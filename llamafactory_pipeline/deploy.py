"""vLLM docker 部署: 命令构建 (纯逻辑) + SSH 调用 (复用 remote.py)。

与训练不同: 训练 docker exec 进已有容器, 部署 docker run 起新容器。
容器即状态: 用 docker ps -a 查, 不另建状态文件。本地另存配置 JSON 便于重建。
"""

from __future__ import annotations

import json
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from . import remote
from .deploy_schema import DeployConfig

_REPO = Path(__file__).resolve().parents[1]
_CFG_DIR = _REPO / "deploy_configs"


# ── 纯逻辑 (可单测) ──

def build_docker_run_argv(d: DeployConfig) -> list[str]:
    """生成 `docker run` argv 列表 (不 quote, 便于测试)。

    d 应已 normalized (container_name 带 vllm- 前缀)。
    """
    d = d.normalized()
    model_path = d.resolved_model_path()
    argv = [
        "docker", "run", "-d",
        "--gpus", f"device={d.gpus}" if d.gpus else "all",
        "--name", d.container_name,
        "--restart", d.restart_policy,
        "-p", f"{d.port}:{d.port}",
        "--log-opt", "max-size=50m",
        "--log-opt", "max-file=3",
        "-v", f"{d.host_model_path}:{model_path}",
        d.image,
        "--model", model_path,
        "--host", "0.0.0.0",
        "--port", str(d.port),
    ]
    if d.api_key:
        argv += ["--api-key", d.api_key]
    argv += [
        "--max-model-len", str(d.max_model_len),
        "--gpu-memory-utilization", str(d.gpu_memory_utilization),
        "--max-num-seqs", str(d.max_num_seqs),
    ]
    if d.reasoning_parser:
        argv += ["--reasoning-parser", d.reasoning_parser]
    if d.enable_auto_tool_choice:
        argv.append("--enable-auto-tool-choice")
    if d.tool_call_parser:
        argv += ["--tool-call-parser", d.tool_call_parser]
    if d.speculative_config:
        argv += ["--speculative-config", d.speculative_config]
    if d.enable_lora:
        argv.append("--enable-lora")
        if d.lora_modules:
            argv += ["--lora-modules", d.lora_modules]
        if d.max_lora_rank > 0:
            argv += ["--max-lora-rank", str(d.max_lora_rank)]
    if d.extra_args.strip():
        # 透传未建模参数 (原样追加, 调用方负责格式)
        argv += d.extra_args.strip().split()
    return argv


def quote_argv(argv: list[str]) -> str:
    return " ".join(shlex.quote(p) for p in argv)


def build_deploy_script(d: DeployConfig) -> str:
    """完整部署脚本: docker run -d, 捕获容器 ID 或失败。

    docker run -d 成功打印容器 ID (重定向到 container_id), 失败非 0 退出。
    不需要 nohup: 容器自带后台 + restart 策略。
    """
    d = d.normalized()
    q = shlex.quote
    argv = build_docker_run_argv(d)
    cmd = quote_argv(argv)
    cfg_dir = _config_dir(d.container_name)
    cid = q(str(cfg_dir / "container_id"))
    err = q(str(cfg_dir / "deploy.err"))
    return "\n".join([
        "set -e",
        f"{cmd} > {cid} 2> {err} || echo FAILED",
        # 容器已存在时 docker run 会失败; 留 err 供诊断
        "echo DEPLOYED",
        "",
    ])


def build_status_script(container_name: str) -> str:
    """docker ps -a 查单容器状态。精确匹配 name=^/name$。"""
    q = shlex.quote
    return "\n".join([
        f"docker ps -a --filter name=^/{q(container_name)}$ "
        f"--format '{{{{.Names}}}}\\t{{{{.Status}}}}\\t{{{{.Ports}}}}'",
        "",
    ])


def parse_container_status(raw: str) -> dict[str, object]:
    """解析 docker ps 输出。空=missing; Up=running; Exited=exited。"""
    line = raw.strip().splitlines()
    if not line or not line[0].strip():
        return {"status": "missing"}
    parts = line[0].split("\t")
    name = parts[0] if len(parts) > 0 else ""
    status_text = parts[1] if len(parts) > 1 else ""
    ports = parts[2] if len(parts) > 2 else ""
    if status_text.startswith("Up"):
        st = "running"
    elif status_text.startswith("Exited"):
        st = "exited"
    elif status_text.startswith("Restarting"):
        st = "restarting"
    else:
        st = "unknown"
    return {"status": st, "name": name, "status_text": status_text, "ports": ports}


def build_list_script() -> str:
    """列所有 vllm- 前缀容器。"""
    return (
        "docker ps -a --filter name=vllm- "
        "--format '{{.Names}}\\t{{.Status}}\\t{{.Ports}}'\n"
    )


def parse_deploy_list(raw: str) -> list[dict[str, object]]:
    out = []
    for line in raw.strip().splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        name = parts[0] if len(parts) > 0 else ""
        if not name:
            continue
        info = parse_container_status("\t".join(parts))
        out.append(info)
    return out


def build_stop_script(container_name: str) -> str:
    """停止并删除容器 (保留本地配置)。"""
    q = shlex.quote
    return "\n".join([
        f"docker stop {q(container_name)} 2>/dev/null || true",
        f"docker rm {q(container_name)} 2>/dev/null || true",
        "echo STOPPED",
        "",
    ])


def build_logs_script(container_name: str, tail: int = 400) -> str:
    """docker logs 取尾部 (增量用 -f, 见 stream_deploy_logs)。"""
    q = shlex.quote
    return f"docker logs --tail {tail} {q(container_name)} 2>&1\n"


# ── vLLM metrics 解析 (纯逻辑) ──

import re as _re

_METRIC_LINE = _re.compile(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+([-+]?[\d.eE+-]+)')


def _parse_metric_line(line: str) -> tuple[str, str | None, float] | None:
    """返回 (name, labels_or_None, value); 注释/空行/解析失败返回 None。"""
    s = line.strip()
    if not s or s.startswith("#"):
        return None
    m = _METRIC_LINE.match(s)
    if not m:
        return None
    try:
        v = float(m.group(3))
    except ValueError:
        return None
    return m.group(1), m.group(2), v


def _histogram_quantile(
    buckets: list[tuple[float, float]], count: float, q: float
) -> float | None:
    """从 Prometheus 直方图桶算分位。

    buckets: [(le, cumulative_count), ...] 含 +Inf 桶; count: 总样本数 (_count);
    q: 分位 (0-1)。找首个 cumulative >= q*count 的 le。
    """
    if not buckets or count <= 0:
        return None
    target = q * count
    prev_le = 0.0
    for le, cum in sorted(buckets):
        if cum >= target:
            # 在 [prev_le, le] 区间内线性插值
            if le == float("inf"):
                return prev_le
            return le  # 简化: 取桶上界 (足够监控用, 不需精确插值)
        prev_le = le
    return None


def parse_prometheus(raw: str) -> dict[str, object]:
    """解析 vLLM /metrics 文本为结构化 dict。

    提取: running / waiting / kv_cache / preemption (gauge 直接值);
    ttft / tpot / e2e (直方图 p50/p95/avg, 秒);
    其余忽略。缺字段填 None, 不抛。
    """
    # 先收集所有样本: name -> [(labels, value)]
    samples: dict[str, list[tuple[str | None, float]]] = {}
    for line in raw.splitlines():
        parsed = _parse_metric_line(line)
        if not parsed:
            continue
        name, labels, v = parsed
        samples.setdefault(name, []).append((labels, v))

    def gauge(name: str) -> float | None:
        xs = samples.get(name)
        return xs[0][1] if xs else None

    def hist(prefix: str) -> dict[str, float | None]:
        """prefix 如 vllm:time_to_first_token_seconds → {p50,p95,avg}。"""
        buckets: list[tuple[float, float]] = []
        sum_v = None
        count_v = None
        for name, lst in samples.items():
            if not name.startswith(prefix):
                continue
            for labels, v in lst:
                if name == prefix + "_sum":
                    sum_v = v
                elif name == prefix + "_count":
                    count_v = v
                elif name == prefix + "_bucket":
                    # labels 形如 {le="0.05"} → 提取 le
                    le = float("inf")
                    if labels:
                        m = _re.search(r'le="([^"]+)"', labels)
                        if m:
                            try:
                                le = float(m.group(1))
                            except ValueError:
                                continue
                    buckets.append((le, v))
        avg = (sum_v / count_v) if (sum_v is not None and count_v) else None
        p50 = _histogram_quantile(buckets, count_v or 0, 0.5) if count_v else None
        p95 = _histogram_quantile(buckets, count_v or 0, 0.95) if count_v else None
        return {"p50": p50, "p95": p95, "avg": avg}

    return {
        "running": gauge("vllm:num_requests_running"),
        "waiting": gauge("vllm:num_requests_waiting"),
        "kv_cache": gauge("vllm:gpu_cache_usage_perc"),
        "preemption": gauge("vllm:num_preemption"),
        "ttft": hist("vllm:time_to_first_token_seconds"),
        "tpot": hist("vllm:time_per_output_token_seconds"),
        "e2e": hist("vllm:e2e_request_latency_seconds"),
    }


# ── 本地配置存储 ──

def _config_dir(container_name: str) -> Path:
    d = _CFG_DIR / container_name
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_config(d: DeployConfig) -> Path:
    """本地存部署配置 JSON (便于重建/编辑; 状态不依赖它)。"""
    d = d.normalized()
    p = _config_dir(d.container_name) / "config.json"
    p.write_text(json.dumps(d.model_dump(), ensure_ascii=False, indent=2),
                 encoding="utf-8")
    return p


def load_config(container_name: str) -> DeployConfig | None:
    p = _CFG_DIR / container_name / "config.json"
    if not p.exists():
        return None
    return DeployConfig.model_validate(json.loads(p.read_text("utf-8")))


def list_saved_configs() -> list[str]:
    if not _CFG_DIR.exists():
        return []
    return sorted(d.name for d in _CFG_DIR.iterdir() if d.is_dir())


def delete_config(container_name: str) -> bool:
    import shutil
    d = _CFG_DIR / container_name
    if d.exists():
        shutil.rmtree(d)
        return True
    return False


# ── SSH 调用 (复用 remote) ──

def deploy(cfg: remote.RemoteConfig, d: DeployConfig) -> dict[str, object]:
    """启动部署: 存配置 → 远端 docker run。返回 {deployed, container_name}。"""
    d = d.normalized()
    save_config(d)
    out = remote.run_remote_script(cfg, build_deploy_script(d), timeout=60)
    last = out.strip().splitlines()[-1] if out.strip() else ""
    return {"deployed": last == "DEPLOYED", "container_name": d.container_name,
            "detail": last}


def list_deployments(cfg: remote.RemoteConfig) -> list[dict[str, object]]:
    raw = remote.run_remote_script(cfg, build_list_script(), timeout=15)
    return parse_deploy_list(raw)


def deployment_status(cfg: remote.RemoteConfig, container_name: str) -> dict[str, object]:
    raw = remote.run_remote_script(cfg, build_status_script(container_name), timeout=15)
    return parse_container_status(raw)


def stop_deployment(cfg: remote.RemoteConfig, container_name: str) -> dict[str, object]:
    out = remote.run_remote_script(cfg, build_stop_script(container_name), timeout=60)
    last = out.strip().splitlines()[-1] if out.strip() else ""
    return {"stopped": last == "STOPPED", "container_name": container_name}


def stream_deploy_logs(
    cfg: remote.RemoteConfig, container_name: str, from_line: int
) -> Iterator[tuple[int, str]]:
    """SSE 用: docker logs -f 持续跟随。复用 remote.stream_logs 的 select 心跳模式。"""
    import select as _select
    q = shlex.quote
    # --tail 从 from_line 近似 (docker logs 按行无直接 offset; 用 tail 行数近似增量)
    start = max(1, from_line)
    remote_cmd = f"docker logs -f --tail {start} {q(container_name)} 2>&1"
    proc = subprocess.Popen(
        remote._ssh_argv(cfg) + [remote_cmd],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        text=True, bufsize=1,
    )
    line_no = start
    try:
        assert proc.stdout is not None
        fd = proc.stdout
        while True:
            if proc.poll() is not None:
                break
            ready, _, _ = _select.select([fd], [], [], 15.0)
            if not ready:
                yield (line_no, "\0heartbeat")
                continue
            line = fd.readline()
            if line == "":
                break
            yield (line_no, line.rstrip("\n"))
            line_no += 1
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

# ── 健康探针 / 指标 / 对话代理 ──

class DeployError(RuntimeError):
    """部署配置缺失或容器不可达。"""


def _resolve_endpoint(cfg: remote.RemoteConfig, container_name: str) -> tuple[int, str]:
    """从本地配置读 (port, api_key); 无配置抛 DeployError。"""
    d = load_config(container_name)
    if d is None:
        raise DeployError(f"容器 {container_name} 无平台配置, 无法监控/对话")
    return d.port, (d.api_key or "")


def probe_health(cfg: remote.RemoteConfig, container_name: str) -> dict[str, object]:
    """探 vLLM /health: 200=就绪, 503=启动中, 其他/超时=不可达。经 SSH curl localhost:port。"""
    port, _ = _resolve_endpoint(cfg, container_name)
    script = (
        f"curl -s -o /dev/null -w '%{{http_code}}' --max-time 5 "
        f"http://localhost:{port}/health\n"
    )
    try:
        out = remote.run_remote_script(cfg, script, timeout=10)
    except remote.RemoteError as e:
        return {"ready": False, "http_code": "", "detail": f"SSH 失败: {e}"}
    code = out.strip().splitlines()[-1].strip("'") if out.strip() else ""
    ready = code == "200"
    if ready:
        detail = "就绪"
    elif code == "503":
        detail = "模型加载中"
    elif code:
        detail = f"HTTP {code}"
    else:
        detail = "无响应"
    return {"ready": ready, "http_code": code, "detail": detail}


def fetch_metrics(cfg: remote.RemoteConfig, container_name: str) -> dict[str, object]:
    """抓 vLLM /metrics 并解析。失败返回空 dict (不抛, 监控页容错)。"""
    port, api_key = _resolve_endpoint(cfg, container_name)
    auth = f"-H 'Authorization: Bearer {api_key}' " if api_key else ""
    script = (
        f"curl -s --max-time 8 {auth}http://localhost:{port}/metrics\n"
    )
    try:
        raw = remote.run_remote_script(cfg, script, timeout=12)
    except remote.RemoteError:
        return {}
    return parse_prometheus(raw)


def chat_proxy_stream(
    cfg: remote.RemoteConfig, container_name: str, body_bytes: bytes
) -> Iterator[bytes]:
    """流式代理: 前端 body → SSH curl -N → vLLM /v1/chat/completions, stdout 原样转发。

    body 是 OpenAI chat completions 请求 JSON (含 messages/stream 等)。
    用 --data @- 经 stdin 传 body, 避免 shell 引用。
    """
    port, api_key = _resolve_endpoint(cfg, container_name)
    auth = f"-H 'Authorization: Bearer {api_key}' " if api_key else ""
    remote_cmd = (
        f"curl -sN {auth}-H 'Content-Type: application/json' --data @- "
        f"http://localhost:{port}/v1/chat/completions"
    )
    proc = subprocess.Popen(
        remote._ssh_argv(cfg) + [remote_cmd],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    assert proc.stdin is not None and proc.stdout is not None
    try:
        proc.stdin.write(body_bytes)
        proc.stdin.close()
        for chunk in iter(lambda: proc.stdout.read(4096), b""):
            yield chunk
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
