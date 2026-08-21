"""通过本机 ssh/scp 在远端服务器提交并观察 LlamaFactory 训练任务。

设计要点:
- 不引入 paramiko, 直接用系统 ssh/scp (复用已有密钥/ssh-agent)。
- 远端目录是任务状态的唯一事实来源, 本地不落库。
- 脚本通过 `ssh ... sh -s` 走 stdin, 避免 shell 引用地狱; 脚本内路径用 shlex.quote。
- 训练用 nohup 脱离 SSH 会话, 浏览器/本地服务重启不影响远端训练。
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import secrets
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterator, Optional

_JOB_ID_RE = re.compile(r"^\d{8}T\d{6}Z-[0-9a-f]{6}$")
_GPUS_RE = re.compile(r"^\d+(,\d+)*$")


class RemoteError(RuntimeError):
    """SSH/SCP 或远端命令失败, 消息已脱敏 (不含私钥/完整命令)。"""


class RemoteConflictError(RemoteError, ValueError):
    """A durable job ID is already bound to different immutable inputs."""


_DEFAULT_CONFIG = os.path.join(os.path.dirname(__file__), "server_config.yaml")


def _load_config_file() -> dict:
    """读取服务器配置 yml (缺文件返回空 dict)。路径可用 TRAIN_CONFIG 覆盖。"""
    path = os.environ.get("TRAIN_CONFIG", _DEFAULT_CONFIG)
    if not os.path.exists(path):
        return {}
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@dataclass(frozen=True)
class RemoteConfig:
    ssh_target: str
    remote_root: str
    llamafactory_dir: str
    ssh_port: str = "22"
    ssh_identity: Optional[str] = None
    # 设置后训练走 `docker exec <container> ...`; 为空则在宿主机直接跑 cli
    docker_container: Optional[str] = None
    # 宿主机模式下, 在跑 cli 前执行的 shell 片段 (如 conda activate); 仅非 docker 时用
    remote_prefix: str = ""

    @classmethod
    def from_env(cls) -> "RemoteConfig":
        """按 yml 配置文件为底、环境变量覆盖的顺序装配。

        yml 路径取 TRAIN_CONFIG, 默认包目录下 server_config.yaml。
        缺文件不报错 (可纯用环境变量); 三个必填项任一缺失才报错。
        """
        y = _load_config_file()

        def pick(env_key: str, yml_key: str, default: Optional[str] = None) -> Optional[str]:
            v = os.environ.get(env_key)
            if v:
                return v
            yv = y.get(yml_key)
            return str(yv) if yv not in (None, "") else default

        ssh_target = pick("TRAIN_SSH_TARGET", "ssh_target")
        remote_root = pick("TRAIN_REMOTE_ROOT", "remote_root")
        llamafactory_dir = pick("LLAMAFACTORY_DIR", "llamafactory_dir")
        missing = [name for name, val in (
            ("ssh_target", ssh_target), ("remote_root", remote_root),
            ("llamafactory_dir", llamafactory_dir),
        ) if not val]
        if missing:
            raise RemoteError(f"缺少配置: {', '.join(missing)} (设 server_config.yaml 或对应环境变量)")
        for tool in ("ssh", "scp"):
            if shutil.which(tool) is None:
                raise RemoteError(f"本机未找到 {tool}")

        identity = pick("TRAIN_SSH_IDENTITY", "ssh_identity")
        return cls(
            ssh_target=ssh_target,
            remote_root=remote_root,
            llamafactory_dir=llamafactory_dir,
            ssh_port=pick("TRAIN_SSH_PORT", "ssh_port", "22"),
            ssh_identity=os.path.expanduser(identity) if identity else None,
            docker_container=pick("TRAIN_DOCKER_CONTAINER", "docker_container") or None,
            remote_prefix=pick("TRAIN_REMOTE_PREFIX", "remote_prefix", "") or "",
        )


# ── 纯逻辑 (可单测) ──

def new_job_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{ts}-{secrets.token_hex(3)}"


def validate_job_id(job_id: str) -> str:
    if not _JOB_ID_RE.match(job_id):
        raise RemoteError("非法任务 ID")
    return job_id


def validate_gpus(gpus: Optional[str]) -> str:
    """校验显卡选择, 返回规范化字符串。空表示不指定 (用容器/环境默认可见卡)。"""
    g = (gpus or "").strip()
    if g == "":
        return ""
    if not _GPUS_RE.match(g):
        raise RemoteError("GPU 需形如 0 或 0,1")
    return g


def job_dir(cfg: RemoteConfig, job_id: str) -> str:
    return f"{cfg.remote_root.rstrip('/')}/{validate_job_id(job_id)}"


_GPU_QUERY = (
    "nvidia-smi --query-gpu=index,name,memory.used,memory.total,memory.free,"
    "utilization.gpu,temperature.gpu,power.draw --format=csv,noheader,nounits\n"
)


def parse_gpu_csv(text: str) -> list[dict[str, object]]:
    """解析新八列 GPU 遥测，并兼容旧五列响应。"""
    gpus: list[dict[str, object]] = []
    for line in text.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        enriched = len(parts) >= 8
        if enriched:
            idx, name, used, total, free, util, temperature, power = parts[:8]
        else:
            idx, name, used, total, util = parts[:5]

        def _num(v: str, *, na_to_none: bool = enriched) -> object:
            if na_to_none and v == "[N/A]":
                return None
            try:
                return int(v)
            except ValueError:
                return v

        row: dict[str, object] = {
            "index": _num(idx), "name": name,
            "mem_used": _num(used), "mem_total": _num(total),
            "util": _num(util),
        }
        if enriched:
            try:
                power_value: object = None if power == "[N/A]" else float(power)
            except ValueError:
                power_value = power
            row.update({
                "mem_free": _num(free),
                "temperature": _num(temperature),
                "power_draw": power_value,
            })
        gpus.append(row)
    return gpus


def _train_env(gpus: str) -> list[tuple[str, str]]:
    """按显卡选择生成注入的环境变量。多卡时开 FORCE_TORCHRUN, 否则 LlamaFactory 只用一张卡。"""
    envs: list[tuple[str, str]] = []
    if gpus:
        envs.append(("CUDA_VISIBLE_DEVICES", gpus))
        if "," in gpus:
            envs.append(("FORCE_TORCHRUN", "1"))
    return envs


def _host_header(cfg: RemoteConfig) -> str:
    """宿主机模式脚本头: 可选环境准备 (conda activate 等) + cd 到 LlamaFactory 目录。

    prefix 与 cd 放在外层脚本, nohup 子进程会继承其 PATH/CWD, 故训练/评测进程都能拿到。
    """
    lines = ["set -e"]
    if cfg.remote_prefix:
        lines.append(cfg.remote_prefix)
    lines.append(f"cd {shlex.quote(cfg.llamafactory_dir)}")
    return "\n".join(lines)


def _build_durable_launch_wrapper(
    header: str,
    directory: str,
    command: str,
    log_path: str,
) -> str:
    """Wrap a remote command with one atomic PID/starttime identity marker."""
    q = shlex.quote
    launch_path = q(directory + "/launch_identity")
    launch_tmp_path = q(directory + "/launch_identity.tmp")
    status_path = q(directory + "/status")
    status_tmp_path = q(directory + "/status.tmp")
    exit_path = q(directory + "/exit_code")
    exit_tmp_path = q(directory + "/exit_code.tmp")
    recovered_terminal = (
        f"printf '%s\\n' 'assistant: recovered incomplete launch markers' "
        f">> {q(log_path)}; "
        'printf 125 > "$D/exit_code.tmp"; '
        'mv "$D/exit_code.tmp" "$D/exit_code"; '
        "echo RECOVERED; echo ALREADY; exit 0"
    )
    inner = "; ".join(
        [
            "P=$$",
            'if [ ! -r "/proc/$P/stat" ]; then '
            f'printf 125 > {exit_tmp_path}; mv {exit_tmp_path} {exit_path}; exit 125; fi',
            'START=$(sed "s/^[^)]*) //" "/proc/$P/stat" | awk "{print \\$20}")',
            'if [ -z "$START" ]; then '
            f'printf 125 > {exit_tmp_path}; mv {exit_tmp_path} {exit_path}; exit 125; fi',
            f'printf "%s %s\\n" "$P" "$START" > {launch_tmp_path}',
            f"mv {launch_tmp_path} {launch_path}",
            f"printf running > {status_tmp_path}",
            f"mv {status_tmp_path} {status_path}",
            command,
            "C=$?",
            f'printf %s "$C" > {exit_tmp_path}',
            f"mv {exit_tmp_path} {exit_path}",
            'exit "$C"',
        ]
    )
    return "\n".join(
        [
            header,
            f"D={q(directory)}",
            'exec 9>"$D/launch.lock"',
            "flock -w 20 9 || { echo BUSY; exit 0; }",
            'identity_alive() { local P="$1" EXPECTED="$2" NOW; '
            'kill -0 "$P" 2>/dev/null || return 1; '
            '[ -r "/proc/$P/stat" ] || return 1; '
            'NOW=$(sed "s/^[^)]*) //" "/proc/$P/stat" | awk "{print \\$20}"); '
            '[ -n "$NOW" ] && [ "$NOW" = "$EXPECTED" ]; }',
            'if [ -f "$D/exit_code" ]; then echo ALREADY; exit 0; fi',
            'if [ -s "$D/launch_identity" ]; then',
            '  if read P START < "$D/launch_identity" '
            '&& identity_alive "$P" "$START"; then echo ALREADY; exit 0; fi',
            "  " + recovered_terminal,
            "fi",
            '# Legacy multi-file markers: trust only a matching live starttime.',
            'if [ -s "$D/pid" ] && [ -s "$D/pid_starttime" ]; then',
            '  P=$(cat "$D/pid"); START=$(cat "$D/pid_starttime")',
            '  if identity_alive "$P" "$START"; then echo ALREADY; exit 0; fi',
            "fi",
            'if [ -e "$D/pid" ] || [ -e "$D/pid_starttime" ] '
            '|| [ -e "$D/status" ] || [ -e "$D/launch_identity" ]; then',
            "  " + recovered_terminal,
            "fi",
            f"nohup setsid bash -c {q(inner)} 9>&- > {q(log_path)} 2>&1 < /dev/null &",
            "P=$!",
            "I=0",
            'while [ "$I" -lt 50 ]; do',
            '  if [ -f "$D/exit_code" ] || [ -s "$D/launch_identity" ]; '
            'then echo STARTED; exit 0; fi',
            '  kill -0 "$P" 2>/dev/null || break',
            "  sleep 0.1; I=$((I + 1))",
            "done",
            "echo NOT_READY",
            "",
        ]
    )


def build_launch_script(cfg: RemoteConfig, job_id: str, gpus: str = "") -> str:
    """生成远端启动脚本; 所有路径/参数经 shlex.quote。

    宿主机始终用 nohup 包装 (脱离 SSH 会话, pid/exit_code 落盘)。docker 模式下
    仅把内层命令换成 `docker exec`, 训练在容器内跑但 pid/日志/退出码仍归宿主机管理。
    """
    d = job_dir(cfg, job_id)
    q = shlex.quote
    cfg_file = d + "/qwen3_lora_sft.yaml"
    envs = _train_env(gpus)

    if cfg.docker_container:
        # 容器内: 用 -e 注入显卡, -w 设工作目录 (相对 output_dir 依赖 cwd)
        parts = ["docker", "exec"]
        for k, v in envs:
            parts += ["-e", f"{k}={v}"]
        parts += ["-w", cfg.llamafactory_dir, cfg.docker_container,
                  "llamafactory-cli", "train", cfg_file]
        train_cmd = " ".join(q(p) for p in parts)
        header = "set -e"
    else:
        # 宿主机: 显卡作 shell 变量前缀, cd 到 LlamaFactory 目录
        prefix = "".join(f"{k}={q(v)} " for k, v in envs)
        train_cmd = f"{prefix}llamafactory-cli train {q(cfg_file)}"
        header = _host_header(cfg)

    return _build_durable_launch_wrapper(
        header, d, train_cmd, d + "/train.log"
    )


def build_status_script(cfg: RemoteConfig, job_id: str) -> str:
    """Resolve status using only an exact PID plus /proc creation starttime.

    New jobs use atomic launch_identity; legacy pid/pid_starttime pairs remain
    readable but command-name or PID-only fallbacks are deliberately rejected.
    """
    d = job_dir(cfg, job_id)
    q = shlex.quote
    return "\n".join([
        f"D={q(d)}",
        'if [ ! -d "$D" ]; then echo NOTFOUND; exit 0; fi',
        'if [ -f "$D/exit_code" ]; then echo "EXIT $(cat "$D/exit_code")"; exit 0; fi',
        'P=""; EXPECTED=""; IDENTITY=0',
        'if [ -s "$D/launch_identity" ]; then '
        'read P EXPECTED < "$D/launch_identity" && IDENTITY=1',
        'elif [ -s "$D/pid" ] && [ -s "$D/pid_starttime" ]; then '
        'P=$(cat "$D/pid"); EXPECTED=$(cat "$D/pid_starttime"); IDENTITY=1; fi',
        'if [ "$IDENTITY" = 1 ] && kill -0 "$P" 2>/dev/null '
        '&& [ -r "/proc/$P/stat" ]; then',
        '  NOW=$(sed "s/^[^)]*) //" "/proc/$P/stat" | awk "{print \\$20}")',
        '  [ -n "$NOW" ] && [ "$NOW" = "$EXPECTED" ] '
        '&& { echo RUNNING; exit 0; }',
        'fi',
        'if [ "$IDENTITY" = 1 ] && [ -f "$D/status" ] '
        '&& [ "$(cat "$D/status")" = "running" ]; then echo INTERRUPTED; exit 0; fi',
        "echo UNKNOWN",
        "",
    ])


def parse_status(output: str) -> dict[str, object]:
    line = output.strip().splitlines()[-1] if output.strip() else ""
    if line == "NOTFOUND":
        return {"status": "not_found"}
    if line == "RUNNING":
        return {"status": "running"}
    if line == "UNKNOWN":
        return {"status": "unknown"}
    if line == "INTERRUPTED":
        # pid 没了但 status 曾 running: 服务器重启/进程被杀, 非正常退出
        return {"status": "interrupted"}
    if line.startswith("EXIT "):
        code = line[5:].strip()
        return {
            "status": "succeeded" if code == "0" else "failed",
            "exit_code": code,
        }
    return {"status": "unknown"}


def build_list_script(cfg: RemoteConfig) -> str:
    """列出 remote_root 下所有任务目录, 每行输出 `id|kind|status`。kind 按 yaml 判 train/eval。"""
    q = shlex.quote
    return "\n".join([
        f"R={q(cfg.remote_root.rstrip('/'))}",
        'for D in "$R"/*/; do',
        '  [ -d "$D" ] || continue',
        '  id=$(basename "$D")',
        '  case "$id" in ????????T??????Z-*) : ;; *) continue ;; esac',
        '  if [ -f "$D/qwen3_lora_sft.yaml" ]; then kind=train; else kind=eval; fi',
        '  if [ -f "$D/exit_code" ]; then st="EXIT $(cat "$D/exit_code")";',
        '  elif [ -s "$D/launch_identity" ] || '
        '{ [ -s "$D/pid" ] && [ -s "$D/pid_starttime" ]; }; then',
        '    if [ -s "$D/launch_identity" ]; then '
        'read P EXPECTED < "$D/launch_identity"; '
        'else P=$(cat "$D/pid"); EXPECTED=$(cat "$D/pid_starttime"); fi',
        '    MATCH=0',
        '    if kill -0 "$P" 2>/dev/null && [ -r "/proc/$P/stat" ]; then',
        '      NOW=$(sed "s/^[^)]*) //" "/proc/$P/stat" | awk "{print \\$20}")',
        '      [ -n "$NOW" ] && [ "$NOW" = "$EXPECTED" ] && MATCH=1',
        '    fi',
        '    if [ "$MATCH" = 1 ]; then st=RUNNING; '
        'elif [ -f "$D/status" ] && [ "$(cat "$D/status")" = running ]; '
        'then st=INTERRUPTED; else st=UNKNOWN; fi',
        '  else st=UNKNOWN; fi',
        r'  printf "%s|%s|%s\n" "$id" "$kind" "$st"',
        'done',
        "",
    ])


def parse_job_list(output: str) -> list[dict[str, object]]:
    """解析 build_list_script 输出, 新任务在前 (id 以时间戳打头, 逆序即最新)。"""
    jobs: list[dict[str, object]] = []
    for line in output.strip().splitlines():
        parts = line.split("|")
        if len(parts) != 3:
            continue
        job_id, kind, st = parts
        info = parse_status(st)
        info.update({"job_id": job_id, "kind": kind})
        jobs.append(info)
    jobs.sort(key=lambda j: j["job_id"], reverse=True)
    return jobs


def build_stop_script(cfg: RemoteConfig, job_id: str) -> str:
    """杀掉任务进程组 (setsid 后 pgid==pid), 并落 exit_code=143 使状态转为 failed。"""
    d = job_dir(cfg, job_id)
    q = shlex.quote
    return "\n".join([
        f"D={q(d)}",
        'if [ -s "$D/launch_identity" ]; then '
        'read P EXPECTED < "$D/launch_identity"; '
        'elif [ -s "$D/pid" ] && [ -s "$D/pid_starttime" ]; then '
        'P=$(cat "$D/pid"); EXPECTED=$(cat "$D/pid_starttime"); '
        'else echo UNVERIFIEDPID; exit 0; fi',
        'kill -0 "$P" 2>/dev/null || { echo NOPROCESS; exit 0; }',
        'MATCH=0',
        'if [ -r "/proc/$P/stat" ]; then',
        '  NOW=$(sed "s/^[^)]*) //" "/proc/$P/stat" | awk "{print \\$20}")',
        '  [ -n "$NOW" ] && [ "$NOW" = "$EXPECTED" ] && MATCH=1',
        'fi',
        '[ "$MATCH" = 1 ] || { echo STALEPID; exit 0; }',
        'kill -TERM -"$P" 2>/dev/null || kill -TERM "$P" 2>/dev/null || true',
        '[ -f "$D/exit_code" ] || printf 143 > "$D/exit_code"',
        "echo STOPPED",
        "",
    ])


def build_disk_script(cfg: RemoteConfig) -> str:
    """查 remote_root 所在盘可用空间 (KB); 目录不存在则向上找最近存在的祖先。"""
    q = shlex.quote
    return "\n".join([
        f"P={q(cfg.remote_root.rstrip('/'))}",
        'while [ ! -d "$P" ] && [ "$P" != "/" ]; do P=$(dirname "$P"); done',
        "df -Pk \"$P\" | awk 'NR==2{print $4}'",
        "",
    ])


# ── IO: ssh/scp ──

def _ssh_argv(cfg: RemoteConfig) -> list[str]:
    argv = [
        "ssh", "-p", cfg.ssh_port,
        "-o", "BatchMode=yes",
        "-o", "ConnectTimeout=10",
    ]
    if cfg.ssh_identity:
        argv += ["-i", cfg.ssh_identity, "-o", "IdentitiesOnly=yes"]
    argv.append(cfg.ssh_target)
    return argv


def _scp_argv(cfg: RemoteConfig, local: str, remote_rel: str) -> list[str]:
    argv = ["scp", "-P", cfg.ssh_port, "-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]
    if cfg.ssh_identity:
        argv += ["-i", cfg.ssh_identity, "-o", "IdentitiesOnly=yes"]
    argv += [local, f"{cfg.ssh_target}:{remote_rel}"]
    return argv


def run_remote_script(cfg: RemoteConfig, script: str, timeout: int = 30) -> str:
    """通过 `ssh ... bash -s` 执行脚本 (脚本走 stdin, 无需再引用整体)。

    用 bash 而非 sh(dash): remote_prefix 里的 `source`/`conda activate` 是 bash 特性。
    """
    try:
        proc = subprocess.run(
            _ssh_argv(cfg) + ["bash", "-s"],
            input=script,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as e:
        raise RemoteError("SSH 命令超时") from e
    except OSError as e:
        raise RemoteError(f"SSH 执行失败: {e}") from e
    if proc.returncode != 0:
        raise RemoteError(f"远端命令失败: {proc.stderr.strip()[:400]}")
    return proc.stdout


def _scp(cfg: RemoteConfig, local: str, remote_rel: str, timeout: int = 120) -> None:
    try:
        proc = subprocess.run(
            _scp_argv(cfg, local, remote_rel),
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired as e:
        raise RemoteError("SCP 超时") from e
    except OSError as e:
        raise RemoteError(f"SCP 执行失败: {e}") from e
    if proc.returncode != 0:
        raise RemoteError(f"上传失败: {proc.stderr.strip()[:400]}")


def read_trainer_log(cfg: RemoteConfig, job_id: str) -> dict[str, object]:
    """读远端 LlamaFactory 训练指标 (trainer_log.jsonl) 供前端画 loss 曲线。

    output_dir 从任务 yaml 读; 相对路径按 llamafactory_dir 解析。文件不存在返回空点。
    """
    out_dir = resolve_output_dir(cfg, job_id)
    if not out_dir:
        return {"points": [], "total_steps": 0}
    jsonl = shlex.quote(out_dir + "/trainer_log.jsonl")
    # tail 增量取尾部: 训练后期该文件可达数十 MB, 前端每 2-3s 轮询全量 cat 很浪费。
    # total_steps / percentage / remaining_time 在每条 log 里都带, tail 末条即最新, 不丢。
    command = f"tail -n 400 {jsonl} 2>/dev/null || true\n"
    if cfg.docker_container:
        command = (
            f"docker exec {shlex.quote(cfg.docker_container)} sh -lc "
            f"{shlex.quote(command)}\n"
        )
    raw = run_remote_script(cfg, command, timeout=15)
    return parse_trainer_log(raw)


def read_job_log_tail(
    cfg: RemoteConfig,
    job_id: str,
    lines: int = 80,
    log_name: str = "train.log",
) -> str:
    """Read a bounded training-log tail for failure classification."""
    validate_job_id(job_id)
    if log_name not in {"train.log", "eval.log"}:
        raise ValueError("unsupported job log name")
    bounded_lines = min(200, max(1, int(lines)))
    path = shlex.quote(job_dir(cfg, job_id) + "/" + log_name)
    raw = run_remote_script(
        cfg,
        f"test -r {path} && tail -n {bounded_lines} {path}\n",
        timeout=15,
    )
    return raw[-12000:]


def resolve_output_dir(cfg: RemoteConfig, job_id: str) -> str:
    """从任务 yaml 读 output_dir 并解析为绝对路径; 读不到返回空串。"""
    import yaml

    d = job_dir(cfg, job_id)
    try:
        y = yaml.safe_load(
            run_remote_script(
                cfg, f"cat {shlex.quote(d + '/qwen3_lora_sft.yaml')}\n"
            )
        )
    except RemoteError:
        return ""
    except yaml.YAMLError as exc:
        raise RemoteError("无法解析远端训练配置") from exc
    if not isinstance(y, dict):
        raise RemoteError("远端训练配置不是对象")
    value = y.get("output_dir", "")
    if value is not None and not isinstance(value, str):
        raise RemoteError("远端训练配置 output_dir 类型非法")
    out_dir = str(value or "").strip()
    if not out_dir:
        return ""
    if not out_dir.startswith("/"):
        out_dir = cfg.llamafactory_dir.rstrip("/") + "/" + out_dir
    return out_dir


def build_list_checkpoints_script(cfg: RemoteConfig, job_id: str) -> str:
    """列 output_dir 下 checkpoint-* 子目录, 每行 `name|size_kb`。

    size 用 du -sk 取目录大小 (KB), 供前端展示占用。checkpoint 不存在则 echo 空。
    """
    d = job_dir(cfg, job_id)
    q = shlex.quote
    prefix = [
        f"D={q(d)}",
        f"LF={q(cfg.llamafactory_dir.rstrip('/'))}",
        'if [ ! -f "$D/qwen3_lora_sft.yaml" ]; then echo NOTFOUND; exit 0; fi',
        # output_dir 从 yaml grep (避免本地再起 python/yaml), 兼容相对/绝对路径
        'OUT=$(grep -E "^output_dir:" "$D/qwen3_lora_sft.yaml" | sed "s/^output_dir:[[:space:]]*//")',
        'if [ -z "$OUT" ]; then echo ""; exit 0; fi',
        'case "$OUT" in /*) ;; *) OUT="$LF/$OUT" ;; esac',
    ]
    runtime = [
        f'if [ ! -d "$OUT" ]; then echo ""; exit 0; fi',
        'for C in "$OUT"/checkpoint-*; do',
        '  [ -d "$C" ] || continue',
        '  SZ=$(du -sk "$C" 2>/dev/null | awk "{print \\$1}")',
        '  printf "%s|%s\\n" "$(basename "$C")" "${SZ:-0}"',
        'done',
        "",
    ]
    if cfg.docker_container:
        inner = "\n".join(['OUT="$1"', *runtime])
        prefix.append(
            f"docker exec {q(cfg.docker_container)} sh -lc {q(inner)} sh \"$OUT\""
        )
        return "\n".join(prefix) + "\n"
    return "\n".join([*prefix, *runtime])


def parse_checkpoint_list(raw: str) -> list[dict[str, object]]:
    """解析 build_list_checkpoints_script 输出。NOTFOUND/空 → 空列表。"""
    out: list[dict[str, object]] = []
    for line in raw.strip().splitlines():
        line = line.strip()
        if not line or line == "NOTFOUND":
            continue
        parts = line.split("|")
        if len(parts) != 2:
            continue
        name, size_kb = parts
        match = re.fullmatch(r"checkpoint-(\d+)", name)
        if match is None:
            continue
        try:
            sz = int(size_kb)
        except ValueError:
            sz = 0
        out.append({"name": name, "size_kb": sz, "step": int(match.group(1))})
    return sorted(out, key=lambda row: int(row["step"]))


def list_checkpoints(cfg: RemoteConfig, job_id: str) -> list[dict[str, object]]:
    """列某训练任务的 checkpoint-* 子目录 (供续训选择)。"""
    out = run_remote_script(cfg, build_list_checkpoints_script(cfg, job_id), timeout=20)
    if "NOTFOUND" in {line.strip() for line in out.splitlines()}:
        raise RemoteError("训练任务配置不存在，无法确认 checkpoint")
    rows = parse_checkpoint_list(out)
    output_dir = resolve_output_dir(cfg, job_id)
    if output_dir:
        for row in rows:
            row["path"] = output_dir.rstrip("/") + "/" + str(row["name"])
    return rows


def build_cleanup_checkpoint_script(
    cfg: RemoteConfig, job_id: str, checkpoint_name: str
) -> str:
    """删除 output_dir 下指定 checkpoint-* 子目录。

    安全: checkpoint_name 必须形如 checkpoint-N (纯数字后缀), 拼到 output_dir 后再
    二次校验路径确实在 output_dir 下, 防 path traversal 误删其他目录。
    """
    if not re.match(r"^checkpoint-\d+$", checkpoint_name):
        raise RemoteError("checkpoint 名必须形如 checkpoint-<数字>")
    d = job_dir(cfg, job_id)
    q = shlex.quote
    prefix = [
        f"D={q(d)}",
        f"LF={q(cfg.llamafactory_dir.rstrip('/'))}",
        'if [ ! -f "$D/qwen3_lora_sft.yaml" ]; then echo NOTFOUND; exit 0; fi',
        'OUT=$(grep -E "^output_dir:" "$D/qwen3_lora_sft.yaml" | sed "s/^output_dir:[[:space:]]*//")',
        'if [ -z "$OUT" ]; then echo NOOUTPUT; exit 0; fi',
        'case "$OUT" in /*) ;; *) OUT="$LF/$OUT" ;; esac',
    ]
    runtime = [
        f'TARGET="$OUT/{checkpoint_name}"',
        # 二次校验: TARGET 必须在 OUT 目录下且确实存在 (防 traversal)
        'case "$TARGET" in',
        f'  "$OUT"/{checkpoint_name}) ;;',
        '  *) echo BADPATH; exit 0 ;;',
        'esac',
        'if [ ! -d "$TARGET" ]; then echo NOTFOUND; exit 0; fi',
        'rm -rf "$TARGET"',
        'echo DELETED',
        "",
    ]
    if cfg.docker_container:
        inner = "\n".join(['OUT="$1"', *runtime])
        prefix.append(
            f"docker exec {q(cfg.docker_container)} sh -lc {q(inner)} sh \"$OUT\""
        )
        return "\n".join(prefix) + "\n"
    return "\n".join([*prefix, *runtime])


def inspect_training_output(
    cfg: RemoteConfig, job_id: str
) -> dict[str, object]:
    """Verify that a successful LoRA run produced a readable adapter artifact."""
    output_dir = resolve_output_dir(cfg, job_id)
    if not output_dir:
        raise RemoteError("训练任务缺少可验证的 output_dir")
    q = shlex.quote
    json_probe = q(
        "import json,sys; value=json.load(open(sys.argv[1])); "
        "sys.exit(0 if isinstance(value, dict) and bool(value) else 1)"
    )
    probe = "\n".join(
        [
            f"P={q(output_dir)}",
            '[ -d "$P" ] && [ -r "$P" ] || { echo MISSING; exit 0; }',
            '[ -s "$P/adapter_config.json" ] || { echo INCOMPLETE; exit 0; }',
            f'python -c {json_probe} "$P/adapter_config.json" '
            '|| { echo INCOMPLETE; exit 0; }',
            '{ [ -s "$P/adapter_model.safetensors" ] || '
            '[ -s "$P/adapter_model.bin" ]; } || { echo INCOMPLETE; exit 0; }',
            "echo COMPLETE",
            "",
        ]
    )
    script = probe
    if cfg.docker_container:
        script = (
            f"docker exec {q(cfg.docker_container)} sh -lc {q(probe)}\n"
        )
    else:
        script = _host_header(cfg) + "\n" + probe
    raw = run_remote_script(cfg, script, timeout=20).strip().splitlines()
    state = raw[-1] if raw else ""
    if state not in {"COMPLETE", "INCOMPLETE", "MISSING"}:
        raise RemoteError("无法解析训练产物证据")
    return {
        "output_evidence_verified": True,
        "output_verified": state == "COMPLETE",
        "output_path": output_dir,
        "output_state": state.lower(),
    }


def cleanup_checkpoint(cfg: RemoteConfig, job_id: str, checkpoint_name: str) -> dict[str, object]:
    """删除某训练任务的指定 checkpoint, 返回 {deleted, detail}。"""
    out = run_remote_script(cfg, build_cleanup_checkpoint_script(cfg, job_id, checkpoint_name),
                            timeout=60).strip().splitlines()
    last = out[-1] if out else ""
    return {"deleted": last == "DELETED", "detail": last}


def parse_trainer_log(raw: str) -> dict[str, object]:
    """把 trainer_log.jsonl 文本解析为 {points, total_steps, 进度/ETA} (纯逻辑, 可单测)。"""
    points: list[dict[str, object]] = []
    total = 0
    meta: dict[str, object] = {"percentage": "", "elapsed_time": "", "remaining_time": ""}
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(rec, dict):
            continue
        if rec.get("total_steps"):
            total = rec["total_steps"]
        for k in ("percentage", "elapsed_time", "remaining_time"):  # 末条为准
            if rec.get(k) not in (None, ""):
                meta[k] = rec[k]
        if "loss" in rec:
            points.append({
                "step": rec.get("current_steps"),
                "loss": rec.get("loss"),
                "epoch": rec.get("epoch"),
                "lr": rec.get("lr"),
            })
        if "eval_loss" in rec:
            points.append({
                "step": rec.get("current_steps"),
                "eval_loss": rec.get("eval_loss"),
                "epoch": rec.get("epoch"),
            })
    return {"points": points, "total_steps": total, **meta}


def gpu_status(cfg: RemoteConfig) -> list[dict[str, object]]:
    """Query GPUs in the same host/container runtime used by training."""
    script = _GPU_QUERY
    if cfg.docker_container:
        script = (
            f"docker exec {shlex.quote(cfg.docker_container)} sh -lc "
            f"{shlex.quote(_GPU_QUERY)}\n"
        )
    return parse_gpu_csv(run_remote_script(cfg, script, timeout=15))


def list_jobs(cfg: RemoteConfig) -> list[dict[str, object]]:
    """列出所有历史任务 (训练+评测) 及状态, 最新在前。"""
    return parse_job_list(run_remote_script(cfg, build_list_script(cfg), timeout=20))


def stop_job(cfg: RemoteConfig, job_id: str) -> dict[str, object]:
    """停止运行中的任务 (train/eval 通用)。"""
    out = run_remote_script(cfg, build_stop_script(cfg, job_id)).strip().splitlines()
    last = out[-1] if out else ""
    return {"stopped": last == "STOPPED", "detail": last}


def disk_free(cfg: RemoteConfig) -> int:
    """remote_root 所在盘可用字节数。"""
    out = run_remote_script(cfg, build_disk_script(cfg), timeout=15).strip()
    try:
        return int(out.splitlines()[-1]) * 1024
    except (ValueError, IndexError) as e:
        raise RemoteError("无法解析磁盘空间") from e


def scp_from(cfg: RemoteConfig, remote_path: str, local_path: str, timeout: int = 300) -> None:
    """从远端拉一个文件到本地。缺文件由调用方判定 (scp 失败即抛)。"""
    argv = ["scp", "-P", cfg.ssh_port, "-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]
    if cfg.ssh_identity:
        argv += ["-i", cfg.ssh_identity, "-o", "IdentitiesOnly=yes"]
    argv += [f"{cfg.ssh_target}:{remote_path}", local_path]
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        raise RemoteError("SCP 下载超时") from e
    except OSError as e:
        raise RemoteError(f"SCP 下载失败: {e}") from e
    if proc.returncode != 0:
        raise RemoteError(f"下载失败: {proc.stderr.strip()[:400]}")


def submit_job(
    cfg: RemoteConfig,
    job_id: str,
    yaml_text: str,
    dataset_info_text: str,
    local_data_path: str,
    data_file_name: str,
    gpus: str = "",
) -> None:
    """Atomically stage immutable inputs, then launch or safely replay."""
    digest = submission_digest(
        {
            "kind": "train",
            "yaml": yaml_text,
            "dataset_info": dataset_info_text,
            "data_file_name": data_file_name,
            "gpus": validate_gpus(gpus),
        },
        [local_data_path],
    )
    state = submission_state(cfg, job_id, digest)
    if state == "CONFLICT":
        raise RemoteConflictError("训练任务 ID 已绑定不同的不可变输入")
    if state == "MISSING":
        staging = (
            f"{cfg.remote_root.rstrip('/')}/.staging-"
            f"{validate_job_id(job_id)}-{secrets.token_hex(8)}"
        )
        run_remote_script(cfg, f"mkdir -p {shlex.quote(staging + '/data')}\n")
        _write_remote_file(cfg, f"{staging}/qwen3_lora_sft.yaml", yaml_text)
        _write_remote_file(
            cfg, f"{staging}/data/dataset_info.json", dataset_info_text
        )
        _write_remote_file(cfg, f"{staging}/gpus", validate_gpus(gpus))
        _scp(cfg, local_data_path, f"{staging}/data/{data_file_name}")
        if submission_digest(
            {
                "kind": "train",
                "yaml": yaml_text,
                "dataset_info": dataset_info_text,
                "data_file_name": data_file_name,
                "gpus": validate_gpus(gpus),
            },
            [local_data_path],
        ) != digest:
            raise RemoteConflictError("本地训练数据在上传期间发生变化")
        output = run_remote_script(
            cfg,
            build_finalize_submission_script(cfg, job_id, staging, digest),
        ).strip().splitlines()
        result = output[-1] if output else ""
        if result == "CONFLICT":
            raise RemoteConflictError("训练任务 ID 已绑定不同的不可变输入")
        if result not in {"CREATED", "SAME"}:
            raise RemoteError("远端训练输入原子发布失败")

    require_launch_ack(
        run_remote_script(cfg, build_launch_script(cfg, job_id, gpus))
    )


def require_launch_ack(output: str) -> str:
    """Accept only a launch protected by durable PID/status/terminal markers."""
    lines = output.strip().splitlines()
    result = lines[-1] if lines else ""
    if result not in {"STARTED", "ALREADY"}:
        raise RemoteError(
            f"remote launch acknowledgement missing or busy: {result or 'empty'}"
        )
    return result


def submission_digest(
    values: dict[str, object], local_paths: list[str] | tuple[str, ...] = ()
) -> str:
    digest = hashlib.sha256()
    digest.update(
        json.dumps(values, ensure_ascii=False, sort_keys=True).encode("utf-8")
    )
    for path in local_paths:
        digest.update(b"\0file\0")
        digest.update(os.path.basename(path).encode("utf-8"))
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def build_submission_state_script(
    cfg: RemoteConfig, job_id: str, digest: str
) -> str:
    d = job_dir(cfg, job_id)
    q = shlex.quote
    return "\n".join(
        [
            f"D={q(d)}",
            f"EXPECTED={q(digest)}",
            'if [ -f "$D/submission.sha256" ]; then',
            '  [ "$(cat "$D/submission.sha256")" = "$EXPECTED" ] && echo SAME || echo CONFLICT',
            'elif [ -e "$D" ]; then echo CONFLICT',
            'else echo MISSING; fi',
            "",
        ]
    )


def submission_state(cfg: RemoteConfig, job_id: str, digest: str) -> str:
    output = run_remote_script(
        cfg, build_submission_state_script(cfg, job_id, digest)
    ).strip().splitlines()
    state = output[-1] if output else ""
    if state not in {"MISSING", "SAME", "CONFLICT"}:
        raise RemoteError("无法确认远端任务输入状态")
    return state


def build_finalize_submission_script(
    cfg: RemoteConfig, job_id: str, staging: str, digest: str
) -> str:
    expected_prefix = (
        f"{cfg.remote_root.rstrip('/')}/.staging-{validate_job_id(job_id)}-"
    )
    if not staging.startswith(expected_prefix) or staging == expected_prefix:
        raise ValueError("invalid remote staging directory")
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("invalid submission digest")
    d = job_dir(cfg, job_id)
    q = shlex.quote
    return "\n".join(
        [
            "set -e",
            f"D={q(d)}",
            f"S={q(staging)}",
            f"EXPECTED={q(digest)}",
            'printf %s "$EXPECTED" > "$S/submission.sha256"',
            'if mv -T "$S" "$D" 2>/dev/null; then echo CREATED; exit 0; fi',
            'if [ -f "$D/submission.sha256" ] && '
            '[ "$(cat "$D/submission.sha256")" = "$EXPECTED" ]; then',
            '  rm -rf -- "$S"; echo SAME; exit 0',
            "fi",
            "echo CONFLICT",
            "",
        ]
    )


def _write_remote_file(cfg: RemoteConfig, remote_path: str, content: str) -> None:
    """把文本写到远端指定路径。

    内容用 base64 内联进脚本, 由 `base64 -d` 从管道解码写盘 —— 不与脚本本身竞争 stdin。
    (Ubuntu 的 /bin/sh=dash 会整块读取脚本 stdin, 早期 `cat > file` 方案会把内容当成命令执行。)
    """
    b64 = base64.b64encode(content.encode("utf-8")).decode("ascii")
    script = f"printf %s {shlex.quote(b64)} | base64 -d > {shlex.quote(remote_path)}\n"
    run_remote_script(cfg, script)


def job_status(cfg: RemoteConfig, job_id: str) -> dict[str, object]:
    try:
        out = run_remote_script(cfg, build_status_script(cfg, job_id))
    except RemoteError as e:
        return {"status": "unknown", "error": str(e)}
    return parse_status(out)


def stream_logs(
    cfg: RemoteConfig, job_id: str, from_line: int, log_name: str = "train.log"
) -> Iterator[tuple[int, str]]:
    """SSE 用: yield (行号, 内容)。行号从 from_line 起递增, 供 Last-Event-ID 续传。

    远端用 `tail -n +N -F` 持续跟随; 客户端断开时上层关闭进程。
    log_name 默认训练日志, 评测传 'eval.log'。
    """
    import select

    d = job_dir(cfg, job_id)
    start = max(1, from_line)
    remote_cmd = f"tail -n +{start} -F {shlex.quote(d + '/' + log_name)}"
    proc = subprocess.Popen(
        _ssh_argv(cfg) + [remote_cmd],
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
            ready, _, _ = select.select([fd], [], [], 15.0)
            if not ready:
                yield (line_no, "\0heartbeat")  # 上层转为 SSE 注释心跳
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
