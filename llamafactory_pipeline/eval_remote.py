"""评测推理阶段: 生成 infer.yaml/runner.py/run_eval.sh, 提交服务器持久任务, 拉回 predictions。

复用 remote.py 的 ssh/scp/job 目录/状态/日志, 不重写连接层。推理在容器内起
`llamafactory-cli api` (vLLM) 端点, runner 遍历评测集出 predictions.jsonl。
"""

from __future__ import annotations

import hashlib
import secrets
import shlex

import yaml

from . import remote
from .eval_schema import EvalRequest, ModelUnderTest

# runner 只用 Python 标准库 (urllib), 避免容器内缺 requests/openai。
RUNNER_PY = r'''#!/usr/bin/env python3
"""服务器端评测 runner: 等端点就绪 → 遍历评测集 → 写 predictions.jsonl。仅用标准库。"""
import argparse, json, os, sys, time, urllib.request, urllib.error, urllib.parse

def _post(api, path, payload, timeout=300):
    req = urllib.request.Request(api.rstrip("/") + path,
        data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())

def _get(api, path, timeout=10):
    with urllib.request.urlopen(api.rstrip("/") + path, timeout=timeout) as r:
        return json.loads(r.read().decode())

def _listener_owned_by_process_tree(api_pid, api):
    """Prove that api_pid or one of its descendants owns the listening socket."""
    parsed = urllib.parse.urlparse(api)
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    socket_inodes = set()
    for table in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            lines = open(table, encoding="ascii").read().splitlines()[1:]
        except OSError:
            continue
        for line in lines:
            fields = line.split()
            if len(fields) < 10 or fields[3] != "0A":
                continue
            try:
                local_port = int(fields[1].rsplit(":", 1)[1], 16)
            except (IndexError, ValueError):
                continue
            if local_port == port:
                socket_inodes.add(fields[9])
    if not socket_inodes:
        return False

    parents = {}
    try:
        proc_entries = [entry for entry in os.listdir("/proc") if entry.isdigit()]
    except OSError:
        return False
    for entry in proc_entries:
        try:
            stat = open(f"/proc/{entry}/stat", encoding="ascii").read()
            parents[int(entry)] = int(stat.split(") ", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            continue
    owned_pids = {api_pid}
    changed = True
    while changed:
        changed = False
        for pid, parent in parents.items():
            if parent in owned_pids and pid not in owned_pids:
                owned_pids.add(pid)
                changed = True
    for pid in owned_pids:
        try:
            fds = os.listdir(f"/proc/{pid}/fd")
        except OSError:
            continue
        for fd in fds:
            try:
                target = os.readlink(f"/proc/{pid}/fd/{fd}")
            except OSError:
                continue
            if target.startswith("socket:[") and target[8:-1] in socket_inodes:
                return True
    return False

def wait_ready(api, timeout, api_pid, expected_model):
    end = time.time() + timeout
    while time.time() < end:
        try:
            os.kill(api_pid, 0)
        except OSError as exc:
            raise RuntimeError("spawned model API exited before readiness") from exc
        try:
            data = _get(api, "/models")
            ids = [m["id"] for m in data.get("data", [])]
            if ids:
                matches = [model_id for model_id in ids if model_id == expected_model]
                if not matches:
                    raise RuntimeError(
                        "model endpoint identity mismatch: " + ",".join(ids)
                    )
                if not _listener_owned_by_process_tree(api_pid, api):
                    raise RuntimeError(
                        "model endpoint listener is not owned by spawned API process"
                    )
                return matches[0]
        except RuntimeError:
            raise
        except Exception:
            pass
        time.sleep(3)
    return None

def _messages(item):
    msgs = []
    if item.get("system"):
        msgs.append({"role": "system", "content": item["system"]})
    msgs.append({"role": "user", "content": item["query"]})
    return msgs

def _load(path):
    if not path or not os.path.exists(path) or os.path.getsize(path) == 0:
        return []
    txt = open(path, encoding="utf-8").read().strip()
    if path.endswith(".jsonl"):
        return [json.loads(l) for l in txt.splitlines() if l.strip()]
    return json.loads(txt)

def run_fc(api, model, items, out, adapter=None):
    for it in items:
        rec = {"id": it["id"], "task_type": "function_call"}
        started = None
        try:
            body = {"model": model,
                "messages": _messages(it), "tools": it["tools"],
                "tool_choice": "auto", "temperature": 0}
            if adapter:
                body["adapter_name"] = adapter
            started = time.perf_counter()
            resp = _post(api, "/chat/completions", body)
            choice = resp["choices"][0]
            rec["finish_reason"] = choice.get("finish_reason")
            msg = choice["message"]
            calls = msg.get("tool_calls") or []
            if calls:
                fn = calls[0]["function"]
                rec["pred_name"] = fn.get("name")
                try:
                    rec["pred_arguments"] = json.loads(fn.get("arguments") or "{}")
                except Exception:
                    rec["pred_arguments"] = {"_raw": fn.get("arguments")}
                    rec["invalid_reason"] = "invalid_arguments_json"
            else:
                rec["pred_name"] = None
                rec["pred_arguments"] = {}
                rec["no_tool_call"] = True
                rec["invalid_reason"] = "no_tool_call"
            rec["raw"] = msg
        except Exception as e:
            rec["error"] = str(e)[:300]
            rec["invalid_reason"] = "request_error"
        finally:
            if started is not None:
                rec["latency_ms"] = round((time.perf_counter() - started) * 1000, 3)
        _append(out, rec)

def run_subjective(api, model, items, out, adapter=None):
    for it in items:
        rec = {"id": it["id"], "task_type": "subjective"}
        started = None
        try:
            body = {"model": model,
                "messages": _messages(it), "temperature": 0}
            if adapter:
                body["adapter_name"] = adapter
            started = time.perf_counter()
            resp = _post(api, "/chat/completions", body)
            choice = resp["choices"][0]
            rec["finish_reason"] = choice.get("finish_reason")
            rec["answer"] = choice["message"].get("content", "")
            if not rec["answer"].strip():
                rec["invalid_reason"] = "empty_answer"
        except Exception as e:
            rec["error"] = str(e)[:300]
            rec["invalid_reason"] = "request_error"
        finally:
            if started is not None:
                rec["latency_ms"] = round((time.perf_counter() - started) * 1000, 3)
        _append(out, rec)

def _append(out, rec):
    with open(out, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ready-timeout", type=int, default=600)
    ap.add_argument("--api-pid", type=int, required=True)
    ap.add_argument("--expected-model", required=True)
    ap.add_argument("--fc", default="")
    ap.add_argument("--subjective", default="")
    ap.add_argument("--adapter", default="", help="多 adapter 端点切换: 传 adapter 路径")
    a = ap.parse_args()
    open(a.out, "w").close()  # 清空/新建
    model = wait_ready(a.api, a.ready_timeout, a.api_pid, a.expected_model)
    if not model:
        print("[runner] 端点未就绪, 放弃", flush=True)
        sys.exit(3)
    print(f"[runner] 使用模型 id: {model}" + (f" adapter={a.adapter}" if a.adapter else ""),
          flush=True)
    fc = _load(a.fc)
    adapter = a.adapter or None
    if fc:
        print(f"[runner] FC {len(fc)} 条", flush=True); run_fc(a.api, model, fc, a.out, adapter)
    subj = _load(a.subjective)
    if subj:
        print(f"[runner] 主观 {len(subj)} 条", flush=True); run_subjective(a.api, model, subj, a.out, adapter)
    print("[runner] 完成", flush=True)

if __name__ == "__main__":
    main()
'''


def new_eval_id() -> str:
    return remote.new_job_id()


def eval_dir(cfg: remote.RemoteConfig, eval_id: str) -> str:
    return remote.job_dir(cfg, eval_id)


def read_train_job(cfg: remote.RemoteConfig, job_id: str) -> dict:
    """读远端某训练任务的 YAML, 解析 base/adapter/template 供微调模型评测。"""
    remote.validate_job_id(job_id)
    d = remote.job_dir(cfg, job_id)
    out = remote.run_remote_script(cfg, f"cat {shlex.quote(d + '/qwen3_lora_sft.yaml')}\n")
    cfg_yaml = yaml.safe_load(out) or {}
    adapter_path = str(cfg_yaml.get("output_dir", "")).strip()
    if adapter_path and not adapter_path.startswith("/"):
        adapter_path = cfg.llamafactory_dir.rstrip("/") + "/" + adapter_path
    return {
        "model_name_or_path": cfg_yaml.get("model_name_or_path", ""),
        "adapter_path": adapter_path,
        "template": cfg_yaml.get("template", "qwen3_5_nothink"),
    }


def build_infer_yaml(model: ModelUnderTest) -> str:
    conf: dict = {
        "model_name_or_path": model.model_name_or_path,
        "template": model.template,
        "infer_backend": "vllm",
        "trust_remote_code": True,
    }
    if model.adapter_path:
        conf["adapter_name_or_path"] = model.adapter_path
    return yaml.safe_dump(conf, allow_unicode=True, sort_keys=False)


def group_models(models: list[ModelUnderTest]) -> list[list[ModelUnderTest]]:
    """把同基座 (相同 model_name_or_path) 且都有 adapter 的微调模型分到一组,
    复用一次 vLLM 冷启动加载基座权重; 纯基座模型各自一组。

    同基座但无 adapter 的 (纯基座) 不合并: 没有切换意义, 且避免无谓耦合。
    不同基座不合并 (无法共享权重加载)。
    """
    groups: list[list[ModelUnderTest]] = []
    # 只合并 "同基座 + 都有 adapter" 的; 其余 (纯基座 / 单 adapter) 各自一组
    by_base: dict[tuple[str, str], list[ModelUnderTest]] = {}
    solo: list[list[ModelUnderTest]] = []
    for m in models:
        if m.adapter_path:
            by_base.setdefault((m.model_name_or_path, m.template), []).append(m)
        else:
            solo.append([m])
    for grp in by_base.values():
        groups.append(grp if len(grp) > 1 else grp)  # 单 adapter 也作为一个组
    return groups + solo


def build_group_infer_yaml(group: list[ModelUnderTest]) -> str:
    """同基座多 adapter 的合并 infer yaml: 一次加载基座 + 所有 adapter。
    adapter_name_or_path 逗号分隔; runner 用对应路径作 adapter_name 切换。"""
    if len(group) == 1:
        return build_infer_yaml(group[0])
    if len({(model.model_name_or_path, model.template) for model in group}) != 1:
        raise ValueError("grouped evaluation models must share base model and template")
    base = group[0].model_name_or_path
    adapters = ",".join(m.adapter_path for m in group)
    conf: dict = {
        "model_name_or_path": base,
        "template": group[0].template,
        "infer_backend": "vllm",
        "trust_remote_code": True,
        "adapter_name_or_path": adapters,
    }
    return yaml.safe_dump(conf, allow_unicode=True, sort_keys=False)


def evaluation_service_id(eval_id: str, group: list[ModelUnderTest]) -> str:
    """Return the deterministic, approval-bound model id exposed by this API."""
    remote.validate_job_id(eval_id)
    material = "\0".join(
        [
            eval_id,
            group[0].model_name_or_path,
            group[0].template,
            *(model.name for model in group),
            *(model.adapter_path or "" for model in group),
        ]
    )
    return "lf-eval-" + hashlib.sha256(material.encode("utf-8")).hexdigest()[:20]


def build_run_eval_script(
    cfg: remote.RemoteConfig, eval_id: str, req: EvalRequest,
    has_fc: bool, has_subj: bool,
) -> str:
    """生成容器内评测各模型的编排脚本 (POSIX sh)。

    同基座多 LoRA 合并到一个 vLLM 端点, 一次冷启动加载基座权重, 组内每模型
    用 --adapter <path> 切换 (省 N-1 次权重加载); 不同基座各自起端点。
    """
    d = eval_dir(cfg, eval_id)
    q = shlex.quote
    port = int(req.api_port)
    fc_arg = f"--fc {q(d + '/evalset/fc.jsonl')} " if has_fc else ""
    subj_arg = f"--subjective {q(d + '/evalset/subjective.jsonl')} " if has_subj else ""

    port_lock = q(f"/tmp/llamafactory-assistant-eval-port-{port}.lock")
    lines = [
        "set -u",
        f"exec 8>{port_lock}",
        "flock -n 8 || { echo 'evaluation port is reserved'; exit 20; }",
    ]
    for group in group_models(req.models):
        # 组 yaml: 同基座共用一个 infer yaml
        if len(group) > 1:
            yaml_path = q(f"{d}/models/group_{group[0].name}.infer.yaml")
        else:
            yaml_path = q(f"{d}/models/{group[0].name}.infer.yaml")
        api_log = q(f"{d}/{group[0].name}.api.log")
        service_id = evaluation_service_id(eval_id, group)
        expected_model = q(service_id)
        lines += [
            f"echo '=== base {group[0].model_name_or_path} ({len(group)} models) ==='",
            f"if ! python -c {q('import socket,sys; s=socket.socket(); s.bind((\"127.0.0.1\", int(sys.argv[1]))); s.close()')} {port}; then echo 'port already in use'; exit 20; fi",
            f"API_MODEL_NAME={q(service_id)} API_PORT={port} "
            f"llamafactory-cli api {yaml_path} > {api_log} 2>&1 &",
            "API_PID=$!",
        ]
        for m in group:
            pred = q(f"{d}/{m.name}.predictions.jsonl")
            adapter_arg = f"--adapter {q(m.adapter_path)} " if m.adapter_path else ""
            lines += [
                f"echo '--- model {m.name} ---'",
                f"if ! python {q(d + '/runner.py')} --api http://127.0.0.1:{port}/v1 "
                f"--out {pred} --ready-timeout {int(req.ready_timeout)} "
                f"--api-pid $API_PID --expected-model {expected_model} "
                f"{adapter_arg}{fc_arg}{subj_arg}"
                f"; then kill $API_PID 2>/dev/null || true; wait $API_PID 2>/dev/null || true; exit 21; fi",
            ]
        lines += [
            "kill $API_PID 2>/dev/null || true",
            "wait $API_PID 2>/dev/null || true",
        ]
    lines.append("echo '=== all models done ==='")
    lines.append("")
    return "\n".join(lines)


def build_eval_launch(cfg: remote.RemoteConfig, eval_id: str, gpus: str) -> str:
    """nohup 包装 run_eval.sh (docker 模式走 docker exec), 复用训练的持久化/状态模型。"""
    gpus = remote.validate_gpus(gpus)
    d = eval_dir(cfg, eval_id)
    q = shlex.quote
    script = d + "/run_eval.sh"

    if cfg.docker_container:
        parts = ["docker", "exec"]
        if gpus:
            parts += ["-e", f"CUDA_VISIBLE_DEVICES={gpus}"]
        parts += ["-w", cfg.llamafactory_dir, cfg.docker_container, "bash", script]
        run_cmd = " ".join(q(p) for p in parts)
        header = "set -e"
    else:
        prefix = f"CUDA_VISIBLE_DEVICES={q(gpus)} " if gpus else ""
        run_cmd = f"{prefix}bash {q(script)}"
        header = remote._host_header(cfg)

    return remote._build_durable_launch_wrapper(
        header, d, run_cmd, d + "/eval.log"
    )


def _evaluation_submission_components(
    cfg: remote.RemoteConfig,
    eval_id: str,
    req: EvalRequest,
    fc_local: str | None,
    subj_local: str | None,
) -> tuple[dict[str, str], str, str]:
    """Build the immutable evaluation payload and its content digest."""
    infer_files: dict[str, str] = {}
    for group in group_models(req.models):
        if len(group) > 1:
            relative = f"models/group_{group[0].name}.infer.yaml"
            infer_files[relative] = build_group_infer_yaml(group)
        else:
            model = group[0]
            infer_files[f"models/{model.name}.infer.yaml"] = build_infer_yaml(
                model
            )
    script = build_run_eval_script(
        cfg, eval_id, req, bool(fc_local), bool(subj_local)
    )
    digest = remote.submission_digest(
        {
            "kind": "evaluation",
            "request": req.model_dump(mode="json"),
            "infer_files": infer_files,
            "runner": RUNNER_PY,
            "run_script": script,
        },
        [path for path in (fc_local, subj_local) if path],
    )
    return infer_files, script, digest


def evaluation_submission_digest(
    cfg: remote.RemoteConfig,
    eval_id: str,
    req: EvalRequest,
    fc_local: str | None,
    subj_local: str | None,
) -> str:
    """Recompute the digest bound to the immutable remote evaluation job."""
    return _evaluation_submission_components(
        cfg, eval_id, req, fc_local, subj_local
    )[2]


def submit_eval(
    cfg: remote.RemoteConfig, eval_id: str, req: EvalRequest,
    fc_local: str | None, subj_local: str | None,
) -> None:
    """Atomically publish immutable eval inputs, then launch or replay."""
    infer_files, script, digest = _evaluation_submission_components(
        cfg, eval_id, req, fc_local, subj_local
    )
    state = remote.submission_state(cfg, eval_id, digest)
    if state == "CONFLICT":
        raise remote.RemoteConflictError("评测任务 ID 已绑定不同的不可变输入")
    if state == "MISSING":
        staging = (
            f"{cfg.remote_root.rstrip('/')}/.staging-"
            f"{remote.validate_job_id(eval_id)}-{secrets.token_hex(8)}"
        )
        remote.run_remote_script(
            cfg,
            f"mkdir -p {shlex.quote(staging + '/evalset')} "
            f"{shlex.quote(staging + '/models')}\n",
        )
        for relative, content in infer_files.items():
            remote._write_remote_file(cfg, f"{staging}/{relative}", content)
        remote._write_remote_file(cfg, f"{staging}/runner.py", RUNNER_PY)
        remote._write_remote_file(cfg, f"{staging}/run_eval.sh", script)
        remote._write_remote_file(cfg, f"{staging}/gpus", req.gpus)
        if fc_local:
            remote._scp(cfg, fc_local, f"{staging}/evalset/fc.jsonl")
        if subj_local:
            remote._scp(cfg, subj_local, f"{staging}/evalset/subjective.jsonl")
        if evaluation_submission_digest(
            cfg, eval_id, req, fc_local, subj_local
        ) != digest:
            raise remote.RemoteConflictError("本地评测数据在上传期间发生变化")
        output = remote.run_remote_script(
            cfg,
            remote.build_finalize_submission_script(
                cfg, eval_id, staging, digest
            ),
        ).strip().splitlines()
        result = output[-1] if output else ""
        if result == "CONFLICT":
            raise remote.RemoteConflictError("评测任务 ID 已绑定不同的不可变输入")
        if result not in {"CREATED", "SAME"}:
            raise remote.RemoteError("远端评测输入原子发布失败")
    remote.require_launch_ack(
        remote.run_remote_script(cfg, build_eval_launch(cfg, eval_id, req.gpus))
    )


def fetch_predictions(
    cfg: remote.RemoteConfig, eval_id: str, model_name: str, local_path: str
) -> None:
    d = eval_dir(cfg, eval_id)
    remote.scp_from(cfg, f"{d}/{model_name}.predictions.jsonl", local_path)
