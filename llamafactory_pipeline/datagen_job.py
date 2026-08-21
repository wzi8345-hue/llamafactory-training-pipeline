"""本地数据生成 job 管理: detached 子进程启动 / 状态 / 日志 tail。

与训练不同, 生成不吃 GPU, 跑在本机。用 start_new_session 脱离后端进程,
后端重启不影响生成; 产出与状态都落在 job 目录 (唯一事实来源)。
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterator

_REPO = Path(__file__).resolve().parents[1]
_BASE = _REPO / "sft_data" / "generated"
_JOB_ID_RE = re.compile(r"^\d{8}T\d{6}Z-[0-9a-f]{6}$")


def _atomic_write_text(path: Path, content: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, path)


def _atomic_write_json(path: Path, value: dict) -> None:
    _atomic_write_text(
        path, json.dumps(value, ensure_ascii=False, indent=2)
    )


def _process_identity(pid: int) -> str:
    """Return a stable process creation identity, not merely a reusable PID."""
    proc_stat = Path(f"/proc/{pid}/stat")
    try:
        fields = proc_stat.read_text("utf-8").split(") ", 1)[1].split()
        if len(fields) > 19:
            return "proc:" + fields[19]
    except (OSError, IndexError):
        pass
    try:
        result = subprocess.run(
            ["ps", "-o", "lstart=", "-p", str(pid)],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        result = None
    if result is not None:
        value = " ".join(result.stdout.split())
        if result.returncode == 0 and value:
            return "ps:" + value
    return ""


def job_dir(job_id: str) -> Path:
    return _BASE / job_id


def list_jobs() -> list[dict]:
    """列出所有生成任务 (最新在前), 复用 status() 判态。"""
    if not _BASE.exists():
        return []
    jobs = []
    for d in _BASE.iterdir():
        if not (d.is_dir() and _JOB_ID_RE.match(d.name)):
            continue
        st = status(d.name)
        # 任务类型从 config.json 读；历史任务没有 finetune_type 时按 SFT。
        task_type = ""
        finetune_type = "sft"
        cfg_p = d / "config.json"
        if cfg_p.exists():
            try:
                cfg = json.loads(cfg_p.read_text("utf-8"))
                task_type = cfg.get("task_type", "")
                finetune_type = cfg.get("finetune_type", "sft")
            except json.JSONDecodeError:
                pass
        jobs.append({"job_id": d.name, "status": st["status"],
                     "accepted": st.get("accepted", 0),
                     "target": st.get("target", 0),
                     "task_type": task_type, "finetune_type": finetune_type})
    jobs.sort(key=lambda j: j["job_id"], reverse=True)
    return jobs


def create_and_launch(job_id: str, config: dict) -> None:
    """幂等地写入配置并以 detached 子进程启动生成。"""
    d = job_dir(job_id)
    d.mkdir(parents=True, exist_ok=True)
    config_path = d / "config.json"
    if config_path.exists():
        try:
            existing = json.loads(config_path.read_text("utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"数据生成任务配置已损坏: {job_id}") from exc
        if existing != config:
            raise ValueError(f"数据生成任务 ID 配置冲突: {job_id}")
        progress = {}
        progress_path = d / "progress.json"
        if progress_path.exists():
            try:
                progress = json.loads(progress_path.read_text("utf-8"))
            except json.JSONDecodeError:
                progress = {}
        if progress.get("state") in {"done", "error"} or _pid_alive(d):
            return
        # A crashed detached child may leave a stale pid and running progress.
        # The runner owns a non-blocking run.lock and resumes output.jsonl, so
        # relaunching with the same immutable config is safe.
        (d / "pid").unlink(missing_ok=True)
        (d / "pid_identity").unlink(missing_ok=True)
    else:
        _atomic_write_json(config_path, config)
        progress = {}
    progress_path = d / "progress.json"
    _atomic_write_json(
        progress_path,
        {
            **progress,
            "state": "starting",
            "target": config.get("count", progress.get("target", 0)),
            "updated_at": time.time(),
        },
    )
    logf = (d / "run.log").open("a", encoding="utf-8")
    try:
        process = subprocess.Popen(
            [sys.executable, "-m", "llamafactory_pipeline.datagen_run", str(d)],
            stdout=logf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            cwd=str(_REPO), start_new_session=True,
        )
    finally:
        logf.close()
    identity = _process_identity(process.pid)
    if not identity:
        process.terminate()
        raise RuntimeError("无法记录数据生成进程身份")
    _atomic_write_text(d / "pid", str(process.pid))
    _atomic_write_text(d / "pid_identity", identity)


def _pid_alive(d: Path) -> bool:
    pid_file = d / "pid"
    if not pid_file.exists():
        return False
    try:
        pid = int(pid_file.read_text().strip())
        identity_file = d / "pid_identity"
        if not identity_file.exists():
            return False
        expected = identity_file.read_text("utf-8").strip()
        if not expected:
            return False
        os.kill(pid, 0)
        return bool(expected) and _process_identity(pid) == expected
    except (OSError, ValueError, ProcessLookupError, PermissionError):
        return False


def status(job_id: str) -> dict:
    d = job_dir(job_id)
    if not d.exists():
        return {"status": "not_found"}
    prog = {}
    p = d / "progress.json"
    if p.exists():
        try:
            prog = json.loads(p.read_text("utf-8"))
        except json.JSONDecodeError:
            pass
    state = prog.get("state")
    alive = _pid_alive(d)
    if state == "done":
        accepted = prog.get("accepted")
        target = prog.get("target")
        st = (
            "failed"
            if isinstance(accepted, int)
            and isinstance(target, int)
            and accepted < target
            else "succeeded"
        )
    elif state == "error":
        st = "failed"
    elif alive:
        st = "running"
    elif (
        state == "starting"
        and isinstance(prog.get("updated_at"), (int, float))
        and time.time() - float(prog["updated_at"]) < 30
    ):
        st = "running"
    else:
        # A durable directory plus running progress and a dead local PID is a
        # terminal interruption, not an unknowable remote observation.
        st = "interrupted"
    return {"status": st, "finetune_type": job_finetune_type(job_id), **prog}


def job_finetune_type(job_id: str) -> str:
    """读取任务微调类型；缺字段或配置损坏时保持历史 SFT 兼容。"""
    p = job_dir(job_id) / "config.json"
    if not p.exists():
        return "sft"
    try:
        value = json.loads(p.read_text("utf-8")).get("finetune_type", "sft")
    except (OSError, json.JSONDecodeError):
        return "sft"
    return value if value in ("sft", "dpo") else "sft"


def stop(job_id: str) -> dict:
    d = job_dir(job_id)
    pid_file = d / "pid"
    if not pid_file.exists():
        return {"stopped": False, "detail": "无 pid"}
    try:
        if not _pid_alive(d):
            return {"stopped": False, "detail": "进程身份已失效"}
        os.kill(int(pid_file.read_text().strip()), signal.SIGTERM)
        return {"stopped": True, "detail": "已发送 SIGTERM"}
    except (ValueError, ProcessLookupError, PermissionError) as e:
        return {"stopped": False, "detail": str(e)}


def tail_logs(job_id: str, from_line: int = 1) -> Iterator[tuple[int, str]]:
    """SSE 用: 增量读本地 run.log。进程结束且无新行则收尾。"""
    d = job_dir(job_id)
    log = d / "run.log"
    sent = 0
    start = max(1, from_line)
    idle = 0
    while True:
        lines = log.read_text("utf-8").splitlines() if log.exists() else []
        while sent < len(lines):
            sent += 1
            if sent >= start:
                yield (sent, lines[sent - 1])
        if not _pid_alive(d):
            idle += 1
            if idle >= 2 and sent >= len(lines):  # 给收尾日志一点时间
                break
        yield (sent, "\0heartbeat")
        time.sleep(2)
