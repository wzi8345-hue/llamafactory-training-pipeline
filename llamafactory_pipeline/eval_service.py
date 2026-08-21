"""Reusable registered-evaluation submission and scoring services."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Literal

from . import dataset_store, eval_judge, eval_remote, eval_schema, remote
from .assistant_diagnosis import compare_models
from .eval_schema import EvalRequest

EVAL_RESULTS = Path(__file__).parent / "eval_results"
_FROZEN_MANIFEST = "frozen_eval_manifest.json"


def _atomic_write_text(path: Path, content: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _request_sha256(req: EvalRequest) -> str:
    raw = json.dumps(
        req.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _persist_frozen_eval_manifest(
    req: EvalRequest,
    results_dir: Path,
    fc_local: str | None,
    subjective_local: str | None,
    submission_sha256: str,
) -> dict:
    """Persist the exact normalized rows and remote submission identity."""
    if len(set(req.task_types)) != len(req.task_types):
        raise ValueError("evaluation task_types must be unique")
    paths = {
        "function_call": fc_local,
        "subjective": subjective_local,
    }
    tasks: dict[str, dict] = {}
    all_ids: set[str] = set()
    for task_type in req.task_types:
        local = paths[task_type]
        if not local:
            raise ValueError(f"normalized evaluation file missing: {task_type}")
        path = Path(local)
        rows = eval_judge._load_jsonl(str(path))
        ids = [str(row.get("id") or "") for row in rows]
        if not ids or any(not item_id for item_id in ids):
            raise ValueError(f"frozen evaluation IDs missing: {task_type}")
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate frozen evaluation IDs: {task_type}")
        overlap = all_ids.intersection(ids)
        if overlap:
            raise ValueError(
                "evaluation IDs must be unique across tasks: "
                + sorted(overlap)[0]
            )
        all_ids.update(ids)
        tasks[task_type] = {
            "file": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "n": len(ids),
            "ids": sorted(ids),
        }
    manifest = {
        "version": 1,
        "request_sha256": _request_sha256(req),
        "submission_sha256": submission_sha256,
        "tasks": tasks,
    }
    _atomic_write_text(
        results_dir / _FROZEN_MANIFEST,
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2),
    )
    return manifest


def _load_verified_frozen_items(
    cfg: remote.RemoteConfig,
    eval_id: str,
    req: EvalRequest,
    results_dir: Path,
) -> tuple[list[dict], list[dict], set[tuple[str, str]]]:
    manifest_path = results_dir / _FROZEN_MANIFEST
    if not manifest_path.is_file():
        raise ValueError("frozen evaluation manifest is missing")
    try:
        manifest = json.loads(manifest_path.read_text("utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("frozen evaluation manifest is invalid") from exc
    if (
        not isinstance(manifest, dict)
        or manifest.get("version") != 1
        or manifest.get("request_sha256") != _request_sha256(req)
    ):
        raise ValueError("frozen evaluation request hash changed")
    tasks = manifest.get("tasks")
    if not isinstance(tasks, dict) or set(tasks) != set(req.task_types):
        raise ValueError("frozen evaluation task manifest changed")
    paths: dict[str, str | None] = {
        "function_call": None,
        "subjective": None,
    }
    rows_by_task: dict[str, list[dict]] = {
        "function_call": [],
        "subjective": [],
    }
    expected_keys: set[tuple[str, str]] = set()
    for task_type in req.task_types:
        expected_name = (
            "fc.items.jsonl"
            if task_type == "function_call"
            else "subjective.items.jsonl"
        )
        entry = tasks.get(task_type)
        if not isinstance(entry, dict) or entry.get("file") != expected_name:
            raise ValueError("frozen evaluation file manifest changed")
        path = results_dir / expected_name
        if not path.is_file():
            raise ValueError("frozen evaluation file is missing")
        actual_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_sha256 != entry.get("sha256"):
            raise ValueError("frozen evaluation file hash changed")
        rows = eval_judge._load_jsonl(str(path))
        ids = [str(row.get("id") or "") for row in rows]
        if (
            ids == []
            or len(ids) != int(entry.get("n") or -1)
            or sorted(ids) != entry.get("ids")
            or len(ids) != len(set(ids))
        ):
            raise ValueError("frozen evaluation ID manifest changed")
        paths[task_type] = str(path)
        rows_by_task[task_type] = rows
        expected_keys.update((task_type, item_id) for item_id in ids)
    local_digest = eval_remote.evaluation_submission_digest(
        cfg,
        eval_id,
        req,
        paths["function_call"],
        paths["subjective"],
    )
    if local_digest != manifest.get("submission_sha256"):
        raise ValueError("frozen evaluation submission hash changed")
    if remote.submission_state(cfg, eval_id, local_digest) != "SAME":
        raise ValueError("remote frozen evaluation submission is not identical")
    return (
        rows_by_task["function_call"],
        rows_by_task["subjective"],
        expected_keys,
    )


def _read_records(path: Path, ext: str) -> list[dict]:
    text = path.read_text("utf-8").strip()
    if ext == ".jsonl":
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        records = json.loads(text)
    if not isinstance(records, list):
        raise ValueError("evaluation dataset must be a JSON array or JSONL records")
    return records


def _write_jsonl(path: Path, records: list[dict]) -> str:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records),
        encoding="utf-8",
    )
    return str(path)


def normalize_registered_evalset(
    dataset_name: str | None,
    task_type: Literal["function_call", "subjective"],
    results_dir: Path,
    output_name: str,
) -> str | None:
    if not dataset_name:
        return None
    path = dataset_store.data_path(dataset_name, "eval")
    meta = dataset_store.dataset_meta(dataset_name, "eval")
    if meta is None or not path.exists():
        raise ValueError(f"evaluation dataset not found: {dataset_name}")
    records = _read_records(path, str(meta.get("ext", ".json")))
    normalized = eval_schema.validate_evalset(records, task_type)
    return _write_jsonl(results_dir / f"{output_name}.items.jsonl", normalized)


def submit_normalized_eval(
    cfg: remote.RemoteConfig,
    req: EvalRequest,
    eval_id: str,
    results_dir: Path,
    fc_local: str | None,
    subjective_local: str | None,
) -> str:
    """Persist an already-normalized request, submit it, and record selected GPUs."""
    req.validate_names()
    results_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(
        results_dir / "config.json", req.model_dump_json(indent=2)
    )
    submission_sha256 = eval_remote.evaluation_submission_digest(
        cfg, eval_id, req, fc_local, subjective_local
    )
    _persist_frozen_eval_manifest(
        req,
        results_dir,
        fc_local,
        subjective_local,
        submission_sha256,
    )
    eval_remote.submit_eval(cfg, eval_id, req, fc_local, subjective_local)
    return eval_id


def submit_registered_eval(
    cfg: remote.RemoteConfig,
    req: EvalRequest,
    fc_dataset_name: str | None,
    subjective_dataset_name: str | None,
    *,
    results_root: Path | None = None,
    eval_id: str | None = None,
) -> str:
    root = Path(results_root) if results_root is not None else EVAL_RESULTS
    eval_id = eval_id or eval_remote.new_eval_id()
    remote.validate_job_id(eval_id)
    need_fc = "function_call" in req.task_types
    need_subjective = "subjective" in req.task_types
    if need_fc and not fc_dataset_name:
        raise ValueError("function_call evaluation requires a registered dataset")
    if need_subjective and not subjective_dataset_name:
        raise ValueError("subjective evaluation requires a registered dataset")
    results_dir = root / eval_id
    results_dir.mkdir(parents=True, exist_ok=True)
    fc_local = normalize_registered_evalset(
        fc_dataset_name if need_fc else None,
        "function_call",
        results_dir,
        "fc",
    )
    subjective_local = normalize_registered_evalset(
        subjective_dataset_name if need_subjective else None,
        "subjective",
        results_dir,
        "subjective",
    )
    return submit_normalized_eval(
        cfg, req, eval_id, results_dir, fc_local, subjective_local
    )


def score_registered_eval(
    cfg: remote.RemoteConfig,
    eval_id: str,
    *,
    results_root: Path | None = None,
    critical_tags: list[str] | tuple[str, ...] = (),
) -> dict:
    remote.validate_job_id(eval_id)
    root = Path(results_root) if results_root is not None else EVAL_RESULTS
    results_dir = root / eval_id
    config_path = results_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"evaluation config not found: {eval_id}")
    req = EvalRequest.model_validate_json(config_path.read_text("utf-8"))
    fc_items, subjective_items, expected_prediction_keys = (
        _load_verified_frozen_items(cfg, eval_id, req, results_dir)
    )
    summary = eval_judge.run_scoring(
        cfg,
        eval_id,
        req,
        fc_items,
        subjective_items,
        str(results_dir),
        expected_prediction_keys=expected_prediction_keys,
    )
    if len(req.models) >= 2:
        baseline_scores = eval_judge._load_jsonl(
            str(results_dir / f"{req.models[0].name}.scores.jsonl")
        )
        candidate_scores = eval_judge._load_jsonl(
            str(results_dir / f"{req.models[1].name}.scores.jsonl")
        )
        if baseline_scores or candidate_scores:
            summary["paired_comparison"] = compare_models(
                baseline_scores,
                candidate_scores,
                critical_tags=critical_tags,
            )
            report = eval_judge.build_report_md(
                eval_id,
                {
                    name: value
                    for name, value in (summary.get("per_model") or {}).items()
                    if "error" not in value
                },
                summary["paired_comparison"],
            )
            (results_dir / "report.md").write_text(report, encoding="utf-8")
            summary["report_md"] = report
    return summary
