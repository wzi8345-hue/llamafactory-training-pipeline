"""Freeze generated training data, create holdouts, and build reproducible profiles."""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from . import dataset_store, schema
from .assistant_schema import DatasetProfile

EvalKind = Literal["function_call", "subjective"]
TaskType = Literal["qa", "qa_multi", "fc"]


@dataclass(frozen=True)
class PreparedDataset:
    train_path: Path
    eval_paths: dict[EvalKind, Path]
    dataset_name: str
    eval_dataset_names: dict[EvalKind, str]
    profile: DatasetProfile


@dataclass(frozen=True)
class _PreparedRecord:
    record: dict[str, Any]
    task_type: TaskType
    eval_kind: EvalKind
    eval_row: dict[str, Any]
    profile_text: str
    label: str | None = None


def holdout_count(n: int, ratio: float) -> int:
    # Small datasets are allowed as smoke/manual-review evidence. Automatic
    # acceptance remains disabled below 30 paired rows by the diagnosis gate.
    if n < 2:
        raise ValueError("at least 2 records are required per task")
    if n < 30:
        return min(n - 1, max(1, round(n * 0.20)))
    if n < 200:
        return min(n - 1, max(20, round(n * ratio)))
    return min(n - 1, max(1, round(n * ratio)))


def estimate_tokens(text: str) -> int:
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    ascii_nonspace = sum(1 for ch in text if ord(ch) < 128 and not ch.isspace())
    other = sum(
        1
        for ch in text
        if ord(ch) >= 128
        and not ("\u4e00" <= ch <= "\u9fff")
        and not ch.isspace()
    )
    return max(1, cjk + (ascii_nonspace + 3) // 4 + other)


def _read_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        records = [
            json.loads(line)
            for line in path.read_text("utf-8").splitlines()
            if line.strip()
        ]
    else:
        records = json.loads(path.read_text("utf-8"))
    if not isinstance(records, list) or not all(isinstance(row, dict) for row in records):
        raise ValueError(f"generated dataset must be a list of objects: {path}")
    return records


def _task_hint(path: Path) -> TaskType | None:
    config_path = path.parent / "config.json"
    if not config_path.exists():
        return None
    try:
        task_type = json.loads(config_path.read_text("utf-8")).get("task_type")
    except (OSError, json.JSONDecodeError, AttributeError):
        return None
    return task_type if task_type in ("qa", "qa_multi", "fc") else None


def _message_value(message: Any) -> str:
    if not isinstance(message, dict):
        return ""
    value = message.get("value", message.get("content", ""))
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _role(message: Any) -> str:
    if not isinstance(message, dict):
        return ""
    return str(message.get("from", message.get("role", "")))


def _conversation(record: dict[str, Any]) -> list[dict[str, Any]]:
    messages = record.get("conversations", record.get("messages", []))
    return messages if isinstance(messages, list) else []


def _last_user_text(record: dict[str, Any]) -> str:
    for message in reversed(_conversation(record)):
        if _role(message) in ("human", "user"):
            return _message_value(message)
    return ""


def _parse_tools(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    if not isinstance(value, list):
        return []
    return [tool for tool in value if isinstance(tool, dict)]


def _parse_arguments(value: Any) -> dict[str, Any] | None:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None
    return value if isinstance(value, dict) else None


def _normalize_call(value: Any) -> dict[str, Any] | None:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None
    if not isinstance(value, dict):
        return None
    function = value.get("function") if isinstance(value.get("function"), dict) else value
    name = function.get("name")
    arguments = _parse_arguments(function.get("arguments", {}))
    if not isinstance(name, str) or not name or arguments is None:
        return None
    return {"name": name, "arguments": arguments}


def _sft_call(record: dict[str, Any]) -> dict[str, Any] | None:
    for message in reversed(_conversation(record)):
        calls = message.get("tool_calls") if isinstance(message, dict) else None
        if isinstance(calls, list) and calls:
            return _normalize_call(calls[0])
    return None


def _dpo_call(record: dict[str, Any]) -> dict[str, Any] | None:
    chosen = record.get("chosen")
    if not isinstance(chosen, dict) or _role(chosen) not in ("function_call", "tool"):
        return None
    return _normalize_call(chosen.get("value", chosen.get("content")))


def _tool_names(tools: list[dict[str, Any]]) -> set[str]:
    names: set[str] = set()
    for tool in tools:
        function = tool.get("function")
        if isinstance(function, dict) and isinstance(function.get("name"), str):
            names.add(function["name"])
    return names


def _record_tags(record: dict[str, Any]) -> dict[str, Any]:
    tags = record.get("tags")
    if not isinstance(tags, dict):
        metadata = record.get("metadata")
        tags = metadata.get("tags") if isinstance(metadata, dict) else None
    return dict(tags) if isinstance(tags, dict) else {}


def _qa_reference(record: dict[str, Any], finetune_type: str) -> str:
    if finetune_type == "dpo":
        return _message_value(record.get("chosen"))
    for message in reversed(_conversation(record)):
        if _role(message) in ("gpt", "assistant") and not message.get("tool_calls"):
            return _message_value(message)
    return ""


def _prepare_record(
    record: dict[str, Any], task_hint: TaskType | None, finetune_type: str
) -> _PreparedRecord | None:
    tools = _parse_tools(record.get("tools"))
    call = _dpo_call(record) if finetune_type == "dpo" else _sft_call(record)
    is_fc = task_hint == "fc" or call is not None or bool(tools)
    query = _last_user_text(record)
    if is_fc:
        if call is None or not tools or call["name"] not in _tool_names(tools):
            return None
        eval_row = {
            "query": query,
            "tools": tools,
            "gold": call,
            "tags": {**_record_tags(record), "task_type": "fc"},
        }
        profile_text = f"{query}\n{json.dumps(call, ensure_ascii=False, sort_keys=True)}"
        return _PreparedRecord(
            record=record,
            task_type="fc",
            eval_kind="function_call",
            eval_row=eval_row,
            profile_text=profile_text,
            label=call["name"],
        )

    task_type: TaskType = "qa_multi" if task_hint == "qa_multi" else "qa"
    reference = _qa_reference(record, finetune_type)
    eval_row = {
        "query": query,
        "reference": reference,
        "tags": {**_record_tags(record), "task_type": task_type},
    }
    return _PreparedRecord(
        record=record,
        task_type=task_type,
        eval_kind="subjective",
        eval_row=eval_row,
        profile_text=f"{query}\n{reference}",
    )


def _safe_component(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]", "_", value)
    return value.strip(".") or "workflow"


def _artifact_name(workflow_id: str, iteration: int, suffix: str) -> str:
    ending = f"_it{iteration}_{suffix}"
    prefix = f"assistant_{_safe_component(workflow_id)}"
    return f"{prefix[: 64 - len(ending)]}{ending}"


def _stable_id(workflow_id: str, iteration: int, index: int) -> str:
    return f"wf_{_safe_component(workflow_id)}_it{iteration}_{index:06d}"


def _normalized_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _quantile(values: list[int], quantile: float) -> int:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(quantile * len(ordered)) - 1)]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
        ),
        encoding="utf-8",
    )


def _generation_stats(paths: list[Path]) -> tuple[float | None, dict[str, int]]:
    accepted = 0
    attempts = 0
    rejection_counts: Counter[str] = Counter()
    seen: set[Path] = set()
    for source_path in paths:
        progress_path = source_path.parent / "progress.json"
        if progress_path in seen or not progress_path.exists():
            continue
        seen.add(progress_path)
        try:
            progress = json.loads(progress_path.read_text("utf-8"))
            accepted += int(progress.get("accepted", 0))
            attempts += int(progress.get("attempts", 0))
            rejection_counts.update(
                {
                    str(key): int(value)
                    for key, value in (progress.get("rejects") or {}).items()
                }
            )
        except (OSError, ValueError, TypeError, AttributeError, json.JSONDecodeError):
            continue
    return (accepted / attempts if attempts else None), dict(rejection_counts)


def prepare_generated_datasets(
    source_paths: list[str | Path],
    workflow_id: str,
    iteration: int,
    holdout_ratio: float,
    split_seed: int,
    output_dir: str | Path,
    *,
    validation_ratio: float = 0.1,
    register: bool = True,
) -> PreparedDataset:
    """Freeze generated outputs into immutable train/eval artifacts and a profile."""
    if not 0 < holdout_ratio < 0.5:
        raise ValueError("holdout_ratio must be in (0, 0.5)")
    if not 0 <= validation_ratio < 0.5:
        raise ValueError("validation_ratio must be in [0, 0.5)")
    if not source_paths:
        raise ValueError("at least one generated dataset is required")

    paths = [Path(path) for path in source_paths]
    prepared_records: list[_PreparedRecord] = []
    invalid_tool_call_count = 0
    finetune_types: set[str] = set()
    for source_path in paths:
        task_hint = _task_hint(source_path)
        for record in _read_records(source_path):
            finetune_type = schema.detect_finetune_type(record)
            finetune_types.add(finetune_type)
            prepared = _prepare_record(record, task_hint, finetune_type)
            if prepared is None:
                invalid_tool_call_count += 1
            else:
                prepared_records.append(prepared)
    if len(finetune_types) != 1:
        raise ValueError("generated datasets must use one finetune_type")

    # Freeze each requested source task independently.  A global shuffle can
    # silently omit a minority task (and qa/qa_multi share one eval kind), so
    # task_type is the stratification and audit boundary.
    by_task: dict[TaskType, list[int]] = {}
    for index, item in enumerate(prepared_records):
        by_task.setdefault(item.task_type, []).append(index)
    holdout_indices: set[int] = set()
    for task_type, indices in sorted(by_task.items()):
        try:
            task_holdout = holdout_count(len(indices), holdout_ratio)
        except ValueError as exc:
            raise ValueError(f"{task_type}: {exc}") from exc
        shuffled = list(indices)
        random.Random(f"{split_seed}:{task_type}").shuffle(shuffled)
        holdout_indices.update(shuffled[:task_holdout])
    train_items = [
        item for index, item in enumerate(prepared_records) if index not in holdout_indices
    ]
    holdout_items = [
        (index, item)
        for index, item in enumerate(prepared_records)
        if index in holdout_indices
    ]

    first_fc = next(
        (index for index, item in enumerate(train_items) if item.task_type == "fc"), None
    )
    if first_fc not in (None, 0):
        train_items.insert(0, train_items.pop(first_fc))

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    dataset_name = _artifact_name(workflow_id, iteration, "train")
    train_path = output / f"{dataset_name}.json"
    _write_json(train_path, [item.record for item in train_items])

    eval_paths: dict[EvalKind, Path] = {}
    eval_dataset_names: dict[EvalKind, str] = {}
    eval_sha256: dict[EvalKind, str] = {}
    suffixes: dict[EvalKind, str] = {
        "function_call": "eval_fc",
        "subjective": "eval_subjective",
    }
    for eval_kind in ("function_call", "subjective"):
        rows = []
        for index, item in holdout_items:
            if item.eval_kind != eval_kind:
                continue
            rows.append(
                {"id": _stable_id(workflow_id, iteration, index), **item.eval_row}
            )
        if not rows:
            continue
        eval_name = _artifact_name(workflow_id, iteration, suffixes[eval_kind])
        eval_path = output / f"{eval_name}.jsonl"
        _write_jsonl(eval_path, rows)
        eval_paths[eval_kind] = eval_path
        eval_dataset_names[eval_kind] = eval_name
        eval_sha256[eval_kind] = _sha256(eval_path)

    profile_texts = [item.profile_text for item in train_items]
    char_counts = [len(text) for text in profile_texts]
    token_counts = [estimate_tokens(text) for text in profile_texts]
    normalized = [_normalized_text(text) for text in profile_texts]
    duplicates = len(normalized) - len(set(normalized))
    label_counts = Counter(item.label for item in train_items if item.label)
    slice_counts: Counter[str] = Counter()
    for _, item in holdout_items:
        tags = item.eval_row.get("tags") or {}
        selectors = set()
        for key, value in tags.items():
            values = value if isinstance(value, list) else [value]
            selectors.add(str(key))
            for member in values:
                selectors.update((str(member), f"{key}={member}"))
        slice_counts.update(selectors)
    acceptance_rate, rejection_counts = _generation_stats(paths)
    task_types = list(
        dict.fromkeys(item.task_type for item in prepared_records)
    )
    finetune_type = next(iter(finetune_types))
    profile = DatasetProfile(
        dataset_name=dataset_name,
        eval_dataset_names=eval_dataset_names,
        sha256=_sha256(train_path),
        eval_sha256=eval_sha256,
        n_records=len(train_items),
        holdout_records=len(holdout_items),
        requested_holdout_ratio=holdout_ratio,
        actual_holdout_ratio=len(holdout_items) / len(prepared_records),
        validation_ratio=validation_ratio,
        split_seed=split_seed,
        finetune_type=finetune_type,
        task_types=task_types,
        char_p50=_quantile(char_counts, 0.50),
        char_p95=_quantile(char_counts, 0.95),
        char_max=max(char_counts),
        token_p50=_quantile(token_counts, 0.50),
        token_p95=_quantile(token_counts, 0.95),
        token_max=max(token_counts),
        truncation_rates={
            str(cutoff): sum(count > cutoff for count in token_counts) / len(token_counts)
            for cutoff in (512, 1024, 2048, 4096, 8192)
        },
        exact_duplicate_rate=duplicates / len(normalized),
        empty_text_count=sum(not text.strip() for text in profile_texts),
        invalid_tool_call_count=invalid_tool_call_count,
        label_counts=dict(label_counts),
        slice_counts=dict(slice_counts),
        generation_acceptance_rate=acceptance_rate,
        rejection_counts=rejection_counts,
    )

    if register:
        dataset_store.register_dataset(
            str(train_path), dataset_name, "train", source="assistant"
        )
        for eval_kind, eval_path in eval_paths.items():
            dataset_store.register_dataset(
                str(eval_path), eval_dataset_names[eval_kind], "eval", source="assistant"
            )

    return PreparedDataset(
        train_path=train_path,
        eval_paths=eval_paths,
        dataset_name=dataset_name,
        eval_dataset_names=eval_dataset_names,
        profile=profile,
    )
