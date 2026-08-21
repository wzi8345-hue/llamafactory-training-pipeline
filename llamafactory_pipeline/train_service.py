"""HTTP-independent, validated LlamaFactory training submission."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

from . import remote, schema


class InsufficientRemoteDisk(RuntimeError):
    def __init__(self, available_bytes: int, required_bytes: int):
        self.available_bytes = available_bytes
        self.required_bytes = required_bytes
        super().__init__(
            f"服务器磁盘不足: 可用 {available_bytes // (1 << 20)}MB, "
            f"需约 {required_bytes // (1 << 20)}MB"
        )


@dataclass(frozen=True)
class TrainDataRef:
    path: Path
    ext: Literal[".json", ".jsonl"]
    source_type: Literal["upload", "datagen", "dataset", "assistant"]
    source_id: str
    cleanup_after_submit: bool = False


def submit_training_job(
    cfg: remote.RemoteConfig,
    train_cfg: schema.TrainConfig,
    data_ref: TrainDataRef,
    gpus: str,
    *,
    job_id: str | None = None,
) -> dict[str, Any]:
    """Validate data/config, create an isolated job, upload it, and launch training."""
    try:
        gpus = remote.validate_gpus(gpus)
    except remote.RemoteError as exc:
        raise ValueError(str(exc)) from exc
    if data_ref.ext not in (".json", ".jsonl"):
        raise ValueError("training data extension must be .json or .jsonl")
    if not data_ref.path.is_file():
        raise ValueError(f"training data does not exist: {data_ref.source_id}")

    available = remote.disk_free(cfg)
    required = data_ref.path.stat().st_size * 2 + (1 << 30)
    if available < required:
        raise InsufficientRemoteDisk(available, required)

    record = schema.read_first_record(
        str(data_ref.path), data_ref.ext == ".jsonl"
    )
    fmt = schema.detect_sharegpt_format(record)
    finetune_type = schema.detect_finetune_type(record)
    if train_cfg.method.stage != finetune_type:
        raise ValueError(
            f"{finetune_type.upper()} data requires stage={finetune_type} "
            f"(current stage={train_cfg.method.stage})"
        )

    job_id = job_id or remote.new_job_id()
    remote.validate_job_id(job_id)
    dataset_name = f"user_{job_id.replace('-', '_')}"
    data_name = f"train{data_ref.ext}"
    data_dir = remote.job_dir(cfg, job_id) + "/data"
    flat = schema.flatten_config(train_cfg, dataset_name, data_dir)
    if not flat.get("resume_from_checkpoint"):
        flat["output_dir"] = flat["output_dir"].rstrip("/") + "/" + job_id
    yaml_text = yaml.safe_dump(flat, allow_unicode=True, sort_keys=False)
    info = schema.build_dataset_info(dataset_name, data_name, fmt, record)
    remote.submit_job(
        cfg,
        job_id,
        yaml_text,
        json.dumps(info, ensure_ascii=False, indent=2),
        str(data_ref.path),
        data_name,
        gpus,
    )
    return {
        "job_id": job_id,
        "dataset": dataset_name,
        "format": fmt,
        "finetune_type": finetune_type,
        "gpus": gpus or "default",
    }
