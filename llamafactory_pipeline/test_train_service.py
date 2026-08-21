"""HTTP-independent training submission tests."""

from __future__ import annotations

import json

import pytest
import yaml

from . import remote, schema
from .train_service import (
    InsufficientRemoteDisk,
    TrainDataRef,
    submit_training_job,
)


def test_submit_training_job_builds_isolated_output(tmp_path, monkeypatch):
    data = tmp_path / "train.json"
    data.write_text(
        json.dumps(
            [
                {
                    "conversations": [
                        {"from": "human", "value": "Q"},
                        {"from": "gpt", "value": "A"},
                    ]
                }
            ]
        ),
        encoding="utf-8",
    )
    captured = []
    monkeypatch.setattr(remote, "disk_free", lambda cfg: 1 << 40)
    monkeypatch.setattr(remote, "new_job_id", lambda: "20260819T010203Z-a1b2c3")
    monkeypatch.setattr(remote, "submit_job", lambda *args: captured.append(args))
    cfg = remote.RemoteConfig("u@h", "/jobs", "/opt/LF")

    result = submit_training_job(
        cfg,
        schema.TrainConfig(),
        TrainDataRef(
            path=data, ext=".json", source_type="assistant", source_id="ds1"
        ),
        "0",
    )

    assert result["job_id"] == "20260819T010203Z-a1b2c3"
    assert result["finetune_type"] == "sft"
    submitted_yaml = yaml.safe_load(captured[0][2])
    assert "20260819T010203Z-a1b2c3" in submitted_yaml["output_dir"]
    assert captured[0][6] == "0"


def test_submit_training_job_uses_preallocated_id(tmp_path, monkeypatch):
    data = tmp_path / "train.json"
    data.write_text(
        '[{"conversations":[{"from":"human","value":"Q"},'
        '{"from":"gpt","value":"A"}]}]',
        encoding="utf-8",
    )
    captured = []
    monkeypatch.setattr(remote, "disk_free", lambda cfg: 1 << 40)
    monkeypatch.setattr(remote, "submit_job", lambda *args: captured.append(args))
    monkeypatch.setattr(remote, "_write_remote_file", lambda *args: None)
    job_id = "20260819T010203Z-a1b2c3"
    result = submit_training_job(
        remote.RemoteConfig("u@h", "/jobs", "/opt/LF"),
        schema.TrainConfig(),
        TrainDataRef(
            path=data, ext=".json", source_type="assistant", source_id="ds1"
        ),
        "0",
        job_id=job_id,
    )
    assert result["job_id"] == job_id
    assert captured[0][1] == job_id


def test_submit_rejects_dpo_data_with_sft_stage(tmp_path, monkeypatch):
    data = tmp_path / "dpo.json"
    data.write_text(
        json.dumps(
            [
                {
                    "conversations": [{"from": "human", "value": "Q"}],
                    "chosen": {"from": "gpt", "value": "A"},
                    "rejected": {"from": "gpt", "value": "B"},
                }
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(remote, "disk_free", lambda cfg: 1 << 40)
    cfg = remote.RemoteConfig("u@h", "/jobs", "/opt/LF")
    with pytest.raises(ValueError, match="stage=dpo"):
        submit_training_job(
            cfg,
            schema.TrainConfig(),
            TrainDataRef(
                path=data, ext=".json", source_type="assistant", source_id="ds1"
            ),
            "0",
        )


def test_submit_reports_required_disk_without_launching(tmp_path, monkeypatch):
    data = tmp_path / "train.json"
    data.write_text(
        '[{"conversations":[{"from":"human","value":"Q"},'
        '{"from":"gpt","value":"A"}]}]',
        encoding="utf-8",
    )
    monkeypatch.setattr(remote, "disk_free", lambda cfg: 1)
    launched = []
    monkeypatch.setattr(remote, "submit_job", lambda *args: launched.append(args))

    with pytest.raises(InsufficientRemoteDisk) as raised:
        submit_training_job(
            remote.RemoteConfig("u@h", "/jobs", "/opt/LF"),
            schema.TrainConfig(),
            TrainDataRef(
                path=data, ext=".json", source_type="dataset", source_id="ds1"
            ),
            "",
        )
    assert raised.value.required_bytes > raised.value.available_bytes
    assert not launched


def test_service_never_deletes_caller_owned_data(tmp_path, monkeypatch):
    data = tmp_path / "train.json"
    data.write_text(
        '[{"conversations":[{"from":"human","value":"Q"},'
        '{"from":"gpt","value":"A"}]}]',
        encoding="utf-8",
    )
    monkeypatch.setattr(remote, "disk_free", lambda cfg: 1 << 40)
    monkeypatch.setattr(remote, "submit_job", lambda *args: None)
    monkeypatch.setattr(remote, "_write_remote_file", lambda *args: None)
    submit_training_job(
        remote.RemoteConfig("u@h", "/jobs", "/opt/LF"),
        schema.TrainConfig(),
        TrainDataRef(
            path=data,
            ext=".json",
            source_type="upload",
            source_id="upload",
            cleanup_after_submit=True,
        ),
        "",
    )
    assert data.exists()
