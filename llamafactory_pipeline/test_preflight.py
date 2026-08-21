"""Read-only model inventory and training preflight tests."""

from __future__ import annotations

import base64
import json
import os
import subprocess

import pytest

from . import remote
from .assistant_schema import GpuDevice, ModelInventory, TrainingPlan
from .preflight import (
    PreflightInputs,
    _disk_required_bytes,
    _host_runtime_script,
    _container_paths_script,
    _probe_output_storage,
    _probe_bf16,
    _resume_checkpoint_script,
    _runtime_probe_script,
    _validate_dataset_meta,
    aggregate_preflight,
    collect_model_inventory,
)
from .schema import TrainConfig


def training_plan() -> TrainingPlan:
    return TrainingPlan(
        config=TrainConfig(),
        dataset_name="assistant_wf_it0_train",
        eval_dataset_names={"function_call": "assistant_wf_it0_eval_fc"},
        gpus="0",
        decisions=[],
        estimated_steps=100,
        estimated_vram_gb=20.0,
    )


def healthy_inputs() -> PreflightInputs:
    return PreflightInputs(
        ssh_ok=True,
        cli_version="0.9.4",
        container_ok=True,
        container_paths_ok=True,
        model=ModelInventory(
            model_path="/models/qwen",
            model_exists=True,
            config_exists=True,
            tokenizer_exists=True,
            parameter_billions=9.0,
            context_length=32768,
        ),
        gpus=[
            GpuDevice(
                index=0,
                name="GPU",
                memory_used_mb=1024,
                memory_total_mb=24576,
                utilization_pct=0,
                temperature_c=45,
            )
        ],
        selected_gpu_ids=[0],
        dataset_ok=True,
        stage_matches=True,
        disk_free_bytes=50 << 30,
        disk_required_bytes=12 << 30,
        bf16_supported=True,
        conflicting_jobs=[],
    )


def test_preflight_blocks_insufficient_vram():
    inputs = healthy_inputs()
    inputs.model = ModelInventory(
        model_path="/models/qwen",
        model_exists=True,
        config_exists=True,
        tokenizer_exists=True,
        parameter_billions=14.0,
        context_length=32768,
    )
    inputs.gpus = [
        GpuDevice(
            index=0,
            name="GPU",
            memory_used_mb=23000,
            memory_total_mb=24576,
            utilization_pct=99,
            temperature_c=70,
        )
    ]
    report = aggregate_preflight(inputs, training_plan())
    assert report.status == "block"
    assert any(
        check.name == "gpu_memory" and check.status == "block"
        for check in report.checks
    )


def test_ddp_memory_check_uses_smallest_selected_gpu_not_sum():
    inputs = healthy_inputs()
    inputs.selected_gpu_ids = [0, 1]
    inputs.gpus = [
        GpuDevice(
            index=index,
            name="GPU",
            memory_used_mb=1024,
            memory_total_mb=13312,
            utilization_pct=0,
            temperature_c=45,
        )
        for index in (0, 1)
    ]
    plan = training_plan().model_copy(update={"gpus": "0,1"})
    report = aggregate_preflight(inputs, plan)
    memory = next(check for check in report.checks if check.name == "gpu_memory")
    assert memory.status == "block"
    assert memory.evidence["minimum_free_gb"] == 12.0


def test_preflight_warns_when_parameter_size_is_unknown():
    inputs = healthy_inputs()
    inputs.model.parameter_billions = None
    report = aggregate_preflight(inputs, training_plan())
    assert report.status == "warn"
    assert next(check for check in report.checks if check.name == "model_size").status == "warn"


def test_preflight_blocks_conflict_and_hot_gpu_and_warns_unknown_job():
    inputs = healthy_inputs()
    inputs.gpus[0].temperature_c = 85
    inputs.conflicting_jobs = ["20260819T010203Z-a1b2c3"]
    inputs.unknown_gpu_jobs = ["20260701T010203Z-deadbe"]
    report = aggregate_preflight(inputs, training_plan())
    by_name = {check.name: check for check in report.checks}
    assert by_name["gpu_temperature"].status == "block"
    assert by_name["gpu_conflicts"].status == "block"
    assert set(by_name) == {
        "ssh", "llamafactory_cli", "container", "model_files", "model_size",
        "dataset", "stage_compatibility", "gpu_selection", "gpu_memory",
        "gpu_temperature", "gpu_conflicts", "disk", "bf16",
        "remote_root", "quantization", "truncation", "eta_budget",
            "resume_checkpoint",
            "output_storage",
        }


def test_preflight_checks_root_quantization_truncation_and_eta_budget():
    inputs = healthy_inputs()
    inputs.remote_root_writable = False
    inputs.quantization_supported = False
    inputs.truncation_rate = 0.08
    plan = training_plan().model_copy(
        update={
            "config": training_plan().config.model_copy(
                update={
                    "model": training_plan().config.model.model_copy(
                        update={"quantization_bit": 4}
                    )
                }
            ),
            "estimated_hours_low": 2.0,
            "estimated_hours_high": 4.0,
            "max_training_hours": 1.0,
        }
    )
    report = aggregate_preflight(inputs, plan)
    checks = {check.name: check for check in report.checks}
    assert checks["remote_root"].status == "block"
    assert checks["quantization"].status == "block"
    assert checks["truncation"].status == "warn"
    assert checks["eta_budget"].status == "block"


def test_preflight_blocks_missing_resume_checkpoint():
    inputs = healthy_inputs()
    inputs.resume_checkpoint_ok = False
    base = training_plan()
    plan = base.model_copy(
        update={
            "config": base.config.model_copy(
                update={
                    "train": base.config.train.model_copy(
                        update={"resume_from_checkpoint": "/missing/checkpoint-20"}
                    )
                }
            )
        }
    )
    report = aggregate_preflight(inputs, plan)
    check = next(item for item in report.checks if item.name == "resume_checkpoint")
    assert check.status == "block"


def test_resume_checkpoint_probe_requires_state_weights_and_optimizer():
    script = _resume_checkpoint_script("/saves/run/checkpoint-100", True)
    for required in (
        "trainer_state.json",
        "adapter_model.safetensors",
        "optimizer.pt",
        "scheduler.pt",
    ):
        assert required in script
    assert '[ -s "$P/trainer_state.json" ]' in script
    assert '[ -s "$P/optimizer.pt" ]' in script
    assert '[ -s "$P/scheduler.pt" ]' in script


def test_resume_checkpoint_probe_rejects_bad_index_and_missing_shards(tmp_path):
    (tmp_path / "trainer_state.json").write_text(
        json.dumps({"global_step": 100}), encoding="utf-8"
    )
    (tmp_path / "optimizer.pt").write_bytes(b"optimizer-state")
    (tmp_path / "scheduler.pt").write_bytes(b"scheduler-state")
    index = tmp_path / "model.safetensors.index.json"
    script = _resume_checkpoint_script(str(tmp_path), True)

    index.write_text("not-json", encoding="utf-8")
    assert subprocess.run(
        ["bash"], input=script, text=True, capture_output=True
    ).stdout.strip() == "CHECKPOINT_BAD"

    index.write_text(
        json.dumps({"weight_map": {"layer": "missing-00001.safetensors"}}),
        encoding="utf-8",
    )
    assert subprocess.run(
        ["bash"], input=script, text=True, capture_output=True
    ).stdout.strip() == "CHECKPOINT_BAD"

    (tmp_path / "model-00001.safetensors").write_bytes(b"weight-shard-data")
    index.write_text(
        json.dumps({"weight_map": {"layer": "model-00001.safetensors"}}),
        encoding="utf-8",
    )
    (tmp_path / "trainer_state.json").write_text("{}", encoding="utf-8")
    assert subprocess.run(
        ["bash"],
        input=script,
        text=True,
        capture_output=True,
        env={**os.environ, "PYTHONOPTIMIZE": "1"},
    ).stdout.strip() == "CHECKPOINT_BAD"
    (tmp_path / "trainer_state.json").write_text(
        json.dumps({"global_step": 100}), encoding="utf-8"
    )
    assert subprocess.run(
        ["bash"], input=script, text=True, capture_output=True
    ).stdout.strip() == "CHECKPOINT_OK"


def test_disk_requirement_covers_dataset_working_space_and_model_outputs():
    required = _disk_required_bytes(
        model_weight_bytes=20 << 30,
        dataset_size_bytes=500 << 20,
    )
    assert required >= (10 << 30) + 2 * (500 << 20)
    assert required >= int((20 << 30) * 0.20)


def test_preflight_dataset_validation_reads_actual_artifact(tmp_path):
    path = tmp_path / "train.data"
    path.write_text(
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
    meta = {"data_path": str(path), "ext": ".json", "finetune_type": "sft"}
    assert _validate_dataset_meta(meta, "sft") == (True, True, path.stat().st_size)
    path.unlink()
    assert _validate_dataset_meta(meta, "sft") == (False, False, 0)


def test_preflight_dataset_validation_blocks_content_hash_drift(tmp_path):
    path = tmp_path / "train.json"
    path.write_text(
        '[{"conversations":[{"from":"human","value":"Q"},'
        '{"from":"gpt","value":"A"}]}]',
        encoding="utf-8",
    )
    meta = {"data_path": str(path), "ext": ".json"}
    assert _validate_dataset_meta(meta, "sft", "0" * 64)[0] is False


def test_preflight_blocks_unwritable_or_full_output_storage():
    inputs = healthy_inputs()
    inputs.output_dir_writable = False
    inputs.output_disk_free_bytes = 1 << 30
    inputs.output_disk_required_bytes = 2 << 30
    report = aggregate_preflight(inputs, training_plan())
    check = next(item for item in report.checks if item.name == "output_storage")
    assert check.status == "block"


def test_output_storage_probe_runs_inside_container(monkeypatch):
    scripts = []

    def fake_run(cfg, script, timeout=15):
        scripts.append(script)
        return "WRITABLE=1\nFREE_KB=2048\n"

    monkeypatch.setattr(remote, "run_remote_script", fake_run)
    writable, free = _probe_output_storage(
        remote.RemoteConfig(
            "u@h", "/jobs", "/opt/LF", docker_container="lf-runtime"
        ),
        training_plan(),
    )
    assert writable and free == 2048 * 1024
    assert scripts[0].startswith("docker exec lf-runtime sh -lc ")


def test_runtime_probes_match_training_environment_and_container_mounts():
    host_cfg = remote.RemoteConfig(
        "u@h",
        "/jobs",
        "/opt/LF",
        remote_prefix="source /opt/conda/bin/activate lf",
    )
    host_script = _host_runtime_script(host_cfg, "python -c 'import bitsandbytes'")
    assert "source /opt/conda/bin/activate lf" in host_script
    assert "cd /opt/LF" in host_script
    assert "import bitsandbytes" in host_script
    resume_script = _runtime_probe_script(
        host_cfg, _resume_checkpoint_script("/saves/checkpoint-100", True)
    )
    assert "source /opt/conda/bin/activate lf" in resume_script
    assert "cd /opt/LF" in resume_script
    assert "CHECKPOINT_OK" in resume_script

    docker_cfg = remote.RemoteConfig(
        "u@h", "/jobs", "/opt/LF", docker_container="lf"
    )
    container_script = _container_paths_script(docker_cfg, "/models/qwen")
    assert "/models/qwen" in container_script
    assert "/opt/LF" in container_script
    assert "/jobs" in container_script


def test_collect_model_inventory_parses_remote_metadata(monkeypatch):
    config = {"max_position_embeddings": 32768, "model_type": "qwen"}
    encoded = base64.b64encode(json.dumps(config).encode()).decode()
    scripts = []

    def fake_run(cfg, script, timeout=30):
        scripts.append(script)
        return (
            "EXISTS=1\nCONFIG_EXISTS=1\n"
            f"CONFIG_B64={encoded}\nTOKENIZER=1\nWEIGHT_BYTES=18000000000\n"
        )

    monkeypatch.setattr(remote, "run_remote_script", fake_run)
    inventory = collect_model_inventory(
        remote.RemoteConfig("u@h", "/jobs", "/opt/LF"), "/models/qwen 9b"
    )
    assert inventory.parameter_billions == 9.0
    assert inventory.context_length == 32768
    assert "'/models/qwen 9b'" in scripts[0]
    assert "find \"$MODEL_PATH\" -maxdepth 1" in scripts[0]


def test_collect_model_inventory_runs_inside_training_container(monkeypatch):
    scripts = []

    def fake_run(cfg, script, timeout=30):
        scripts.append(script)
        return "EXISTS=1\nCONFIG_EXISTS=0\nCONFIG_B64=\nTOKENIZER=1\nWEIGHT_BYTES=0\n"

    monkeypatch.setattr(remote, "run_remote_script", fake_run)
    collect_model_inventory(
        remote.RemoteConfig(
            "u@h", "/jobs", "/opt/LF", docker_container="lf-runtime"
        ),
        "/models/qwen",
    )

    assert scripts[0].startswith("docker exec lf-runtime sh -lc ")
    assert "MODEL_PATH=" in scripts[0]


def test_bf16_probe_runs_inside_training_container(monkeypatch):
    scripts = []
    monkeypatch.setattr(
        remote,
        "run_remote_script",
        lambda cfg, script, timeout=15: scripts.append(script) or "0, 8.9\n",
    )
    assert _probe_bf16(
        remote.RemoteConfig(
            "u@h", "/jobs", "/opt/LF", docker_container="lf-runtime"
        ),
        [0],
    ) is True
    assert scripts[0].startswith("docker exec lf-runtime sh -lc ")


@pytest.mark.parametrize("path", ["", "relative/model", "/bad\npath"])
def test_collect_model_inventory_rejects_unsafe_paths(path):
    with pytest.raises(ValueError, match="model path"):
        collect_model_inventory(remote.RemoteConfig("u@h", "/jobs", "/opt/LF"), path)
