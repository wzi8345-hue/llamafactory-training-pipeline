"""Typed assistant action/observation adapter tests."""

from __future__ import annotations

import pytest

from . import datagen_job, dataset_store, eval_remote, eval_service, remote, train_service
from .assistant_schema import (
    DataPlan,
    DataPlanItem,
    EvaluationPlan,
    SuccessCriteria,
    TrainingPlan,
)
from .assistant_tools import AssistantTools
from .datagen_schema import DatagenConfig
from .eval_schema import EvalRequest, ModelUnderTest
from .schema import TrainConfig


def data_plan() -> DataPlan:
    return DataPlan(
        items=[
            DataPlanItem(
                task_type="fc",
                config=DatagenConfig(
                    task_type="fc", finetune_type="sft", count=100
                ),
                rationale="FC routing examples",
            )
        ],
        rationale="Improve tool routing",
    )


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


def evaluation_plan() -> EvaluationPlan:
    return EvaluationPlan(
        eval_dataset_names={"function_call": "assistant_wf_it0_eval_fc"},
        task_types=["function_call"],
        gpus="0",
        success_criteria=SuccessCriteria(primary_metric="tool_name_accuracy"),
    )


def fake_remote_config() -> remote.RemoteConfig:
    return remote.RemoteConfig("u@h", "/jobs", "/opt/LF")


@pytest.mark.parametrize(
    ("kind", "expected"),
    [
        ("datagen", "datagen"),
        ("training", "remote"),
        ("evaluation", "remote"),
    ],
)
def test_stop_external_job_routes_to_identity_safe_adapter(
    monkeypatch, kind, expected
):
    calls = []
    monkeypatch.setattr(
        datagen_job,
        "stop",
        lambda job_id: calls.append(("datagen", job_id))
        or {"stopped": True, "detail": "stopped"},
    )
    monkeypatch.setattr(
        remote,
        "stop_job",
        lambda cfg, job_id: calls.append(("remote", job_id))
        or {"stopped": True, "detail": "STOPPED"},
    )

    result = AssistantTools(fake_remote_config()).stop_external_job(kind, "job_1")

    assert calls == [(expected, "job_1")]
    assert result["terminal"] is True


def test_stop_external_job_rejects_unsupported_kind():
    with pytest.raises(ValueError, match="unsupported external job kind"):
        AssistantTools(fake_remote_config()).stop_external_job("other", "job_1")


def test_start_datagen_calls_existing_launcher(monkeypatch):
    calls = []
    monkeypatch.setattr(
        datagen_job, "create_and_launch", lambda jid, cfg: calls.append((jid, cfg))
    )
    monkeypatch.setattr(
        remote, "new_job_id", lambda: "20260819T010203Z-a1b2c3"
    )
    launches = AssistantTools().start_datagen("wf_1", data_plan())
    assert launches == [
        {"job_id": "20260819T010203Z-a1b2c3", "task_type": "fc"}
    ]
    assert calls[0][1]["task_type"] == "fc"


def test_start_training_uses_registered_dataset(monkeypatch, tmp_path):
    path = tmp_path / "train.data"
    path.write_text("[]", encoding="utf-8")
    monkeypatch.setattr(dataset_store, "data_path", lambda name, kind: path)
    monkeypatch.setattr(
        dataset_store, "dataset_meta", lambda name, kind: {"ext": ".json"}
    )
    monkeypatch.setattr(
        train_service,
        "submit_training_job",
        lambda *args, **kwargs: {"job_id": "j1"},
    )
    assert (
        AssistantTools(fake_remote_config()).start_training("wf_1", training_plan())
        == "j1"
    )


def test_start_evaluation_uses_only_registered_names(monkeypatch):
    captured = []
    monkeypatch.setattr(
        eval_service,
        "submit_registered_eval",
        lambda *args, **kwargs: captured.append((args, kwargs)) or "e1",
    )
    request = EvalRequest(
        models=[ModelUnderTest(name="base", model_name_or_path="/m")],
        task_types=["function_call"],
    )
    result = AssistantTools(fake_remote_config()).start_evaluation(
        "wf_1", request, evaluation_plan()
    )
    assert result == "e1"
    assert captured[0][0][2] == "assistant_wf_it0_eval_fc"
    assert captured[0][0][3] is None


def test_score_evaluation_forwards_critical_slices(monkeypatch):
    captured = []
    monkeypatch.setattr(
        eval_service,
        "score_registered_eval",
        lambda *args, **kwargs: captured.append((args, kwargs)) or {"ok": True},
    )
    result = AssistantTools(fake_remote_config()).score_evaluation(
        "20260819T030405Z-e7f8a9", ["critical", "intent=plan"]
    )
    assert result == {"ok": True}
    assert captured[0][1]["critical_tags"] == ["critical", "intent=plan"]


def test_resolve_train_job_uses_read_only_eval_resolver(monkeypatch):
    monkeypatch.setattr(
        eval_remote,
        "read_train_job",
        lambda cfg, job_id: {
            "model_name_or_path": "/models/champion-base",
            "adapter_path": "/saves/champion",
            "template": "qwen3_5_nothink",
        },
    )
    resolved = AssistantTools(fake_remote_config()).resolve_train_job(
        "20260818T010203Z-a1b2c3"
    )
    assert resolved["adapter_path"] == "/saves/champion"


def test_training_observation_includes_checkpoints_and_bounded_log_tail(monkeypatch):
    monkeypatch.setattr(remote, "job_status", lambda cfg, job_id: {"status": "failed"})
    monkeypatch.setattr(remote, "read_trainer_log", lambda cfg, job_id: {"points": []})
    monkeypatch.setattr(remote, "gpu_status", lambda cfg: [])
    monkeypatch.setattr(
        remote, "list_checkpoints", lambda cfg, job_id: [{"name": "checkpoint-20"}]
    )
    monkeypatch.setattr(
        remote,
        "read_job_log_tail",
        lambda cfg, job_id, lines=80: "CUDA out of memory",
    )
    observation = AssistantTools(fake_remote_config()).inspect_training(
        "20260819T020304Z-d4e5f6"
    )
    assert observation["checkpoints"] == [{"name": "checkpoint-20"}]
    assert observation["log_tail"] == "CUDA out of memory"


def test_successful_training_observation_verifies_adapter_artifacts(monkeypatch):
    monkeypatch.setattr(
        remote, "job_status", lambda cfg, job_id: {"status": "succeeded"}
    )
    monkeypatch.setattr(remote, "read_trainer_log", lambda cfg, job_id: {"points": []})
    monkeypatch.setattr(remote, "gpu_status", lambda cfg: [])
    monkeypatch.setattr(remote, "list_checkpoints", lambda cfg, job_id: [])
    monkeypatch.setattr(remote, "read_job_log_tail", lambda *args, **kwargs: "done")
    monkeypatch.setattr(
        remote,
        "inspect_training_output",
        lambda cfg, job_id: {
            "output_evidence_verified": True,
            "output_verified": True,
            "output_path": "/saves/adapter",
            "output_state": "complete",
        },
    )
    observation = AssistantTools(fake_remote_config()).inspect_training(
        "20260819T020304Z-d4e5f6"
    )
    assert observation["output_verified"] is True
    assert observation["output_path"] == "/saves/adapter"


def test_terminal_training_observation_survives_metrics_and_gpu_probe_failure(
    monkeypatch,
):
    monkeypatch.setattr(
        remote, "job_status", lambda cfg, job_id: {"status": "failed"}
    )
    monkeypatch.setattr(
        remote,
        "read_trainer_log",
        lambda cfg, job_id: (_ for _ in ()).throw(remote.RemoteError("metrics")),
    )
    monkeypatch.setattr(
        remote,
        "gpu_status",
        lambda cfg: (_ for _ in ()).throw(remote.RemoteError("gpu")),
    )
    monkeypatch.setattr(remote, "list_checkpoints", lambda cfg, job_id: [])
    monkeypatch.setattr(
        remote, "read_job_log_tail", lambda *args, **kwargs: "failed"
    )

    observation = AssistantTools(fake_remote_config()).inspect_training(
        "20260819T020304Z-d4e5f6"
    )

    assert observation["status"]["status"] == "failed"
    assert observation["metrics"] == {}
    assert observation["metrics_verified"] is False
    assert observation["gpus"] == []
    assert observation["gpus_verified"] is False


def test_failed_evaluation_includes_verified_bounded_log_tail(monkeypatch):
    monkeypatch.setattr(remote, "job_status", lambda cfg, job_id: {"status": "failed"})
    captured = []

    def read_tail(cfg, job_id, lines=80, log_name="train.log"):
        captured.append((lines, log_name))
        return "runner failed"

    monkeypatch.setattr(remote, "read_job_log_tail", read_tail)
    observation = AssistantTools(fake_remote_config()).inspect_evaluation(
        "20260819T030405Z-e7f8a9"
    )
    assert observation["log_tail"] == "runner failed"
    assert observation["log_verified"] is True
    assert observation["log_state"] == "readable"
    assert captured == [(80, "eval.log")]


def test_missing_terminal_log_is_not_treated_as_verified(monkeypatch):
    monkeypatch.setattr(remote, "job_status", lambda cfg, job_id: {"status": "failed"})

    def missing(*args, **kwargs):
        raise remote.RemoteError("log not readable")

    monkeypatch.setattr(remote, "read_job_log_tail", missing)
    observation = AssistantTools(fake_remote_config()).inspect_evaluation(
        "20260819T030405Z-e7f8a9"
    )

    assert observation["log_tail"] == ""
    assert observation["log_verified"] is False
    assert observation["log_state"] == "unavailable"


def test_empty_but_readable_terminal_log_is_verified(monkeypatch):
    monkeypatch.setattr(remote, "job_status", lambda cfg, job_id: {"status": "failed"})
    monkeypatch.setattr(remote, "read_job_log_tail", lambda *args, **kwargs: "")
    observation = AssistantTools(fake_remote_config()).inspect_evaluation(
        "20260819T030405Z-e7f8a9"
    )

    assert observation["log_verified"] is True
    assert observation["log_state"] == "empty"


def test_adapter_does_not_expose_destructive_or_arbitrary_methods():
    tools = AssistantTools(fake_remote_config())
    for name in ("stop", "cleanup", "delete", "run_command", "run_remote_script"):
        assert not hasattr(tools, name)
