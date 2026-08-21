"""Approval-gated assistant workflow orchestration tests."""

from __future__ import annotations

from contextlib import nullcontext
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from .assistant_schema import (
    ActionKind,
    BaselineSpec,
    CheckResult,
    DataPlan,
    DataPlanItem,
    DataSourceSpec,
    DatasetProfile,
    EvaluationPlan,
    GpuDevice,
    ModelInventory,
    PreflightReport,
    RequirementDraft,
    RequirementField,
    RequirementExtraction,
    SuccessCriteria,
    TrainingAdjustment,
    TrainingObjective,
    TrainingPlan,
    WorkflowState,
)
from .assistant_service import (
    ApprovalConflict,
    AssistantService,
    sanitize_error,
    sanitize_log_excerpt,
)
from .assistant_state import plan_hash
from .assistant_store import AssistantStore
from .assistant_worker import run_once
from .datagen_schema import DatagenConfig
from .schema import TrainConfig
from .remote import RemoteConflictError, RemoteError


def objective() -> TrainingObjective:
    return TrainingObjective(
        goal="Improve FC routing",
        task_types=["fc"],
        base_model_path="/models/qwen",
        baseline=BaselineSpec(kind="base_model", name="base"),
        data_source=DataSourceSpec(fc_seed_file="sft_data/router_fc/seed.json"),
        success_criteria=SuccessCriteria(primary_metric="tool_name_accuracy"),
    )


def data_plan(*, count: int = 100, rationale: str = "Improve FC routing") -> DataPlan:
    return DataPlan(
        items=[
            DataPlanItem(
                task_type="fc",
                config=DatagenConfig(
                    task_type="fc", finetune_type="sft", count=count
                ),
            )
        ],
        rationale=rationale,
    )


def dataset_profile() -> DatasetProfile:
    return DatasetProfile(
        dataset_name="assistant_wf_it0_train",
        eval_dataset_names={"function_call": "assistant_wf_it0_eval_fc"},
        sha256="a" * 64,
        eval_sha256={"function_call": "b" * 64},
        n_records=80,
        holdout_records=20,
        requested_holdout_ratio=0.1,
        actual_holdout_ratio=0.2,
        validation_ratio=0.1,
        split_seed=42,
        finetune_type="sft",
        task_types=["fc"],
        char_p50=40,
        char_p95=80,
        char_max=120,
        token_p50=20,
        token_p95=40,
        token_max=60,
        truncation_rates={
            "512": 0.0,
            "1024": 0.0,
            "2048": 0.0,
            "4096": 0.0,
            "8192": 0.0,
        },
        exact_duplicate_rate=0.0,
        empty_text_count=0,
        invalid_tool_call_count=0,
        label_counts={"plan": 80},
    )


class ReadyPlanner:
    def __init__(self):
        self.iteration_calls = []

    def extract_requirements(self, messages):
        return RequirementExtraction(
            assistant_reply="Plan ready",
            ready=True,
            missing_fields=[],
            objective=objective(),
        )

    def extract_requirement_draft(self, messages):
        message_id = next(
            row["message_id"] for row in messages if row["role"] == "user"
        )

        def field(value):
            return RequirementField(
                value=value,
                source="user",
                evidence_message_ids=[message_id],
            )

        return RequirementDraft(
            assistant_reply="需求理解已整理，请确认。",
            ready_for_review=True,
            missing_fields=[],
            scenario=field("训练 FC 路由"),
            current_problem=field("训练 FC 路由"),
            desired_behavior=field("训练 FC 路由"),
            proposed_objective=objective(),
            assumptions=["测试默认目标"],
        )

    def create_data_plan(self, value):
        return data_plan()

    def explain_diagnosis(self, diagnosis, comparison):
        return "A/B 证据已完成规则诊断。"

    def create_iteration_plan(
        self, objective, previous_plan, previous_training_plan, diagnosis, comparison
    ):
        self.iteration_calls.append(
            (objective, previous_plan, previous_training_plan, diagnosis, comparison)
        )
        return data_plan(count=60, rationale="Diagnosis-derived coverage repair")


class FakeTools:
    def __init__(self):
        self.datagen_calls = []
        self.training_calls = []
        self.evaluation_calls = []
        self.job_ids = iter(
            [
                "20260819T010203Z-a1b2c3",
                "20260819T020304Z-d4e5f6",
                "20260819T030405Z-e7f8a9",
            ]
        )

    def new_job_id(self):
        return next(self.job_ids)

    def start_datagen(self, workflow_id, plan, launches=None):
        self.datagen_calls.append((workflow_id, plan))
        return launches or [
            {"job_id": "20260819T010203Z-a1b2c3", "task_type": "fc"}
        ]

    def start_training(self, workflow_id, plan, job_id=None):
        self.training_calls.append((workflow_id, plan))
        return job_id or "20260819T020304Z-d4e5f6"

    def start_evaluation(self, workflow_id, request, plan, eval_id=None):
        self.evaluation_calls.append((workflow_id, request, plan))
        return eval_id or "20260819T030405Z-e7f8a9"

    def score_evaluation(self, eval_id, critical_tags=()):
        assert eval_id == "20260819T030405Z-e7f8a9"
        return comparison(primary_improvement=0.05)

    def resolve_train_job(self, job_id):
        if job_id == "20260819T020304Z-d4e5f6":
            return {
                "model_name_or_path": "/models/qwen",
                "adapter_path": (
                    "/opt/LF/saves/qwen3-4b/lora/sft/"
                    "20260819T020304Z-d4e5f6"
                ),
                "template": "qwen3_5_nothink",
            }
        return {
            "model_name_or_path": "/models/champion-base",
            "adapter_path": "/saves/champion",
            "template": "qwen3_5_nothink",
        }


class CrashAfterTrainingSubmitTools(FakeTools):
    def __init__(self):
        super().__init__()
        self.submit_job_ids = []

    def start_training(self, workflow_id, plan, job_id=None):
        self.submit_job_ids.append(job_id)
        if len(self.submit_job_ids) == 1:
            raise RuntimeError("process crashed after remote submit")
        return job_id


class PermanentTrainingErrorTools(FakeTools):
    def start_training(self, workflow_id, plan, job_id=None):
        raise ValueError("dataset artifact is invalid")


def fake_policy(*args, **kwargs):
    return TrainingPlan(
        config=TrainConfig(),
        dataset_name="assistant_wf_it0_train",
        eval_dataset_names={"function_call": "assistant_wf_it0_eval_fc"},
        gpus="0",
        decisions=[],
        estimated_steps=10,
        estimated_vram_gb=20.0,
    )


def test_workflow_error_evidence_redacts_credentials_and_home_paths():
    value = (
        "Bearer abcdefghijklmnop api_key=topsecret sk-abcdefghijklmnop "
        "/Users/alice/.ssh/id_ed25519"
    )
    redacted_error = sanitize_error(value)
    redacted_log = sanitize_log_excerpt(value)
    for redacted in (redacted_error, redacted_log):
        assert "abcdefghijklmnop" not in redacted
        assert "topsecret" not in redacted
        assert "/Users/alice" not in redacted


def test_execution_error_classification_distinguishes_conflict_from_transport():
    assert AssistantService.is_retryable_execution_error(RemoteError("timeout"))
    assert not AssistantService.is_retryable_execution_error(
        RemoteConflictError("immutable input conflict")
    )


def test_evidence_gate_blocks_missing_requested_critical_slice():
    summary = {"subjective": {"n": 2, "answer_accuracy": 0.8}}
    issues = AssistantService._evaluation_evidence_issues(
        summary,
        summary,
        {
            "n": 2,
            "baseline_missing": 0,
            "candidate_missing": 0,
            "slices": {},
            "missing_critical_slices": ["must_cover"],
        },
        ["subjective"],
        "answer_accuracy",
        ["must_cover"],
    )
    assert "critical_slice_must_cover_missing" in issues


def test_evidence_gate_blocks_requested_source_task_without_holdout_rows():
    summary = {"subjective": {"n": 30, "combined_accuracy": 0.8}}
    issues = AssistantService._evaluation_evidence_issues(
        summary,
        summary,
        {
            "n": 30,
            "baseline_missing": 0,
            "candidate_missing": 0,
            "slices": {"qa": {"n": 30}},
        },
        ["subjective"],
        "combined_accuracy",
        [],
        ["qa", "fc"],
    )
    assert "task_slice_fc_missing" in issues


def test_evidence_gate_blocks_pairs_without_a_usable_business_score():
    summary = {"subjective": {"n": 30, "answer_accuracy": 0.8}}
    issues = AssistantService._evaluation_evidence_issues(
        summary,
        summary,
        {
            "n": 30,
            "paired_score_n": 29,
            "baseline_missing": 0,
            "candidate_missing": 0,
            "slices": {"qa": {"n": 30}},
        },
        ["subjective"],
        "answer_accuracy",
        ["qa"],
        ["qa"],
    )

    assert "paired_business_score_incomplete" in issues


def preflight(status, summary=None):
    return PreflightReport(
        status=status,
        checks=[
            CheckResult(name="ssh", status=status, summary=summary or status)
        ],
        model=ModelInventory(
            model_path="/models/qwen",
            model_exists=True,
            config_exists=True,
            tokenizer_exists=True,
            parameter_billions=9.0,
        ),
        gpus=[
            GpuDevice(
                index=0,
                name="GPU",
                memory_used_mb=0,
                memory_total_mb=24576,
                utilization_pct=0,
            )
        ],
    )


def pass_preflight(*args, **kwargs):
    return preflight("pass")


def block_preflight(*args, **kwargs):
    return preflight("block")


def fake_data_preparer(*args, **kwargs):
    return SimpleNamespace(profile=dataset_profile())


def service(tmp_path, planner, tools, policy, preflight_runner, data_preparer):
    return AssistantService(
        store=AssistantStore(tmp_path / "assistant.sqlite"),
        planner=planner,
        tools=tools,
        policy=policy,
        preflight_runner=preflight_runner,
        data_preparer=data_preparer,
    )


def create_data_plan_ready(svc, message="训练 FC 路由"):
    review = svc.create_workflow(message)
    assert review["state"] == WorkflowState.REQUIREMENTS_REVIEW.value
    approval = review["pending_approvals"][0]
    assert approval["action"] == ActionKind.CONFIRM_REQUIREMENTS.value
    preparing = svc.approve(
        review["workflow_id"], approval["approval_id"], approval["plan_hash"]
    )
    assert preparing["state"] == WorkflowState.DATA_PLAN_PREPARING.value
    run_once(svc.store, svc.tools, svc)
    return svc.snapshot(review["workflow_id"])


def test_snapshot_contains_visual_steps_and_available_actions(tmp_path):
    svc = service(
        tmp_path,
        ReadyPlanner(),
        FakeTools(),
        fake_policy,
        pass_preflight,
        fake_data_preparer,
    )

    snapshot = svc.create_workflow("训练 FC 路由")

    assert len(snapshot["workflow_steps"]) == 8
    assert snapshot["workflow_steps"][0]["status"] == "needs_confirmation"
    assert snapshot["available_actions"] == [
        "confirm_requirements",
        "revise_requirements",
        "cancel",
    ]


def test_cancel_completion_preserves_training_artifact_references(tmp_path):
    svc = service(
        tmp_path,
        ReadyPlanner(),
        FakeTools(),
        fake_policy,
        pass_preflight,
        fake_data_preparer,
    )
    workflow_id = svc.store.create_workflow()
    for event in (
        "requirements_completed",
        "datagen_started",
        "datagen_completed",
        "train_plan_created",
        "preflight_passed",
        "training_started",
    ):
        svc.store.transition(workflow_id, event, {})
    svc.store.update_workflow_fields(workflow_id, train_job_id="train_1")
    before = svc.snapshot(workflow_id)["workflow_steps"][5]["artifacts"]
    svc.tools.stop_external_job = Mock(
        return_value={"stopped": True, "terminal": True, "detail": "STOPPED"}
    )

    svc.cancel(workflow_id, "用户手动中止")
    run_once(svc.store, svc.tools, svc, limit=1)
    after = svc.snapshot(workflow_id)

    assert after["state"] == "cancelled"
    assert after["workflow_steps"][5]["artifacts"] == before


def approve_datagen(svc):
    snapshot = create_data_plan_ready(svc)
    approval = snapshot["pending_approvals"][0]
    return svc.approve(
        snapshot["workflow_id"], approval["approval_id"], approval["plan_hash"]
    )


def test_direct_approval_execution_keeps_its_action_lease_alive(tmp_path):
    svc = service(
        tmp_path,
        ReadyPlanner(),
        FakeTools(),
        fake_policy,
        pass_preflight,
        fake_data_preparer,
    )
    heartbeat = Mock(return_value=nullcontext())
    svc.store.action_lease_heartbeat = heartbeat

    approve_datagen(svc)

    assert heartbeat.call_count == 3
    lease_seconds = [call.kwargs["lease_seconds"] for call in heartbeat.call_args_list]
    assert lease_seconds == [600, 120, 600]


def successful_datagen_results():
    return [
        {
            "job_id": "20260819T010203Z-a1b2c3",
            "task_type": "fc",
            "status": "succeeded",
            "output": "/tmp/fc-output.json",
        }
    ]


def comparison(*, primary_improvement: float) -> dict:
    return {
        "per_model": {
            "baseline": {
                "function_call": {
                    "n": 100,
                    "tool_name_accuracy": 0.80,
                    "invalid_rate": 0.01,
                }
            },
            "candidate": {
                "function_call": {
                    "n": 100,
                    "tool_name_accuracy": 0.80 + primary_improvement,
                    "invalid_rate": 0.01,
                }
            },
        },
        "paired_comparison": {
            "n": 100,
            "paired_score_n": 100,
            "mean_delta": primary_improvement,
            "critical_slice_rate_regression": 0.0,
            "critical_slice_score_regression": 0.0,
            "slices": {"fc": {"n": 100}},
        },
    }


def reach_evaluating(svc: AssistantService) -> dict:
    generating = approve_datagen(svc)
    ready = svc.on_datagen_terminal(
        generating["workflow_id"], successful_datagen_results()
    )
    training_approval = next(
        item
        for item in ready["pending_approvals"]
        if item["action"] == "start_training"
    )
    training = svc.approve(
        ready["workflow_id"],
        training_approval["approval_id"],
        training_approval["plan_hash"],
    )
    ab_ready = svc.on_training_observation(
        training["workflow_id"], {"status": "succeeded"}
    )
    evaluation_approval = next(
        item
        for item in ab_ready["pending_approvals"]
        if item["action"] == "start_evaluation"
    )
    return svc.approve(
        ab_ready["workflow_id"],
        evaluation_approval["approval_id"],
        evaluation_approval["plan_hash"],
    )


def test_ready_requirement_creates_data_plan_approval(tmp_path):
    fake_tools = FakeTools()
    svc = service(
        tmp_path,
        ReadyPlanner(),
        fake_tools,
        fake_policy,
        pass_preflight,
        fake_data_preparer,
    )
    review = svc.create_workflow("训练 FC 路由")
    assert review["state"] == WorkflowState.REQUIREMENTS_REVIEW.value
    assert [a["action"] for a in review["pending_approvals"]] == [
        "confirm_requirements"
    ]
    assert review["data_plan"] is None

    snapshot = create_data_plan_ready(
        service(
            tmp_path / "prepared",
            ReadyPlanner(),
            fake_tools,
            fake_policy,
            pass_preflight,
            fake_data_preparer,
        )
    )
    assert snapshot["state"] == WorkflowState.DATA_PLAN_READY.value
    assert [a["action"] for a in snapshot["pending_approvals"]] == [
        "start_datagen"
    ]
    assert fake_tools.datagen_calls == []
    assert "plan_json" not in snapshot["pending_approvals"][0]


def test_ambiguous_9b_fc_never_creates_data_plan_or_external_approval(tmp_path):
    class IncompletePlanner(ReadyPlanner):
        def __init__(self):
            super().__init__()
            self.data_plan_calls = []

        def extract_requirement_draft(self, messages):
            return RequirementDraft(
                assistant_reply=(
                    "暂定方案：9B 模型进行 FC SFT；请补充业务场景、"
                    "当前失败表现和期望行为。"
                ),
                ready_for_review=False,
                missing_fields=[
                    "scenario",
                    "current_problem",
                    "desired_behavior",
                ],
                proposed_objective=objective(),
                assumptions=["训练目标尚未得到用户确认"],
            )

        def create_data_plan(self, value):
            self.data_plan_calls.append(value)
            return super().create_data_plan(value)

    planner = IncompletePlanner()
    tools = FakeTools()
    svc = service(
        tmp_path,
        planner,
        tools,
        fake_policy,
        pass_preflight,
        fake_data_preparer,
    )

    snapshot = svc.create_workflow("我想训练9b模型")
    snapshot = svc.add_message(snapshot["workflow_id"], "fc")

    assert snapshot["state"] == WorkflowState.COLLECTING_REQUIREMENTS.value
    assert snapshot["data_plan"] is None
    assert snapshot["pending_approvals"] == []
    assert planner.data_plan_calls == []
    assert tools.datagen_calls == []


def test_permanent_data_plan_error_returns_to_review_and_can_retry(tmp_path):
    class InvalidDataPlanPlanner(ReadyPlanner):
        def create_data_plan(self, value):
            raise ValueError("planner schema mismatch")

    svc = service(
        tmp_path,
        InvalidDataPlanPlanner(),
        FakeTools(),
        fake_policy,
        pass_preflight,
        fake_data_preparer,
    )
    review = svc.create_workflow("训练 FC 路由")
    approval = review["pending_approvals"][0]
    preparing = svc.approve(
        review["workflow_id"], approval["approval_id"], approval["plan_hash"]
    )

    summary = run_once(svc.store, svc.tools, svc)
    failed = svc.snapshot(review["workflow_id"])

    assert preparing["state"] == WorkflowState.DATA_PLAN_PREPARING.value
    assert summary.failed == 1
    assert failed["state"] == WorkflowState.REQUIREMENTS_REVIEW.value
    assert any(
        event["event_type"] == "data_plan_preparation_failed"
        for event in failed["events"]
    )
    retried = svc.retry_data_plan(review["workflow_id"])
    assert retried["state"] == WorkflowState.DATA_PLAN_PREPARING.value


def test_rejecting_initial_data_plan_reopens_requirements_revision(tmp_path):
    svc = service(
        tmp_path,
        ReadyPlanner(),
        FakeTools(),
        fake_policy,
        pass_preflight,
        fake_data_preparer,
    )
    planned = create_data_plan_ready(svc)
    approval = planned["pending_approvals"][0]

    revising = svc.reject(planned["workflow_id"], approval["approval_id"])

    assert revising["state"] == WorkflowState.COLLECTING_REQUIREMENTS.value
    assert revising["pending_approvals"] == []
    review = svc.add_message(planned["workflow_id"], "把数据量改为 120 条")
    approval = review["pending_approvals"][0]
    svc.approve(
        review["workflow_id"], approval["approval_id"], approval["plan_hash"]
    )
    run_once(svc.store, svc.tools, svc)
    replanned = svc.snapshot(review["workflow_id"])
    assert replanned["state"] == WorkflowState.DATA_PLAN_READY.value
    assert [item["action"] for item in replanned["pending_approvals"]] == [
        "start_datagen"
    ]


def test_rejecting_initial_data_plan_does_not_use_split_store_operations(
    tmp_path, monkeypatch
):
    svc = service(
        tmp_path,
        ReadyPlanner(),
        FakeTools(),
        fake_policy,
        pass_preflight,
        fake_data_preparer,
    )
    planned = create_data_plan_ready(svc)
    approval = planned["pending_approvals"][0]

    def split_operation_was_used(*args, **kwargs):
        raise AssertionError("initial revision must be one store transaction")

    monkeypatch.setattr(svc.store, "reject_approval", split_operation_was_used)
    monkeypatch.setattr(svc.store, "transition", split_operation_was_used)
    monkeypatch.setattr(svc.store, "append_event", split_operation_was_used)
    monkeypatch.setattr(svc.store, "append_message", split_operation_was_used)

    revising = svc.reject(planned["workflow_id"], approval["approval_id"])

    assert revising["state"] == WorkflowState.COLLECTING_REQUIREMENTS.value
    assert svc.store.get_approval(
        planned["workflow_id"], approval["approval_id"]
    )["status"] == "rejected"
    assert [event["event_type"] for event in revising["events"][-2:]] == [
        "approval_rejected",
        "data_plan_revision_requested",
    ]


def test_approval_executes_exactly_once(tmp_path):
    fake_tools = FakeTools()
    svc = service(
        tmp_path,
        ReadyPlanner(),
        fake_tools,
        fake_policy,
        pass_preflight,
        fake_data_preparer,
    )
    snapshot = create_data_plan_ready(svc)
    approval = snapshot["pending_approvals"][0]
    first = svc.approve(
        snapshot["workflow_id"], approval["approval_id"], approval["plan_hash"]
    )
    assert first["state"] == WorkflowState.DATA_GENERATING.value
    assert len(fake_tools.datagen_calls) == 1
    with pytest.raises(ApprovalConflict):
        svc.approve(
            snapshot["workflow_id"], approval["approval_id"], approval["plan_hash"]
        )
    assert len(fake_tools.datagen_calls) == 1


def test_datagen_completion_creates_train_approval_only_after_passed_preflight(
    tmp_path,
):
    svc = service(
        tmp_path,
        ReadyPlanner(),
        FakeTools(),
        fake_policy,
        pass_preflight,
        fake_data_preparer,
    )
    snapshot = approve_datagen(svc)
    next_snapshot = svc.on_datagen_terminal(
        snapshot["workflow_id"], successful_datagen_results()
    )
    assert next_snapshot["state"] == WorkflowState.TRAIN_READY.value
    assert [a["action"] for a in next_snapshot["pending_approvals"]] == [
        "start_training"
    ]
    completed_event = next(
        event
        for event in next_snapshot["events"]
        if event["event_type"] == "datagen_completed"
    )
    assert completed_event["payload"]["dataset_profile"]["sha256"] == "a" * 64
    assert completed_event["payload"]["dataset_profile"]["eval_sha256"] == {
        "function_call": "b" * 64
    }


def test_missing_critical_slice_in_frozen_holdout_blocks_training(tmp_path):
    class CriticalSlicePlanner(ReadyPlanner):
        def extract_requirement_draft(self, messages):
            draft = super().extract_requirement_draft(messages)
            return draft.model_copy(
                update={
                    "proposed_objective": objective().model_copy(
                        update={"critical_slices": ["tool_name=missing_tool"]}
                    )
                }
            )

    policy = Mock(side_effect=AssertionError("training plan must not be created"))
    svc = service(
        tmp_path,
        CriticalSlicePlanner(),
        FakeTools(),
        policy,
        pass_preflight,
        fake_data_preparer,
    )
    generating = approve_datagen(svc)

    blocked = svc.on_datagen_terminal(
        generating["workflow_id"], successful_datagen_results()
    )

    assert blocked["state"] == WorkflowState.DATA_PLAN_READY.value
    assert [item["action"] for item in blocked["pending_approvals"]] == [
        "start_datagen"
    ]
    failed = next(
        event for event in blocked["events"]
        if event["event_type"] == "datagen_failed"
    )
    assert failed["payload"]["missing_critical_slices"] == [
        "tool_name=missing_tool"
    ]
    policy.assert_not_called()


def test_iteration_training_adjustment_recalculates_plan_estimates():
    adjusted = AssistantService._apply_training_adjustments(
        fake_policy(),
        [
            TrainingAdjustment(
                parameter="num_train_epochs",
                value=6,
                reason="underfit evidence",
            ),
            TrainingAdjustment(
                parameter="lora_rank",
                value=16,
                reason="underfit capacity evidence",
            ),
        ],
    )

    assert adjusted.config.train.num_train_epochs == 6
    assert adjusted.config.method.lora_rank == 16
    assert adjusted.estimated_steps == 20
    assert adjusted.estimated_vram_gb == 40.0
    decision = next(
        item for item in adjusted.decisions if item.parameter == "lora_rank"
    )
    assert decision.value == 16


def test_blocked_preflight_has_no_training_approval(tmp_path):
    svc = service(
        tmp_path,
        ReadyPlanner(),
        FakeTools(),
        fake_policy,
        block_preflight,
        fake_data_preparer,
    )
    snapshot = approve_datagen(svc)
    next_snapshot = svc.on_datagen_terminal(
        snapshot["workflow_id"], successful_datagen_results()
    )
    assert next_snapshot["state"] == WorkflowState.PREFLIGHT_BLOCKED.value
    assert next_snapshot["pending_approvals"] == []


def test_datagen_failure_returns_to_approved_retry_without_training(tmp_path):
    preparer = Mock()
    svc = service(
        tmp_path,
        ReadyPlanner(),
        FakeTools(),
        fake_policy,
        pass_preflight,
        preparer,
    )
    snapshot = approve_datagen(svc)
    retry = svc.on_datagen_terminal(
        snapshot["workflow_id"],
        [
            {
                "job_id": "fc1",
                "task_type": "fc",
                "status": "failed",
                "error": "sanitized",
            }
        ],
    )
    assert retry["state"] == WorkflowState.DATA_PLAN_READY.value
    assert [a["action"] for a in retry["pending_approvals"]] == [
        "start_datagen"
    ]
    preparer.assert_not_called()


def test_dataset_preparation_failure_is_retryable_from_generating_state(tmp_path):
    preparer = Mock(
        side_effect=[RuntimeError("temporary local write failure"), fake_data_preparer()]
    )
    svc = service(
        tmp_path,
        ReadyPlanner(),
        FakeTools(),
        fake_policy,
        pass_preflight,
        preparer,
    )
    generating = approve_datagen(svc)

    with pytest.raises(RuntimeError, match="temporary"):
        svc.on_datagen_terminal(
            generating["workflow_id"], successful_datagen_results()
        )
    assert (
        svc.store.get_workflow(generating["workflow_id"])["state"]
        == WorkflowState.DATA_GENERATING.value
    )

    recovered = svc.on_datagen_terminal(
        generating["workflow_id"], successful_datagen_results()
    )
    assert recovered["state"] == WorkflowState.TRAIN_READY.value


def test_training_approval_rechecks_preflight_before_remote_start(tmp_path):
    checks = iter([pass_preflight(), block_preflight()])
    tools = FakeTools()
    svc = service(
        tmp_path,
        ReadyPlanner(),
        tools,
        fake_policy,
        lambda *args, **kwargs: next(checks),
        fake_data_preparer,
    )
    snapshot = approve_datagen(svc)
    ready = svc.on_datagen_terminal(
        snapshot["workflow_id"], successful_datagen_results()
    )
    approval = ready["pending_approvals"][0]
    blocked = svc.approve(
        ready["workflow_id"], approval["approval_id"], approval["plan_hash"]
    )
    assert blocked["state"] == WorkflowState.PREFLIGHT_BLOCKED.value
    assert tools.training_calls == []


def test_training_start_replays_durable_intent_with_same_job_id(tmp_path):
    tools = CrashAfterTrainingSubmitTools()
    svc = service(
        tmp_path,
        ReadyPlanner(),
        tools,
        fake_policy,
        pass_preflight,
        fake_data_preparer,
    )
    generating = approve_datagen(svc)
    ready = svc.on_datagen_terminal(
        generating["workflow_id"], successful_datagen_results()
    )
    for stale_monitor in svc.store.lease_due_actions(
        datetime.now(timezone.utc), limit=10
    ):
        svc.store.complete_action(
            stale_monitor["action_id"], stale_monitor["lease_token"]
        )
    approval = ready["pending_approvals"][0]

    queued = svc.approve(
        ready["workflow_id"], approval["approval_id"], approval["plan_hash"]
    )

    assert queued["state"] == WorkflowState.TRAIN_READY.value
    assert svc.store.get_approval(
        ready["workflow_id"], approval["approval_id"]
    )["status"] == "executing"
    assert svc.store.count_pending_actions("execute_approval") == 1

    summary = run_once(
        svc.store,
        tools,
        svc,
        now=datetime.now(timezone.utc) + timedelta(seconds=61),
    )

    assert summary.failed == 0
    recovered = svc.snapshot(ready["workflow_id"])
    assert recovered["state"] == WorkflowState.TRAINING.value
    assert tools.submit_job_ids == [
        "20260819T020304Z-d4e5f6",
        "20260819T020304Z-d4e5f6",
    ]
    assert svc.store.count_pending_actions("execute_approval") == 0
    assert svc.store.count_pending_actions("monitor_training") == 1


def test_local_approval_crash_replays_idempotently_from_durable_outbox(tmp_path):
    svc = service(
        tmp_path,
        ReadyPlanner(),
        FakeTools(),
        fake_policy,
        pass_preflight,
        fake_data_preparer,
    )
    generating = approve_datagen(svc)
    ready = svc.on_datagen_terminal(
        generating["workflow_id"], successful_datagen_results()
    )
    train_approval = ready["pending_approvals"][0]
    training = svc.approve(
        ready["workflow_id"],
        train_approval["approval_id"],
        train_approval["plan_hash"],
    )
    ab_ready = svc.on_training_observation(
        training["workflow_id"], {"status": "succeeded"}
    )
    skip = next(
        item for item in ab_ready["pending_approvals"]
        if item["action"] == "skip_evaluation"
    )
    original_commit = svc.store.commit_local_execution
    calls = 0

    def crash_once(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("process exited after state transition")
        return original_commit(**kwargs)

    svc.store.commit_local_execution = crash_once
    crashed = svc.approve(
        ab_ready["workflow_id"], skip["approval_id"], skip["plan_hash"]
    )
    assert crashed["state"] == WorkflowState.COMPLETED.value
    assert svc.store.count_pending_actions("execute_approval") == 1

    # A second process opening the store must not orphan the live outbox.
    AssistantStore(svc.store.path)
    replay = next(
        action
        for action in svc.store.lease_due_actions(
            datetime.now(timezone.utc) + timedelta(seconds=61)
        )
        if action["idempotency_key"]
        == f"approval-execution:{skip['approval_id']}"
    )
    svc.execute_approval_action(replay)
    assert svc.store.get_approval(
        ab_ready["workflow_id"], skip["approval_id"]
    )["status"] == "consumed"


def test_permanent_training_submit_error_reopens_approval(tmp_path):
    svc = service(
        tmp_path,
        ReadyPlanner(),
        PermanentTrainingErrorTools(),
        fake_policy,
        pass_preflight,
        fake_data_preparer,
    )
    generating = approve_datagen(svc)
    ready = svc.on_datagen_terminal(
        generating["workflow_id"], successful_datagen_results()
    )
    approval = ready["pending_approvals"][0]

    recovered = svc.approve(
        ready["workflow_id"], approval["approval_id"], approval["plan_hash"]
    )

    assert recovered["state"] == WorkflowState.TRAIN_READY.value
    assert svc.store.get_approval(
        ready["workflow_id"], approval["approval_id"]
    )["status"] == "failed"
    assert [item["action"] for item in recovered["pending_approvals"]] == [
        "start_training"
    ]
    action = svc.store.get_action_by_key(
        f"approval-execution:{approval['approval_id']}"
    )
    assert action["status"] == "failed"


def test_permanent_data_preparation_error_returns_to_datagen_recovery(tmp_path):
    def broken_preparer(*args, **kwargs):
        raise ValueError("fc: insufficient accepted records")

    svc = service(
        tmp_path,
        ReadyPlanner(),
        FakeTools(),
        fake_policy,
        pass_preflight,
        broken_preparer,
    )
    generating = approve_datagen(svc)

    recovered = svc.on_datagen_terminal(
        generating["workflow_id"], successful_datagen_results()
    )

    assert recovered["state"] == WorkflowState.DATA_PLAN_READY.value
    assert [item["action"] for item in recovered["pending_approvals"]] == [
        "start_datagen"
    ]
    failed = next(
        event for event in recovered["events"]
        if event["event_type"] == "datagen_failed"
    )
    assert "insufficient accepted records" in failed["payload"]["error"]


def test_changed_warning_requires_fresh_approval(tmp_path):
    checks = iter([preflight("warn", "warm"), preflight("warn", "hotter")])
    tools = FakeTools()
    svc = service(
        tmp_path,
        ReadyPlanner(),
        tools,
        fake_policy,
        lambda *args, **kwargs: next(checks),
        fake_data_preparer,
    )
    generating = approve_datagen(svc)
    ready = svc.on_datagen_terminal(
        generating["workflow_id"], successful_datagen_results()
    )
    old = ready["pending_approvals"][0]
    refreshed = svc.approve(
        ready["workflow_id"], old["approval_id"], old["plan_hash"]
    )
    assert tools.training_calls == []
    assert len(refreshed["pending_approvals"]) == 1
    assert refreshed["pending_approvals"][0]["approval_id"] != old["approval_id"]


def test_successful_training_asks_before_starting_ab_evaluation(tmp_path):
    tools = FakeTools()
    svc = service(
        tmp_path,
        ReadyPlanner(),
        tools,
        fake_policy,
        pass_preflight,
        fake_data_preparer,
    )
    generating = approve_datagen(svc)
    ready = svc.on_datagen_terminal(
        generating["workflow_id"], successful_datagen_results()
    )
    approval = ready["pending_approvals"][0]
    training = svc.approve(
        ready["workflow_id"], approval["approval_id"], approval["plan_hash"]
    )
    assert training["state"] == WorkflowState.TRAINING.value

    completed = svc.on_training_observation(
        training["workflow_id"], {"status": "succeeded"}
    )
    assert completed["state"] == WorkflowState.AB_PLAN_READY.value
    assert {item["action"] for item in completed["pending_approvals"]} == {
        "start_evaluation",
        "skip_evaluation",
    }
    assert completed["training_plan"]["dataset_sha256"] == "a" * 64
    assert completed["evaluation_plan"]["eval_sha256"] == {
        "function_call": "b" * 64
    }
    assert completed["evaluation_plan"]["critical_slices"] == ["fc"]
    assert completed["evaluation_plan"]["execution_request"]["models"][1][
        "adapter_path"
    ].endswith(training["train_job_id"])
    assert completed["evaluation_plan"]["baseline_source_hash"].startswith(
        "sha256:"
    )
    assert completed["evaluation_plan"]["candidate_source_hash"].startswith(
        "sha256:"
    )
    duplicate = svc.on_training_observation(
        training["workflow_id"], {"status": "succeeded"}
    )
    assert duplicate["state"] == WorkflowState.AB_PLAN_READY.value
    assert len(duplicate["pending_approvals"]) == 2


def test_training_terminal_transition_and_ab_approvals_are_atomic(tmp_path):
    svc = service(
        tmp_path,
        ReadyPlanner(),
        FakeTools(),
        fake_policy,
        pass_preflight,
        fake_data_preparer,
    )
    generating = approve_datagen(svc)
    ready = svc.on_datagen_terminal(
        generating["workflow_id"], successful_datagen_results()
    )
    approval = ready["pending_approvals"][0]
    training = svc.approve(
        ready["workflow_id"], approval["approval_id"], approval["plan_hash"]
    )
    svc.store.create_approval = Mock(
        side_effect=RuntimeError("legacy non-atomic approval path")
    )

    completed = svc.on_training_observation(
        training["workflow_id"], {"status": "succeeded"}
    )

    assert completed["state"] == WorkflowState.AB_PLAN_READY.value
    assert {item["action"] for item in completed["pending_approvals"]} == {
        "start_evaluation",
        "skip_evaluation",
    }


def test_failed_training_offers_checkpoint_recovery_with_log_evidence(tmp_path):
    svc = service(
        tmp_path,
        ReadyPlanner(),
        FakeTools(),
        fake_policy,
        pass_preflight,
        fake_data_preparer,
    )
    generating = approve_datagen(svc)
    ready = svc.on_datagen_terminal(
        generating["workflow_id"], successful_datagen_results()
    )
    approval = ready["pending_approvals"][0]
    training = svc.approve(
        ready["workflow_id"], approval["approval_id"], approval["plan_hash"]
    )

    failed = svc.on_training_observation(
        training["workflow_id"],
        {
            "status": "failed",
            "failure_category": "oom",
            "error": "CUDA out of memory",
            "log_tail": "Traceback\nCUDA out of memory\nsecret\x00removed",
            "checkpoints": [
                {
                    "name": "checkpoint-100",
                    "path": "/opt/LF/saves/run/checkpoint-100",
                },
                {
                    "name": "checkpoint-20",
                    "path": "/opt/LF/saves/run/checkpoint-20",
                }
            ],
        },
    )

    assert failed["state"] == WorkflowState.TRAIN_FAILED.value
    recovery = next(
        item
        for item in failed["pending_approvals"]
        if item["action"] == "recover_training"
    )
    failure_event = next(
        event
        for event in reversed(failed["events"])
        if event["event_type"] == "training_failed"
    )
    assert "CUDA out of memory" in failure_event["payload"]["log_excerpt"]
    assert "\x00" not in failure_event["payload"]["log_excerpt"]

    recovered = svc.approve(
        failed["workflow_id"],
        recovery["approval_id"],
        recovery["plan_hash"],
    )
    assert recovered["state"] == WorkflowState.TRAIN_READY.value
    assert recovered["training_plan"]["config"]["train"][
        "resume_from_checkpoint"
    ] == "/opt/LF/saves/run/checkpoint-100"
    assert any(
        item["action"] == "start_training"
        for item in recovered["pending_approvals"]
    )


def test_evaluation_builds_diagnosis_derived_next_data_plan(tmp_path):
    planner = ReadyPlanner()
    svc = service(
        tmp_path,
        planner,
        FakeTools(),
        fake_policy,
        pass_preflight,
        fake_data_preparer,
    )
    evaluating = reach_evaluating(svc)
    heartbeat = Mock(return_value=nullcontext())
    svc.store.action_lease_heartbeat = heartbeat

    diagnosed = svc.on_evaluation_terminal(
        evaluating["workflow_id"], {"status": "succeeded"}
    )

    assert diagnosed["state"] == WorkflowState.DIAGNOSIS_READY.value
    assert diagnosed["data_plan"]["rationale"] == "Diagnosis-derived coverage repair"
    assert len(planner.iteration_calls) == 1
    assert {item["action"] for item in diagnosed["pending_approvals"]} == {
        "finish_without_accepting",
        "start_iteration",
    }
    heartbeat.assert_called_once()
    assert heartbeat.call_args.kwargs["lease_seconds"] == 600
    duplicate = svc.on_evaluation_terminal(
        evaluating["workflow_id"], {"status": "succeeded"}
    )
    assert duplicate["state"] == WorkflowState.DIAGNOSIS_READY.value
    assert len(planner.iteration_calls) == 1


def test_diagnosis_uses_persisted_training_trends_and_success_event_is_redacted(
    tmp_path,
):
    svc = service(
        tmp_path,
        ReadyPlanner(),
        FakeTools(),
        fake_policy,
        pass_preflight,
        fake_data_preparer,
    )
    generating = approve_datagen(svc)
    ready = svc.on_datagen_terminal(
        generating["workflow_id"], successful_datagen_results()
    )
    approval = next(
        item for item in ready["pending_approvals"]
        if item["action"] == "start_training"
    )
    training = svc.approve(
        ready["workflow_id"], approval["approval_id"], approval["plan_hash"]
    )
    ab_ready = svc.on_training_observation(
        training["workflow_id"],
        {
            "status": "succeeded",
            "metrics": {
                "points": [
                    {"step": 1, "loss": 2.0},
                    {"step": 20, "loss": 1.99},
                ],
                "total_steps": 20,
            },
            "log_tail": "api_key=topsecret /Users/alice/.ssh/id_ed25519",
        },
    )
    training_event = next(
        event for event in ab_ready["events"]
        if event["event_type"] == "training_succeeded"
    )
    assert training_event["payload"]["training_evidence"]["loss_trend"] == "flat"
    assert "log_tail" not in training_event["payload"]
    assert "topsecret" not in str(training_event)

    eval_approval = next(
        item for item in ab_ready["pending_approvals"]
        if item["action"] == "start_evaluation"
    )
    evaluating = svc.approve(
        ab_ready["workflow_id"],
        eval_approval["approval_id"],
        eval_approval["plan_hash"],
    )
    diagnosed = svc.on_evaluation_terminal(
        evaluating["workflow_id"],
        {"status": "succeeded", "training_evidence": {"loss_trend": "down"}},
    )
    assert diagnosed["diagnosis"]["category"] == "underfit"


def test_iteration_planner_failure_does_not_lose_completed_diagnosis(tmp_path):
    planner = ReadyPlanner()
    planner.create_iteration_plan = Mock(
        side_effect=RuntimeError("planner temporarily unavailable")
    )
    svc = service(
        tmp_path,
        planner,
        FakeTools(),
        fake_policy,
        pass_preflight,
        fake_data_preparer,
    )
    evaluating = reach_evaluating(svc)

    diagnosed = svc.on_evaluation_terminal(
        evaluating["workflow_id"], {"status": "succeeded"}
    )

    assert diagnosed["state"] == WorkflowState.DIAGNOSIS_READY.value
    assert diagnosed["diagnosis"]["accepted"] is False
    assert [
        item["action"] for item in diagnosed["pending_approvals"]
    ] == ["finish_without_accepting"]
    assert svc.store.count_pending_actions("plan_iteration") == 1


def test_accepted_candidate_does_not_propose_another_iteration(tmp_path):
    planner = ReadyPlanner()
    tools = FakeTools()
    tools.score_evaluation = Mock(return_value=comparison(primary_improvement=0.20))
    svc = service(
        tmp_path,
        planner,
        tools,
        fake_policy,
        pass_preflight,
        fake_data_preparer,
    )
    evaluating = reach_evaluating(svc)

    diagnosed = svc.on_evaluation_terminal(
        evaluating["workflow_id"], {"status": "succeeded"}
    )

    assert diagnosed["diagnosis"]["accepted"] is True
    assert [
        item["action"] for item in diagnosed["pending_approvals"]
    ] == ["accept_candidate"]
    assert planner.iteration_calls == []
    assert not any(
        event["event_type"] == "manual_review_required"
        for event in diagnosed["events"]
    )
    pending = svc.store.get_action_by_key(
        f"{diagnosed['workflow_id']}:diagnosis-explanation:0"
    )
    action = svc.store.lease_action(pending["action_id"])
    assert action["action"] == "explain_diagnosis"
    svc.explain_diagnosis_action(action)
    explained = svc.snapshot(diagnosed["workflow_id"])
    assert explained["messages"][-1]["content"] == "A/B 证据已完成规则诊断。"


def test_evaluation_quality_issue_retries_evidence_instead_of_retraining(tmp_path):
    planner = ReadyPlanner()
    tools = FakeTools()
    result = comparison(primary_improvement=0.01)
    result["paired_comparison"]["n"] = 20
    result["paired_comparison"]["paired_score_n"] = 20
    result["paired_comparison"]["slices"]["fc"]["n"] = 20
    tools.score_evaluation = Mock(return_value=result)
    svc = service(
        tmp_path, planner, tools, fake_policy, pass_preflight, fake_data_preparer
    )
    evaluating = reach_evaluating(svc)

    diagnosed = svc.on_evaluation_terminal(
        evaluating["workflow_id"], {"status": "succeeded"}
    )

    assert diagnosed["diagnosis"]["category"] == "evaluation_quality_issue"
    assert {item["action"] for item in diagnosed["pending_approvals"]} == {
        "finish_without_accepting",
        "start_evaluation",
    }
    assert svc.store.count_pending_actions("plan_iteration") == 0


def test_failed_evaluation_returns_to_fresh_retry_or_skip_approval(tmp_path):
    svc = service(
        tmp_path,
        ReadyPlanner(),
        FakeTools(),
        fake_policy,
        pass_preflight,
        fake_data_preparer,
    )
    evaluating = reach_evaluating(svc)

    retry = svc.on_evaluation_terminal(
        evaluating["workflow_id"],
        {"status": "failed", "error": "judge temporarily unavailable"},
    )

    assert retry["state"] == WorkflowState.AB_PLAN_READY.value
    assert {item["action"] for item in retry["pending_approvals"]} == {
        "start_evaluation",
        "skip_evaluation",
    }


def test_permanent_scoring_error_returns_to_ab_retry_without_acceptance(tmp_path):
    tools = FakeTools()
    tools.score_evaluation = Mock(
        side_effect=ValueError("scores.jsonl contains corrupt interior row")
    )
    svc = service(
        tmp_path,
        ReadyPlanner(),
        tools,
        fake_policy,
        pass_preflight,
        fake_data_preparer,
    )
    evaluating = reach_evaluating(svc)

    retry = svc.on_evaluation_terminal(
        evaluating["workflow_id"], {"status": "succeeded"}
    )

    assert retry["state"] == WorkflowState.AB_PLAN_READY.value
    assert retry["diagnosis"] is None
    assert {item["action"] for item in retry["pending_approvals"]} == {
        "retry_scoring",
        "skip_evaluation",
    }
    failed = next(
        event for event in retry["events"]
        if event["event_type"] == "evaluation_failed"
    )
    assert failed["payload"]["status"] == "scoring_failed"

    original_eval_id = retry["eval_id"]
    retry_approval = next(
        item for item in retry["pending_approvals"]
        if item["action"] == "retry_scoring"
    )
    pending_before = svc.store.count_pending_actions("monitor_evaluation")
    resumed = svc.approve(
        retry["workflow_id"], retry_approval["approval_id"], retry_approval["plan_hash"]
    )
    assert resumed["state"] == WorkflowState.EVALUATING.value
    assert resumed["eval_id"] == original_eval_id
    assert len(tools.evaluation_calls) == 1
    assert svc.store.count_pending_actions("monitor_evaluation") == pending_before + 1


def test_scoring_retry_outbox_reconciles_after_monitor_already_advanced_state(tmp_path):
    tools = FakeTools()
    tools.score_evaluation = Mock(
        side_effect=ValueError("judge unavailable")
    )
    svc = service(
        tmp_path,
        ReadyPlanner(),
        tools,
        fake_policy,
        pass_preflight,
        fake_data_preparer,
    )
    evaluating = reach_evaluating(svc)
    retry = svc.on_evaluation_terminal(
        evaluating["workflow_id"], {"status": "succeeded"}
    )
    approval = next(
        item for item in retry["pending_approvals"]
        if item["action"] == "retry_scoring"
    )
    payload = svc._approval_payload_for_action(
        retry["workflow_id"], ActionKind.RETRY_SCORING
    )
    execution = svc.store.prepare_external_execution(
        retry["workflow_id"],
        approval["approval_id"],
        approval["plan_hash"],
        payload,
        {},
    )
    svc._execute_approved(
        retry["workflow_id"],
        ActionKind.RETRY_SCORING,
        payload,
        execution_key=execution["action_id"],
    )
    tools.score_evaluation = Mock(return_value=comparison(primary_improvement=0.2))
    advanced = svc.on_evaluation_terminal(
        retry["workflow_id"], {"status": "succeeded"}
    )
    assert advanced["state"] == WorkflowState.DIAGNOSIS_READY.value

    svc.execute_approval_action(execution)

    assert svc.store.get_approval(
        retry["workflow_id"], approval["approval_id"]
    )["status"] == "consumed"


def test_missing_baseline_evidence_returns_to_ab_retry_without_acceptance(tmp_path):
    planner = ReadyPlanner()
    tools = FakeTools()
    tools.score_evaluation = Mock(
        return_value={
            "per_model": {
                "baseline": {"error": "predictions download failed"},
                "candidate": comparison(primary_improvement=0.30)["per_model"][
                    "candidate"
                ],
            },
            "paired_comparison": {
                "n": 100,
                "baseline_invalid_rate": 1.0,
                "candidate_invalid_rate": 0.0,
                "mean_delta": 0.30,
            },
        }
    )
    svc = service(
        tmp_path,
        planner,
        tools,
        fake_policy,
        pass_preflight,
        fake_data_preparer,
    )
    evaluating = reach_evaluating(svc)

    retry = svc.on_evaluation_terminal(
        evaluating["workflow_id"], {"status": "succeeded"}
    )

    assert retry["state"] == WorkflowState.AB_PLAN_READY.value
    assert retry["diagnosis"] is None
    assert {item["action"] for item in retry["pending_approvals"]} == {
        "start_evaluation",
        "skip_evaluation",
    }
    assert planner.iteration_calls == []


def test_preflight_rejects_inapplicable_workflow_state(tmp_path):
    svc = service(
        tmp_path,
        ReadyPlanner(),
        FakeTools(),
        fake_policy,
        pass_preflight,
        fake_data_preparer,
    )
    snapshot = svc.create_workflow("train FC")
    with pytest.raises(ApprovalConflict, match="preflight"):
        svc.run_preflight(snapshot["workflow_id"])


def test_two_consecutive_missed_improvements_stop_iteration_proposal(tmp_path):
    svc = service(
        tmp_path,
        ReadyPlanner(),
        FakeTools(),
        fake_policy,
        pass_preflight,
        fake_data_preparer,
    )
    evaluating = reach_evaluating(svc)
    svc.store.append_event(
        evaluating["workflow_id"],
        "evaluation_completed",
        {
            "primary_improvement": 0.05,
            "iteration": -1,
            "train_job_id": "20260818T020304Z-abcdef",
            "eval_id": "20260818T030405Z-abcdef",
        },
    )

    diagnosed = svc.on_evaluation_terminal(
        evaluating["workflow_id"], {"status": "succeeded"}
    )

    assert [
        item["action"] for item in diagnosed["pending_approvals"]
    ] == ["finish_without_accepting"]
    attention = [
        event
        for event in diagnosed["events"]
        if event["event_type"] == "manual_review_required"
    ]
    assert attention[-1]["payload"]["reason"] == (
        "two consecutive evaluations missed min_improvement"
    )


def test_repeated_scoring_of_same_candidate_does_not_count_as_new_iteration(tmp_path):
    svc = service(
        tmp_path,
        ReadyPlanner(),
        FakeTools(),
        fake_policy,
        pass_preflight,
        fake_data_preparer,
    )
    evaluating = reach_evaluating(svc)
    svc.store.append_event(
        evaluating["workflow_id"],
        "evaluation_completed",
        {
            "primary_improvement": 0.05,
            "iteration": evaluating["iteration"],
            "train_job_id": evaluating["train_job_id"],
            "eval_id": evaluating["eval_id"],
        },
    )

    diagnosed = svc.on_evaluation_terminal(
        evaluating["workflow_id"], {"status": "succeeded"}
    )

    assert "start_iteration" in {
        item["action"] for item in diagnosed["pending_approvals"]
    }


def test_approved_iteration_clears_previous_round_runtime_fields(tmp_path):
    svc = service(
        tmp_path,
        ReadyPlanner(),
        FakeTools(),
        fake_policy,
        pass_preflight,
        fake_data_preparer,
    )
    evaluating = reach_evaluating(svc)
    diagnosed = svc.on_evaluation_terminal(
        evaluating["workflow_id"], {"status": "succeeded"}
    )
    approval = next(
        item
        for item in diagnosed["pending_approvals"]
        if item["action"] == "start_iteration"
    )
    next_round = svc.approve(
        diagnosed["workflow_id"], approval["approval_id"], approval["plan_hash"]
    )
    assert next_round["iteration"] == 1
    assert next_round["state"] == WorkflowState.DATA_PLAN_READY.value
    for field in (
        "dataset_profile",
        "training_plan",
        "preflight",
        "evaluation_plan",
        "diagnosis",
        "train_job_id",
        "eval_id",
    ):
        assert next_round[field] is None


def test_train_job_baseline_is_resolved_to_its_adapter(tmp_path):
    tools = FakeTools()
    svc = service(
        tmp_path,
        ReadyPlanner(),
        tools,
        fake_policy,
        pass_preflight,
        fake_data_preparer,
    )
    baseline_objective = objective().model_copy(
        update={
            "baseline": BaselineSpec(
                kind="train_job",
                name="champion",
                train_job_id="20260818T010203Z-a1b2c3",
            )
        }
    )
    plan = EvaluationPlan(
        eval_dataset_names={"function_call": "assistant_wf_it0_eval_fc"},
        task_types=["function_call"],
        gpus="0",
        success_criteria=baseline_objective.success_criteria,
    )

    request = svc._build_eval_request(
        {
            "objective": baseline_objective.model_dump(mode="json"),
            "training_plan": fake_policy().model_dump(mode="json"),
            "train_job_id": "20260819T020304Z-d4e5f6",
        },
        plan,
    )

    assert request.models[0].model_name_or_path == "/models/champion-base"
    assert request.models[0].adapter_path == "/saves/champion"
    assert request.models[0].template == request.models[1].template
    assert request.models[1].adapter_path.endswith(
        "/20260819T020304Z-d4e5f6"
    )

    locked = plan.model_copy(
        update={
            "execution_request": request,
            "baseline_source_hash": plan_hash(
                request.models[0].model_dump(mode="json")
            ),
        }
    )
    tools.resolve_train_job = Mock(
        return_value={
            "model_name_or_path": "/models/changed",
            "adapter_path": "/saves/champion",
            "template": "qwen3_5_nothink",
        }
    )
    with pytest.raises(ValueError, match="fingerprint changed"):
        svc._build_eval_request(
            {
                "objective": baseline_objective.model_dump(mode="json"),
                "training_plan": fake_policy().model_dump(mode="json"),
                "train_job_id": "20260819T020304Z-d4e5f6",
            },
            locked,
        )


def test_recovery_evaluation_uses_resumed_output_directory(tmp_path):
    tools = FakeTools()
    tools.resolve_train_job = Mock(
        return_value={
            "model_name_or_path": "/models/qwen",
            "adapter_path": "/saves/run",
            "template": "qwen3_5_nothink",
        }
    )
    svc = service(
        tmp_path,
        ReadyPlanner(),
        tools,
        fake_policy,
        pass_preflight,
        fake_data_preparer,
    )
    objective_payload = objective().model_dump(mode="json")
    training_plan = fake_policy()
    training_plan = training_plan.model_copy(
        update={
            "config": training_plan.config.model_copy(
                update={
                    "train": training_plan.config.train.model_copy(
                        update={
                            "resume_from_checkpoint": "/saves/run/checkpoint-100"
                        }
                    ),
                    "output": training_plan.config.output.model_copy(
                        update={"output_dir": "/saves/run"}
                    ),
                }
            )
        }
    )
    plan = EvaluationPlan(
        eval_dataset_names={"function_call": "assistant_wf_it0_eval_fc"},
        task_types=["function_call"],
        gpus="0",
        success_criteria=objective().success_criteria,
    )

    request = svc._build_eval_request(
        {
            "objective": objective_payload,
            "training_plan": training_plan.model_dump(mode="json"),
            "train_job_id": "20260819T020304Z-d4e5f6",
        },
        plan,
    )

    assert request.models[1].adapter_path == "/saves/run"


def test_train_job_baseline_template_mismatch_blocks_ab_request(tmp_path):
    tools = FakeTools()
    tools.resolve_train_job = Mock(
        return_value={
            "model_name_or_path": "/models/champion-base",
            "adapter_path": "/saves/champion",
            "template": "different_template",
        }
    )
    svc = service(
        tmp_path,
        ReadyPlanner(),
        tools,
        fake_policy,
        pass_preflight,
        fake_data_preparer,
    )
    baseline_objective = objective().model_copy(
        update={
            "baseline": BaselineSpec(
                kind="train_job",
                name="champion",
                train_job_id="20260818T010203Z-a1b2c3",
            )
        }
    )
    plan = EvaluationPlan(
        eval_dataset_names={"function_call": "assistant_wf_it0_eval_fc"},
        task_types=["function_call"],
        gpus="0",
        success_criteria=baseline_objective.success_criteria,
    )

    with pytest.raises(ValueError, match="template differs"):
        svc._build_eval_request(
            {
                "objective": baseline_objective.model_dump(mode="json"),
                "training_plan": fake_policy().model_dump(mode="json"),
                "train_job_id": "20260819T020304Z-d4e5f6",
            },
            plan,
        )
