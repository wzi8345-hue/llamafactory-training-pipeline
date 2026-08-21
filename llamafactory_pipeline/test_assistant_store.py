from datetime import datetime, timedelta, timezone
import sqlite3

import pytest

from .assistant_schema import (
    ActionKind,
    BaselineSpec,
    DataSourceSpec,
    RequirementDraft,
    RequirementField,
    SuccessCriteria,
    TrainingObjective,
    WorkflowState,
)
from .assistant_state import InvalidTransition, plan_hash
from .assistant_store import AssistantStore


def _objective() -> TrainingObjective:
    return TrainingObjective(
        goal="客服函数调用参数完整",
        task_types=["fc"],
        base_model_path="/models/qwen",
        baseline=BaselineSpec(kind="base_model", name="base"),
        data_source=DataSourceSpec(fc_seed_file="seed.json"),
        success_criteria=SuccessCriteria(primary_metric="param_score_mean"),
    )


def _requirement_draft() -> RequirementDraft:
    def field(value: str) -> RequirementField:
        return RequirementField(
            value=value,
            source="user",
            evidence_message_ids=[1],
        )

    return RequirementDraft(
        assistant_reply="需求理解已整理，请确认。",
        ready_for_review=True,
        missing_fields=[],
        scenario=field("客服函数调用"),
        current_problem=field("复杂参数缺失"),
        desired_behavior=field("必填参数完整"),
        proposed_objective=_objective(),
        assumptions=["默认参数分为主指标"],
    )


def test_store_migrates_requirement_and_cancel_columns(tmp_path):
    path = tmp_path / "assistant.sqlite"
    conn = sqlite3.connect(path)
    conn.execute(
        """CREATE TABLE workflows (
             workflow_id TEXT PRIMARY KEY,state TEXT NOT NULL,
             iteration INTEGER NOT NULL DEFAULT 0,objective_json TEXT,
             data_plan_json TEXT,dataset_profile_json TEXT,
             training_plan_json TEXT,preflight_json TEXT,
             evaluation_plan_json TEXT,diagnosis_json TEXT,
             datagen_jobs_json TEXT NOT NULL DEFAULT '[]',
             train_job_id TEXT,eval_id TEXT,created_at TEXT NOT NULL,
             updated_at TEXT NOT NULL
           )"""
    )
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        """INSERT INTO workflows(workflow_id,state,created_at,updated_at)
           VALUES('wf_legacy_1','collecting_requirements',?,?)""",
        (now, now),
    )
    conn.commit()
    conn.close()

    workflow = AssistantStore(path).get_workflow("wf_legacy_1")

    assert workflow["requirement_draft"] is None
    assert workflow["confirmed_objective"] is None
    assert workflow["objective_hash"] is None
    assert workflow["cancel_request"] is None


def test_publish_requirement_review_is_atomic_and_only_confirms_requirements(tmp_path):
    store = AssistantStore(tmp_path / "assistant.sqlite")
    workflow_id = store.create_workflow()
    store.append_message(workflow_id, "user", "客服函数调用参数经常缺失")

    store.publish_requirement_review(workflow_id, _requirement_draft())

    workflow = store.get_workflow(workflow_id)
    approvals = store.list_pending_approvals(workflow_id)
    assert workflow["state"] == WorkflowState.REQUIREMENTS_REVIEW.value
    assert workflow["requirement_draft"]["scenario"]["source"] == "user"
    assert workflow["data_plan"] is None
    assert [row["action"] for row in approvals] == [
        ActionKind.CONFIRM_REQUIREMENTS.value
    ]


def test_confirm_requirements_atomically_schedules_recoverable_data_planning(tmp_path):
    store = AssistantStore(tmp_path / "assistant.sqlite")
    workflow_id = store.create_workflow()
    store.append_message(workflow_id, "user", "客服函数调用参数经常缺失")
    store.publish_requirement_review(workflow_id, _requirement_draft())
    listed = store.list_pending_approvals(workflow_id)[0]
    approval = store.get_approval(workflow_id, listed["approval_id"])
    execution = store.prepare_external_execution(
        workflow_id,
        approval["approval_id"],
        approval["plan_hash"],
        approval["payload"],
        {},
    )

    assert store.confirm_requirements_and_schedule_plan(
        workflow_id=workflow_id,
        objective=_objective(),
        objective_hash=plan_hash(_objective()),
        approval_id=approval["approval_id"],
        action_id=execution["action_id"],
        lease_token=execution["lease_token"],
    )

    workflow = store.get_workflow(workflow_id)
    assert workflow["state"] == WorkflowState.DATA_PLAN_PREPARING.value
    assert workflow["confirmed_objective"]["goal"] == _objective().goal
    assert workflow["objective_hash"] == plan_hash(_objective())
    assert store.get_approval(workflow_id, approval["approval_id"])[
        "status"
    ] == "consumed"
    assert store.count_pending_actions("prepare_data_plan") == 1


def test_request_cancellation_is_idempotent_and_fences_existing_monitor(tmp_path):
    store = AssistantStore(tmp_path / "assistant.sqlite")
    workflow_id = store.create_workflow()
    store.transition(workflow_id, "requirements_completed", {})
    store.update_workflow_fields(
        workflow_id,
        datagen_jobs_json=[{"job_id": "job_1", "task_type": "fc"}],
    )
    store.transition(workflow_id, "datagen_started", {"job_id": "job_1"})
    now = datetime.now(timezone.utc)
    store.schedule_action(
        workflow_id,
        "monitor_datagen",
        now,
        {"launches": [{"job_id": "job_1", "task_type": "fc"}]},
        f"{workflow_id}:datagen:0",
    )
    leased = store.lease_due_actions(now, limit=1)[0]

    first = store.request_cancellation(workflow_id, "用户手动中止")
    second = store.request_cancellation(workflow_id, "重复点击")

    assert first["cancel_request_id"] == second["cancel_request_id"]
    assert store.get_workflow(workflow_id)["state"] == "cancelling"
    assert store.count_pending_actions("cancel_external_job") == 1
    assert store.list_pending_approvals(workflow_id) == []
    assert not store.reschedule_action(
        leased["action_id"],
        leased["lease_token"],
        now + timedelta(seconds=60),
        leased["payload"],
    )


def test_cancellation_without_external_job_completes_immediately(tmp_path):
    store = AssistantStore(tmp_path / "assistant.sqlite")
    workflow_id = store.create_workflow()

    request = store.request_cancellation(workflow_id, "不再训练")

    assert request["targets"] == []
    assert store.get_workflow(workflow_id)["state"] == "cancelled"


def test_workflow_transition_and_event_are_atomic(tmp_path):
    store = AssistantStore(tmp_path / "assistant.sqlite")
    workflow_id = store.create_workflow()
    store.transition(workflow_id, "requirements_completed", {"source": "test"})
    row = store.get_workflow(workflow_id)
    assert row["state"] == WorkflowState.DATA_PLAN_READY.value
    assert store.list_events(workflow_id, 0)[-1]["event_type"] == "requirements_completed"

    with pytest.raises(InvalidTransition):
        store.transition(workflow_id, "training_started", {"should": "rollback"})
    assert store.get_workflow(workflow_id)["state"] == WorkflowState.DATA_PLAN_READY.value
    assert [event["event_type"] for event in store.list_events(workflow_id, 0)] == [
        "workflow_created",
        "requirements_completed",
    ]


def test_messages_and_allowed_workflow_fields_round_trip(tmp_path):
    store = AssistantStore(tmp_path / "assistant.sqlite")
    workflow_id = store.create_workflow()
    store.append_message(workflow_id, "user", "先训练 FC")
    store.append_message(workflow_id, "assistant", "请提供基座模型")
    store.update_workflow_fields(
        workflow_id,
        objective_json='{"goal":"fc"}',
        datagen_jobs_json='[{"job_id":"j1"}]',
    )
    row = store.get_workflow(workflow_id)
    assert row["objective"] == {"goal": "fc"}
    assert row["datagen_jobs"] == [{"job_id": "j1"}]
    assert [message["role"] for message in store.list_messages(workflow_id)] == [
        "user",
        "assistant",
    ]
    with pytest.raises(ValueError, match="unsupported workflow field"):
        store.update_workflow_fields(workflow_id, state="completed")


def test_approval_can_only_be_claimed_once(tmp_path):
    store = AssistantStore(tmp_path / "assistant.sqlite")
    workflow_id = store.create_workflow()
    approval = store.create_approval(
        workflow_id, ActionKind.START_DATAGEN, {"count": 100}, "build data"
    )
    assert approval["plan_hash"].startswith("sha256:")
    assert store.claim_approval(workflow_id, approval["approval_id"], approval["plan_hash"])
    assert not store.claim_approval(workflow_id, approval["approval_id"], approval["plan_hash"])
    store.finish_approval(approval["approval_id"], succeeded=True)
    assert store.get_approval(workflow_id, approval["approval_id"])["status"] == "consumed"


def test_only_one_alternative_approval_can_execute(tmp_path):
    store = AssistantStore(tmp_path / "assistant.sqlite")
    workflow_id = store.create_workflow()
    start = store.create_approval(
        workflow_id, ActionKind.START_EVALUATION, {"eval": "paired"}, "run A/B"
    )
    skip = store.create_approval(
        workflow_id, ActionKind.SKIP_EVALUATION, {"eval": "paired"}, "skip A/B"
    )
    assert store.claim_approval(workflow_id, start["approval_id"], start["plan_hash"])
    assert not store.claim_approval(workflow_id, skip["approval_id"], skip["plan_hash"])
    store.finish_approval(start["approval_id"], succeeded=True)
    store.stale_other_pending_approvals(workflow_id, start["approval_id"])
    assert store.list_pending_approvals(workflow_id) == []
    assert store.get_approval(workflow_id, skip["approval_id"])["status"] == "stale"


def test_external_execution_outbox_survives_store_restart(tmp_path):
    path = tmp_path / "assistant.sqlite"
    store = AssistantStore(path)
    workflow_id = store.create_workflow()
    approval = store.create_approval(
        workflow_id, ActionKind.START_TRAINING, {"epochs": 3}, "train"
    )
    action = store.prepare_external_execution(
        workflow_id,
        approval["approval_id"],
        approval["plan_hash"],
        approval["payload"],
        {"job_id": "20260819T010203Z-a1b2c3"},
    )
    assert action["approval_id"] == approval["approval_id"]

    reopened = AssistantStore(path)
    assert reopened.get_approval(
        workflow_id, approval["approval_id"]
    )["status"] == "executing"
    assert action["status"] == "leased"
    assert reopened.lease_due_actions(
        datetime.now(timezone.utc) + timedelta(seconds=1), limit=1
    ) == []
    leased = reopened.lease_due_actions(
        datetime.fromisoformat(action["lease_until"]) + timedelta(seconds=1),
        limit=1,
    )
    assert leased[0]["payload"]["external_refs"]["job_id"] == (
        "20260819T010203Z-a1b2c3"
    )


def test_orphaned_legacy_execution_is_marked_failed_on_restart(tmp_path):
    path = tmp_path / "assistant.sqlite"
    store = AssistantStore(path)
    workflow_id = store.create_workflow()
    approval = store.create_approval(
        workflow_id, ActionKind.SKIP_EVALUATION, {"eval": "paired"}, "skip"
    )
    assert store.claim_approval(
        workflow_id, approval["approval_id"], approval["plan_hash"]
    )

    reopened = AssistantStore(path)
    recovered = reopened.get_approval(workflow_id, approval["approval_id"])
    assert recovered["status"] == "failed"
    assert recovered["error"] == "orphaned execution intent"


def test_new_approval_stales_same_action_and_reject_is_scoped(tmp_path):
    store = AssistantStore(tmp_path / "assistant.sqlite")
    workflow_id = store.create_workflow()
    old = store.create_approval(
        workflow_id, ActionKind.START_TRAINING, {"epochs": 3}, "old"
    )
    new = store.create_approval(
        workflow_id,
        ActionKind.START_TRAINING,
        {"epochs": 2},
        "new",
        decision_warnings=["warm", "warm"],
    )
    assert store.get_approval(workflow_id, old["approval_id"])["status"] == "stale"
    assert store.get_approval(workflow_id, new["approval_id"])["payload"]["decision_warnings"] == ["warm"]
    listed = store.list_pending_approvals(workflow_id)[0]
    assert listed["decision_warnings"] == ["warm"]
    assert "payload" not in listed and "plan_json" not in listed
    assert store.reject_approval(workflow_id, new["approval_id"])
    assert not store.reject_approval(workflow_id, new["approval_id"])


def test_expired_lease_can_be_reclaimed(tmp_path):
    store = AssistantStore(tmp_path / "assistant.sqlite")
    workflow_id = store.create_workflow()
    now = datetime.now(timezone.utc)
    store.schedule_action(workflow_id, "monitor_training", now, {"job_id": "j1"}, "wf:j1:0")
    store.schedule_action(workflow_id, "monitor_training", now, {"job_id": "j1"}, "wf:j1:0")
    first = store.lease_due_actions(now, limit=20, lease_seconds=1)
    assert len(first) == 1
    second = store.lease_due_actions(now + timedelta(seconds=2), limit=20, lease_seconds=120)
    assert len(second) == 1 and second[0]["action_id"] == first[0]["action_id"]
    assert second[0]["lease_token"] != first[0]["lease_token"]
    assert not store.complete_action(
        first[0]["action_id"], first[0]["lease_token"]
    )
    assert store.complete_action(
        second[0]["action_id"], second[0]["lease_token"]
    )


def test_active_worker_can_renew_action_lease(tmp_path):
    store = AssistantStore(tmp_path / "assistant.sqlite")
    workflow_id = store.create_workflow()
    now = datetime.now(timezone.utc)
    store.schedule_action(workflow_id, "monitor_evaluation", now, {}, "wf:renew")
    leased = store.lease_due_actions(now, limit=1, lease_seconds=1)[0]

    assert store.renew_action_lease(
        leased["action_id"], leased["lease_token"], lease_seconds=120
    )
    assert store.lease_due_actions(
        now + timedelta(seconds=2), limit=1, lease_seconds=1
    ) == []


def test_scheduled_action_retry_and_completion(tmp_path):
    store = AssistantStore(tmp_path / "assistant.sqlite")
    workflow_id = store.create_workflow()
    now = datetime.now(timezone.utc)
    action_id = store.schedule_action(
        workflow_id, "monitor_training", now, {"job_id": "j1"}, "wf:j1:retry"
    )
    leased = store.lease_due_actions(now, limit=1)[0]
    retry_at = now + timedelta(minutes=2)
    store.retry_action(
        leased["action_id"], leased["lease_token"], retry_at, "ssh unavailable"
    )
    assert store.count_pending_actions("monitor_training") == 1
    leased_again = store.lease_due_actions(retry_at, limit=1)[0]
    assert leased_again["action_id"] == action_id
    store.complete_action(action_id, leased_again["lease_token"])
    assert store.count_pending_actions("monitor_training") == 0


def test_training_history_returns_only_compatible_successes(tmp_path):
    store = AssistantStore(tmp_path / "assistant.sqlite")
    workflow_id = store.create_workflow()
    common = {
        "workflow_id": workflow_id,
        "iteration": 0,
        "stage": "sft",
        "model_parameter_billions": 9.0,
        "gpu_names": ["RTX 4090"],
        "gpu_count": 1,
        "cutoff_len": 2048,
        "quantization_bit": None,
        "estimated_steps": 100,
        "actual_steps": 100,
        "initial_eta_seconds": 1000,
        "calibrated_eta_seconds": 800,
        "duration_seconds": 500,
        "steps_per_second": 0.2,
    }
    store.record_training_run(train_job_id="good", terminal_status="succeeded", **common)
    store.record_training_run(train_job_id="failed", terminal_status="failed", **common)
    store.record_training_run(
        train_job_id="other-gpu",
        terminal_status="succeeded",
        **{**common, "gpu_names": ["A100"]},
    )
    rows = store.list_compatible_training_runs(
        stage="sft",
        model_parameter_billions=10.0,
        gpu_names=["RTX 4090"],
        gpu_count=1,
        cutoff_len=2048,
        quantization_bit=None,
    )
    assert [row["train_job_id"] for row in rows] == ["good"]
    assert rows[0]["gpu_names"] == ["RTX 4090"]


def test_invalid_lookup_id_is_rejected_before_query(tmp_path):
    store = AssistantStore(tmp_path / "assistant.sqlite")
    with pytest.raises(ValueError, match="invalid workflow_id"):
        store.get_workflow("../assistant.sqlite")
