"""One-shot leased monitoring worker and reducer tests."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import Mock

import pytest

from .assistant_store import AssistantStore
from .remote import RemoteError
from .assistant_worker import (
    MonitorMemory,
    classify_training_failure,
    reduce_datagen_observation,
    reduce_training_observation,
    run_once,
)


def running_observation(step, total):
    return {
        "status": {"status": "running"},
        "metrics": {
            "points": [
                {"step": step, "loss": 1.2, "epoch": 1.0, "lr": 1e-4}
            ],
            "total_steps": total,
            "percentage": step * 100 / total,
            "remaining_time": "00:30:00",
        },
        "gpus": [
            {
                "util": 90,
                "mem_used": 20000,
                "mem_total": 24576,
                "temperature": 70,
            }
        ],
    }


def same_observation():
    return running_observation(25, 100)


def seeded_training_store(tmp_path):
    store = AssistantStore(tmp_path / "assistant.sqlite")
    workflow_id = store.create_workflow()
    for event in (
        "requirements_completed",
        "datagen_started",
        "datagen_completed",
        "train_plan_created",
        "preflight_passed",
        "training_started",
    ):
        store.transition(workflow_id, event, {})
    store.update_workflow_fields(
        workflow_id, train_job_id="20260819T010203Z-a1b2c3"
    )
    now = datetime.now(timezone.utc)
    store.schedule_action(
        workflow_id,
        "monitor_training",
        now,
        {"job_id": "20260819T010203Z-a1b2c3", "memory": {}},
        f"{workflow_id}:monitor_training:20260819T010203Z-a1b2c3:{int(now.timestamp())}",
    )
    return store


class FakeTools:
    training_observation = None

    def inspect_training(self, job_id):
        return self.training_observation


def _cancelling_training_store(tmp_path):
    store = AssistantStore(tmp_path / "assistant.sqlite")
    workflow_id = store.create_workflow()
    for event in (
        "requirements_completed",
        "datagen_started",
        "datagen_completed",
        "train_plan_created",
        "preflight_passed",
        "training_started",
    ):
        store.transition(workflow_id, event, {})
    store.update_workflow_fields(workflow_id, train_job_id="train_1")
    request = store.request_cancellation(workflow_id, "用户手动中止")
    return store, workflow_id, request


def test_cancel_worker_retries_transport_error_without_marking_cancelled(tmp_path):
    store, workflow_id, _ = _cancelling_training_store(tmp_path)
    tools = Mock()
    tools.stop_external_job.side_effect = RemoteError("ssh timeout")
    service = Mock()

    result = run_once(
        store, tools, service, now=datetime.now(timezone.utc), limit=1
    )

    assert result.failed == 1
    assert store.get_workflow(workflow_id)["state"] == "cancelling"
    assert store.count_pending_actions("cancel_external_job") == 1
    service.on_cancellation_observation.assert_not_called()


def test_cancel_worker_records_attention_after_five_failed_stop_attempts(tmp_path):
    store, workflow_id, request = _cancelling_training_store(tmp_path)
    tools = Mock()
    tools.stop_external_job.side_effect = RemoteError("ssh timeout")
    service = Mock()
    now = datetime.now(timezone.utc)

    for _ in range(5):
        run_once(store, tools, service, now=now, limit=1)
        action = store.get_action_by_key(
            f"{workflow_id}:cancel:{request['cancel_request_id']}:"
            "training:train_1"
        )
        now = datetime.fromisoformat(action["due_at"]) + timedelta(seconds=1)

    attention = [
        event
        for event in store.list_events(workflow_id, 0, limit=100)
        if event["event_type"] == "cancellation_needs_attention"
    ]
    assert len(attention) == 1


def test_datagen_reducer_emits_changed_progress_without_poll_spam():
    observation = {
        "status": "running",
        "accepted": 188,
        "target": 1000,
        "attempts": 260,
        "rejects": {"judge": 17},
    }

    event, memory = reduce_datagen_observation(
        {}, [observation], now="2026-08-21T03:00:00+00:00"
    )

    assert event == {
        "event_type": "datagen_progress",
        "payload": {
            "accepted": 188,
            "target": 1000,
            "attempts": 260,
            "acceptance_rate": pytest.approx(188 / 260),
            "rejects": {"judge": 17},
            "eta_seconds": None,
        },
    }
    duplicate, _ = reduce_datagen_observation(
        memory, [observation], now="2026-08-21T03:01:00+00:00"
    )
    assert duplicate is None


def test_datagen_reducer_estimates_eta_from_accepted_record_rate():
    memory = {
        "last_accepted": 100,
        "last_observed_at": "2026-08-21T03:00:00+00:00",
        "last_emitted_at": "2026-08-21T03:00:00+00:00",
        "fingerprint": "old",
        "rates": [],
    }

    event, _ = reduce_datagen_observation(
        memory,
        [{"status": "running", "accepted": 130, "target": 190, "attempts": 150}],
        now="2026-08-21T03:01:00+00:00",
    )

    assert event["payload"]["eta_seconds"] == 120


def test_terminal_training_waits_for_verified_checkpoint_and_log_evidence(tmp_path):
    store = seeded_training_store(tmp_path)
    tools = FakeTools()
    tools.training_observation = {
        "status": {"status": "failed"},
        "metrics": {"points": []},
        "gpus": [],
        "checkpoints": [],
        "checkpoints_verified": False,
        "log_tail": "",
        "log_verified": False,
    }
    service = Mock()

    result = run_once(
        store, tools, service, now=datetime.now(timezone.utc), limit=1
    )

    assert result.failed == 1
    service.on_training_observation.assert_not_called()
    assert store.count_pending_actions("monitor_training") == 1


def test_successful_training_waits_for_verified_output_evidence(tmp_path):
    store = seeded_training_store(tmp_path)
    tools = FakeTools()
    tools.training_observation = {
        "status": {"status": "succeeded"},
        "metrics": {"points": []},
        "gpus": [],
        "checkpoints": [],
        "checkpoints_verified": False,
        "log_tail": "",
        "log_verified": False,
        "output_evidence_verified": False,
        "output_verified": False,
    }
    service = Mock()

    result = run_once(
        store, tools, service, now=datetime.now(timezone.utc), limit=1
    )

    assert result.failed == 1
    service.on_training_observation.assert_not_called()


def test_successful_training_advances_with_verified_adapter_output(tmp_path):
    store = seeded_training_store(tmp_path)
    tools = FakeTools()
    tools.training_observation = {
        "status": {"status": "succeeded"},
        "metrics": {"points": []},
        "gpus": [],
        "checkpoints": [],
        "checkpoints_verified": False,
        "log_tail": "",
        "log_verified": False,
        "output_evidence_verified": True,
        "output_verified": True,
    }
    service = Mock()

    result = run_once(
        store, tools, service, now=datetime.now(timezone.utc), limit=1
    )

    assert result.failed == 0
    service.on_training_observation.assert_called_once()
    assert store.count_pending_actions("monitor_training") == 0


@pytest.mark.parametrize(
    ("action", "payload", "tool_name", "observation", "service_name"),
    [
        (
            "monitor_datagen",
            {"launches": [{"job_id": "dg1", "task_type": "fc"}]},
            "inspect_datagen",
            [{"status": "not_found", "job_id": "dg1"}],
            "on_datagen_terminal",
        ),
        (
            "monitor_training",
            {"job_id": "train1", "memory": {}},
            "inspect_training",
            {"status": "unknown"},
            "on_training_observation",
        ),
        (
            "monitor_evaluation",
            {"eval_id": "eval1"},
            "inspect_evaluation",
            {"status": "not_found"},
            "on_evaluation_terminal",
        ),
    ],
)
def test_unknown_remote_status_keeps_monitor_and_escalates_after_five_polls(
    tmp_path, action, payload, tool_name, observation, service_name
):
    store = AssistantStore(tmp_path / f"{action}.sqlite")
    workflow_id = store.create_workflow()
    now = datetime.now(timezone.utc)
    key = f"{workflow_id}:{action}"
    store.schedule_action(workflow_id, action, now, payload, key)
    tools = Mock()
    getattr(tools, tool_name).return_value = observation
    service = Mock()

    for _ in range(5):
        last_poll = now
        summary = run_once(store, tools, service, now=now, limit=1)
        assert summary.failed == 0
        scheduled = store.get_action_by_key(key)
        assert scheduled["status"] == "pending"
        now = datetime.fromisoformat(scheduled["due_at"]) + timedelta(seconds=1)

    assert scheduled["payload"]["unknown_polls"] == 5
    getattr(service, service_name).assert_not_called()
    attention = [
        event
        for event in store.list_events(workflow_id, 0, limit=100)
        if event["event_type"] == "monitor_needs_attention"
    ]
    assert len(attention) == 1
    assert datetime.fromisoformat(scheduled["due_at"]) >= last_poll + timedelta(
        seconds=899
    )


def test_repeated_scoring_error_exits_evaluating_with_recovery_actions(tmp_path):
    store = AssistantStore(tmp_path / "assistant.sqlite")
    workflow_id = store.create_workflow()
    now = datetime.now(timezone.utc)
    key = f"{workflow_id}:monitor_evaluation"
    store.schedule_action(
        workflow_id,
        "monitor_evaluation",
        now,
        {"eval_id": "eval1"},
        key,
    )
    tools = Mock()
    tools.inspect_evaluation.return_value = {"status": "succeeded"}
    service = Mock()
    service.on_evaluation_terminal.side_effect = RuntimeError(
        "judge authentication failed"
    )

    for _ in range(3):
        summary = run_once(store, tools, service, now=now, limit=1)
        scheduled = store.get_action_by_key(key)
        if scheduled["status"] == "pending":
            now = datetime.fromisoformat(scheduled["due_at"]) + timedelta(seconds=1)

    assert summary.failed == 1
    service.on_evaluation_processing_failed.assert_called_once()
    assert store.get_action_by_key(key)["status"] == "done"


def test_successful_remote_evaluation_persists_scoring_progress(tmp_path):
    store = AssistantStore(tmp_path / "assistant.sqlite")
    workflow_id = store.create_workflow()
    now = datetime.now(timezone.utc)
    store.schedule_action(
        workflow_id,
        "monitor_evaluation",
        now,
        {"eval_id": "eval1"},
        f"{workflow_id}:monitor_evaluation",
    )
    tools = Mock()
    tools.inspect_evaluation.return_value = {"status": "succeeded"}
    service = Mock()

    result = run_once(store, tools, service, now=now, limit=1)

    assert result.failed == 0
    progress = [
        event
        for event in store.list_events(workflow_id, 0, limit=100)
        if event["event_type"] == "evaluation_progress"
    ]
    assert progress[-1]["payload"] == {
        "phase": "scoring",
        "percentage": 75.0,
        "eval_id": "eval1",
    }


def test_reducer_emits_each_progress_milestone_once():
    memory = MonitorMemory(
        reported_milestones=[],
        last_step=20,
        last_step_at="2026-08-19T00:00:00+00:00",
    )
    events, updated = reduce_training_observation(
        memory,
        running_observation(25, 100),
        now="2026-08-19T00:01:00+00:00",
    )
    assert [event["event_type"] for event in events] == ["training_progress"]
    assert events[0]["payload"]["loss"] == 1.2
    assert events[0]["payload"]["epoch"] == 1.0
    assert events[0]["payload"]["learning_rate"] == 1e-4
    assert events[0]["payload"]["eta_seconds"] == 1800
    assert updated.reported_milestones == [25]
    again, _ = reduce_training_observation(
        updated, same_observation(), now="2026-08-19T00:02:00+00:00"
    )
    assert again == []


def test_reducer_calibrates_eta_from_recent_step_rate():
    memory = MonitorMemory(
        last_step=20,
        last_step_at="2026-08-19T00:00:00+00:00",
        recent_step_rates=[0.5, 1.5],
    )
    _, updated = reduce_training_observation(
        memory,
        running_observation(80, 100),
        now="2026-08-19T00:01:00+00:00",
    )
    # New rate is 1 step/s; median(0.5, 1.0, 1.5)=1.0.
    assert updated.calibrated_eta_seconds == 20
    assert updated.recent_step_rates[-1] == 1.0


def test_reducer_collapses_large_jump_and_detects_gpu_pressure():
    memory = MonitorMemory(
        reported_milestones=[10],
        last_step=10,
        last_step_at="2026-08-19T00:00:00+00:00",
    )
    observation = running_observation(92, 100)
    observation["gpus"][0].update(mem_used=24000, temperature=85)
    events, updated = reduce_training_observation(
        memory, observation, now="2026-08-19T00:01:00+00:00"
    )
    assert [event["event_type"] for event in events] == [
        "training_progress",
        "training_gpu_pressure",
    ]
    assert updated.reported_milestones == [10, 25, 50, 75, 90]
    assert events[0]["payload"]["milestone"] == 90


def test_reducer_detects_stall_and_invalid_loss():
    memory = MonitorMemory(
        reported_milestones=[],
        last_step=10,
        last_step_at="2026-08-19T00:00:00+00:00",
    )
    observation = running_observation(10, 100)
    observation["metrics"]["points"][-1]["loss"] = float("nan")
    observation["gpus"][0]["util"] = 2
    events, updated = reduce_training_observation(
        memory, observation, now="2026-08-19T00:11:00+00:00"
    )
    assert {event["event_type"] for event in events} == {
        "training_stalled",
        "training_loss_invalid",
    }
    repeated, _ = reduce_training_observation(
        updated, observation, now="2026-08-19T00:12:00+00:00"
    )
    assert repeated == []


def test_failure_classifier_uses_stable_categories():
    assert classify_training_failure("CUDA out of memory") == "oom"
    assert classify_training_failure("NCCL watchdog timeout") == "distributed"
    assert classify_training_failure("Tokenizer class cannot be loaded") == "tokenizer"
    assert classify_training_failure("Traceback: bad dataset column") == "dataset"


def test_worker_reschedules_running_training(tmp_path):
    store = seeded_training_store(tmp_path)
    fake_tools = FakeTools()
    fake_service = Mock()
    fake_tools.training_observation = running_observation(step=10, total=100)
    now = datetime.now(timezone.utc)
    summary = run_once(
        store,
        fake_tools,
        fake_service,
        now=now,
        limit=20,
    )
    assert summary.processed == 1
    assert summary.failed == 0
    assert store.count_pending_actions("monitor_training") == 1
    next_lease = store.lease_due_actions(
        now + timedelta(seconds=61),
        limit=1,
    )[0]
    assert next_lease["attempts"] == 1


def test_worker_leases_each_action_only_when_it_is_ready_to_dispatch(tmp_path):
    store = AssistantStore(tmp_path / "assistant.sqlite")
    workflow_id = store.create_workflow()
    now = datetime.now(timezone.utc)
    store.schedule_action(
        workflow_id,
        "execute_approval",
        now,
        {},
        "first-action",
    )
    store.schedule_action(
        workflow_id,
        "execute_approval",
        now,
        {},
        "second-action",
    )
    service = Mock()
    statuses_seen = []

    def execute(_action):
        if not statuses_seen:
            statuses_seen.append(store.get_action_by_key("second-action")["status"])

    service.execute_approval_action.side_effect = execute

    summary = run_once(store, Mock(), service, now=now, limit=2)

    assert summary.processed == 2
    assert summary.failed == 0
    assert statuses_seen == ["pending"]


def test_retryable_external_reconciliation_is_never_abandoned_by_attempt_count(
    tmp_path,
):
    store = AssistantStore(tmp_path / "assistant.sqlite")
    workflow_id = store.create_workflow()
    now = datetime.now(timezone.utc)
    key = "approval-reconciliation"
    store.schedule_action(
        workflow_id,
        "execute_approval",
        now,
        {"external_refs": {"job_id": "fixed-job"}},
        key,
    )
    service = Mock()
    service.execute_approval_action.side_effect = RuntimeError(
        "remote started; local commit unavailable"
    )
    service.is_retryable_execution_error.return_value = True

    for _ in range(6):
        run_once(store, Mock(), service, now=now, limit=1)
        action = store.get_action_by_key(key)
        assert action["status"] == "pending"
        assert action["payload"]["external_refs"]["job_id"] == "fixed-job"
        now = datetime.fromisoformat(action["due_at"]) + timedelta(seconds=1)

    service.fail_execution_action.assert_not_called()


def test_reschedule_does_not_fork_monitor_chain_if_completion_crashes(tmp_path):
    store = seeded_training_store(tmp_path)
    fake_tools = FakeTools()
    fake_tools.training_observation = running_observation(step=10, total=100)
    store.complete_action = Mock(side_effect=RuntimeError("crash after reschedule"))

    summary = run_once(
        store,
        fake_tools,
        Mock(),
        now=datetime.now(timezone.utc),
        limit=20,
    )

    assert summary.failed == 1
    assert store.count_pending_actions("monitor_training") == 1


def test_worker_advances_succeeded_training(tmp_path):
    store = seeded_training_store(tmp_path)
    fake_tools = FakeTools()
    fake_service = Mock()
    fake_tools.training_observation = {
        "status": {"status": "succeeded"},
        "metrics": {},
        "gpus": [],
        "output_evidence_verified": True,
        "output_verified": True,
    }
    run_once(
        store,
        fake_tools,
        fake_service,
        now=datetime.now(timezone.utc),
        limit=20,
    )
    fake_service.on_training_observation.assert_called_once()
    assert store.count_pending_actions("monitor_training") == 0
