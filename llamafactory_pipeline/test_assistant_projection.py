from .assistant_projection import build_workflow_steps


def _workflow(state: str, **updates):
    base = {
        "workflow_id": "wf_20260821T010203Z_abcdef",
        "state": state,
        "iteration": 0,
        "created_at": "2026-08-21T01:02:03+00:00",
        "updated_at": "2026-08-21T01:12:03+00:00",
        "requirement_draft": None,
        "confirmed_objective": None,
        "objective_hash": None,
        "objective": None,
        "data_plan": None,
        "dataset_profile": None,
        "training_plan": None,
        "preflight": None,
        "evaluation_plan": None,
        "diagnosis": None,
        "datagen_jobs": [],
        "train_job_id": None,
        "eval_id": None,
        "cancel_request": None,
    }
    return {**base, **updates}


def test_projection_always_returns_eight_ordered_steps():
    steps = build_workflow_steps(_workflow("collecting_requirements"), [], [])

    assert len(steps) == 8
    assert [step.sequence for step in steps] == list(range(1, 9))
    assert [step.key for step in steps] == [
        "requirements",
        "data_plan",
        "data_build",
        "data_review",
        "train_plan",
        "training",
        "evaluation",
        "diagnosis",
    ]
    assert steps[0].status == "active"
    assert steps[1].status == "pending"


def test_requirement_review_exposes_confirmation_and_revision_actions():
    steps = build_workflow_steps(
        _workflow(
            "requirements_review",
            requirement_draft={"ready_for_review": True},
        ),
        [],
        [{"action": "confirm_requirements"}],
    )

    assert steps[0].status == "needs_confirmation"
    assert steps[0].actions == [
        "confirm_requirements",
        "revise_requirements",
        "cancel",
    ]


def test_failed_data_plan_review_exposes_retry_instead_of_reconfirmation():
    steps = build_workflow_steps(
        _workflow(
            "requirements_review",
            confirmed_objective={"goal": "confirmed"},
            objective_hash="sha256:" + "a" * 64,
        ),
        [{"event_type": "data_plan_preparation_failed", "payload": {}}],
        [],
    )

    assert steps[0].actions == [
        "retry_data_plan",
        "revise_requirements",
        "cancel",
    ]


def test_data_generation_projection_exposes_progress_and_artifacts():
    events = [
        {
            "event_id": 9,
            "event_type": "datagen_progress",
            "created_at": "2026-08-21T03:00:00+00:00",
            "payload": {
                "accepted": 188,
                "target": 1000,
                "attempts": 260,
                "acceptance_rate": 0.723,
                "rejects": {"judge": 17},
                "eta_seconds": 1260,
            },
        }
    ]
    workflow = _workflow(
        "data_generating",
        objective_hash="sha256:" + "a" * 64,
        datagen_jobs=[{"job_id": "job_1", "task_type": "fc"}],
    )

    steps = build_workflow_steps(workflow, events, [])
    data = steps[2]

    assert data.status == "active"
    assert "1 个数据任务" in data.summary
    assert data.progress.current == 188
    assert data.progress.target == 1000
    assert data.progress.percentage == 18.8
    assert data.progress.eta_seconds == 1260
    assert data.progress.details["rejects"] == {"judge": 17}
    assert {artifact.kind for artifact in data.artifacts} == {
        "datagen_output",
        "datagen_report",
        "datagen_log",
    }
    assert all(
        not (artifact.download_url or "").startswith("/Users/")
        for artifact in data.artifacts
    )


def test_legacy_workflow_is_visible_with_an_explicit_requirement_warning():
    steps = build_workflow_steps(
        _workflow(
            "training",
            objective={"goal": "legacy FC"},
            train_job_id="train_1",
        ),
        [],
        [],
    )

    assert steps[0].status == "succeeded"
    assert steps[0].issues[0]["code"] == "legacy_requirement_gate_missing"
    assert steps[5].status == "active"


def test_cancelled_workflow_keeps_existing_job_artifacts():
    steps = build_workflow_steps(
        _workflow(
            "cancelled",
            train_job_id="train_1",
            cancel_request={
                "cancel_request_id": "cancel_1",
                "targets": [{"kind": "training", "job_id": "train_1"}],
            },
        ),
        [{"event_type": "cancellation_completed", "payload": {}}],
        [],
    )

    training = steps[5]
    assert training.status == "cancelled"
    assert "train_1" in training.summary
    assert {artifact.kind for artifact in training.artifacts} >= {
        "training_log",
        "training_metrics",
        "checkpoint_index",
    }
    assert training.actions == []
