import pytest

from .assistant_schema import ApprovalPayload, ActionKind, WorkflowState
from .assistant_state import InvalidTransition, next_state, plan_hash


def test_plan_hash_is_key_order_independent():
    assert plan_hash({"b": 2, "a": 1}) == plan_hash({"a": 1, "b": 2})
    assert plan_hash({"a": 2}) != plan_hash({"a": 1})


def test_plan_hash_accepts_pydantic_contracts():
    payload = ApprovalPayload(
        action=ActionKind.START_TRAINING,
        plan={"epochs": 3},
        decision_warnings=["gpu: warm"],
    )
    assert plan_hash(payload).startswith("sha256:")
    assert len(plan_hash(payload)) == 71


def test_known_transition_reaches_data_plan_ready():
    assert next_state(
        WorkflowState.COLLECTING_REQUIREMENTS, "requirements_draft_ready"
    ) == WorkflowState.REQUIREMENTS_REVIEW


def test_requirement_review_precedes_data_plan_ready():
    assert next_state(
        WorkflowState.REQUIREMENTS_REVIEW, "requirements_confirmed"
    ) == WorkflowState.DATA_PLAN_PREPARING
    assert next_state(
        WorkflowState.DATA_PLAN_PREPARING, "data_plan_created"
    ) == WorkflowState.DATA_PLAN_READY


@pytest.mark.parametrize(
    "state",
    [
        state
        for state in WorkflowState
        if state not in {WorkflowState.COMPLETED, WorkflowState.CANCELLED}
        and state != WorkflowState.CANCELLING
    ],
)
def test_every_active_state_can_request_cancellation(state):
    assert next_state(state, "cancellation_requested") == WorkflowState.CANCELLING


def test_cancellation_has_an_explicit_terminal_transition():
    assert next_state(
        WorkflowState.CANCELLING, "cancellation_completed"
    ) == WorkflowState.CANCELLED


@pytest.mark.parametrize(
    "state", [WorkflowState.COMPLETED, WorkflowState.CANCELLED]
)
def test_terminal_states_reject_cancellation(state):
    with pytest.raises(InvalidTransition):
        next_state(state, "cancellation_requested")


def test_blocked_preflight_can_pass_after_resources_recover():
    assert next_state(
        WorkflowState.PREFLIGHT_BLOCKED, "preflight_passed"
    ) == WorkflowState.TRAIN_READY


def test_scoring_retry_reuses_the_existing_evaluation_run():
    assert next_state(
        WorkflowState.AB_PLAN_READY, "evaluation_scoring_retried"
    ) == WorkflowState.EVALUATING


def test_unknown_transition_is_rejected():
    with pytest.raises(InvalidTransition):
        next_state(WorkflowState.TRAINING, "start_datagen")
