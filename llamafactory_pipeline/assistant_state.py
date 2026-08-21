"""Canonical approval hashes and explicit workflow transitions."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from pydantic import BaseModel

from .assistant_schema import WorkflowState


class InvalidTransition(ValueError):
    pass


def canonical_json(value: Any) -> str:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def plan_hash(value: Any) -> str:
    digest = hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


TRANSITIONS = {
    (WorkflowState.COLLECTING_REQUIREMENTS, "requirements_draft_ready"): WorkflowState.REQUIREMENTS_REVIEW,
    (WorkflowState.REQUIREMENTS_REVIEW, "requirements_revision_requested"): WorkflowState.COLLECTING_REQUIREMENTS,
    (WorkflowState.REQUIREMENTS_REVIEW, "requirements_confirmed"): WorkflowState.DATA_PLAN_PREPARING,
    (WorkflowState.DATA_PLAN_PREPARING, "data_plan_created"): WorkflowState.DATA_PLAN_READY,
    (WorkflowState.DATA_PLAN_PREPARING, "data_plan_preparation_failed"): WorkflowState.REQUIREMENTS_REVIEW,
    (WorkflowState.COLLECTING_REQUIREMENTS, "requirements_completed"): WorkflowState.DATA_PLAN_READY,
    (WorkflowState.DATA_PLAN_READY, "data_plan_revision_requested"): WorkflowState.COLLECTING_REQUIREMENTS,
    (WorkflowState.DATA_PLAN_READY, "datagen_started"): WorkflowState.DATA_GENERATING,
    (WorkflowState.DATA_GENERATING, "datagen_completed"): WorkflowState.DATA_REVIEW,
    (WorkflowState.DATA_GENERATING, "datagen_failed"): WorkflowState.DATA_PLAN_READY,
    (WorkflowState.DATA_REVIEW, "train_plan_created"): WorkflowState.TRAIN_PLAN_READY,
    (WorkflowState.TRAIN_PLAN_READY, "preflight_passed"): WorkflowState.TRAIN_READY,
    (WorkflowState.TRAIN_PLAN_READY, "preflight_blocked"): WorkflowState.PREFLIGHT_BLOCKED,
    (WorkflowState.PREFLIGHT_BLOCKED, "plan_revised"): WorkflowState.TRAIN_PLAN_READY,
    (WorkflowState.PREFLIGHT_BLOCKED, "preflight_passed"): WorkflowState.TRAIN_READY,
    (WorkflowState.TRAIN_READY, "preflight_blocked"): WorkflowState.PREFLIGHT_BLOCKED,
    (WorkflowState.TRAIN_READY, "training_started"): WorkflowState.TRAINING,
    (WorkflowState.TRAINING, "training_succeeded"): WorkflowState.AB_PLAN_READY,
    (WorkflowState.TRAINING, "training_failed"): WorkflowState.TRAIN_FAILED,
    (WorkflowState.TRAIN_FAILED, "recovery_plan_created"): WorkflowState.TRAIN_PLAN_READY,
    (WorkflowState.AB_PLAN_READY, "evaluation_started"): WorkflowState.EVALUATING,
    (WorkflowState.AB_PLAN_READY, "evaluation_scoring_retried"): WorkflowState.EVALUATING,
    (WorkflowState.AB_PLAN_READY, "evaluation_skipped"): WorkflowState.COMPLETED,
    (WorkflowState.EVALUATING, "evaluation_completed"): WorkflowState.DIAGNOSIS_READY,
    (WorkflowState.EVALUATING, "evaluation_failed"): WorkflowState.AB_PLAN_READY,
    (WorkflowState.DIAGNOSIS_READY, "evaluation_retried"): WorkflowState.EVALUATING,
    (WorkflowState.DIAGNOSIS_READY, "candidate_accepted"): WorkflowState.COMPLETED,
    (WorkflowState.DIAGNOSIS_READY, "candidate_rejected"): WorkflowState.COMPLETED,
    (WorkflowState.DIAGNOSIS_READY, "iteration_started"): WorkflowState.DATA_PLAN_READY,
    (WorkflowState.CANCELLING, "cancellation_retry_scheduled"): WorkflowState.CANCELLING,
    (WorkflowState.CANCELLING, "cancellation_completed"): WorkflowState.CANCELLED,
}

for _state in WorkflowState:
    if _state not in {
        WorkflowState.CANCELLING,
        WorkflowState.CANCELLED,
        WorkflowState.COMPLETED,
    }:
        TRANSITIONS[(_state, "cancellation_requested")] = WorkflowState.CANCELLING


def next_state(current: WorkflowState, event: str) -> WorkflowState:
    try:
        return TRANSITIONS[(current, event)]
    except KeyError as exc:
        raise InvalidTransition(f"{current.value} cannot handle {event}") from exc
