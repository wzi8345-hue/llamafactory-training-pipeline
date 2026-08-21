"""Approval-gated orchestration for personal LlamaFactory training workflows."""

from __future__ import annotations

import math
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .assistant_schema import (
    ActionKind,
    ApprovalPayload,
    DataPlan,
    DatasetProfile,
    EvaluationDiagnosis,
    EvaluationPlan,
    ParameterDecision,
    PreflightReport,
    RequirementDraft,
    TrainingObjective,
    TrainingAdjustment,
    TrainingPlan,
    WorkflowState,
)
from .assistant_diagnosis import diagnose_evaluation
from .assistant_projection import available_actions, build_workflow_steps
from .assistant_state import plan_hash
from .eval_schema import EvalRequest, ModelUnderTest


class ApprovalConflict(RuntimeError):
    pass


_SECRET_PATTERNS = (
    (re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{8,}"), "Bearer [REDACTED]"),
    (re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b"), "[REDACTED_API_KEY]"),
    (
        re.compile(
            r"(?i)\b(api[_-]?key|access[_-]?token|secret|password)"
            r"\s*[:=]\s*[^\s,;]+"
        ),
        r"\1=[REDACTED]",
    ),
    (re.compile(r"/(Users|home)/[^/\s]+"), r"/\1/[REDACTED]"),
)


def _redact_sensitive(value: Any) -> str:
    text = str(value or "").replace("\x00", "")
    for pattern, replacement in _SECRET_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def sanitize_error(exc: Exception | str) -> str:
    text = " ".join(_redact_sensitive(exc).split())
    return text[:400] or "unknown error"


def sanitize_log_excerpt(value: Any) -> str:
    text = _redact_sensitive(value)
    return text[-2000:]


def redact_evidence(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): redact_evidence(item) for key, item in value.items()}
    if isinstance(value, list):
        return [redact_evidence(item) for item in value]
    if isinstance(value, tuple):
        return [redact_evidence(item) for item in value]
    return _redact_sensitive(value) if isinstance(value, str) else value


class AssistantService:
    EXTERNAL_ACTIONS = {
        ActionKind.START_DATAGEN,
        ActionKind.START_TRAINING,
        ActionKind.START_EVALUATION,
    }

    def __init__(
        self,
        store,
        planner,
        tools,
        policy,
        preflight_runner,
        data_preparer,
    ):
        self.store = store
        self.planner = planner
        self.tools = tools
        self.policy = policy
        self.preflight_runner = preflight_runner
        self.data_preparer = data_preparer

    def create_workflow(self, message: str) -> dict[str, Any]:
        workflow_id = self.store.create_workflow()
        return self.add_message(workflow_id, message)

    def add_message(self, workflow_id: str, message: str) -> dict[str, Any]:
        if not isinstance(message, str) or not message.strip():
            raise ValueError("message must not be empty")
        workflow = self.store.get_workflow(workflow_id)
        self.store.append_message(workflow_id, "user", message.strip())
        if workflow["state"] == WorkflowState.REQUIREMENTS_REVIEW.value:
            self.store.reopen_requirements(workflow_id)
            workflow = self.store.get_workflow(workflow_id)
        if workflow["state"] != WorkflowState.COLLECTING_REQUIREMENTS.value:
            self.store.append_message(
                workflow_id,
                "assistant",
                "当前方案已进入执行流程；请先处理待确认操作，或创建新的训练任务。",
            )
            return self.snapshot(workflow_id)

        try:
            draft = self.planner.extract_requirement_draft(
                self.store.list_messages(workflow_id)
            )
            if draft.ready_for_review:
                self.store.publish_requirement_review(workflow_id, draft)
            else:
                self.store.publish_incomplete_requirement_draft(
                    workflow_id, draft
                )
        except Exception as exc:
            self.store.append_event(
                workflow_id,
                "planner_invalid_output",
                {"error": sanitize_error(exc)},
            )
            self.store.append_message(
                workflow_id,
                "assistant",
                "方案输出未通过结构校验，请补充或换一种方式描述训练目标。",
            )
        return self.snapshot(workflow_id)

    def revise_requirements(self, workflow_id: str) -> dict[str, Any]:
        workflow = self.store.get_workflow(workflow_id)
        if workflow["state"] != WorkflowState.REQUIREMENTS_REVIEW.value:
            raise ApprovalConflict("workflow is not ready for requirement revision")
        self.store.reopen_requirements(workflow_id)
        self.store.append_message(
            workflow_id,
            "assistant",
            "已回到需求收集，请直接补充或修改场景、问题、目标与成功标准。",
        )
        return self.snapshot(workflow_id)

    def retry_data_plan(self, workflow_id: str) -> dict[str, Any]:
        if not self.store.retry_data_plan(workflow_id):
            raise ApprovalConflict("workflow cannot retry data plan preparation")
        return self.snapshot(workflow_id)

    def cancel(self, workflow_id: str, reason: str) -> dict[str, Any]:
        self.store.request_cancellation(workflow_id, reason)
        return self.snapshot(workflow_id)

    def on_cancellation_observation(
        self, action: dict[str, Any], result: dict[str, Any]
    ) -> None:
        """Publish one fenced stop result; incomplete identity evidence retries."""
        if not result.get("terminal"):
            raise RuntimeError(
                "external job stop was not confirmed: "
                + sanitize_error(result.get("detail") or "unknown")
            )
        payload = action["payload"]
        evidence = {
            "cancel_request_id": payload["cancel_request_id"],
            "kind": payload["kind"],
            "job_id": payload["job_id"],
            "stopped": bool(result.get("stopped")),
            "detail": sanitize_error(result.get("detail") or "already terminal"),
        }
        if not self.store.complete_cancellation_action(
            workflow_id=action["workflow_id"],
            cancel_request_id=payload["cancel_request_id"],
            action_id=action["action_id"],
            lease_token=action["lease_token"],
            result=evidence,
        ):
            raise RuntimeError("cancellation action lease or workflow fence was lost")

    def snapshot(self, workflow_id: str) -> dict[str, Any]:
        workflow = self.store.get_workflow(workflow_id)
        events = self.store.list_events(workflow_id, 0, limit=1000)[-100:]
        pending_approvals = self.store.list_pending_approvals(workflow_id)
        steps = build_workflow_steps(workflow, events, pending_approvals)
        return {
            **workflow,
            "messages": self.store.list_messages(workflow_id),
            "pending_approvals": pending_approvals,
            "events": events,
            "workflow_steps": [
                step.model_dump(mode="json") for step in steps
            ],
            "available_actions": available_actions(steps),
        }

    def list_workflows(self, limit: int = 50) -> list[dict[str, Any]]:
        return self.store.list_workflows(limit)

    def list_events(self, workflow_id: str, after_id: int = 0) -> list[dict[str, Any]]:
        return self.store.list_events(workflow_id, after_id, limit=100)

    def approve(
        self, workflow_id: str, approval_id: str, expected_hash: str
    ) -> dict[str, Any]:
        approval = self.store.get_approval(workflow_id, approval_id)
        if approval["status"] != "pending":
            raise ApprovalConflict("approval is no longer pending")
        action = ActionKind(approval["action"])
        current_payload = self._approval_payload_for_action(workflow_id, action)
        if plan_hash(current_payload) != expected_hash:
            self.store.mark_approval_stale(approval_id)
            raise ApprovalConflict("plan changed; request a new approval")
        if action == ActionKind.START_TRAINING:
            if not self._refresh_training_preflight_before_claim(
                workflow_id, approval_id, current_payload
            ):
                return self.snapshot(workflow_id)
        refs = self._new_external_refs(action, current_payload)
        execution = self.store.prepare_external_execution(
            workflow_id,
            approval_id,
            expected_hash,
            current_payload,
            refs,
        )
        if execution is None:
            raise ApprovalConflict("approval is no longer pending")
        try:
            with self.store.action_lease_heartbeat(
                execution, lease_seconds=600
            ):
                self.execute_approval_action(execution)
        except Exception as exc:
            error = sanitize_error(exc)
            if self.is_retryable_execution_error(exc):
                self.store.defer_action(
                    execution["action_id"],
                    execution["lease_token"],
                    datetime.now(timezone.utc) + timedelta(seconds=60),
                    error,
                )
                self.store.append_event_once(
                    workflow_id,
                    "approval_execution_deferred",
                    {"approval_id": approval_id, "error": error},
                )
                self.store.append_message(
                    workflow_id,
                    "assistant",
                    "外部任务提交尚未完成，已保存执行意图；monitor 将使用同一任务 ID 重试。",
                )
            else:
                self.fail_execution_action(execution, exc)
        return self.snapshot(workflow_id)

    @staticmethod
    def is_retryable_execution_error(exc: Exception) -> bool:
        return not isinstance(exc, (ValueError, TypeError, ApprovalConflict))

    def fail_execution_action(
        self, action: dict[str, Any], exc: Exception | str
    ) -> None:
        error = sanitize_error(exc)
        if not self.store.fail_external_execution(
            workflow_id=action["workflow_id"],
            approval_id=action["approval_id"],
            action_id=action["action_id"],
            lease_token=action["lease_token"],
            error=error,
            message=(
                f"该操作未能执行：{error}。已保留工作流并生成新的审批，"
                "请修正数据、配置或运行环境后再确认。"
            ),
        ):
            raise RuntimeError("execution lease was lost before failure handling")

    def reject(self, workflow_id: str, approval_id: str) -> dict[str, Any]:
        approval = self.store.get_approval(workflow_id, approval_id)
        workflow = self.store.get_workflow(workflow_id)
        if (
            approval["action"] == ActionKind.START_DATAGEN.value
            and workflow["state"] == WorkflowState.DATA_PLAN_READY.value
        ):
            rejected = self.store.reject_data_plan_for_revision(
                workflow_id,
                approval_id,
                "已退回需求收集，请直接说明希望修改的数据量、任务、prompt 或门槛。",
            )
        else:
            rejected = self.store.reject_approval(workflow_id, approval_id)
        if not rejected:
            raise ApprovalConflict("approval is no longer pending")
        return self.snapshot(workflow_id)

    def _approval_payload_for_action(
        self, workflow_id: str, action: ActionKind
    ) -> ApprovalPayload:
        workflow = self.store.get_workflow(workflow_id)
        warnings: list[str] = []
        if action == ActionKind.CONFIRM_REQUIREMENTS:
            draft = RequirementDraft.model_validate(workflow["requirement_draft"])
            if not draft.ready_for_review or draft.proposed_objective is None:
                raise ApprovalConflict("requirements are not ready for confirmation")
            objective_hash = plan_hash(draft.proposed_objective)
            plan = {
                "requirement_draft": draft.model_dump(mode="json"),
                "objective_hash": objective_hash,
            }
        elif action == ActionKind.START_DATAGEN:
            plan = DataPlan.model_validate(workflow["data_plan"])
            objective_hash = workflow.get("objective_hash")
            if not objective_hash:
                raise ApprovalConflict("requirements were not explicitly confirmed")
            warnings = [f"objective_hash:{objective_hash}"]
        elif action == ActionKind.START_TRAINING:
            plan = TrainingPlan.model_validate(workflow["training_plan"])
            if workflow["preflight"]:
                warnings = self._warning_fingerprints(
                    PreflightReport.model_validate(workflow["preflight"])
                )
        elif action in (
            ActionKind.START_EVALUATION,
            ActionKind.RETRY_SCORING,
            ActionKind.SKIP_EVALUATION,
        ):
            plan = EvaluationPlan.model_validate(workflow["evaluation_plan"])
        elif action in (
            ActionKind.ACCEPT_CANDIDATE,
            ActionKind.FINISH_WITHOUT_ACCEPTING,
        ):
            plan = workflow["diagnosis"] or {"decision": "accept_candidate"}
        elif action == ActionKind.START_ITERATION:
            plan = DataPlan.model_validate(workflow["data_plan"])
        elif action == ActionKind.RECOVER_TRAINING:
            plan = TrainingPlan.model_validate(workflow["training_plan"])
        else:  # pragma: no cover - enum exhaustiveness guard
            raise ValueError(f"unsupported action: {action.value}")
        if hasattr(plan, "model_dump"):
            plan = plan.model_dump(mode="json")
        return ApprovalPayload(
            action=action, plan=plan, decision_warnings=sorted(warnings)
        )

    def _execute_approved(
        self,
        workflow_id: str,
        action: ActionKind,
        payload: ApprovalPayload,
        execution_key: str | None = None,
    ) -> None:
        workflow = self.store.get_workflow(workflow_id)
        if action == ActionKind.RETRY_SCORING:
            if not workflow.get("eval_id"):
                raise ApprovalConflict("no evaluation artifacts are available")
            if execution_key is None:
                raise RuntimeError("scoring retry requires a durable execution key")
            monitor_key = (
                f"{workflow_id}:evaluation-score-retry:"
                f"{workflow['eval_id']}:{execution_key}"
            )
            try:
                persisted_monitor = self.store.get_action_by_key(monitor_key)
            except KeyError:
                persisted_monitor = None
            if persisted_monitor is not None:
                # The atomic transition already published its monitor. The
                # monitor may even have advanced the workflow before a crashed
                # approval executor reconciles its outbox row.
                return
            if workflow["state"] == WorkflowState.AB_PLAN_READY.value:
                self.store.transition_bundle(
                    workflow_id,
                    "evaluation_scoring_retried",
                    {"eval_id": workflow["eval_id"]},
                    message="复用现有 A/B 推理产物，重新执行本地评分与诊断。",
                    scheduled_actions=[
                        {
                            "action": "monitor_evaluation",
                            "payload": {"eval_id": workflow["eval_id"]},
                            "idempotency_key": monitor_key,
                        }
                    ],
                )
            else:
                raise ApprovalConflict("workflow is not ready to retry scoring")
            return
        if action == ActionKind.SKIP_EVALUATION:
            if workflow["state"] == WorkflowState.AB_PLAN_READY.value:
                self.store.transition(workflow_id, "evaluation_skipped", {})
            elif workflow["state"] != WorkflowState.COMPLETED.value:
                raise ApprovalConflict("workflow is not ready to skip evaluation")
            return
        if action == ActionKind.ACCEPT_CANDIDATE:
            diagnosis = EvaluationDiagnosis.model_validate(workflow["diagnosis"])
            if not diagnosis.accepted:
                raise ApprovalConflict("candidate did not pass the acceptance gates")
            if workflow["state"] == WorkflowState.DIAGNOSIS_READY.value:
                self.store.transition(
                    workflow_id, "candidate_accepted", payload.plan
                )
            elif workflow["state"] != WorkflowState.COMPLETED.value:
                raise ApprovalConflict("workflow is not ready to accept candidate")
            return
        if action == ActionKind.FINISH_WITHOUT_ACCEPTING:
            if workflow["state"] == WorkflowState.DIAGNOSIS_READY.value:
                self.store.transition(
                    workflow_id, "candidate_rejected", payload.plan
                )
            elif workflow["state"] != WorkflowState.COMPLETED.value:
                raise ApprovalConflict("workflow is not ready to finish")
            return
        if action == ActionKind.START_ITERATION:
            if workflow["state"] == WorkflowState.DATA_PLAN_READY.value:
                return
            if workflow["state"] != WorkflowState.DIAGNOSIS_READY.value:
                raise ApprovalConflict("workflow is not ready for another iteration")
            if workflow["iteration"] >= 3:
                raise ApprovalConflict("maximum iteration count reached")
            next_iteration = workflow["iteration"] + 1
            plan = DataPlan.model_validate(payload.plan)
            self.store.transition_bundle(
                workflow_id,
                "iteration_started",
                {"iteration": next_iteration, "data_plan": payload.plan},
                workflow_updates={
                    "iteration": next_iteration,
                    "data_plan_json": plan,
                    "dataset_profile_json": None,
                    "training_plan_json": None,
                    "preflight_json": None,
                    "evaluation_plan_json": None,
                    "diagnosis_json": None,
                    "train_job_id": None,
                    "eval_id": None,
                },
                approvals=[
                    {
                        "action": ActionKind.START_DATAGEN,
                        "plan": plan,
                        "summary": self._data_plan_summary(plan),
                        "decision_warnings": self._objective_binding_warning(
                            workflow
                        ),
                    }
                ],
            )
            return
        if action == ActionKind.RECOVER_TRAINING:
            plan = TrainingPlan.model_validate(payload.plan)
            if workflow["state"] == WorkflowState.TRAIN_FAILED.value:
                self.store.transition_bundle(
                    workflow_id,
                    "recovery_plan_created",
                    {"training_plan": plan.model_dump(mode="json")},
                    workflow_updates={"training_plan_json": plan},
                )
                workflow = self.store.get_workflow(workflow_id)
            if workflow["state"] in {
                WorkflowState.TRAIN_PLAN_READY.value,
                WorkflowState.PREFLIGHT_BLOCKED.value,
            }:
                self.run_preflight(workflow_id)
            elif workflow["state"] != WorkflowState.TRAIN_READY.value:
                raise ApprovalConflict("workflow is not recoverable")
            return
        raise ValueError(f"unsupported action: {action.value}")

    def _new_external_refs(
        self, action: ActionKind, payload: ApprovalPayload
    ) -> dict[str, Any]:
        if action == ActionKind.START_DATAGEN:
            plan = DataPlan.model_validate(payload.plan)
            return {
                "launches": [
                    {
                        "job_id": self.tools.new_job_id(),
                        "task_type": item.task_type,
                    }
                    for item in plan.items
                ]
            }
        if action == ActionKind.START_TRAINING:
            return {"job_id": self.tools.new_job_id()}
        if action == ActionKind.START_EVALUATION:
            return {"eval_id": self.tools.new_job_id()}
        return {}

    def execute_approval_action(self, execution: dict[str, Any]) -> None:
        """Replay a durable external execution using its preallocated IDs."""
        workflow_id = execution["workflow_id"]
        approval_id = execution["approval_id"]
        payload = ApprovalPayload.model_validate(
            execution["payload"]["approval_payload"]
        )
        refs = execution["payload"]["external_refs"]
        workflow = self.store.get_workflow(workflow_id)

        if payload.action == ActionKind.CONFIRM_REQUIREMENTS:
            draft = RequirementDraft.model_validate(
                payload.plan["requirement_draft"]
            )
            objective = draft.proposed_objective
            if objective is None:
                raise ApprovalConflict("confirmed requirements have no objective")
            if not self.store.confirm_requirements_and_schedule_plan(
                workflow_id=workflow_id,
                objective=objective,
                objective_hash=payload.plan["objective_hash"],
                approval_id=approval_id,
                action_id=execution["action_id"],
                lease_token=execution["lease_token"],
            ):
                raise ApprovalConflict("requirements are no longer confirmable")
            return

        if payload.action not in self.EXTERNAL_ACTIONS:
            self._execute_approved(
                workflow_id,
                payload.action,
                payload,
                execution_key=execution["action_id"],
            )
            if not self.store.commit_local_execution(
                workflow_id=workflow_id,
                approval_id=approval_id,
                action_id=execution["action_id"],
                lease_token=execution["lease_token"],
            ):
                raise RuntimeError("local approval execution is no longer active")
            return

        if payload.action == ActionKind.START_DATAGEN:
            self._assert_datagen_objective_binding(workflow_id, workflow, payload)
            plan = DataPlan.model_validate(payload.plan)
            launches = self.tools.start_datagen(
                workflow_id, plan, list(refs["launches"])
            )
            if launches != refs["launches"]:
                raise RuntimeError("data generation returned different job IDs")
            by_task = {item.task_type: item for item in plan.items}
            records = [
                {
                    **launch,
                    "iteration": workflow["iteration"],
                    "plan_item_hash": plan_hash(by_task[launch["task_type"]]),
                }
                for launch in launches
            ]
            prior = list(workflow["datagen_jobs"] or [])
            self.store.commit_external_start(
                workflow_id=workflow_id,
                approval_id=approval_id,
                action_id=execution["action_id"],
                lease_token=execution["lease_token"],
                transition_event="datagen_started",
                workflow_updates={"datagen_jobs_json": [*prior, *records]},
                event_payload={"launches": records},
                monitor_action="monitor_datagen",
                monitor_payload={"launches": records},
                monitor_key=(
                    f"{workflow_id}:datagen:{workflow['iteration']}:{len(prior)}"
                ),
            )
            return

        if payload.action == ActionKind.START_TRAINING:
            if workflow["state"] != WorkflowState.TRAIN_READY.value:
                raise ApprovalConflict("workflow is not train_ready")
            plan = TrainingPlan.model_validate(payload.plan)
            expected_job_id = refs["job_id"]
            job_id = self.tools.start_training(
                workflow_id, plan, expected_job_id
            )
            if job_id != expected_job_id:
                raise RuntimeError("training returned a different job ID")
            self.store.commit_external_start(
                workflow_id=workflow_id,
                approval_id=approval_id,
                action_id=execution["action_id"],
                lease_token=execution["lease_token"],
                transition_event="training_started",
                workflow_updates={"train_job_id": job_id},
                event_payload={"job_id": job_id},
                monitor_action="monitor_training",
                monitor_payload={"job_id": job_id},
                monitor_key=f"{workflow_id}:training:{job_id}",
            )
            return

        if payload.action == ActionKind.START_EVALUATION:
            plan = EvaluationPlan.model_validate(payload.plan)
            request = self._build_eval_request(workflow, plan)
            expected_eval_id = refs["eval_id"]
            eval_id = self.tools.start_evaluation(
                workflow_id, request, plan, expected_eval_id
            )
            if eval_id != expected_eval_id:
                raise RuntimeError("evaluation returned a different job ID")
            self.store.commit_external_start(
                workflow_id=workflow_id,
                approval_id=approval_id,
                action_id=execution["action_id"],
                lease_token=execution["lease_token"],
                transition_event=(
                    "evaluation_retried"
                    if workflow["state"] == WorkflowState.DIAGNOSIS_READY.value
                    else "evaluation_started"
                ),
                workflow_updates={"eval_id": eval_id},
                event_payload={"eval_id": eval_id},
                monitor_action="monitor_evaluation",
                monitor_payload={"eval_id": eval_id},
                monitor_key=f"{workflow_id}:evaluation:{eval_id}",
            )
            return

        raise ValueError(f"unsupported external action: {payload.action.value}")

    def prepare_data_plan_action(self, action: dict[str, Any]) -> None:
        workflow = self.store.get_workflow(action["workflow_id"])
        if workflow["state"] != WorkflowState.DATA_PLAN_PREPARING.value:
            self.store.complete_action(
                action["action_id"], action["lease_token"]
            )
            return
        objective = TrainingObjective.model_validate(
            workflow["confirmed_objective"]
        )
        expected_hash = action["payload"]["objective_hash"]
        if plan_hash(objective) != expected_hash:
            raise ApprovalConflict("confirmed objective changed before planning")
        data_plan = self.planner.create_data_plan(objective)
        if not self.store.publish_confirmed_data_plan(
            workflow_id=action["workflow_id"],
            action_id=action["action_id"],
            lease_token=action["lease_token"],
            objective_hash=expected_hash,
            data_plan=data_plan,
            summary=self._data_plan_summary(data_plan),
        ):
            raise RuntimeError("data plan publication lease was lost")

    @staticmethod
    def is_retryable_planning_error(exc: Exception) -> bool:
        return not isinstance(exc, (ValueError, TypeError, ApprovalConflict))

    def fail_data_plan_action(
        self, action: dict[str, Any], exc: Exception | str
    ) -> None:
        if not self.store.fail_data_plan_preparation(
            workflow_id=action["workflow_id"],
            action_id=action["action_id"],
            lease_token=action["lease_token"],
            error=sanitize_error(exc),
        ):
            raise RuntimeError("data plan failure lease was lost")

    def _assert_datagen_objective_binding(
        self,
        workflow_id: str,
        workflow: dict[str, Any],
        payload: ApprovalPayload,
    ) -> None:
        objective_hash = workflow.get("objective_hash")
        if not objective_hash or not workflow.get("confirmed_objective"):
            raise ApprovalConflict("requirements were not explicitly confirmed")
        if plan_hash(workflow["confirmed_objective"]) != objective_hash:
            raise ApprovalConflict("confirmed objective hash changed")
        if f"objective_hash:{objective_hash}" not in payload.decision_warnings:
            raise ApprovalConflict("data plan approval is not bound to objective")
        confirmed = self.store.list_recent_events(
            workflow_id, "requirements_confirmed", limit=20
        )
        if not any(
            event["payload"].get("objective_hash") == objective_hash
            for event in confirmed
        ):
            raise ApprovalConflict("requirement confirmation evidence is missing")

    def on_datagen_terminal(
        self, workflow_id: str, results: list[dict[str, Any]]
    ) -> dict[str, Any]:
        workflow = self.store.get_workflow(workflow_id)
        if workflow["state"] != WorkflowState.DATA_GENERATING.value:
            return self.snapshot(workflow_id)
        all_jobs = list(workflow["datagen_jobs"] or [])
        current_jobs = [
            row
            for row in all_jobs
            if row.get("iteration") == workflow["iteration"]
        ]
        by_job = {row.get("job_id"): dict(row) for row in current_jobs}
        for result in results:
            merged = {**by_job.get(result.get("job_id"), {}), **result}
            merged.setdefault("iteration", workflow["iteration"])
            by_job[result.get("job_id")] = merged
        merged_current_jobs = list(by_job.values())
        merged_jobs = [
            row
            for row in all_jobs
            if row.get("iteration") != workflow["iteration"]
        ] + merged_current_jobs
        self.store.update_workflow_fields(
            workflow_id, datagen_jobs_json=merged_jobs
        )

        failed_tasks = {
            result.get("task_type")
            for result in results
            if result.get("status") in {"failed", "error", "interrupted"}
        }
        plan = DataPlan.model_validate(workflow["data_plan"])
        if failed_tasks:
            retry = DataPlan(
                items=[
                    item for item in plan.items if item.task_type in failed_tasks
                ],
                holdout_ratio=plan.holdout_ratio,
                validation_ratio=plan.validation_ratio,
                split_seed=plan.split_seed,
                rationale="Retry failed data-generation task types",
                risks=[*plan.risks, "Previous data generation attempt failed"],
                training_adjustments=plan.training_adjustments,
            )
            self.store.transition_bundle(
                workflow_id,
                "datagen_failed",
                {"failed_task_types": sorted(failed_tasks), "results": results},
                workflow_updates={"data_plan_json": retry},
                approvals=[
                    {
                        "action": ActionKind.START_DATAGEN,
                        "plan": retry,
                        "summary": self._data_plan_summary(retry),
                        "decision_warnings": self._objective_binding_warning(
                            workflow
                        ),
                    }
                ],
            )
            return self.snapshot(workflow_id)

        successful = {
            row.get("task_type"): row
            for row in merged_current_jobs
            if row.get("status") in {"succeeded", "done"} and row.get("output")
        }
        requested = {item.task_type for item in plan.items}
        if not requested <= set(successful):
            return self.snapshot(workflow_id)

        # A retry plan contains only failed task types, while successful artifacts from
        # earlier attempts remain part of the frozen dataset for this iteration.
        all_successful_outputs = [
            successful[task]["output"] for task in sorted(successful)
        ]

        artifact_dir = (
            Path(self.store.path).parent
            / "assistant_artifacts"
            / workflow_id
            / f"it{workflow['iteration']}"
        )
        try:
            prepared = self.data_preparer(
                all_successful_outputs,
                workflow_id,
                workflow["iteration"],
                plan.holdout_ratio,
                plan.split_seed,
                artifact_dir,
                validation_ratio=plan.validation_ratio,
                register=True,
            )
        except (OSError, TypeError, ValueError) as exc:
            error = sanitize_error(exc)
            retry = plan.model_copy(
                update={
                    "rationale": "Repair generated data coverage or format",
                    "risks": [
                        *plan.risks,
                        f"Previous data preparation failed: {error}",
                    ],
                }
            )
            self.store.transition_bundle(
                workflow_id,
                "datagen_failed",
                {"error": error, "results": results},
                workflow_updates={"data_plan_json": retry},
                message=(
                    "生成产物未通过数据覆盖/格式校验，已保留现有产物。"
                    "请修复数据源或生成参数后重新确认。"
                ),
                approvals=[
                    {
                        "action": ActionKind.START_DATAGEN,
                        "plan": retry,
                        "summary": self._data_plan_summary(retry),
                        "decision_warnings": self._objective_binding_warning(
                            workflow
                        ),
                    }
                ],
            )
            return self.snapshot(workflow_id)
        profile = DatasetProfile.model_validate(prepared.profile)
        objective = TrainingObjective.model_validate(workflow["objective"])
        missing_critical_slices = sorted(
            selector
            for selector in objective.critical_slices
            if int(profile.slice_counts.get(selector, 0)) <= 0
        )
        if missing_critical_slices:
            retry = plan.model_copy(
                update={
                    "rationale": "Repair missing critical-slice coverage",
                    "risks": [
                        *plan.risks,
                        "Frozen holdout is missing critical slices: "
                        + ", ".join(missing_critical_slices),
                    ],
                }
            )
            self.store.transition_bundle(
                workflow_id,
                "datagen_failed",
                {
                    "missing_critical_slices": missing_critical_slices,
                    "dataset_profile": profile.model_dump(mode="json"),
                },
                workflow_updates={"data_plan_json": retry},
                message=(
                    "冻结 holdout 未覆盖全部关键切片，不会进入训练。"
                    "请调整数据源或生成 prompt 后重新确认。"
                ),
                approvals=[
                    {
                        "action": ActionKind.START_DATAGEN,
                        "plan": retry,
                        "summary": self._data_plan_summary(retry),
                        "decision_warnings": self._objective_binding_warning(
                            workflow
                        ),
                    }
                ],
            )
            return self.snapshot(workflow_id)
        training_plan = TrainingPlan.model_validate(self.policy(objective, profile))
        training_plan = training_plan.model_copy(
            update={
                "dataset_sha256": profile.sha256,
                "eval_sha256": profile.eval_sha256,
            }
        )
        training_plan = self._apply_training_adjustments(
            training_plan, plan.training_adjustments
        )
        self.store.finish_data_preparation(
            workflow_id,
            outputs=all_successful_outputs,
            dataset_profile=profile,
            training_plan=training_plan,
        )
        return self.run_preflight(workflow_id)

    @staticmethod
    def _apply_training_adjustments(
        plan: TrainingPlan, adjustments: list[TrainingAdjustment]
    ) -> TrainingPlan:
        if not adjustments:
            return plan
        config = plan.config
        decisions = list(plan.decisions)
        estimated_steps = plan.estimated_steps
        estimated_vram = plan.estimated_vram_gb
        eta_low = plan.estimated_hours_low
        eta_high = plan.estimated_hours_high
        risks = list(plan.risks)
        for adjustment in adjustments:
            parameter = adjustment.parameter
            if parameter in {"learning_rate", "num_train_epochs"}:
                old_value = getattr(config.train, parameter)
                value = float(adjustment.value)
                config = config.model_copy(
                    update={
                        "train": config.train.model_copy(
                            update={parameter: value}
                        )
                    }
                )
                if parameter == "num_train_epochs" and old_value:
                    ratio = value / float(old_value)
                    estimated_steps = max(1, round(estimated_steps * ratio))
                    eta_low = eta_low * ratio if eta_low is not None else None
                    eta_high = eta_high * ratio if eta_high is not None else None
            else:
                old_value = getattr(config.method, parameter)
                value = (
                    int(adjustment.value)
                    if parameter == "lora_rank"
                    else float(adjustment.value)
                )
                config = config.model_copy(
                    update={
                        "method": config.method.model_copy(
                            update={parameter: value}
                        )
                    }
                )
                if (
                    parameter == "lora_rank"
                    and estimated_vram is not None
                    and float(old_value) > 0
                    and float(value) > float(old_value)
                ):
                    estimated_vram = round(
                        estimated_vram * float(value) / float(old_value), 2
                    )
                    risks.append(
                        "LoRA rank 上调后按 rank 比例保守放大显存估算，"
                        "最终以重新预检的最小单卡余量为准。"
                    )
            decisions.append(
                ParameterDecision(
                    parameter=parameter,
                    value=value,
                    reason=(
                        f"{adjustment.reason} (diagnosis adjustment: "
                        f"{old_value} -> {value})"
                    ),
                    confidence="medium",
                )
            )
        return plan.model_copy(
            update={
                "config": config,
                "decisions": decisions,
                "estimated_steps": estimated_steps,
                "estimated_vram_gb": estimated_vram,
                "estimated_hours_low": eta_low,
                "estimated_hours_high": eta_high,
                "risks": risks,
            }
        )

    def run_preflight(self, workflow_id: str) -> dict[str, Any]:
        workflow = self.store.get_workflow(workflow_id)
        allowed_states = {
            WorkflowState.TRAIN_PLAN_READY.value,
            WorkflowState.PREFLIGHT_BLOCKED.value,
            WorkflowState.TRAIN_READY.value,
        }
        if workflow["state"] not in allowed_states:
            raise ApprovalConflict("preflight is not available in the current state")
        training_plan = TrainingPlan.model_validate(workflow["training_plan"])
        profile = DatasetProfile.model_validate(workflow["dataset_profile"])
        report = PreflightReport.model_validate(
            self.preflight_runner(workflow_id, training_plan, profile)
        )
        self.store.publish_preflight(
            workflow_id=workflow_id,
            report=report,
            plan=training_plan,
            summary=self._training_plan_summary(training_plan, report),
            decision_warnings=self._warning_fingerprints(report),
        )
        return self.snapshot(workflow_id)

    def _refresh_training_preflight_before_claim(
        self,
        workflow_id: str,
        approval_id: str,
        old_payload: ApprovalPayload,
    ) -> bool:
        workflow = self.store.get_workflow(workflow_id)
        plan = TrainingPlan.model_validate(workflow["training_plan"])
        profile = DatasetProfile.model_validate(workflow["dataset_profile"])
        report = PreflightReport.model_validate(
            self.preflight_runner(workflow_id, plan, profile)
        )
        new_warnings = self._warning_fingerprints(report)
        if report.status == "block":
            self.store.publish_preflight(
                workflow_id=workflow_id,
                report=report,
                plan=plan,
                summary=self._training_plan_summary(plan, report),
                decision_warnings=new_warnings,
            )
            return False
        if sorted(old_payload.decision_warnings) != sorted(new_warnings):
            self.store.publish_preflight(
                workflow_id=workflow_id,
                report=report,
                plan=plan,
                summary=self._training_plan_summary(plan, report),
                decision_warnings=new_warnings,
            )
            return False
        self.store.update_workflow_fields(workflow_id, preflight_json=report)
        return True

    def on_training_observation(
        self, workflow_id: str, observation: dict[str, Any]
    ) -> dict[str, Any]:
        workflow = self.store.get_workflow(workflow_id)
        if workflow["state"] != WorkflowState.TRAINING.value:
            return self.snapshot(workflow_id)
        observed_job_id = observation.get("job_id")
        if observed_job_id and observed_job_id != workflow["train_job_id"]:
            self.store.append_event_once(
                workflow_id,
                "stale_training_observation_ignored",
                {"job_id": observed_job_id},
            )
            return self.snapshot(workflow_id)
        status_value = observation.get("status")
        if isinstance(status_value, dict):
            status_value = status_value.get("status")
        if status_value == "running":
            if observation.get("milestone") or observation.get("anomaly"):
                self.store.append_event(
                    workflow_id, "training_observation", observation
                )
            return self.snapshot(workflow_id)
        if status_value == "succeeded":
            profile = DatasetProfile.model_validate(workflow["dataset_profile"])
            objective = TrainingObjective.model_validate(workflow["objective"])
            metrics = self._safe_training_metrics(observation.get("metrics") or {})
            training_evidence = self._training_trends(metrics)
            eval_plan = EvaluationPlan(
                eval_dataset_names=profile.eval_dataset_names,
                task_types=list(profile.eval_dataset_names),
                gpus=TrainingPlan.model_validate(workflow["training_plan"]).gpus,
                success_criteria=objective.success_criteria,
                # Every requested task is an automatic non-regression slice;
                # user-selected task_type aliases may add an equivalent view.
                critical_slices=list(
                    dict.fromkeys(
                        [*objective.task_types, *objective.critical_slices]
                    )
                ),
                eval_sha256=profile.eval_sha256,
            )
            execution_request = self._build_eval_request(workflow, eval_plan)
            eval_plan = eval_plan.model_copy(
                update={
                    "execution_request": execution_request,
                    "baseline_source_hash": plan_hash(
                        execution_request.models[0].model_dump(mode="json")
                    ),
                    "candidate_source_hash": plan_hash(
                        execution_request.models[1].model_dump(mode="json")
                    ),
                }
            )
            self.store.transition_bundle(
                workflow_id,
                "training_succeeded",
                {
                    "job_id": workflow["train_job_id"],
                    "status": "succeeded",
                    "metrics": metrics,
                    "training_evidence": training_evidence,
                },
                workflow_updates={"evaluation_plan_json": eval_plan},
                message=(
                    "训练已完成。是否现在使用冻结评测集进行基座模型"
                    "与新模型的 A/B 测试？"
                ),
                approvals=self._evaluation_approval_specs(eval_plan),
            )
        elif status_value in {"failed", "interrupted"}:
            failure_category = str(observation.get("failure_category") or "unknown")
            reason = sanitize_error(observation.get("error", "training failed"))
            checkpoints = observation.get("checkpoints", [])
            log_excerpt = sanitize_log_excerpt(observation.get("log_tail"))
            recovery_plan = self._recovery_training_plan(workflow, checkpoints)
            recovery_approvals = []
            updates = {}
            if recovery_plan is not None:
                updates["training_plan_json"] = recovery_plan
                recovery_approvals.append(
                    {
                        "action": ActionKind.RECOVER_TRAINING,
                        "plan": recovery_plan,
                        "summary": "使用最新 checkpoint 重新预检后续训",
                    }
                )
            self.store.transition_bundle(
                workflow_id,
                "training_failed",
                {
                    "status": status_value,
                    "failure_category": failure_category,
                    "reason": reason,
                    "checkpoints": checkpoints,
                    "log_excerpt": log_excerpt,
                },
                workflow_updates=updates,
                message=(
                    f"训练未完成（{failure_category}）：{reason}。"
                    f"已发现 {len(checkpoints)} 个 checkpoint；"
                    "助手不会自动重启或删除，请先核对日志与 checkpoint "
                    "后决定续训或新建任务。"
                ),
                approvals=recovery_approvals,
            )
        return self.snapshot(workflow_id)

    @staticmethod
    def _recovery_training_plan(
        workflow: dict[str, Any], checkpoints: list[dict[str, Any]]
    ) -> TrainingPlan | None:
        def checkpoint_step(row: dict[str, Any]) -> int:
            name = str(row.get("name") or Path(str(row.get("path", ""))).name)
            prefix = "checkpoint-"
            suffix = name[len(prefix):] if name.startswith(prefix) else ""
            return int(suffix) if suffix.isdigit() else -1

        usable = [
            row
            for row in checkpoints
            if isinstance(row, dict)
            and isinstance(row.get("path"), str)
            and checkpoint_step(row) >= 0
        ]
        if not usable:
            return None
        checkpoint = max(usable, key=checkpoint_step)["path"]
        current = TrainingPlan.model_validate(workflow["training_plan"])
        train = current.config.train.model_copy(
            update={"resume_from_checkpoint": checkpoint}
        )
        output = current.config.output.model_copy(
            update={
                "output_dir": str(Path(checkpoint).parent),
                "overwrite_output_dir": False,
            }
        )
        config = current.config.model_copy(
            update={"train": train, "output": output}
        )
        decision = ParameterDecision(
            parameter="resume_from_checkpoint",
            value=checkpoint,
            reason="训练中断后保留的最新 checkpoint，需用户批准并重新预检",
            confidence="high",
        )
        return current.model_copy(
            update={
                "config": config,
                "decisions": [*current.decisions, decision],
                "risks": [
                    *current.risks,
                    "续训前需确认 checkpoint 与当前数据、模型和训练参数一致。",
                ],
            }
        )

    def on_evaluation_terminal(
        self, workflow_id: str, observation: dict[str, Any]
    ) -> dict[str, Any]:
        workflow = self.store.get_workflow(workflow_id)
        if workflow["state"] != WorkflowState.EVALUATING.value:
            return self.snapshot(workflow_id)
        observed_eval_id = observation.get("eval_id")
        if observed_eval_id and observed_eval_id != workflow["eval_id"]:
            self.store.append_event_once(
                workflow_id,
                "stale_evaluation_observation_ignored",
                {"eval_id": observed_eval_id},
            )
            return self.snapshot(workflow_id)
        status = observation.get("status")
        if isinstance(status, dict):
            status = status.get("status")
        if status in {"failed", "interrupted"}:
            eval_plan = EvaluationPlan.model_validate(workflow["evaluation_plan"])
            log_excerpt = sanitize_log_excerpt(observation.get("log_tail"))
            error = sanitize_error(
                observation.get("error") or log_excerpt or f"evaluation {status}"
            )
            self.store.transition_bundle(
                workflow_id,
                "evaluation_failed",
                {
                    "status": status,
                    "error": error,
                    "log_excerpt": log_excerpt,
                },
                message=(
                    "A/B 评测未完成，已保留现有产物。"
                    "可重新批准评测，或跳过本轮。"
                ),
                approvals=self._evaluation_approval_specs(eval_plan),
            )
            return self.snapshot(workflow_id)
        if status != "succeeded":
            return self.snapshot(workflow_id)
        objective = TrainingObjective.model_validate(workflow["objective"])
        eval_plan = EvaluationPlan.model_validate(workflow["evaluation_plan"])
        try:
            comparison = redact_evidence(
                self.tools.score_evaluation(
                    workflow["eval_id"], eval_plan.critical_slices
                )
            )
        except (FileNotFoundError, TypeError, ValueError) as exc:
            return self.on_evaluation_processing_failed(
                workflow_id, sanitize_error(exc)
            )
        per_model = comparison.get("per_model") or {}
        baseline_summary = per_model.get(eval_plan.baseline_name) or {}
        candidate_summary = per_model.get(eval_plan.candidate_name) or {}
        primary_metric = objective.success_criteria.primary_metric
        evidence_issues = self._evaluation_evidence_issues(
            baseline_summary,
            candidate_summary,
            comparison.get("paired_comparison") or {},
            eval_plan.task_types,
            primary_metric,
            eval_plan.critical_slices,
            objective.task_types,
        )
        if evidence_issues:
            self.store.transition_bundle(
                workflow_id,
                "evaluation_failed",
                {
                    "status": "evidence_incomplete",
                    "issues": evidence_issues,
                    "comparison": comparison,
                },
                message=(
                    "A/B 证据不完整，不会接受候选模型。"
                    "已保留产物，请重新批准评测或跳过本轮。"
                ),
                approvals=self._evaluation_approval_specs(eval_plan),
            )
            return self.snapshot(workflow_id)
        paired = comparison.get("paired_comparison") or {}
        requested_task_slice_counts = {
            task: int((paired.get("slices") or {}).get(task, {}).get("n") or 0)
            for task in objective.task_types
        }
        paired["requested_task_slice_counts"] = requested_task_slice_counts
        paired["minimum_requested_task_slice_n"] = min(
            requested_task_slice_counts.values(), default=0
        )
        training_events = self.store.list_recent_events(
            workflow_id, "training_succeeded", limit=1
        )
        persisted_training_evidence = (
            training_events[0]["payload"].get("training_evidence")
            if training_events
            else {}
        )
        diagnosis = EvaluationDiagnosis.model_validate(
            diagnose_evaluation(
                baseline=self._diagnosis_metrics(baseline_summary, primary_metric),
                candidate=self._diagnosis_metrics(candidate_summary, primary_metric),
                paired=paired or {"n": 0},
                criteria=objective.success_criteria,
                training={
                    "finetune_type": TrainingPlan.model_validate(
                        workflow["training_plan"]
                    ).config.method.stage,
                    **(persisted_training_evidence or {}),
                },
            )
        )
        diagnosis_json = diagnosis.model_dump(mode="json")
        primary_improvement = self._primary_improvement(
            baseline_summary, candidate_summary, primary_metric
        )
        stop_reason = None
        if not diagnosis.accepted:
            stop_reason = self._iteration_stop_reason(
                workflow_id,
                current_improvement=primary_improvement,
                threshold=objective.success_criteria.min_improvement,
                iteration=workflow["iteration"],
                train_job_id=workflow["train_job_id"],
            )
        plan_key = f"{workflow_id}:iteration-plan:{workflow['iteration']}"
        explanation_key = (
            f"{workflow_id}:diagnosis-explanation:{workflow['iteration']}"
        )
        scheduled_actions = [
            {
                "action": "explain_diagnosis",
                "payload": {"comparison": comparison},
                "idempotency_key": explanation_key,
            }
        ]
        iteration_scheduled = (
            not diagnosis.accepted
            and diagnosis.category != "evaluation_quality_issue"
            and stop_reason is None
        )
        if iteration_scheduled:
            scheduled_actions.append(
                {
                    "action": "plan_iteration",
                    "payload": {
                        "comparison": comparison,
                        "diagnosis": diagnosis_json,
                    },
                    "idempotency_key": plan_key,
                }
            )
        extra_events = []
        if stop_reason is not None:
            extra_events.append(
                {
                    "event_type": "manual_review_required",
                    "payload": {"reason": stop_reason},
                }
            )
        decision_approval = (
            {
                "action": ActionKind.ACCEPT_CANDIDATE,
                "plan": diagnosis_json,
                "summary": "接受已通过全部门槛的候选模型",
            }
            if diagnosis.accepted
            else {
                "action": ActionKind.FINISH_WITHOUT_ACCEPTING,
                "plan": diagnosis_json,
                "summary": "结束本轮且不接受未达标候选模型",
            }
        )
        approvals = [decision_approval]
        if diagnosis.category == "evaluation_quality_issue":
            approvals.append(
                {
                    "action": ActionKind.START_EVALUATION,
                    "plan": eval_plan,
                    "summary": "修复 judge/评测证据后重跑冻结 A/B 评测",
                }
            )
        self.store.transition_bundle(
            workflow_id,
            "evaluation_completed",
            {
                "comparison": comparison,
                "diagnosis": diagnosis_json,
                "primary_improvement": primary_improvement,
                "iteration": workflow["iteration"],
                "train_job_id": workflow["train_job_id"],
                "eval_id": workflow["eval_id"],
            },
            workflow_updates={"diagnosis_json": diagnosis},
            message=diagnosis.summary,
            approvals=approvals,
            scheduled_actions=scheduled_actions,
            extra_events=extra_events,
        )
        if iteration_scheduled:
            published = self.store.get_action_by_key(plan_key)
            action = self.store.lease_action(published["action_id"])
            if action is None:
                return self.snapshot(workflow_id)
            try:
                with self.store.action_lease_heartbeat(
                    action, lease_seconds=600
                ):
                    self.plan_iteration_action(action)
            except Exception as exc:
                error = sanitize_error(exc)
                self.store.defer_action(
                    action["action_id"],
                    action["lease_token"],
                    datetime.now(timezone.utc) + timedelta(seconds=60),
                    error,
                )
                self.store.append_event_once(
                    workflow_id,
                    "iteration_planning_deferred",
                    {"error": error, "iteration": workflow["iteration"]},
                )
        return self.snapshot(workflow_id)

    def on_evaluation_processing_failed(
        self, workflow_id: str, error: str
    ) -> dict[str, Any]:
        """Leave EVALUATING after permanent/repeated local scoring failures."""
        workflow = self.store.get_workflow(workflow_id)
        if workflow["state"] != WorkflowState.EVALUATING.value:
            return self.snapshot(workflow_id)
        eval_plan = EvaluationPlan.model_validate(workflow["evaluation_plan"])
        safe_error = sanitize_error(error)
        self.store.transition_bundle(
            workflow_id,
            "evaluation_failed",
            {"status": "scoring_failed", "error": safe_error},
            message=(
                "A/B 推理产物已保留，但本地评分未能完成。"
                "可修复 judge/评分文件后重新批准评测，或跳过本轮。"
            ),
            approvals=self._scoring_recovery_approval_specs(eval_plan),
        )
        return self.snapshot(workflow_id)

    def plan_iteration_action(self, action: dict[str, Any]) -> None:
        workflow_id = action["workflow_id"]
        workflow = self.store.get_workflow(workflow_id)
        if workflow["state"] != WorkflowState.DIAGNOSIS_READY.value:
            self.store.complete_action(
                action["action_id"], action["lease_token"]
            )
            return
        diagnosis = EvaluationDiagnosis.model_validate(workflow["diagnosis"])
        previous_data_plan = self._approved_iteration_data_plan(
            workflow_id,
            workflow["iteration"],
            DataPlan.model_validate(workflow["data_plan"]),
        )
        plan = self.planner.create_iteration_plan(
            TrainingObjective.model_validate(workflow["objective"]),
            previous_data_plan,
            TrainingPlan.model_validate(workflow["training_plan"]),
            diagnosis,
            action["payload"]["comparison"],
        )
        self.store.publish_iteration_plan(
            workflow_id=workflow_id,
            action_id=action["action_id"],
            lease_token=action["lease_token"],
            plan=plan,
            summary="根据诊断开始下一轮数据与训练迭代",
        )

    def _approved_iteration_data_plan(
        self, workflow_id: str, iteration: int, fallback: DataPlan
    ) -> DataPlan:
        event_type = "requirements_completed" if iteration == 0 else "iteration_started"
        for event in self.store.list_recent_events(workflow_id, event_type, limit=100):
            payload = event["payload"]
            if iteration and payload.get("iteration") != iteration:
                continue
            candidate = payload.get("data_plan")
            if candidate:
                return DataPlan.model_validate(candidate)
        return fallback

    def explain_diagnosis_action(self, action: dict[str, Any]) -> None:
        workflow = self.store.get_workflow(action["workflow_id"])
        diagnosis = EvaluationDiagnosis.model_validate(workflow["diagnosis"])
        explanation = self.planner.explain_diagnosis(
            diagnosis, action["payload"]["comparison"]
        )
        if not isinstance(explanation, str) or not explanation.strip():
            raise ValueError("diagnosis explanation was empty")
        if not self.store.publish_diagnosis_explanation(
            workflow_id=action["workflow_id"],
            action_id=action["action_id"],
            lease_token=action["lease_token"],
            explanation=explanation.strip(),
        ):
            raise RuntimeError("diagnosis explanation lease was lost")

    @classmethod
    def _primary_improvement(
        cls,
        baseline_summary: dict[str, Any],
        candidate_summary: dict[str, Any],
        primary_metric: str,
    ) -> float:
        baseline = cls._diagnosis_metrics(baseline_summary, primary_metric)["primary"]
        candidate = cls._diagnosis_metrics(candidate_summary, primary_metric)["primary"]
        return candidate - baseline

    @staticmethod
    def _evaluation_evidence_issues(
        baseline: dict[str, Any],
        candidate: dict[str, Any],
        paired: dict[str, Any],
        task_types: list[str],
        primary_metric: str,
        critical_slices: list[str],
        source_task_types: list[str] | None = None,
    ) -> list[str]:
        issues: list[str] = []
        for label, summary in (("baseline", baseline), ("candidate", candidate)):
            if not isinstance(summary, dict) or summary.get("error"):
                issues.append(f"{label}_summary_error")
                continue
            for task_type in task_types:
                task_summary = summary.get(task_type)
                if not isinstance(task_summary, dict):
                    issues.append(f"{label}_{task_type}_summary_missing")
                    continue
                if not isinstance(task_summary.get("n"), int) or task_summary["n"] <= 0:
                    issues.append(f"{label}_{task_type}_sample_count_missing")
                primary_value = task_summary.get(primary_metric)
                if (
                    not isinstance(primary_value, (int, float))
                    or isinstance(primary_value, bool)
                    or not math.isfinite(float(primary_value))
                ):
                    issues.append(f"{label}_{task_type}_{primary_metric}_missing")
        paired_n = paired.get("n")
        if (
            not isinstance(paired_n, int)
            or isinstance(paired_n, bool)
            or paired_n <= 0
        ):
            issues.append("paired_sample_count_missing")
        else:
            paired_score_n = paired.get("paired_score_n")
            if (
                not isinstance(paired_score_n, int)
                or isinstance(paired_score_n, bool)
                or paired_score_n != paired_n
            ):
                issues.append("paired_business_score_incomplete")
        for label in ("baseline", "candidate"):
            missing = paired.get(f"{label}_missing")
            if isinstance(missing, int) and missing > 0:
                issues.append(f"{label}_paired_rows_missing")
        missing_critical = set(paired.get("missing_critical_slices") or [])
        missing_critical.update(
            tag
            for tag in critical_slices
            if not isinstance((paired.get("slices") or {}).get(tag), dict)
            or int((paired.get("slices") or {}).get(tag, {}).get("n") or 0) <= 0
        )
        issues.extend(
            f"critical_slice_{tag}_missing" for tag in sorted(missing_critical)
        )
        slices = paired.get("slices") or {}
        for task_type in source_task_types or []:
            task_slice = slices.get(task_type)
            if not isinstance(task_slice, dict) or int(task_slice.get("n") or 0) <= 0:
                issues.append(f"task_slice_{task_type}_missing")
        return sorted(set(issues))

    def _iteration_stop_reason(
        self,
        workflow_id: str,
        *,
        current_improvement: float,
        threshold: float,
        iteration: int,
        train_job_id: str | None,
    ) -> str | None:
        if iteration >= 3:
            return "maximum iteration count reached"
        prior = []
        seen_rounds: set[tuple[int, str]] = set()
        for event in self.store.list_recent_events(
            workflow_id, "evaluation_completed", limit=20
        ):
            payload = event["payload"]
            previous_job = payload.get("train_job_id")
            previous_iteration = payload.get("iteration")
            if (
                not isinstance(previous_job, str)
                or not isinstance(previous_iteration, int)
                or previous_job == train_job_id
            ):
                continue
            round_key = (previous_iteration, previous_job)
            if round_key in seen_rounds:
                continue
            seen_rounds.add(round_key)
            value = payload.get("primary_improvement")
            if isinstance(value, (int, float)):
                prior.append(float(value))
                break
        recent = [*prior, current_improvement]
        if len(recent) >= 2 and all(value < threshold for value in recent[-2:]):
            return "two consecutive evaluations missed min_improvement"
        return None

    @staticmethod
    def _diagnosis_metrics(
        summary: dict[str, Any], primary_metric: str
    ) -> dict[str, Any]:
        primary_values = []
        invalid_rates = []
        metric_values: dict[str, list[float]] = {}
        for task_summary in summary.values():
            if not isinstance(task_summary, dict):
                continue
            value = task_summary.get(primary_metric)
            if isinstance(value, (int, float)):
                primary_values.append(float(value))
            invalid = task_summary.get("invalid_rate")
            if isinstance(invalid, (int, float)):
                invalid_rates.append(float(invalid))
            for key, metric_value in task_summary.items():
                if isinstance(metric_value, (int, float)) and not isinstance(
                    metric_value, bool
                ):
                    metric_values.setdefault(key, []).append(float(metric_value))
        return {
            "primary": (
                sum(primary_values) / len(primary_values) if primary_values else 0.0
            ),
            "invalid_rate": max(invalid_rates, default=0.0),
            "metrics": {
                key: sum(values) / len(values)
                for key, values in metric_values.items()
            },
        }

    @staticmethod
    def _safe_training_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
        safe_points = []
        for point in metrics.get("points") or []:
            if not isinstance(point, dict):
                continue
            safe = {
                key: value
                for key, value in point.items()
                if key in {"step", "loss", "eval_loss", "epoch", "lr"}
                and isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
            }
            if safe:
                safe_points.append(safe)
        safe_metrics = {"points": safe_points[-400:]}
        for key in ("total_steps", "percentage"):
            value = metrics.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if math.isfinite(float(value)):
                    safe_metrics[key] = value
        return safe_metrics

    @staticmethod
    def _training_trends(metrics: dict[str, Any]) -> dict[str, str]:
        points = metrics.get("points") or []
        train = [float(row["loss"]) for row in points if "loss" in row]
        validation = [
            float(row["eval_loss"]) for row in points if "eval_loss" in row
        ]

        def direction(values: list[float], *, validation_loss: bool = False) -> str:
            if len(values) < 2:
                return "unknown"
            scale = max(abs(values[0]), 1e-9)
            delta = (values[-1] - values[0]) / scale
            if abs(delta) <= 0.02:
                return "flat"
            if delta < 0:
                return "better" if validation_loss else "down"
            return "worse" if validation_loss else "up"

        return {
            "loss_trend": direction(train),
            "validation_trend": direction(validation, validation_loss=True),
        }

    def _build_eval_request(
        self, workflow: dict[str, Any], plan: EvaluationPlan
    ) -> EvalRequest:
        objective = TrainingObjective.model_validate(workflow["objective"])
        if plan.execution_request is not None:
            current_baseline = self._resolve_baseline_model(
                objective, plan.baseline_name
            )
            current_hash = plan_hash(current_baseline.model_dump(mode="json"))
            if not plan.baseline_source_hash or current_hash != plan.baseline_source_hash:
                raise ValueError(
                    "baseline model/template fingerprint changed after approval"
                )
            current_candidate = self._resolve_candidate_model(
                objective, workflow["train_job_id"], plan.candidate_name
            )
            candidate_hash = plan_hash(
                current_candidate.model_dump(mode="json")
            )
            if (
                not plan.candidate_source_hash
                or candidate_hash != plan.candidate_source_hash
            ):
                raise ValueError(
                    "candidate model/template fingerprint changed after approval"
                )
            return plan.execution_request
        baseline = self._resolve_baseline_model(objective, plan.baseline_name)
        models = [
            baseline,
            self._resolve_candidate_model(
                objective, workflow["train_job_id"], plan.candidate_name
            ),
        ]
        return EvalRequest(
            models=models,
            task_types=plan.task_types,
            gpus=plan.gpus,
        )

    def _resolve_baseline_model(
        self, objective: TrainingObjective, baseline_name: str
    ) -> ModelUnderTest:
        if objective.baseline.kind == "train_job":
            resolved = self.tools.resolve_train_job(objective.baseline.train_job_id)
            if not resolved.get("model_name_or_path"):
                raise ValueError("baseline training job has no base model")
            if not str(resolved["model_name_or_path"]).startswith("/"):
                raise ValueError("baseline training job base model path is not absolute")
            adapter_path = resolved.get("adapter_path") or None
            if adapter_path is not None and not str(adapter_path).startswith("/"):
                raise ValueError("baseline training job adapter path is not absolute")
            baseline_template = resolved.get("template") or objective.template
            if baseline_template != objective.template:
                raise ValueError(
                    "baseline training job template differs from the locked "
                    "evaluation template"
                )
            return ModelUnderTest(
                name=baseline_name,
                model_name_or_path=resolved["model_name_or_path"],
                adapter_path=adapter_path,
                template=objective.template,
            )
        return ModelUnderTest(
            name=baseline_name,
            model_name_or_path=objective.base_model_path,
            template=objective.template,
        )

    def _resolve_candidate_model(
        self,
        objective: TrainingObjective,
        train_job_id: str,
        candidate_name: str,
    ) -> ModelUnderTest:
        resolved = self.tools.resolve_train_job(train_job_id)
        base = resolved.get("model_name_or_path")
        adapter = resolved.get("adapter_path")
        template = resolved.get("template") or objective.template
        if base != objective.base_model_path:
            raise ValueError("candidate training job base model differs from objective")
        if template != objective.template:
            raise ValueError("candidate training job template differs from objective")
        if not isinstance(adapter, str) or not adapter.startswith("/"):
            raise ValueError("candidate training job adapter path is not absolute")
        return ModelUnderTest(
            name=candidate_name,
            model_name_or_path=base,
            adapter_path=adapter,
            template=template,
        )

    def _stale_training_approvals(self, workflow_id: str) -> None:
        for approval in self.store.list_pending_approvals(workflow_id):
            if approval["action"] == ActionKind.START_TRAINING.value:
                self.store.mark_approval_stale(approval["approval_id"])

    def _create_training_approval(
        self,
        workflow_id: str,
        plan: TrainingPlan,
        report: PreflightReport,
    ) -> None:
        self.store.create_approval(
            workflow_id,
            ActionKind.START_TRAINING,
            plan,
            self._training_plan_summary(plan, report),
            decision_warnings=self._warning_fingerprints(report),
        )

    @staticmethod
    def _evaluation_approval_specs(plan: EvaluationPlan) -> list[dict[str, Any]]:
        return [
            {
                "action": ActionKind.START_EVALUATION,
                "plan": plan,
                "summary": "运行基座与候选模型 A/B 评测",
            },
            {
                "action": ActionKind.SKIP_EVALUATION,
                "plan": plan,
                "summary": "跳过本轮 A/B 评测并结束",
            },
        ]

    @staticmethod
    def _scoring_recovery_approval_specs(
        plan: EvaluationPlan,
    ) -> list[dict[str, Any]]:
        return [
            {
                "action": ActionKind.RETRY_SCORING,
                "plan": plan,
                "summary": "复用现有推理产物并重新执行本地评分",
            },
            {
                "action": ActionKind.SKIP_EVALUATION,
                "plan": plan,
                "summary": "跳过本轮 A/B 评测并结束",
            },
        ]

    @staticmethod
    def _warning_fingerprints(report: PreflightReport) -> list[str]:
        return sorted(
            plan_hash(
                {
                    "name": check.name,
                    "summary": check.summary,
                    "remediation": check.remediation,
                }
            )
            for check in report.checks
            if check.status == "warn"
        )

    @staticmethod
    def _objective_binding_warning(workflow: dict[str, Any]) -> list[str]:
        objective_hash = workflow.get("objective_hash")
        if not objective_hash:
            raise ApprovalConflict("requirements were not explicitly confirmed")
        return [f"objective_hash:{objective_hash}"]

    @staticmethod
    def _data_plan_summary(plan: DataPlan) -> str:
        task_types = ", ".join(item.task_type for item in plan.items)
        records = sum(item.config.count for item in plan.items)
        return f"生成 {records} 条 {task_types} 训练数据并冻结评测集"

    @staticmethod
    def _training_plan_summary(
        plan: TrainingPlan, report: PreflightReport
    ) -> str:
        return (
            f"在 GPU {plan.gpus} 上启动 {plan.config.method.stage.upper()} 训练，"
            f"预计 {plan.estimated_steps} 步；预检状态 {report.status}"
        )
