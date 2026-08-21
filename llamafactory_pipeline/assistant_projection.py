"""Stable eight-stage UI projection for assistant workflows."""

from __future__ import annotations

from typing import Any

from .assistant_schema import ArtifactRef, StepProgress, WorkflowStep


STEP_DEFINITIONS = (
    ("requirements", "需求理解与确认"),
    ("data_plan", "数据策略规划"),
    ("data_build", "数据构建"),
    ("data_review", "数据质量复核"),
    ("train_plan", "训练方案与预检"),
    ("training", "模型训练"),
    ("evaluation", "A/B 评测"),
    ("diagnosis", "诊断与下一轮决策"),
)

STATE_STEP = {
    "collecting_requirements": 0,
    "requirements_review": 0,
    "data_plan_preparing": 1,
    "data_plan_ready": 1,
    "data_generating": 2,
    "data_review": 3,
    "train_plan_ready": 4,
    "preflight_blocked": 4,
    "train_ready": 4,
    "training": 5,
    "train_failed": 5,
    "ab_plan_ready": 6,
    "evaluating": 6,
    "diagnosis_ready": 7,
    "completed": 7,
}

CONFIRMATION_STATES = {
    "requirements_review",
    "data_plan_ready",
    "train_ready",
    "ab_plan_ready",
    "diagnosis_ready",
}

TERMINAL_STATES = {"cancelled", "completed"}


def _latest_event(events: list[dict[str, Any]], event_type: str) -> dict[str, Any] | None:
    for event in reversed(events):
        if event.get("event_type") == event_type:
            return event
    return None


def _cancel_step(workflow: dict[str, Any]) -> int:
    request = workflow.get("cancel_request") or {}
    targets = request.get("targets") or []
    target_steps = {
        "datagen": 2,
        "training": 5,
        "evaluation": 6,
    }
    indices = [target_steps[row.get("kind")] for row in targets if row.get("kind") in target_steps]
    if indices:
        return max(indices)
    if workflow.get("eval_id"):
        return 6
    if workflow.get("train_job_id"):
        return 5
    if workflow.get("datagen_jobs"):
        return 2
    if workflow.get("data_plan"):
        return 1
    return 0


def _current_step(workflow: dict[str, Any]) -> int:
    state = str(workflow.get("state") or "collecting_requirements")
    if state in {"cancelling", "cancelled"}:
        return _cancel_step(workflow)
    return STATE_STEP.get(state, 0)


def _step_status(state: str, index: int, current: int) -> str:
    if state == "cancelling":
        if index < current:
            return "succeeded"
        return "cancelling" if index == current else "pending"
    if state == "cancelled":
        if index < current:
            return "succeeded"
        return "cancelled" if index == current else "pending"
    if state == "completed":
        return "succeeded"
    if index < current:
        return "succeeded"
    if index > current:
        return "pending"
    if state in CONFIRMATION_STATES:
        return "needs_confirmation"
    if state == "preflight_blocked":
        return "blocked"
    if state == "train_failed":
        return "failed"
    return "active"


def _approval_actions(approvals: list[dict[str, Any]]) -> list[str]:
    actions: list[str] = []
    for approval in approvals:
        action = approval.get("action")
        if isinstance(action, str) and action not in actions:
            actions.append(action)
    return actions


def _actions_for_step(
    workflow: dict[str, Any],
    index: int,
    approvals: list[dict[str, Any]],
    events: list[dict[str, Any]],
) -> list[str]:
    state = str(workflow.get("state") or "")
    if state in TERMINAL_STATES or state == "cancelling":
        return []
    current = _current_step(workflow)
    if index != current:
        return []
    if state == "requirements_review":
        if workflow.get("confirmed_objective") and _latest_event(
            events, "data_plan_preparation_failed"
        ):
            return ["retry_data_plan", "revise_requirements", "cancel"]
        return ["confirm_requirements", "revise_requirements", "cancel"]
    actions = _approval_actions(approvals)
    if state == "preflight_blocked":
        actions.append("rerun_preflight")
    if state == "data_plan_preparing" and workflow.get("confirmed_objective"):
        failed = _latest_event(events, "data_plan_preparation_failed")
        if failed:
            actions.append("retry_data_plan")
    if "cancel" not in actions:
        actions.append("cancel")
    return actions


def _artifact(
    name: str,
    kind: str,
    *,
    download_url: str | None = None,
    preview_url: str | None = None,
    sha256: str | None = None,
) -> ArtifactRef:
    return ArtifactRef(
        name=name,
        kind=kind,
        download_url=download_url,
        preview_url=preview_url,
        sha256=sha256,
    )


def _artifacts(workflow: dict[str, Any], index: int) -> list[ArtifactRef]:
    result: list[ArtifactRef] = []
    if index == 0 and workflow.get("objective_hash"):
        result.append(
            _artifact(
                "已确认训练目标",
                "confirmed_objective",
                sha256=str(workflow["objective_hash"]),
            )
        )
    elif index == 1 and workflow.get("data_plan"):
        result.append(_artifact("数据构建方案", "data_plan"))
    elif index == 2:
        for row in workflow.get("datagen_jobs") or []:
            job_id = row.get("job_id")
            if not job_id:
                continue
            result.extend(
                [
                    _artifact(
                        f"{job_id} 数据",
                        "datagen_output",
                        download_url=f"/api/datagen/jobs/{job_id}/download",
                    ),
                    _artifact(
                        f"{job_id} 报告",
                        "datagen_report",
                        preview_url=f"/api/datagen/jobs/{job_id}/report",
                    ),
                    _artifact(
                        f"{job_id} 日志",
                        "datagen_log",
                        preview_url=f"/api/datagen/jobs/{job_id}/logs",
                    ),
                ]
            )
    elif index == 3 and workflow.get("dataset_profile"):
        profile = workflow["dataset_profile"]
        result.append(
            _artifact(
                str(profile.get("dataset_name") or "冻结数据集"),
                "dataset_profile",
                sha256=profile.get("sha256"),
            )
        )
    elif index == 4 and workflow.get("training_plan"):
        result.append(_artifact("训练参数方案", "training_plan"))
        if workflow.get("preflight"):
            result.append(_artifact("环境预检报告", "preflight_report"))
    elif index == 5 and workflow.get("train_job_id"):
        job_id = workflow["train_job_id"]
        result.extend(
            [
                _artifact(
                    f"{job_id} 日志",
                    "training_log",
                    preview_url=f"/api/jobs/{job_id}/logs",
                ),
                _artifact(
                    f"{job_id} 指标",
                    "training_metrics",
                    preview_url=f"/api/jobs/{job_id}/metrics",
                ),
                _artifact(
                    f"{job_id} Checkpoints",
                    "checkpoint_index",
                    preview_url=f"/api/jobs/{job_id}/checkpoints",
                ),
            ]
        )
    elif index == 6 and workflow.get("eval_id"):
        eval_id = workflow["eval_id"]
        result.extend(
            [
                _artifact(
                    f"{eval_id} 日志",
                    "evaluation_log",
                    preview_url=f"/api/eval/jobs/{eval_id}/logs",
                ),
                _artifact(
                    f"{eval_id} A/B 报告",
                    "evaluation_report",
                    preview_url=f"/api/eval/jobs/{eval_id}/report",
                ),
            ]
        )
    elif index == 7 and workflow.get("diagnosis"):
        result.append(_artifact("诊断与对照分析", "diagnosis_report"))
    return result


def _progress(events: list[dict[str, Any]], index: int) -> StepProgress | None:
    event_type = {2: "datagen_progress", 5: "training_progress", 6: "evaluation_progress"}.get(index)
    event = _latest_event(events, event_type) if event_type else None
    if not event:
        return None
    payload = event.get("payload") or {}
    if index == 2:
        current = payload.get("accepted")
        target = payload.get("target")
        percentage = (
            round(float(current) * 100 / float(target), 2)
            if isinstance(current, (int, float))
            and isinstance(target, (int, float))
            and target > 0
            else None
        )
        details = {
            key: payload[key]
            for key in ("attempts", "acceptance_rate", "rejects")
            if key in payload
        }
        return StepProgress(
            current=current,
            target=target,
            percentage=percentage,
            eta_seconds=payload.get("eta_seconds"),
            updated_at=event.get("created_at"),
            details=details,
        )
    current = payload.get("step") or payload.get("current")
    target = payload.get("total_steps") or payload.get("target")
    return StepProgress(
        current=current,
        target=target,
        percentage=payload.get("percentage"),
        eta_seconds=payload.get("eta_seconds"),
        updated_at=event.get("created_at"),
        details={
            key: value
            for key, value in payload.items()
            if key not in {"step", "current", "total_steps", "target", "percentage", "eta_seconds"}
        },
    )


def _summary(workflow: dict[str, Any], index: int) -> str:
    if index == 0:
        draft = workflow.get("requirement_draft") or {}
        objective = workflow.get("confirmed_objective") or workflow.get("objective") or {}
        return str(
            draft.get("assistant_reply")
            or objective.get("goal")
            or "请说明业务场景、当前问题和期望行为。"
        )
    if index == 1:
        plan = workflow.get("data_plan") or {}
        return str(plan.get("rationale") or "尚未生成数据构建方案。")
    if index == 2:
        jobs = workflow.get("datagen_jobs") or []
        tasks = sorted({str(row.get("task_type")) for row in jobs if row.get("task_type")})
        return (
            f"{len(jobs)} 个数据任务"
            + (f"：{', '.join(tasks)}" if tasks else "")
            if jobs
            else "等待数据构建任务。"
        )
    if index == 3:
        profile = workflow.get("dataset_profile") or {}
        if profile:
            return (
                f"训练 {profile.get('n_records', 0)} 条，"
                f"冻结评测 {profile.get('holdout_records', 0)} 条。"
            )
        return "等待数据质量、覆盖率与冻结 holdout 检查。"
    if index == 4:
        plan = workflow.get("training_plan") or {}
        if plan:
            return (
                f"预计 {plan.get('estimated_steps', 0)} steps，"
                f"约 {plan.get('estimated_vram_gb', 0)} GB 显存。"
            )
        return "等待训练参数推荐和环境预检。"
    if index == 5:
        job_id = workflow.get("train_job_id")
        return f"训练任务 {job_id}" if job_id else "训练尚未启动。"
    if index == 6:
        eval_id = workflow.get("eval_id")
        plan = workflow.get("evaluation_plan") or {}
        tasks = plan.get("task_types") or []
        if eval_id:
            return f"评测任务 {eval_id}"
        if tasks:
            return "冻结评测任务：" + ", ".join(str(item) for item in tasks)
        return "等待候选模型与基线的成对 A/B 评测。"
    diagnosis = workflow.get("diagnosis") or {}
    return str(diagnosis.get("summary") or "等待评测证据后给出对照分析与下一轮建议。")


def _decisions(workflow: dict[str, Any], index: int) -> list[dict[str, Any]]:
    if index == 0:
        return [
            {"kind": "assumption", "value": value}
            for value in (workflow.get("requirement_draft") or {}).get("assumptions", [])
        ]
    if index == 1:
        plan = workflow.get("data_plan") or {}
        return [
            {"kind": "risk", "value": value}
            for value in plan.get("risks", [])
        ]
    if index == 4:
        return list((workflow.get("training_plan") or {}).get("decisions") or [])
    if index == 7:
        diagnosis = workflow.get("diagnosis") or {}
        result: list[dict[str, Any]] = []
        for kind in ("next_data_changes", "next_training_changes"):
            for value in diagnosis.get(kind, []) or []:
                result.append({"kind": kind, "value": value})
        return result
    return []


def build_workflow_steps(
    workflow: dict[str, Any],
    events: list[dict[str, Any]],
    pending_approvals: list[dict[str, Any]],
) -> list[WorkflowStep]:
    """Project storage details into a stable, safe UI contract."""
    state = str(workflow.get("state") or "collecting_requirements")
    current = _current_step(workflow)
    created_at = workflow.get("created_at")
    updated_at = workflow.get("updated_at")
    steps: list[WorkflowStep] = []
    for index, (key, title) in enumerate(STEP_DEFINITIONS):
        status = _step_status(state, index, current)
        if state == "completed" and _latest_event(
            events, "evaluation_skipped"
        ) and index in {6, 7}:
            status = "skipped"
        issues: list[dict[str, Any]] = []
        if (
            index == 0
            and current > 0
            and not workflow.get("objective_hash")
        ):
            issues.append(
                {
                    "code": "legacy_requirement_gate_missing",
                    "message": "该历史流程没有显式需求确认证据，仅作兼容展示。",
                }
            )
        steps.append(
            WorkflowStep(
                key=key,
                sequence=index + 1,
                title=title,
                status=status,
                started_at=created_at if index <= current else None,
                updated_at=updated_at if index == current else None,
                finished_at=updated_at if status in {"succeeded", "cancelled"} else None,
                summary=_summary(workflow, index),
                decisions=_decisions(workflow, index),
                progress=_progress(events, index),
                artifacts=_artifacts(workflow, index),
                issues=issues,
                actions=_actions_for_step(
                    workflow, index, pending_approvals, events
                ),
            )
        )
    return steps


def available_actions(steps: list[WorkflowStep]) -> list[str]:
    """Return de-duplicated actions for clients that do not render steps yet."""
    result: list[str] = []
    for step in steps:
        for action in step.actions:
            if action not in result:
                result.append(action)
    return result
