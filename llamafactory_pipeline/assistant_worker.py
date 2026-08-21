"""Lease-based one-shot monitor for assistant workflows."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from .assistant_service import sanitize_error

MILESTONES = (10, 25, 50, 75, 90, 100)
UNCERTAIN_REMOTE_STATUSES = {"unknown", "not_found"}


class EvaluationScoringError(RuntimeError):
    """A terminal remote evaluation could not be scored locally."""


def classify_training_failure(log_tail: str) -> str:
    text = (log_tail or "").lower()
    if any(
        pattern in text
        for pattern in ("out of memory", "cuda oom", "cublas_status_alloc_failed")
    ):
        return "oom"
    if any(
        pattern in text
        for pattern in ("nccl", "torchrun", "distributed", "watchdog timeout")
    ):
        return "distributed"
    if any(pattern in text for pattern in ("tokenizer", "sentencepiece", "vocab")):
        return "tokenizer"
    if any(
        pattern in text
        for pattern in ("dataset", "data_files", "column", "jsondecode")
    ):
        return "dataset"
    if "traceback" in text:
        return "runtime"
    return "unknown"


@dataclass
class MonitorMemory:
    reported_milestones: list[int] = field(default_factory=list)
    last_step: int | None = None
    last_step_at: str | None = None
    last_eta_seconds: int | None = None
    recent_losses: list[float] = field(default_factory=list)
    recent_step_rates: list[float] = field(default_factory=list)
    calibrated_eta_seconds: int | None = None
    stalled_active: bool = False
    gpu_pressure_active: bool = False
    loss_spike_active: bool = False
    loss_invalid_active: bool = False


@dataclass(frozen=True)
class WorkerSummary:
    processed: int
    failed: int


def _parse_eta(value: Any) -> int | None:
    if isinstance(value, (int, float)) and value > 0:
        return int(value)
    if not isinstance(value, str) or not value.strip():
        return None
    parts = value.strip().split(":")
    try:
        if len(parts) == 3:
            hours, minutes, seconds = (int(part) for part in parts)
            return hours * 3600 + minutes * 60 + seconds
        if len(parts) == 2:
            minutes, seconds = (int(part) for part in parts)
            return minutes * 60 + seconds
    except ValueError:
        return None
    return None


def _latest_point(observation: dict[str, Any]) -> dict[str, Any]:
    points = (observation.get("metrics") or {}).get("points") or []
    return points[-1] if points and isinstance(points[-1], dict) else {}


def _gpu_value(row: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = row.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def reduce_training_observation(
    memory: MonitorMemory,
    observation: dict[str, Any],
    *,
    now: str,
) -> tuple[list[dict[str, Any]], MonitorMemory]:
    """Reduce noisy polls to durable progress and anomaly events."""
    metrics = observation.get("metrics") or {}
    point = _latest_point(observation)
    total = metrics.get("total_steps")
    current_step = point.get("step")
    if not isinstance(current_step, (int, float)):
        current_step = None
    else:
        current_step = int(current_step)
    if not isinstance(total, (int, float)) or total <= 0:
        total = None
    else:
        total = int(total)
    percentage = metrics.get("percentage")
    if not isinstance(percentage, (int, float)) and current_step is not None and total:
        percentage = current_step * 100 / total
    percentage = float(percentage) if isinstance(percentage, (int, float)) else 0.0
    previous_percentage = (
        memory.last_step * 100 / total
        if memory.last_step is not None and total
        else 0.0
    )

    events: list[dict[str, Any]] = []
    already = set(memory.reported_milestones)
    crossed = [
        milestone
        for milestone in MILESTONES
        if previous_percentage < milestone <= percentage and milestone not in already
    ]
    reported = sorted(already | set(crossed))
    if crossed:
        eta_seconds = _parse_eta(metrics.get("remaining_time"))
        events.append(
            {
                "event_type": "training_progress",
                "payload": {
                    "milestone": max(crossed),
                    "percentage": round(percentage, 2),
                    "step": current_step,
                    "total_steps": total,
                    "remaining_time": metrics.get("remaining_time", ""),
                    "eta_seconds": eta_seconds,
                    "loss": point.get("loss"),
                    "epoch": point.get("epoch"),
                    "learning_rate": point.get("lr"),
                },
            }
        )

    eta_seconds = _parse_eta(metrics.get("remaining_time"))
    if (
        memory.last_eta_seconds
        and eta_seconds
        and abs(eta_seconds - memory.last_eta_seconds) / memory.last_eta_seconds > 0.20
    ):
        events.append(
            {
                "event_type": "training_eta_changed",
                "payload": {
                    "previous_eta_seconds": memory.last_eta_seconds,
                    "eta_seconds": eta_seconds,
                },
            }
        )

    gpus = [row for row in observation.get("gpus", []) if isinstance(row, dict)]
    utilizations = [
        value
        for row in gpus
        if (value := _gpu_value(row, "util", "utilization_pct")) is not None
    ]
    last_step_at = memory.last_step_at
    stalled_active = False
    step_rates = [
        float(value)
        for value in memory.recent_step_rates
        if isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) > 0
    ][-20:]
    if current_step is not None and (
        memory.last_step is None or current_step > memory.last_step
    ):
        if memory.last_step is not None and memory.last_step_at:
            elapsed = (
                datetime.fromisoformat(now)
                - datetime.fromisoformat(memory.last_step_at)
            ).total_seconds()
            if elapsed > 0:
                step_rates = [
                    *step_rates,
                    (current_step - memory.last_step) / elapsed,
                ][-20:]
        last_step_at = now
    elif current_step == memory.last_step and memory.last_step_at:
        idle_seconds = (
            datetime.fromisoformat(now) - datetime.fromisoformat(memory.last_step_at)
        ).total_seconds()
        if idle_seconds >= 600 and max(utilizations or [0]) < 10:
            stalled_active = True
            if not memory.stalled_active:
                events.append(
                    {
                        "event_type": "training_stalled",
                        "payload": {
                            "step": current_step,
                            # Stable episode identity keeps append_event_once
                            # effective if the process crashes before memory is
                            # rescheduled.
                            "stalled_since": memory.last_step_at,
                        },
                    }
                )

    loss = point.get("loss")
    recent = [
        float(value)
        for value in memory.recent_losses
        if isinstance(value, (int, float)) and math.isfinite(value) and value > 0
    ][-20:]
    if isinstance(loss, (int, float)):
        numeric_loss = float(loss)
        loss_invalid_active = not math.isfinite(numeric_loss)
        loss_spike_active = False
        if not math.isfinite(numeric_loss):
            if not memory.loss_invalid_active:
                events.append(
                    {
                        "event_type": "training_loss_invalid",
                        "payload": {"step": current_step, "loss": str(loss)},
                    }
                )
        elif numeric_loss > 0:
            if len(recent) >= 20 and numeric_loss > 3 * statistics.median(recent):
                loss_spike_active = True
                if not memory.loss_spike_active:
                    events.append(
                        {
                            "event_type": "training_loss_spike",
                            "payload": {
                                "step": current_step,
                                "loss": numeric_loss,
                                "previous_median": statistics.median(recent),
                            },
                        }
                    )
            recent = [*recent, numeric_loss][-20:]
    else:
        loss_invalid_active = False
        loss_spike_active = False

    pressure = []
    for row in gpus:
        used = _gpu_value(row, "mem_used", "memory_used_mb")
        total_memory = _gpu_value(row, "mem_total", "memory_total_mb")
        temperature = _gpu_value(row, "temperature", "temperature_c")
        ratio = used / total_memory if used is not None and total_memory else None
        if (ratio is not None and ratio >= 0.95) or (
            temperature is not None and temperature >= 85
        ):
            pressure.append(
                {
                    "index": row.get("index"),
                    "memory_ratio": round(ratio, 4) if ratio is not None else None,
                    "temperature_c": temperature,
                }
            )
    if pressure and not memory.gpu_pressure_active:
        events.append(
            {"event_type": "training_gpu_pressure", "payload": {"gpus": pressure}}
        )

    calibrated = memory.calibrated_eta_seconds
    if current_step is not None and current_step >= 20 and total and step_rates:
        median_rate = statistics.median(step_rates)
        calibrated = max(0, round((total - current_step) / median_rate))
    updated = MonitorMemory(
        reported_milestones=reported,
        last_step=current_step if current_step is not None else memory.last_step,
        last_step_at=last_step_at,
        last_eta_seconds=eta_seconds or memory.last_eta_seconds,
        recent_losses=recent,
        recent_step_rates=step_rates,
        calibrated_eta_seconds=calibrated,
        stalled_active=stalled_active,
        gpu_pressure_active=bool(pressure),
        loss_spike_active=loss_spike_active,
        loss_invalid_active=loss_invalid_active,
    )
    return events, updated


def _status_value(observation: dict[str, Any]) -> str:
    status = observation.get("status", "unknown")
    if isinstance(status, dict):
        status = status.get("status", "unknown")
    return str(status)


def _schedule(
    store,
    action: dict[str, Any],
    due_at: datetime,
    payload: dict[str, Any],
) -> None:
    if not store.reschedule_action(
        action["action_id"], action["lease_token"], due_at, payload
    ):
        raise RuntimeError("monitor action lease was lost before reschedule")


def _schedule_uncertain(
    store,
    action: dict[str, Any],
    now: datetime,
    payload: dict[str, Any],
    normal_delay_seconds: int,
) -> None:
    """Persist consecutive unknown polls without ending the monitor chain."""
    unknown_polls = int(action["payload"].get("unknown_polls") or 0) + 1
    if unknown_polls == 5:
        store.append_event_once(
            action["workflow_id"],
            "monitor_needs_attention",
            {
                "action": action["action"],
                "reason": "remote task status remained unknown for five polls",
            },
        )
    delay = 900 if unknown_polls >= 5 else normal_delay_seconds
    _schedule(
        store,
        action,
        now + timedelta(seconds=delay),
        {**payload, "unknown_polls": unknown_polls},
    )


def reduce_datagen_observation(
    memory: dict[str, Any],
    observations: list[dict[str, Any]],
    *,
    now: str,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Aggregate data-generation polls and emit only changed/heartbeat progress."""
    accepted = 0
    target = 0
    attempts = 0
    rejects: dict[str, int] = {}
    has_progress = False
    for row in observations:
        if not isinstance(row, dict):
            continue
        for key in ("accepted", "target", "attempts"):
            value = row.get(key)
            if isinstance(value, (int, float)) and value >= 0:
                has_progress = True
                if key == "accepted":
                    accepted += int(value)
                elif key == "target":
                    target += int(value)
                else:
                    attempts += int(value)
        for reason, count in (row.get("rejects") or {}).items():
            if isinstance(reason, str) and isinstance(count, (int, float)) and count >= 0:
                rejects[reason] = rejects.get(reason, 0) + int(count)

    if not has_progress:
        return None, dict(memory)

    previous_accepted = memory.get("last_accepted")
    previous_at = memory.get("last_observed_at")
    rates = [
        float(value)
        for value in memory.get("rates", [])
        if isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) > 0
    ][-10:]
    if isinstance(previous_accepted, (int, float)) and isinstance(previous_at, str):
        try:
            elapsed = (
                datetime.fromisoformat(now) - datetime.fromisoformat(previous_at)
            ).total_seconds()
        except ValueError:
            elapsed = 0
        delta = accepted - int(previous_accepted)
        if elapsed > 0 and delta > 0:
            rates = [*rates, delta / elapsed][-10:]
    eta_seconds = None
    if target > accepted and rates:
        eta_seconds = int(round((target - accepted) / statistics.median(rates)))
    acceptance_rate = accepted / attempts if attempts > 0 else None
    payload = {
        "accepted": accepted,
        "target": target,
        "attempts": attempts,
        "acceptance_rate": acceptance_rate,
        "rejects": rejects,
        "eta_seconds": eta_seconds,
    }
    fingerprint = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    last_emitted_at = memory.get("last_emitted_at")
    heartbeat_due = False
    if isinstance(last_emitted_at, str):
        try:
            heartbeat_due = (
                datetime.fromisoformat(now) - datetime.fromisoformat(last_emitted_at)
            ).total_seconds() >= 300
        except ValueError:
            heartbeat_due = True
    event = None
    if fingerprint != memory.get("fingerprint") or heartbeat_due:
        event = {"event_type": "datagen_progress", "payload": payload}
        last_emitted_at = now
    return event, {
        "last_accepted": accepted,
        "last_observed_at": now,
        "last_emitted_at": last_emitted_at,
        "fingerprint": fingerprint,
        "rates": rates,
    }


def _dispatch_datagen(store, tools, service, action, now: datetime) -> None:
    workflow_id = action["workflow_id"]
    launches = action["payload"].get("launches", [])
    observations = tools.inspect_datagen(launches)
    event, progress_memory = reduce_datagen_observation(
        action["payload"].get("progress_memory") or {},
        observations,
        now=now.isoformat(),
    )
    if event:
        store.append_event_once(
            workflow_id, event["event_type"], event["payload"]
        )
    statuses = {_status_value(row) for row in observations}
    if not statuses or statuses & UNCERTAIN_REMOTE_STATUSES:
        _schedule_uncertain(
            store,
            action,
            now,
            {"launches": launches, "progress_memory": progress_memory},
            60,
        )
    elif "running" in statuses:
        due = now + timedelta(seconds=60)
        _schedule(
            store,
            action,
            due,
            {"launches": launches, "progress_memory": progress_memory},
        )
    else:
        service.on_datagen_terminal(workflow_id, observations)


def _record_training_history(
    store,
    workflow_id: str,
    observation: dict[str, Any],
    memory: MonitorMemory,
    now: datetime,
) -> None:
    try:
        workflow = store.get_workflow(workflow_id)
        plan = workflow.get("training_plan") or {}
        config = plan["config"]
        points = (observation.get("metrics") or {}).get("points") or []
        actual_steps = int(points[-1].get("step") or 0) if points else 0
        resume_path = config["train"].get("resume_from_checkpoint")
        resume_match = re.search(r"/checkpoint-(\d+)$", str(resume_path or ""))
        initial_step = int(resume_match.group(1)) if resume_match else 0
        comparable_steps = max(0, actual_steps - initial_step)
        training_events = [
            event
            for event in store.list_events(workflow_id, 0, limit=1000)
            if event["event_type"] == "training_started"
        ]
        started = datetime.fromisoformat(training_events[-1]["created_at"])
        duration = max(1, int((now - started).total_seconds()))
        gpu_rows = observation.get("gpus") or []
        preflight = workflow.get("preflight") or {}
        model_size = (preflight.get("model") or {}).get("parameter_billions")
        estimated_low = plan.get("estimated_hours_low")
        estimated_high = plan.get("estimated_hours_high")
        initial_eta = None
        if isinstance(estimated_low, (int, float)) and isinstance(
            estimated_high, (int, float)
        ):
            initial_eta = int((estimated_low + estimated_high) * 1800)
        store.record_training_run(
            train_job_id=workflow["train_job_id"],
            workflow_id=workflow_id,
            iteration=workflow["iteration"],
            stage=config["method"]["stage"],
            model_parameter_billions=model_size,
            gpu_names=[str(row.get("name", "GPU")) for row in gpu_rows],
            gpu_count=len(gpu_rows),
            cutoff_len=config["dataset"]["cutoff_len"],
            quantization_bit=config["model"].get("quantization_bit"),
            estimated_steps=plan["estimated_steps"],
            actual_steps=comparable_steps or None,
            initial_eta_seconds=initial_eta,
            calibrated_eta_seconds=memory.calibrated_eta_seconds,
            duration_seconds=duration,
            steps_per_second=(
                comparable_steps / duration if comparable_steps else None
            ),
            terminal_status=_status_value(observation),
        )
    except Exception as exc:
        store.append_event(
            workflow_id,
            "training_history_record_failed",
            {"error": sanitize_error(exc)},
        )


def _dispatch_training(store, tools, service, action, now: datetime) -> None:
    workflow_id = action["workflow_id"]
    payload = action["payload"]
    job_id = payload["job_id"]
    observation = tools.inspect_training(job_id)
    status = _status_value(observation)
    memory = MonitorMemory(**(payload.get("memory") or {}))
    if status in UNCERTAIN_REMOTE_STATUSES:
        _schedule_uncertain(
            store,
            action,
            now,
            {
                "job_id": job_id,
                "memory": asdict(memory),
                "monitor_started_at": (
                    payload.get("monitor_started_at") or action["created_at"]
                ),
            },
            60,
        )
        return
    events, updated = reduce_training_observation(
        memory, observation, now=now.isoformat()
    )
    for event in events:
        store.append_event_once(
            workflow_id, event["event_type"], event["payload"]
        )
    if status == "running":
        service.on_training_observation(
            workflow_id,
            {**observation, "job_id": job_id, "reducer_events": events},
        )
        monitor_started_at = payload.get("monitor_started_at") or action["created_at"]
        created = datetime.fromisoformat(monitor_started_at)
        elapsed = (now - created).total_seconds()
        anomaly = any(
            event["event_type"]
            not in {"training_progress", "training_eta_changed"}
            for event in events
        )
        delay = 60 if elapsed < 600 or (updated.last_eta_seconds or 10**9) < 600 or anomaly else 180
        due = now + timedelta(seconds=delay)
        _schedule(
            store,
            action,
            due,
            {
                "job_id": job_id,
                "memory": asdict(updated),
                "monitor_started_at": monitor_started_at,
            },
        )
        return
    if status == "succeeded":
        if not observation.get("output_evidence_verified", False):
            raise RuntimeError(
                "training output evidence is unavailable; inspection will be retried"
            )
        if not observation.get("output_verified", False):
            status = "failed"
            observation = {
                **observation,
                "status": {"status": "failed"},
                "failure_category": "output_artifact_missing",
                "error": "training exited successfully but adapter artifacts are missing",
            }
    if status in {"failed", "interrupted"} and not all(
        observation.get(flag, True)
        for flag in ("checkpoints_verified", "log_verified")
    ):
        raise RuntimeError(
            "training terminal evidence is incomplete; checkpoint/log inspection "
            "will be retried"
        )
    _record_training_history(store, workflow_id, observation, updated, now)
    if status in {"failed", "interrupted"}:
        category = observation.get("failure_category") or classify_training_failure(
            str(observation.get("log_tail") or "")
        )
        observation = {
            **observation,
            "failure_category": category,
            "error": observation.get("error") or f"training {status}: {category}",
        }
    service.on_training_observation(
        workflow_id,
        {**observation, "job_id": job_id, "reducer_events": events},
    )


def _dispatch_evaluation(store, tools, service, action, now: datetime) -> None:
    workflow_id = action["workflow_id"]
    eval_id = action["payload"]["eval_id"]
    observation = tools.inspect_evaluation(eval_id)
    status = _status_value(observation)
    if status in UNCERTAIN_REMOTE_STATUSES:
        _schedule_uncertain(
            store,
            action,
            now,
            {"eval_id": eval_id},
            120,
        )
    elif status == "running":
        due = now + timedelta(seconds=120)
        _schedule(
            store,
            action,
            due,
            {"eval_id": eval_id},
        )
    else:
        if status in {"failed", "interrupted"} and not observation.get(
            "log_verified", True
        ):
            raise RuntimeError(
                "evaluation terminal log evidence is incomplete; inspection "
                "will be retried"
            )
        if status == "succeeded":
            store.append_event_once(
                workflow_id,
                "evaluation_progress",
                {
                    "phase": "scoring",
                    "percentage": 75.0,
                    "eval_id": eval_id,
                },
            )
        try:
            service.on_evaluation_terminal(
                workflow_id, {**observation, "eval_id": eval_id}
            )
        except Exception as exc:
            raise EvaluationScoringError(sanitize_error(exc)) from exc


def dispatch_action(store, tools, service, action, now: datetime) -> None:
    if action["action"] == "execute_approval":
        service.execute_approval_action(action)
    elif action["action"] == "prepare_data_plan":
        service.prepare_data_plan_action(action)
    elif action["action"] == "plan_iteration":
        service.plan_iteration_action(action)
    elif action["action"] == "explain_diagnosis":
        service.explain_diagnosis_action(action)
    elif action["action"] == "monitor_datagen":
        _dispatch_datagen(store, tools, service, action, now)
    elif action["action"] == "monitor_training":
        _dispatch_training(store, tools, service, action, now)
    elif action["action"] == "monitor_evaluation":
        _dispatch_evaluation(store, tools, service, action, now)
    elif action["action"] == "cancel_external_job":
        payload = action["payload"]
        result = tools.stop_external_job(payload["kind"], payload["job_id"])
        service.on_cancellation_observation(action, result)
    else:
        raise ValueError(f"unknown scheduled action: {action['action']}")


def run_once(store, tools, service, now=None, limit=20) -> WorkerSummary:
    fixed_now = now
    now = fixed_now or datetime.now(timezone.utc)
    action_ceiling = store.max_scheduled_action_id()
    processed = 0
    failed = 0
    for _ in range(max(1, int(limit))):
        action_now = fixed_now or datetime.now(timezone.utc)
        actions = store.lease_due_actions(
            action_now,
            limit=1,
            lease_seconds=120,
            max_action_id=action_ceiling,
        )
        if not actions:
            break
        action = actions[0]
        processed += 1
        try:
            with store.action_lease_heartbeat(action, lease_seconds=120):
                dispatch_action(store, tools, service, action, action_now)
            store.complete_action(action["action_id"], action["lease_token"])
        except Exception as exc:
            failed += 1
            attempts = int(action["attempts"])
            if action["action"] == "prepare_data_plan" and not (
                service.is_retryable_planning_error(exc)
            ):
                service.fail_data_plan_action(action, exc)
                continue
            if action["action"] == "execute_approval" and not (
                service.is_retryable_execution_error(exc)
            ):
                service.fail_execution_action(action, exc)
                continue
            if (
                action["action"] == "monitor_evaluation"
                and isinstance(exc, EvaluationScoringError)
                and attempts >= 3
            ):
                if not store.renew_action_lease(
                    action["action_id"], action["lease_token"], lease_seconds=120
                ):
                    continue
                service.on_evaluation_processing_failed(
                    action["workflow_id"], sanitize_error(exc)
                )
                if not store.complete_action(
                    action["action_id"], action["lease_token"]
                ):
                    raise RuntimeError(
                        "evaluation recovery lease was lost before completion"
                    )
                continue
            if attempts == 5:
                event_type = (
                    "cancellation_needs_attention"
                    if action["action"] == "cancel_external_job"
                    else "monitor_needs_attention"
                )
                store.append_event(
                    action["workflow_id"],
                    event_type,
                    {"action": action["action"], "error": sanitize_error(exc)},
                )
            delay = 900 if attempts >= 5 else min(900, 60 * (2 ** min(attempts, 4)))
            store.retry_action(
                action["action_id"],
                action["lease_token"],
                action_now + timedelta(seconds=delay),
                sanitize_error(exc),
            )
    return WorkerSummary(processed=processed, failed=failed)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", required=True)
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args(argv)
    from .assistant_api import build_default_service

    service = build_default_service()
    summary = run_once(service.store, service.tools, service, limit=args.limit)
    print(json.dumps(asdict(summary), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
