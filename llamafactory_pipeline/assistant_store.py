"""Durable SQLite store for personal training-assistant workflows."""

from __future__ import annotations

import json
import re
import secrets
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator

from pydantic import BaseModel

from .assistant_schema import (
    ActionKind,
    ApprovalPayload,
    RequirementDraft,
    WorkflowState,
)
from .assistant_state import InvalidTransition, canonical_json, next_state, plan_hash

_WORKFLOW_JSON_FIELDS = {
    "requirement_draft_json": "requirement_draft",
    "confirmed_objective_json": "confirmed_objective",
    "objective_json": "objective",
    "data_plan_json": "data_plan",
    "dataset_profile_json": "dataset_profile",
    "training_plan_json": "training_plan",
    "preflight_json": "preflight",
    "evaluation_plan_json": "evaluation_plan",
    "diagnosis_json": "diagnosis",
    "datagen_jobs_json": "datagen_jobs",
    "cancel_request_json": "cancel_request",
}
_UPDATABLE_WORKFLOW_FIELDS = set(_WORKFLOW_JSON_FIELDS) | {
    "iteration",
    "train_job_id",
    "eval_id",
    "objective_hash",
}
_ID_RE = re.compile(r"^[a-z]+_[A-Za-z0-9_]+$")
_DIRECT_EXECUTION_LEASE_SECONDS = 600


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _iso(value: datetime | str) -> str:
    if isinstance(value, str):
        return value
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat()


def _new_id(prefix: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}_{stamp}_{secrets.token_hex(3)}"


def _validate_id(value: str, kind: str) -> str:
    prefix = "wf" if kind == "workflow_id" else "apr"
    if not isinstance(value, str) or not _ID_RE.fullmatch(value) or not value.startswith(f"{prefix}_"):
        raise ValueError(f"invalid {kind}")
    return value


def _json_value(value: Any) -> str:
    if isinstance(value, str):
        json.loads(value)
        return value
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return canonical_json(value)


class AssistantStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=5, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self) -> None:
        conn = self._connect()
        try:
            conn.executescript(
                """
                PRAGMA journal_mode=WAL;
                PRAGMA foreign_keys=ON;
                PRAGMA busy_timeout=5000;

                CREATE TABLE IF NOT EXISTS workflows (
                  workflow_id TEXT PRIMARY KEY,
                  state TEXT NOT NULL,
                  iteration INTEGER NOT NULL DEFAULT 0,
                  requirement_draft_json TEXT,
                  confirmed_objective_json TEXT,
                  objective_hash TEXT,
                  objective_json TEXT,
                  data_plan_json TEXT,
                  dataset_profile_json TEXT,
                  training_plan_json TEXT,
                  preflight_json TEXT,
                  evaluation_plan_json TEXT,
                  diagnosis_json TEXT,
                  datagen_jobs_json TEXT NOT NULL DEFAULT '[]',
                  train_job_id TEXT,
                  eval_id TEXT,
                  cancel_request_json TEXT,
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS messages (
                  message_id INTEGER PRIMARY KEY AUTOINCREMENT,
                  workflow_id TEXT NOT NULL REFERENCES workflows(workflow_id),
                  role TEXT NOT NULL CHECK(role IN ('user','assistant')),
                  content TEXT NOT NULL,
                  created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS events (
                  event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                  workflow_id TEXT NOT NULL REFERENCES workflows(workflow_id),
                  event_type TEXT NOT NULL,
                  payload_json TEXT NOT NULL,
                  created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS approvals (
                  approval_id TEXT PRIMARY KEY,
                  workflow_id TEXT NOT NULL REFERENCES workflows(workflow_id),
                  action TEXT NOT NULL,
                  plan_hash TEXT NOT NULL,
                  plan_json TEXT NOT NULL,
                  summary TEXT NOT NULL,
                  status TEXT NOT NULL CHECK(status IN (
                    'pending','executing','consumed','failed','rejected','stale'
                  )),
                  error TEXT NOT NULL DEFAULT '',
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS scheduled_actions (
                  action_id INTEGER PRIMARY KEY AUTOINCREMENT,
                  workflow_id TEXT NOT NULL REFERENCES workflows(workflow_id),
                  approval_id TEXT REFERENCES approvals(approval_id),
                  action TEXT NOT NULL,
                  due_at TEXT NOT NULL,
                  lease_until TEXT,
                  lease_token TEXT,
                  attempts INTEGER NOT NULL DEFAULT 0,
                  payload_json TEXT NOT NULL,
                  status TEXT NOT NULL CHECK(status IN ('pending','leased','done','failed')),
                  idempotency_key TEXT NOT NULL UNIQUE,
                  last_error TEXT NOT NULL DEFAULT '',
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS training_runs (
                  train_job_id TEXT PRIMARY KEY,
                  workflow_id TEXT NOT NULL REFERENCES workflows(workflow_id),
                  iteration INTEGER NOT NULL,
                  stage TEXT NOT NULL,
                  model_parameter_billions REAL,
                  gpu_names_json TEXT NOT NULL,
                  gpu_count INTEGER NOT NULL,
                  cutoff_len INTEGER NOT NULL,
                  quantization_bit INTEGER,
                  estimated_steps INTEGER NOT NULL,
                  actual_steps INTEGER,
                  initial_eta_seconds INTEGER,
                  calibrated_eta_seconds INTEGER,
                  duration_seconds INTEGER,
                  steps_per_second REAL,
                  terminal_status TEXT NOT NULL,
                  created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_scheduled_due
                  ON scheduled_actions(status, due_at, lease_until);
                CREATE INDEX IF NOT EXISTS idx_events_workflow
                  ON events(workflow_id, event_id);
                CREATE INDEX IF NOT EXISTS idx_training_runs_compat
                  ON training_runs(stage, cutoff_len, gpu_count, terminal_status);
                """
            )
            workflow_columns = {
                row[1]
                for row in conn.execute("PRAGMA table_info(workflows)")
            }
            workflow_migrations = {
                "requirement_draft_json": "TEXT",
                "confirmed_objective_json": "TEXT",
                "objective_hash": "TEXT",
                "cancel_request_json": "TEXT",
            }
            for column, declaration in workflow_migrations.items():
                if column not in workflow_columns:
                    conn.execute(
                        f"ALTER TABLE workflows ADD COLUMN {column} {declaration}"  # noqa: S608
                    )
            columns = {
                row[1]
                for row in conn.execute("PRAGMA table_info(scheduled_actions)")
            }
            if "approval_id" not in columns:
                conn.execute(
                    "ALTER TABLE scheduled_actions ADD COLUMN approval_id TEXT"
                )
            if "lease_token" not in columns:
                conn.execute(
                    "ALTER TABLE scheduled_actions ADD COLUMN lease_token TEXT"
                )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_scheduled_approval "
                "ON scheduled_actions(approval_id, status)"
            )
            conn.execute(
                """UPDATE approvals
                   SET status='failed',error='orphaned execution intent',updated_at=?
                   WHERE status='executing' AND NOT EXISTS (
                     SELECT 1 FROM scheduled_actions action
                     WHERE action.approval_id=approvals.approval_id
                       AND action.action='execute_approval'
                       AND action.status IN ('pending','leased')
                   )""",
                (utc_now(),),
            )
        finally:
            conn.close()

    def create_workflow(self) -> str:
        workflow_id = _new_id("wf")
        now = utc_now()
        with self.transaction() as conn:
            conn.execute(
                """INSERT INTO workflows(workflow_id,state,created_at,updated_at)
                   VALUES(?,?,?,?)""",
                (workflow_id, WorkflowState.COLLECTING_REQUIREMENTS.value, now, now),
            )
            conn.execute(
                """INSERT INTO events(workflow_id,event_type,payload_json,created_at)
                   VALUES(?,?,?,?)""",
                (workflow_id, "workflow_created", "{}", now),
            )
        return workflow_id

    def _decode_workflow(self, row: sqlite3.Row) -> dict[str, Any]:
        data = dict(row)
        for column, public_name in _WORKFLOW_JSON_FIELDS.items():
            raw = data.pop(column)
            data[public_name] = json.loads(raw) if raw is not None else None
        return data

    def get_workflow(self, workflow_id: str) -> dict[str, Any]:
        _validate_id(workflow_id, "workflow_id")
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM workflows WHERE workflow_id=?", (workflow_id,)
            ).fetchone()
        finally:
            conn.close()
        if row is None:
            raise KeyError(workflow_id)
        return self._decode_workflow(row)

    def list_workflows(self, limit: int = 50) -> list[dict[str, Any]]:
        limit = min(100, max(1, int(limit)))
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT * FROM workflows ORDER BY updated_at DESC LIMIT ?", (limit,)
            ).fetchall()
        finally:
            conn.close()
        return [self._decode_workflow(row) for row in rows]

    def update_workflow_fields(self, workflow_id: str, **fields: Any) -> None:
        _validate_id(workflow_id, "workflow_id")
        unsupported = set(fields) - _UPDATABLE_WORKFLOW_FIELDS
        if unsupported:
            raise ValueError(f"unsupported workflow field: {sorted(unsupported)[0]}")
        if not fields:
            return
        normalized = {
            key: _json_value(value) if key in _WORKFLOW_JSON_FIELDS else value
            for key, value in fields.items()
        }
        normalized["updated_at"] = utc_now()
        assignments = ",".join(f"{key}=?" for key in normalized)
        values = [normalized[key] for key in normalized]
        with self.transaction() as conn:
            cur = conn.execute(
                f"UPDATE workflows SET {assignments} WHERE workflow_id=?",  # noqa: S608
                (*values, workflow_id),
            )
            if cur.rowcount != 1:
                raise KeyError(workflow_id)

    def append_message(self, workflow_id: str, role: str, content: str) -> int:
        _validate_id(workflow_id, "workflow_id")
        if role not in {"user", "assistant"}:
            raise ValueError("invalid message role")
        now = utc_now()
        with self.transaction() as conn:
            cur = conn.execute(
                """INSERT INTO messages(workflow_id,role,content,created_at)
                   VALUES(?,?,?,?)""",
                (workflow_id, role, content, now),
            )
            conn.execute(
                "UPDATE workflows SET updated_at=? WHERE workflow_id=?", (now, workflow_id)
            )
            return int(cur.lastrowid)

    def list_messages(self, workflow_id: str) -> list[dict[str, Any]]:
        _validate_id(workflow_id, "workflow_id")
        conn = self._connect()
        try:
            rows = conn.execute(
                """SELECT message_id,role,content,created_at FROM messages
                   WHERE workflow_id=? ORDER BY message_id""",
                (workflow_id,),
            ).fetchall()
        finally:
            conn.close()
        return [dict(row) for row in rows]

    def publish_requirement_review(
        self,
        workflow_id: str,
        draft: RequirementDraft | dict[str, Any],
    ) -> WorkflowState:
        validated = RequirementDraft.model_validate(draft)
        if not validated.ready_for_review or validated.proposed_objective is None:
            raise ValueError("requirement draft is not ready for review")
        objective_hash = plan_hash(validated.proposed_objective)
        approval_plan = {
            "requirement_draft": validated.model_dump(mode="json"),
            "objective_hash": objective_hash,
        }
        return self.transition_bundle(
            workflow_id,
            "requirements_draft_ready",
            {"objective_hash": objective_hash},
            workflow_updates={"requirement_draft_json": validated},
            message=validated.assistant_reply,
            approvals=[
                {
                    "action": ActionKind.CONFIRM_REQUIREMENTS,
                    "plan": approval_plan,
                    "summary": "确认训练场景、当前问题、期望行为与成功标准",
                }
            ],
        )

    def publish_incomplete_requirement_draft(
        self,
        workflow_id: str,
        draft: RequirementDraft | dict[str, Any],
    ) -> None:
        validated = RequirementDraft.model_validate(draft)
        if validated.ready_for_review:
            raise ValueError("reviewable requirement draft needs confirmation")
        now = utc_now()
        with self.transaction() as conn:
            row = conn.execute(
                "SELECT state FROM workflows WHERE workflow_id=?", (workflow_id,)
            ).fetchone()
            if row is None:
                raise KeyError(workflow_id)
            if row["state"] != WorkflowState.COLLECTING_REQUIREMENTS.value:
                raise InvalidTransition(
                    f"{row['state']} cannot publish an incomplete requirement draft"
                )
            conn.execute(
                """UPDATE workflows SET requirement_draft_json=?,updated_at=?
                   WHERE workflow_id=?""",
                (_json_value(validated), now, workflow_id),
            )
            conn.execute(
                """INSERT INTO messages(workflow_id,role,content,created_at)
                   VALUES(?,'assistant',?,?)""",
                (workflow_id, validated.assistant_reply, now),
            )
            conn.execute(
                """INSERT INTO events(workflow_id,event_type,payload_json,created_at)
                   VALUES(?,'requirements_draft_updated',?,?)""",
                (
                    workflow_id,
                    _json_value({"missing_fields": validated.missing_fields}),
                    now,
                ),
            )

    def reopen_requirements(self, workflow_id: str) -> WorkflowState:
        """Atomically invalidate requirement confirmation and reopen chat edits."""
        _validate_id(workflow_id, "workflow_id")
        now = utc_now()
        with self.transaction() as conn:
            row = conn.execute(
                "SELECT state FROM workflows WHERE workflow_id=?", (workflow_id,)
            ).fetchone()
            if row is None:
                raise KeyError(workflow_id)
            current = WorkflowState(row["state"])
            if current == WorkflowState.COLLECTING_REQUIREMENTS:
                return current
            state = next_state(current, "requirements_revision_requested")
            conn.execute(
                "UPDATE workflows SET state=?,updated_at=? WHERE workflow_id=?",
                (state.value, now, workflow_id),
            )
            conn.execute(
                """UPDATE approvals SET status='stale',updated_at=?
                   WHERE workflow_id=? AND action=? AND status='pending'""",
                (
                    now,
                    workflow_id,
                    ActionKind.CONFIRM_REQUIREMENTS.value,
                ),
            )
            conn.execute(
                """INSERT INTO events(workflow_id,event_type,payload_json,created_at)
                   VALUES(?,'requirements_revision_requested','{}',?)""",
                (workflow_id, now),
            )
            return state

    def confirm_requirements_and_schedule_plan(
        self,
        *,
        workflow_id: str,
        objective: BaseModel | dict[str, Any],
        objective_hash: str,
        approval_id: str,
        action_id: int,
        lease_token: str,
    ) -> bool:
        """Consume requirement confirmation and durably publish planning work."""
        _validate_id(workflow_id, "workflow_id")
        _validate_id(approval_id, "approval_id")
        objective_payload = (
            objective.model_dump(mode="json")
            if isinstance(objective, BaseModel)
            else objective
        )
        if plan_hash(objective_payload) != objective_hash:
            raise ValueError("objective hash changed before confirmation")
        now = utc_now()
        planning_key = (
            f"{workflow_id}:prepare-data-plan:{objective_hash}:{approval_id}"
        )
        with self.transaction() as conn:
            workflow = conn.execute(
                "SELECT state FROM workflows WHERE workflow_id=?",
                (workflow_id,),
            ).fetchone()
            if workflow is None:
                raise KeyError(workflow_id)
            if workflow["state"] != WorkflowState.REQUIREMENTS_REVIEW.value:
                return False
            approval = conn.execute(
                """SELECT plan_json FROM approvals
                   WHERE approval_id=? AND workflow_id=? AND action=?
                     AND status='executing'""",
                (
                    approval_id,
                    workflow_id,
                    ActionKind.CONFIRM_REQUIREMENTS.value,
                ),
            ).fetchone()
            if approval is None:
                return False
            approval_payload = ApprovalPayload.model_validate_json(
                approval["plan_json"]
            )
            if approval_payload.plan.get("objective_hash") != objective_hash:
                raise ValueError("confirmed objective no longer matches approval")
            action = conn.execute(
                """SELECT 1 FROM scheduled_actions
                   WHERE action_id=? AND workflow_id=? AND approval_id=?
                     AND status='leased' AND lease_token=?""",
                (int(action_id), workflow_id, approval_id, lease_token),
            ).fetchone()
            if action is None:
                return False
            state = next_state(
                WorkflowState(workflow["state"]), "requirements_confirmed"
            )
            objective_json = _json_value(objective_payload)
            conn.execute(
                """UPDATE workflows
                   SET state=?,objective_json=?,confirmed_objective_json=?,
                       objective_hash=?,updated_at=?
                   WHERE workflow_id=?""",
                (
                    state.value,
                    objective_json,
                    objective_json,
                    objective_hash,
                    now,
                    workflow_id,
                ),
            )
            conn.execute(
                """INSERT INTO events(workflow_id,event_type,payload_json,created_at)
                   VALUES(?,'requirements_confirmed',?,?)""",
                (workflow_id, _json_value({"objective_hash": objective_hash}), now),
            )
            conn.execute(
                """INSERT OR IGNORE INTO scheduled_actions(
                     workflow_id,action,due_at,payload_json,status,idempotency_key,
                     created_at,updated_at
                   ) VALUES(?,'prepare_data_plan',?,?,'pending',?,?,?)""",
                (
                    workflow_id,
                    now,
                    _json_value({"objective_hash": objective_hash}),
                    planning_key,
                    now,
                    now,
                ),
            )
            conn.execute(
                """UPDATE approvals SET status='consumed',error='',updated_at=?
                   WHERE approval_id=? AND status='executing'""",
                (now, approval_id),
            )
            conn.execute(
                """UPDATE approvals SET status='stale',updated_at=?
                   WHERE workflow_id=? AND approval_id<>? AND status='pending'""",
                (now, workflow_id, approval_id),
            )
            consumed = conn.execute(
                """UPDATE scheduled_actions
                   SET status='done',lease_until=NULL,lease_token=NULL,
                       last_error='',updated_at=?
                   WHERE action_id=? AND approval_id=? AND status='leased'
                     AND lease_token=?""",
                (now, int(action_id), approval_id, lease_token),
            )
            if consumed.rowcount != 1:
                raise RuntimeError("requirement confirmation lease was lost")
            return True

    def publish_confirmed_data_plan(
        self,
        *,
        workflow_id: str,
        action_id: int,
        lease_token: str,
        objective_hash: str,
        data_plan: BaseModel | dict[str, Any],
        summary: str,
    ) -> bool:
        """Fence planner output, transition and create the datagen approval."""
        plan_payload = (
            data_plan.model_dump(mode="json")
            if isinstance(data_plan, BaseModel)
            else data_plan
        )
        approval_payload = ApprovalPayload(
            action=ActionKind.START_DATAGEN,
            plan=plan_payload,
            decision_warnings=[f"objective_hash:{objective_hash}"],
        )
        approval_id = _new_id("apr")
        now = utc_now()
        with self.transaction() as conn:
            workflow = conn.execute(
                """SELECT state,objective_hash FROM workflows
                   WHERE workflow_id=?""",
                (workflow_id,),
            ).fetchone()
            if workflow is None:
                raise KeyError(workflow_id)
            if (
                workflow["state"] != WorkflowState.DATA_PLAN_PREPARING.value
                or workflow["objective_hash"] != objective_hash
            ):
                return False
            action = conn.execute(
                """UPDATE scheduled_actions
                   SET status='done',lease_until=NULL,lease_token=NULL,
                       last_error='',updated_at=?
                   WHERE action_id=? AND workflow_id=?
                     AND action='prepare_data_plan' AND status='leased'
                     AND lease_token=?""",
                (now, int(action_id), workflow_id, lease_token),
            )
            if action.rowcount != 1:
                return False
            state = next_state(
                WorkflowState(workflow["state"]), "data_plan_created"
            )
            conn.execute(
                """UPDATE workflows SET state=?,data_plan_json=?,updated_at=?
                   WHERE workflow_id=?""",
                (state.value, _json_value(plan_payload), now, workflow_id),
            )
            conn.execute(
                """INSERT INTO events(workflow_id,event_type,payload_json,created_at)
                   VALUES(?,'data_plan_created',?,?)""",
                (
                    workflow_id,
                    _json_value(
                        {
                            "objective_hash": objective_hash,
                            "data_plan": plan_payload,
                        }
                    ),
                    now,
                ),
            )
            conn.execute(
                """UPDATE approvals SET status='stale',updated_at=?
                   WHERE workflow_id=? AND action=? AND status='pending'""",
                (now, workflow_id, ActionKind.START_DATAGEN.value),
            )
            conn.execute(
                """INSERT INTO approvals(
                     approval_id,workflow_id,action,plan_hash,plan_json,summary,
                     status,created_at,updated_at
                   ) VALUES(?,?,?,?,?,?,'pending',?,?)""",
                (
                    approval_id,
                    workflow_id,
                    ActionKind.START_DATAGEN.value,
                    plan_hash(approval_payload),
                    canonical_json(approval_payload),
                    summary,
                    now,
                    now,
                ),
            )
            return True

    def fail_data_plan_preparation(
        self,
        *,
        workflow_id: str,
        action_id: int,
        lease_token: str,
        error: str,
    ) -> bool:
        now = utc_now()
        with self.transaction() as conn:
            workflow = conn.execute(
                "SELECT state FROM workflows WHERE workflow_id=?", (workflow_id,)
            ).fetchone()
            if workflow is None:
                raise KeyError(workflow_id)
            if workflow["state"] != WorkflowState.DATA_PLAN_PREPARING.value:
                return False
            failed = conn.execute(
                """UPDATE scheduled_actions
                   SET status='failed',lease_until=NULL,lease_token=NULL,
                       last_error=?,updated_at=?
                   WHERE action_id=? AND workflow_id=?
                     AND action='prepare_data_plan' AND status='leased'
                     AND lease_token=?""",
                (error, now, int(action_id), workflow_id, lease_token),
            )
            if failed.rowcount != 1:
                return False
            state = next_state(
                WorkflowState(workflow["state"]),
                "data_plan_preparation_failed",
            )
            conn.execute(
                "UPDATE workflows SET state=?,updated_at=? WHERE workflow_id=?",
                (state.value, now, workflow_id),
            )
            conn.execute(
                """INSERT INTO events(workflow_id,event_type,payload_json,created_at)
                   VALUES(?,'data_plan_preparation_failed',?,?)""",
                (workflow_id, _json_value({"error": error}), now),
            )
            conn.execute(
                """INSERT INTO messages(workflow_id,role,content,created_at)
                   VALUES(?,'assistant',?,?)""",
                (
                    workflow_id,
                    "数据方案未通过结构校验，已保留确认目标；请重试生成方案或修订需求。",
                    now,
                ),
            )
            return True

    def retry_data_plan(self, workflow_id: str) -> bool:
        _validate_id(workflow_id, "workflow_id")
        now = utc_now()
        with self.transaction() as conn:
            workflow = conn.execute(
                """SELECT state,objective_hash,confirmed_objective_json
                   FROM workflows WHERE workflow_id=?""",
                (workflow_id,),
            ).fetchone()
            if workflow is None:
                raise KeyError(workflow_id)
            if (
                workflow["state"] != WorkflowState.REQUIREMENTS_REVIEW.value
                or not workflow["objective_hash"]
                or not workflow["confirmed_objective_json"]
            ):
                return False
            state = WorkflowState.DATA_PLAN_PREPARING
            retry_key = (
                f"{workflow_id}:prepare-data-plan:{workflow['objective_hash']}:"
                f"{_new_id('plan')}"
            )
            conn.execute(
                "UPDATE workflows SET state=?,updated_at=? WHERE workflow_id=?",
                (state.value, now, workflow_id),
            )
            conn.execute(
                """INSERT INTO events(workflow_id,event_type,payload_json,created_at)
                   VALUES(?,'data_plan_retry_started',?,?)""",
                (
                    workflow_id,
                    _json_value({"objective_hash": workflow["objective_hash"]}),
                    now,
                ),
            )
            conn.execute(
                """INSERT OR IGNORE INTO scheduled_actions(
                     workflow_id,action,due_at,payload_json,status,idempotency_key,
                     created_at,updated_at
                   ) VALUES(?,'prepare_data_plan',?,?,'pending',?,?,?)""",
                (
                    workflow_id,
                    now,
                    _json_value({"objective_hash": workflow["objective_hash"]}),
                    retry_key,
                    now,
                    now,
                ),
            )
            return True

    @staticmethod
    def _cancellation_targets(
        workflow: sqlite3.Row,
        execution_rows: list[sqlite3.Row],
    ) -> list[dict[str, str]]:
        targets: dict[tuple[str, str], dict[str, str]] = {}

        def add(kind: str, target_id: Any) -> None:
            if isinstance(target_id, str) and target_id:
                targets[(kind, target_id)] = {"kind": kind, "job_id": target_id}

        state = WorkflowState(workflow["state"])
        if state == WorkflowState.DATA_GENERATING:
            for row in json.loads(workflow["datagen_jobs_json"] or "[]"):
                add("datagen", row.get("job_id"))
        elif state == WorkflowState.TRAINING:
            add("training", workflow["train_job_id"])
        elif state == WorkflowState.EVALUATING:
            add("evaluation", workflow["eval_id"])

        for row in execution_rows:
            payload = json.loads(row["payload_json"])
            approval_payload = payload.get("approval_payload") or {}
            action = approval_payload.get("action")
            refs = payload.get("external_refs") or {}
            if action == ActionKind.START_DATAGEN.value:
                for launch in refs.get("launches") or []:
                    add("datagen", launch.get("job_id"))
            elif action == ActionKind.START_TRAINING.value:
                add("training", refs.get("job_id"))
            elif action == ActionKind.START_EVALUATION.value:
                add("evaluation", refs.get("eval_id"))
        return list(targets.values())

    def request_cancellation(self, workflow_id: str, reason: str) -> dict[str, Any]:
        """Persist a cancellation intent, fence normal work and schedule stops."""
        _validate_id(workflow_id, "workflow_id")
        reason = str(reason or "").strip()
        if not reason:
            raise ValueError("cancellation reason must not be empty")
        now = utc_now()
        with self.transaction() as conn:
            workflow = conn.execute(
                "SELECT * FROM workflows WHERE workflow_id=?", (workflow_id,)
            ).fetchone()
            if workflow is None:
                raise KeyError(workflow_id)
            existing = workflow["cancel_request_json"]
            state = WorkflowState(workflow["state"])
            if state in {WorkflowState.CANCELLING, WorkflowState.CANCELLED} and existing:
                return json.loads(existing)
            next_value = next_state(state, "cancellation_requested")
            executions = conn.execute(
                """SELECT payload_json FROM scheduled_actions
                   WHERE workflow_id=? AND action='execute_approval'
                     AND status IN ('pending','leased')""",
                (workflow_id,),
            ).fetchall()
            targets = self._cancellation_targets(workflow, list(executions))
            request = {
                "cancel_request_id": _new_id("cancel"),
                "reason": reason[:500],
                "requested_at": now,
                "targets": targets,
            }
            request_json = _json_value(request)
            conn.execute(
                """UPDATE workflows SET state=?,cancel_request_json=?,updated_at=?
                   WHERE workflow_id=?""",
                (next_value.value, request_json, now, workflow_id),
            )
            conn.execute(
                """UPDATE approvals SET status='stale',updated_at=?
                   WHERE workflow_id=? AND status='pending'""",
                (now, workflow_id),
            )
            conn.execute(
                """UPDATE approvals SET status='failed',
                       error='workflow cancellation requested',updated_at=?
                   WHERE workflow_id=? AND status='executing'""",
                (now, workflow_id),
            )
            conn.execute(
                """UPDATE scheduled_actions
                   SET status='failed',lease_until=NULL,lease_token=NULL,
                       last_error='workflow cancellation requested',updated_at=?
                   WHERE workflow_id=? AND action<>'cancel_external_job'
                     AND status IN ('pending','leased')""",
                (now, workflow_id),
            )
            conn.execute(
                """INSERT INTO events(workflow_id,event_type,payload_json,created_at)
                   VALUES(?,'cancellation_requested',?,?)""",
                (workflow_id, request_json, now),
            )
            for target in targets:
                conn.execute(
                    """INSERT OR IGNORE INTO scheduled_actions(
                         workflow_id,action,due_at,payload_json,status,
                         idempotency_key,created_at,updated_at
                       ) VALUES(?,'cancel_external_job',?,?,'pending',?,?,?)""",
                    (
                        workflow_id,
                        now,
                        _json_value(
                            {
                                "cancel_request_id": request["cancel_request_id"],
                                **target,
                            }
                        ),
                        (
                            f"{workflow_id}:cancel:{request['cancel_request_id']}:"
                            f"{target['kind']}:{target['job_id']}"
                        ),
                        now,
                        now,
                    ),
                )
            if not targets:
                final_state = next_state(next_value, "cancellation_completed")
                conn.execute(
                    "UPDATE workflows SET state=?,updated_at=? WHERE workflow_id=?",
                    (final_state.value, now, workflow_id),
                )
                conn.execute(
                    """INSERT INTO events(
                         workflow_id,event_type,payload_json,created_at
                       ) VALUES(?,'cancellation_completed',?,?)""",
                    (
                        workflow_id,
                        _json_value(
                            {
                                "cancel_request_id": request["cancel_request_id"],
                                "targets": [],
                            }
                        ),
                        now,
                    ),
                )
            return request

    def complete_cancellation_action(
        self,
        *,
        workflow_id: str,
        cancel_request_id: str,
        action_id: int,
        lease_token: str,
        result: dict[str, Any],
    ) -> bool:
        now = utc_now()
        with self.transaction() as conn:
            workflow = conn.execute(
                "SELECT state,cancel_request_json FROM workflows WHERE workflow_id=?",
                (workflow_id,),
            ).fetchone()
            if workflow is None:
                raise KeyError(workflow_id)
            request = json.loads(workflow["cancel_request_json"] or "{}")
            if (
                workflow["state"] != WorkflowState.CANCELLING.value
                or request.get("cancel_request_id") != cancel_request_id
            ):
                return False
            completed = conn.execute(
                """UPDATE scheduled_actions
                   SET status='done',lease_until=NULL,lease_token=NULL,
                       last_error='',updated_at=?
                   WHERE action_id=? AND workflow_id=?
                     AND action='cancel_external_job' AND status='leased'
                     AND lease_token=?""",
                (now, int(action_id), workflow_id, lease_token),
            )
            if completed.rowcount != 1:
                return False
            conn.execute(
                """INSERT INTO events(workflow_id,event_type,payload_json,created_at)
                   VALUES(?,'external_job_stopped',?,?)""",
                (workflow_id, _json_value(result), now),
            )
            remaining = conn.execute(
                """SELECT 1 FROM scheduled_actions
                   WHERE workflow_id=? AND action='cancel_external_job'
                     AND status IN ('pending','leased') LIMIT 1""",
                (workflow_id,),
            ).fetchone()
            if remaining is None:
                final_state = next_state(
                    WorkflowState(workflow["state"]), "cancellation_completed"
                )
                conn.execute(
                    "UPDATE workflows SET state=?,updated_at=? WHERE workflow_id=?",
                    (final_state.value, now, workflow_id),
                )
                conn.execute(
                    """INSERT INTO events(
                         workflow_id,event_type,payload_json,created_at
                       ) VALUES(?,'cancellation_completed',?,?)""",
                    (
                        workflow_id,
                        _json_value({"cancel_request_id": cancel_request_id}),
                        now,
                    ),
                )
            return True

    def append_event(self, workflow_id: str, event_type: str, payload: Any) -> int:
        _validate_id(workflow_id, "workflow_id")
        now = utc_now()
        with self.transaction() as conn:
            cur = conn.execute(
                """INSERT INTO events(workflow_id,event_type,payload_json,created_at)
                   VALUES(?,?,?,?)""",
                (workflow_id, event_type, _json_value(payload), now),
            )
            conn.execute(
                "UPDATE workflows SET updated_at=? WHERE workflow_id=?", (now, workflow_id)
            )
            return int(cur.lastrowid)

    def append_event_once(
        self, workflow_id: str, event_type: str, payload: Any
    ) -> int | None:
        """Insert an identical reducer event at most once for a workflow."""
        _validate_id(workflow_id, "workflow_id")
        now = utc_now()
        payload_json = _json_value(payload)
        with self.transaction() as conn:
            existing = conn.execute(
                """SELECT event_id FROM events
                   WHERE workflow_id=? AND event_type=? AND payload_json=?
                   LIMIT 1""",
                (workflow_id, event_type, payload_json),
            ).fetchone()
            if existing is not None:
                return None
            cur = conn.execute(
                """INSERT INTO events(workflow_id,event_type,payload_json,created_at)
                   VALUES(?,?,?,?)""",
                (workflow_id, event_type, payload_json, now),
            )
            conn.execute(
                "UPDATE workflows SET updated_at=? WHERE workflow_id=?",
                (now, workflow_id),
            )
            return int(cur.lastrowid)

    def transition(self, workflow_id: str, event: str, payload: Any) -> WorkflowState:
        _validate_id(workflow_id, "workflow_id")
        now = utc_now()
        with self.transaction() as conn:
            row = conn.execute(
                "SELECT state FROM workflows WHERE workflow_id=?", (workflow_id,)
            ).fetchone()
            if row is None:
                raise KeyError(workflow_id)
            state = next_state(WorkflowState(row["state"]), event)
            conn.execute(
                "UPDATE workflows SET state=?,updated_at=? WHERE workflow_id=?",
                (state.value, now, workflow_id),
            )
            conn.execute(
                """INSERT INTO events(workflow_id,event_type,payload_json,created_at)
                   VALUES(?,?,?,?)""",
                (workflow_id, event, _json_value(payload), now),
            )
            return state

    def transition_bundle(
        self,
        workflow_id: str,
        event: str,
        payload: Any,
        *,
        workflow_updates: dict[str, Any] | None = None,
        message: str | None = None,
        approvals: list[dict[str, Any]] | None = None,
        scheduled_actions: list[dict[str, Any]] | None = None,
        extra_events: list[dict[str, Any]] | None = None,
    ) -> WorkflowState:
        """Atomically transition state and publish all required follow-up work."""
        _validate_id(workflow_id, "workflow_id")
        workflow_updates = workflow_updates or {}
        approvals = approvals or []
        scheduled_actions = scheduled_actions or []
        extra_events = extra_events or []
        unsupported = set(workflow_updates) - _UPDATABLE_WORKFLOW_FIELDS
        if unsupported:
            raise ValueError(
                f"unsupported workflow field: {sorted(unsupported)[0]}"
            )
        normalized = {
            key: _json_value(value) if key in _WORKFLOW_JSON_FIELDS else value
            for key, value in workflow_updates.items()
        }
        prepared_approvals = []
        for spec in approvals:
            action = ActionKind(spec["action"])
            plan = spec["plan"]
            if isinstance(plan, BaseModel):
                plan = plan.model_dump(mode="json")
            approval_payload = ApprovalPayload(
                action=action,
                plan=plan,
                decision_warnings=sorted(set(spec.get("decision_warnings", []))),
            )
            prepared_approvals.append(
                (
                    _new_id("apr"),
                    action,
                    approval_payload,
                    plan_hash(approval_payload),
                    spec["summary"],
                )
            )
        now = utc_now()
        with self.transaction() as conn:
            row = conn.execute(
                "SELECT state FROM workflows WHERE workflow_id=?", (workflow_id,)
            ).fetchone()
            if row is None:
                raise KeyError(workflow_id)
            state = next_state(WorkflowState(row["state"]), event)
            updates = {**normalized, "state": state.value, "updated_at": now}
            assignments = ",".join(f"{key}=?" for key in updates)
            conn.execute(
                f"UPDATE workflows SET {assignments} WHERE workflow_id=?",  # noqa: S608
                (*updates.values(), workflow_id),
            )
            conn.execute(
                """INSERT INTO events(workflow_id,event_type,payload_json,created_at)
                   VALUES(?,?,?,?)""",
                (workflow_id, event, _json_value(payload), now),
            )
            for extra in extra_events:
                conn.execute(
                    """INSERT INTO events(
                         workflow_id,event_type,payload_json,created_at
                       ) VALUES(?,?,?,?)""",
                    (
                        workflow_id,
                        extra["event_type"],
                        _json_value(extra.get("payload", {})),
                        now,
                    ),
                )
            if message:
                conn.execute(
                    """INSERT INTO messages(workflow_id,role,content,created_at)
                       VALUES(?,'assistant',?,?)""",
                    (workflow_id, message, now),
                )
            for approval_id, action, approval_payload, digest, summary in prepared_approvals:
                conn.execute(
                    """UPDATE approvals SET status='stale',updated_at=?
                       WHERE workflow_id=? AND action=? AND status='pending'""",
                    (now, workflow_id, action.value),
                )
                conn.execute(
                    """INSERT INTO approvals(
                         approval_id,workflow_id,action,plan_hash,plan_json,summary,
                         status,created_at,updated_at
                       ) VALUES(?,?,?,?,?,?,'pending',?,?)""",
                    (
                        approval_id,
                        workflow_id,
                        action.value,
                        digest,
                        canonical_json(approval_payload),
                        summary,
                        now,
                        now,
                    ),
                )
            for scheduled in scheduled_actions:
                conn.execute(
                    """INSERT OR IGNORE INTO scheduled_actions(
                         workflow_id,action,due_at,payload_json,status,
                         idempotency_key,created_at,updated_at
                       ) VALUES(?,?,?,?,'pending',?,?,?)""",
                    (
                        workflow_id,
                        scheduled["action"],
                        _iso(scheduled.get("due_at", now)),
                        _json_value(scheduled.get("payload", {})),
                        scheduled["idempotency_key"],
                        now,
                        now,
                    ),
                )
            return state

    def publish_preflight(
        self,
        *,
        workflow_id: str,
        report: BaseModel | dict[str, Any],
        plan: BaseModel | dict[str, Any],
        summary: str,
        decision_warnings: list[str],
    ) -> WorkflowState:
        """Atomically publish preflight state, report and replacement approval."""
        _validate_id(workflow_id, "workflow_id")
        report_payload = (
            report.model_dump(mode="json") if isinstance(report, BaseModel) else report
        )
        plan_payload = (
            plan.model_dump(mode="json") if isinstance(plan, BaseModel) else plan
        )
        status = report_payload["status"]
        approval_payload = ApprovalPayload(
            action=ActionKind.START_TRAINING,
            plan=plan_payload,
            decision_warnings=sorted(set(decision_warnings)),
        )
        approval_id = _new_id("apr")
        now = utc_now()
        with self.transaction() as conn:
            row = conn.execute(
                "SELECT state FROM workflows WHERE workflow_id=?",
                (workflow_id,),
            ).fetchone()
            if row is None:
                raise KeyError(workflow_id)
            current = WorkflowState(row["state"])
            if status == "block":
                if current == WorkflowState.PREFLIGHT_BLOCKED:
                    state = current
                    event = "preflight_refreshed"
                else:
                    state = next_state(current, "preflight_blocked")
                    event = "preflight_blocked"
            elif current == WorkflowState.TRAIN_READY:
                state = current
                event = "preflight_refreshed"
            else:
                state = next_state(current, "preflight_passed")
                event = "preflight_passed"
            conn.execute(
                """UPDATE workflows SET state=?,preflight_json=?,updated_at=?
                   WHERE workflow_id=?""",
                (state.value, _json_value(report_payload), now, workflow_id),
            )
            conn.execute(
                """INSERT INTO events(workflow_id,event_type,payload_json,created_at)
                   VALUES(?,?,?,?)""",
                (workflow_id, event, _json_value({"report": report_payload}), now),
            )
            conn.execute(
                """UPDATE approvals SET status='stale',updated_at=?
                   WHERE workflow_id=? AND action=? AND status='pending'""",
                (now, workflow_id, ActionKind.START_TRAINING.value),
            )
            if status != "block":
                conn.execute(
                    """INSERT INTO approvals(
                         approval_id,workflow_id,action,plan_hash,plan_json,summary,
                         status,created_at,updated_at
                       ) VALUES(?,?,?,?,?,?,'pending',?,?)""",
                    (
                        approval_id,
                        workflow_id,
                        ActionKind.START_TRAINING.value,
                        plan_hash(approval_payload),
                        canonical_json(approval_payload),
                        summary,
                        now,
                        now,
                    ),
                )
            return state

    def get_action_by_key(self, idempotency_key: str) -> dict[str, Any]:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM scheduled_actions WHERE idempotency_key=?",
                (idempotency_key,),
            ).fetchone()
        finally:
            conn.close()
        if row is None:
            raise KeyError(idempotency_key)
        return self._decode_action(row)

    def lease_action(
        self, action_id: int, lease_seconds: int = 600
    ) -> dict[str, Any] | None:
        """Lease one freshly published action for an immediate in-process attempt."""
        now_dt = datetime.now(timezone.utc)
        now = _iso(now_dt)
        lease_until = _iso(now_dt + timedelta(seconds=max(1, lease_seconds)))
        token = secrets.token_hex(16)
        with self.transaction() as conn:
            updated = conn.execute(
                """UPDATE scheduled_actions
                   SET status='leased',lease_until=?,lease_token=?,
                       attempts=attempts+1,updated_at=?
                   WHERE action_id=? AND status='pending' AND due_at<=?""",
                (lease_until, token, now, int(action_id), now),
            )
            if updated.rowcount != 1:
                return None
            row = conn.execute(
                "SELECT * FROM scheduled_actions WHERE action_id=?",
                (int(action_id),),
            ).fetchone()
        return self._decode_action(row)

    def publish_iteration_plan(
        self,
        *,
        workflow_id: str,
        action_id: int,
        lease_token: str,
        plan: BaseModel | dict[str, Any],
        summary: str,
    ) -> None:
        """Atomically publish a diagnosis-derived plan and its approval."""
        _validate_id(workflow_id, "workflow_id")
        plan_payload = (
            plan.model_dump(mode="json") if isinstance(plan, BaseModel) else plan
        )
        approval_payload = ApprovalPayload(
            action=ActionKind.START_ITERATION,
            plan=plan_payload,
        )
        approval_id = _new_id("apr")
        now = utc_now()
        with self.transaction() as conn:
            workflow = conn.execute(
                "SELECT state FROM workflows WHERE workflow_id=?", (workflow_id,)
            ).fetchone()
            if workflow is None:
                raise KeyError(workflow_id)
            if workflow["state"] != WorkflowState.DIAGNOSIS_READY.value:
                raise RuntimeError("workflow is not ready for an iteration plan")
            conn.execute(
                """UPDATE workflows SET data_plan_json=?,updated_at=?
                   WHERE workflow_id=?""",
                (_json_value(plan_payload), now, workflow_id),
            )
            conn.execute(
                """UPDATE approvals SET status='stale',updated_at=?
                   WHERE workflow_id=? AND action=? AND status='pending'""",
                (now, workflow_id, ActionKind.START_ITERATION.value),
            )
            conn.execute(
                """INSERT INTO approvals(
                     approval_id,workflow_id,action,plan_hash,plan_json,summary,
                     status,created_at,updated_at
                   ) VALUES(?,?,?,?,?,?,'pending',?,?)""",
                (
                    approval_id,
                    workflow_id,
                    ActionKind.START_ITERATION.value,
                    plan_hash(approval_payload),
                    canonical_json(approval_payload),
                    summary,
                    now,
                    now,
                ),
            )
            conn.execute(
                """INSERT INTO events(workflow_id,event_type,payload_json,created_at)
                   VALUES(?, 'iteration_plan_created', ?, ?)""",
                (workflow_id, _json_value({"data_plan": plan_payload}), now),
            )
            action = conn.execute(
                """UPDATE scheduled_actions
                   SET status='done',lease_until=NULL,lease_token=NULL,
                       last_error='',updated_at=?
                   WHERE action_id=? AND workflow_id=?
                     AND action='plan_iteration'
                     AND status='leased' AND lease_token=?""",
                (now, int(action_id), workflow_id, lease_token),
            )
            if action.rowcount != 1:
                raise RuntimeError("iteration planning action is no longer active")

    def finish_data_preparation(
        self,
        workflow_id: str,
        *,
        outputs: list[str],
        dataset_profile: BaseModel | dict[str, Any],
        training_plan: BaseModel | dict[str, Any],
    ) -> WorkflowState:
        """Atomically publish frozen-data metadata and the resulting train plan."""
        _validate_id(workflow_id, "workflow_id")
        now = utc_now()
        profile_json = _json_value(dataset_profile)
        plan_json = _json_value(training_plan)
        profile_payload = (
            dataset_profile.model_dump(mode="json")
            if isinstance(dataset_profile, BaseModel)
            else dataset_profile
        )
        with self.transaction() as conn:
            row = conn.execute(
                "SELECT state FROM workflows WHERE workflow_id=?", (workflow_id,)
            ).fetchone()
            if row is None:
                raise KeyError(workflow_id)
            current = WorkflowState(row["state"])
            data_review = next_state(current, "datagen_completed")
            train_ready = next_state(data_review, "train_plan_created")
            conn.execute(
                """UPDATE workflows
                   SET state=?,dataset_profile_json=?,training_plan_json=?,updated_at=?
                   WHERE workflow_id=?""",
                (train_ready.value, profile_json, plan_json, now, workflow_id),
            )
            conn.execute(
                """INSERT INTO events(workflow_id,event_type,payload_json,created_at)
                   VALUES(?,?,?,?)""",
                (
                    workflow_id,
                    "datagen_completed",
                    _json_value(
                        {
                            "outputs": outputs,
                            "dataset_profile": profile_payload,
                        }
                    ),
                    now,
                ),
            )
            conn.execute(
                """INSERT INTO events(workflow_id,event_type,payload_json,created_at)
                   VALUES(?,?,?,?)""",
                (
                    workflow_id,
                    "train_plan_created",
                    _json_value(
                        {
                            "training_plan": (
                                training_plan.model_dump(mode="json")
                                if isinstance(training_plan, BaseModel)
                                else training_plan
                            )
                        }
                    ),
                    now,
                ),
            )
            return train_ready

    def list_events(
        self, workflow_id: str, after_id: int = 0, limit: int = 100
    ) -> list[dict[str, Any]]:
        _validate_id(workflow_id, "workflow_id")
        limit = min(1000, max(1, int(limit)))
        conn = self._connect()
        try:
            rows = conn.execute(
                """SELECT event_id,event_type,payload_json,created_at FROM events
                   WHERE workflow_id=? AND event_id>? ORDER BY event_id LIMIT ?""",
                (workflow_id, max(0, int(after_id)), limit),
            ).fetchall()
        finally:
            conn.close()
        return [
            {
                "event_id": row["event_id"],
                "event_type": row["event_type"],
                "payload": json.loads(row["payload_json"]),
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def list_recent_events(
        self, workflow_id: str, event_type: str, limit: int = 2
    ) -> list[dict[str, Any]]:
        _validate_id(workflow_id, "workflow_id")
        limit = min(100, max(1, int(limit)))
        conn = self._connect()
        try:
            rows = conn.execute(
                """SELECT event_id,event_type,payload_json,created_at FROM events
                   WHERE workflow_id=? AND event_type=?
                   ORDER BY event_id DESC LIMIT ?""",
                (workflow_id, event_type, limit),
            ).fetchall()
        finally:
            conn.close()
        return [
            {
                "event_id": row["event_id"],
                "event_type": row["event_type"],
                "payload": json.loads(row["payload_json"]),
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def create_approval(
        self,
        workflow_id: str,
        action: ActionKind,
        plan: BaseModel | dict[str, Any],
        summary: str,
        decision_warnings: list[str] | tuple[str, ...] = (),
    ) -> dict[str, Any]:
        _validate_id(workflow_id, "workflow_id")
        if isinstance(plan, BaseModel):
            plan = plan.model_dump(mode="json")
        payload = ApprovalPayload(
            action=action,
            plan=plan,
            decision_warnings=sorted(set(decision_warnings)),
        )
        approval_id = _new_id("apr")
        now = utc_now()
        digest = plan_hash(payload)
        with self.transaction() as conn:
            conn.execute(
                """UPDATE approvals SET status='stale',updated_at=?
                   WHERE workflow_id=? AND action=? AND status='pending'""",
                (now, workflow_id, action.value),
            )
            conn.execute(
                """INSERT INTO approvals(
                     approval_id,workflow_id,action,plan_hash,plan_json,summary,status,
                     created_at,updated_at
                   ) VALUES(?,?,?,?,?,?,'pending',?,?)""",
                (
                    approval_id,
                    workflow_id,
                    action.value,
                    digest,
                    canonical_json(payload),
                    summary,
                    now,
                    now,
                ),
            )
        return self.get_approval(workflow_id, approval_id)

    def _decode_approval(self, row: sqlite3.Row, include_payload: bool = True) -> dict[str, Any]:
        data = dict(row)
        raw = data.pop("plan_json")
        payload = json.loads(raw)
        if include_payload:
            data["payload"] = payload
        else:
            data["decision_warnings"] = payload.get("decision_warnings", [])
        return data

    def get_approval(self, workflow_id: str, approval_id: str) -> dict[str, Any]:
        _validate_id(workflow_id, "workflow_id")
        _validate_id(approval_id, "approval_id")
        conn = self._connect()
        try:
            row = conn.execute(
                """SELECT * FROM approvals
                   WHERE workflow_id=? AND approval_id=?""",
                (workflow_id, approval_id),
            ).fetchone()
        finally:
            conn.close()
        if row is None:
            raise KeyError(approval_id)
        return self._decode_approval(row)

    def list_pending_approvals(self, workflow_id: str) -> list[dict[str, Any]]:
        _validate_id(workflow_id, "workflow_id")
        conn = self._connect()
        try:
            rows = conn.execute(
                """SELECT * FROM approvals WHERE workflow_id=? AND status='pending'
                   ORDER BY created_at,approval_id""",
                (workflow_id,),
            ).fetchall()
        finally:
            conn.close()
        return [self._decode_approval(row, include_payload=False) for row in rows]

    def claim_approval(self, workflow_id: str, approval_id: str, expected_hash: str) -> bool:
        _validate_id(workflow_id, "workflow_id")
        _validate_id(approval_id, "approval_id")
        now = utc_now()
        with self.transaction() as conn:
            cur = conn.execute(
                """UPDATE approvals SET status='executing',updated_at=?
                   WHERE approval_id=? AND workflow_id=? AND plan_hash=? AND status='pending'
                     AND NOT EXISTS (
                       SELECT 1 FROM approvals active
                       WHERE active.workflow_id=? AND active.status='executing'
                     )""",
                (now, approval_id, workflow_id, expected_hash, workflow_id),
            )
            return cur.rowcount == 1

    def prepare_external_execution(
        self,
        workflow_id: str,
        approval_id: str,
        expected_hash: str,
        payload: Any,
        external_refs: Any,
    ) -> dict[str, Any] | None:
        """Claim an approval and lease its durable outbox to the API caller."""
        _validate_id(workflow_id, "workflow_id")
        _validate_id(approval_id, "approval_id")
        now_dt = datetime.now(timezone.utc)
        now = _iso(now_dt)
        lease_until = _iso(
            now_dt + timedelta(seconds=_DIRECT_EXECUTION_LEASE_SECONDS)
        )
        lease_token = secrets.token_hex(16)
        action_payload = {
            "approval_id": approval_id,
            "approval_payload": (
                payload.model_dump(mode="json")
                if isinstance(payload, BaseModel)
                else payload
            ),
            "external_refs": external_refs,
        }
        idempotency_key = f"approval-execution:{approval_id}"
        with self.transaction() as conn:
            claimed = conn.execute(
                """UPDATE approvals SET status='executing',updated_at=?
                   WHERE approval_id=? AND workflow_id=? AND plan_hash=?
                     AND status='pending' AND NOT EXISTS (
                       SELECT 1 FROM approvals active
                       WHERE active.workflow_id=? AND active.status='executing'
                     )""",
                (now, approval_id, workflow_id, expected_hash, workflow_id),
            )
            if claimed.rowcount != 1:
                return None
            conn.execute(
                """INSERT INTO scheduled_actions(
                     workflow_id,approval_id,action,due_at,lease_until,lease_token,attempts,
                     payload_json,status,idempotency_key,created_at,updated_at
                   ) VALUES(?,?,'execute_approval',?,?,?,1,?,'leased',?,?,?)""",
                (
                    workflow_id,
                    approval_id,
                    now,
                    lease_until,
                    lease_token,
                    _json_value(action_payload),
                    idempotency_key,
                    now,
                    now,
                ),
            )
            row = conn.execute(
                "SELECT * FROM scheduled_actions WHERE idempotency_key=?",
                (idempotency_key,),
            ).fetchone()
        return self._decode_action(row)

    def defer_action(
        self,
        action_id: int,
        lease_token: str,
        due_at: datetime | str,
        error: str,
    ) -> bool:
        """Defer a direct or leased outbox attempt without losing its intent."""
        with self.transaction() as conn:
            cur = conn.execute(
                """UPDATE scheduled_actions
                   SET status='pending',due_at=?,lease_until=NULL,lease_token=NULL,
                       last_error=?,updated_at=?
                   WHERE action_id=? AND status='leased' AND lease_token=?""",
                (
                    _iso(due_at),
                    error,
                    utc_now(),
                    int(action_id),
                    lease_token,
                ),
            )
            return cur.rowcount == 1

    def commit_external_start(
        self,
        *,
        workflow_id: str,
        approval_id: str,
        action_id: int,
        lease_token: str,
        transition_event: str,
        workflow_updates: dict[str, Any],
        event_payload: Any,
        monitor_action: str,
        monitor_payload: Any,
        monitor_key: str,
    ) -> WorkflowState:
        """Atomically register a submitted job, monitor, transition and approval."""
        _validate_id(workflow_id, "workflow_id")
        _validate_id(approval_id, "approval_id")
        unsupported = set(workflow_updates) - _UPDATABLE_WORKFLOW_FIELDS
        if unsupported:
            raise ValueError(
                f"unsupported workflow field: {sorted(unsupported)[0]}"
            )
        now = utc_now()
        normalized = {
            key: _json_value(value) if key in _WORKFLOW_JSON_FIELDS else value
            for key, value in workflow_updates.items()
        }
        with self.transaction() as conn:
            row = conn.execute(
                "SELECT state FROM workflows WHERE workflow_id=?", (workflow_id,)
            ).fetchone()
            if row is None:
                raise KeyError(workflow_id)
            state = next_state(WorkflowState(row["state"]), transition_event)
            updates = {**normalized, "state": state.value, "updated_at": now}
            assignments = ",".join(f"{key}=?" for key in updates)
            conn.execute(
                f"UPDATE workflows SET {assignments} WHERE workflow_id=?",  # noqa: S608
                (*updates.values(), workflow_id),
            )
            conn.execute(
                """INSERT INTO events(workflow_id,event_type,payload_json,created_at)
                   VALUES(?,?,?,?)""",
                (workflow_id, transition_event, _json_value(event_payload), now),
            )
            conn.execute(
                """INSERT OR IGNORE INTO scheduled_actions(
                     workflow_id,action,due_at,payload_json,status,idempotency_key,
                     created_at,updated_at
                   ) VALUES(?,?,?,?,'pending',?,?,?)""",
                (
                    workflow_id,
                    monitor_action,
                    now,
                    _json_value(monitor_payload),
                    monitor_key,
                    now,
                    now,
                ),
            )
            approval = conn.execute(
                """UPDATE approvals SET status='consumed',error='',updated_at=?
                   WHERE approval_id=? AND workflow_id=? AND status='executing'""",
                (now, approval_id, workflow_id),
            )
            if approval.rowcount != 1:
                raise RuntimeError("approval execution is no longer active")
            conn.execute(
                """UPDATE approvals SET status='stale',updated_at=?
                   WHERE workflow_id=? AND approval_id<>? AND status='pending'""",
                (now, workflow_id, approval_id),
            )
            action = conn.execute(
                """UPDATE scheduled_actions
                   SET status='done',lease_until=NULL,lease_token=NULL,
                       last_error='',updated_at=?
                   WHERE action_id=? AND approval_id=?
                     AND status='leased' AND lease_token=?""",
                (now, int(action_id), approval_id, lease_token),
            )
            if action.rowcount != 1:
                raise RuntimeError("execution outbox is no longer active")
            return state

    def fail_external_execution(
        self,
        *,
        workflow_id: str,
        approval_id: str,
        action_id: int,
        lease_token: str,
        error: str,
        message: str,
    ) -> bool:
        """Fence a permanent/exhausted submit and atomically reopen approval."""
        _validate_id(workflow_id, "workflow_id")
        _validate_id(approval_id, "approval_id")
        now = utc_now()
        replacement_id = _new_id("apr")
        with self.transaction() as conn:
            action = conn.execute(
                """UPDATE scheduled_actions
                   SET status='failed',lease_until=NULL,lease_token=NULL,
                       last_error=?,updated_at=?
                   WHERE action_id=? AND workflow_id=? AND approval_id=?
                     AND status='leased' AND lease_token=?""",
                (
                    error,
                    now,
                    int(action_id),
                    workflow_id,
                    approval_id,
                    lease_token,
                ),
            )
            if action.rowcount != 1:
                return False
            approval = conn.execute(
                """SELECT action,plan_hash,plan_json,summary FROM approvals
                   WHERE approval_id=? AND workflow_id=? AND status='executing'""",
                (approval_id, workflow_id),
            ).fetchone()
            if approval is None:
                raise RuntimeError("approval execution is no longer active")
            conn.execute(
                """UPDATE approvals SET status='failed',error=?,updated_at=?
                   WHERE approval_id=?""",
                (error, now, approval_id),
            )
            conn.execute(
                """UPDATE approvals SET status='stale',updated_at=?
                   WHERE workflow_id=? AND action=? AND status='pending'""",
                (now, workflow_id, approval["action"]),
            )
            conn.execute(
                """INSERT INTO approvals(
                     approval_id,workflow_id,action,plan_hash,plan_json,summary,
                     status,created_at,updated_at
                   ) VALUES(?,?,?,?,?,?,'pending',?,?)""",
                (
                    replacement_id,
                    workflow_id,
                    approval["action"],
                    approval["plan_hash"],
                    approval["plan_json"],
                    approval["summary"],
                    now,
                    now,
                ),
            )
            conn.execute(
                """INSERT INTO events(workflow_id,event_type,payload_json,created_at)
                   VALUES(?,'external_execution_failed',?,?)""",
                (
                    workflow_id,
                    _json_value(
                        {
                            "approval_id": approval_id,
                            "action": approval["action"],
                            "error": error,
                        }
                    ),
                    now,
                ),
            )
            conn.execute(
                """INSERT INTO messages(workflow_id,role,content,created_at)
                   VALUES(?,'assistant',?,?)""",
                (workflow_id, message, now),
            )
            conn.execute(
                "UPDATE workflows SET updated_at=? WHERE workflow_id=?",
                (now, workflow_id),
            )
            return True

    def commit_local_execution(
        self,
        *,
        workflow_id: str,
        approval_id: str,
        action_id: int,
        lease_token: str,
    ) -> bool:
        """Atomically consume a replayable local approval after its state effect."""
        _validate_id(workflow_id, "workflow_id")
        _validate_id(approval_id, "approval_id")
        now = utc_now()
        with self.transaction() as conn:
            original = conn.execute(
                "SELECT created_at FROM approvals WHERE approval_id=?",
                (approval_id,),
            ).fetchone()
            if original is None:
                return False
            approval = conn.execute(
                """UPDATE approvals SET status='consumed',error='',updated_at=?
                   WHERE approval_id=? AND workflow_id=? AND status='executing'""",
                (now, approval_id, workflow_id),
            )
            if approval.rowcount != 1:
                return False
            conn.execute(
                """UPDATE approvals SET status='stale',updated_at=?
                   WHERE workflow_id=? AND approval_id<>? AND status='pending'
                     AND created_at<=?""",
                (now, workflow_id, approval_id, original["created_at"]),
            )
            action = conn.execute(
                """UPDATE scheduled_actions
                   SET status='done',lease_until=NULL,lease_token=NULL,
                       last_error='',updated_at=?
                   WHERE action_id=? AND approval_id=? AND status='leased'
                     AND lease_token=?""",
                (now, int(action_id), approval_id, lease_token),
            )
            if action.rowcount != 1:
                raise RuntimeError("local execution outbox is no longer active")
            return True

    def finish_approval(
        self, approval_id: str, succeeded: bool, error: str = ""
    ) -> bool:
        _validate_id(approval_id, "approval_id")
        status = "consumed" if succeeded else "failed"
        with self.transaction() as conn:
            cur = conn.execute(
                """UPDATE approvals SET status=?,error=?,updated_at=?
                   WHERE approval_id=? AND status='executing'""",
                (status, error, utc_now(), approval_id),
            )
            return cur.rowcount == 1

    def mark_approval_stale(self, approval_id: str) -> bool:
        _validate_id(approval_id, "approval_id")
        with self.transaction() as conn:
            cur = conn.execute(
                """UPDATE approvals SET status='stale',updated_at=?
                   WHERE approval_id=? AND status='pending'""",
                (utc_now(), approval_id),
            )
            return cur.rowcount == 1

    def stale_other_pending_approvals(
        self, workflow_id: str, keep_approval_id: str
    ) -> int:
        _validate_id(workflow_id, "workflow_id")
        _validate_id(keep_approval_id, "approval_id")
        with self.transaction() as conn:
            cur = conn.execute(
                """UPDATE approvals SET status='stale',updated_at=?
                   WHERE workflow_id=? AND approval_id<>? AND status='pending'""",
                (utc_now(), workflow_id, keep_approval_id),
            )
            return cur.rowcount

    def reject_approval(self, workflow_id: str, approval_id: str) -> bool:
        _validate_id(workflow_id, "workflow_id")
        _validate_id(approval_id, "approval_id")
        now = utc_now()
        with self.transaction() as conn:
            cur = conn.execute(
                """UPDATE approvals SET status='rejected',updated_at=?
                   WHERE workflow_id=? AND approval_id=? AND status='pending'""",
                (now, workflow_id, approval_id),
            )
            if cur.rowcount != 1:
                return False
            conn.execute(
                """INSERT INTO events(workflow_id,event_type,payload_json,created_at)
                   VALUES(?,'approval_rejected',?,?)""",
                (workflow_id, canonical_json({"approval_id": approval_id}), now),
            )
            conn.execute(
                "UPDATE workflows SET updated_at=? WHERE workflow_id=?",
                (now, workflow_id),
            )
            return True

    def reject_data_plan_for_revision(
        self,
        workflow_id: str,
        approval_id: str,
        message: str,
    ) -> bool:
        """Atomically reject the initial data plan and reopen requirements."""
        _validate_id(workflow_id, "workflow_id")
        _validate_id(approval_id, "approval_id")
        now = utc_now()
        with self.transaction() as conn:
            row = conn.execute(
                """SELECT w.state,a.action,a.status
                   FROM workflows AS w
                   JOIN approvals AS a ON a.workflow_id=w.workflow_id
                   WHERE w.workflow_id=? AND a.approval_id=?""",
                (workflow_id, approval_id),
            ).fetchone()
            if row is None:
                raise KeyError(approval_id)
            if (
                row["status"] != "pending"
                or row["action"] != ActionKind.START_DATAGEN.value
                or row["state"] != WorkflowState.DATA_PLAN_READY.value
            ):
                return False
            state = next_state(
                WorkflowState(row["state"]), "data_plan_revision_requested"
            )
            rejected = conn.execute(
                """UPDATE approvals SET status='rejected',updated_at=?
                   WHERE workflow_id=? AND approval_id=? AND status='pending'""",
                (now, workflow_id, approval_id),
            )
            if rejected.rowcount != 1:
                return False
            conn.execute(
                "UPDATE workflows SET state=?,updated_at=? WHERE workflow_id=?",
                (state.value, now, workflow_id),
            )
            payload = canonical_json({"approval_id": approval_id})
            conn.execute(
                """INSERT INTO events(workflow_id,event_type,payload_json,created_at)
                   VALUES(?,'approval_rejected',?,?)""",
                (workflow_id, payload, now),
            )
            conn.execute(
                """INSERT INTO events(workflow_id,event_type,payload_json,created_at)
                   VALUES(?,'data_plan_revision_requested',?,?)""",
                (workflow_id, payload, now),
            )
            conn.execute(
                """INSERT INTO messages(workflow_id,role,content,created_at)
                   VALUES(?,'assistant',?,?)""",
                (workflow_id, message, now),
            )
            return True

    def schedule_action(
        self,
        workflow_id: str,
        action: str,
        due_at: datetime | str,
        payload: Any,
        idempotency_key: str,
    ) -> int:
        _validate_id(workflow_id, "workflow_id")
        now = utc_now()
        with self.transaction() as conn:
            conn.execute(
                """INSERT OR IGNORE INTO scheduled_actions(
                     workflow_id,action,due_at,payload_json,status,idempotency_key,
                     created_at,updated_at
                   ) VALUES(?,?,?,?,'pending',?,?,?)""",
                (
                    workflow_id,
                    action,
                    _iso(due_at),
                    _json_value(payload),
                    idempotency_key,
                    now,
                    now,
                ),
            )
            row = conn.execute(
                "SELECT action_id FROM scheduled_actions WHERE idempotency_key=?",
                (idempotency_key,),
            ).fetchone()
            return int(row["action_id"])

    def lease_due_actions(
        self,
        now: datetime | str,
        limit: int = 20,
        lease_seconds: int = 120,
        max_action_id: int | None = None,
    ) -> list[dict[str, Any]]:
        now_iso = _iso(now)
        base_dt = datetime.fromisoformat(now_iso)
        lease_until = _iso(base_dt + timedelta(seconds=lease_seconds))
        limit = min(100, max(1, int(limit)))
        with self.transaction() as conn:
            action_ceiling = (
                int(max_action_id) if max_action_id is not None else 2**63 - 1
            )
            rows = conn.execute(
                """SELECT * FROM scheduled_actions
                   WHERE due_at<=? AND (
                     status='pending' OR (status='leased' AND lease_until<?)
                   ) AND action_id<=?
                   ORDER BY due_at,action_id LIMIT ?""",
                (now_iso, now_iso, action_ceiling, limit),
            ).fetchall()
            if not rows:
                return []
            ids = []
            for row in rows:
                lease_token = secrets.token_hex(16)
                updated = conn.execute(
                    """UPDATE scheduled_actions
                       SET status='leased',lease_until=?,lease_token=?,
                           attempts=attempts+1,updated_at=?
                       WHERE action_id=? AND (
                         status='pending' OR (status='leased' AND lease_until<?)
                       )""",
                    (
                        lease_until,
                        lease_token,
                        now_iso,
                        row["action_id"],
                        now_iso,
                    ),
                )
                if updated.rowcount == 1:
                    ids.append(row["action_id"])
            if not ids:
                return []
            placeholders = ",".join("?" for _ in ids)
            leased = conn.execute(
                f"SELECT * FROM scheduled_actions WHERE action_id IN ({placeholders}) ORDER BY action_id",  # noqa: S608
                ids,
            ).fetchall()
        return [self._decode_action(row) for row in leased]

    def max_scheduled_action_id(self) -> int:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT COALESCE(MAX(action_id), 0) AS value FROM scheduled_actions"
            ).fetchone()
        finally:
            conn.close()
        return int(row["value"])

    def _decode_action(self, row: sqlite3.Row) -> dict[str, Any]:
        data = dict(row)
        data["payload"] = json.loads(data.pop("payload_json"))
        return data

    def renew_action_lease(
        self, action_id: int, lease_token: str, lease_seconds: int = 120
    ) -> bool:
        now = datetime.now(timezone.utc)
        lease_until = _iso(now + timedelta(seconds=max(1, lease_seconds)))
        with self.transaction() as conn:
            cur = conn.execute(
                """UPDATE scheduled_actions SET lease_until=?,updated_at=?
                   WHERE action_id=? AND status='leased' AND lease_token=?""",
                (lease_until, _iso(now), int(action_id), lease_token),
            )
            return cur.rowcount == 1

    @contextmanager
    def action_lease_heartbeat(
        self, action: dict[str, Any], lease_seconds: int = 120
    ) -> Iterator[None]:
        """Verify and renew one action lease for the whole side-effect window."""
        action_id = int(action["action_id"])
        lease_token = str(action["lease_token"])
        if not self.renew_action_lease(action_id, lease_token, lease_seconds):
            raise RuntimeError("scheduled action lease was lost before dispatch")
        stopped = threading.Event()
        renewal_lost = threading.Event()

        def renew() -> None:
            interval = max(1, int(lease_seconds) // 3)
            while not stopped.wait(interval):
                try:
                    owned = self.renew_action_lease(
                        action_id, lease_token, lease_seconds
                    )
                except Exception:
                    renewal_lost.set()
                    return
                if not owned:
                    renewal_lost.set()
                    return

        thread = threading.Thread(target=renew, daemon=True)
        thread.start()
        try:
            yield
        finally:
            stopped.set()
            thread.join(timeout=1)
        if renewal_lost.is_set():
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT status,lease_token FROM scheduled_actions "
                    "WHERE action_id=?",
                    (action_id,),
                ).fetchone()
            finally:
                conn.close()
            if (
                row is not None
                and row["status"] == "leased"
                and row["lease_token"] != lease_token
            ):
                raise RuntimeError("scheduled action lease changed during dispatch")

    def complete_action(self, action_id: int, lease_token: str) -> bool:
        with self.transaction() as conn:
            cur = conn.execute(
                """UPDATE scheduled_actions
                   SET status='done',lease_until=NULL,lease_token=NULL,updated_at=?
                   WHERE action_id=? AND status='leased' AND lease_token=?""",
                (utc_now(), int(action_id), lease_token),
            )
            return cur.rowcount == 1

    def publish_diagnosis_explanation(
        self,
        *,
        workflow_id: str,
        action_id: int,
        lease_token: str,
        explanation: str,
    ) -> bool:
        """Fence and publish the optional LLM explanation after terminal state."""
        now = utc_now()
        with self.transaction() as conn:
            cur = conn.execute(
                """UPDATE scheduled_actions
                   SET status='done',lease_until=NULL,lease_token=NULL,updated_at=?
                   WHERE action_id=? AND workflow_id=?
                     AND status='leased' AND lease_token=?""",
                (now, int(action_id), workflow_id, lease_token),
            )
            if cur.rowcount != 1:
                return False
            conn.execute(
                """INSERT INTO messages(workflow_id,role,content,created_at)
                   VALUES(?,'assistant',?,?)""",
                (workflow_id, explanation, now),
            )
            conn.execute(
                """INSERT INTO events(workflow_id,event_type,payload_json,created_at)
                   VALUES(?,?,?,?)""",
                (
                    workflow_id,
                    "diagnosis_explained",
                    _json_value({"explanation": explanation}),
                    now,
                ),
            )
            return True

    def retry_action(
        self,
        action_id: int,
        lease_token: str,
        due_at: datetime | str,
        error: str,
    ) -> bool:
        with self.transaction() as conn:
            cur = conn.execute(
                """UPDATE scheduled_actions
                   SET status='pending',due_at=?,lease_until=NULL,lease_token=NULL,
                       last_error=?,updated_at=?
                   WHERE action_id=? AND status='leased' AND lease_token=?""",
                (_iso(due_at), error, utc_now(), int(action_id), lease_token),
            )
            return cur.rowcount == 1

    def reschedule_action(
        self,
        action_id: int,
        lease_token: str,
        due_at: datetime | str,
        payload: Any,
    ) -> bool:
        """Move a leased action back to pending with updated monitor memory."""
        with self.transaction() as conn:
            cur = conn.execute(
                """UPDATE scheduled_actions
                   SET status='pending',due_at=?,payload_json=?,lease_until=NULL,
                       lease_token=NULL,
                       attempts=0,last_error='',updated_at=?
                   WHERE action_id=? AND status='leased' AND lease_token=?""",
                (
                    _iso(due_at),
                    _json_value(payload),
                    utc_now(),
                    int(action_id),
                    lease_token,
                ),
            )
            return cur.rowcount == 1

    def count_pending_actions(self, action: str) -> int:
        now = utc_now()
        conn = self._connect()
        try:
            row = conn.execute(
                """SELECT COUNT(*) AS n FROM scheduled_actions
                   WHERE action=? AND (
                     status='pending' OR (status='leased' AND lease_until>?)
                   )""",
                (action, now),
            ).fetchone()
        finally:
            conn.close()
        return int(row["n"])

    def record_training_run(self, **run: Any) -> None:
        required = {
            "train_job_id",
            "workflow_id",
            "iteration",
            "stage",
            "model_parameter_billions",
            "gpu_names",
            "gpu_count",
            "cutoff_len",
            "quantization_bit",
            "estimated_steps",
            "actual_steps",
            "initial_eta_seconds",
            "calibrated_eta_seconds",
            "duration_seconds",
            "steps_per_second",
            "terminal_status",
        }
        missing = required - set(run)
        if missing:
            raise ValueError(f"missing training run field: {sorted(missing)[0]}")
        _validate_id(run["workflow_id"], "workflow_id")
        gpu_names_json = canonical_json(sorted(run["gpu_names"]))
        with self.transaction() as conn:
            conn.execute(
                """INSERT INTO training_runs(
                     train_job_id,workflow_id,iteration,stage,model_parameter_billions,
                     gpu_names_json,gpu_count,cutoff_len,quantization_bit,estimated_steps,
                     actual_steps,initial_eta_seconds,calibrated_eta_seconds,duration_seconds,
                     steps_per_second,terminal_status,created_at
                   ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(train_job_id) DO UPDATE SET
                     actual_steps=excluded.actual_steps,
                     calibrated_eta_seconds=excluded.calibrated_eta_seconds,
                     duration_seconds=excluded.duration_seconds,
                     steps_per_second=excluded.steps_per_second,
                     terminal_status=excluded.terminal_status""",
                (
                    run["train_job_id"],
                    run["workflow_id"],
                    run["iteration"],
                    run["stage"],
                    run["model_parameter_billions"],
                    gpu_names_json,
                    run["gpu_count"],
                    run["cutoff_len"],
                    run["quantization_bit"],
                    run["estimated_steps"],
                    run["actual_steps"],
                    run["initial_eta_seconds"],
                    run["calibrated_eta_seconds"],
                    run["duration_seconds"],
                    run["steps_per_second"],
                    run["terminal_status"],
                    utc_now(),
                ),
            )

    def list_compatible_training_runs(
        self,
        *,
        stage: str,
        model_parameter_billions: float | None,
        gpu_names: list[str],
        gpu_count: int,
        cutoff_len: int,
        quantization_bit: int | None,
    ) -> list[dict[str, Any]]:
        conn = self._connect()
        try:
            rows = conn.execute(
                """SELECT * FROM training_runs
                   WHERE stage=? AND gpu_names_json=? AND gpu_count=? AND cutoff_len=?
                     AND ((quantization_bit IS NULL AND ? IS NULL) OR quantization_bit=?)
                     AND terminal_status='succeeded' AND steps_per_second>0
                   ORDER BY created_at DESC""",
                (
                    stage,
                    canonical_json(sorted(gpu_names)),
                    gpu_count,
                    cutoff_len,
                    quantization_bit,
                    quantization_bit,
                ),
            ).fetchall()
        finally:
            conn.close()
        decoded = []
        for row in rows:
            data = dict(row)
            stored_size = data["model_parameter_billions"]
            if model_parameter_billions is not None:
                if stored_size is None or abs(stored_size - model_parameter_billions) / model_parameter_billions > 0.20:
                    continue
            data["gpu_names"] = json.loads(data.pop("gpu_names_json"))
            decoded.append(data)
        return decoded
