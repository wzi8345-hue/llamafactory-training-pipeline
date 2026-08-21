"""FastAPI router and default dependency graph for the personal training assistant."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterator

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from . import dataset_store, preflight, remote
from .assistant_data import prepare_generated_datasets
from .assistant_planner import AssistantPlanner, PlannerOutputError
from .assistant_service import ApprovalConflict, AssistantService, sanitize_error
from .assistant_schema import CancelRequest
from .assistant_state import InvalidTransition
from .assistant_store import AssistantStore
from .assistant_tools import AssistantTools
from .training_policy import recommend_training

router = APIRouter(prefix="/api/assistant", tags=["training-assistant"])


class MessageRequest(BaseModel):
    message: str = Field(min_length=1, max_length=20000)


class ApprovalRequest(BaseModel):
    plan_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")


def build_default_service() -> AssistantService:
    default_db = Path(__file__).parent / "assistant_state" / "assistant.sqlite"
    store = AssistantStore(Path(os.environ.get("ASSISTANT_DB_PATH", default_db)))
    tools = AssistantTools()
    planner = AssistantPlanner()

    def policy(objective, profile):
        cfg = tools.remote_config()
        inventory = preflight.collect_model_inventory(cfg, objective.base_model_path)
        gpus = preflight.collect_gpus(cfg)
        draft = recommend_training(
            objective, profile, inventory, gpus, objective.template
        )
        history = store.list_compatible_training_runs(
            stage=draft.config.method.stage,
            model_parameter_billions=inventory.parameter_billions,
            gpu_names=[gpu.name for gpu in gpus],
            gpu_count=len(gpus),
            cutoff_len=draft.config.dataset.cutoff_len,
            quantization_bit=draft.config.model.quantization_bit,
        )
        return recommend_training(
            objective,
            profile,
            inventory,
            gpus,
            objective.template,
            historical_runs=history,
        )

    def preflight_runner(workflow_id, plan, profile):
        del workflow_id
        meta = dataset_store.dataset_meta(plan.dataset_name, "train")
        if meta is not None:
            meta = {
                **meta,
                "data_path": str(
                    dataset_store.data_path(plan.dataset_name, "train")
                ),
            }
        return preflight.run_preflight(tools.remote_config(), plan, meta, profile)

    return AssistantService(
        store=store,
        planner=planner,
        tools=tools,
        policy=policy,
        preflight_runner=preflight_runner,
        data_preparer=prepare_generated_datasets,
    )


@lru_cache(maxsize=1)
def get_assistant_service() -> AssistantService:
    return build_default_service()


def _http_error(exc: Exception) -> HTTPException:
    if isinstance(exc, KeyError):
        return HTTPException(status_code=404, detail="workflow or approval not found")
    if isinstance(exc, (ApprovalConflict, InvalidTransition)):
        return HTTPException(status_code=409, detail=sanitize_error(exc))
    if isinstance(exc, PlannerOutputError):
        return HTTPException(status_code=502, detail="planner returned invalid output")
    if isinstance(exc, remote.RemoteError):
        return HTTPException(status_code=502, detail=sanitize_error(exc))
    if isinstance(exc, ValueError):
        return HTTPException(status_code=422, detail=sanitize_error(exc))
    return HTTPException(status_code=500, detail="assistant operation failed")


def _sse(event: str, data: Any) -> str:
    payload = json.dumps(data, ensure_ascii=False, default=str)
    return f"event: {event}\ndata: {payload}\n\n"


def _assistant_reply(snapshot: dict[str, Any]) -> str:
    for message in reversed(snapshot.get("messages") or []):
        if message.get("role") == "assistant":
            return str(message.get("content") or "")
    return ""


def _reply_chunks(text: str, size: int = 24) -> Iterator[str]:
    for offset in range(0, len(text), size):
        yield text[offset:offset + size]


def _stream_operation(
    operation: Callable[[], dict[str, Any]],
) -> Iterator[str]:
    yield _sse("progress", {
        "stage": "planning",
        "message": "正在理解需求并更新训练方案…",
    })
    try:
        snapshot = operation()
    except Exception as exc:
        error = _http_error(exc)
        yield _sse("error", {
            "status": error.status_code,
            "detail": error.detail,
        })
        return

    yield _sse("progress", {
        "stage": "validated",
        "message": "方案已校验，正在输出…",
    })
    for chunk in _reply_chunks(_assistant_reply(snapshot)):
        yield _sse("assistant_delta", {"delta": chunk})
    yield _sse("snapshot", snapshot)
    yield _sse("done", {"workflow_id": snapshot.get("workflow_id")})


def _streaming_response(operation: Callable[[], dict[str, Any]]) -> StreamingResponse:
    return StreamingResponse(
        _stream_operation(operation),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/workflows")
def create_workflow(
    body: MessageRequest, service: AssistantService = Depends(get_assistant_service)
):
    try:
        return service.create_workflow(body.message)
    except Exception as exc:
        raise _http_error(exc) from exc


@router.post("/workflows/stream")
def create_workflow_stream(
    body: MessageRequest, service: AssistantService = Depends(get_assistant_service)
):
    return _streaming_response(lambda: service.create_workflow(body.message))


@router.get("/workflows")
def list_workflows(service: AssistantService = Depends(get_assistant_service)):
    try:
        return {"workflows": service.list_workflows(limit=50)}
    except Exception as exc:
        raise _http_error(exc) from exc


@router.get("/workflows/{workflow_id}")
def get_workflow(
    workflow_id: str, service: AssistantService = Depends(get_assistant_service)
):
    try:
        return service.snapshot(workflow_id)
    except Exception as exc:
        raise _http_error(exc) from exc


@router.post("/workflows/{workflow_id}/messages")
def add_message(
    workflow_id: str,
    body: MessageRequest,
    service: AssistantService = Depends(get_assistant_service),
):
    try:
        return service.add_message(workflow_id, body.message)
    except Exception as exc:
        raise _http_error(exc) from exc


@router.post("/workflows/{workflow_id}/messages/stream")
def add_message_stream(
    workflow_id: str,
    body: MessageRequest,
    service: AssistantService = Depends(get_assistant_service),
):
    return _streaming_response(
        lambda: service.add_message(workflow_id, body.message)
    )


@router.post("/workflows/{workflow_id}/requirements/revise")
def revise_requirements(
    workflow_id: str,
    service: AssistantService = Depends(get_assistant_service),
):
    try:
        return service.revise_requirements(workflow_id)
    except Exception as exc:
        raise _http_error(exc) from exc


@router.post("/workflows/{workflow_id}/data-plan/retry")
def retry_data_plan(
    workflow_id: str,
    service: AssistantService = Depends(get_assistant_service),
):
    try:
        return service.retry_data_plan(workflow_id)
    except Exception as exc:
        raise _http_error(exc) from exc


@router.post("/workflows/{workflow_id}/cancel")
def cancel_workflow(
    workflow_id: str,
    body: CancelRequest,
    service: AssistantService = Depends(get_assistant_service),
):
    try:
        return service.cancel(workflow_id, body.reason)
    except Exception as exc:
        raise _http_error(exc) from exc


@router.get("/workflows/{workflow_id}/events")
def list_events(
    workflow_id: str,
    after_id: int = 0,
    service: AssistantService = Depends(get_assistant_service),
):
    try:
        return {"events": service.list_events(workflow_id, after_id)}
    except Exception as exc:
        raise _http_error(exc) from exc


@router.post("/workflows/{workflow_id}/approvals/{approval_id}/approve")
def approve(
    workflow_id: str,
    approval_id: str,
    body: ApprovalRequest,
    service: AssistantService = Depends(get_assistant_service),
):
    try:
        return service.approve(workflow_id, approval_id, body.plan_hash)
    except Exception as exc:
        raise _http_error(exc) from exc


@router.post("/workflows/{workflow_id}/approvals/{approval_id}/reject")
def reject(
    workflow_id: str,
    approval_id: str,
    service: AssistantService = Depends(get_assistant_service),
):
    try:
        return service.reject(workflow_id, approval_id)
    except Exception as exc:
        raise _http_error(exc) from exc


@router.post("/workflows/{workflow_id}/preflight")
def run_preflight(
    workflow_id: str, service: AssistantService = Depends(get_assistant_service)
):
    try:
        return service.run_preflight(workflow_id)
    except Exception as exc:
        raise _http_error(exc) from exc
