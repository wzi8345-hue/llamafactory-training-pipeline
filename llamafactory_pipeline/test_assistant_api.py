"""FastAPI router tests for the personal training assistant."""

from __future__ import annotations

from unittest.mock import Mock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from .assistant_api import get_assistant_service, router
from .assistant_service import ApprovalConflict


def client(fake_service):
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_assistant_service] = lambda: fake_service
    return TestClient(app)


def fake_service():
    service = Mock()
    service.create_workflow.return_value = {
        "workflow_id": "wf_1",
        "state": "collecting_requirements",
    }
    service.list_events.return_value = []
    service.list_workflows.return_value = []
    return service


def test_create_workflow_returns_snapshot():
    service = fake_service()
    response = client(service).post(
        "/api/assistant/workflows", json={"message": "train FC"}
    )
    assert response.status_code == 200
    assert response.json()["workflow_id"] == "wf_1"


def test_create_workflow_streams_progress_validated_reply_and_snapshot():
    service = fake_service()
    service.create_workflow.return_value = {
        "workflow_id": "wf_1",
        "state": "data_plan_ready",
        "messages": [
            {"role": "user", "content": "训练 FC"},
            {"role": "assistant", "content": "这是已经校验的训练方案。"},
        ],
        "events": [],
    }

    response = client(service).post(
        "/api/assistant/workflows/stream", json={"message": "训练 FC"}
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    body = response.text
    assert "event: progress" in body
    assert "event: assistant_delta" in body
    assert "event: snapshot" in body
    assert "event: done" in body
    assert body.index("event: progress") < body.index("event: assistant_delta")
    assert body.index("event: assistant_delta") < body.index("event: snapshot")
    assert "这是已经校验的训练方案。" in body


def test_add_message_stream_maps_errors_to_sanitized_sse_event():
    service = fake_service()
    service.add_message.side_effect = ValueError(
        "api_key=topsecret /Users/alice/.ssh/id_ed25519"
    )

    response = client(service).post(
        "/api/assistant/workflows/wf_1/messages/stream",
        json={"message": "继续"},
    )

    assert response.status_code == 200
    assert "event: error" in response.text
    assert "topsecret" not in response.text
    assert "/Users/alice" not in response.text
    assert '"status": 422' in response.text


def test_approve_requires_matching_hash():
    service = fake_service()
    service.approve.side_effect = ApprovalConflict("plan changed")
    response = client(service).post(
        "/api/assistant/workflows/wf_1/approvals/apr_1/approve",
        json={"plan_hash": "sha256:" + "a" * 64},
    )
    assert response.status_code == 409


def test_events_are_incremental():
    service = fake_service()
    response = client(service).get(
        "/api/assistant/workflows/wf_1/events?after_id=8"
    )
    assert response.status_code == 200
    service.list_events.assert_called_once_with("wf_1", 8)


def test_invalid_approval_hash_is_rejected_by_request_validation():
    response = client(fake_service()).post(
        "/api/assistant/workflows/wf_1/approvals/apr_1/approve",
        json={"plan_hash": "old"},
    )
    assert response.status_code == 422


def test_missing_workflow_maps_to_404():
    service = fake_service()
    service.snapshot.side_effect = KeyError("wf_missing")
    response = client(service).get("/api/assistant/workflows/wf_missing")
    assert response.status_code == 404


def test_api_errors_do_not_expose_credentials_or_home_paths():
    service = fake_service()
    service.snapshot.side_effect = ValueError(
        "api_key=topsecret /Users/alice/.ssh/id_ed25519"
    )
    response = client(service).get("/api/assistant/workflows/wf_1")
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert "topsecret" not in detail
    assert "/Users/alice" not in detail


def test_cancel_endpoint_returns_latest_snapshot():
    service = fake_service()
    service.cancel.return_value = {
        "workflow_id": "wf_1",
        "state": "cancelling",
    }

    response = client(service).post(
        "/api/assistant/workflows/wf_1/cancel",
        json={"reason": "用户手动中止"},
    )

    assert response.status_code == 200
    assert response.json()["state"] == "cancelling"
    service.cancel.assert_called_once_with("wf_1", "用户手动中止")


def test_requirement_revision_and_data_plan_retry_endpoints():
    service = fake_service()
    service.revise_requirements.return_value = {
        "workflow_id": "wf_1",
        "state": "collecting_requirements",
    }
    service.retry_data_plan.return_value = {
        "workflow_id": "wf_1",
        "state": "data_plan_preparing",
    }
    test_client = client(service)

    revised = test_client.post(
        "/api/assistant/workflows/wf_1/requirements/revise"
    )
    retried = test_client.post(
        "/api/assistant/workflows/wf_1/data-plan/retry"
    )

    assert revised.status_code == 200
    assert retried.status_code == 200
    service.revise_requirements.assert_called_once_with("wf_1")
    service.retry_data_plan.assert_called_once_with("wf_1")
