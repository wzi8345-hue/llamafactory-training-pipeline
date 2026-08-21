"""Offline old-A/new-B behavior comparison acceptance tests."""

from __future__ import annotations

import json

from .assistant_ab import run_behavior_ab


def test_behavior_ab_proves_no_premature_execution_and_no_real_tools(tmp_path):
    report = run_behavior_ab(tmp_path)

    assert report["A"]["premature_execution_rate"] > 0
    assert report["B"]["premature_execution_rate"] == 0
    assert report["B"]["requirement_evidence_coverage"] == 1.0
    assert report["B"]["approval_sequence_accuracy"] == 1.0
    assert report["B"]["visual_stage_coverage"] == 1.0
    assert report["B"]["cancel_success_rate"] == 1.0
    assert report["B"]["cancel_idempotency_rate"] == 1.0
    assert report["B"]["artifact_preservation_rate"] == 1.0
    assert report["external_side_effect_calls"] == 0
    assert (tmp_path / "report.json").is_file()
    assert (tmp_path / "report.md").is_file()


def test_behavior_ab_reports_per_scenario_state_action_and_cancel_evidence(tmp_path):
    report = run_behavior_ab(tmp_path)
    scenarios = report["B"]["scenarios"]

    assert scenarios["ambiguous_9b_then_fc"]["state_sequence"] == [
        "collecting_requirements",
        "collecting_requirements",
    ]
    assert scenarios["ambiguous_9b_then_fc"]["approvals"] == []
    assert scenarios["explicit_fc_goal"]["state_sequence"] == [
        "requirements_review",
        "data_plan_preparing",
        "data_plan_ready",
    ]
    assert scenarios["explicit_fc_goal"]["approvals"] == [
        "confirm_requirements",
        "start_datagen",
    ]
    assert set(scenarios["cancellation"]) == {"datagen", "training", "evaluation"}
    assert all(
        row["state_after_restart"] == "cancelled"
        for row in scenarios["cancellation"].values()
    )
    on_disk = json.loads((tmp_path / "report.json").read_text("utf-8"))
    assert on_disk == report
