"""Paired evaluation statistics and deterministic diagnosis tests."""

from __future__ import annotations

from .assistant_diagnosis import (
    compare_models,
    diagnose_evaluation,
    exact_mcnemar_p,
    paired_bootstrap_ci,
)
from .assistant_schema import SuccessCriteria


def criteria() -> SuccessCriteria:
    return SuccessCriteria(
        primary_metric="primary",
        min_improvement=0.15,
        max_invalid_rate_increase=0.01,
        max_critical_slice_rate_regression=0.02,
        max_critical_slice_score_regression=0.10,
    )


def test_exact_mcnemar_is_symmetric():
    assert exact_mcnemar_p(1, 9) == exact_mcnemar_p(9, 1)
    assert 0 <= exact_mcnemar_p(1, 9) <= 1


def test_paired_bootstrap_is_reproducible():
    one = paired_bootstrap_ci([0.2, 0.1, -0.1, 0.4], iterations=2000, seed=42)
    two = paired_bootstrap_ci([0.2, 0.1, -0.1, 0.4], iterations=2000, seed=42)
    assert one == two


def test_compare_models_aligns_ids_and_reports_critical_slice():
    baseline = [
        {
            "id": "1",
            "task_type": "function_call",
            "tool_name_correct": True,
            "param_score": 5,
            "invalid": False,
            "tags": {"slice": "critical"},
        },
        {
            "id": "2",
            "task_type": "function_call",
            "tool_name_correct": False,
            "param_score": 2,
            "invalid": False,
        },
    ]
    candidate = [
        {
            "id": "1",
            "task_type": "function_call",
            "tool_name_correct": False,
            "param_score": 3,
            "invalid": False,
            "tags": {"slice": "critical"},
        },
        {
            "id": "3",
            "task_type": "function_call",
            "tool_name_correct": True,
            "param_score": 5,
            "invalid": False,
        },
    ]
    comparison = compare_models(baseline, candidate, critical_tags=["critical"])
    assert comparison["n"] == 3
    assert comparison["baseline_only_correct"] == 1
    assert comparison["candidate_only_correct"] == 1
    assert comparison["worst_critical_slice"]["rate_regression"] == 1.0
    assert comparison["worst_regressions"][0]["id"] == "1"
    assert comparison["worst_regressions"][0]["delta"] == -2.0


def test_compare_models_reports_requested_critical_slice_without_samples():
    rows = [
        {
            "id": "1",
            "task_type": "subjective",
            "score": 4,
            "invalid": False,
            "tags": {"slice": "ordinary"},
        }
    ]
    comparison = compare_models(
        rows, rows, critical_tags=["must_cover"]
    )
    assert comparison["missing_critical_slices"] == ["must_cover"]


def test_compare_models_matches_each_value_in_multi_source_tag():
    row = {
        "id": "1",
        "task_type": "subjective",
        "score": 4,
        "invalid": False,
        "tags": {"source_doc": ["manual-a.pdf", "manual-b.pdf"]},
    }

    comparison = compare_models(
        [row], [row], critical_tags=["source_doc=manual-b.pdf"]
    )

    assert comparison["missing_critical_slices"] == []
    assert comparison["slices"]["source_doc=manual-b.pdf"]["n"] == 1


def test_subjective_critical_slice_counts_invalid_candidate_as_regression():
    baseline = [
        {
            "id": "1",
            "task_type": "subjective",
            "score": 4,
            "invalid": False,
            "tags": {"slice": "critical"},
        }
    ]
    candidate = [
        {
            "id": "1",
            "task_type": "subjective",
            "score": None,
            "invalid": True,
            "invalid_reason": "missing_prediction",
            "tags": {"slice": "critical"},
        }
    ]

    comparison = compare_models(
        baseline, candidate, critical_tags=["critical"]
    )

    critical = comparison["slices"]["critical"]
    assert critical["baseline_correct_rate"] == 1.0
    assert critical["candidate_correct_rate"] == 0.0
    assert critical["candidate_invalid_rate"] == 1.0
    assert critical["invalid_rate_regression"] == 1.0
    assert critical["mean_score_delta"] == -4.0
    assert comparison["critical_slice_rate_regression"] == 1.0


def test_malformed_fc_scores_do_not_fall_back_to_tool_name_boolean():
    malformed = [
        {
            "id": str(index),
            "task_type": "function_call",
            "tool_name_correct": True,
        }
        for index in range(30)
    ]

    comparison = compare_models(malformed, malformed)

    assert comparison["n"] == 30
    assert comparison["paired_score_n"] == 0
    assert comparison["mean_delta"] is None


def test_out_of_range_judge_scores_are_not_business_evidence():
    malformed = [
        {
            "id": str(index),
            "task_type": "subjective",
            "score": 999.0,
            "invalid": False,
        }
        for index in range(30)
    ]

    comparison = compare_models(malformed, malformed)

    assert comparison["paired_score_n"] == 0
    assert comparison["mean_delta"] is None


def test_retryable_judge_failures_are_not_business_zero_scores():
    retryable = [
        {
            "id": str(index),
            "task_type": "subjective",
            "score": None,
            "reason": "judge_score_invalid",
            "invalid": True,
        }
        for index in range(30)
    ]

    comparison = compare_models(retryable, retryable)

    assert comparison["paired_score_n"] == 0
    assert comparison["mean_delta"] is None


def test_critical_slice_gates_take_separate_worst_rate_and_score_slices():
    baseline = [
        {
            "id": "rate",
            "task_type": "subjective",
            "score": 4.0,
            "invalid": False,
            "tags": {"slice": "rate"},
        },
        {
            "id": "score",
            "task_type": "subjective",
            "score": 5.0,
            "invalid": False,
            "tags": {"slice": "score"},
        },
    ]
    candidate = [
        {
            "id": "rate",
            "task_type": "subjective",
            "score": 3.9,
            "invalid": False,
            "tags": {"slice": "rate"},
        },
        {
            "id": "score",
            "task_type": "subjective",
            "score": 4.0,
            "invalid": False,
            "tags": {"slice": "score"},
        },
    ]

    comparison = compare_models(
        baseline, candidate, critical_tags=["rate", "score"]
    )

    assert comparison["critical_slice_rate_regression"] == 1.0
    assert comparison["critical_slice_score_regression"] == 1.0
    assert comparison["worst_critical_rate_slice"]["tag"] == "rate"
    assert comparison["worst_critical_score_slice"]["tag"] == "score"
    diagnosis = diagnose_evaluation(
        baseline={"primary": 0.6, "invalid_rate": 0.0},
        candidate={"primary": 0.8, "invalid_rate": 0.0},
        paired={**comparison, "n": 60},
        criteria=criteria(),
        training={"loss_trend": "down"},
    )
    assert not diagnosis.accepted
    assert diagnosis.category == "data_coverage_gap"


def test_invalid_regression_is_protocol_diagnosis():
    diagnosis = diagnose_evaluation(
        baseline={"primary": 0.70, "invalid_rate": 0.01},
        candidate={"primary": 0.80, "invalid_rate": 0.08},
        paired={
            "n": 100,
            "mean_delta": 0.10,
            "critical_slice_rate_regression": 0.0,
            "critical_slice_score_regression": 0.0,
        },
        criteria=criteria(),
        training={"loss_trend": "down"},
    )
    assert not diagnosis.accepted
    assert diagnosis.category == "template_or_protocol_mismatch"


def test_candidate_is_accepted_only_when_all_gates_pass():
    diagnosis = diagnose_evaluation(
        baseline={"primary": 0.70, "invalid_rate": 0.01},
        candidate={"primary": 0.90, "invalid_rate": 0.01},
        paired={
            "n": 200,
            "mean_delta": 0.20,
            "critical_slice_rate_regression": 0.0,
            "critical_slice_score_regression": 0.0,
        },
        criteria=criteria(),
        training={"loss_trend": "down"},
    )
    assert diagnosis.accepted
    assert diagnosis.category == "accept_candidate"


def test_candidate_is_rejected_when_non_regression_metric_declines():
    guarded = criteria().model_copy(
        update={"non_regression_metrics": {"tool_name_accuracy": 0.0}}
    )
    diagnosis = diagnose_evaluation(
        baseline={
            "primary": 3.5,
            "invalid_rate": 0.01,
            "metrics": {"tool_name_accuracy": 0.90},
        },
        candidate={
            "primary": 3.8,
            "invalid_rate": 0.01,
            "metrics": {"tool_name_accuracy": 0.70},
        },
        paired={
            "n": 200,
            "mean_delta": 0.30,
            "critical_slice_rate_regression": 0.0,
            "critical_slice_score_regression": 0.0,
        },
        criteria=guarded,
        training={"loss_trend": "down"},
    )
    assert not diagnosis.accepted
    assert diagnosis.category == "data_coverage_gap"
    assert "non_regression:tool_name_accuracy" in " ".join(diagnosis.evidence)


def test_small_evaluation_is_manual_review_only():
    diagnosis = diagnose_evaluation(
        baseline={"primary": 0.5, "invalid_rate": 0.0},
        candidate={"primary": 0.9, "invalid_rate": 0.0},
        paired={"n": 10, "mean_delta": 0.4},
        criteria=criteria(),
        training={"loss_trend": "down"},
    )
    assert diagnosis.category == "evaluation_quality_issue"
    assert not diagnosis.accepted


def test_small_requested_task_slice_is_manual_review_even_if_total_is_large():
    diagnosis = diagnose_evaluation(
        baseline={"primary": 0.5, "invalid_rate": 0.0},
        candidate={"primary": 0.9, "invalid_rate": 0.0},
        paired={
            "n": 60,
            "minimum_requested_task_slice_n": 20,
            "mean_delta": 0.4,
        },
        criteria=criteria(),
        training={"loss_trend": "down"},
    )
    assert diagnosis.category == "evaluation_quality_issue"
    assert not diagnosis.accepted


def test_one_sided_missing_predictions_can_never_be_accepted():
    diagnosis = diagnose_evaluation(
        baseline={"primary": 0.0, "invalid_rate": 0.0},
        candidate={"primary": 0.9, "invalid_rate": 0.0},
        paired={
            "n": 100,
            "mean_delta": 0.9,
            "baseline_invalid_rate": 1.0,
            "candidate_invalid_rate": 0.0,
            "critical_slice_rate_regression": 0.0,
            "critical_slice_score_regression": 0.0,
        },
        criteria=criteria(),
        training={"loss_trend": "down"},
    )
    assert diagnosis.category == "evaluation_quality_issue"
    assert not diagnosis.accepted
