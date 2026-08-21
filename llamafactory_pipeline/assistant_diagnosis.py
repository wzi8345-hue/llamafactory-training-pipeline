"""Dependency-free paired evaluation statistics and deterministic diagnosis."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any, Iterable

from .assistant_schema import EvaluationDiagnosis, SuccessCriteria


def exact_mcnemar_p(
    baseline_only_correct: int, candidate_only_correct: int
) -> float:
    n = baseline_only_correct + candidate_only_correct
    if n == 0:
        return 1.0
    k = min(baseline_only_correct, candidate_only_correct)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def paired_bootstrap_ci(
    deltas: list[float], iterations: int = 2000, seed: int = 42
) -> dict[str, float | None]:
    if not deltas:
        return {"mean": None, "low": None, "high": None}
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    rng = random.Random(seed)
    n = len(deltas)
    means = []
    for _ in range(iterations):
        means.append(sum(deltas[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    low = means[math.floor(0.025 * (iterations - 1))]
    high = means[math.ceil(0.975 * (iterations - 1))]
    return {"mean": sum(deltas) / n, "low": low, "high": high}


def _numeric_score(row: dict[str, Any] | None) -> float | None:
    # Missing/invalid paired output is a business failure, not a row to drop
    # from the delta denominator.
    if not row:
        return 0.0
    if row.get("invalid") is True:
        # A persisted prediction failure has an invalid_reason and counts as a
        # business zero.  A judge-only failure must be retried, never used to
        # inflate either model's paired delta.
        if not row.get("invalid_reason"):
            return None
        return 0.0
    if row.get("invalid") is not False:
        return None
    key = (
        "param_score"
        if row.get("task_type") == "function_call"
        else "score" if row.get("task_type") == "subjective" else ""
    )
    value = row.get(key) if key else None
    if (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and 1.0 <= float(value) <= 5.0
    ):
        return float(value)
    return None


def _correct(row: dict[str, Any] | None) -> bool:
    if not row or row.get("invalid") is not False:
        return False
    if row.get("task_type") == "subjective":
        score = row.get("score")
        return (
            isinstance(score, (int, float))
            and not isinstance(score, bool)
            and math.isfinite(float(score))
            and 1.0 <= float(score) <= 5.0
            and float(score) >= 4.0
        )
    if row.get("task_type") == "function_call":
        score = row.get("param_score")
        return row.get("tool_name_correct") is True and (
            isinstance(score, (int, float))
            and not isinstance(score, bool)
            and math.isfinite(float(score))
            and 1.0 <= float(score) <= 5.0
            and float(score) >= 4.0
        )
    return False


def _tag_values(*rows: dict[str, Any] | None) -> set[str]:
    values = set()
    for row in rows:
        tags = (row or {}).get("tags") or {}
        if not isinstance(tags, dict):
            continue
        for key, value in tags.items():
            members = value if isinstance(value, list) else [value]
            values.add(str(key))
            for member in members:
                values.add(str(member))
                values.add(f"{key}={member}")
    return values


def compare_models(
    baseline_scores: list[dict[str, Any]],
    candidate_scores: list[dict[str, Any]],
    *,
    critical_tags: Iterable[str] = (),
    bootstrap_iterations: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    """Align rows by ID, retain missing rows as invalid, and calculate paired evidence."""
    baseline = {row["id"]: row for row in baseline_scores if row.get("id")}
    candidate = {row["id"]: row for row in candidate_scores if row.get("id")}
    ids = sorted(set(baseline) | set(candidate))
    critical = set(critical_tags)
    baseline_only = 0
    candidate_only = 0
    baseline_missing = 0
    candidate_missing = 0
    baseline_invalid = 0
    candidate_invalid = 0
    wins = ties = losses = 0
    deltas: list[float] = []
    regressions: list[dict[str, Any]] = []
    slice_rows: dict[str, list[tuple[dict | None, dict | None]]] = defaultdict(list)

    for item_id in ids:
        base_row = baseline.get(item_id)
        candidate_row = candidate.get(item_id)
        baseline_missing += int(base_row is None)
        candidate_missing += int(candidate_row is None)
        base_invalid = base_row is None or bool(base_row.get("invalid"))
        cand_invalid = candidate_row is None or bool(candidate_row.get("invalid"))
        baseline_invalid += int(base_invalid)
        candidate_invalid += int(cand_invalid)
        base_correct = _correct(base_row)
        cand_correct = _correct(candidate_row)
        baseline_only += int(base_correct and not cand_correct)
        candidate_only += int(cand_correct and not base_correct)
        base_score = _numeric_score(base_row)
        cand_score = _numeric_score(candidate_row)
        if base_score is not None and cand_score is not None:
            delta = cand_score - base_score
            deltas.append(delta)
            if delta > 0:
                wins += 1
            elif delta < 0:
                losses += 1
                reason = ""
                if candidate_row:
                    reason = str(
                        candidate_row.get("reason")
                        or candidate_row.get("param_reason")
                        or candidate_row.get("invalid_reason")
                        or ""
                    )[:120]
                regressions.append(
                    {
                        "id": str(item_id)[:120],
                        "delta": delta,
                        "baseline_missing": base_row is None,
                        "candidate_missing": candidate_row is None,
                        "baseline_invalid": base_invalid,
                        "candidate_invalid": cand_invalid,
                        "candidate_reason": reason,
                        "tags": sorted(_tag_values(base_row, candidate_row))[:20],
                    }
                )
            else:
                ties += 1
        for tag in _tag_values(base_row, candidate_row):
            slice_rows[tag].append((base_row, candidate_row))

    slices = {}
    for tag, rows in sorted(slice_rows.items()):
        base_rate = sum(_correct(base) for base, _ in rows) / len(rows)
        cand_rate = sum(_correct(candidate) for _, candidate in rows) / len(rows)
        base_invalid_rate = sum(
            base is None or bool(base.get("invalid")) for base, _ in rows
        ) / len(rows)
        candidate_invalid_rate = sum(
            candidate is None or bool(candidate.get("invalid"))
            for _, candidate in rows
        ) / len(rows)
        score_deltas = []
        for base_row, candidate_row in rows:
            base_score = _numeric_score(base_row)
            candidate_score = _numeric_score(candidate_row)
            if base_score is not None and candidate_score is not None:
                score_deltas.append(candidate_score - base_score)
        mean_delta = sum(score_deltas) / len(score_deltas) if score_deltas else None
        slices[tag] = {
            "n": len(rows),
            "baseline_correct_rate": base_rate,
            "candidate_correct_rate": cand_rate,
            "rate_regression": max(0.0, base_rate - cand_rate),
            "baseline_invalid_rate": base_invalid_rate,
            "candidate_invalid_rate": candidate_invalid_rate,
            "invalid_rate_regression": max(
                0.0, candidate_invalid_rate - base_invalid_rate
            ),
            "mean_score_delta": mean_delta,
            "score_regression": max(0.0, -(mean_delta or 0.0)),
        }
    critical_slices = {
        tag: value for tag, value in slices.items() if tag in critical
    }
    missing_critical_slices = sorted(critical - set(critical_slices))
    worst = None
    worst_rate = None
    worst_score = None
    if critical_slices:
        worst_tag, worst_value = max(
            critical_slices.items(),
            key=lambda item: (
                max(item[1]["rate_regression"], item[1]["score_regression"]),
                item[0],
            ),
        )
        worst = {"tag": worst_tag, **worst_value}
        worst_rate_tag, worst_rate_value = max(
            critical_slices.items(),
            key=lambda item: (item[1]["rate_regression"], item[0]),
        )
        worst_rate = {"tag": worst_rate_tag, **worst_rate_value}
        worst_score_tag, worst_score_value = max(
            critical_slices.items(),
            key=lambda item: (item[1]["score_regression"], item[0]),
        )
        worst_score = {"tag": worst_score_tag, **worst_score_value}
    bootstrap = paired_bootstrap_ci(
        deltas, iterations=bootstrap_iterations, seed=seed
    )
    return {
        "n": len(ids),
        "baseline_invalid": baseline_invalid,
        "candidate_invalid": candidate_invalid,
        "baseline_missing": baseline_missing,
        "candidate_missing": candidate_missing,
        "baseline_invalid_rate": baseline_invalid / len(ids) if ids else None,
        "candidate_invalid_rate": candidate_invalid / len(ids) if ids else None,
        "baseline_missing_rate": baseline_missing / len(ids) if ids else None,
        "candidate_missing_rate": candidate_missing / len(ids) if ids else None,
        "baseline_only_correct": baseline_only,
        "candidate_only_correct": candidate_only,
        "mcnemar_p": exact_mcnemar_p(baseline_only, candidate_only),
        "paired_score_n": len(deltas),
        "mean_delta": bootstrap["mean"],
        "bootstrap_low": bootstrap["low"],
        "bootstrap_high": bootstrap["high"],
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "worst_regressions": sorted(
            regressions, key=lambda row: (row["delta"], row["id"])
        )[:5],
        "slices": slices,
        "missing_critical_slices": missing_critical_slices,
        "worst_critical_slice": worst,
        "worst_critical_rate_slice": worst_rate,
        "worst_critical_score_slice": worst_score,
        "critical_slice_rate_regression": (
            worst_rate["rate_regression"] if worst_rate else 0.0
        ),
        "critical_slice_score_regression": (
            worst_score["score_regression"] if worst_score else 0.0
        ),
    }


def diagnose_evaluation(
    *,
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    paired: dict[str, Any],
    criteria: SuccessCriteria,
    training: dict[str, Any],
) -> EvaluationDiagnosis:
    """Apply fixed-precedence acceptance and retraining rules."""
    baseline_primary = float(baseline.get("primary") or 0.0)
    candidate_primary = float(candidate.get("primary") or 0.0)
    improvement = candidate_primary - baseline_primary
    baseline_invalid = max(
        float(baseline.get("invalid_rate") or 0.0),
        float(paired.get("baseline_invalid_rate") or 0.0),
    )
    candidate_invalid = max(
        float(candidate.get("invalid_rate") or 0.0),
        float(paired.get("candidate_invalid_rate") or 0.0),
    )
    invalid_delta = candidate_invalid - baseline_invalid
    rate_regression = float(paired.get("critical_slice_rate_regression") or 0.0)
    score_regression = float(paired.get("critical_slice_score_regression") or 0.0)
    mean_delta = paired.get("mean_delta")
    baseline_metrics = baseline.get("metrics") or {}
    candidate_metrics = candidate.get("metrics") or {}
    missing_non_regression: list[str] = []
    failed_non_regression: list[tuple[str, float, float]] = []
    non_regression_evidence: list[str] = []
    for metric, minimum_delta in sorted(criteria.non_regression_metrics.items()):
        baseline_value = baseline_metrics.get(metric)
        candidate_value = candidate_metrics.get(metric)
        if not isinstance(baseline_value, (int, float)) or not isinstance(
            candidate_value, (int, float)
        ):
            missing_non_regression.append(metric)
            non_regression_evidence.append(f"non_regression:{metric}=missing")
            continue
        delta = float(candidate_value) - float(baseline_value)
        non_regression_evidence.append(
            f"non_regression:{metric}:delta={delta:.6f}:minimum={minimum_delta:.6f}"
        )
        if delta < minimum_delta:
            failed_non_regression.append((metric, delta, minimum_delta))
    evidence = [
        f"primary_delta={improvement:.6f}",
        f"invalid_rate_delta={invalid_delta:.6f}",
        f"critical_slice_rate_regression={rate_regression:.6f}",
        f"critical_slice_score_regression={score_regression:.6f}",
        *non_regression_evidence,
    ]

    category = "data_coverage_gap"
    summary = "整体门槛未全部通过，失败集中在局部样本或覆盖范围。"
    next_data: list[str] = ["按失败 ID 与标签切片补充覆盖并复核标注"]
    next_training: list[str] = []
    minimum_task_slice_n = paired.get("minimum_requested_task_slice_n")
    if int(paired.get("n") or 0) < 30 or (
        isinstance(minimum_task_slice_n, int) and minimum_task_slice_n < 30
    ):
        category = "evaluation_quality_issue"
        summary = "整体或任务切片的成对样本少于 30，仅可视为冒烟测试，需要人工复核。"
        next_data = []
    elif max(baseline_invalid, candidate_invalid) > 0.10:
        category = "evaluation_quality_issue"
        summary = "评测无效率超过 10%，当前结果不适合作为训练结论。"
        next_data = []
    elif invalid_delta > criteria.max_invalid_rate_increase:
        category = "template_or_protocol_mismatch"
        summary = "候选模型无效率增幅超过门槛，优先检查模板和工具协议。"
        next_data = []
        next_training = ["核对 template、tool_format 与函数调用输出协议后重训"]
    elif (
        rate_regression > criteria.max_critical_slice_rate_regression
        or score_regression > criteria.max_critical_slice_score_regression
    ):
        category = "data_coverage_gap"
        summary = "关键切片出现不可接受的回归，需要补充对应数据覆盖。"
    elif missing_non_regression:
        category = "evaluation_quality_issue"
        summary = "评测结果缺少必须的非回归指标，需先修复评测配置。"
        next_data = []
    elif failed_non_regression:
        category = "data_coverage_gap"
        summary = "候选模型的非回归指标低于门槛，不能接受本轮模型。"
    elif (
        training.get("finetune_type") == "dpo"
        and isinstance(mean_delta, (int, float))
        and (
            mean_delta < -0.10
            or (
                isinstance(paired.get("bootstrap_high"), (int, float))
                and paired["bootstrap_high"] < 0
            )
        )
    ):
        category = "annotation_or_pair_quality"
        summary = "DPO 协议有效但成对得分广泛下降，应优先复核偏好对质量。"
        next_data = ["抽查 chosen/rejected 顺序、难度与标签一致性并重构偏好对"]
    elif training.get("loss_trend") == "flat" and improvement < criteria.min_improvement:
        category = "underfit"
        summary = "训练损失基本持平且主指标提升不足，表现为欠拟合。"
        next_data = []
        next_training = ["在显存允许范围内调整学习率、epoch 或 LoRA 容量"]
    elif (
        training.get("loss_trend") == "down"
        and training.get("validation_trend") in {"worse", "up"}
        and improvement < criteria.min_improvement
    ):
        category = "overfit"
        summary = "训练损失下降但验证表现恶化，表现为过拟合。"
        next_data = []
        next_training = [
            "减少 epoch 或保守增加 LoRA dropout，加强验证与数据多样性并保留早停点"
        ]
    elif improvement >= criteria.min_improvement:
        return EvaluationDiagnosis(
            category="accept_candidate",
            accepted=True,
            summary="主指标提升达到门槛，且无效率和关键切片均未回归。",
            evidence=evidence,
        )

    return EvaluationDiagnosis(
        category=category,
        accepted=False,
        summary=summary,
        evidence=evidence,
        next_data_changes=next_data,
        next_training_changes=next_training,
    )
