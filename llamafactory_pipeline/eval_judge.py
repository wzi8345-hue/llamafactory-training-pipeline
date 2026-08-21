"""评测打分阶段 (本地, 可重跑): 拉回 predictions → LLM judge 打分 → 聚合报告。

judge 复用项目生成模型 (rag_eval_plan.common.build_llm)。纯逻辑函数 (匹配/解析/聚合)
不依赖网络, 可单测; judge 调用通过可注入的 callable 抽象。
"""

from __future__ import annotations

import json
import math
import os
import fcntl
from collections import Counter
from typing import Any, Callable, Optional

# judge 抽象: (system, user) -> LLM 原始文本
JudgeFn = Callable[[str, str], str]


class RetryableJudgeEvidenceError(RuntimeError):
    """Judge evidence was persisted but must be retried before diagnosis."""


# ── 纯逻辑 ──

def tool_name_correct(gold_name: str, pred_name: Optional[str]) -> bool:
    return bool(pred_name) and pred_name == gold_name


def parse_judge_score(raw: str) -> tuple[Optional[int], str]:
    """从 judge 输出解析 {score, reason}。解析失败返回 (None, 原文截断)。"""
    from rag_eval_plan.common import extract_json_object
    obj = extract_json_object(raw or "")
    score = obj.get("score")
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        return None, str(obj.get("reason") or (raw or ""))[:200]
    s = int(round(score))
    if s < 1 or s > 5:
        return None, f"score 越界: {score}"
    return s, str(obj.get("reason") or "")[:200]


def _mean(vals: list[Optional[float]]) -> Optional[float]:
    xs = [v for v in vals if v is not None]
    return round(sum(xs) / len(xs), 3) if xs else None


def _index_by_id(items: list[dict]) -> dict[str, dict]:
    return {it["id"]: it for it in items if isinstance(it, dict) and it.get("id")}


def _copy_prediction_evidence(row: dict, prediction: dict | None) -> None:
    if prediction is None:
        row["invalid_reason"] = "missing_prediction"
        return
    for key in ("latency_ms", "finish_reason", "invalid_reason"):
        if key in prediction:
            row[key] = prediction[key]


def score_model(
    judge: JudgeFn,
    fc_items: list[dict],
    subj_items: list[dict],
    preds: list[dict],
    fc_prompt: str,
    subj_prompt: str,
    *,
    existing_scores: list[dict] | None = None,
    on_score: Callable[[dict], None] | None = None,
) -> tuple[list[dict], dict[str, Any]]:
    """对单个模型的预测打分, 返回 (逐条得分, 汇总)。"""
    pred_by_id = _index_by_id(preds)
    scores: list[dict] = list(existing_scores or [])
    completed = {
        (row.get("task_type"), row.get("id"))
        for row in scores
        if row.get("task_type") and row.get("id")
    }

    def record(row: dict) -> None:
        if not _score_row_schema_valid(row):
            raise ValueError("judge produced a score row outside the evidence contract")
        scores.append(row)
        completed.add((row.get("task_type"), row.get("id")))
        if on_score is not None:
            on_score(row)

    for it in fc_items:
        if ("function_call", it["id"]) in completed:
            continue
        p = pred_by_id.get(it["id"])
        gold = it.get("gold", {})
        row: dict[str, Any] = {
            "id": it["id"], "task_type": "function_call",
            "gold_name": gold.get("name"),
            "pred_name": (p or {}).get("pred_name"),
        }
        if it.get("tags"):
            row["tags"] = dict(it["tags"])
        _copy_prediction_evidence(row, p)
        if p is None:
            row.update(tool_name_correct=False, param_score=None,
                       param_reason="缺少预测", invalid=True)
            record(row)
            continue
        row["tool_name_correct"] = tool_name_correct(gold.get("name", ""), p.get("pred_name"))
        if p.get("invalid_reason"):
            row.update(
                param_score=None,
                param_reason=str(p["invalid_reason"]),
                invalid=True,
            )
            record(row)
            continue
        user = (
            f"用户请求: {it['query']}\n"
            f"期望参数(gold): {json.dumps(gold.get('arguments', {}), ensure_ascii=False)}\n"
            f"模型参数(pred): {json.dumps(p.get('pred_arguments', {}), ensure_ascii=False)}"
        )
        s, reason = parse_judge_score(judge(fc_prompt, user))
        row.update(
            param_score=s,
            param_reason=reason or "judge_score_invalid",
            invalid=(s is None or bool(row.get("invalid_reason"))),
        )
        record(row)

    for it in subj_items:
        if ("subjective", it["id"]) in completed:
            continue
        p = pred_by_id.get(it["id"])
        row = {"id": it["id"], "task_type": "subjective"}
        if it.get("tags"):
            row["tags"] = dict(it["tags"])
        _copy_prediction_evidence(row, p)
        if p is None or "answer" not in p:
            row.update(score=None, reason="缺少预测", invalid=True)
            record(row)
            continue
        if p.get("invalid_reason"):
            row.update(score=None, reason=p["invalid_reason"], invalid=True)
            record(row)
            continue
        user = (
            f"问题: {it['query']}\n"
            f"参考答案: {it.get('reference') or '(无)'}\n"
            f"模型回答: {p.get('answer', '')}"
        )
        s, reason = parse_judge_score(judge(subj_prompt, user))
        row.update(
            score=s,
            reason=reason or "judge_score_invalid",
            invalid=(s is None),
        )
        record(row)

    return scores, summarize(scores)


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    if len(xs) == 1:
        return float(xs[0])
    pos = (len(xs) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(xs[lo])
    return round(xs[lo] + (xs[hi] - xs[lo]) * (pos - lo), 3)


def _failure_evidence(rows: list[dict], judge_reason_key: str) -> dict[str, Any]:
    invalid_reasons = Counter(
        str(row["invalid_reason"])[:120]
        for row in rows
        if row.get("invalid_reason")
    )
    finish_reasons = Counter(
        str(row["finish_reason"])[:120]
        for row in rows
        if row.get("finish_reason")
    )
    judge_reasons = Counter(
        str(row[judge_reason_key])[:120]
        for row in rows
        if row.get("invalid") and row.get(judge_reason_key)
    )
    examples = []
    for row in rows:
        if not row.get("invalid"):
            continue
        example = {"id": str(row.get("id", ""))[:120]}
        if row.get("invalid_reason"):
            example["invalid_reason"] = str(row["invalid_reason"])[:120]
        if row.get("finish_reason"):
            example["finish_reason"] = str(row["finish_reason"])[:120]
        if row.get(judge_reason_key):
            example["judge_reason"] = str(row[judge_reason_key])[:120]
        examples.append(example)
        if len(examples) >= 5:
            break
    quality_issue_counts: Counter[str] = Counter()
    quality_reason_counts: Counter[str] = Counter()
    quality_examples = []
    for row in rows:
        if row.get("invalid"):
            continue
        issues = []
        if row.get("task_type") == "function_call":
            if row.get("tool_name_correct") is False:
                issues.append("tool_name_incorrect")
            if isinstance(row.get("param_score"), (int, float)) and (
                float(row["param_score"]) < 4
            ):
                issues.append("param_score_below_4")
        elif isinstance(row.get("score"), (int, float)) and (
            float(row["score"]) < 4
        ):
            issues.append("score_below_4")
        if not issues:
            continue
        quality_issue_counts.update(issues)
        reason = str(row.get(judge_reason_key) or "").strip()[:120]
        if reason:
            quality_reason_counts[reason] += 1
        example = {
            "id": str(row.get("id", ""))[:120],
            "issues": issues,
        }
        if reason:
            example["judge_reason"] = reason
        tags = row.get("tags")
        if isinstance(tags, dict):
            example["tags"] = {
                str(key)[:80]: str(value)[:80]
                for key, value in list(tags.items())[:10]
            }
        quality_examples.append(example)
        if len(quality_examples) >= 5:
            break
    return {
        "invalid_reason_counts": dict(sorted(invalid_reasons.items())),
        "finish_reason_counts": dict(sorted(finish_reasons.items())),
        "judge_failure_reason_counts": dict(sorted(judge_reasons.items())),
        "failure_examples": examples,
        "quality_issue_counts": dict(sorted(quality_issue_counts.items())),
        "quality_reason_counts": dict(sorted(quality_reason_counts.items())),
        "quality_examples": quality_examples,
    }


def summarize(scores: list[dict]) -> dict[str, Any]:
    fc = [r for r in scores if r["task_type"] == "function_call"]
    subj = [r for r in scores if r["task_type"] == "subjective"]
    out: dict[str, Any] = {}
    if fc:
        latencies = [
            float(r["latency_ms"])
            for r in fc
            if isinstance(r.get("latency_ms"), (int, float))
        ]
        param_score_mean = _mean([r.get("param_score") for r in fc])
        combined_accuracy = _mean(
            [
                1.0
                if r.get("tool_name_correct")
                and isinstance(r.get("param_score"), (int, float))
                and r["param_score"] >= 4
                else 0.0
                for r in fc
            ]
        )
        out["function_call"] = {
            "n": len(fc),
            "tool_name_accuracy": _mean([1.0 if r.get("tool_name_correct") else 0.0 for r in fc]),
            "param_score_mean": param_score_mean,
            "param_score": param_score_mean,
            "combined_accuracy": combined_accuracy,
            "invalid": sum(1 for r in fc if r.get("invalid")),
            "invalid_rate": round(sum(1 for r in fc if r.get("invalid")) / len(fc), 6),
            "no_tool_call_rate": round(
                sum(1 for r in fc if r.get("invalid_reason") == "no_tool_call") / len(fc),
                6,
            ),
            "latency_p50_ms": percentile(latencies, 0.50),
            "latency_p95_ms": percentile(latencies, 0.95),
            **_failure_evidence(fc, "param_reason"),
        }
    if subj:
        latencies = [
            float(r["latency_ms"])
            for r in subj
            if isinstance(r.get("latency_ms"), (int, float))
        ]
        score_mean = _mean([r.get("score") for r in subj])
        answer_accuracy = (
            round(score_mean / 5.0, 6) if score_mean is not None else None
        )
        out["subjective"] = {
            "n": len(subj),
            "score_mean": score_mean,
            "answer_accuracy": answer_accuracy,
            "combined_accuracy": answer_accuracy,
            "invalid": sum(1 for r in subj if r.get("invalid")),
            "invalid_rate": round(
                sum(1 for r in subj if r.get("invalid")) / len(subj), 6
            ),
            "latency_p50_ms": percentile(latencies, 0.50),
            "latency_p95_ms": percentile(latencies, 0.95),
            **_failure_evidence(subj, "reason"),
        }
    return out


def build_report_md(
    eval_id: str,
    per_model: dict[str, dict],
    paired: dict[str, Any] | None = None,
) -> str:
    """多模型对比报告。per_model: {model_name: summary}。"""
    lines = [f"# 评测报告 {eval_id}", ""]
    has_fc = any("function_call" in s for s in per_model.values())
    has_subj = any("subjective" in s for s in per_model.values())

    if has_fc:
        lines += ["## Function Call", "",
                  "| 模型 | 样本 | 工具名正确率 | 参数均分(1-5) | 无效率 | P50 ms | P95 ms |",
                  "|------|------|------|------|------|------|------|"]
        for name, s in per_model.items():
            fc = s.get("function_call")
            if fc:
                lines.append(
                    f"| {name} | {fc['n']} | {_fmt(fc['tool_name_accuracy'])} "
                    f"| {_fmt(fc['param_score_mean'])} | {_pct(fc.get('invalid_rate'))} "
                    f"| {_fmt(fc.get('latency_p50_ms'))} "
                    f"| {_fmt(fc.get('latency_p95_ms'))} |"
                )
        lines.append("")

    if has_subj:
        lines += ["## 主观任务", "",
                  "| 模型 | 样本 | 得分均值(1-5) | 无效率 | P50 ms | P95 ms |",
                  "|------|------|------|------|------|------|"]
        for name, s in per_model.items():
            sj = s.get("subjective")
            if sj:
                lines.append(
                    f"| {name} | {sj['n']} | {_fmt(sj['score_mean'])} "
                    f"| {_pct(sj.get('invalid_rate'))} "
                    f"| {_fmt(sj.get('latency_p50_ms'))} "
                    f"| {_fmt(sj.get('latency_p95_ms'))} |"
                )
        lines.append("")

    if paired:
        lines += [
            "## 成对对照证据",
            "",
            f"- paired n={int(paired.get('n') or 0)}; "
            f"business scores={int(paired.get('paired_score_n') or 0)}/"
            f"{int(paired.get('n') or 0)}; "
            f"baseline missing={int(paired.get('baseline_missing') or 0)}; "
            f"candidate missing={int(paired.get('candidate_missing') or 0)}",
            f"- win / tie / loss: {int(paired.get('wins') or 0)} / "
            f"{int(paired.get('ties') or 0)} / {int(paired.get('losses') or 0)}",
            f"- mean delta={_fmt(paired.get('mean_delta'))}; "
            f"95% bootstrap CI=[{_fmt(paired.get('bootstrap_low'))}, "
            f"{_fmt(paired.get('bootstrap_high'))}]",
            f"- McNemar exact p={_fmt6(paired.get('mcnemar_p'))}",
            "",
        ]
        slices = paired.get("slices") or {}
        if slices:
            lines += [
                "### 切片证据",
                "",
                "| 切片 | n | baseline correct | candidate correct | rate regression | invalid regression | score delta | score regression |",
                "|------|------|------|------|------|------|------|------|",
            ]
            for tag, row in sorted(slices.items()):
                lines.append(
                    f"| {tag} | {int(row.get('n') or 0)} "
                    f"| {_pct(row.get('baseline_correct_rate'))} "
                    f"| {_pct(row.get('candidate_correct_rate'))} "
                    f"| {_pct(row.get('rate_regression'))} "
                    f"| {_pct(row.get('invalid_rate_regression'))} "
                    f"| {_fmt(row.get('mean_score_delta'))} "
                    f"| {_fmt(row.get('score_regression'))} |"
                )
            lines.append("")
        missing_slices = paired.get("missing_critical_slices") or []
        if missing_slices:
            lines += [
                "### 缺失的关键切片",
                "",
                "- " + ", ".join(str(tag) for tag in missing_slices),
                "",
            ]
    failure_lines = []
    for name, summary in per_model.items():
        for task_type in ("function_call", "subjective"):
            task = summary.get(task_type) or {}
            groups = []
            for label, key in (
                ("invalid", "invalid_reason_counts"),
                ("finish", "finish_reason_counts"),
                ("judge", "judge_failure_reason_counts"),
                ("quality", "quality_issue_counts"),
                ("quality_reason", "quality_reason_counts"),
            ):
                counts = task.get(key) or {}
                if counts:
                    rendered = ", ".join(
                        f"{reason}={count}"
                        for reason, count in sorted(counts.items())
                    )
                    groups.append(f"{label}: {rendered}")
            if groups:
                failure_lines.append(
                    f"- {name}/{task_type}: " + "; ".join(groups)
                )
    if failure_lines:
        lines += ["## 失败原因", "", *failure_lines, ""]
    quality_example_lines = []
    for name, summary in per_model.items():
        for task_type in ("function_call", "subjective"):
            for example in (summary.get(task_type) or {}).get(
                "quality_examples", []
            )[:5]:
                issues = ",".join(str(item) for item in example.get("issues") or [])
                reason = str(example.get("judge_reason") or "")
                quality_example_lines.append(
                    f"- {name}/{task_type} id={example.get('id', '')}; "
                    f"issues={issues}; reason={reason}"
                )
    if paired:
        for example in (paired.get("worst_regressions") or [])[:5]:
            quality_example_lines.append(
                f"- paired id={example.get('id', '')}; "
                f"delta={_fmt(example.get('delta'))}; "
                f"reason={example.get('candidate_reason', '')}"
            )
    if quality_example_lines:
        lines += ["## 低质量与最大回归样例", "", *quality_example_lines, ""]
    return "\n".join(lines)


def _fmt(v: Optional[float]) -> str:
    return "-" if v is None else f"{v:.3f}"


def _fmt6(v: Optional[float]) -> str:
    return "-" if v is None else f"{v:.6f}"


def _pct(v: Optional[float]) -> str:
    return "-" if v is None else f"{v * 100:.3f}%"


# ── judge 工厂 + 协调 ──

def make_default_judge() -> JudgeFn:
    """复用项目生成模型做 judge。"""
    from rag_eval_plan.common import build_llm
    llm = build_llm()

    def _judge(system: str, user: str) -> str:
        return llm.chat(system=system, user=user, temperature=0.0,
                        disable_thinking=True).get("answer", "")

    return _judge


def _load_jsonl(path: str, *, repair_trailing: bool = False) -> list[dict]:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return []
    with open(path, "rb") as handle:
        raw = handle.read()
    rows: list[dict] = []
    valid_end = 0
    lines = raw.splitlines(keepends=True)
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            valid_end += len(line)
            continue
        try:
            rows.append(json.loads(stripped.decode("utf-8")))
        except (UnicodeDecodeError, json.JSONDecodeError):
            recoverable_tail = (
                repair_trailing
                and index == len(lines) - 1
                and not line.endswith((b"\n", b"\r"))
            )
            if not recoverable_tail:
                raise
            with open(path, "r+b") as handle:
                handle.truncate(valid_end)
            break
        valid_end += len(line)
    else:
        if repair_trailing and raw and not raw.endswith((b"\n", b"\r")):
            with open(path, "ab") as handle:
                handle.write(b"\n")
    return rows


def _valid_judge_score(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and 1.0 <= float(value) <= 5.0
    )


def _valid_tags(tags: Any) -> bool:
    if not isinstance(tags, dict):
        return False
    for key, value in tags.items():
        if not isinstance(key, str) or not key:
            return False
        members = value if isinstance(value, list) else [value]
        if not members or not all(
            isinstance(member, str) and bool(member) for member in members
        ):
            return False
    return True


def _score_row_schema_valid(row: Any) -> bool:
    if not isinstance(row, dict):
        return False
    if not isinstance(row.get("id"), str) or not row["id"]:
        return False
    task_type = row.get("task_type")
    if task_type not in {"function_call", "subjective"}:
        return False
    if not isinstance(row.get("invalid"), bool):
        return False
    if "tags" in row and not _valid_tags(row["tags"]):
        return False
    latency = row.get("latency_ms")
    if "latency_ms" in row and not (
        isinstance(latency, (int, float))
        and not isinstance(latency, bool)
        and math.isfinite(float(latency))
        and float(latency) >= 0
    ):
        return False
    if "finish_reason" in row and row["finish_reason"] is not None and not isinstance(
        row["finish_reason"], str
    ):
        return False
    for key in ("invalid_reason", "param_reason", "reason"):
        if key in row and not isinstance(row[key], str):
            return False
    score = row.get("param_score" if task_type == "function_call" else "score")
    if task_type == "function_call" and not isinstance(
        row.get("tool_name_correct"), bool
    ):
        return False
    if row["invalid"]:
        reasons = [
            row.get(key) for key in ("invalid_reason", "param_reason", "reason")
        ]
        return (
            any(isinstance(reason, str) and bool(reason) for reason in reasons)
            and (score is None or _valid_judge_score(score))
        )
    return _valid_judge_score(score)


def _retryable_judge_failure(row: Any) -> bool:
    """Retry malformed cache rows and judge failures with otherwise usable output."""
    if not _score_row_schema_valid(row):
        return True
    invalid = row.get("invalid")
    if invalid is False:
        return False
    return not bool(row.get("invalid_reason"))


def _normalized_existing_scores(rows: list[Any]) -> list[dict[str, Any]]:
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        if _retryable_judge_failure(row):
            continue
        key = (str(row.get("task_type") or ""), str(row.get("id") or ""))
        if all(key):
            latest[key] = row
    return list(latest.values())


def run_scoring(
    cfg, eval_id: str, req, fc_items: list[dict], subj_items: list[dict],
    results_dir: str, judge: Optional[JudgeFn] = None,
    *,
    expected_prediction_keys: set[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    """拉回各模型 predictions → 打分 → 写 scores/report/config。返回汇总。"""
    from . import eval_remote

    os.makedirs(results_dir, exist_ok=True)
    judge = judge or make_default_judge()
    per_model: dict[str, dict] = {}

    for m in req.models:
        pred_path = os.path.join(results_dir, f"{m.name}.predictions.jsonl")
        score_path = os.path.join(results_dir, f"{m.name}.scores.jsonl")
        lock_path = score_path + ".lock"
        with open(lock_path, "a", encoding="utf-8") as lock_handle:
            fcntl.flock(lock_handle, fcntl.LOCK_EX)
            try:
                eval_remote.fetch_predictions(cfg, eval_id, m.name, pred_path)
            except Exception as e:  # 某模型缺 predictions 不阻断其余
                per_model[m.name] = {"error": str(e)[:200]}
                continue
            preds = _load_jsonl(pred_path)
            if expected_prediction_keys is not None:
                if not all(isinstance(row, dict) for row in preds):
                    raise ValueError(
                        f"{m.name} predictions contain a non-object row"
                    )
                prediction_keys = [
                    (
                        str(row.get("task_type") or ""),
                        str(row.get("id") or ""),
                    )
                    for row in preds
                ]
                if (
                    len(prediction_keys) != len(expected_prediction_keys)
                    or set(prediction_keys) != expected_prediction_keys
                ):
                    raise ValueError(
                        f"{m.name} prediction ID set does not match frozen manifest"
                    )
            loaded_scores = _load_jsonl(score_path, repair_trailing=True)
            if expected_prediction_keys is not None:
                score_keys = {
                    (
                        str(row.get("task_type") or ""),
                        str(row.get("id") or ""),
                    )
                    for row in loaded_scores
                    if isinstance(row, dict)
                }
                if not score_keys <= expected_prediction_keys:
                    raise ValueError(
                        f"{m.name} score ID set does not match frozen manifest"
                    )
            existing_scores = _normalized_existing_scores(loaded_scores)
            if existing_scores != loaded_scores:
                replacement = score_path + ".repair"
                with open(replacement, "w", encoding="utf-8") as handle:
                    for row in existing_scores:
                        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(replacement, score_path)
            with open(score_path, "a", encoding="utf-8") as handle:
                def persist(row: dict) -> None:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    handle.flush()

                scores, summary = score_model(
                    judge,
                    fc_items,
                    subj_items,
                    preds,
                    req.fc_param_prompt,
                    req.subjective_prompt,
                    existing_scores=existing_scores,
                    on_score=persist,
                )
                retryable = [
                    row for row in scores if _retryable_judge_failure(row)
                ]
                if retryable:
                    raise RetryableJudgeEvidenceError(
                        f"{m.name} has {len(retryable)} retryable judge rows"
                    )
                if expected_prediction_keys is not None:
                    completed_keys = {
                        (str(row.get("task_type") or ""), str(row.get("id") or ""))
                        for row in scores
                    }
                    if completed_keys != expected_prediction_keys:
                        raise ValueError(
                            f"{m.name} score evidence is incomplete for frozen manifest"
                        )
        per_model[m.name] = summary

    report = build_report_md(eval_id, {k: v for k, v in per_model.items() if "error" not in v})
    with open(os.path.join(results_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write(report)
    return {"eval_id": eval_id, "per_model": per_model, "report_md": report}
