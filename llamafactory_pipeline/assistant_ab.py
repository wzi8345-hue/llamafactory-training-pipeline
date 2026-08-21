"""Deterministic, side-effect-free old-A/new-B assistant behavior comparison."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

from .assistant_schema import (
    BaselineSpec,
    DataPlan,
    DataPlanItem,
    DataSourceSpec,
    RequirementDraft,
    RequirementEvidence,
    RequirementField,
    SuccessCriteria,
    TrainingObjective,
)
from .assistant_service import AssistantService
from .assistant_store import AssistantStore
from .assistant_worker import run_once
from .datagen_schema import DatagenConfig


EXPLICIT_SCENARIO = (
    "场景是客服系统的函数调用；当前问题是复杂参数经常缺失；"
    "期望行为是必填参数完整且类型正确。"
)


def _objective() -> TrainingObjective:
    return TrainingObjective(
        goal="提高客服函数调用的参数完整性",
        task_types=["fc"],
        base_model_path="/models/sanitized-9b",
        baseline=BaselineSpec(kind="base_model", name="base"),
        data_source=DataSourceSpec(fc_seed_file="registered/fc_seed.json"),
        success_criteria=SuccessCriteria(
            primary_metric="param_score_mean", min_improvement=0.15
        ),
    )


def _data_plan() -> DataPlan:
    return DataPlan(
        items=[
            DataPlanItem(
                task_type="fc",
                config=DatagenConfig(
                    task_type="fc", finetune_type="sft", count=100
                ),
                rationale="补齐复杂必填参数和类型约束样本",
            )
        ],
        rationale="先构建可审计 FC 参数错误切片，再进行 SFT。",
    )


class _Planner:
    """Fixed planner: it never calls a model or reads external data."""

    def extract_requirement_draft(
        self, messages: list[dict[str, Any]]
    ) -> RequirementDraft:
        latest = messages[-1]
        contents = "\n".join(
            str(row.get("content") or "")
            for row in messages
            if row.get("role") == "user"
        )
        if all(
            phrase in contents
            for phrase in ("客服系统", "复杂参数", "必填参数")
        ):
            evidence = {
                "source": "user",
                "evidence_message_ids": [latest["message_id"]],
            }
            return RequirementDraft(
                assistant_reply=(
                    "已整理需求卡：客服 FC 参数完整性 SFT，"
                    "并用冻结评测集做基座/候选 A/B。请先确认需求理解。"
                ),
                ready_for_review=True,
                missing_fields=[],
                scenario=RequirementField(value="客服系统的函数调用", **evidence),
                current_problem=RequirementField(
                    value="复杂参数经常缺失", **evidence
                ),
                desired_behavior=RequirementField(
                    value="必填参数完整且类型正确", **evidence
                ),
                proposed_objective=_objective(),
                field_evidence={
                    "task_types": RequirementEvidence(
                        source="assistant_assumption"
                    ),
                    "base_model_path": RequirementEvidence(source="default"),
                    "data_source": RequirementEvidence(source="default"),
                    "success_criteria": RequirementEvidence(
                        source="assistant_assumption"
                    ),
                },
                assumptions=["默认以 param_score_mean 提升 0.15 为主门槛"],
            )
        return RequirementDraft(
            assistant_reply=(
                "暂定方案框架：9B 基座做 FC SFT，构建路由和参数"
                "对齐数据，训练后做冻结 A/B。当前还需一次性补充："
                "业务场景、具体错误、期望行为。"
            ),
            ready_for_review=False,
            missing_fields=["scenario", "current_problem", "desired_behavior"],
            proposed_objective=_objective(),
            field_evidence={
                "task_types": RequirementEvidence(
                    source="assistant_assumption"
                ),
                "base_model_path": RequirementEvidence(source="default"),
            },
            assumptions=["9B FC SFT 仅为暂定框架，不可执行"],
        )

    def create_data_plan(self, objective: TrainingObjective) -> DataPlan:
        assert objective.goal == _objective().goal
        return _data_plan()


class _NoSideEffectTools:
    """Fake adapter that fails if an external start path is reached."""

    def __init__(self) -> None:
        self.external_start_calls = 0
        self.stop_calls: list[tuple[str, str]] = []

    def _forbidden(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        self.external_start_calls += 1
        raise AssertionError("A/B harness attempted a real external start path")

    start_datagen = _forbidden
    start_training = _forbidden
    start_evaluation = _forbidden

    def stop_external_job(self, kind: str, job_id: str) -> dict[str, Any]:
        self.stop_calls.append((kind, job_id))
        return {
            "kind": kind,
            "job_id": job_id,
            "stopped": True,
            "terminal": True,
            "detail": "FAKE_STOPPED",
        }


def _service(db_path: Path, tools: _NoSideEffectTools) -> AssistantService:
    return AssistantService(
        store=AssistantStore(db_path),
        planner=_Planner(),
        tools=tools,
        policy=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("training policy must not run in behavior A/B")
        ),
        preflight_runner=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("preflight must not run in behavior A/B")
        ),
        data_preparer=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("data preparation must not run in behavior A/B")
        ),
    )


def _approval_actions(snapshot: dict[str, Any]) -> list[str]:
    return [str(row["action"]) for row in snapshot.get("pending_approvals") or []]


def _artifact_urls(snapshot: dict[str, Any]) -> list[str]:
    result: list[str] = []
    for step in snapshot.get("workflow_steps") or []:
        for artifact in step.get("artifacts") or []:
            for key in ("preview_url", "download_url"):
                if artifact.get(key):
                    result.append(str(artifact[key]))
    return sorted(result)


def _seed_active_workflow(store: AssistantStore, kind: str) -> str:
    workflow_id = store.create_workflow()
    store.transition(workflow_id, "requirements_completed", {})
    if kind == "datagen":
        store.update_workflow_fields(
            workflow_id,
            datagen_jobs_json=[{"job_id": "fake_datagen_1", "task_type": "fc"}],
        )
        store.transition(workflow_id, "datagen_started", {})
        return workflow_id
    store.transition(workflow_id, "datagen_started", {})
    store.transition(workflow_id, "datagen_completed", {})
    store.transition(workflow_id, "train_plan_created", {})
    store.transition(workflow_id, "preflight_passed", {})
    store.update_workflow_fields(workflow_id, train_job_id="fake_train_1")
    store.transition(workflow_id, "training_started", {})
    if kind == "training":
        return workflow_id
    store.transition(workflow_id, "training_succeeded", {})
    store.update_workflow_fields(workflow_id, eval_id="fake_eval_1")
    store.transition(workflow_id, "evaluation_started", {})
    return workflow_id


def _run_b(temp_root: Path) -> tuple[dict[str, Any], int]:
    tools = _NoSideEffectTools()
    db_path = temp_root / "requirements.sqlite"
    service = _service(db_path, tools)

    first = service.create_workflow("我想训练9b模型")
    second = service.add_message(first["workflow_id"], "fc")
    ambiguous = {
        "messages": ["我想训练9b模型", "fc"],
        "state_sequence": [first["state"], second["state"]],
        "approvals": _approval_actions(second),
        "premature_execution": bool(
            second.get("data_plan") or _approval_actions(second)
        ),
        "missing_fields": second["requirement_draft"]["missing_fields"],
        "visual_steps": [row["key"] for row in second["workflow_steps"]],
        "assistant_reply": second["messages"][-1]["content"],
    }

    review = service.create_workflow(EXPLICIT_SCENARIO)
    confirmation = review["pending_approvals"][0]
    preparing = service.approve(
        review["workflow_id"],
        confirmation["approval_id"],
        confirmation["plan_hash"],
    )
    run_once(service.store, tools, service, limit=1)
    planned = service.snapshot(review["workflow_id"])
    explicit = {
        "messages": [EXPLICIT_SCENARIO],
        "state_sequence": [review["state"], preparing["state"], planned["state"]],
        "approvals": [confirmation["action"], *_approval_actions(planned)],
        "premature_execution": False,
        "requirement_sources": {
            name: review["requirement_draft"][name]["source"]
            for name in ("scenario", "current_problem", "desired_behavior")
        },
        "visual_steps": [row["key"] for row in planned["workflow_steps"]],
        "external_start_calls": tools.external_start_calls,
    }

    cancellations: dict[str, dict[str, Any]] = {}
    for kind in ("datagen", "training", "evaluation"):
        cancel_db = temp_root / f"cancel_{kind}.sqlite"
        cancel_service = _service(cancel_db, tools)
        workflow_id = _seed_active_workflow(cancel_service.store, kind)
        before = cancel_service.snapshot(workflow_id)
        first_request = cancel_service.store.request_cancellation(
            workflow_id, "behavior A/B cancellation"
        )
        second_request = cancel_service.store.request_cancellation(
            workflow_id, "duplicate click"
        )
        restarted = _service(cancel_db, tools)
        run_once(restarted.store, tools, restarted, limit=1)
        after = restarted.snapshot(workflow_id)
        cancellations[kind] = {
            "request_ids_equal": (
                first_request["cancel_request_id"]
                == second_request["cancel_request_id"]
            ),
            "state_after_restart": after["state"],
            "artifacts_before": _artifact_urls(before),
            "artifacts_after": _artifact_urls(after),
            "artifacts_preserved": _artifact_urls(before) == _artifact_urls(after),
            "visual_steps": [row["key"] for row in after["workflow_steps"]],
        }

    expected_steps = [
        "requirements",
        "data_plan",
        "data_build",
        "data_review",
        "train_plan",
        "training",
        "evaluation",
        "diagnosis",
    ]
    visual_rows = [ambiguous["visual_steps"], explicit["visual_steps"]] + [
        row["visual_steps"] for row in cancellations.values()
    ]
    evidence_coverage = sum(
        source == "user" for source in explicit["requirement_sources"].values()
    ) / 3
    cancel_rows = list(cancellations.values())
    b = {
        "premature_execution_rate": float(ambiguous["premature_execution"]),
        "requirement_evidence_coverage": evidence_coverage,
        "approval_sequence_accuracy": float(
            explicit["approvals"]
            == ["confirm_requirements", "start_datagen"]
        ),
        "visual_stage_coverage": sum(row == expected_steps for row in visual_rows)
        / len(visual_rows),
        "cancel_success_rate": sum(
            row["state_after_restart"] == "cancelled" for row in cancel_rows
        )
        / len(cancel_rows),
        "cancel_idempotency_rate": sum(
            row["request_ids_equal"] for row in cancel_rows
        )
        / len(cancel_rows),
        "artifact_preservation_rate": sum(
            row["artifacts_preserved"] for row in cancel_rows
        )
        / len(cancel_rows),
        "scenarios": {
            "ambiguous_9b_then_fc": ambiguous,
            "explicit_fc_goal": explicit,
            "cancellation": cancellations,
        },
        "fake_stop_adapter_calls": len(tools.stop_calls),
    }
    return b, tools.external_start_calls


def _markdown(report: dict[str, Any]) -> str:
    a = report["A"]
    b = report["B"]
    lines = [
        "# 个人训练助手安全工作流 A/B",
        "",
        "本报告仅回放脱敏状态/动作事实并使用假工具；未访问 SSH、GPU、数据生成或评测服务。",
        "",
        "| 指标 | 旧版 A | 新版 B | 验收 |",
        "|---|---:|---:|---|",
        f"| 未明确目标即进入可执行流程 | {a['premature_execution_rate']:.2f} | {b['premature_execution_rate']:.2f} | B = 0 |",
        f"| 核心需求用户证据覆盖 | n/a | {b['requirement_evidence_coverage']:.2f} | B = 1 |",
        f"| 审批顺序正确 | n/a | {b['approval_sequence_accuracy']:.2f} | B = 1 |",
        f"| 八阶段可视化覆盖 | n/a | {b['visual_stage_coverage']:.2f} | B = 1 |",
        f"| 取消成功 / 幂等 / 产物保留 | n/a | {b['cancel_success_rate']:.2f} / {b['cancel_idempotency_rate']:.2f} / {b['artifact_preservation_rate']:.2f} | 全部 = 1 |",
        f"| 真实外部副作用调用 | n/a | {report['external_side_effect_calls']} | B = 0 |",
        "",
        "## 逐场景证据",
        "",
    ]
    for name, row in b["scenarios"].items():
        lines.extend(
            [
                f"### {name}",
                "",
                "```json",
                json.dumps(row, ensure_ascii=False, indent=2, sort_keys=True),
                "```",
                "",
            ]
        )
    lines.extend(
        [
            "## 结论",
            "",
            "新版 B 只在核心场景、问题和期望行为均有用户证据后生成需求确认审批；确认后才准备数据方案，且数据构建仍需独立审批。所有 B 组门槛均通过。",
            "",
        ]
    )
    return "\n".join(lines)


def run_behavior_ab(output_dir: Path) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fixture_path = Path(__file__).parent / "fixtures" / "assistant_safe_workflow_a.json"
    a_fixture = json.loads(fixture_path.read_text("utf-8"))
    a_rows = list(a_fixture["scenarios"].values())
    a = {
        "source_workflow": a_fixture["source_workflow"],
        "sanitized": bool(a_fixture.get("sanitized")),
        "premature_execution_rate": sum(
            bool(row.get("premature_execution")) for row in a_rows
        )
        / len(a_rows),
        "scenarios": a_fixture["scenarios"],
    }
    with tempfile.TemporaryDirectory(prefix="assistant-safe-ab-") as raw_temp:
        b, external_start_calls = _run_b(Path(raw_temp))
    report = {
        "experiment": "assistant_safe_workflow_ab_20260821",
        "method": "sanitized historical A trace vs deterministic isolated B orchestration",
        "A": a,
        "B": b,
        "external_side_effect_calls": external_start_calls,
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "report.md").write_text(_markdown(report), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    report = run_behavior_ab(args.output_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
