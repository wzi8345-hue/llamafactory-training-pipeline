import json

import pytest

from .assistant_planner import AssistantPlanner, PlannerOutputError
from .assistant_schema import (
    BaselineSpec,
    DataPlan,
    DataSourceSpec,
    RequirementDraft,
    SuccessCriteria,
    TrainingAdjustment,
    TrainingObjective,
)
from .test_assistant_service import fake_policy


class FakeLLM:
    def __init__(self, answers):
        self.answers = iter(answers)
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        return {"answer": next(self.answers)}


class UnexpectedLLM:
    def chat(self, **kwargs):
        raise AssertionError(f"LLM should not be called: {kwargs}")


def objective():
    return TrainingObjective(
        goal="提高 FC 工具选择",
        task_types=["fc"],
        base_model_path="/models/qwen",
        template="qwen3_5_nothink",
        baseline=BaselineSpec(kind="base_model", name="base"),
        data_source=DataSourceSpec(fc_seed_file="sft_data/router_fc/seed.json"),
        success_criteria=SuccessCriteria(primary_metric="tool_name_accuracy"),
        requested_finetune_type="dpo",
        max_datagen_records=200,
    )


def requirement_draft_json(
    *,
    ready: bool,
    source: str,
    message_ids: list[int],
    missing_fields: list[str] | None = None,
) -> str:
    return json.dumps(
        {
            "assistant_reply": (
                "暂定方案：9B 基座做 FC SFT，先构建参数缺失样本并用 A/B 验证。"
                "请一次补充：业务场景？当前失败表现？期望行为？"
            ),
            "ready_for_review": ready,
            "missing_fields": missing_fields or [],
            "scenario": {
                "value": "客服函数调用",
                "source": source,
                "evidence_message_ids": message_ids,
            },
            "current_problem": {
                "value": "复杂参数经常缺失",
                "source": source,
                "evidence_message_ids": message_ids,
            },
            "desired_behavior": {
                "value": "必填参数完整且类型正确",
                "source": source,
                "evidence_message_ids": message_ids,
            },
            "proposed_objective": {
                "goal": "客服函数调用的必填参数完整且类型正确",
                "task_types": ["fc"],
                "base_model_path": "/models/qwen",
                "baseline": {"kind": "base_model", "name": "base"},
                "data_source": {
                    "fc_seed_file": "sft_data/router_fc/seed.json"
                },
                "success_criteria": {
                    "primary_metric": "param_score_mean"
                },
            },
            "field_evidence": {},
            "assumptions": ["默认以参数分作为主指标"],
        },
        ensure_ascii=False,
    )


def test_9b_then_fc_cannot_become_reviewable_without_scenario_goal_evidence():
    planner = AssistantPlanner(
        FakeLLM(
            [
                requirement_draft_json(
                    ready=False,
                    source="assistant_assumption",
                    message_ids=[],
                    missing_fields=[
                        "scenario",
                        "current_problem",
                        "desired_behavior",
                    ],
                )
            ]
        )
    )
    messages = [
        {"message_id": 1, "role": "user", "content": "我想训练9b模型"},
        {"message_id": 2, "role": "user", "content": "fc"},
    ]

    draft = planner.extract_requirement_draft(messages)

    assert isinstance(draft, RequirementDraft)
    assert draft.ready_for_review is False
    assert set(draft.missing_fields) >= {
        "scenario",
        "current_problem",
        "desired_behavior",
    }
    assert draft.proposed_objective is not None
    assert "暂定方案" in draft.assistant_reply
    assert draft.assistant_reply.count("？") <= 3


def test_explicit_scenario_problem_and_target_reach_review_only():
    fake = FakeLLM(
        [requirement_draft_json(ready=True, source="user", message_ids=[1])]
    )
    messages = [
        {
            "message_id": 1,
            "role": "user",
            "content": (
                "业务场景是客服函数调用；当前问题是复杂参数经常缺失；"
                "期望行为是必填参数完整且类型正确。"
            ),
        }
    ]

    draft = AssistantPlanner(fake).extract_requirement_draft(messages)

    assert draft.ready_for_review is True
    assert draft.proposed_objective.task_types == ["fc"]
    assert draft.scenario.evidence_message_ids == [1]


def test_forged_user_evidence_ids_are_downgraded_to_missing_fields():
    fake = FakeLLM(
        [requirement_draft_json(ready=True, source="user", message_ids=[999])]
    )
    draft = AssistantPlanner(fake).extract_requirement_draft(
        [{"message_id": 1, "role": "user", "content": "明确的业务描述"}]
    )

    assert draft.ready_for_review is False
    assert set(draft.missing_fields) == {
        "scenario",
        "current_problem",
        "desired_behavior",
    }


def test_requirement_draft_prompt_is_proposal_first_and_batches_questions():
    fake = FakeLLM(
        [
            requirement_draft_json(
                ready=False,
                source="assistant_assumption",
                message_ids=[],
                missing_fields=["scenario"],
            )
        ]
    )

    AssistantPlanner(fake).extract_requirement_draft(
        [{"message_id": 1, "role": "user", "content": "训练 FC"}]
    )

    system = fake.calls[0]["system"]
    assert "暂定训练方案框架" in system
    assert "最多三个" in system
    assert "一次只询问一个" not in system


def test_extract_requirements_validates_ready_objective():
    raw = json.dumps({
        "assistant_reply": "已理解，将为 FC 路由构建 DPO 数据。",
        "ready": True,
        "missing_fields": [],
        "objective": {
            "goal": "提高 FC 工具选择",
            "task_types": ["fc"],
            "base_model_path": "/models/qwen",
            "baseline": {"kind": "base_model", "name": "base"},
            "data_source": {"fc_seed_file": "sft_data/router_fc/seed.json"},
            "success_criteria": {"primary_metric": "tool_name_accuracy"},
        },
    }, ensure_ascii=False)
    fake = FakeLLM([raw])
    out = AssistantPlanner(fake).extract_requirements([
        {"role": "user", "content": "训练 FC 路由"}
    ])
    assert out.ready and out.objective.task_types == ["fc"]
    assert fake.calls[0]["temperature"] == 0.0
    assert "tools" not in fake.calls[0]


def test_requirement_prompt_supplies_safe_defaults_for_delegated_drafts():
    raw = json.dumps({
        "assistant_reply": "请只确认任务类型。",
        "ready": False,
        "missing_fields": ["task_types"],
        "objective": None,
    }, ensure_ascii=False)
    fake = FakeLLM([raw])

    AssistantPlanner(fake).extract_requirements([
        {"role": "user", "content": "我想训练一个模型，但还没想好任务"}
    ])

    call = fake.calls[0]
    payload = json.loads(call["user"])
    defaults = payload["safe_defaults"]
    fc = defaults["objective_defaults"]["fc"]
    assert fc["base_model_path"] == "/data/wangzhengyan/Qwen3.5-9B/"
    assert fc["template"] == "qwen3_5_nothink"
    assert fc["data_source"]["fc_seed_file"] == (
        "sft_data/router_fc/sft_router_fc_shougang_1160.json"
    )
    assert fc["baseline"] == {"kind": "base_model", "name": "base"}
    assert fc["success_criteria"] == {
        "primary_metric": "param_score_mean",
        "min_improvement": 0.15,
        "non_regression_metrics": {"tool_name_accuracy": 0.0},
        "max_invalid_rate_increase": 0.01,
    }
    assert fc["requested_finetune_type"] == "sft"
    assert fc["max_datagen_records"] == 1000
    assert "goal" not in fc
    assert "data_sources" not in fc
    assert "一次只询问一个" in call["system"]
    assert "安全默认值" in call["system"]


def test_delegated_fc_request_returns_default_draft_without_repeated_questions():
    out = AssistantPlanner(UnexpectedLLM()).extract_requirements([
        {"role": "user", "content": "我想训练9b模型"},
        {
            "role": "user",
            "content": (
                "提升FC参数准确性，任务是FC，9b模型路径你帮我找，"
                "没有历史baseline，随机生成数据，提升所有指标"
            ),
        },
        {"role": "user", "content": "我不管，按默认方案继续"},
    ])

    assert out.ready is True
    assert out.missing_fields == []
    assert out.objective.task_types == ["fc"]
    assert out.objective.base_model_path == "/data/wangzhengyan/Qwen3.5-9B/"
    assert out.objective.baseline.kind == "base_model"
    assert out.objective.success_criteria.primary_metric == "param_score_mean"
    assert out.objective.success_criteria.non_regression_metrics == {
        "tool_name_accuracy": 0.0
    }
    assert "等待审批" in out.assistant_reply
    assert "尚未执行" in out.assistant_reply


def test_delegated_request_without_task_asks_only_for_task_type():
    out = AssistantPlanner(UnexpectedLLM()).extract_requirements([
        {"role": "user", "content": "我想训练 9B 模型，其余按默认"}
    ])

    assert out.ready is False
    assert out.objective is None
    assert out.missing_fields == ["task_types"]
    assert "任务类型" in out.assistant_reply


def test_fc_param_primary_adds_tool_name_non_regression_default():
    raw = json.dumps({
        "assistant_reply": "FC 方案已就绪。",
        "ready": True,
        "missing_fields": [],
        "objective": {
            "goal": "提高 FC 参数识别",
            "task_types": ["fc"],
            "base_model_path": "/models/qwen",
            "baseline": {"kind": "base_model", "name": "base"},
            "data_source": {"fc_seed_file": "sft_data/router_fc/seed.json"},
            "success_criteria": {
                "primary_metric": "param_score",
                "min_improvement": 0.15,
            },
        },
    }, ensure_ascii=False)
    out = AssistantPlanner(FakeLLM([raw])).extract_requirements([
        {"role": "user", "content": "训练 FC 参数"}
    ])
    assert out.objective.success_criteria.non_regression_metrics == {
        "tool_name_accuracy": 0.0
    }


def test_invalid_planner_json_raises_typed_error():
    with pytest.raises(PlannerOutputError, match="JSON"):
        AssistantPlanner(FakeLLM(["not-json"])).extract_requirements([])


def plan_json(
    *, count=200, source="/invented/by/model.json", finetune="dpo", adjustments=None
):
    return json.dumps({
        "items": [{
            "task_type": "fc",
            "config": {
                "finetune_type": finetune,
                "task_type": "fc",
                "count": count,
                "fc_seed_file": source,
                "gen_prompt": "围绕工具路由目标构造多样化用户请求",
                "judge_prompt": "检查意图、工具名与参数是否一致",
                "rejected_prompt": "构造自然但明确错误的工具调用",
                "pair_judge_prompt": "判断 chosen 是否显著优于 rejected",
            },
            "rationale": "构造 wrong_tool/wrong_args/direct_answer 偏好对",
        }],
        "holdout_ratio": 0.1,
        "validation_ratio": 0.1,
        "split_seed": 42,
        "rationale": "先修复 FC 决策",
        "risks": [],
        "training_adjustments": adjustments or [],
    }, ensure_ascii=False)


def test_data_plan_preserves_task_and_requested_type():
    plan = AssistantPlanner(FakeLLM([plan_json()])).create_data_plan(objective())
    assert plan.items[0].config.finetune_type == "dpo"


def test_schema_invalid_data_plan_falls_back_to_safe_bound_defaults():
    malformed = json.dumps({
        "items": [{
            "task_type": "fc",
            "config": {"seed_file": "/invented.json"},
            "rationale": "unsupported schema",
        }],
        "rationale": "bad model output",
    }, ensure_ascii=False)

    plan = AssistantPlanner(FakeLLM([malformed])).create_data_plan(objective())

    assert len(plan.items) == 1
    item = plan.items[0]
    assert item.task_type == "fc"
    assert item.config.task_type == "fc"
    assert item.config.finetune_type == "dpo"
    assert item.config.count == 200
    assert item.config.fc_seed_file == "sft_data/router_fc/seed.json"
    assert item.config.gen_prompt.strip()
    assert item.config.judge_prompt.strip()
    assert item.config.rejected_prompt.strip()
    assert item.config.pair_judge_prompt.strip()
    assert "确定性安全默认" in plan.rationale


def test_data_plan_rejects_budget_overrun():
    with pytest.raises(PlannerOutputError, match="budget"):
        AssistantPlanner(FakeLLM([plan_json(count=201)])).create_data_plan(objective())


def test_data_plan_rejects_changed_requested_finetune_type():
    with pytest.raises(PlannerOutputError, match="finetune"):
        AssistantPlanner(FakeLLM([plan_json(finetune="sft")])).create_data_plan(objective())


def test_data_plan_rejects_sft_when_error_type_slice_requires_dpo():
    target = objective().model_copy(
        update={
            "requested_finetune_type": None,
            "critical_slices": ["error_type=wrong_args"],
        }
    )
    with pytest.raises(PlannerOutputError, match="error_type"):
        AssistantPlanner(FakeLLM([plan_json(finetune="sft")])).create_data_plan(
            target
        )


def test_data_plan_binds_user_source_instead_of_model_source():
    plan = AssistantPlanner(FakeLLM([plan_json()])).create_data_plan(objective())
    assert plan.items[0].config.fc_seed_file == "sft_data/router_fc/seed.json"


def test_data_plan_rejects_empty_generation_or_quality_prompts():
    raw = json.loads(plan_json())
    raw["items"][0]["config"]["gen_prompt"] = "  "
    with pytest.raises(PlannerOutputError, match="gen_prompt"):
        AssistantPlanner(FakeLLM([json.dumps(raw, ensure_ascii=False)])).create_data_plan(
            objective()
        )


def test_dpo_data_plan_requires_preference_pair_prompts():
    raw = json.loads(plan_json())
    raw["items"][0]["config"]["rejected_prompt"] = ""
    with pytest.raises(PlannerOutputError, match="rejected_prompt"):
        AssistantPlanner(FakeLLM([json.dumps(raw, ensure_ascii=False)])).create_data_plan(
            objective()
        )


def test_initial_data_plan_rejects_training_adjustments_without_diagnosis():
    raw = plan_json(
        adjustments=[
            TrainingAdjustment(
                parameter="learning_rate",
                value=5e-5,
                reason="unsupported initial guess",
            ).model_dump(mode="json")
        ]
    )
    with pytest.raises(PlannerOutputError, match="initial data plan"):
        AssistantPlanner(FakeLLM([raw])).create_data_plan(objective())


def test_diagnosis_explanation_is_grounded_in_supplied_evidence():
    planner = AssistantPlanner(FakeLLM([
        '{"summary":"无效率上升 7 个百分点，先检查模板与工具协议。"}'
    ]))
    summary = planner.explain_diagnosis(
        {"category": "template_or_protocol_mismatch", "evidence": ["invalid +0.07"]},
        {"baseline_invalid": 0.01, "candidate_invalid": 0.08},
    )
    assert "无效率" in summary


def test_iteration_plan_is_grounded_in_diagnosis_and_rebinds_source():
    fake = FakeLLM([plan_json(count=120)])
    planner = AssistantPlanner(fake)
    previous = DataPlan.model_validate_json(plan_json(count=200))

    revised = planner.create_iteration_plan(
        objective(),
        previous,
        fake_policy(),
        {
            "category": "data_coverage_gap",
            "accepted": False,
            "next_data_changes": ["补充失败工具切片"],
            "next_training_changes": [],
        },
        {"paired_comparison": {"n": 100, "mean_delta": 0.05}},
    )

    request = json.loads(fake.calls[0]["user"])
    assert request["diagnosis"]["next_data_changes"] == ["补充失败工具切片"]
    assert request["previous_training_plan"]["config"]["train"][
        "learning_rate"
    ] == 0.0001
    assert revised.items[0].config.count == 120
    assert revised.items[0].config.fc_seed_file == "sft_data/router_fc/seed.json"


def test_iteration_plan_cannot_switch_previous_finetune_type():
    unrestricted = objective().model_copy(
        update={"requested_finetune_type": None}
    )
    previous = DataPlan.model_validate_json(plan_json(finetune="sft"))
    planner = AssistantPlanner(FakeLLM([plan_json(finetune="dpo")]))

    with pytest.raises(PlannerOutputError, match="previous finetune type"):
        planner.create_iteration_plan(
            unrestricted,
            previous,
            fake_policy(),
            {
                "category": "data_coverage_gap",
                "accepted": False,
                "next_data_changes": ["add coverage"],
                "next_training_changes": [],
            },
            {},
        )


def test_iteration_plan_rejects_adjustment_without_diagnosis_support():
    adjustment = TrainingAdjustment(
        parameter="learning_rate", value=2e-4, reason="unsupported"
    ).model_dump(mode="json")
    planner = AssistantPlanner(
        FakeLLM([plan_json(finetune="dpo", adjustments=[adjustment])])
    )

    with pytest.raises(PlannerOutputError, match="diagnosis"):
        planner.create_iteration_plan(
            objective(),
            DataPlan.model_validate_json(plan_json()),
            fake_policy(),
            {
                "category": "data_coverage_gap",
                "accepted": False,
                "next_data_changes": ["add coverage"],
                "next_training_changes": [],
            },
            {},
        )


def test_underfit_iteration_rejects_epoch_reduction():
    adjustment = TrainingAdjustment(
        parameter="num_train_epochs", value=1.0, reason="wrong direction"
    ).model_dump(mode="json")
    planner = AssistantPlanner(
        FakeLLM([plan_json(finetune="dpo", adjustments=[adjustment])])
    )

    with pytest.raises(PlannerOutputError, match="direction"):
        planner.create_iteration_plan(
            objective(),
            DataPlan.model_validate_json(plan_json()),
            fake_policy(),
            {
                "category": "underfit",
                "accepted": False,
                "next_data_changes": [],
                "next_training_changes": ["increase capacity"],
            },
            {},
        )


def test_iteration_rejects_parameter_not_allowed_by_diagnosis_category():
    adjustment = TrainingAdjustment(
        parameter="learning_rate", value=5e-5, reason="not an overfit action"
    ).model_dump(mode="json")
    planner = AssistantPlanner(
        FakeLLM([plan_json(finetune="dpo", adjustments=[adjustment])])
    )

    with pytest.raises(PlannerOutputError, match="does not allow learning_rate"):
        planner.create_iteration_plan(
            objective(),
            DataPlan.model_validate_json(plan_json()),
            fake_policy(),
            {
                "category": "overfit",
                "accepted": False,
                "next_data_changes": [],
                "next_training_changes": ["reduce epochs"],
            },
            {},
        )
