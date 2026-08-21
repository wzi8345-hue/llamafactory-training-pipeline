import pytest
from pydantic import ValidationError

from .assistant_schema import (
    ActionKind,
    BaselineSpec,
    DataPlan,
    DataPlanItem,
    DataSourceSpec,
    RequirementDraft,
    RequirementField,
    RequirementExtraction,
    SuccessCriteria,
    TrainingAdjustment,
    TrainingObjective,
    WorkflowState,
)
from .datagen_schema import DatagenConfig


def _fc_objective() -> TrainingObjective:
    return TrainingObjective(
        goal="客服函数调用参数需要更完整",
        task_types=["fc"],
        base_model_path="/models/Qwen3.5-9B",
        baseline=BaselineSpec(kind="base_model", name="base"),
        data_source=DataSourceSpec(fc_seed_file="seeds.json"),
        success_criteria=SuccessCriteria(primary_metric="param_score_mean"),
    )


def _user_field(value: str, message_id: int = 1) -> RequirementField:
    return RequirementField(
        value=value,
        source="user",
        evidence_message_ids=[message_id],
    )


def test_requirement_draft_requires_user_evidence_for_core_fields():
    with pytest.raises(ValidationError, match="user evidence"):
        RequirementDraft(
            assistant_reply="暂定 FC 训练方案；请确认业务场景。",
            ready_for_review=True,
            missing_fields=[],
            scenario=RequirementField(
                value="客服函数调用",
                source="assistant_assumption",
                evidence_message_ids=[],
            ),
            current_problem=_user_field("复杂参数经常缺失"),
            desired_behavior=_user_field("必填参数完整且类型正确"),
            proposed_objective=_fc_objective(),
            assumptions=["客服场景由助手推断"],
        )


def test_requirement_draft_allows_review_with_three_user_evidence_fields():
    draft = RequirementDraft(
        assistant_reply="需求理解已整理，请确认。",
        ready_for_review=True,
        missing_fields=[],
        scenario=_user_field("客服函数调用"),
        current_problem=_user_field("复杂参数经常缺失"),
        desired_behavior=_user_field("必填参数完整且类型正确"),
        proposed_objective=_fc_objective(),
        field_evidence={},
        assumptions=["默认以参数分作为主指标"],
    )

    assert draft.ready_for_review is True
    assert draft.scenario.source == "user"


def test_incomplete_requirement_draft_can_keep_a_non_executable_proposal():
    draft = RequirementDraft(
        assistant_reply="先给出 FC 暂定训练框架，再补齐三个业务问题。",
        ready_for_review=False,
        missing_fields=["scenario", "current_problem", "desired_behavior"],
        proposed_objective=_fc_objective(),
        assumptions=["当前目标只是建议，不能执行"],
    )

    assert draft.proposed_objective is not None
    assert draft.ready_for_review is False


def test_requirement_extraction_requires_objective_when_ready():
    with pytest.raises(ValidationError):
        RequirementExtraction(assistant_reply="ready", ready=True, missing_fields=[])


def test_requirement_extraction_rejects_contradictory_ready_state():
    with pytest.raises(ValidationError, match="missing_fields"):
        RequirementExtraction(
            assistant_reply="ready",
            ready=True,
            missing_fields=["base_model_path"],
            objective=TrainingObjective(
                goal="QA",
                task_types=["qa"],
                base_model_path="/models/qwen",
                baseline=BaselineSpec(kind="base_model"),
                data_source=DataSourceSpec(kb_source_dir="uploads"),
                success_criteria=SuccessCriteria(primary_metric="answer_accuracy"),
            ),
        )


def test_objective_requires_a_concrete_data_source():
    with pytest.raises(ValidationError):
        TrainingObjective(
            goal="提高回答准确率",
            task_types=["qa"],
            base_model_path="/models/Qwen3.5-9B",
            baseline=BaselineSpec(kind="base_model", name="base"),
            data_source=DataSourceSpec(),
            success_criteria=SuccessCriteria(primary_metric="answer_accuracy"),
        )


def test_objective_requires_the_source_for_each_task_type():
    with pytest.raises(ValidationError, match="fc_seed_file"):
        TrainingObjective(
            goal="提高回答与工具路由",
            task_types=["qa", "fc"],
            base_model_path="/models/Qwen3.5-9B",
            baseline=BaselineSpec(kind="base_model", name="base"),
            data_source=DataSourceSpec(kb_source_dir="uploads"),
            success_criteria=SuccessCriteria(primary_metric="combined_accuracy"),
        )


def test_objective_rejects_duplicate_task_types():
    with pytest.raises(ValidationError, match="unique"):
        TrainingObjective(
            goal="提高回答准确率",
            task_types=["qa", "qa"],
            base_model_path="/models/Qwen3.5-9B",
            baseline=BaselineSpec(kind="base_model", name="base"),
            data_source=DataSourceSpec(kb_source_dir="uploads"),
            success_criteria=SuccessCriteria(primary_metric="answer_accuracy"),
        )


def test_mixed_fc_and_qa_require_a_metric_emitted_for_both_tasks():
    with pytest.raises(ValidationError, match="combined_accuracy"):
        TrainingObjective(
            goal="提高回答与工具路由",
            task_types=["qa", "fc"],
            base_model_path="/models/Qwen3.5-9B",
            baseline=BaselineSpec(kind="base_model", name="base"),
            data_source=DataSourceSpec(
                kb_source_dir="uploads", fc_seed_file="seeds.json"
            ),
            success_criteria=SuccessCriteria(primary_metric="param_score"),
        )


def test_objective_rejects_metric_not_emitted_by_requested_task():
    with pytest.raises(ValidationError, match="answer_accuracy"):
        TrainingObjective(
            goal="提高 FC",
            task_types=["fc"],
            base_model_path="/models/Qwen3.5-9B",
            baseline=BaselineSpec(kind="base_model", name="base"),
            data_source=DataSourceSpec(fc_seed_file="seeds.json"),
            success_criteria=SuccessCriteria(primary_metric="answer_accuracy"),
        )


def test_objective_rejects_unexecutable_custom_critical_slice():
    with pytest.raises(ValidationError, match="critical_slices"):
        TrainingObjective(
            goal="提高 QA",
            task_types=["qa"],
            base_model_path="/models/Qwen3.5-9B",
            baseline=BaselineSpec(kind="base_model", name="base"),
            data_source=DataSourceSpec(kb_source_dir="uploads"),
            success_criteria=SuccessCriteria(primary_metric="answer_accuracy"),
            critical_slices=["long_context"],
        )


def test_objective_accepts_a_requested_task_as_auditable_critical_slice():
    objective = TrainingObjective(
        goal="提高 QA",
        task_types=["qa"],
        base_model_path="/models/Qwen3.5-9B",
        baseline=BaselineSpec(kind="base_model", name="base"),
        data_source=DataSourceSpec(kb_source_dir="uploads"),
        success_criteria=SuccessCriteria(primary_metric="answer_accuracy"),
        critical_slices=["task_type=qa"],
    )
    assert objective.critical_slices == ["task_type=qa"]


@pytest.mark.parametrize(
    ("task_types", "data_source", "critical_slice"),
    [
        (["fc"], DataSourceSpec(fc_seed_file="seed.json"), "tool_name=search"),
        (["qa"], DataSourceSpec(kb_source_dir="uploads"), "source_doc=manual.pdf"),
        (["fc"], DataSourceSpec(fc_seed_file="seed.json"), "error_type=wrong_args"),
    ],
)
def test_objective_accepts_generated_slice_taxonomy(
    task_types, data_source, critical_slice
):
    metric = "tool_name_accuracy" if task_types == ["fc"] else "answer_accuracy"
    objective = TrainingObjective(
        goal="提高关键切片",
        task_types=task_types,
        base_model_path="/models/Qwen3.5-9B",
        baseline=BaselineSpec(kind="base_model", name="base"),
        data_source=data_source,
        success_criteria=SuccessCriteria(primary_metric=metric),
        critical_slices=[critical_slice],
        max_datagen_records=10,
    )
    assert objective.critical_slices == [critical_slice]


def test_objective_rejects_slice_taxonomy_incompatible_with_tasks():
    with pytest.raises(ValidationError, match="tool_name"):
        TrainingObjective(
            goal="提高 QA",
            task_types=["qa"],
            base_model_path="/models/Qwen3.5-9B",
            baseline=BaselineSpec(kind="base_model", name="base"),
            data_source=DataSourceSpec(kb_source_dir="uploads"),
            success_criteria=SuccessCriteria(primary_metric="answer_accuracy"),
            critical_slices=["tool_name=search"],
            max_datagen_records=10,
        )


def test_objective_allows_a_ten_record_smoke_budget():
    objective = TrainingObjective(
        goal="smoke",
        task_types=["qa"],
        base_model_path="/models/Qwen3.5-9B",
        baseline=BaselineSpec(kind="base_model", name="base"),
        data_source=DataSourceSpec(kb_source_dir="uploads"),
        success_criteria=SuccessCriteria(primary_metric="answer_accuracy"),
        max_datagen_records=10,
    )
    assert objective.max_datagen_records == 10


def test_train_job_baseline_requires_job_id():
    with pytest.raises(ValidationError, match="train_job_id"):
        BaselineSpec(kind="train_job", name="champion")


def test_data_plan_task_type_must_match_datagen_config():
    with pytest.raises(ValidationError):
        DataPlan(
            items=[DataPlanItem(
                task_type="fc",
                config=DatagenConfig(task_type="qa", finetune_type="sft", count=100),
            )],
            rationale="FC hard cases",
        )


def test_data_plan_cannot_mix_sft_and_dpo_outputs():
    with pytest.raises(ValidationError):
        DataPlan(
            items=[
                DataPlanItem(
                    task_type="qa",
                    config=DatagenConfig(task_type="qa", finetune_type="sft", count=50),
                ),
                DataPlanItem(
                    task_type="fc",
                    config=DatagenConfig(task_type="fc", finetune_type="dpo", count=50),
                ),
            ],
            rationale="mixed incompatible outputs",
        )


def test_data_plan_rejects_duplicate_training_adjustment_parameters():
    with pytest.raises(ValidationError, match="unique"):
        DataPlan(
            items=[
                DataPlanItem(
                    task_type="fc",
                    config=DatagenConfig(
                        task_type="fc", finetune_type="sft", count=50
                    ),
                )
            ],
            rationale="retry",
            training_adjustments=[
                TrainingAdjustment(
                    parameter="learning_rate", value=2e-4, reason="first"
                ),
                TrainingAdjustment(
                    parameter="learning_rate", value=3e-4, reason="second"
                ),
            ],
        )


def test_enum_values_are_stable():
    assert WorkflowState.TRAINING.value == "training"
    assert ActionKind.START_TRAINING.value == "start_training"
