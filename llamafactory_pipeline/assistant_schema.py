"""Validated contracts shared by the personal training assistant."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .datagen_schema import DatagenConfig
from .eval_schema import EvalRequest
from .schema import TrainConfig

STRICT = ConfigDict(extra="forbid")


class WorkflowState(str, Enum):
    COLLECTING_REQUIREMENTS = "collecting_requirements"
    REQUIREMENTS_REVIEW = "requirements_review"
    DATA_PLAN_PREPARING = "data_plan_preparing"
    DATA_PLAN_READY = "data_plan_ready"
    DATA_GENERATING = "data_generating"
    DATA_REVIEW = "data_review"
    TRAIN_PLAN_READY = "train_plan_ready"
    PREFLIGHT_BLOCKED = "preflight_blocked"
    TRAIN_READY = "train_ready"
    TRAINING = "training"
    TRAIN_FAILED = "train_failed"
    AB_PLAN_READY = "ab_plan_ready"
    EVALUATING = "evaluating"
    DIAGNOSIS_READY = "diagnosis_ready"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"
    COMPLETED = "completed"


class ActionKind(str, Enum):
    CONFIRM_REQUIREMENTS = "confirm_requirements"
    START_DATAGEN = "start_datagen"
    START_TRAINING = "start_training"
    START_EVALUATION = "start_evaluation"
    RETRY_SCORING = "retry_scoring"
    SKIP_EVALUATION = "skip_evaluation"
    ACCEPT_CANDIDATE = "accept_candidate"
    FINISH_WITHOUT_ACCEPTING = "finish_without_accepting"
    START_ITERATION = "start_iteration"
    RECOVER_TRAINING = "recover_training"


class ApprovalPayload(BaseModel):
    model_config = STRICT
    action: ActionKind
    plan: dict[str, Any]
    decision_warnings: list[str] = Field(default_factory=list)


class BaselineSpec(BaseModel):
    model_config = STRICT
    kind: Literal["base_model", "train_job"]
    name: str = "baseline"
    train_job_id: Optional[str] = None

    @model_validator(mode="after")
    def require_job_id(self):
        if self.kind == "train_job" and not self.train_job_id:
            raise ValueError("train_job baseline requires train_job_id")
        return self


class DataSourceSpec(BaseModel):
    model_config = STRICT
    kb_source_dir: Optional[str] = None
    fc_seed_file: Optional[str] = None
    collection: Optional[str] = None

    @model_validator(mode="after")
    def require_source(self):
        if not any((self.kb_source_dir, self.fc_seed_file, self.collection)):
            raise ValueError("at least one data source is required")
        return self


class SuccessCriteria(BaseModel):
    model_config = STRICT
    primary_metric: str = Field(min_length=1)
    min_improvement: float = Field(default=0.15, ge=0)
    non_regression_metrics: dict[str, float] = Field(default_factory=dict)
    max_invalid_rate_increase: float = Field(default=0.01, ge=0)
    max_critical_slice_rate_regression: float = Field(default=0.02, ge=0)
    max_critical_slice_score_regression: float = Field(default=0.10, ge=0)


class RequirementEvidence(BaseModel):
    model_config = STRICT
    source: Literal["user", "default", "assistant_assumption"]
    evidence_message_ids: list[int] = Field(default_factory=list)

    @model_validator(mode="after")
    def user_evidence_requires_message_ids(self):
        if self.source == "user" and not self.evidence_message_ids:
            raise ValueError("user evidence requires message ids")
        if len(set(self.evidence_message_ids)) != len(self.evidence_message_ids):
            raise ValueError("evidence_message_ids must be unique")
        if any(message_id <= 0 for message_id in self.evidence_message_ids):
            raise ValueError("evidence_message_ids must be positive")
        return self


class RequirementField(RequirementEvidence):
    value: str = Field(min_length=1, max_length=4000)


class TrainingObjective(BaseModel):
    model_config = STRICT
    goal: str = Field(min_length=1)
    task_types: list[Literal["qa", "qa_multi", "fc"]] = Field(min_length=1)
    base_model_path: str = Field(min_length=1)
    template: str = "qwen3_5_nothink"
    baseline: BaselineSpec
    data_source: DataSourceSpec
    success_criteria: SuccessCriteria
    requested_finetune_type: Optional[Literal["sft", "dpo"]] = None
    max_datagen_records: int = Field(default=1000, ge=2, le=100000)
    max_training_hours: Optional[float] = Field(default=None, gt=0)
    critical_slices: list[str] = Field(default_factory=list)
    examples: list[dict[str, Any]] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique_tasks_and_matching_sources(self):
        if len(set(self.task_types)) != len(self.task_types):
            raise ValueError("task_types must be unique")
        required = {
            "qa": "kb_source_dir",
            "qa_multi": "collection",
            "fc": "fc_seed_file",
        }
        missing = [
            required[task]
            for task in self.task_types
            if not getattr(self.data_source, required[task])
        ]
        if missing:
            raise ValueError(f"missing task data sources: {', '.join(sorted(missing))}")
        minimum_budget = 2 * len(self.task_types)
        if self.max_datagen_records < minimum_budget:
            raise ValueError(
                f"max_datagen_records must be at least {minimum_budget} "
                "for per-task train/eval coverage"
            )
        qa_metrics = {"answer_accuracy", "score_mean", "combined_accuracy"}
        fc_metrics = {
            "tool_name_accuracy",
            "param_score",
            "param_score_mean",
            "combined_accuracy",
        }
        has_qa = bool({"qa", "qa_multi"}.intersection(self.task_types))
        has_fc = "fc" in self.task_types
        primary = self.success_criteria.primary_metric
        if has_qa and has_fc and primary != "combined_accuracy":
            raise ValueError(
                "mixed QA/FC objectives require primary_metric=combined_accuracy"
            )
        supported = (qa_metrics if has_qa else set()) | (
            fc_metrics if has_fc else set()
        )
        if primary not in supported:
            raise ValueError(
                f"primary metric {primary!r} is not emitted by requested tasks"
            )
        unknown_non_regression = sorted(
            set(self.success_criteria.non_regression_metrics) - supported
        )
        if unknown_non_regression:
            raise ValueError(
                "non-regression metrics are not emitted by requested tasks: "
                + ", ".join(unknown_non_regression)
            )
        if len(set(self.critical_slices)) != len(self.critical_slices):
            raise ValueError("critical_slices must be unique")
        unsupported_slices = []
        has_qa_task = bool({"qa", "qa_multi"}.intersection(self.task_types))
        for selector in self.critical_slices:
            if selector in self.task_types:
                continue
            if "=" not in selector:
                unsupported_slices.append(selector)
                continue
            key, value = selector.split("=", 1)
            if (
                key not in {"task_type", "tool_name", "source_doc", "error_type"}
                or not value
                or len(value) > 256
                or any(char in value for char in ("\n", "\r", "\x00"))
            ):
                unsupported_slices.append(selector)
                continue
            if key == "task_type" and value not in self.task_types:
                unsupported_slices.append(selector)
            elif key == "tool_name" and "fc" not in self.task_types:
                unsupported_slices.append(selector)
            elif key == "source_doc" and not has_qa_task:
                unsupported_slices.append(selector)
            elif key == "error_type" and (
                self.requested_finetune_type == "sft"
                or value not in {"wrong_tool", "wrong_args", "direct_answer"}
            ):
                unsupported_slices.append(selector)
        if unsupported_slices:
            raise ValueError(
                "critical_slices must use an executable task_type, tool_name, "
                "source_doc, or DPO error_type selector compatible with the "
                "requested tasks: "
                + ", ".join(unsupported_slices)
            )
        return self


class RequirementDraft(BaseModel):
    model_config = STRICT
    assistant_reply: str = Field(min_length=1)
    ready_for_review: bool
    missing_fields: list[str]
    scenario: Optional[RequirementField] = None
    current_problem: Optional[RequirementField] = None
    desired_behavior: Optional[RequirementField] = None
    proposed_objective: Optional[TrainingObjective] = None
    field_evidence: dict[str, RequirementEvidence] = Field(default_factory=dict)
    assumptions: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def review_requires_user_evidence(self):
        if not self.ready_for_review:
            return self
        if self.missing_fields:
            raise ValueError("reviewable requirement draft has missing fields")
        if self.proposed_objective is None:
            raise ValueError("reviewable requirement draft requires an objective")
        for name in ("scenario", "current_problem", "desired_behavior"):
            field = getattr(self, name)
            if field is None or field.source != "user":
                raise ValueError(f"{name} requires user evidence")
        return self


class CancelRequest(BaseModel):
    model_config = STRICT
    reason: str = Field(default="用户手动中止", min_length=1, max_length=500)


class StepProgress(BaseModel):
    model_config = STRICT
    current: Optional[float] = None
    target: Optional[float] = None
    percentage: Optional[float] = Field(default=None, ge=0, le=100)
    eta_seconds: Optional[int] = Field(default=None, ge=0)
    updated_at: Optional[str] = None
    details: dict[str, Any] = Field(default_factory=dict)


class ArtifactRef(BaseModel):
    model_config = STRICT
    name: str = Field(min_length=1)
    kind: str = Field(min_length=1)
    sha256: Optional[str] = None
    size_bytes: Optional[int] = Field(default=None, ge=0)
    preview_url: Optional[str] = None
    download_url: Optional[str] = None
    created_at: Optional[str] = None


class WorkflowStep(BaseModel):
    model_config = STRICT
    key: Literal[
        "requirements",
        "data_plan",
        "data_build",
        "data_review",
        "train_plan",
        "training",
        "evaluation",
        "diagnosis",
    ]
    sequence: int = Field(ge=1, le=8)
    title: str = Field(min_length=1)
    status: Literal[
        "pending",
        "active",
        "needs_confirmation",
        "blocked",
        "failed",
        "cancelling",
        "cancelled",
        "succeeded",
        "skipped",
    ]
    started_at: Optional[str] = None
    updated_at: Optional[str] = None
    finished_at: Optional[str] = None
    summary: str = ""
    inputs: list[dict[str, Any]] = Field(default_factory=list)
    decisions: list[dict[str, Any]] = Field(default_factory=list)
    progress: Optional[StepProgress] = None
    artifacts: list[ArtifactRef] = Field(default_factory=list)
    issues: list[dict[str, Any]] = Field(default_factory=list)
    actions: list[str] = Field(default_factory=list)


class RequirementExtraction(BaseModel):
    model_config = STRICT
    assistant_reply: str
    ready: bool
    missing_fields: list[str]
    objective: Optional[TrainingObjective] = None

    @model_validator(mode="after")
    def ready_requires_objective(self):
        if self.ready and self.objective is None:
            raise ValueError("ready extraction requires objective")
        if self.ready and self.missing_fields:
            raise ValueError("ready extraction requires empty missing_fields")
        if not self.ready and self.objective is not None:
            raise ValueError("incomplete extraction must not include an objective")
        return self


class DataPlanItem(BaseModel):
    model_config = STRICT
    task_type: Literal["qa", "qa_multi", "fc"]
    config: DatagenConfig
    rationale: str = ""

    @model_validator(mode="after")
    def matching_task(self):
        if self.task_type != self.config.task_type:
            raise ValueError("task_type must match config.task_type")
        return self


class TrainingAdjustment(BaseModel):
    model_config = STRICT
    parameter: Literal[
        "learning_rate",
        "num_train_epochs",
        "lora_rank",
        "lora_dropout",
    ]
    value: float
    reason: str = Field(min_length=1)

    @model_validator(mode="after")
    def bounded_value(self):
        bounds = {
            "learning_rate": (1e-7, 1e-2),
            "num_train_epochs": (0.1, 20.0),
            "lora_rank": (1.0, 256.0),
            "lora_dropout": (0.0, 0.5),
        }
        low, high = bounds[self.parameter]
        if not low <= self.value <= high:
            raise ValueError(f"{self.parameter} adjustment is outside safe bounds")
        if self.parameter == "lora_rank" and not float(self.value).is_integer():
            raise ValueError("lora_rank adjustment must be an integer")
        return self


class DataPlan(BaseModel):
    model_config = STRICT
    items: list[DataPlanItem] = Field(min_length=1)
    holdout_ratio: float = Field(default=0.10, gt=0, lt=0.5)
    validation_ratio: float = Field(default=0.10, ge=0, lt=0.5)
    split_seed: int = 42
    rationale: str
    risks: list[str] = Field(default_factory=list)
    training_adjustments: list[TrainingAdjustment] = Field(default_factory=list)

    @model_validator(mode="after")
    def one_finetune_type(self):
        kinds = {item.config.finetune_type for item in self.items}
        if len(kinds) != 1:
            raise ValueError("all data-plan items must use one finetune_type")
        undersized = [
            item.task_type for item in self.items if item.config.count < 2
        ]
        if undersized:
            raise ValueError(
                "each task needs at least 2 generated records: "
                + ", ".join(sorted(undersized))
            )
        parameters = [item.parameter for item in self.training_adjustments]
        if len(parameters) != len(set(parameters)):
            raise ValueError("training adjustment parameters must be unique")
        return self


class DatasetProfile(BaseModel):
    model_config = STRICT
    dataset_name: str
    eval_dataset_names: dict[Literal["function_call", "subjective"], str]
    sha256: str
    eval_sha256: dict[Literal["function_call", "subjective"], str]
    n_records: int = Field(ge=1)
    holdout_records: int = Field(ge=1)
    requested_holdout_ratio: float = Field(gt=0, lt=0.5)
    actual_holdout_ratio: float = Field(gt=0, lt=1)
    validation_ratio: float = Field(ge=0, lt=0.5)
    split_seed: int
    finetune_type: Literal["sft", "dpo"]
    task_types: list[Literal["qa", "qa_multi", "fc"]] = Field(min_length=1)
    char_p50: int = Field(ge=0)
    char_p95: int = Field(ge=0)
    char_max: int = Field(ge=0)
    token_p50: int = Field(ge=0)
    token_p95: int = Field(ge=0)
    token_max: int = Field(ge=0)
    token_estimate_method: Literal["cjk_char_ascii4_v1"] = "cjk_char_ascii4_v1"
    truncation_rates: dict[str, float]
    exact_duplicate_rate: float = Field(ge=0, le=1)
    empty_text_count: int = Field(ge=0)
    invalid_tool_call_count: int = Field(ge=0)
    label_counts: dict[str, int]
    slice_counts: dict[str, int] = Field(default_factory=dict)
    generation_acceptance_rate: Optional[float] = Field(default=None, ge=0, le=1)
    rejection_counts: dict[str, int] = Field(default_factory=dict)


class ModelInventory(BaseModel):
    model_config = STRICT
    model_path: str
    model_exists: bool
    config_exists: bool
    tokenizer_exists: bool
    parameter_billions: Optional[float] = Field(default=None, gt=0)
    context_length: Optional[int] = Field(default=None, gt=0)
    weight_bytes: Optional[int] = Field(default=None, ge=0)


class GpuDevice(BaseModel):
    model_config = STRICT
    index: int = Field(ge=0)
    name: str
    memory_used_mb: int = Field(ge=0)
    memory_total_mb: int = Field(gt=0)
    utilization_pct: int = Field(ge=0, le=100)
    temperature_c: Optional[int] = None
    power_draw_w: Optional[float] = None

    @property
    def memory_free_mb(self) -> int:
        return max(0, self.memory_total_mb - self.memory_used_mb)


class ParameterDecision(BaseModel):
    model_config = STRICT
    parameter: str
    value: Any
    reason: str
    confidence: Literal["low", "medium", "high"]


class TrainingPlan(BaseModel):
    model_config = STRICT
    config: TrainConfig
    dataset_name: str
    eval_dataset_names: dict[Literal["function_call", "subjective"], str]
    dataset_sha256: Optional[str] = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    eval_sha256: dict[Literal["function_call", "subjective"], str] = Field(
        default_factory=dict
    )
    gpus: str
    decisions: list[ParameterDecision]
    estimated_steps: int = Field(ge=1)
    estimated_vram_gb: Optional[float] = Field(default=None, gt=0)
    estimated_hours_low: Optional[float] = Field(default=None, ge=0)
    estimated_hours_high: Optional[float] = Field(default=None, ge=0)
    eta_confidence: Literal["low", "medium", "high"] = "low"
    eta_basis: str = "cold_start"
    max_training_hours: Optional[float] = Field(default=None, gt=0)
    risks: list[str] = Field(default_factory=list)


class CheckResult(BaseModel):
    model_config = STRICT
    name: str
    status: Literal["pass", "warn", "block"]
    summary: str
    evidence: dict[str, Any] = Field(default_factory=dict)
    remediation: str = ""


class PreflightReport(BaseModel):
    model_config = STRICT
    status: Literal["pass", "warn", "block"]
    checks: list[CheckResult]
    model: ModelInventory
    gpus: list[GpuDevice]


class EvaluationPlan(BaseModel):
    model_config = STRICT
    baseline_name: str = "baseline"
    candidate_name: str = "candidate"
    eval_dataset_names: dict[Literal["function_call", "subjective"], str]
    task_types: list[Literal["function_call", "subjective"]] = Field(min_length=1)
    gpus: str
    success_criteria: SuccessCriteria
    critical_slices: list[str] = Field(default_factory=list)
    execution_request: Optional[EvalRequest] = None
    eval_sha256: dict[Literal["function_call", "subjective"], str] = Field(
        default_factory=dict
    )
    baseline_source_hash: Optional[str] = Field(
        default=None, pattern=r"^sha256:[0-9a-f]{64}$"
    )
    candidate_source_hash: Optional[str] = Field(
        default=None, pattern=r"^sha256:[0-9a-f]{64}$"
    )


class EvaluationDiagnosis(BaseModel):
    model_config = STRICT
    category: Literal[
        "accept_candidate",
        "data_coverage_gap",
        "annotation_or_pair_quality",
        "template_or_protocol_mismatch",
        "underfit",
        "overfit",
        "evaluation_quality_issue",
    ]
    accepted: bool
    summary: str
    evidence: list[str]
    next_data_changes: list[str] = Field(default_factory=list)
    next_training_changes: list[str] = Field(default_factory=list)
