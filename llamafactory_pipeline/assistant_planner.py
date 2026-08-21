"""LLM-backed proposal generation with strict deterministic validation."""

from __future__ import annotations

import json
import re
from typing import Any, TypeVar

from pydantic import BaseModel
from rag_eval_plan.common import build_llm, extract_json_object

from .datagen_schema import DatagenConfig
from .assistant_schema import (
    DataPlan,
    DataPlanItem,
    RequirementDraft,
    RequirementField,
    RequirementExtraction,
    TrainingObjective,
    TrainingPlan,
)
from .schema import TrainConfig

T = TypeVar("T", bound=BaseModel)

REQUIREMENT_SYSTEM = """你是个人 LlamaFactory 训练助手的需求分析器，只能提出方案，不能调用工具或执行任何操作。
只返回一个 JSON 对象，键必须严格为 assistant_reply、ready、missing_fields、objective。
你的工作方式是“草案优先”，不是逐字段填写表单。输入包含 messages 与 safe_defaults。
用户说“你决定”“我不管”“按默认”“随机生成”“没有基线”或要求你帮忙选择时，表示授权使用安全默认值；不得重复追问已有默认值的字段。
baseline 表示 A/B 对照模型，不是要求用户提供历史数值；“没有基线”应使用匹配任务安全草案中的 baseline。
safe_defaults.objective_defaults 按 qa、qa_multi、fc、mixed 提供符合 TrainingObjective 字段形状的安全草案；选中匹配任务的草案、复制其中字段并补充从对话总结的 goal，不得把 objective_defaults 等包装键输出到 objective。
objective 只允许 goal、task_types、base_model_path、template、baseline、data_source、success_criteria、requested_finetune_type、max_datagen_records、max_training_hours、critical_slices、examples。
用户要求 9B 模型且未给路径时，采用对应安全草案的 base_model_path，并说明稍后会在预检中验证路径、GPU 和磁盘，不得声称无法查看服务器。
FC 用户要求随机/自动生成数据时，采用 FC 安全草案中的 data_source.fc_seed_file；QA 与 QA-multi 同理使用对应默认来源。
用户说“提升所有指标”时，采用对应安全草案的 success_criteria；不得要求用户提供当前数值才能形成相对提升方案。
只有训练目标或 task_types 等不存在安全默认值、也无法从对话判断的真正阻塞项才可追问；每次最多一次只询问一个最关键问题，同时先给出已确定的初版方案与假设。
当用户信息与安全默认值合起来足以构造合法 objective 时，必须 ready=true，并在 assistant_reply 中用简洁中文给出模型、任务、数据、baseline、成功标准和“等待审批、尚未执行”的方案摘要。
在 ready=true 前必须从用户或 safe_defaults 取得：训练目标、task_types、base_model_path、baseline、每个 task 对应的数据来源和 success_criteria。
QA 数据来源使用 kb_source_dir，QA-multi 使用 collection，FC 使用 fc_seed_file。
若仍缺真正阻塞字段，ready=false、objective=null，并在 assistant_reply 中先总结草案，再一次只询问一个字段。
指标只允许：QA 用 answer_accuracy/score_mean/combined_accuracy，FC 用 tool_name_accuracy/param_score/param_score_mean/combined_accuracy；QA+FC 混合任务必须用 combined_accuracy。
若用户未覆盖：FC 默认以参数分提升 0.15 为主门槛并要求工具名准确率不回归；主观 QA 默认分数提升 0.15。小数据可用于 smoke/人工复核，但成对评测少于 30 条不得自动接受候选。
不得输出 shell、SSH 命令、密钥、工具调用或 JSON 之外的文字。"""

REQUIREMENT_DRAFT_SYSTEM = """你是个人 LlamaFactory 训练助手的需求分析器，只能提出方案，不能调用工具或执行任何操作。
只返回一个 JSON 对象，键严格为 assistant_reply、ready_for_review、missing_fields、scenario、current_problem、desired_behavior、proposed_objective、field_evidence、assumptions。
你的第一职责是先给出“暂定训练方案框架”，包括可能的任务类型、数据构建方式、训练方式与 A/B 验证思路；没有用户证据的内容必须放入 assumptions 并标为 assistant_assumption，不能冒充事实。
框架后一次性询问当前最关键的最多三个问题，三个问题必须在同一轮成组提出，也不得重复追问已有用户证据的字段。
scenario、current_problem、desired_behavior 都使用 {value,source,evidence_message_ids}，其中 source 只允许 user、default、assistant_assumption。
只有用户消息明确表达的业务场景、当前失败表现和期望行为才可标 source=user；evidence_message_ids 必须引用输入 messages 中对应的 user 消息，value 应尽量逐字摘录用户原文。
用户只说模型规模、QA/FC 任务类型、模型路径或“按默认”时，不代表已经说明业务场景、当前问题和期望行为；此时必须 ready_for_review=false，并把缺失项列入 missing_fields，但仍要提供非执行性的暂定框架。
safe_defaults 可用于 proposed_objective 的模型路径、template、baseline、兼容数据来源和成功指标；这些字段在 field_evidence 中标 default 或 assistant_assumption。
当且仅当 scenario、current_problem、desired_behavior 都有用户证据且 TrainingObjective 完整合法时，ready_for_review=true、missing_fields=[]。
proposed_objective 只是一份待确认建议；无论 ready_for_review 为何都不能声称已经执行数据构建或训练。
QA 指标只允许 answer_accuracy/score_mean/combined_accuracy；FC 指标只允许 tool_name_accuracy/param_score/param_score_mean/combined_accuracy；QA+FC 混合任务必须使用 combined_accuracy。
不得输出 shell、SSH 命令、密钥、工具调用或 JSON 之外的文字。"""

DATA_PLAN_SYSTEM = """你是个人 LlamaFactory 训练助手的数据方案规划器，只返回符合 DataPlan 的 JSON 对象。
每个请求的 task_type 必须恰好对应一个 DataPlanItem；只允许 qa、qa_multi、fc 和 sft、dpo。
保留 objective.requested_finetune_type；所有 item 必须使用同一种 finetune_type。
config 只使用 DatagenConfig 字段，每个 item count 至少 2，总 count 不得超过 objective.max_datagen_records。
默认 holdout_ratio=0.1、validation_ratio=0.1、split_seed=42。
根据目标填写有意义的 gen_prompt、judge_prompt；DPO 还要填写 rejected_prompt、pair_judge_prompt。
来源字段会由确定性代码重新绑定，禁止依赖你生成的路径。不得调用工具或执行任务。"""

DIAGNOSIS_SYSTEM = """你只负责把确定性诊断结果解释为简洁中文。
唯一证据是输入的 diagnosis 与 comparison；不得发明根因、指标或已执行动作。
不确定的原因必须明确写为假设。只返回精确 JSON 结构 {"summary":"..."}。"""

ITERATION_PLAN_SYSTEM = """你是个人 LlamaFactory 训练助手的迭代数据规划器，只返回符合 DataPlan 的 JSON 对象。
仅根据 objective、previous_plan、确定性 diagnosis 和 comparison 证据调整方案。
将 diagnosis.next_data_changes 落入覆盖范围、生成 prompt 或质检 prompt。
只有 diagnosis.next_training_changes 有明确证据时，才参考 previous_training_plan 的当前绝对值，在 training_adjustments 中输出具体的 learning_rate、num_train_epochs、lora_rank 或 lora_dropout 新值与理由；值必须保守、方向与诊断一致且在结构限制内。
每个 objective.task_types 必须恰好一项，总 count 不超过原数据预算，不得改变用户锁定的 SFT/DPO。
来源字段由确定性代码重新绑定；不得发明路径、collection，不得调用工具或执行任务。"""


class PlannerOutputError(ValueError):
    pass


CORE_REQUIREMENT_FIELDS = ("scenario", "current_problem", "desired_behavior")


def _normalized_evidence_text(value: Any) -> str:
    return re.sub(r"[\s，。；：、,.!?！？:;‘’“”\-_]+", "", str(value or "")).lower()


def _field_has_user_evidence(
    field: RequirementField | None,
    user_messages: dict[int, str],
) -> bool:
    if field is None or field.source != "user" or not field.evidence_message_ids:
        return False
    needle = _normalized_evidence_text(field.value)
    if not needle:
        return False
    return any(
        needle in _normalized_evidence_text(user_messages.get(message_id, ""))
        for message_id in field.evidence_message_ids
    )


def validate_requirement_evidence(
    draft: RequirementDraft,
    messages: list[dict[str, Any]],
) -> RequirementDraft:
    """Fail closed when core requirement claims are not present in user text."""
    user_messages = {
        int(row["message_id"]): str(row.get("content") or "")
        for row in messages
        if row.get("role") == "user" and row.get("message_id") is not None
    }
    updates: dict[str, Any] = {}
    missing = set(draft.missing_fields)
    assumptions = list(draft.assumptions)
    for name in CORE_REQUIREMENT_FIELDS:
        field = getattr(draft, name)
        if _field_has_user_evidence(field, user_messages):
            continue
        missing.add(name)
        if field is not None and field.source == "user":
            updates[name] = field.model_copy(
                update={
                    "source": "assistant_assumption",
                    "evidence_message_ids": [],
                }
            )
            note = f"{name} 未能在引用的用户原文中验证"
            if note not in assumptions:
                assumptions.append(note)
    if missing:
        updates.update(
            ready_for_review=False,
            missing_fields=sorted(missing),
            assumptions=assumptions,
        )
    return draft.model_copy(update=updates) if updates else draft


class AssistantPlanner:
    def __init__(self, llm=None):
        self.llm = llm or build_llm(timeout=180, max_retries=2)

    @staticmethod
    def _safe_requirement_defaults() -> dict[str, Any]:
        train = TrainConfig()
        datagen = {
            task_type: DatagenConfig(task_type=task_type, count=2)
            for task_type in ("qa", "qa_multi", "fc")
        }
        common = {
            "base_model_path": train.model.model_name_or_path,
            "template": train.dataset.template,
            "baseline": {"kind": "base_model", "name": "base"},
            "requested_finetune_type": "sft",
            "max_datagen_records": train.dataset.max_samples,
        }
        return {
            "objective_defaults": {
                "qa": {
                    **common,
                    "task_types": ["qa"],
                    "data_source": {"kb_source_dir": datagen["qa"].kb_source_dir},
                    "success_criteria": {
                        "primary_metric": "score_mean",
                        "min_improvement": 0.15,
                        "non_regression_metrics": {},
                        "max_invalid_rate_increase": 0.01,
                    },
                },
                "qa_multi": {
                    **common,
                    "task_types": ["qa_multi"],
                    "data_source": {
                        "collection": datagen["qa_multi"].collection
                    },
                    "success_criteria": {
                        "primary_metric": "score_mean",
                        "min_improvement": 0.15,
                        "non_regression_metrics": {},
                        "max_invalid_rate_increase": 0.01,
                    },
                },
                "fc": {
                    **common,
                    "task_types": ["fc"],
                    "data_source": {
                        "fc_seed_file": datagen["fc"].fc_seed_file
                    },
                    "success_criteria": {
                        "primary_metric": "param_score_mean",
                        "min_improvement": 0.15,
                        "non_regression_metrics": {
                            "tool_name_accuracy": 0.0
                        },
                        "max_invalid_rate_increase": 0.01,
                    },
                },
                "mixed": {
                    **common,
                    "task_types": ["qa", "qa_multi", "fc"],
                    "data_source": {
                        "kb_source_dir": datagen["qa"].kb_source_dir,
                        "collection": datagen["qa_multi"].collection,
                        "fc_seed_file": datagen["fc"].fc_seed_file,
                    },
                    "success_criteria": {
                        "primary_metric": "combined_accuracy",
                        "min_improvement": 0.15,
                        "non_regression_metrics": {},
                        "max_invalid_rate_increase": 0.01,
                    },
                },
            },
            "selection_rules": {
                "delegation_phrases": [
                    "你决定", "我不管", "按默认", "随机生成", "没有基线"
                ],
                "proposal_is_not_execution": True,
                "preflight_verifies_model_and_resources": True,
            },
        }

    @staticmethod
    def _delegation_requested(messages: list[dict[str, str]]) -> bool:
        text = "\n".join(
            str(message.get("content") or "")
            for message in messages
            if message.get("role") == "user"
        )
        return any(
            phrase in text
            for phrase in (
                "我不管",
                "你决定",
                "按默认",
                "其余按默认",
                "其他按默认",
                "你看着办",
                "随你",
            )
        )

    @staticmethod
    def _infer_task_types(messages: list[dict[str, str]]) -> list[str]:
        text = "\n".join(
            str(message.get("content") or "")
            for message in messages
            if message.get("role") == "user"
        ).lower()
        tasks: list[str] = []
        if (
            re.search(
                r"(?:^|[^a-z0-9])qa(?:_multi|-multi)(?:$|[^a-z0-9])",
                text,
            )
            or any(
                word in text
                for word in ("多文档问答", "多篇问答", "跨文档问答")
            )
        ):
            tasks.append("qa_multi")
        qa_without_multi = re.sub(r"qa(?:_multi|-multi)", "", text)
        if (
            re.search(r"(?:^|[^a-z0-9])qa(?:$|[^a-z0-9])", qa_without_multi)
            or "知识问答" in text
        ):
            tasks.append("qa")
        if (
            re.search(r"(?:^|[^a-z0-9])fc(?:$|[^a-z0-9])", text)
            or any(
                word in text for word in ("函数调用", "工具调用", "function call")
            )
        ):
            tasks.append("fc")
        return [task for task in ("qa", "qa_multi", "fc") if task in tasks]

    @classmethod
    def _delegated_default_draft(
        cls, messages: list[dict[str, str]]
    ) -> RequirementExtraction | None:
        if not cls._delegation_requested(messages):
            return None
        tasks = cls._infer_task_types(messages)
        defaults = cls._safe_requirement_defaults()["objective_defaults"]
        if not tasks:
            model_path = defaults["fc"]["base_model_path"]
            return RequirementExtraction(
                assistant_reply=(
                    f"我先按 9B 基座模型 {model_path}、SFT、基础模型 A/B 对照"
                    "准备默认草案，模型路径、GPU 和磁盘会在预检时验证。"
                    "目前唯一阻塞项是任务类型：你要训练 QA、跨文档 QA，"
                    "还是 FC 工具调用？"
                ),
                ready=False,
                missing_fields=["task_types"],
                objective=None,
            )

        if len(tasks) == 1:
            objective_data = dict(defaults[tasks[0]])
        else:
            sources: dict[str, str] = {}
            for task in tasks:
                sources.update(defaults[task]["data_source"])
            objective_data = {
                **{
                    key: value
                    for key, value in defaults["mixed"].items()
                    if key not in {"task_types", "data_source"}
                },
                "task_types": tasks,
                "data_source": sources,
                "success_criteria": {
                    **defaults["mixed"]["success_criteria"],
                    "non_regression_metrics": (
                        {"tool_name_accuracy": 0.0} if "fc" in tasks else {}
                    ),
                },
            }
        labels = {"qa": "单文档 QA", "qa_multi": "跨文档 QA", "fc": "FC 工具调用"}
        task_summary = "、".join(labels[task] for task in tasks)
        objective_data["goal"] = f"提升{task_summary}的业务准确性与稳定性"
        user_text = "\n".join(
            str(message.get("content") or "")
            for message in messages
            if message.get("role") == "user"
        ).lower()
        if "dpo" in user_text:
            objective_data["requested_finetune_type"] = "dpo"
        objective = TrainingObjective.model_validate(objective_data)
        source_summary = "、".join(
            f"{key}={value}" for key, value in objective.data_source.model_dump(
                exclude_none=True
            ).items()
        )
        criteria = objective.success_criteria
        return RequirementExtraction(
            assistant_reply=(
                "已按你的授权生成初版方案：\n"
                f"- 模型：{objective.base_model_path}（预检时验证路径、GPU 与磁盘）\n"
                f"- 任务：{task_summary}；方式：{objective.requested_finetune_type.upper()}\n"
                f"- 数据：基于 {source_summary} 自动构建，预算 {objective.max_datagen_records} 条\n"
                "- 对照：基础模型作为 baseline\n"
                f"- 成功标准：{criteria.primary_metric} 至少提升 "
                f"{criteria.min_improvement:g}，并执行非回归与无效率门槛\n"
                "当前仅生成方案并等待审批，尚未执行数据构建或训练。"
            ),
            ready=True,
            missing_fields=[],
            objective=objective,
        )

    def _validated(self, system: str, payload: dict[str, Any], model_cls: type[T]) -> T:
        response = self.llm.chat(
            system=system,
            user=json.dumps(payload, ensure_ascii=False),
            temperature=0.0,
            max_tokens=4096,
            disable_thinking=True,
        )
        obj = extract_json_object(response.get("answer", ""))
        if not obj:
            raise PlannerOutputError("planner did not return a JSON object")
        try:
            return model_cls.model_validate(obj)
        except Exception as exc:
            raise PlannerOutputError(str(exc)) from exc

    def extract_requirements(self, messages: list[dict[str, str]]) -> RequirementExtraction:
        extraction = self._delegated_default_draft(messages)
        if extraction is None:
            extraction = self._validated(
                REQUIREMENT_SYSTEM,
                {
                    "messages": messages,
                    "safe_defaults": self._safe_requirement_defaults(),
                },
                RequirementExtraction,
            )
        objective = extraction.objective
        if objective and "fc" in objective.task_types:
            criteria = objective.success_criteria
            guarded = dict(criteria.non_regression_metrics)
            if (
                criteria.primary_metric != "tool_name_accuracy"
                and "tool_name_accuracy" not in guarded
            ):
                guarded["tool_name_accuracy"] = 0.0
                criteria = criteria.model_copy(
                    update={"non_regression_metrics": guarded}
                )
                objective = objective.model_copy(
                    update={"success_criteria": criteria}
                )
                extraction = extraction.model_copy(update={"objective": objective})
        return extraction

    def extract_requirement_draft(
        self, messages: list[dict[str, Any]]
    ) -> RequirementDraft:
        draft = self._validated(
            REQUIREMENT_DRAFT_SYSTEM,
            {
                "messages": messages,
                "safe_defaults": self._safe_requirement_defaults(),
            },
            RequirementDraft,
        )
        objective = draft.proposed_objective
        if objective and "fc" in objective.task_types:
            criteria = objective.success_criteria
            guarded = dict(criteria.non_regression_metrics)
            if (
                criteria.primary_metric != "tool_name_accuracy"
                and "tool_name_accuracy" not in guarded
            ):
                guarded["tool_name_accuracy"] = 0.0
                objective = objective.model_copy(
                    update={
                        "success_criteria": criteria.model_copy(
                            update={"non_regression_metrics": guarded}
                        )
                    }
                )
                draft = draft.model_copy(update={"proposed_objective": objective})
        return validate_requirement_evidence(draft, messages)

    def create_data_plan(self, objective: TrainingObjective) -> DataPlan:
        payload = {
            "objective": objective.model_dump(mode="json"),
            "allowed_finetune_types": ["sft", "dpo"],
            "allowed_task_types": ["qa", "qa_multi", "fc"],
            "defaults": {
                "holdout_ratio": 0.1,
                "validation_ratio": 0.1,
                "split_seed": 42,
            },
        }
        try:
            plan = self._validated(DATA_PLAN_SYSTEM, payload, DataPlan)
        except PlannerOutputError:
            plan = self._safe_default_data_plan(objective)
        return self._bind_and_validate_data_plan(
            plan, objective, allow_training_adjustments=False
        )

    @staticmethod
    def _safe_default_data_plan(objective: TrainingObjective) -> DataPlan:
        finetune_type = objective.requested_finetune_type or (
            "dpo"
            if any(
                selector.startswith("error_type=")
                for selector in objective.critical_slices
            )
            else "sft"
        )
        per_task = objective.max_datagen_records // len(objective.task_types)
        remainder = objective.max_datagen_records % len(objective.task_types)
        source_fields = {
            "qa": "kb_source_dir",
            "qa_multi": "collection",
            "fc": "fc_seed_file",
        }
        items = []
        for index, task_type in enumerate(objective.task_types):
            count = per_task + (1 if index < remainder else 0)
            base = DatagenConfig(
                task_type=task_type,
                finetune_type=finetune_type,
                count=count,
            )
            source_field = source_fields[task_type]
            update = {
                source_field: getattr(objective.data_source, source_field),
                "gen_prompt": base.resolved_gen_prompt(),
                "judge_prompt": base.resolved_judge_prompt(),
            }
            if finetune_type == "dpo":
                update.update({
                    "rejected_prompt": base.resolved_rejected_prompt(),
                    "pair_judge_prompt": base.resolved_pair_judge_prompt(),
                })
            config = base.model_copy(update=update)
            items.append(DataPlanItem(
                task_type=task_type,
                config=config,
                rationale="采用内置任务模板构建多样化数据并执行严格质检",
            ))
        return DataPlan(
            items=items,
            holdout_ratio=0.1,
            validation_ratio=0.1,
            split_seed=42,
            rationale="模型方案结构异常时采用确定性安全默认数据方案",
            risks=["执行前请在审批卡中复核数据来源、数量和 prompt"],
            training_adjustments=[],
        )

    def create_iteration_plan(
        self,
        objective: TrainingObjective,
        previous_plan: DataPlan,
        previous_training_plan: TrainingPlan,
        diagnosis: Any,
        comparison: Any,
    ) -> DataPlan:
        if isinstance(diagnosis, BaseModel):
            diagnosis = diagnosis.model_dump(mode="json")
        payload = {
            "objective": objective.model_dump(mode="json"),
            "previous_plan": previous_plan.model_dump(mode="json"),
            "previous_training_plan": previous_training_plan.model_dump(mode="json"),
            "diagnosis": diagnosis,
            "comparison": comparison,
            "record_budget": objective.max_datagen_records,
        }
        plan = self._validated(ITERATION_PLAN_SYSTEM, payload, DataPlan)
        return self._bind_and_validate_data_plan(
            plan,
            objective,
            allow_training_adjustments=True,
            previous_plan=previous_plan,
            previous_training_plan=previous_training_plan,
            diagnosis=diagnosis,
        )

    def _bind_and_validate_data_plan(
        self,
        plan: DataPlan,
        objective: TrainingObjective,
        *,
        allow_training_adjustments: bool,
        previous_plan: DataPlan | None = None,
        previous_training_plan: TrainingPlan | None = None,
        diagnosis: dict[str, Any] | None = None,
    ) -> DataPlan:
        if plan.training_adjustments and not allow_training_adjustments:
            raise PlannerOutputError(
                "initial data plan cannot contain training adjustments"
            )
        requested = set(objective.task_types)
        actual = [item.task_type for item in plan.items]
        if len(actual) != len(set(actual)) or set(actual) != requested:
            raise PlannerOutputError(
                "data plan must contain each requested task type exactly once"
            )
        if sum(item.config.count for item in plan.items) > objective.max_datagen_records:
            raise PlannerOutputError("data plan exceeds the approved record budget")
        if any(
            selector.startswith("error_type=")
            for selector in objective.critical_slices
        ) and any(item.config.finetune_type != "dpo" for item in plan.items):
            raise PlannerOutputError(
                "error_type critical slices require a DPO data plan"
            )

        rebound = []
        previous_types = (
            {
                item.task_type: item.config.finetune_type
                for item in previous_plan.items
            }
            if previous_plan
            else {}
        )
        for item in plan.items:
            required_prompts = ["gen_prompt", "judge_prompt"]
            if item.config.finetune_type == "dpo":
                required_prompts.extend(["rejected_prompt", "pair_judge_prompt"])
            missing_prompts = [
                field
                for field in required_prompts
                if not str(getattr(item.config, field, "")).strip()
            ]
            if missing_prompts:
                raise PlannerOutputError(
                    f"data plan {item.task_type} requires non-empty "
                    + ", ".join(missing_prompts)
                )
            if (
                objective.requested_finetune_type
                and item.config.finetune_type != objective.requested_finetune_type
            ):
                raise PlannerOutputError("data plan changed requested finetune type")
            if previous_types and (
                item.config.finetune_type != previous_types[item.task_type]
            ):
                raise PlannerOutputError(
                    "iteration data plan changed previous finetune type"
                )
            source_field = {
                "qa": "kb_source_dir",
                "qa_multi": "collection",
                "fc": "fc_seed_file",
            }[item.task_type]
            source_value = getattr(objective.data_source, source_field)
            config = item.config.model_copy(update={source_field: source_value})
            rebound.append(item.model_copy(update={"config": config}))
        if plan.training_adjustments:
            if not diagnosis or not diagnosis.get("next_training_changes"):
                raise PlannerOutputError(
                    "training adjustment is not supported by diagnosis"
                )
            category = diagnosis.get("category")
            if category not in {"underfit", "overfit"}:
                raise PlannerOutputError(
                    "training adjustment is not supported by diagnosis category"
                )
            if previous_training_plan is None:
                raise PlannerOutputError(
                    "previous training plan is required for adjustments"
                )
            previous_values = {
                "learning_rate": previous_training_plan.config.train.learning_rate,
                "num_train_epochs": (
                    previous_training_plan.config.train.num_train_epochs
                ),
                "lora_rank": previous_training_plan.config.method.lora_rank,
                "lora_dropout": previous_training_plan.config.method.lora_dropout,
            }
            allowed_parameters = {
                "underfit": {"learning_rate", "num_train_epochs", "lora_rank"},
                "overfit": {"num_train_epochs", "lora_dropout"},
            }[category]
            increasing = {"learning_rate", "num_train_epochs", "lora_rank"}
            for adjustment in plan.training_adjustments:
                if adjustment.parameter not in allowed_parameters:
                    raise PlannerOutputError(
                        f"{category} diagnosis does not allow "
                        f"{adjustment.parameter} adjustment"
                    )
                previous_value = float(previous_values[adjustment.parameter])
                new_value = float(adjustment.value)
                if category == "underfit":
                    direction_ok = (
                        new_value >= previous_value
                        if adjustment.parameter in increasing
                        else new_value <= previous_value
                    )
                else:
                    direction_ok = (
                        new_value <= previous_value
                        if adjustment.parameter in increasing
                        else new_value >= previous_value
                    )
                if not direction_ok:
                    raise PlannerOutputError(
                        f"{adjustment.parameter} adjustment direction conflicts "
                        f"with {category} diagnosis"
                    )
        return plan.model_copy(update={"items": rebound})

    def explain_diagnosis(self, diagnosis: Any, comparison: Any) -> str:
        if isinstance(diagnosis, BaseModel):
            diagnosis = diagnosis.model_dump(mode="json")
        response = self.llm.chat(
            system=DIAGNOSIS_SYSTEM,
            user=json.dumps(
                {"diagnosis": diagnosis, "comparison": comparison}, ensure_ascii=False
            ),
            temperature=0.0,
            max_tokens=1200,
            disable_thinking=True,
        )
        obj = extract_json_object(response.get("answer", ""))
        summary = obj.get("summary") if obj else None
        if not isinstance(summary, str) or not summary.strip():
            raise PlannerOutputError("diagnosis explanation is missing summary")
        return summary.strip()
