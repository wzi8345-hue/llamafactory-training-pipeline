"""训练参数模型 + 表单描述 + ShareGPT 数据集注册。

只暴露官方 examples/train_lora/qwen3_lora_sft.yaml 中出现的字段。字段按
model/method/dataset/output/train/eval 分组, 便于前端生成折叠表单。此模块为纯逻辑,
不做任何网络/SSH 操作, 可被 test_remote.py 直接导入测试。
"""

from __future__ import annotations

import json
import typing
from typing import Any, Literal, Optional, get_args, get_origin

from pydantic import BaseModel, ConfigDict, Field

_STRICT = ConfigDict(extra="forbid")


class ModelParams(BaseModel):
    model_config = _STRICT
    # 指向服务器本地权重, 避免容器内联网下载 (需按相同路径挂进容器)
    model_name_or_path: str = "/data/wangzhengyan/Qwen3.5-9B/"
    trust_remote_code: bool = True
    flash_attn: Literal["auto", "disabled", "sdpa", "fa2"] = "auto"
    quantization_bit: Optional[Literal[4, 8]] = None
    quantization_method: Literal["bitsandbytes", "hqq", "eetq"] = "bitsandbytes"
    double_quantization: bool = True
    disable_gradient_checkpointing: bool = False


class MethodParams(BaseModel):
    model_config = _STRICT
    stage: Literal["sft", "pt", "rm", "ppo", "dpo", "kto"] = "sft"
    do_train: bool = True
    finetuning_type: Literal["lora", "full", "freeze"] = "lora"
    lora_rank: int = 8
    lora_alpha: Optional[int] = None
    lora_dropout: float = 0.0
    lora_target: str = "all"
    pref_beta: Optional[float] = None
    pref_loss: Optional[Literal[
        "sigmoid", "hinge", "ipo", "kto_pair", "orpo", "simpo",
    ]] = None
    include_effective_tokens_per_second: bool = False


class DatasetParams(BaseModel):
    model_config = _STRICT
    # dataset 由服务端按上传数据自动覆盖, 前端只读展示
    dataset: str = Field("(由上传数据自动填充)", json_schema_extra={"readonly": True})
    template: str = "qwen3_5_nothink"  # Qwen3.5 对应模板; 换模型记得改
    cutoff_len: int = 2048
    max_samples: int = 1000
    preprocessing_num_workers: int = 16
    dataloader_num_workers: int = 4
    packing: Optional[bool] = None
    tool_format: Optional[str] = None


class OutputParams(BaseModel):
    model_config = _STRICT
    output_dir: str = "saves/qwen3-4b/lora/sft"
    logging_steps: int = 10
    # epoch=每轮存一次 checkpoint; steps=按 save_steps 存; no=不存中间 checkpoint
    save_strategy: Literal["epoch", "steps", "no"] = "epoch"
    save_steps: int = 500  # 仅 save_strategy=steps 时生效
    plot_loss: bool = True
    overwrite_output_dir: bool = True
    save_only_model: bool = False
    report_to: Literal["none", "wandb", "tensorboard", "swanlab", "mlflow"] = "none"


class TrainParams(BaseModel):
    model_config = _STRICT
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1.0e-4
    num_train_epochs: float = 3.0
    lr_scheduler_type: Literal[
        "linear", "cosine", "cosine_with_restarts", "polynomial", "constant",
        "constant_with_warmup",
    ] = "cosine"
    warmup_ratio: float = 0.1
    bf16: bool = True
    max_grad_norm: float = 1.0
    seed: int = 42
    ddp_timeout: int = 180000000
    # DeepSpeed ZeRO 策略 (多卡显存不够时用); 值映射到 LlamaFactory 自带 ds 配置
    deepspeed: Literal[
        "none", "ds_z0", "ds_z2", "ds_z2_offload", "ds_z3", "ds_z3_offload",
    ] = "none"
    resume_from_checkpoint: Optional[str] = None


class EvalParams(BaseModel):
    model_config = _STRICT
    eval_dataset: Optional[str] = None
    val_size: Optional[float] = None
    per_device_eval_batch_size: Optional[int] = None
    eval_strategy: Optional[Literal["no", "steps", "epoch"]] = None
    eval_steps: Optional[int] = None


class TrainConfig(BaseModel):
    model_config = _STRICT
    model: ModelParams = Field(default_factory=ModelParams)
    method: MethodParams = Field(default_factory=MethodParams)
    dataset: DatasetParams = Field(default_factory=DatasetParams)
    output: OutputParams = Field(default_factory=OutputParams)
    train: TrainParams = Field(default_factory=TrainParams)
    eval: EvalParams = Field(default_factory=EvalParams)


# ZeRO 选项 → LlamaFactory 自带配置 (相对其目录, 启动脚本已 cd/-w 到该目录)
_DEEPSPEED_CONFIGS = {
    "ds_z0": "examples/deepspeed/ds_z0_config.json",
    "ds_z2": "examples/deepspeed/ds_z2_config.json",
    "ds_z2_offload": "examples/deepspeed/ds_z2_offload_config.json",
    "ds_z3": "examples/deepspeed/ds_z3_config.json",
    "ds_z3_offload": "examples/deepspeed/ds_z3_offload_config.json",
}


_GROUPS: list[tuple[str, type[BaseModel]]] = [
    ("model", ModelParams),
    ("method", MethodParams),
    ("dataset", DatasetParams),
    ("output", OutputParams),
    ("train", TrainParams),
    ("eval", EvalParams),
]


def _field_type(annotation: Any) -> tuple[str, Optional[list[str]]]:
    """把注解映射为前端控件类型, 返回 (type, enum)。"""
    origin = get_origin(annotation)
    if origin is typing.Union:  # Optional[X] → 取非 None 分支
        inner = [a for a in get_args(annotation) if a is not type(None)]
        if inner:
            return _field_type(inner[0])
    if origin is Literal:
        return "select", [str(v) for v in get_args(annotation)]
    if annotation is bool:
        return "bool", None
    if annotation is int:
        return "int", None
    if annotation is float:
        return "float", None
    return "str", None


def describe_schema() -> dict[str, Any]:
    """生成分组表单描述, 供前端渲染。默认值直接来自模型, 不额外维护。"""
    groups = []
    for group_name, model_cls in _GROUPS:
        fields = []
        for fname, finfo in model_cls.model_fields.items():
            ftype, enum = _field_type(finfo.annotation)
            extra = finfo.json_schema_extra or {}
            fields.append({
                "name": fname,
                "type": ftype,
                "enum": enum,
                "default": finfo.get_default(call_default_factory=True),
                "readonly": bool(extra.get("readonly", False)),
            })
        groups.append({"name": group_name, "fields": fields})
    return {"groups": groups}


def flatten_config(cfg: TrainConfig, dataset_name: str, dataset_dir: str) -> dict[str, Any]:
    """把分组配置摊平为 LlamaFactory 的扁平 YAML dict。

    - 丢弃 None (含未启用的 eval 字段与 resume_from_checkpoint)。
    - dataset 强制为上传数据集名, dataset_dir 为任务隔离目录 (前端不可编辑)。
    """
    flat: dict[str, Any] = {}
    for group_name, _ in _GROUPS:
        group_val = getattr(cfg, group_name).model_dump()
        for k, v in group_val.items():
            if v is None:
                continue
            flat[k] = v
    flat["dataset"] = dataset_name
    flat["dataset_dir"] = dataset_dir
    if flat["stage"] != "dpo":
        flat.pop("pref_beta", None)
        flat.pop("pref_loss", None)
    if "quantization_bit" not in flat:
        flat.pop("quantization_method", None)
        flat.pop("double_quantization", None)
    # deepspeed: "none" → 不启用 (删掉); 否则换成自带配置路径
    ds = flat.pop("deepspeed", "none")
    if ds != "none":
        flat["deepspeed"] = _DEEPSPEED_CONFIGS[ds]
    return flat


# ── ShareGPT 数据识别与注册 ──

def read_first_record(path: str, is_jsonl: bool) -> dict[str, Any]:
    """只读取首条记录, 避免把接近 500MB 的文件整体载入内存。

    - JSONL: 取首个非空行 json.loads。
    - JSON 数组: 定位首个 '[' 后用 raw_decode 解析第一个元素。
    """
    if is_jsonl:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    if not isinstance(rec, dict):
                        raise ValueError("JSONL 首条记录不是对象")
                    return rec
        raise ValueError("JSONL 文件为空")

    # JSON 数组: 有界读取头部
    head = _read_head(path, max_bytes=8 * 1024 * 1024)
    stripped = head.lstrip()
    if not stripped.startswith("["):
        raise ValueError("JSON 顶层必须是数组")
    body = stripped[1:].lstrip()
    if body.startswith("]"):
        raise ValueError("JSON 数组为空")
    rec, _ = json.JSONDecoder().raw_decode(body)
    if not isinstance(rec, dict):
        raise ValueError("JSON 数组首元素不是对象")
    return rec


def _read_head(path: str, max_bytes: int) -> str:
    with open(path, "rb") as f:
        raw = f.read(max_bytes)
    return raw.decode("utf-8", errors="strict")


def detect_sharegpt_format(record: dict[str, Any]) -> str:
    """返回 'conversations' 或 'messages', 无法识别则抛错。"""
    conv = record.get("conversations")
    if isinstance(conv, list) and conv and isinstance(conv[0], dict) \
            and "from" in conv[0] and "value" in conv[0]:
        return "conversations"
    msgs = record.get("messages")
    if isinstance(msgs, list) and msgs and isinstance(msgs[0], dict) \
            and "role" in msgs[0] and "content" in msgs[0]:
        return "messages"
    raise ValueError(
        "无法识别 ShareGPT 结构: 需要 conversations(from/value) 或 messages(role/content)"
    )


def detect_finetune_type(record: dict[str, Any]) -> Literal["sft", "dpo"]:
    """按首条记录识别普通监督数据或偏好数据，并拒绝半条偏好记录。"""
    has_chosen = "chosen" in record
    has_rejected = "rejected" in record
    if has_chosen != has_rejected:
        raise ValueError("DPO 数据必须同时包含 chosen 和 rejected")
    if not has_chosen:
        return "sft"
    if not isinstance(record["chosen"], dict) or not isinstance(record["rejected"], dict):
        raise ValueError("DPO chosen 和 rejected 必须是消息对象")
    return "dpo"


def build_dataset_info(
    dataset_name: str, file_name: str, fmt: str, record: dict[str, Any]
) -> dict[str, Any]:
    """按识别到的结构生成 dataset_info.json 内容。"""
    if fmt == "conversations":
        columns: dict[str, str] = {"messages": "conversations"}
        if "system" in record:
            columns["system"] = "system"
        if "tools" in record:
            columns["tools"] = "tools"
        entry: dict[str, Any] = {
            "file_name": file_name,
            "formatting": "sharegpt",
            "columns": columns,
        }
    else:
        # messages / role / content (OpenAI 风格)
        entry = {
            "file_name": file_name,
            "formatting": "sharegpt",
            "columns": {"messages": "messages"},
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
                "system_tag": "system",
            },
        }
    if detect_finetune_type(record) == "dpo":
        entry["ranking"] = True
        entry["columns"].update({"chosen": "chosen", "rejected": "rejected"})
    return {dataset_name: entry}
