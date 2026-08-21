"""Deterministic training-parameter policy for the personal assistant."""

from __future__ import annotations

import math
import os
from collections.abc import Iterable, Mapping
from typing import Any

from .assistant_schema import (
    DatasetProfile,
    GpuDevice,
    ModelInventory,
    ParameterDecision,
    TrainingObjective,
    TrainingPlan,
)
from .schema import TrainConfig

CUTOFF_BUCKETS = (512, 1024, 2048, 4096, 8192)


def choose_cutoff(token_p95: int, context_length: int | None) -> int:
    target = max(1, math.ceil(token_p95 * 1.10))
    chosen = next((n for n in CUTOFF_BUCKETS if n >= target), CUTOFF_BUCKETS[-1])
    if context_length:
        chosen = min(chosen, context_length)
    return chosen


def choose_epochs(n_records: int) -> int:
    if n_records < 500:
        return 4
    if n_records < 2000:
        return 3
    if n_records < 10000:
        return 2
    return 1


def estimate_vram_gb(
    parameter_billions: float, stage: str, quantization_bit: int | None
) -> float:
    if quantization_bit == 4:
        base = parameter_billions / 2.0
    elif quantization_bit == 8:
        base = parameter_billions
    else:
        base = parameter_billions * 2.0
    if stage == "dpo":
        base *= 1.35
    return round(base * 1.20, 2)


def _decision(parameter: str, value: Any, reason: str) -> ParameterDecision:
    return ParameterDecision(
        parameter=parameter, value=value, reason=reason, confidence="high"
    )


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _compatible_speeds(
    runs: Iterable[Mapping[str, Any]],
    *,
    stage: str,
    model_size: float | None,
    gpu_names: list[str],
    cutoff_len: int,
    quantization_bit: int | None,
) -> list[float]:
    expected_names = sorted(gpu_names)
    speeds: list[float] = []
    for run in runs:
        if run.get("stage") != stage:
            continue
        if run.get("terminal_status", "succeeded") != "succeeded":
            continue
        if int(run.get("gpu_count", -1)) != len(expected_names):
            continue
        if sorted(run.get("gpu_names") or []) != expected_names:
            continue
        if int(run.get("cutoff_len", -1)) != cutoff_len:
            continue
        if run.get("quantization_bit") != quantization_bit:
            continue
        stored_size = run.get("model_parameter_billions")
        if model_size is not None:
            if stored_size is None:
                continue
            if abs(float(stored_size) - model_size) / model_size > 0.20:
                continue
        try:
            speed = float(run.get("steps_per_second", 0))
        except (TypeError, ValueError):
            continue
        if speed > 0:
            speeds.append(speed)
    return sorted(speeds)


def _estimate_hours(
    estimated_steps: int, speeds: list[float]
) -> tuple[float, float, str, str]:
    if len(speeds) >= 3:
        slow = _percentile(speeds, 0.25)
        fast = _percentile(speeds, 0.75)
        confidence = "high"
        basis = f"historical_p25_p75_{len(speeds)}"
    elif speeds:
        slow = min(speeds) * 0.80
        fast = max(speeds) * 1.20
        confidence = "medium"
        basis = f"historical_expanded_{len(speeds)}"
    else:
        try:
            slow = float(
                os.environ.get(
                    "ASSISTANT_COLD_START_STEPS_PER_SECOND_LOW", "0.05"
                )
            )
            fast = float(
                os.environ.get(
                    "ASSISTANT_COLD_START_STEPS_PER_SECOND_HIGH", "1.0"
                )
            )
        except ValueError as exc:
            raise ValueError("cold-start throughput bounds must be numeric") from exc
        if slow <= 0 or fast <= 0 or slow > fast:
            raise ValueError("cold-start throughput bounds must be positive and ordered")
        confidence = "low"
        basis = "cold_start_config"
    low_hours = estimated_steps / fast / 3600
    high_hours = estimated_steps / slow / 3600
    return (
        round(low_hours, 6),
        round(high_hours, 6),
        confidence,
        basis,
    )


def recommend_training(
    objective: TrainingObjective,
    profile: DatasetProfile,
    model: ModelInventory,
    gpus: list[GpuDevice],
    template: str,
    historical_runs: Iterable[Mapping[str, Any]] = (),
) -> TrainingPlan:
    """Recommend a validated, reproducible LlamaFactory training configuration."""
    if not gpus:
        raise ValueError("at least one GPU is required")
    selected_gpus = sorted(gpus, key=lambda item: item.index)
    gpu_count = len(selected_gpus)
    gpu_names = [item.name for item in selected_gpus]
    gpu_selector = ",".join(str(item.index) for item in selected_gpus)
    # LlamaFactory's FORCE_TORCHRUN path uses replicated DDP unless an explicit
    # sharding strategy is selected, so the model must fit the smallest device.
    available_per_device_gb = min(item.memory_free_mb for item in selected_gpus) / 1024

    stage = profile.finetune_type
    cutoff_len = choose_cutoff(profile.token_p95, model.context_length)
    epochs = choose_epochs(profile.n_records)
    target_global_batch = 16 if stage == "dpo" else 32
    gradient_accumulation = min(
        64, max(1, math.ceil(target_global_batch / gpu_count))
    )
    actual_global_batch = gpu_count * gradient_accumulation
    learning_rate = 5.0e-6 if stage == "dpo" else 1.0e-4

    risks: list[str] = []
    quantization_bit: int | None = None
    estimated_vram: float | None = None
    if model.parameter_billions is not None:
        estimated_vram = estimate_vram_gb(model.parameter_billions, stage, None)
        if estimated_vram > available_per_device_gb:
            quantization_bit = 4
            estimated_vram = estimate_vram_gb(model.parameter_billions, stage, 4)
            risks.append(
                "预计非量化 LoRA 显存不足，已切换 4-bit bitsandbytes QLoRA；"
                "需在预检中确认量化依赖与实际余量。"
            )
            if estimated_vram > available_per_device_gb:
                risks.append(
                    "阻断风险：即使使用 4-bit QLoRA，预计显存仍超过当前可用显存。"
                )
    else:
        risks.append("模型参数规模未知，无法在参数规划阶段估算显存。")

    effective_records = max(
        1, math.floor(profile.n_records * (1 - profile.validation_ratio))
    )
    estimated_steps = math.ceil(effective_records * epochs / actual_global_batch)
    speeds = _compatible_speeds(
        historical_runs,
        stage=stage,
        model_size=model.parameter_billions,
        gpu_names=gpu_names,
        cutoff_len=cutoff_len,
        quantization_bit=quantization_bit,
    )
    eta_low, eta_high, eta_confidence, eta_basis = _estimate_hours(
        estimated_steps, speeds
    )

    config = TrainConfig.model_validate(
        {
            "model": {
                "model_name_or_path": objective.base_model_path,
                "trust_remote_code": True,
                "flash_attn": "auto",
                "quantization_bit": quantization_bit,
                "quantization_method": "bitsandbytes",
                "double_quantization": True,
                "disable_gradient_checkpointing": False,
            },
            "method": {
                "stage": stage,
                "do_train": True,
                "finetuning_type": "lora",
                "lora_rank": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.0,
                "lora_target": "all",
                "pref_beta": 0.1 if stage == "dpo" else None,
                "pref_loss": "sigmoid" if stage == "dpo" else None,
                "include_effective_tokens_per_second": True,
            },
            "dataset": {
                "dataset": profile.dataset_name,
                "template": template,
                "cutoff_len": cutoff_len,
                "max_samples": profile.n_records,
                "packing": False,
                "tool_format": "qwen" if "fc" in profile.task_types else None,
            },
            "output": {
                "output_dir": f"saves/assistant/{profile.dataset_name}/{stage}",
                "logging_steps": 10,
                "save_strategy": "epoch",
                "plot_loss": True,
                "overwrite_output_dir": True,
                "save_only_model": False,
                "report_to": "none",
            },
            "train": {
                "per_device_train_batch_size": 1,
                "gradient_accumulation_steps": gradient_accumulation,
                "learning_rate": learning_rate,
                "num_train_epochs": epochs,
                "lr_scheduler_type": "cosine",
                "warmup_ratio": 0.1,
                "bf16": True,
                "max_grad_norm": 1.0,
                "seed": 42,
                "deepspeed": "none",
            },
            "eval": {
                "val_size": profile.validation_ratio,
                "per_device_eval_batch_size": 1,
                "eval_strategy": "epoch" if profile.validation_ratio else "no",
            },
        }
    )
    decisions = [
        _decision("stage", stage, "训练数据类型决定 LlamaFactory 阶段"),
        _decision("cutoff_len", cutoff_len, "覆盖 P95 token 长度并受上下文窗口约束"),
        _decision("max_samples", profile.n_records, "使用冻结训练集的全部记录"),
        _decision("learning_rate", learning_rate, "采用 LoRA 阶段基线学习率"),
        _decision("num_train_epochs", epochs, "按训练样本量分桶"),
        _decision("global_batch", actual_global_batch, "按阶段目标批量和 GPU 数量计算"),
    ]
    if quantization_bit is not None:
        decisions.append(
            _decision("quantization_bit", quantization_bit, "非量化 LoRA 预计显存不足")
        )

    return TrainingPlan(
        config=config,
        dataset_name=profile.dataset_name,
        eval_dataset_names=profile.eval_dataset_names,
        gpus=gpu_selector,
        decisions=decisions,
        estimated_steps=estimated_steps,
        estimated_vram_gb=estimated_vram,
        estimated_hours_low=eta_low,
        estimated_hours_high=eta_high,
        eta_confidence=eta_confidence,
        eta_basis=eta_basis,
        max_training_hours=objective.max_training_hours,
        risks=risks,
    )
