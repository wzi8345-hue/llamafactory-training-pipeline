"""Deterministic training-parameter recommendation tests."""

from __future__ import annotations

import pytest

from .assistant_schema import (
    BaselineSpec,
    DataSourceSpec,
    DatasetProfile,
    GpuDevice,
    ModelInventory,
    SuccessCriteria,
    TrainingObjective,
)
from .training_policy import choose_cutoff, choose_epochs, recommend_training


def objective() -> TrainingObjective:
    return TrainingObjective(
        goal="提高 FC 工具调用准确率",
        task_types=["fc"],
        base_model_path="/models/qwen",
        template="qwen3_5_nothink",
        baseline=BaselineSpec(kind="base_model", name="base"),
        data_source=DataSourceSpec(fc_seed_file="sft_data/router_fc/seed.json"),
        success_criteria=SuccessCriteria(primary_metric="tool_name_accuracy"),
    )


def profile(n=1000, token_p95=1800, finetune="sft") -> DatasetProfile:
    return DatasetProfile(
        dataset_name="assistant_wf_it0_train",
        eval_dataset_names={"function_call": "assistant_wf_it0_eval_fc"},
        sha256="a" * 64,
        eval_sha256={"function_call": "b" * 64},
        n_records=n,
        holdout_records=max(1, round(n * 0.1)),
        requested_holdout_ratio=0.1,
        actual_holdout_ratio=0.1,
        validation_ratio=0.1,
        split_seed=42,
        finetune_type=finetune,
        task_types=["fc"],
        char_p50=200,
        char_p95=600,
        char_max=900,
        token_p50=600,
        token_p95=token_p95,
        token_max=5000,
        truncation_rates={
            "512": 0.8,
            "1024": 0.3,
            "2048": 0.03,
            "4096": 0.01,
            "8192": 0.0,
        },
        exact_duplicate_rate=0.01,
        empty_text_count=0,
        invalid_tool_call_count=0,
        label_counts={"plan": n},
    )


def model(size=9.0) -> ModelInventory:
    return ModelInventory(
        model_path="/models/qwen",
        model_exists=True,
        config_exists=True,
        tokenizer_exists=True,
        parameter_billions=size,
        context_length=32768,
    )


def gpu(total=24576, used=1024, index=0) -> GpuDevice:
    return GpuDevice(
        index=index,
        name="RTX 4090",
        memory_used_mb=used,
        memory_total_mb=total,
        utilization_pct=0,
        temperature_c=40,
    )


def test_cutoff_and_epoch_buckets():
    assert choose_cutoff(1800, 32768) == 2048
    assert choose_cutoff(4000, 4096) == 4096
    assert choose_epochs(499) == 4
    assert choose_epochs(500) == 3
    assert choose_epochs(2000) == 2
    assert choose_epochs(10000) == 1


def test_sft_9b_balanced_defaults():
    plan = recommend_training(
        objective(), profile(), model(), [gpu()], template="qwen3_5_nothink"
    )
    assert plan.config.method.stage == "sft"
    assert plan.config.dataset.cutoff_len == 2048
    assert plan.config.dataset.max_samples == 1000
    assert plan.config.dataset.tool_format == "qwen"
    assert plan.config.train.learning_rate == 1.0e-4
    assert plan.config.train.num_train_epochs == 3
    assert plan.config.train.gradient_accumulation_steps == 32
    assert plan.config.model.quantization_bit is None
    assert plan.estimated_hours_low is not None
    assert plan.estimated_hours_high >= plan.estimated_hours_low
    assert plan.eta_confidence == "low"
    assert plan.eta_basis == "cold_start_config"


def test_dpo_uses_lower_lr_and_smaller_global_batch():
    plan = recommend_training(
        objective(), profile(n=2500, finetune="dpo"), model(), [gpu()],
        template="qwen3_5_nothink",
    )
    assert plan.config.method.pref_beta == 0.1
    assert plan.config.method.pref_loss == "sigmoid"
    assert plan.config.train.learning_rate == 5.0e-6
    assert plan.config.train.gradient_accumulation_steps == 16
    assert plan.config.train.num_train_epochs == 2


def test_14b_on_24gb_falls_back_to_four_bit():
    plan = recommend_training(
        objective(), profile(), model(14.0), [gpu()], template="qwen3_5_nothink"
    )
    assert plan.config.model.quantization_bit == 4
    assert plan.config.model.quantization_method == "bitsandbytes"
    assert any("QLoRA" in risk for risk in plan.risks)


def test_ddp_uses_per_device_memory_and_reduces_accumulation():
    plan = recommend_training(
        objective(), profile(), model(14.0), [gpu(index=2), gpu(index=0)],
        template="qwen3_5_nothink",
    )
    assert plan.gpus == "0,2"
    assert plan.config.train.gradient_accumulation_steps == 16
    assert plan.config.model.quantization_bit == 4


def test_compatible_history_produces_stable_high_confidence_eta():
    history = [
        {
            "stage": "sft",
            "model_parameter_billions": 9.0,
            "gpu_names": ["RTX 4090"],
            "gpu_count": 1,
            "cutoff_len": 2048,
            "quantization_bit": None,
            "steps_per_second": speed,
            "terminal_status": "succeeded",
        }
        for speed in (0.4, 0.5, 0.6, 0.7)
    ]
    first = recommend_training(
        objective(), profile(), model(), [gpu()], "qwen3_5_nothink", history
    )
    second = recommend_training(
        objective(), profile(), model(), [gpu()], "qwen3_5_nothink", history
    )
    assert first.eta_confidence == "high"
    assert first.eta_basis == "historical_p25_p75_4"
    assert first.model_dump_json() == second.model_dump_json()


def test_invalid_cold_start_bounds_are_rejected(monkeypatch):
    monkeypatch.setenv("ASSISTANT_COLD_START_STEPS_PER_SECOND_LOW", "2")
    monkeypatch.setenv("ASSISTANT_COLD_START_STEPS_PER_SECOND_HIGH", "1")
    with pytest.raises(ValueError, match="cold-start"):
        recommend_training(
            objective(), profile(), model(), [gpu()], "qwen3_5_nothink"
        )


def test_training_time_budget_is_preserved_for_preflight():
    budgeted = objective().model_copy(update={"max_training_hours": 2.0})
    plan = recommend_training(
        budgeted, profile(), model(), [gpu()], "qwen3_5_nothink"
    )
    assert plan.max_training_hours == 2.0
