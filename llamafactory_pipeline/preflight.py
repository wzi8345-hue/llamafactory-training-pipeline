"""Read-only remote probes and deterministic training preflight aggregation."""

from __future__ import annotations

import base64
import hashlib
import json
import posixpath
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import remote, schema
from .assistant_schema import (
    CheckResult,
    GpuDevice,
    ModelInventory,
    PreflightReport,
    TrainingPlan,
)


@dataclass
class PreflightInputs:
    ssh_ok: bool
    cli_version: str
    container_ok: bool
    container_paths_ok: bool
    model: ModelInventory
    gpus: list[GpuDevice]
    selected_gpu_ids: list[int]
    dataset_ok: bool
    stage_matches: bool
    disk_free_bytes: int
    disk_required_bytes: int
    bf16_supported: bool | None
    conflicting_jobs: list[str]
    unknown_gpu_jobs: list[str] = field(default_factory=list)
    remote_root_writable: bool = True
    quantization_supported: bool | None = True
    truncation_rate: float | None = None
    resume_checkpoint_ok: bool = True
    output_dir_writable: bool = True
    output_disk_free_bytes: int | None = None
    output_disk_required_bytes: int = 0
    container_configured: bool = False


def _disk_required_bytes(
    *, model_weight_bytes: int, dataset_size_bytes: int
) -> int:
    dataset_workspace = (10 << 30) + max(0, dataset_size_bytes) * 2
    model_outputs = int(max(0, model_weight_bytes) * 0.20)
    return max(dataset_workspace, model_outputs)


def _validate_dataset_meta(
    dataset_meta: dict[str, Any] | None,
    expected_stage: str,
    expected_sha256: str | None = None,
) -> tuple[bool, bool, int]:
    if not dataset_meta or not isinstance(dataset_meta.get("data_path"), str):
        return False, False, 0
    path = Path(dataset_meta["data_path"])
    try:
        if not path.is_file() or not path.stat().st_size:
            return False, False, 0
        if expected_sha256:
            actual_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual_sha256 != expected_sha256:
                return False, False, path.stat().st_size
        record = schema.read_first_record(
            str(path), dataset_meta.get("ext") == ".jsonl"
        )
        schema.detect_sharegpt_format(record)
        actual_stage = schema.detect_finetune_type(record)
        return True, actual_stage == expected_stage, path.stat().st_size
    except (OSError, ValueError, json.JSONDecodeError):
        return False, False, 0


def _output_dir(cfg: remote.RemoteConfig, plan: TrainingPlan) -> str:
    output = plan.config.output.output_dir.rstrip("/")
    if not output.startswith("/"):
        output = cfg.llamafactory_dir.rstrip("/") + "/" + output
    return output


def _path_storage_script(path: str) -> str:
    quoted = shlex.quote(path)
    return "\n".join(
        [
            f"P={quoted}",
            'while [ ! -d "$P" ] && [ "$P" != "/" ]; do P=$(dirname "$P"); done',
            'if [ -d "$P" ] && [ -w "$P" ]; then echo WRITABLE=1; else echo WRITABLE=0; fi',
            'df -Pk "$P" | awk \'NR==2{printf "FREE_KB=%s\\n", $4}\'',
            "",
        ]
    )


def _resume_checkpoint_script(path: str, require_optimizer: bool) -> str:
    q = shlex.quote(path)
    required_state = (
        ' && [ -s "$P/optimizer.pt" ] && [ -s "$P/scheduler.pt" ]'
        if require_optimizer
        else ""
    )
    checkpoint_probe = shlex.quote(
        """import json, os, sys
p = os.path.realpath(sys.argv[1])
with open(os.path.join(p, "trainer_state.json"), encoding="utf-8") as handle:
    state = json.load(handle)
step = state.get("global_step") if isinstance(state, dict) else None
if not (isinstance(step, int) and not isinstance(step, bool) and step >= 0):
    raise SystemExit(1)
direct_names = (
    "adapter_model.safetensors", "adapter_model.bin", "model.safetensors",
    "pytorch_model.bin",
)
direct = any(
    os.path.isfile(os.path.join(p, name)) and os.path.getsize(os.path.join(p, name)) > 8
    for name in direct_names
)
if not direct:
    indexes = [
        os.path.join(p, name)
        for name in ("adapter_model.safetensors.index.json", "model.safetensors.index.json")
        if os.path.isfile(os.path.join(p, name)) and os.path.getsize(os.path.join(p, name)) > 0
    ]
    if not indexes:
        raise SystemExit(1)
    with open(indexes[0], encoding="utf-8") as handle:
        index = json.load(handle)
    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if not (isinstance(weight_map, dict) and weight_map):
        raise SystemExit(1)
    shards = set(weight_map.values())
    if not (shards and all(isinstance(name, str) and name for name in shards)):
        raise SystemExit(1)
    for name in shards:
        full = os.path.realpath(os.path.join(p, name))
        if os.path.commonpath((p, full)) != p:
            raise SystemExit(1)
        if not (os.path.isfile(full) and os.path.getsize(full) > 8):
            raise SystemExit(1)
if sys.argv[2] == "1":
    for name in ("optimizer.pt", "scheduler.pt"):
        full = os.path.join(p, name)
        if not (os.path.isfile(full) and os.path.getsize(full) > 8):
            raise SystemExit(1)
"""
    )
    return "\n".join(
        [
            f"P={q}",
            'if [ -d "$P" ] && [ -r "$P" ] && [ -x "$P" ] '
            '&& [ -s "$P/trainer_state.json" ] '
            f'&& python -c {checkpoint_probe} "$P" '
            + ("1" if require_optimizer else "0")
            + " "
            + required_state
            + "; then echo CHECKPOINT_OK; else echo CHECKPOINT_BAD; fi",
            "",
        ]
    )


def _probe_output_storage(
    cfg: remote.RemoteConfig, plan: TrainingPlan
) -> tuple[bool, int]:
    script = _path_storage_script(_output_dir(cfg, plan))
    if cfg.docker_container:
        script = (
            f"docker exec {shlex.quote(cfg.docker_container)} sh -lc "
            f"{shlex.quote(script)}\n"
        )
    else:
        script = _host_runtime_script(cfg, script)
    raw = remote.run_remote_script(cfg, script, timeout=15)
    values = {
        key: value
        for line in raw.splitlines()
        for key, separator, value in [line.partition("=")]
        if separator
    }
    try:
        free_bytes = int(values.get("FREE_KB", "0")) * 1024
    except ValueError:
        free_bytes = 0
    return values.get("WRITABLE") == "1", free_bytes


def _host_runtime_script(cfg: remote.RemoteConfig, command: str) -> str:
    """Run a probe in the same prefix, cwd and PATH as host-mode training."""
    return remote._host_header(cfg) + "\n" + command.rstrip() + "\n"


def _runtime_probe_script(cfg: remote.RemoteConfig, command: str) -> str:
    """Run a Python/CLI probe inside the exact host or container runtime."""
    if cfg.docker_container:
        return (
            f"docker exec {shlex.quote(cfg.docker_container)} sh -lc "
            f"{shlex.quote(command)}\n"
        )
    return _host_runtime_script(cfg, command)


def _container_paths_script(
    cfg: remote.RemoteConfig, model_path: str
) -> str:
    paths = [model_path, cfg.llamafactory_dir, cfg.remote_root]
    tests = " && ".join(f"test -d {shlex.quote(path)}" for path in paths)
    return tests + "\n"


def _validated_model_path(model_path: str) -> str:
    if (
        not model_path
        or not model_path.startswith("/")
        or "\n" in model_path
        or "\r" in model_path
        or "\x00" in model_path
    ):
        raise ValueError("model path must be a safe absolute POSIX path")
    return model_path


def collect_model_inventory(
    cfg: remote.RemoteConfig, model_path: str
) -> ModelInventory:
    """Inspect model files over SSH without modifying the server or using an LLM."""
    model_path = _validated_model_path(model_path)
    quoted = shlex.quote(model_path)
    script = f"""set -e
MODEL_PATH={quoted}
if [ -d "$MODEL_PATH" ]; then printf 'EXISTS=1\\n'; else printf 'EXISTS=0\\n'; fi
if [ -f "$MODEL_PATH/config.json" ]; then
  printf 'CONFIG_EXISTS=1\\nCONFIG_B64='
  base64 -w0 "$MODEL_PATH/config.json"
  printf '\\n'
else
  printf 'CONFIG_EXISTS=0\\nCONFIG_B64=\\n'
fi
if [ -f "$MODEL_PATH/tokenizer.json" ] || [ -f "$MODEL_PATH/tokenizer_config.json" ]; then
  printf 'TOKENIZER=1\\n'
else
  printf 'TOKENIZER=0\\n'
fi
find "$MODEL_PATH" -maxdepth 1 -name '*.safetensors' -printf '%s\\n' 2>/dev/null |
  awk '{{ total += $1 }} END {{ printf "WEIGHT_BYTES=%d\\n", total + 0 }}'
"""
    probe_script = script
    if cfg.docker_container:
        probe_script = (
            f"docker exec {shlex.quote(cfg.docker_container)} sh -lc "
            f"{shlex.quote(script)}\n"
        )
    raw = remote.run_remote_script(cfg, probe_script, timeout=30)
    values: dict[str, str] = {}
    for line in raw.splitlines():
        key, separator, value = line.partition("=")
        if separator:
            values[key] = value

    model_exists = values.get("EXISTS") == "1"
    config_exists = values.get("CONFIG_EXISTS") == "1"
    tokenizer_exists = values.get("TOKENIZER") == "1"
    config: dict[str, Any] = {}
    encoded_config = values.get("CONFIG_B64", "")
    if config_exists and encoded_config:
        try:
            config = json.loads(base64.b64decode(encoded_config).decode("utf-8"))
        except (ValueError, UnicodeDecodeError, json.JSONDecodeError):
            config_exists = False

    context_length = None
    for key in ("max_position_embeddings", "model_max_length", "seq_length"):
        value = config.get(key)
        if isinstance(value, int) and value > 0:
            context_length = value
            break
    try:
        weight_bytes = max(0, int(values.get("WEIGHT_BYTES", "0")))
    except ValueError:
        weight_bytes = 0
    parameter_billions = (
        weight_bytes / 2 / 1_000_000_000 if weight_bytes > 0 else None
    )
    return ModelInventory(
        model_path=model_path,
        model_exists=model_exists,
        config_exists=config_exists,
        tokenizer_exists=tokenizer_exists,
        parameter_billions=parameter_billions,
        context_length=context_length,
        weight_bytes=weight_bytes,
    )


def collect_gpus(cfg: remote.RemoteConfig) -> list[GpuDevice]:
    devices = []
    for row in remote.gpu_status(cfg):
        if not isinstance(row.get("index"), int) or not isinstance(
            row.get("mem_total"), int
        ):
            continue
        used = row.get("mem_used") if isinstance(row.get("mem_used"), int) else 0
        utilization = row.get("util") if isinstance(row.get("util"), int) else 0
        temperature = (
            row.get("temperature") if isinstance(row.get("temperature"), int) else None
        )
        power = row.get("power_draw")
        devices.append(
            GpuDevice(
                index=row["index"],
                name=str(row.get("name", "GPU")),
                memory_used_mb=used,
                memory_total_mb=row["mem_total"],
                utilization_pct=utilization,
                temperature_c=temperature,
                power_draw_w=float(power) if isinstance(power, (int, float)) else None,
            )
        )
    return devices


def _check(
    name: str,
    status: str,
    summary: str,
    *,
    evidence: dict[str, Any] | None = None,
    remediation: str = "",
) -> CheckResult:
    return CheckResult(
        name=name,
        status=status,
        summary=summary,
        evidence=evidence or {},
        remediation=remediation,
    )


def aggregate_preflight(
    inputs: PreflightInputs, plan: TrainingPlan
) -> PreflightReport:
    """Convert probe facts into a stable pass/warn/block report."""
    checks: list[CheckResult] = []
    checks.append(
        _check(
            "ssh",
            "pass" if inputs.ssh_ok else "block",
            "SSH 可连接" if inputs.ssh_ok else "SSH 无法连接",
            remediation="检查服务器地址、密钥和网络" if not inputs.ssh_ok else "",
        )
    )
    checks.append(
        _check(
            "llamafactory_cli",
            "pass" if inputs.cli_version else "block",
            f"LlamaFactory CLI {inputs.cli_version}"
            if inputs.cli_version
            else "未找到 LlamaFactory CLI",
            remediation="确认环境或容器内已安装 llamafactory-cli"
            if not inputs.cli_version
            else "",
        )
    )
    container_passed = inputs.container_ok and inputs.container_paths_ok
    container_summary = (
        "容器状态与路径可见性正常"
        if inputs.container_configured and container_passed
        else "宿主机模式，不需要容器检查"
        if not inputs.container_configured and container_passed
        else "配置的容器未运行或训练路径不可见"
    )
    checks.append(
        _check(
            "container",
            "pass" if container_passed else "block",
            container_summary,
            evidence={
                "running": inputs.container_ok,
                "paths_visible": inputs.container_paths_ok,
            },
            remediation="启动容器并挂载模型、任务目录和 LlamaFactory 目录"
            if not container_passed
            else "",
        )
    )
    checks.append(
        _check(
            "remote_root",
            "pass" if inputs.remote_root_writable else "block",
            "远程任务目录存在且可写"
            if inputs.remote_root_writable
            else "远程任务目录不存在或不可写",
            remediation="创建远程任务目录并修正当前用户的写权限"
            if not inputs.remote_root_writable
            else "",
        )
    )
    resume_path = plan.config.train.resume_from_checkpoint
    checks.append(
        _check(
            "resume_checkpoint",
            "pass" if inputs.resume_checkpoint_ok else "block",
            (
                f"续训 checkpoint 可读：{resume_path}"
                if resume_path and inputs.resume_checkpoint_ok
                else (
                    f"续训 checkpoint 不可读：{resume_path}"
                    if resume_path
                    else "当前方案不使用 checkpoint 续训"
                )
            ),
            remediation="选择存在且在实际训练环境可读的 checkpoint"
            if not inputs.resume_checkpoint_ok
            else "",
        )
    )

    model_files_ok = (
        inputs.model.model_exists
        and inputs.model.config_exists
        and inputs.model.tokenizer_exists
    )
    checks.append(
        _check(
            "model_files",
            "pass" if model_files_ok else "block",
            "模型、配置和 tokenizer 文件完整"
            if model_files_ok
            else "模型目录缺少必要文件",
            evidence={
                "model_exists": inputs.model.model_exists,
                "config_exists": inputs.model.config_exists,
                "tokenizer_exists": inputs.model.tokenizer_exists,
            },
            remediation="补齐 config.json 和 tokenizer 文件并核对模型路径"
            if not model_files_ok
            else "",
        )
    )
    size_known = inputs.model.parameter_billions is not None
    checks.append(
        _check(
            "model_size",
            "pass" if size_known else "warn",
            f"估算模型规模 {inputs.model.parameter_billions:.2f}B"
            if size_known
            else "无法从权重文件估算模型规模",
            evidence={"weight_bytes": inputs.model.weight_bytes},
            remediation="确认顶层 safetensors 权重完整；当前显存估算需人工复核"
            if not size_known
            else "",
        )
    )
    checks.append(
        _check(
            "dataset",
            "pass" if inputs.dataset_ok else "block",
            "训练数据集已注册" if inputs.dataset_ok else "训练数据集不存在或不可读",
            remediation="重新冻结并注册数据集" if not inputs.dataset_ok else "",
        )
    )
    checks.append(
        _check(
            "stage_compatibility",
            "pass" if inputs.stage_matches else "block",
            "训练 stage 与数据类型一致"
            if inputs.stage_matches
            else "训练 stage 与 SFT/DPO 数据类型不一致",
            remediation="重新生成与数据 finetune_type 一致的训练计划"
            if not inputs.stage_matches
            else "",
        )
    )
    if plan.config.model.quantization_bit is None:
        quantization_status = "pass"
        quantization_summary = "当前方案不需要量化依赖"
    elif inputs.quantization_supported is True:
        quantization_status = "pass"
        quantization_summary = "已确认量化依赖可用"
    elif inputs.quantization_supported is False:
        quantization_status = "block"
        quantization_summary = "训练方案需要量化，但运行环境缺少 bitsandbytes"
    else:
        quantization_status = "warn"
        quantization_summary = "无法确认 bitsandbytes 量化依赖"
    checks.append(
        _check(
            "quantization",
            quantization_status,
            quantization_summary,
            remediation="在 LlamaFactory 实际运行环境安装并验证 bitsandbytes"
            if quantization_status != "pass"
            else "",
        )
    )
    truncation_rate = inputs.truncation_rate
    truncation_warn = isinstance(truncation_rate, (int, float)) and truncation_rate > 0.05
    checks.append(
        _check(
            "truncation",
            "warn" if truncation_warn else "pass",
            (
                f"预计截断率 {truncation_rate:.2%} 超过 5%"
                if truncation_warn
                else (
                    f"预计截断率 {truncation_rate:.2%}"
                    if isinstance(truncation_rate, (int, float))
                    else "未发现可用的截断率估算"
                )
            ),
            evidence={"estimated_rate": truncation_rate},
            remediation="检查长样本并考虑提高 cutoff_len 或重构数据"
            if truncation_warn
            else "",
        )
    )

    by_id = {gpu.index: gpu for gpu in inputs.gpus}
    selected = [by_id[index] for index in inputs.selected_gpu_ids if index in by_id]
    selection_ok = bool(inputs.selected_gpu_ids) and len(selected) == len(
        inputs.selected_gpu_ids
    )
    checks.append(
        _check(
            "gpu_selection",
            "pass" if selection_ok else "block",
            "所选 GPU 均存在" if selection_ok else "GPU 选择为空或包含不存在的设备",
            evidence={
                "selected": inputs.selected_gpu_ids,
                "available": sorted(by_id),
            },
            remediation="刷新 GPU 列表并重新生成训练计划" if not selection_ok else "",
        )
    )
    free_by_gpu_gb = {
        gpu.index: gpu.memory_free_mb / 1024 for gpu in selected
    }
    free_gb = min(free_by_gpu_gb.values(), default=0.0)
    required_gb = plan.estimated_vram_gb
    if not selection_ok:
        memory_status, memory_summary = "block", "无法对无效 GPU 选择进行显存预检"
    elif required_gb is None:
        memory_status, memory_summary = "warn", "缺少显存需求估算，需人工复核"
    elif free_gb < required_gb:
        memory_status, memory_summary = (
            "block",
            f"最小单卡可用显存 {free_gb:.2f}GB，小于预计单卡需求 {required_gb:.2f}GB",
        )
    else:
        memory_status, memory_summary = (
            "pass",
            f"最小单卡可用显存 {free_gb:.2f}GB，预计单卡需求 {required_gb:.2f}GB",
        )
    checks.append(
        _check(
            "gpu_memory",
            memory_status,
            memory_summary,
            evidence={
                "minimum_free_gb": round(free_gb, 2),
                "free_by_gpu_gb": {
                    str(index): round(value, 2)
                    for index, value in free_by_gpu_gb.items()
                },
                "required_per_gpu_gb": required_gb,
            },
            remediation="释放显存、减少并发、改用 QLoRA 或更换 GPU"
            if memory_status == "block"
            else "",
        )
    )

    temperatures = [
        gpu.temperature_c for gpu in selected if gpu.temperature_c is not None
    ]
    hottest = max(temperatures) if temperatures else None
    if hottest is not None and hottest >= 85:
        temp_status, temp_summary = "block", f"GPU 温度达到 {hottest}℃"
    elif hottest is not None and hottest >= 80:
        temp_status, temp_summary = "warn", f"GPU 温度偏高：{hottest}℃"
    else:
        temp_status, temp_summary = "pass", (
            f"GPU 最高温度 {hottest}℃" if hottest is not None else "未获得 GPU 温度"
        )
    checks.append(
        _check(
            "gpu_temperature",
            temp_status,
            temp_summary,
            evidence={"max_temperature_c": hottest},
            remediation="等待 GPU 降温并检查散热" if temp_status != "pass" else "",
        )
    )

    if inputs.conflicting_jobs:
        conflict_status = "block"
        conflict_summary = "所选 GPU 正被其他训练任务占用"
    elif inputs.unknown_gpu_jobs:
        conflict_status = "warn"
        conflict_summary = "历史运行任务缺少 GPU 记录，无法排除冲突"
    else:
        conflict_status = "pass"
        conflict_summary = "未发现 GPU 任务冲突"
    checks.append(
        _check(
            "gpu_conflicts",
            conflict_status,
            conflict_summary,
            evidence={
                "conflicting_jobs": inputs.conflicting_jobs,
                "unknown_gpu_jobs": inputs.unknown_gpu_jobs,
            },
            remediation="等待冲突任务结束；缺少记录的历史任务需人工确认"
            if conflict_status != "pass"
            else "",
        )
    )

    disk_ok = inputs.disk_free_bytes >= inputs.disk_required_bytes
    checks.append(
        _check(
            "disk",
            "pass" if disk_ok else "block",
            "远端磁盘空间充足" if disk_ok else "远端磁盘空间不足",
            evidence={
                "free_bytes": inputs.disk_free_bytes,
                "required_bytes": inputs.disk_required_bytes,
            },
            remediation="清理无关文件或更换输出目录" if not disk_ok else "",
        )
    )
    output_free = (
        inputs.output_disk_free_bytes
        if inputs.output_disk_free_bytes is not None
        else inputs.disk_free_bytes
    )
    output_ok = (
        inputs.output_dir_writable
        and output_free >= inputs.output_disk_required_bytes
    )
    checks.append(
        _check(
            "output_storage",
            "pass" if output_ok else "block",
            "训练输出目录可写且空间充足"
            if output_ok
            else "训练输出目录不可写或空间不足",
            evidence={
                "writable": inputs.output_dir_writable,
                "free_bytes": output_free,
                "required_bytes": inputs.output_disk_required_bytes,
            },
            remediation="修正实际 output_dir 挂载权限、清理空间或更换输出目录"
            if not output_ok
            else "",
        )
    )
    budget = plan.max_training_hours
    eta_low = plan.estimated_hours_low
    eta_high = plan.estimated_hours_high
    if budget is None:
        eta_status, eta_summary = "pass", "未设置训练时长上限"
    elif eta_low is None or eta_high is None:
        eta_status, eta_summary = "warn", "已设时长上限，但 ETA 不完整"
    elif eta_low > budget:
        eta_status = "block"
        eta_summary = f"乐观 ETA {eta_low:.2f}h 仍超过上限 {budget:.2f}h"
    elif eta_high > budget:
        eta_status = "warn"
        eta_summary = f"ETA 上界 {eta_high:.2f}h 超过上限 {budget:.2f}h"
    else:
        eta_status = "pass"
        eta_summary = f"ETA {eta_low:.2f}–{eta_high:.2f}h 在上限 {budget:.2f}h 内"
    checks.append(
        _check(
            "eta_budget",
            eta_status,
            eta_summary,
            evidence={
                "estimated_hours_low": eta_low,
                "estimated_hours_high": eta_high,
                "max_training_hours": budget,
            },
            remediation="减少数据量/epoch，或调整时间预算"
            if eta_status != "pass"
            else "",
        )
    )
    if inputs.bf16_supported is True:
        bf16_status, bf16_summary = "pass", "所选 GPU 支持 BF16"
    elif inputs.bf16_supported is False:
        bf16_status, bf16_summary = "block", "所选 GPU 不支持当前 BF16 配置"
    else:
        bf16_status, bf16_summary = "warn", "无法确认所选 GPU 是否支持 BF16"
    checks.append(
        _check(
            "bf16",
            bf16_status,
            bf16_summary,
            remediation="改用兼容精度或人工确认 GPU 算力"
            if bf16_status != "pass"
            else "",
        )
    )

    statuses = {check.status for check in checks}
    status = "block" if "block" in statuses else "warn" if "warn" in statuses else "pass"
    return PreflightReport(
        status=status, checks=checks, model=inputs.model, gpus=inputs.gpus
    )


def _probe_bf16(
    cfg: remote.RemoteConfig, selected_gpu_ids: list[int]
) -> bool | None:
    try:
        script = (
            "nvidia-smi --query-gpu=index,compute_cap "
            "--format=csv,noheader,nounits\n"
        )
        if cfg.docker_container:
            script = (
                f"docker exec {shlex.quote(cfg.docker_container)} sh -lc "
                f"{shlex.quote(script)}\n"
            )
        raw = remote.run_remote_script(cfg, script, timeout=15)
        capabilities = {}
        for line in raw.splitlines():
            index, separator, capability = line.partition(",")
            if separator:
                capabilities[int(index.strip())] = float(capability.strip())
        if not selected_gpu_ids or any(index not in capabilities for index in selected_gpu_ids):
            return None
        return all(capabilities[index] >= 8.0 for index in selected_gpu_ids)
    except (remote.RemoteError, ValueError):
        return None


def _probe_conflicts(
    cfg: remote.RemoteConfig, selected_gpu_ids: list[int]
) -> tuple[list[str], list[str]]:
    selected = set(selected_gpu_ids)
    conflicts: list[str] = []
    unknown: list[str] = []
    try:
        jobs = remote.list_jobs(cfg)
    except remote.RemoteError:
        return conflicts, unknown
    for job in jobs:
        if job.get("status") != "running" or not isinstance(job.get("job_id"), str):
            continue
        job_id = job["job_id"]
        path = shlex.quote(f"{remote.job_dir(cfg, job_id)}/gpus")
        try:
            raw = remote.run_remote_script(
                cfg,
                f"if [ -f {path} ]; then cat {path}; else printf MISSING; fi\n",
                timeout=10,
            ).strip()
        except remote.RemoteError:
            unknown.append(job_id)
            continue
        if not raw or raw == "MISSING":
            unknown.append(job_id)
            continue
        try:
            job_gpus = {int(value) for value in raw.split(",")}
        except ValueError:
            unknown.append(job_id)
            continue
        if selected & job_gpus:
            conflicts.append(job_id)
    return sorted(conflicts), sorted(unknown)


def probe_remote(
    cfg: remote.RemoteConfig,
    plan: TrainingPlan,
    dataset_meta: dict[str, Any] | None,
    dataset_profile: Any | None = None,
) -> PreflightInputs:
    """Collect all remote facts using read-only commands."""
    dataset_ok, stage_matches, dataset_size_bytes = _validate_dataset_meta(
        dataset_meta, plan.config.method.stage, plan.dataset_sha256
    )
    try:
        remote.run_remote_script(cfg, "printf SSH_OK\\n", timeout=10)
        ssh_ok = True
    except remote.RemoteError:
        ssh_ok = False

    cli_version = ""
    container_ok = True
    container_paths_ok = True
    if ssh_ok:
        try:
            if cfg.docker_container:
                container = shlex.quote(cfg.docker_container)
                container_ok = (
                    remote.run_remote_script(
                        cfg,
                        f"docker inspect -f '{{{{.State.Running}}}}' {container}\n",
                        timeout=15,
                    ).strip()
                    == "true"
                )
                if container_ok:
                    path_probe = _container_paths_script(
                        cfg, plan.config.model.model_name_or_path
                    )
                    container_paths_ok = (
                        remote.run_remote_script(
                            cfg,
                            f"docker exec {container} sh -lc {shlex.quote(path_probe)}\n",
                            timeout=15,
                        ).strip()
                        == ""
                    )
                    cli_version = remote.run_remote_script(
                        cfg,
                        f"docker exec {container} llamafactory-cli version\n",
                        timeout=15,
                    ).strip().splitlines()[-1]
            else:
                cli_version = remote.run_remote_script(
                    cfg,
                    _host_runtime_script(cfg, "llamafactory-cli version"),
                    timeout=15,
                ).strip().splitlines()[-1]
        except (remote.RemoteError, IndexError):
            cli_version = ""
            if cfg.docker_container:
                container_paths_ok = False

    remote_root_writable = False
    if ssh_ok:
        try:
            root_path = shlex.quote(cfg.remote_root)
            remote_root_writable = (
                remote.run_remote_script(
                    cfg,
                    f"if [ -d {root_path} ] && [ -w {root_path} ]; then printf WRITABLE; else printf BLOCKED; fi\n",
                    timeout=10,
                ).strip()
                == "WRITABLE"
            )
        except remote.RemoteError:
            remote_root_writable = False

    quantization_supported: bool | None = True
    if plan.config.model.quantization_bit is not None:
        quantization_supported = None
        if ssh_ok:
            probe = "python -c 'import bitsandbytes' >/dev/null 2>&1 && printf OK || printf MISSING"
            if cfg.docker_container:
                probe = (
                    f"docker exec {shlex.quote(cfg.docker_container)} sh -lc "
                    f"{shlex.quote(probe)}"
                )
            else:
                probe = _host_runtime_script(cfg, probe)
            try:
                quantization_supported = remote.run_remote_script(
                    cfg, probe.rstrip() + "\n", timeout=15
                ).strip() == "OK"
            except remote.RemoteError:
                quantization_supported = None

    resume_checkpoint_ok = True
    resume_path = plan.config.train.resume_from_checkpoint
    if resume_path and ssh_ok:
        expected_parent = posixpath.normpath(_output_dir(cfg, plan))
        resume_checkpoint_ok = (
            not plan.config.output.save_only_model
            and
            resume_path.startswith("/")
            and posixpath.dirname(posixpath.normpath(resume_path))
            == expected_parent
        )
        resume_check = _resume_checkpoint_script(
            resume_path, True
        )
        resume_probe = _runtime_probe_script(cfg, resume_check)
        if resume_checkpoint_ok:
            try:
                lines = remote.run_remote_script(
                    cfg, resume_probe, timeout=15
                ).strip().splitlines()
                resume_checkpoint_ok = bool(lines) and lines[-1] == "CHECKPOINT_OK"
            except remote.RemoteError:
                resume_checkpoint_ok = False

    try:
        model = collect_model_inventory(cfg, plan.config.model.model_name_or_path)
    except (remote.RemoteError, ValueError):
        model = ModelInventory(
            model_path=plan.config.model.model_name_or_path,
            model_exists=False,
            config_exists=False,
            tokenizer_exists=False,
        )
    try:
        gpus = collect_gpus(cfg) if ssh_ok else []
    except remote.RemoteError:
        gpus = []
    selected_gpu_ids = [int(value) for value in plan.gpus.split(",") if value]
    try:
        disk_free_bytes = remote.disk_free(cfg) if ssh_ok else 0
    except remote.RemoteError:
        disk_free_bytes = 0
    disk_required_bytes = (10 << 30) + dataset_size_bytes * 2
    output_disk_required_bytes = max(
        1 << 30, int((model.weight_bytes or 0) * 0.20)
    )
    output_dir_writable = False
    output_disk_free_bytes = 0
    if ssh_ok:
        try:
            output_dir_writable, output_disk_free_bytes = _probe_output_storage(
                cfg, plan
            )
        except remote.RemoteError:
            pass
    profile = (
        dataset_profile.model_dump(mode="json")
        if hasattr(dataset_profile, "model_dump")
        else (dataset_profile or {})
    )
    truncation_rate = (profile.get("truncation_rates") or {}).get(
        str(plan.config.dataset.cutoff_len)
    )
    conflicts, unknown = _probe_conflicts(cfg, selected_gpu_ids) if ssh_ok else ([], [])
    return PreflightInputs(
        ssh_ok=ssh_ok,
        cli_version=cli_version,
        container_ok=container_ok,
        container_paths_ok=container_paths_ok,
        model=model,
        gpus=gpus,
        selected_gpu_ids=selected_gpu_ids,
        dataset_ok=dataset_ok,
        stage_matches=stage_matches,
        disk_free_bytes=disk_free_bytes,
        disk_required_bytes=disk_required_bytes,
        bf16_supported=_probe_bf16(cfg, selected_gpu_ids) if ssh_ok else None,
        conflicting_jobs=conflicts,
        unknown_gpu_jobs=unknown,
        remote_root_writable=remote_root_writable,
        quantization_supported=quantization_supported,
        truncation_rate=truncation_rate,
        resume_checkpoint_ok=resume_checkpoint_ok,
        output_dir_writable=output_dir_writable,
        output_disk_free_bytes=output_disk_free_bytes,
        output_disk_required_bytes=output_disk_required_bytes,
        container_configured=bool(cfg.docker_container),
    )


def run_preflight(
    cfg: remote.RemoteConfig,
    plan: TrainingPlan,
    dataset_meta: dict[str, Any] | None,
    dataset_profile: Any | None = None,
) -> PreflightReport:
    return aggregate_preflight(
        probe_remote(cfg, plan, dataset_meta, dataset_profile), plan
    )
