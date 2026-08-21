"""部署子系统纯逻辑自检: 命令构建 / 状态解析 / 配置存取。不依赖真实服务器/docker。

运行: python -m pytest llamafactory_pipeline/test_deploy.py -q
"""

from __future__ import annotations

import json
import subprocess
import tempfile

import pytest

from llamafactory_pipeline import deploy, deploy_schema
from llamafactory_pipeline.deploy_schema import DeployConfig


def _cfg(**kw) -> DeployConfig:
    base = dict(container_name="llm2", host_model_path="./Qwen3.6-27B",
                port=8003, api_key="secret", max_model_len=260000,
                gpu_memory_utilization=0.9, max_num_seqs=128,
                reasoning_parser="qwen3", enable_auto_tool_choice=True,
                tool_call_parser="qwen3_xml",
                speculative_config='{"method": "mtp", "num_speculative_tokens": 1}')
    base.update(kw)
    return DeployConfig(**base)


# ── 容器名规范化 ──

def test_normalize_adds_prefix():
    assert deploy_schema.normalize_container_name("llm2") == "vllm-llm2"
    assert deploy_schema.normalize_container_name("vllm-llm2") == "vllm-llm2"


def test_normalize_rejects_bad():
    for bad in ["", "vllm-", "vllm-bad name!", "a;b", "../../etc"]:
        with pytest.raises(ValueError):
            deploy_schema.normalize_container_name(bad)


# ── docker run argv 构建 ──

def test_build_argv_matches_user_command():
    """对照用户给的启动命令, 核心参数齐全。"""
    d = _cfg(gpus="3")
    argv = deploy.build_docker_run_argv(d)
    assert argv[0:3] == ["docker", "run", "-d"]
    assert "--gpus" in argv
    assert argv[argv.index("--gpus") + 1] == "device=3"
    assert "--name" in argv and argv[argv.index("--name") + 1] == "vllm-llm2"
    assert "--restart" in argv and argv[argv.index("--restart") + 1] == "unless-stopped"
    assert "-p" in argv and "8003:8003" in argv
    assert "--log-opt" in argv
    # 挂载: host_model_path → /models/Qwen3.6-27B (basename 推导)
    v_idx = argv.index("-v")
    assert argv[v_idx + 1] == "./Qwen3.6-27B:/models/Qwen3.6-27B"
    assert argv[argv.index("--model") + 1] == "/models/Qwen3.6-27B"
    assert argv[argv.index("--port") + 1] == "8003"
    assert argv[argv.index("--api-key") + 1] == "secret"
    assert argv[argv.index("--max-model-len") + 1] == "260000"
    assert argv[argv.index("--gpu-memory-utilization") + 1] == "0.9"
    assert argv[argv.index("--max-num-seqs") + 1] == "128"
    assert argv[argv.index("--reasoning-parser") + 1] == "qwen3"
    assert "--enable-auto-tool-choice" in argv
    assert argv[argv.index("--tool-call-parser") + 1] == "qwen3_xml"
    assert argv[argv.index("--speculative-config") + 1] == '{"method": "mtp", "num_speculative_tokens": 1}'


def test_build_argv_gpus_all_when_empty():
    d = _cfg(gpus="")
    argv = deploy.build_docker_run_argv(d)
    assert argv[argv.index("--gpus") + 1] == "all"


def test_build_argv_multi_gpu():
    d = _cfg(gpus="0,1")
    argv = deploy.build_docker_run_argv(d)
    assert argv[argv.index("--gpus") + 1] == "device=0,1"


def test_build_argv_omits_empty_optionals():
    d = _cfg(reasoning_parser="", tool_call_parser="", speculative_config="",
             api_key="", enable_auto_tool_choice=False)
    argv = deploy.build_docker_run_argv(d)
    assert "--reasoning-parser" not in argv
    assert "--tool-call-parser" not in argv
    assert "--speculative-config" not in argv
    assert "--api-key" not in argv
    assert "--enable-auto-tool-choice" not in argv


def test_build_argv_lora():
    d = _cfg(enable_lora=True, lora_modules="m1=/out1 m2=/out2", max_lora_rank=64)
    argv = deploy.build_docker_run_argv(d)
    assert "--enable-lora" in argv
    assert argv[argv.index("--lora-modules") + 1] == "m1=/out1 m2=/out2"
    assert argv[argv.index("--max-lora-rank") + 1] == "64"


def test_build_argv_lora_disabled_omits_modules():
    d = _cfg(enable_lora=False, lora_modules="m1=/out1")
    argv = deploy.build_docker_run_argv(d)
    assert "--lora-modules" not in argv


def test_build_argv_extra_args_passthrough():
    d = _cfg(extra_args="--disable-log-requests --uvicorn-log-level warning")
    argv = deploy.build_docker_run_argv(d)
    assert "--disable-log-requests" in argv
    assert "--uvicorn-log-level" in argv


def test_build_argv_explicit_model_path():
    d = _cfg(model_path="/custom/path")
    argv = deploy.build_docker_run_argv(d)
    assert argv[argv.index("--model") + 1] == "/custom/path"
    assert argv[argv.index("-v") + 1].endswith(":/custom/path")


def test_quote_argv_quotes_spaces():
    argv = ["--gpus", "device=0,1", "--model", "/my model/path"]
    s = deploy.quote_argv(argv)
    assert "'/my model/path'" in s


# ── 部署脚本语法 ──

def test_deploy_script_sh_valid():
    s = deploy.build_deploy_script(_cfg())
    # 去掉配置目录创建依赖, 仅校验命令行语法
    with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False) as f:
        f.write(s); p = f.name
    r = subprocess.run(["sh", "-n", p], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


# ── 状态解析 ──

def test_parse_status_running():
    raw = "vllm-llm2\tUp 2 hours\t0.0.0.0:8003->8003/tcp"
    s = deploy.parse_container_status(raw)
    assert s["status"] == "running"
    assert s["name"] == "vllm-llm2"
    assert "8003" in s["ports"]


def test_parse_status_exited():
    s = deploy.parse_container_status("vllm-llm2\tExited (0) 5 minutes ago\t")
    assert s["status"] == "exited"


def test_parse_status_missing():
    assert deploy.parse_container_status("")["status"] == "missing"
    assert deploy.parse_container_status("\n")["status"] == "missing"


def test_parse_status_restarting():
    s = deploy.parse_container_status("vllm-x\tRestarting (1) 3 seconds ago\t")
    assert s["status"] == "restarting"


def test_parse_deploy_list():
    raw = (
        "vllm-llm2\tUp 2 hours\t0.0.0.0:8003->8003/tcp\n"
        "vllm-llm1\tExited (1) 10 min ago\t\n"
        "\n"
    )
    lst = deploy.parse_deploy_list(raw)
    assert len(lst) == 2
    assert lst[0]["status"] == "running"
    assert lst[1]["status"] == "exited"


# ── 脚本结构 ──

def test_status_script_exact_name_filter():
    s = deploy.build_status_script("vllm-llm2")
    assert "name=^/vllm-llm2$" in s  # 精确匹配, 防 vllm-llm2 命中 vllm-llm20


def test_stop_script_quotes_name():
    s = deploy.build_stop_script("vllm-llm2")
    assert "docker stop" in s and "docker rm" in s
    # 注入测试: 名字被 quote
    s2 = deploy.build_stop_script("vllm-a;rm -rf /")
    # normalize 会拒绝非法名, 但 stop 直接接名: 校验调用方应先 normalize
    assert "rm -rf /" not in s2 or "'vllm-a;rm -rf /'" in s2


# ── 配置存取 ──

def test_save_load_config(tmp_path, monkeypatch):
    monkeypatch.setattr(deploy, "_CFG_DIR", tmp_path)
    d = _cfg()
    deploy.save_config(d)
    loaded = deploy.load_config("vllm-llm2")
    assert loaded is not None
    assert loaded.container_name == "vllm-llm2"
    assert loaded.max_model_len == 260000


def test_list_saved_configs(tmp_path, monkeypatch):
    monkeypatch.setattr(deploy, "_CFG_DIR", tmp_path)
    deploy.save_config(_cfg(container_name="a"))
    deploy.save_config(_cfg(container_name="b"))
    names = deploy.list_saved_configs()
    assert "vllm-a" in names and "vllm-b" in names


def test_delete_config(tmp_path, monkeypatch):
    monkeypatch.setattr(deploy, "_CFG_DIR", tmp_path)
    deploy.save_config(_cfg(container_name="a"))
    assert deploy.delete_config("vllm-a") is True
    assert deploy.load_config("vllm-a") is None
    assert deploy.delete_config("vllm-a") is False  # 已删


# ── vLLM metrics 解析 ──

_VLLM_METRICS = """# HELP vllm:num_requests_running Number of requests currently running on GPU.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{model_name="qwen"} 2
# HELP vllm:num_requests_waiting Number of requests waiting.
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting{model_name="qwen"} 0
# HELP vllm:gpu_cache_usage_perc GPU cache usage percentage.
# TYPE vllm:gpu_cache_usage_perc gauge
vllm:gpu_cache_usage_perc 0.35
# HELP vllm:num_preemption Number of preemptions.
# TYPE vllm:num_preemption counter
vllm:num_preemption 0
# HELP vllm:time_to_first_token_seconds Histogram of TTFT.
# TYPE vllm:time_to_first_token_seconds histogram
vllm:time_to_first_token_seconds_bucket{le="0.05"} 10
vllm:time_to_first_token_seconds_bucket{le="0.1"} 40
vllm:time_to_first_token_seconds_bucket{le="0.5"} 80
vllm:time_to_first_token_seconds_bucket{le="+Inf"} 100
vllm:time_to_first_token_seconds_sum 25.0
vllm:time_to_first_token_seconds_count 100
# HELP vllm:time_per_output_token_seconds Histogram of TPOT.
# TYPE vllm:time_per_output_token_seconds histogram
vllm:time_per_output_token_seconds_bucket{le="0.02"} 50
vllm:time_per_output_token_seconds_bucket{le="0.05"} 90
vllm:time_per_output_token_seconds_bucket{le="+Inf"} 100
vllm:time_per_output_token_seconds_sum 4.0
vllm:time_per_output_token_seconds_count 100
"""


def test_parse_prometheus_gauges():
    m = deploy.parse_prometheus(_VLLM_METRICS)
    assert m["running"] == 2
    assert m["waiting"] == 0
    assert m["kv_cache"] == 0.35
    assert m["preemption"] == 0


def test_parse_prometheus_histogram():
    m = deploy.parse_prometheus(_VLLM_METRICS)
    ttft = m["ttft"]
    # 100 样本, sum=25 → avg=0.25
    assert ttft["avg"] == 0.25
    # p50: target=50, le=0.1 时 cum=40<50, le=0.5 时 cum=80>=50 → 0.5
    assert ttft["p50"] == 0.5
    # p95: target=95, le=0.5 cum=80<95, le=+Inf cum=100>=95 → 前一桶 0.5
    assert ttft["p95"] == 0.5
    tpot = m["tpot"]
    assert tpot["avg"] == 0.04
    # p50: target=50, le=0.02 cum=50>=50 → 0.02
    assert tpot["p50"] == 0.02


def test_parse_prometheus_missing_fields():
    """缺字段填 None, 不抛。"""
    m = deploy.parse_prometheus("# only comment\n")
    assert m["running"] is None
    assert m["kv_cache"] is None
    assert m["ttft"]["p50"] is None and m["ttft"]["avg"] is None


def test_parse_prometheus_empty():
    m = deploy.parse_prometheus("")
    assert m["running"] is None and m["waiting"] is None


def test_histogram_quantile_boundaries():
    buckets = [(0.1, 40), (0.5, 80), (float("inf"), 100)]
    assert deploy._histogram_quantile(buckets, 100, 0.5) == 0.5
    assert deploy._histogram_quantile(buckets, 100, 0.95) == 0.5
    assert deploy._histogram_quantile(buckets, 0, 0.5) is None  # count=0
    assert deploy._histogram_quantile([], 100, 0.5) is None     # 空桶


def test_parse_metric_line_skips_comments():
    assert deploy._parse_metric_line("# comment") is None
    assert deploy._parse_metric_line("") is None
    assert deploy._parse_metric_line("vllm:foo 1.5") == ("vllm:foo", None, 1.5)
    assert deploy._parse_metric_line('vllm:bar{le="0.1"} 3') == ("vllm:bar", '{le="0.1"}', 3.0)
