"""初版纯逻辑自检: 参数模型 / ShareGPT 识别 / 任务ID / 脚本引用 / 状态解析。

运行: python -m pytest llamafactory_pipeline/test_remote.py -q
不依赖真实服务器。
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile

import pytest
import yaml

from . import remote, schema

_DEFAULTS = os.path.join(os.path.dirname(__file__), "defaults.yaml")


def test_official_defaults_validate():
    """defaults.yaml 的可编辑字段应能通过参数模型 (分组后)。"""
    cfg = schema.TrainConfig()  # 模型默认值即官方默认
    flat = schema.flatten_config(cfg, "ds", "data")
    assert flat["stage"] == "sft"
    assert flat["finetuning_type"] == "lora"
    assert flat["dataset"] == "ds"
    assert flat["dataset_dir"] == "data"
    assert "resume_from_checkpoint" not in flat  # None 被丢弃
    assert "eval_dataset" not in flat            # 未启用 eval 被丢弃
    assert "deepspeed" not in flat               # 默认 none 不启用


def test_flatten_dpo_policy_fields():
    cfg = schema.TrainConfig.model_validate({
        "model": {
            "model_name_or_path": "/models/qwen",
            "flash_attn": "auto",
            "quantization_bit": 4,
            "quantization_method": "bitsandbytes",
        },
        "method": {
            "stage": "dpo",
            "finetuning_type": "lora",
            "lora_rank": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "pref_beta": 0.1,
            "pref_loss": "sigmoid",
            "include_effective_tokens_per_second": True,
        },
        "dataset": {"packing": False, "tool_format": "qwen"},
        "train": {"seed": 42, "max_grad_norm": 1.0},
    })
    flat = schema.flatten_config(cfg, "dpo_data", "/jobs/j/data")
    assert flat["stage"] == "dpo"
    assert flat["pref_beta"] == 0.1
    assert flat["pref_loss"] == "sigmoid"
    assert flat["quantization_bit"] == 4
    assert flat["quantization_method"] == "bitsandbytes"
    assert flat["seed"] == 42


def test_sft_defaults_do_not_emit_dpo_or_inactive_quantization_fields():
    flat = schema.flatten_config(schema.TrainConfig(), "sft_data", "/jobs/j/data")
    assert "pref_beta" not in flat
    assert "pref_loss" not in flat
    assert "quantization_bit" not in flat
    assert "quantization_method" not in flat
    assert "double_quantization" not in flat


def test_from_env_reads_yaml_and_env_override(tmp_path, monkeypatch):
    cfg_file = tmp_path / "server_config.yaml"
    cfg_file.write_text(
        "ssh_target: ubuntu@host\nremote_root: /r\nllamafactory_dir: /lf\n"
        "ssh_port: '2222'\ndocker_container: box\n", encoding="utf-8")
    monkeypatch.setenv("TRAIN_CONFIG", str(cfg_file))
    for k in ("TRAIN_SSH_TARGET", "TRAIN_REMOTE_ROOT", "LLAMAFACTORY_DIR",
              "TRAIN_SSH_PORT", "TRAIN_SSH_IDENTITY", "TRAIN_DOCKER_CONTAINER"):
        monkeypatch.delenv(k, raising=False)

    c = remote.RemoteConfig.from_env()
    assert c.ssh_target == "ubuntu@host" and c.ssh_port == "2222"
    assert c.docker_container == "box"

    monkeypatch.setenv("TRAIN_SSH_TARGET", "root@other")  # 环境变量覆盖 yml
    assert remote.RemoteConfig.from_env().ssh_target == "root@other"


def test_from_env_missing_reports(tmp_path, monkeypatch):
    monkeypatch.setenv("TRAIN_CONFIG", str(tmp_path / "nope.yaml"))
    for k in ("TRAIN_SSH_TARGET", "TRAIN_REMOTE_ROOT", "LLAMAFACTORY_DIR"):
        monkeypatch.delenv(k, raising=False)
    with pytest.raises(remote.RemoteError):
        remote.RemoteConfig.from_env()


def test_host_launch_includes_remote_prefix():
    cfg = remote.RemoteConfig(
        ssh_target="u@h", remote_root="/r", llamafactory_dir="/lf",
        remote_prefix="conda activate lf")
    script = remote.build_launch_script(cfg, "20240101T000000Z-abcdef")
    assert "conda activate lf" in script
    assert "docker exec" not in script  # 宿主机模式


def test_docker_launch_ignores_remote_prefix():
    cfg = remote.RemoteConfig(
        ssh_target="u@h", remote_root="/r", llamafactory_dir="/lf",
        docker_container="box", remote_prefix="conda activate lf")
    script = remote.build_launch_script(cfg, "20240101T000000Z-abcdef")
    assert "docker exec" in script
    assert "conda activate lf" not in script


def test_parse_trainer_log():
    raw = (
        '{"current_steps": 5, "total_steps": 30, "loss": 2.5, "lr": 9e-05, "epoch": 0.5,'
        ' "percentage": 16.6, "elapsed_time": "0:01:00", "remaining_time": "0:05:00"}\n'
        '\n'
        '{"current_steps": 10, "total_steps": 30, "loss": 1.8, "epoch": 1.0,'
        ' "percentage": 33.3, "remaining_time": "0:04:00"}\n'
        'not json\n'
        '{"current_steps": 30, "total_steps": 30, "eval_loss": 1.2, "epoch": 3.0}\n'
    )
    out = remote.parse_trainer_log(raw)
    assert out["total_steps"] == 30
    assert len(out["points"]) == 3
    assert out["points"][0] == {"step": 5, "loss": 2.5, "epoch": 0.5, "lr": 9e-05}
    assert out["points"][1]["loss"] == 1.8
    assert out["points"][2] == {"step": 30, "eval_loss": 1.2, "epoch": 3.0}
    assert out["percentage"] == 33.3            # 末条为准
    assert out["remaining_time"] == "0:04:00"
    assert out["elapsed_time"] == "0:01:00"     # 末条无该字段则保留上一条


def test_parse_trainer_log_ignores_valid_non_object_json_rows():
    out = remote.parse_trainer_log(
        '[]\n42\n"text"\nnull\n'
        '{"current_steps":1,"total_steps":2,"loss":1.5}\n'
    )

    assert out["total_steps"] == 2
    assert out["points"] == [
        {"step": 1, "loss": 1.5, "epoch": None, "lr": None}
    ]


def test_read_trainer_log_uses_tail(monkeypatch):
    """read_trainer_log 应 tail 增量取尾部, 而非 cat 全量 (大文件轮询优化)。"""
    calls: list[str] = []

    def fake_run(cfg, script, timeout=30):
        calls.append(script)
        if "qwen3_lora_sft.yaml" in script:           # 读 yaml
            return "output_dir: /saves/run1\n"
        return '{"current_steps": 1, "total_steps": 10, "loss": 3.0, "epoch": 0.1}\n'

    monkeypatch.setattr(remote, "run_remote_script", fake_run)
    out = remote.read_trainer_log(_cfg(), "20240101T000000Z-abcdef")
    log_call = [c for c in calls if "trainer_log" in c][0]
    assert "tail -n 400" in log_call
    assert "cat " not in log_call.split("trainer_log")[0]  # 非 cat
    assert out["total_steps"] == 10 and out["points"][0]["loss"] == 3.0


def test_parse_job_list():
    out = (
        "20260721T071950Z-c0e935|train|EXIT 0\n"
        "20260721T071311Z-5bd111|train|RUNNING\n"
        "20260720T010101Z-aaaaaa|eval|EXIT 1\n"
        "garbage line\n"
    )
    jobs = remote.parse_job_list(out)
    assert [j["job_id"] for j in jobs] == [  # 最新在前
        "20260721T071950Z-c0e935",
        "20260721T071311Z-5bd111",
        "20260720T010101Z-aaaaaa",
    ]
    assert jobs[0]["status"] == "succeeded" and jobs[0]["kind"] == "train"
    assert jobs[1]["status"] == "running"
    assert jobs[2]["status"] == "failed" and jobs[2]["exit_code"] == "1"


def test_stop_script_kills_group():
    cfg = remote.RemoteConfig(ssh_target="u@h", remote_root="/r", llamafactory_dir="/lf")
    s = remote.build_stop_script(cfg, "20260721T071950Z-c0e935")
    assert 'kill -TERM -"$P"' in s      # 先杀进程组
    assert "printf 143" in s            # 落 exit_code 使状态转 failed


def test_list_script_matches_job_dirs():
    cfg = remote.RemoteConfig(ssh_target="u@h", remote_root="/r/", llamafactory_dir="/lf")
    s = remote.build_list_script(cfg)
    assert "R=/r\n" in s                # rstrip 尾斜杠
    assert "qwen3_lora_sft.yaml" in s   # train/eval 区分依据


def test_parse_trainer_log_empty():
    out = remote.parse_trainer_log("")
    assert out["points"] == [] and out["total_steps"] == 0
    assert out["percentage"] == "" and out["remaining_time"] == ""


def test_parse_gpu_csv():
    text = "0, NVIDIA A100, 1024, 81920, 5\n1, NVIDIA A100, 40000, 81920, 97\n"
    gpus = remote.parse_gpu_csv(text)
    assert len(gpus) == 2
    assert gpus[0] == {"index": 0, "name": "NVIDIA A100",
                       "mem_used": 1024, "mem_total": 81920, "util": 5}
    assert gpus[1]["util"] == 97


def test_parse_gpu_csv_handles_na_and_blank():
    gpus = remote.parse_gpu_csv("0, GPU, 100, 200, [N/A]\n\nbad line\n")
    assert len(gpus) == 1
    assert gpus[0]["util"] == "[N/A]"


def test_parse_enriched_gpu_csv():
    raw = "0, NVIDIA A100, 1024, 81920, 80896, 5, 42, 71.5\n"
    assert remote.parse_gpu_csv(raw) == [{
        "index": 0,
        "name": "NVIDIA A100",
        "mem_used": 1024,
        "mem_total": 81920,
        "mem_free": 80896,
        "util": 5,
        "temperature": 42,
        "power_draw": 71.5,
    }]


def test_deepspeed_maps_to_config_path():
    cfg = schema.TrainConfig()
    cfg.train.deepspeed = "ds_z3_offload"
    flat = schema.flatten_config(cfg, "ds", "data")
    assert flat["deepspeed"] == "examples/deepspeed/ds_z3_offload_config.json"


def test_extra_field_rejected():
    with pytest.raises(Exception):
        schema.TrainConfig.model_validate({"train": {"nonexistent_field": 1}})


def test_defaults_yaml_matches_model_keys():
    """defaults.yaml 里出现的键 (去掉运行时注入) 都在模型内, 防止清单漂移。"""
    raw = yaml.safe_load(open(_DEFAULTS, encoding="utf-8"))
    described = schema.describe_schema()
    model_keys = {
        field["name"]
        for group in described["groups"]
        for field in group["fields"]
    }
    for k in raw:
        assert k in model_keys, f"defaults.yaml 的 {k} 不在模型中"


def test_sharegpt_conversations():
    rec = {"conversations": [{"from": "human", "value": "hi"},
                             {"from": "gpt", "value": "yo"}], "system": "s"}
    assert schema.detect_sharegpt_format(rec) == "conversations"
    info = schema.build_dataset_info("ds", "train.json", "conversations", rec)
    cols = info["ds"]["columns"]
    assert info["ds"]["formatting"] == "sharegpt"
    assert cols["messages"] == "conversations"
    assert cols["system"] == "system"      # 记录含 system → 映射
    assert "tools" not in cols             # 记录无 tools → 不映射


def test_sharegpt_messages():
    rec = {"messages": [{"role": "user", "content": "hi"},
                        {"role": "assistant", "content": "yo"}]}
    assert schema.detect_sharegpt_format(rec) == "messages"
    info = schema.build_dataset_info("ds", "train.jsonl", "messages", rec)
    assert info["ds"]["tags"]["role_tag"] == "role"
    assert info["ds"]["columns"]["messages"] == "messages"


def test_sharegpt_preference_dataset_info():
    rec = {
        "conversations": [{"from": "human", "value": "Q?"}],
        "chosen": {"from": "gpt", "value": "good"},
        "rejected": {"from": "gpt", "value": "bad"},
    }
    assert schema.detect_finetune_type(rec) == "dpo"
    info = schema.build_dataset_info("ds", "train.json", "conversations", rec)["ds"]
    assert info["ranking"] is True
    assert info["columns"]["chosen"] == "chosen"
    assert info["columns"]["rejected"] == "rejected"


def test_sharegpt_rejects_one_sided_preference_record():
    rec = {
        "conversations": [{"from": "human", "value": "Q?"}],
        "chosen": {"from": "gpt", "value": "good"},
    }
    with pytest.raises(ValueError, match="chosen.*rejected"):
        schema.detect_finetune_type(rec)


def test_sharegpt_unrecognized():
    with pytest.raises(ValueError):
        schema.detect_sharegpt_format({"instruction": "x", "output": "y"})


def test_read_first_record_json_and_jsonl():
    recs = [{"messages": [{"role": "user", "content": "a"}]},
            {"messages": [{"role": "user", "content": "b"}]}]
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(recs, f)
        jpath = f.name
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
        jlpath = f.name
    try:
        assert schema.read_first_record(jpath, is_jsonl=False)["messages"][0]["content"] == "a"
        assert schema.read_first_record(jlpath, is_jsonl=True)["messages"][0]["content"] == "a"
    finally:
        os.unlink(jpath); os.unlink(jlpath)


def test_read_first_record_rejects_non_array_json():
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump({"not": "array"}, f)
        p = f.name
    try:
        with pytest.raises(ValueError):
            schema.read_first_record(p, is_jsonl=False)
    finally:
        os.unlink(p)


def test_job_id_roundtrip_and_validation():
    jid = remote.new_job_id()
    assert remote.validate_job_id(jid) == jid
    for bad in ["../etc", "abc", "2026-01-01", "x; rm -rf /", ""]:
        with pytest.raises(remote.RemoteError):
            remote.validate_job_id(bad)


def _cfg():
    return remote.RemoteConfig(
        ssh_target="u@h", remote_root="/data/jobs",
        llamafactory_dir="/opt/LLaMA-Factory",
    )


def test_launch_script_structure():
    jid = remote.new_job_id()
    s = remote.build_launch_script(_cfg(), jid)
    assert "nohup setsid bash -c" in s  # setsid: 自成进程组, 可整组停止
    assert f"/data/jobs/{jid}/qwen3_lora_sft.yaml" in s
    assert f"/data/jobs/{jid}/train.log" in s
    assert f"/data/jobs/{jid}/launch_identity" in s


def test_launch_script_quotes_special_chars():
    """含空格/分号的目录必须被 shlex.quote 包裹, 防止命令注入。"""
    cfg = remote.RemoteConfig(
        ssh_target="u@h", remote_root="/data/jobs x; touch pwned",
        llamafactory_dir="/opt/LF",
    )
    jid = remote.new_job_id()
    s = remote.build_launch_script(cfg, jid)
    assert "'/data/jobs x; touch pwned/" + jid in s   # 整体被引用
    assert "; touch pwned/" + jid + "/launch_identity'" in s


def test_gpus_validation():
    assert remote.validate_gpus("") == ""
    assert remote.validate_gpus(" 0 ") == "0"
    assert remote.validate_gpus("0,1,2") == "0,1,2"
    for bad in ["a", "0,", ",0", "0 1", "0;1", "-1"]:
        with pytest.raises(remote.RemoteError):
            remote.validate_gpus(bad)


def test_launch_single_gpu_host():
    jid = remote.new_job_id()
    s = remote.build_launch_script(_cfg(), jid, gpus="0")
    assert "CUDA_VISIBLE_DEVICES=0 llamafactory-cli train" in s
    assert "FORCE_TORCHRUN" not in s          # 单卡不开分布式
    assert "cd /opt/LLaMA-Factory" in s


def test_launch_multi_gpu_host_enables_torchrun():
    jid = remote.new_job_id()
    s = remote.build_launch_script(_cfg(), jid, gpus="0,1")
    assert "CUDA_VISIBLE_DEVICES=0,1" in s
    assert "FORCE_TORCHRUN=1" in s


def _cfg_docker():
    return remote.RemoteConfig(
        ssh_target="u@h", remote_root="/data/jobs",
        llamafactory_dir="/opt/LLaMA-Factory", docker_container="llamafactory",
    )


def test_launch_docker_uses_exec_and_gpu_env():
    jid = remote.new_job_id()
    s = remote.build_launch_script(_cfg_docker(), jid, gpus="0,1")
    assert "docker exec" in s
    assert "-e CUDA_VISIBLE_DEVICES=0,1" in s
    assert "-e FORCE_TORCHRUN=1" in s
    assert "-w /opt/LLaMA-Factory llamafactory llamafactory-cli train" in s
    assert "cd /opt/LLaMA-Factory" not in s   # 容器内用 -w, 不在宿主机 cd


def test_launch_docker_no_gpu_still_valid():
    jid = remote.new_job_id()
    s = remote.build_launch_script(_cfg_docker(), jid, gpus="")
    assert "docker exec -w /opt/LLaMA-Factory llamafactory llamafactory-cli train" in s
    assert "CUDA_VISIBLE_DEVICES" not in s


def test_status_script_and_parse():
    jid = remote.new_job_id()
    s = remote.build_status_script(_cfg(), jid)
    assert "kill -0" in s and "NOTFOUND" in s
    assert remote.parse_status("EXIT 0") == {"status": "succeeded", "exit_code": "0"}
    assert remote.parse_status("EXIT 1") == {"status": "failed", "exit_code": "1"}
    assert remote.parse_status("RUNNING") == {"status": "running"}
    assert remote.parse_status("NOTFOUND") == {"status": "not_found"}
    assert remote.parse_status("INTERRUPTED") == {"status": "interrupted"}
    assert remote.parse_status("garbage") == {"status": "unknown"}


def test_status_script_checks_cmdline_for_pid_reuse():
    """PID reuse is fenced by the process starttime, for train and eval jobs."""
    s = remote.build_status_script(_cfg(), remote.new_job_id())
    assert "/proc/$P/stat" in s
    assert "pid_starttime" in s
    assert "cmdline" not in s


def test_list_and_stop_scripts_fence_pid_reuse_with_starttime():
    list_script = remote.build_list_script(_cfg())
    stop_script = remote.build_stop_script(_cfg(), remote.new_job_id())
    for script in (list_script, stop_script):
        assert "/proc/$P/stat" in script
        assert "pid_starttime" in script
        assert "cmdline" not in script
    assert "STALEPID" in stop_script
    assert "st=INTERRUPTED" in list_script


def test_status_script_writes_running_status_file():
    """launch 脚本应在训练前写 status=running, 供重启后判定曾启动。"""
    s = remote.build_launch_script(_cfg(), remote.new_job_id())
    assert "printf running >" in s and "/status" in s
    assert "pid_starttime" in s


def test_training_launch_is_guarded_against_duplicate_job_id():
    s = remote.build_launch_script(_cfg(), remote.new_job_id(), "0")
    assert "launch_identity" in s
    assert "RECOVERED" in s
    assert "flock -w 20 9" in s
    assert "echo BUSY" in s
    assert "echo STARTED" in s


def test_training_launch_ack_requires_complete_process_identity():
    s = remote.build_launch_script(_cfg(), remote.new_job_id(), "0")
    identity_write = s.index("launch_identity.tmp")
    status_write = s.index("status.tmp")
    train_start = s.index("llamafactory-cli train")

    assert identity_write < status_write < train_start
    assert 'if [ -e "$D/pid" ] || [ -e "$D/pid_starttime" ]' in s
    assert "echo NOT_READY" in s


def test_submit_job_requires_durable_launch_ack(tmp_path, monkeypatch):
    data = tmp_path / "train.json"
    data.write_text("[]", encoding="utf-8")
    monkeypatch.setattr(remote, "submission_state", lambda *args: "SAME")
    monkeypatch.setattr(
        remote, "run_remote_script", lambda *args, **kwargs: "BUSY\n"
    )

    with pytest.raises(remote.RemoteError, match="launch acknowledgement"):
        remote.submit_job(
            _cfg(),
            "20240101T000000Z-abcdef",
            "model_name_or_path: /m\n",
            "{}",
            str(data),
            "train.json",
            "0",
        )


def test_submit_job_replay_with_matching_manifest_never_rewrites_inputs(
    tmp_path, monkeypatch
):
    data = tmp_path / "train.json"
    data.write_text("[]", encoding="utf-8")
    scripts = []

    def fake_run(cfg, script, timeout=30):
        scripts.append(script)
        if "submission.sha256" in script and "echo MISSING" in script:
            return "SAME\n"
        return "ALREADY\n"

    monkeypatch.setattr(remote, "run_remote_script", fake_run)
    monkeypatch.setattr(
        remote, "_write_remote_file", lambda *args, **kwargs: pytest.fail("rewrite")
    )
    monkeypatch.setattr(remote, "_scp", lambda *args, **kwargs: pytest.fail("rewrite"))

    remote.submit_job(
        _cfg(),
        "20240101T000000Z-abcdef",
        "model_name_or_path: /m\n",
        "{}",
        str(data),
        "train.json",
        "0",
    )

    assert any("launch.lock" in script for script in scripts)


def test_submit_job_first_publish_stages_then_atomically_finalizes(
    tmp_path, monkeypatch
):
    data = tmp_path / "train.json"
    data.write_text("[]", encoding="utf-8")
    scripts, writes, copies = [], [], []

    def fake_run(cfg, script, timeout=30):
        scripts.append(script)
        if "echo MISSING" in script:
            return "MISSING\n"
        if "mv -T" in script:
            return "CREATED\n"
        return "STARTED\n"

    monkeypatch.setattr(remote, "run_remote_script", fake_run)
    monkeypatch.setattr(
        remote, "_write_remote_file", lambda cfg, path, content: writes.append(path)
    )
    monkeypatch.setattr(
        remote, "_scp", lambda cfg, local, target: copies.append(target)
    )

    remote.submit_job(
        _cfg(),
        "20240101T000000Z-abcdef",
        "model_name_or_path: /m\n",
        "{}",
        str(data),
        "train.json",
        "0",
    )

    assert writes and all("/.staging-" in path for path in writes)
    assert copies and "/.staging-" in copies[0]
    assert any("mv -T" in script for script in scripts)
    assert any("launch.lock" in script for script in scripts)


def test_submission_finalize_is_atomic_and_rejects_hash_conflict():
    script = remote.build_finalize_submission_script(
        _cfg(),
        "20240101T000000Z-abcdef",
        "/data/jobs/.staging-20240101T000000Z-abcdef-token",
        "a" * 64,
    )
    assert "mv -T \"$S\" \"$D\"" in script
    assert "submission.sha256" in script
    assert "CONFLICT" in script


def test_status_script_prefers_exit_code_over_status():
    """exit_code 存在即终态, 不被 status=running 覆盖。脚本文本中 exit_code 判定在前。"""
    s = remote.build_status_script(_cfg(), remote.new_job_id())
    exit_idx = s.index('exit_code')
    status_idx = s.index('"$D/status"')
    assert exit_idx < status_idx  # exit_code 分支在前


def test_log_tail_probe_requires_a_readable_artifact(monkeypatch):
    scripts = []

    def fake_run(cfg, script, timeout=30):
        scripts.append(script)
        raise remote.RemoteError("log unavailable")

    monkeypatch.setattr(remote, "run_remote_script", fake_run)

    with pytest.raises(remote.RemoteError, match="unavailable"):
        remote.read_job_log_tail(
            _cfg(), "20240101T000000Z-abcdef", log_name="eval.log"
        )

    assert "test -r" in scripts[0]
    assert "|| true" not in scripts[0]


# ── checkpoint 列表 ──

def test_parse_checkpoint_list():
    raw = (
        "checkpoint-100|5120\n"
        "checkpoint-20|5120\n"
        "NOTFOUND\n"      # 不应出现 (脚本先 exit), 但解析器要能容忍
        "bad|line|x\n"    # 格式错, 跳过
        "\n"
    )
    ckpts = remote.parse_checkpoint_list(raw)
    assert [c["name"] for c in ckpts] == ["checkpoint-20", "checkpoint-100"]
    assert ckpts[0]["size_kb"] == 5120


def test_parse_checkpoint_list_empty():
    assert remote.parse_checkpoint_list("") == []
    assert remote.parse_checkpoint_list("NOTFOUND") == []


def test_list_checkpoints_script_structure():
    s = remote.build_list_checkpoints_script(_cfg(), "20240101T000000Z-abcdef")
    assert "checkpoint-*" in s
    assert "du -sk" in s
    assert "/opt/LLaMA-Factory" in s
    assert 'case "$OUT" in' in s
    # sh -n 语法校验
    import tempfile, subprocess
    with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False) as f:
        f.write(s); p = f.name
    r = subprocess.run(["sh", "-n", p], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_checkpoint_listing_and_cleanup_run_inside_training_container():
    cfg = _cfg_docker()
    listing = remote.build_list_checkpoints_script(
        cfg, "20240101T000000Z-abcdef"
    )
    cleanup = remote.build_cleanup_checkpoint_script(
        cfg, "20240101T000000Z-abcdef", "checkpoint-100"
    )
    assert "docker exec llamafactory sh -lc" in listing
    assert "docker exec llamafactory sh -lc" in cleanup
    assert "/opt/LLaMA-Factory" in listing


def test_gpu_status_queries_the_training_container(monkeypatch):
    scripts = []
    monkeypatch.setattr(
        remote,
        "run_remote_script",
        lambda cfg, script, timeout=15: scripts.append(script)
        or "0, GPU, 1, 100, 99, 0, 40, 20\n",
    )
    rows = remote.gpu_status(_cfg_docker())
    assert rows[0]["index"] == 0
    assert scripts[0].startswith("docker exec llamafactory sh -lc ")


def test_training_output_probe_requires_adapter_config_and_weights(monkeypatch):
    monkeypatch.setattr(
        remote,
        "resolve_output_dir",
        lambda cfg, job_id: "/opt/LLaMA-Factory/saves/adapter",
    )
    scripts = []
    monkeypatch.setattr(
        remote,
        "run_remote_script",
        lambda cfg, script, timeout=20: scripts.append(script) or "COMPLETE\n",
    )
    evidence = remote.inspect_training_output(
        _cfg_docker(), "20240101T000000Z-abcdef"
    )
    assert evidence["output_evidence_verified"] is True
    assert evidence["output_verified"] is True
    assert scripts[0].startswith("docker exec llamafactory sh -lc ")
    assert "adapter_config.json" in scripts[0]
    assert '[ -s "$P/adapter_config.json" ]' in scripts[0]
    assert '[ -s "$P/adapter_model.safetensors" ]' in scripts[0]


def test_training_output_probe_rejects_empty_config_under_python_optimize(
    tmp_path, monkeypatch
):
    (tmp_path / "adapter_config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "adapter_model.safetensors").write_bytes(b"model-data")
    cfg = remote.RemoteConfig("u@h", "/jobs", str(tmp_path))
    monkeypatch.setattr(
        remote, "resolve_output_dir", lambda cfg, job_id: str(tmp_path)
    )

    def run_locally(cfg, script, timeout=20):
        result = subprocess.run(
            ["bash"],
            input=script,
            text=True,
            capture_output=True,
            env={**os.environ, "PYTHONOPTIMIZE": "1"},
            check=True,
        )
        return result.stdout

    monkeypatch.setattr(remote, "run_remote_script", run_locally)

    evidence = remote.inspect_training_output(
        cfg, "20240101T000000Z-abcdef"
    )

    assert evidence["output_verified"] is False
    assert evidence["output_state"] == "incomplete"


def test_host_training_output_probe_uses_training_runtime(monkeypatch):
    cfg = remote.RemoteConfig(
        "u@h",
        "/jobs",
        "/opt/LF",
        remote_prefix="source /opt/conda/bin/activate lf",
    )
    monkeypatch.setattr(
        remote,
        "resolve_output_dir",
        lambda cfg, job_id: "/opt/LF/saves/adapter",
    )
    scripts = []
    monkeypatch.setattr(
        remote,
        "run_remote_script",
        lambda cfg, script, timeout=20: scripts.append(script) or "COMPLETE\n",
    )

    remote.inspect_training_output(cfg, "20240101T000000Z-abcdef")

    assert "source /opt/conda/bin/activate lf" in scripts[0]
    assert "cd /opt/LF" in scripts[0]


def test_resolve_output_dir_handles_relative(monkeypatch):
    """相对 output_dir 应解析为 llamafactory_dir 下的绝对路径。"""
    calls = []

    def fake_run(cfg, script, timeout=30):
        calls.append(script)
        return "output_dir: saves/run1/lora\n"

    monkeypatch.setattr(remote, "run_remote_script", fake_run)
    out = remote.resolve_output_dir(_cfg(), "20240101T000000Z-abcdef")
    assert out == "/opt/LLaMA-Factory/saves/run1/lora"


@pytest.mark.parametrize("payload", ["[]", "not: [valid", "42"])
def test_resolve_output_dir_wraps_malformed_yaml_as_remote_error(
    monkeypatch, payload
):
    monkeypatch.setattr(
        remote, "run_remote_script", lambda *args, **kwargs: payload
    )

    with pytest.raises(remote.RemoteError, match="训练配置"):
        remote.resolve_output_dir(_cfg(), "20240101T000000Z-abcdef")


# ── checkpoint 清理 ──

def test_cleanup_checkpoint_rejects_bad_name():
    """非法 checkpoint 名 (含 traversal) 必须被拦, 不生成脚本。"""
    cfg = _cfg()
    for bad in ["../etc", "checkpoint-x", "checkpoint-1; rm -rf /", "saves", "checkpoint--1"]:
        with pytest.raises(remote.RemoteError):
            remote.build_cleanup_checkpoint_script(cfg, "20240101T000000Z-abcdef", bad)


def test_cleanup_checkpoint_script_structure():
    s = remote.build_cleanup_checkpoint_script(_cfg(), "20240101T000000Z-abcdef", "checkpoint-100")
    assert "checkpoint-100" in s
    assert "rm -rf" in s
    assert "BADPATH" in s  # 二次校验分支
    import tempfile, subprocess
    with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False) as f:
        f.write(s); p = f.name
    r = subprocess.run(["sh", "-n", p], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_cleanup_checkpoint_only_allows_numeric_suffix():
    """checkpoint-100 合法, checkpoint-100abc 非法。"""
    remote.build_cleanup_checkpoint_script(_cfg(), "20240101T000000Z-abcdef", "checkpoint-100")
    with pytest.raises(remote.RemoteError):
        remote.build_cleanup_checkpoint_script(
            _cfg(), "20240101T000000Z-abcdef", "checkpoint-100abc")


def test_job_dir_rejects_bad_id():
    with pytest.raises(remote.RemoteError):
        remote.job_dir(_cfg(), "bad id")
