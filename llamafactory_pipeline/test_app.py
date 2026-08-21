"""app 层接口测试: from_datagen_job 送训路径, mock remote 不打真实 SSH/服务器。

运行: python -m pytest llamafactory_pipeline/test_app.py -q
"""

from __future__ import annotations

import json
from html.parser import HTMLParser
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from llamafactory_pipeline import app as app_module
from llamafactory_pipeline import datagen_job, deploy_schema, remote


class _ControlParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = set()
        self.select_options = {}
        self.scripts = []
        self._select = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        element_id = attrs.get("id")
        if element_id:
            self.ids.add(element_id)
        if tag == "select":
            self._select = element_id
            if element_id:
                self.select_options[element_id] = []
        elif tag == "option" and self._select:
            self.select_options[self._select].append(attrs.get("value"))
        elif tag == "script" and attrs.get("src"):
            self.scripts.append(attrs["src"])

    def handle_endtag(self, tag):
        if tag == "select":
            self._select = None


@pytest.fixture
def client(monkeypatch):
    """TestClient + 桩掉 remote.RemoteConfig.from_env 与所有 SSH 调用。"""
    cfg = remote.RemoteConfig(
        ssh_target="u@h", remote_root="/data/jobs", llamafactory_dir="/opt/LF")

    monkeypatch.setattr(remote.RemoteConfig, "from_env", lambda cls=remote.RemoteConfig: cfg)
    # disk_free 返回大值, 不触发 507
    monkeypatch.setattr(remote, "disk_free", lambda c: 1 << 40)
    # submit_job 记录调用参数, 不真上传
    submitted: list[dict] = []
    monkeypatch.setattr(remote, "submit_job",
                        lambda c, jid, yaml_t, info_t, data_path, data_name, gpus:
                        submitted.append({"job_id": jid, "yaml": yaml_t,
                                          "data_path": data_path, "data_name": data_name,
                                          "gpus": gpus}))
    monkeypatch.setattr(remote, "_write_remote_file", lambda *args: None)
    monkeypatch.setattr(app_module, "_save_upload_capped",
                        lambda f: f"/tmp/fake_{f.filename}")
    # datagen 产物目录桩到一个临时根
    base = Path(app_module.__file__).parent / "sft_data" / "generated"
    # 用模块级 datagen_job._BASE, 测试内单独覆盖
    yield TestClient(app_module.app), submitted


def test_index_exposes_sft_dpo_generation_controls(client):
    test_client, _ = client
    response = test_client.get("/")
    assert response.status_code == 200
    parser = _ControlParser()
    parser.feed(response.text)
    assert parser.select_options["dg-finetune"] == ["sft", "dpo"]
    assert {"dg-rejectedp", "dg-pairp", "dg-type"} <= parser.ids
    assert "/static/datagen.js" in parser.scripts
    assert test_client.get("/static/datagen.js").status_code == 200


def test_index_exposes_personal_training_assistant(client):
    test_client, _ = client
    response = test_client.get("/")
    parser = _ControlParser()
    parser.feed(response.text)
    assert {
        "tab-assistant",
        "view-assistant",
        "as-workflows",
        "as-messages",
        "as-input",
        "as-send",
        "as-current",
        "as-events",
    } <= parser.ids
    assert "/static/assistant.js" in parser.scripts
    assert test_client.get("/static/assistant.js").status_code == 200


@pytest.mark.parametrize(
    "script_name",
    [
        "run-llamafactory.sh",
        "run-llamafactory-assistant-monitor.sh",
    ],
)
def test_launch_scripts_expose_eval_package_on_pythonpath(script_name):
    script = Path(__file__).parents[1] / "scripts" / script_name
    content = script.read_text(encoding="utf-8")

    pythonpath_lines = [
        line for line in content.splitlines() if line.startswith("export PYTHONPATH=")
    ]
    assert len(pythonpath_lines) == 1
    assert ".:eval" in pythonpath_lines[0]


def test_create_job_from_datagen_job(client, tmp_path, monkeypatch):
    """from_datagen_job 时直接用生成产物 output.json, 不走上传。"""
    test_client, submitted = client
    # 造一个生成任务产物
    jid = "20260721T071950Z-c0e935"
    monkeypatch.setattr(datagen_job, "_BASE", tmp_path)
    d = datagen_job.job_dir(jid)
    d.mkdir(parents=True)
    (d / "output.json").write_text(json.dumps([
        {"conversations": [{"from": "human", "value": "Q?"},
                           {"from": "gpt", "value": "A."}]}],
        ensure_ascii=False), encoding="utf-8")

    params = {"model": {"model_name_or_path": "/x"}, "dataset": {"template": "qwen3_5_nothink"}}
    r = test_client.post("/api/jobs", data={
        "params": json.dumps(params),
        "gpus": "0",
        "from_datagen_job": jid,
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["format"] == "conversations"
    assert submitted, "submit_job 应被调用"
    # 数据文件路径应指向生成产物 (非上传临时文件)
    assert submitted[0]["data_path"].endswith("output.json")
    assert submitted[0]["data_name"] == "train.json"


def test_create_job_requires_file_or_datagen(client):
    """既无文件又无 from_datagen_job → 400。"""
    test_client, _ = client
    r = test_client.post("/api/jobs", data={"params": json.dumps({}), "gpus": ""})
    assert r.status_code == 400
    assert "上传文件或指定 from_datagen_job" in r.json()["detail"]


def test_create_job_rejects_both_file_and_datagen(client, tmp_path, monkeypatch):
    """同时上传文件和指定 from_datagen_job → 400。"""
    test_client, _ = client
    jid = "20260721T071950Z-c0e935"
    monkeypatch.setattr(datagen_job, "_BASE", tmp_path)
    d = datagen_job.job_dir(jid)
    d.mkdir(parents=True)
    (d / "output.json").write_text("[]", encoding="utf-8")
    r = test_client.post("/api/jobs", data={
        "params": json.dumps({}),
        "from_datagen_job": jid,
    }, files={"file": ("a.json", b"[]", "application/json")})
    assert r.status_code == 400
    assert "不能同时" in r.json()["detail"]


def test_create_job_datagen_not_found(client, tmp_path, monkeypatch):
    """from_datagen_job 指向的任务无产出 → 404。"""
    test_client, _ = client
    monkeypatch.setattr(datagen_job, "_BASE", tmp_path)
    r = test_client.post("/api/jobs", data={
        "params": json.dumps({}),
        "from_datagen_job": "20260721T071950Z-c0e935",
    })
    assert r.status_code == 404


def test_list_datagen_jobs_endpoint(client, tmp_path, monkeypatch):
    test_client, _ = client
    monkeypatch.setattr(datagen_job, "_BASE", tmp_path)
    jid = "20260721T071950Z-c0e935"
    d = datagen_job.job_dir(jid)
    d.mkdir(parents=True)
    (d / "config.json").write_text(
        json.dumps({"task_type": "qa", "count": 10}), encoding="utf-8")
    (d / "progress.json").write_text(
        json.dumps({"state": "done", "accepted": 5, "target": 10}), encoding="utf-8")
    r = test_client.get("/api/datagen/jobs")
    assert r.status_code == 200
    jobs = r.json()["jobs"]
    assert len(jobs) == 1 and jobs[0]["job_id"] == jid
    assert jobs[0]["finetune_type"] == "sft"


def test_create_datagen_job_returns_finetune_type(client, monkeypatch):
    test_client, _ = client
    launched = []
    monkeypatch.setattr(
        datagen_job, "create_and_launch",
        lambda jid, cfg: launched.append((jid, cfg)))
    r = test_client.post("/api/datagen/jobs", data={
        "config": json.dumps({
            "finetune_type": "dpo", "task_type": "fc", "count": 2})})
    assert r.status_code == 200, r.text
    assert r.json()["finetune_type"] == "dpo"
    assert launched[0][1]["finetune_type"] == "dpo"


def test_download_datagen_output_uses_dpo_filename(client, tmp_path, monkeypatch):
    test_client, _ = client
    jid = "20260721T071950Z-c0e935"
    monkeypatch.setattr(datagen_job, "_BASE", tmp_path)
    d = datagen_job.job_dir(jid)
    d.mkdir(parents=True)
    (d / "config.json").write_text(json.dumps({
        "finetune_type": "dpo", "task_type": "qa", "count": 1}), encoding="utf-8")
    (d / "output.json").write_text("[]", encoding="utf-8")
    r = test_client.get(f"/api/datagen/jobs/{jid}/download")
    assert r.status_code == 200
    assert f"dpo_{jid}.json" in r.headers["content-disposition"]


# ── 续训: checkpoints 端点 + output_dir 复用 ──

def test_checkpoints_endpoint(client, monkeypatch):
    """GET /api/jobs/{id}/checkpoints 返回 checkpoint 列表 + 绝对路径。"""
    test_client, _ = client
    monkeypatch.setattr(remote, "list_checkpoints",
                        lambda c, jid: [{"name": "checkpoint-100", "size_kb": 5120}])
    monkeypatch.setattr(remote, "resolve_output_dir",
                        lambda c, jid: "/opt/LLaMA-Factory/saves/run1")
    r = test_client.get("/api/jobs/20260721T071950Z-c0e935/checkpoints")
    assert r.status_code == 200
    body = r.json()
    assert body["output_dir"] == "/opt/LLaMA-Factory/saves/run1"
    assert body["checkpoints"][0]["path"].endswith("checkpoint-100")


def _make_datagen_output(tmp_path, jid="20260721T071950Z-c0e935"):
    """造一个生成任务产物 output.json (conversations 格式)。"""
    d = datagen_job.job_dir(jid)
    d.mkdir(parents=True)
    (d / "output.json").write_text(json.dumps([
        {"conversations": [{"from": "human", "value": "Q?"},
                           {"from": "gpt", "value": "A."}]}],
        ensure_ascii=False), encoding="utf-8")
    return jid


def _make_dpo_datagen_output(tmp_path, jid="20260721T071950Z-c0e935"):
    d = datagen_job.job_dir(jid)
    d.mkdir(parents=True)
    (d / "config.json").write_text(json.dumps({
        "finetune_type": "dpo", "task_type": "qa", "count": 1}), encoding="utf-8")
    (d / "output.json").write_text(json.dumps([{
        "conversations": [{"from": "human", "value": "Q?"}],
        "chosen": {"from": "gpt", "value": "good"},
        "rejected": {"from": "gpt", "value": "bad"},
    }], ensure_ascii=False), encoding="utf-8")
    return jid


def test_create_job_rejects_dpo_generated_data_with_sft_stage(client, tmp_path, monkeypatch):
    test_client, submitted = client
    monkeypatch.setattr(datagen_job, "_BASE", tmp_path)
    jid = _make_dpo_datagen_output(tmp_path)
    r = test_client.post("/api/jobs", data={
        "params": json.dumps({"method": {"stage": "sft"}}),
        "from_datagen_job": jid,
    })
    assert r.status_code == 400
    assert "DPO" in r.json()["detail"] and "stage=dpo" in r.json()["detail"]
    assert not submitted


def test_create_job_accepts_dpo_generated_data_with_dpo_stage(client, tmp_path, monkeypatch):
    test_client, submitted = client
    monkeypatch.setattr(datagen_job, "_BASE", tmp_path)
    jid = _make_dpo_datagen_output(tmp_path)
    r = test_client.post("/api/jobs", data={
        "params": json.dumps({"method": {"stage": "dpo"}}),
        "from_datagen_job": jid,
    })
    assert r.status_code == 200, r.text
    import yaml as _yaml
    assert _yaml.safe_load(submitted[0]["yaml"])["stage"] == "dpo"


def test_create_job_maps_insufficient_disk_to_507(client, tmp_path, monkeypatch):
    test_client, submitted = client
    monkeypatch.setattr(datagen_job, "_BASE", tmp_path)
    jid = _make_datagen_output(tmp_path)
    monkeypatch.setattr(remote, "disk_free", lambda cfg: 1)
    response = test_client.post(
        "/api/jobs",
        data={"params": json.dumps({}), "from_datagen_job": jid},
    )
    assert response.status_code == 507
    assert "磁盘不足" in response.json()["detail"]
    assert not submitted


def test_create_job_rejects_sft_generated_data_with_dpo_stage(client, tmp_path, monkeypatch):
    test_client, submitted = client
    monkeypatch.setattr(datagen_job, "_BASE", tmp_path)
    jid = _make_datagen_output(tmp_path)
    r = test_client.post("/api/jobs", data={
        "params": json.dumps({"method": {"stage": "dpo"}}),
        "from_datagen_job": jid,
    })
    assert r.status_code == 400
    assert "SFT" in r.json()["detail"] and "stage=sft" in r.json()["detail"]
    assert not submitted


def test_resume_keeps_output_dir(client, tmp_path, monkeypatch):
    """提交带 resume_from_checkpoint 时, yaml 的 output_dir 不被 job_id 追加 (复用原目录)。"""
    test_client, submitted = client
    monkeypatch.setattr(datagen_job, "_BASE", tmp_path)
    jid = _make_datagen_output(tmp_path)
    params = {
        "model": {"model_name_or_path": "/x"},
        "dataset": {"template": "qwen3_5_nothink"},
        "output": {"output_dir": "saves/qwen3/lora/sft"},
        "train": {"resume_from_checkpoint": "/opt/LLaMA-Factory/saves/qwen3/lora/sft/20260721T071950Z-c0e935/checkpoint-100"},
    }
    r = test_client.post("/api/jobs", data={
        "params": json.dumps(params), "gpus": "", "from_datagen_job": jid})
    assert r.status_code == 200, r.text
    # submit_job 被调, 检查 yaml 里 output_dir 未追加 job_id
    import yaml as _yaml
    flat = _yaml.safe_load(submitted[0]["yaml"])
    assert flat["output_dir"] == "saves/qwen3/lora/sft"  # 原样, 未追加


def test_no_resume_appends_job_id(client, tmp_path, monkeypatch):
    """无 resume_from_checkpoint 时, output_dir 追加 job_id (隔离)。"""
    test_client, submitted = client
    monkeypatch.setattr(datagen_job, "_BASE", tmp_path)
    jid = _make_datagen_output(tmp_path)
    params = {
        "model": {"model_name_or_path": "/x"},
        "dataset": {"template": "qwen3_5_nothink"},
        "output": {"output_dir": "saves/qwen3/lora/sft"},
    }
    r = test_client.post("/api/jobs", data={
        "params": json.dumps(params), "gpus": "", "from_datagen_job": jid})
    assert r.status_code == 200, r.text
    import yaml as _yaml
    flat = _yaml.safe_load(submitted[0]["yaml"])
    new_jid = submitted[0]["job_id"]
    assert flat["output_dir"] == f"saves/qwen3/lora/sft/{new_jid}"


# ── 产物清理 ──

def test_cleanup_endpoint_calls_remote(client, monkeypatch):
    """POST /api/jobs/{id}/cleanup 透传 checkpoint_name 到 remote.cleanup_checkpoint。"""
    test_client, _ = client
    called = {}
    monkeypatch.setattr(remote, "cleanup_checkpoint",
                        lambda c, jid, name: called.update(name=name) or
                        {"deleted": True, "detail": "DELETED"})
    r = test_client.post("/api/jobs/20260721T071950Z-c0e935/cleanup",
                         data={"checkpoint_name": "checkpoint-100"})
    assert r.status_code == 200
    assert r.json()["deleted"] is True
    assert called["name"] == "checkpoint-100"


def test_cleanup_endpoint_rejects_bad_name(client, monkeypatch):
    """非法 checkpoint 名 → 400 (不传给远端)。"""
    test_client, _ = client
    monkeypatch.setattr(remote, "cleanup_checkpoint",
                        lambda c, jid, name: (_ for _ in ()).throw(
                            remote.RemoteError("checkpoint 名必须形如 checkpoint-<数字>")))
    r = test_client.post("/api/jobs/20260721T071950Z-c0e935/cleanup",
                         data={"checkpoint_name": "../etc"})
    assert r.status_code == 400


def test_download_endpoint_rejects_bad_name(client):
    """下载 name 非 checkpoint-N → 400。"""
    test_client, _ = client
    r = test_client.get("/api/jobs/20260721T071950Z-c0e935/download?name=../etc")
    assert r.status_code == 400


# ── 模型部署 ──

from llamafactory_pipeline import deploy as deploy_mod


def test_deploy_schema_endpoint(client):
    test_client, _ = client
    r = test_client.get("/api/deploy/schema")
    assert r.status_code == 200
    names = [f["name"] for f in r.json()["fields"]]
    assert "container_name" in names and "max_model_len" in names


def test_deploy_endpoint_calls_docker_run(client, monkeypatch, tmp_path):
    """POST /api/deploy 生成 docker run 命令并远端执行。"""
    test_client, _ = client
    monkeypatch.setattr(deploy_mod, "_CFG_DIR", tmp_path)
    captured = []
    monkeypatch.setattr(remote, "run_remote_script",
                        lambda c, script, timeout=30: captured.append(script) or "DEPLOYED\n")
    r = test_client.post("/api/deploy", data={"config": json.dumps({
        "container_name": "llm2", "host_model_path": "./Qwen3.6-27B",
        "gpus": "3", "port": 8003, "api_key": "k", "max_model_len": 260000,
    })})
    assert r.status_code == 200, r.text
    assert r.json()["deployed"] is True
    assert r.json()["container_name"] == "vllm-llm2"
    script = captured[0]
    assert "docker run -d" in script
    assert "--gpus device=3" in script
    assert "vllm-llm2" in script


def test_deploy_rejects_bad_container_name(client):
    test_client, _ = client
    r = test_client.post("/api/deploy", data={"config": json.dumps({
        "container_name": "bad name!", "host_model_path": "/x"})})
    assert r.status_code == 400


def test_deploy_list_merges_configs_and_containers(client, monkeypatch, tmp_path):
    test_client, _ = client
    monkeypatch.setattr(deploy_mod, "_CFG_DIR", tmp_path)
    # 本地存一个配置, 远端 docker ps 返回一个运行中容器
    deploy_mod.save_config(deploy_schema.DeployConfig(
        container_name="saved", host_model_path="/x"))
    monkeypatch.setattr(remote, "run_remote_script",
                        lambda c, script, timeout=30:
                        "vllm-running\tUp 1 hour\t0.0.0.0:8000->8000/tcp\n"
                        if "ps -a" in script else "")
    r = test_client.get("/api/deploy")
    deps = r.json()["deployments"]
    names = [d["name"] for d in deps]
    assert "vllm-running" in names and "vllm-saved" in names  # 容器 + 本地配置
    saved = next(d for d in deps if d["name"] == "vllm-saved")
    assert saved["status"] == "missing"  # 本地有配置但容器不存在


def test_deploy_stop_endpoint(client, monkeypatch):
    test_client, _ = client
    captured = []
    monkeypatch.setattr(remote, "run_remote_script",
                        lambda c, script, timeout=30: captured.append(script) or "STOPPED\n")
    r = test_client.post("/api/deploy/vllm-llm2/stop")
    assert r.status_code == 200
    assert r.json()["stopped"] is True
    assert "docker stop" in captured[0] and "docker rm" in captured[0]


def test_deploy_delete_removes_config(client, monkeypatch, tmp_path):
    test_client, _ = client
    monkeypatch.setattr(deploy_mod, "_CFG_DIR", tmp_path)
    deploy_mod.save_config(deploy_schema.DeployConfig(
        container_name="llm2", host_model_path="/x"))
    monkeypatch.setattr(remote, "run_remote_script",
                        lambda c, script, timeout=30: "STOPPED\n")
    r = test_client.delete("/api/deploy/vllm-llm2")
    assert r.status_code == 200
    assert deploy_mod.load_config("vllm-llm2") is None  # 配置已删


# ── 健康探针 / 指标 / 对话代理 ──

def _save_deploy_config(monkeypatch, tmp_path, port=8003, api_key="k"):
    monkeypatch.setattr(deploy_mod, "_CFG_DIR", tmp_path)
    deploy_mod.save_config(deploy_schema.DeployConfig(
        container_name="llm2", host_model_path="/x", port=port, api_key=api_key))


def test_deploy_health_ready(client, monkeypatch, tmp_path):
    test_client, _ = client
    _save_deploy_config(monkeypatch, tmp_path)
    monkeypatch.setattr(remote, "run_remote_script",
                        lambda c, script, timeout=30: "200\n")
    r = test_client.get("/api/deploy/vllm-llm2/health")
    assert r.status_code == 200
    body = r.json()
    assert body["ready"] is True and body["http_code"] == "200"


def test_deploy_health_loading(client, monkeypatch, tmp_path):
    test_client, _ = client
    _save_deploy_config(monkeypatch, tmp_path)
    monkeypatch.setattr(remote, "run_remote_script",
                        lambda c, script, timeout=30: "503\n")
    r = test_client.get("/api/deploy/vllm-llm2/health")
    assert r.json()["ready"] is False
    assert "加载中" in r.json()["detail"]


def test_deploy_health_no_config(client, monkeypatch, tmp_path):
    test_client, _ = client
    monkeypatch.setattr(deploy_mod, "_CFG_DIR", tmp_path)
    r = test_client.get("/api/deploy/vllm-llm2/health")
    assert r.status_code == 400


def test_deploy_metrics(client, monkeypatch, tmp_path):
    test_client, _ = client
    _save_deploy_config(monkeypatch, tmp_path)
    sample = ('vllm:num_requests_running 2\n'
              'vllm:gpu_cache_usage_perc 0.35\n'
              'vllm:time_to_first_token_seconds_sum 25.0\n'
              'vllm:time_to_first_token_seconds_count 100\n')
    monkeypatch.setattr(remote, "run_remote_script",
                        lambda c, script, timeout=30: sample)
    r = test_client.get("/api/deploy/vllm-llm2/metrics")
    assert r.status_code == 200
    m = r.json()
    assert m["running"] == 2 and m["kv_cache"] == 0.35
    assert m["ttft"]["avg"] == 0.25


def test_deploy_metrics_failure_returns_empty(client, monkeypatch, tmp_path):
    test_client, _ = client
    _save_deploy_config(monkeypatch, tmp_path)
    def raise_err(c, script, timeout=30):
        raise remote.RemoteError("SSH fail")
    monkeypatch.setattr(remote, "run_remote_script", raise_err)
    r = test_client.get("/api/deploy/vllm-llm2/metrics")
    assert r.status_code == 200
    assert r.json() == {}


def test_deploy_chat_proxy(client, monkeypatch, tmp_path):
    """POST /chat 透传 body 到 vLLM, 流式返回。"""
    test_client, _ = client
    _save_deploy_config(monkeypatch, tmp_path)

    class FakeProc:
        def __init__(self):
            self.stdin = type("S", (), {"write": lambda s, b: None, "close": lambda s: None})()
            self._chunks = [b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n', b'data: [DONE]\n\n']
            self._i = 0
        @property
        def stdout(self):
            return self
        def read(self, n):
            if self._i >= len(self._chunks):
                return b""
            ch = self._chunks[self._i]; self._i += 1
            return ch
        def terminate(self): pass
        def wait(self, timeout=None): pass
        def kill(self): pass

    monkeypatch.setattr(deploy_mod.subprocess, "Popen", lambda *a, **kw: FakeProc())
    r = test_client.post("/api/deploy/vllm-llm2/chat",
                         json={"messages": [{"role": "user", "content": "hi"}], "stream": True})
    assert r.status_code == 200
    body = b"".join(r.iter_bytes())
    assert b"data: " in body and b"[DONE]" in body


def test_deploy_chat_no_config(client, monkeypatch, tmp_path):
    test_client, _ = client
    monkeypatch.setattr(deploy_mod, "_CFG_DIR", tmp_path)
    r = test_client.post("/api/deploy/vllm-llm2/chat", json={"messages": []})
    assert r.status_code == 400


# ── 数据管理 ──

from llamafactory_pipeline import dataset_store as ds
from llamafactory_pipeline import app as _app_mod

# client fixture 桩了 _save_upload_capped 返假路径; 数据集注册需真实落盘, 这里保存原始实现
_REAL_SAVE_UPLOAD = _app_mod._save_upload_capped


def _make_train_file(tmp_path, n=2):
    p = tmp_path / "train.json"
    p.write_text(json.dumps([
        {"conversations": [{"from": "human", "value": f"Q{i}?"},
                           {"from": "gpt", "value": f"A{i}."}]} for i in range(n)],
        ensure_ascii=False), encoding="utf-8")
    return p


def test_register_dataset_endpoint(client, monkeypatch, tmp_path):
    test_client, _ = client
    monkeypatch.setattr(_app_mod, "_save_upload_capped", _REAL_SAVE_UPLOAD)
    monkeypatch.setattr(ds, "_ROOT", tmp_path)
    p = _make_train_file(tmp_path)
    with open(p, "rb") as f:
        r = test_client.post("/api/datasets", data={"name": "qa1", "kind": "train"},
                             files={"file": ("train.json", f, "application/json")})
    assert r.status_code == 200, r.text
    meta = r.json()
    assert meta["name"] == "qa1" and meta["n_records"] == 2


def test_list_datasets_endpoint(client, monkeypatch, tmp_path):
    test_client, _ = client
    monkeypatch.setattr(_app_mod, "_save_upload_capped", _REAL_SAVE_UPLOAD)
    monkeypatch.setattr(ds, "_ROOT", tmp_path)
    p = _make_train_file(tmp_path)
    with open(p, "rb") as f:
        test_client.post("/api/datasets", data={"name": "qa1", "kind": "train"},
                         files={"file": ("train.json", f, "application/json")})
    r = test_client.get("/api/datasets?kind=train")
    assert r.status_code == 200
    names = [d["name"] for d in r.json()["items"]]
    assert "qa1" in names


def test_delete_dataset_endpoint(client, monkeypatch, tmp_path):
    test_client, _ = client
    monkeypatch.setattr(_app_mod, "_save_upload_capped", _REAL_SAVE_UPLOAD)
    monkeypatch.setattr(ds, "_ROOT", tmp_path)
    p = _make_train_file(tmp_path)
    with open(p, "rb") as f:
        test_client.post("/api/datasets", data={"name": "qa1", "kind": "train"},
                         files={"file": ("train.json", f, "application/json")})
    r = test_client.delete("/api/datasets/qa1?kind=train")
    assert r.status_code == 200
    assert r.json()["deleted"] is True


def test_create_job_from_dataset_name(client, monkeypatch, tmp_path):
    """训练时用 dataset_name 引用本地注册数据集, 跳过上传。"""
    test_client, submitted = client
    monkeypatch.setattr(_app_mod, "_save_upload_capped", _REAL_SAVE_UPLOAD)
    monkeypatch.setattr(ds, "_ROOT", tmp_path)
    # 注册一个训练集
    p = _make_train_file(tmp_path)
    with open(p, "rb") as f:
        test_client.post("/api/datasets", data={"name": "qa1", "kind": "train"},
                         files={"file": ("train.json", f, "application/json")})
    r = test_client.post("/api/jobs", data={
        "params": json.dumps({"model": {"model_name_or_path": "/x"},
                              "dataset": {"template": "qwen3_5_nothink"}}),
        "gpus": "", "dataset_name": "qa1"})
    assert r.status_code == 200, r.text
    assert r.json()["format"] == "conversations"
    # submit_job 数据路径指向本地注册的数据文件
    assert submitted[0]["data_path"].endswith("qa1.data")


def test_create_job_dataset_not_found(client, monkeypatch, tmp_path):
    test_client, _ = client
    monkeypatch.setattr(_app_mod, "_save_upload_capped", _REAL_SAVE_UPLOAD)
    monkeypatch.setattr(ds, "_ROOT", tmp_path)
    r = test_client.post("/api/jobs", data={
        "params": json.dumps({}), "dataset_name": "nope"})
    assert r.status_code == 404
