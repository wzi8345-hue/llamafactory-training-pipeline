"""评测子系统纯逻辑自检 (不依赖真实服务器/GPU/LLM)。

运行: 从仓库根 `python -m pytest llamafactory_pipeline/test_eval.py -q`
"""

from __future__ import annotations

import json
import subprocess
import tempfile

import pytest

from llamafactory_pipeline import eval_judge, eval_remote, eval_schema
from llamafactory_pipeline import remote


# ── eval_schema 校验 ──

def test_validate_fc_item_ok():
    it = {"id": "a", "query": "q", "tools": [{"function": {"name": "f"}}],
          "gold": {"name": "f", "arguments": {"x": 1}}}
    assert eval_schema.validate_fc_item(it) is it


@pytest.mark.parametrize("bad", [
    {"query": "q", "tools": [{"function": {"name": "f"}}], "gold": {"name": "f"}},   # 无 id
    {"id": "a", "tools": [{"function": {"name": "f"}}], "gold": {"name": "f"}},        # 无 query
    {"id": "a", "query": "q", "tools": [], "gold": {"name": "f"}},                     # tools 空
    {"id": "a", "query": "q", "tools": [{"function": {}}], "gold": {"name": "f"}},     # tool 无 name
    {"id": "a", "query": "q", "tools": [{"function": {"name": "f"}}], "gold": {}},     # gold 无 name
])
def test_validate_fc_item_bad(bad):
    with pytest.raises(ValueError):
        eval_schema.validate_fc_item(bad)


def test_validate_subjective_item():
    assert eval_schema.validate_subjective_item({"id": "s", "query": "q"})
    with pytest.raises(ValueError):
        eval_schema.validate_subjective_item({"id": "s"})
    with pytest.raises(ValueError):
        eval_schema.validate_subjective_item({"id": "s", "query": "q", "reference": 5})


def test_eval_item_tags_allow_strict_multi_value_source_documents():
    tagged = {"id": "s", "query": "q", "tags": {"slice": "critical"}}
    assert eval_schema.validate_subjective_item(tagged) is tagged
    multi = {
        "id": "m",
        "query": "q",
        "tags": {"source_doc": ["doc-a", "doc-b"]},
    }
    assert eval_schema.validate_subjective_item(multi) is multi
    for bad in (
        {"slice": 1},
        {"source_doc": []},
        {"source_doc": ["doc-a", ""]},
        {"source_doc": ["doc-a", 2]},
    ):
        with pytest.raises(ValueError, match="tags"):
            eval_schema.validate_subjective_item(
                {"id": "s", "query": "q", "tags": bad}
            )


def test_validate_evalset_empty():
    with pytest.raises(ValueError):
        eval_schema.validate_evalset([], "subjective")


def test_eval_request_names():
    m = {"name": "a", "model_name_or_path": "/x"}
    req = eval_schema.EvalRequest(models=[m], task_types=["subjective"])
    req.validate_names()
    with pytest.raises(ValueError):
        eval_schema.EvalRequest(
            models=[m, {"name": "a", "model_name_or_path": "/y"}],
            task_types=["subjective"],
        ).validate_names()
    with pytest.raises(ValueError):
        eval_schema.EvalRequest(
            models=[{"name": "bad name!", "model_name_or_path": "/y"}],
            task_types=["subjective"],
        ).validate_names()


# ── eval_judge 纯逻辑 ──

def test_tool_name_correct():
    assert eval_judge.tool_name_correct("f", "f")
    assert not eval_judge.tool_name_correct("f", "g")
    assert not eval_judge.tool_name_correct("f", None)


def test_parse_judge_score():
    assert eval_judge.parse_judge_score('{"score": 4, "reason": "ok"}')[0] == 4
    assert eval_judge.parse_judge_score('```json\n{"score": 5, "reason": "x"}\n```')[0] == 5
    assert eval_judge.parse_judge_score("not json")[0] is None
    assert eval_judge.parse_judge_score('{"score": 9}')[0] is None  # 越界


def test_score_model_and_summary():
    judge = lambda system, user: '{"score": 4, "reason": "ok"}'  # noqa: E731
    fc_items = [
        {"id": "a", "query": "q", "tools": [{"function": {"name": "f"}}],
         "gold": {"name": "f", "arguments": {"x": 1}}},
        {"id": "b", "query": "q", "tools": [{"function": {"name": "g"}}],
         "gold": {"name": "g", "arguments": {}}},
    ]
    subj_items = [{"id": "s", "query": "q", "reference": "r"}]
    preds = [
        {"id": "a", "task_type": "function_call", "pred_name": "f", "pred_arguments": {"x": 1}},
        {"id": "s", "task_type": "subjective", "answer": "hi"},
        # b 缺失预测
    ]
    scores, summary = eval_judge.score_model(
        judge, fc_items, subj_items, preds, "fcp", "subjp")
    fc = summary["function_call"]
    assert fc["n"] == 2
    assert fc["tool_name_accuracy"] == 0.5     # a 命中, b 缺失
    assert fc["param_score_mean"] == 4.0       # 仅 a 有效
    assert fc["param_score"] == 4.0
    assert fc["combined_accuracy"] == 0.5
    assert fc["invalid"] == 1                  # b
    assert summary["subjective"]["score_mean"] == 4.0
    assert summary["subjective"]["answer_accuracy"] == 0.8


def test_summary_includes_invalid_rate_and_latency_percentiles():
    rows = [
        {
            "id": "1",
            "task_type": "function_call",
            "tool_name_correct": True,
            "param_score": 5,
            "invalid": False,
            "latency_ms": 100,
        },
        {
            "id": "2",
            "task_type": "function_call",
            "tool_name_correct": False,
            "param_score": None,
            "invalid": True,
            "latency_ms": 300,
            "invalid_reason": "no_tool_call",
            "finish_reason": "length",
            "param_reason": "judge output was not valid JSON",
        },
    ]
    summary = eval_judge.summarize(rows)["function_call"]
    assert summary["invalid_rate"] == 0.5
    assert summary["no_tool_call_rate"] == 0.5
    assert summary["latency_p50_ms"] == 200.0
    assert summary["latency_p95_ms"] == 290.0
    assert summary["invalid_reason_counts"] == {"no_tool_call": 1}
    assert summary["finish_reason_counts"] == {"length": 1}
    assert summary["judge_failure_reason_counts"] == {
        "judge output was not valid JSON": 1
    }
    assert summary["failure_examples"] == [
        {
            "id": "2",
            "invalid_reason": "no_tool_call",
            "finish_reason": "length",
            "judge_reason": "judge output was not valid JSON",
        }
    ]


def test_summary_preserves_valid_low_quality_reasons_for_iteration_planning():
    rows = [
        {
            "id": "fc-low",
            "task_type": "function_call",
            "tool_name_correct": False,
            "param_score": 2,
            "param_reason": "wrong tool and missing required date",
            "invalid": False,
        },
        {
            "id": "qa-low",
            "task_type": "subjective",
            "score": 2,
            "reason": "answer omitted the cited constraint",
            "invalid": False,
        },
    ]

    summary = eval_judge.summarize(rows)

    assert summary["function_call"]["quality_issue_counts"] == {
        "param_score_below_4": 1,
        "tool_name_incorrect": 1,
    }
    assert summary["function_call"]["quality_examples"][0]["judge_reason"] == (
        "wrong tool and missing required date"
    )
    assert summary["subjective"]["quality_issue_counts"] == {
        "score_below_4": 1
    }
    assert summary["subjective"]["quality_examples"][0]["id"] == "qa-low"


def test_build_report_md():
    per_model = {
        "m1": {"function_call": {"n": 2, "tool_name_accuracy": 0.5,
                                 "param_score_mean": 4.0, "invalid": 1}},
        "m2": {"subjective": {"n": 1, "score_mean": 3.0, "invalid": 0}},
    }
    md = eval_judge.build_report_md("run1", per_model)
    assert "m1" in md and "m2" in md
    assert "0.500" in md and "4.000" in md


def test_report_includes_aggregated_failure_reasons():
    md = eval_judge.build_report_md(
        "run1",
        {
            "m1": {
                "function_call": {
                    "n": 2,
                    "tool_name_accuracy": 0.5,
                    "param_score_mean": 4.0,
                    "invalid": 1,
                    "invalid_reason_counts": {"no_tool_call": 1},
                    "finish_reason_counts": {"length": 1},
                    "judge_failure_reason_counts": {"invalid JSON": 1},
                    "quality_issue_counts": {"tool_name_incorrect": 1},
                    "quality_reason_counts": {"wrong tool": 1},
                    "quality_examples": [
                        {
                            "id": "case-7",
                            "issues": ["tool_name_incorrect"],
                            "judge_reason": "wrong tool",
                        }
                    ],
                }
            }
        },
    )
    assert "no_tool_call=1" in md
    assert "length=1" in md
    assert "invalid JSON=1" in md
    assert "tool_name_incorrect=1" in md
    assert "wrong tool=1" in md
    assert "case-7" in md


def test_report_includes_full_paired_acceptance_evidence():
    per_model = {
        "baseline": {
            "subjective": {
                "n": 30,
                "score_mean": 3.0,
                "invalid": 3,
                "invalid_rate": 0.1,
                "latency_p50_ms": 100.0,
                "latency_p95_ms": 250.0,
            }
        },
        "candidate": {
            "subjective": {
                "n": 30,
                "score_mean": 3.5,
                "invalid": 1,
                "invalid_rate": 0.033333,
                "latency_p50_ms": 110.0,
                "latency_p95_ms": 300.0,
            }
        },
    }
    paired = {
        "n": 31,
        "paired_score_n": 30,
        "baseline_missing": 1,
        "candidate_missing": 0,
        "wins": 12,
        "ties": 10,
        "losses": 8,
        "mean_delta": 0.5,
        "bootstrap_low": 0.1,
        "bootstrap_high": 0.9,
        "mcnemar_p": 0.03125,
        "slices": {
            "qa": {
                "n": 30,
                "baseline_correct_rate": 0.7,
                "candidate_correct_rate": 0.8,
                "rate_regression": 0.0,
                "mean_score_delta": 0.5,
                "score_regression": 0.0,
            }
        },
        "missing_critical_slices": ["safety"],
    }

    md = eval_judge.build_report_md("run1", per_model, paired)

    for expected in (
        "10.000%",
        "100.000",
        "250.000",
        "12 / 10 / 8",
        "[0.100, 0.900]",
        "0.031250",
        "baseline missing=1",
        "business scores=30/31",
        "qa",
        "safety",
    ):
        assert expected in md


def test_scoring_persists_each_judged_row_and_resumes(tmp_path, monkeypatch):
    req = eval_schema.EvalRequest(
        models=[eval_schema.ModelUnderTest(name="m1", model_name_or_path="/m")],
        task_types=["function_call"],
    )
    items = [
        {
            "id": item_id,
            "query": "q",
            "tools": [{"function": {"name": "f"}}],
            "gold": {"name": "f", "arguments": {}},
        }
        for item_id in ("1", "2")
    ]
    predictions = [
        {
            "id": item_id,
            "task_type": "function_call",
            "pred_name": "f",
            "pred_arguments": {},
        }
        for item_id in ("1", "2")
    ]

    def fetch(cfg, eval_id, model_name, local_path):
        with open(local_path, "w", encoding="utf-8") as handle:
            for row in predictions:
                handle.write(json.dumps(row) + "\n")

    monkeypatch.setattr(eval_remote, "fetch_predictions", fetch)
    calls = []

    def flaky_judge(system, user):
        calls.append(user)
        if len(calls) == 2:
            raise RuntimeError("judge unavailable")
        return '{"score":4,"reason":"ok"}'

    with pytest.raises(RuntimeError, match="judge unavailable"):
        eval_judge.run_scoring(
            _cfg(), "20240101T000000Z-abcdef", req, items, [], str(tmp_path), flaky_judge
        )
    score_path = tmp_path / "m1.scores.jsonl"
    assert len(score_path.read_text("utf-8").splitlines()) == 1
    with score_path.open("a", encoding="utf-8") as handle:
        handle.write('{"id":"partial"')

    resumed_calls = []
    eval_judge.run_scoring(
        _cfg(),
        "20240101T000000Z-abcdef",
        req,
        items,
        [],
        str(tmp_path),
        lambda system, user: resumed_calls.append(user)
        or '{"score":4,"reason":"ok"}',
    )
    assert len(resumed_calls) == 1
    assert len(score_path.read_text("utf-8").splitlines()) == 2


def test_scoring_retries_judge_invalid_row_without_duplicate_summary(tmp_path, monkeypatch):
    req = eval_schema.EvalRequest(
        models=[eval_schema.ModelUnderTest(name="m1", model_name_or_path="/m")],
        task_types=["subjective"],
    )
    items = [{"id": "1", "query": "q", "reference": "a"}]

    def fetch(cfg, eval_id, model_name, local_path):
        with open(local_path, "w", encoding="utf-8") as handle:
            handle.write(json.dumps({"id": "1", "answer": "a"}) + "\n")

    monkeypatch.setattr(eval_remote, "fetch_predictions", fetch)
    with pytest.raises(eval_judge.RetryableJudgeEvidenceError):
        eval_judge.run_scoring(
            _cfg(), "20240101T000000Z-abcdef", req, [], items, str(tmp_path),
            lambda system, user: "not-json",
        )
    first_rows = eval_judge._load_jsonl(str(tmp_path / "m1.scores.jsonl"))
    assert len(first_rows) == 1
    assert first_rows[0]["invalid"] is True

    second = eval_judge.run_scoring(
        _cfg(), "20240101T000000Z-abcdef", req, [], items, str(tmp_path),
        lambda system, user: '{"score":5,"reason":"ok"}',
    )
    rows = eval_judge._load_jsonl(str(tmp_path / "m1.scores.jsonl"))
    assert len(rows) == 1
    assert rows[0]["score"] == 5
    assert second["per_model"]["m1"]["subjective"]["n"] == 1


def test_malformed_cached_fc_score_is_rejudged():
    malformed = {
        "id": "1",
        "task_type": "function_call",
        "tool_name_correct": True,
    }

    assert eval_judge._normalized_existing_scores([malformed]) == []


def test_out_of_range_and_non_object_cached_scores_are_rejudged():
    out_of_range = {
        "id": "1",
        "task_type": "subjective",
        "score": 999.0,
        "invalid": False,
    }

    assert eval_judge._normalized_existing_scores(
        [out_of_range, ["not", "an", "object"], 7]
    ) == []
    assert eval_judge._normalized_existing_scores(
        [
            {
                "id": "2",
                "task_type": "subjective",
                "score": None,
                "invalid": True,
            }
        ]
    ) == []


@pytest.mark.parametrize(
    "bad_evidence",
    [
        {"tags": "corrupt"},
        {"tags": {"source_doc": []}},
        {"latency_ms": float("nan")},
        {"latency_ms": -1},
        {"finish_reason": {"unexpected": "object"}},
    ],
)
def test_cached_scores_with_malformed_optional_evidence_are_rejudged(bad_evidence):
    row = {
        "id": "1",
        "task_type": "subjective",
        "score": 5,
        "reason": "ok",
        "invalid": False,
        **bad_evidence,
    }

    assert eval_judge._normalized_existing_scores([row]) == []


def test_scoring_does_not_rewrite_immutable_request_config(tmp_path, monkeypatch):
    req = eval_schema.EvalRequest(
        models=[eval_schema.ModelUnderTest(name="m1", model_name_or_path="/m")],
        task_types=["subjective"],
    )
    item = {"id": "1", "query": "q", "reference": "a"}
    config_path = tmp_path / "config.json"
    immutable = req.model_dump_json()
    config_path.write_text(immutable, encoding="utf-8")

    def fetch(cfg, eval_id, model_name, local_path):
        with open(local_path, "w", encoding="utf-8") as handle:
            handle.write(json.dumps({"id": "1", "answer": "a"}) + "\n")

    monkeypatch.setattr(eval_remote, "fetch_predictions", fetch)
    eval_judge.run_scoring(
        _cfg(),
        "20240101T000000Z-abcdef",
        req,
        [],
        [item],
        str(tmp_path),
        lambda system, user: '{"score":5,"reason":"ok"}',
    )

    assert config_path.read_text("utf-8") == immutable


# ── eval_remote 生成物 ──

def _cfg(docker=None):
    return remote.RemoteConfig(
        ssh_target="user@host", remote_root="/root/runs",
        llamafactory_dir="/data/LLaMA-Factory", docker_container=docker)


def test_build_infer_yaml_adapter():
    m_base = eval_schema.ModelUnderTest(name="b", model_name_or_path="/x")
    assert "adapter_name_or_path" not in eval_remote.build_infer_yaml(m_base)
    m_ft = eval_schema.ModelUnderTest(name="f", model_name_or_path="/x", adapter_path="/out")
    assert "adapter_name_or_path" in eval_remote.build_infer_yaml(m_ft)


def test_read_train_job_resolves_relative_output_to_absolute_path(monkeypatch):
    monkeypatch.setattr(
        remote,
        "run_remote_script",
        lambda *args, **kwargs: (
            "model_name_or_path: /models/qwen\n"
            "output_dir: saves/run\n"
            "template: qwen3_5_nothink\n"
        ),
    )
    resolved = eval_remote.read_train_job(
        _cfg(), "20240101T000000Z-abcdef"
    )
    assert resolved["adapter_path"] == "/data/LLaMA-Factory/saves/run"


def _req(gpus=""):
    return eval_schema.EvalRequest(
        models=[eval_schema.ModelUnderTest(name="m1", model_name_or_path="/x")],
        task_types=["function_call", "subjective"], gpus=gpus)


def test_run_eval_script_sh_valid():
    script = eval_remote.build_run_eval_script(_cfg(), "20240101T000000Z-abcdef",
                                               _req(), has_fc=True, has_subj=True)
    assert "m1" in script and "llamafactory-cli api" in script
    with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False) as f:
        f.write(script); path = f.name
    r = subprocess.run(["sh", "-n", path], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_run_eval_script_binds_runner_to_the_spawned_api_identity():
    script = eval_remote.build_run_eval_script(
        _cfg(),
        "20240101T000000Z-abcdef",
        _req(),
        has_fc=True,
        has_subj=True,
    )

    assert "port already in use" in script
    assert "eval-port-8000.lock" in script
    assert "flock -n 8" in script
    assert "--api-pid $API_PID" in script
    service_id = eval_remote.evaluation_service_id(
        "20240101T000000Z-abcdef", _req().models
    )
    assert f"API_MODEL_NAME={service_id}" in script
    assert f"--expected-model {service_id}" in script
    assert "if ! python" in script
    assert "exit 21" in script


def test_runner_rejects_an_unrelated_model_endpoint():
    namespace = {"__name__": "runner_test"}
    exec(eval_remote.RUNNER_PY, namespace)
    namespace["_get"] = lambda *args, **kwargs: {
        "data": [{"id": "unrelated-model"}]
    }
    namespace["os"].kill = lambda pid, signal: None

    with pytest.raises(RuntimeError, match="identity mismatch"):
        namespace["wait_ready"](
            "http://127.0.0.1:8000/v1", 1, 123, "/models/expected"
        )


def test_runner_rejects_same_basename_from_a_different_model_path():
    namespace = {"__name__": "runner_test"}
    exec(eval_remote.RUNNER_PY, namespace)
    namespace["_get"] = lambda *args, **kwargs: {
        "data": [{"id": "/wrong/location/same-model"}]
    }
    namespace["os"].kill = lambda pid, signal: None

    with pytest.raises(RuntimeError, match="identity mismatch"):
        namespace["wait_ready"](
            "http://127.0.0.1:8000/v1",
            1,
            123,
            "/approved/location/same-model",
        )


def test_runner_rejects_expected_model_from_unrelated_listener():
    namespace = {"__name__": "runner_test"}
    exec(eval_remote.RUNNER_PY, namespace)
    namespace["_get"] = lambda *args, **kwargs: {
        "data": [{"id": "/approved/model"}]
    }
    namespace["_listener_owned_by_process_tree"] = lambda *args: False
    namespace["os"].kill = lambda pid, signal: None

    with pytest.raises(RuntimeError, match="listener is not owned"):
        namespace["wait_ready"](
            "http://127.0.0.1:8000/v1", 1, 123, "/approved/model"
        )


# ── 多 adapter 合并 ──

def _ft(name, base="/base", adapter="/out", template="qwen3_5_nothink"):
    return eval_schema.ModelUnderTest(
        name=name,
        model_name_or_path=base,
        adapter_path=adapter,
        template=template,
    )


def test_evaluation_service_id_is_deterministic_and_request_bound():
    eval_id = "20240101T000000Z-abcdef"
    group = [_ft("a1", adapter="/o1"), _ft("a2", adapter="/o2")]
    first = eval_remote.evaluation_service_id(eval_id, group)
    assert first == eval_remote.evaluation_service_id(eval_id, group)
    assert first != eval_remote.evaluation_service_id(
        eval_id, [_ft("a1", adapter="/different")]
    )


def test_group_models_merges_same_base():
    """同基座多 adapter 合并一组; 不同基座分开; 纯基座单独。"""
    ms = [
        _ft("a1", adapter="/o1"), _ft("a2", adapter="/o2"),  # 同基座
        _ft("b1", base="/other", adapter="/o3"),             # 不同基座
        eval_schema.ModelUnderTest(name="raw", model_name_or_path="/base"),  # 纯基座
    ]
    groups = eval_remote.group_models(ms)
    # 同基座 a1,a2 合并; b1 单独; raw 单独
    gnames = [sorted(m.name for m in g) for g in groups]
    assert ["a1", "a2"] in gnames
    assert ["b1"] in gnames
    assert ["raw"] in gnames


def test_group_models_never_merges_different_templates():
    groups = eval_remote.group_models([
        _ft("a1", adapter="/o1", template="qwen3_5_nothink"),
        _ft("a2", adapter="/o2", template="chatml"),
    ])
    assert [sorted(model.name for model in group) for group in groups] == [
        ["a1"],
        ["a2"],
    ]


def test_group_infer_yaml_merges_adapters():
    grp = [_ft("a1", adapter="/o1"), _ft("a2", adapter="/o2")]
    y = eval_remote.build_group_infer_yaml(grp)
    import yaml as _yaml
    conf = _yaml.safe_load(y)
    assert conf["adapter_name_or_path"] == "/o1,/o2"
    assert conf["model_name_or_path"] == "/base"


def test_eval_merges_same_base_adapters():
    """同基座两 adapter → 脚本只起一次 llamafactory-cli api。"""
    req = eval_schema.EvalRequest(
        models=[_ft("a1", adapter="/o1"), _ft("a2", adapter="/o2")],
        task_types=["subjective"])
    script = eval_remote.build_run_eval_script(_cfg(), "20240101T000000Z-abcdef",
                                               req, has_fc=False, has_subj=True)
    # 只一次冷启动
    assert script.count("llamafactory-cli api") == 1
    # 两个模型各自的 runner 调用, 带 --adapter
    assert script.count("--adapter") == 2
    assert "a1" in script and "a2" in script
    # sh 语法校验
    with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False) as f:
        f.write(script); path = f.name
    r = subprocess.run(["sh", "-n", path], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_eval_fallback_serial_for_diff_base():
    """不同基座 → 各自起 api (不合并)。"""
    req = eval_schema.EvalRequest(
        models=[_ft("a1", base="/b1", adapter="/o1"),
                _ft("a2", base="/b2", adapter="/o2")],
        task_types=["subjective"])
    script = eval_remote.build_run_eval_script(_cfg(), "20240101T000000Z-abcdef",
                                               req, has_fc=False, has_subj=True)
    assert script.count("llamafactory-cli api") == 2  # 各起一次


def test_eval_pure_base_no_adapter_arg():
    """纯基座模型 runner 不带 --adapter。"""
    req = eval_schema.EvalRequest(
        models=[eval_schema.ModelUnderTest(name="raw", model_name_or_path="/b")],
        task_types=["subjective"])
    script = eval_remote.build_run_eval_script(_cfg(), "20240101T000000Z-abcdef",
                                               req, has_fc=False, has_subj=True)
    assert "--adapter" not in script


def test_runner_py_has_adapter_arg():
    """runner 脚本支持 --adapter 参数。"""
    assert "--adapter" in eval_remote.RUNNER_PY
    assert "adapter_name" in eval_remote.RUNNER_PY  # 注入请求 body


def test_runner_py_compiles():
    compile(eval_remote.RUNNER_PY, "runner.py", "exec")


def test_runner_records_latency_finish_reason_and_invalid_reason(tmp_path, monkeypatch):
    namespace = {"__name__": "runner_test"}
    exec(compile(eval_remote.RUNNER_PY, "runner.py", "exec"), namespace)
    monkeypatch.setattr(
        namespace["time"], "perf_counter", iter((1.0, 1.125)).__next__
    )
    namespace["_post"] = lambda *args, **kwargs: {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"content": "direct answer", "tool_calls": []},
            }
        ]
    }
    output = tmp_path / "predictions.jsonl"
    namespace["run_fc"](
        "http://api",
        "model",
        [{"id": "1", "query": "Q", "tools": []}],
        str(output),
    )
    row = json.loads(output.read_text("utf-8"))
    assert row["latency_ms"] == 125.0
    assert row["finish_reason"] == "stop"
    assert row["invalid_reason"] == "no_tool_call"


def test_launch_docker_and_host():
    eid = "20240101T000000Z-abcdef"
    host = eval_remote.build_eval_launch(_cfg(), eid, "0")
    assert "nohup" in host and "CUDA_VISIBLE_DEVICES=0" in host
    assert "launch_identity" in host
    assert "echo RECOVERED" in host
    assert "flock -w 20 9" in host
    assert "echo STARTED" in host
    assert "pid_starttime" in host
    dock = eval_remote.build_eval_launch(_cfg(docker="lf"), eid, "0,1")
    assert "docker exec" in dock and "CUDA_VISIBLE_DEVICES=0,1" in dock


def test_eval_launch_ack_requires_complete_process_identity():
    script = eval_remote.build_eval_launch(
        _cfg(), "20240101T000000Z-abcdef", "0"
    )
    identity_write = script.index("launch_identity.tmp")
    status_write = script.index("status.tmp")
    eval_start = script.index("run_eval.sh")

    assert identity_write < status_write < eval_start
    assert 'if [ -e "$D/pid" ] || [ -e "$D/pid_starttime" ]' in script
    assert "echo NOT_READY" in script


def test_launch_rejects_bad_gpus():
    with pytest.raises(remote.RemoteError):
        eval_remote.build_eval_launch(_cfg(), "20240101T000000Z-abcdef", "0;rm -rf /")


def test_eval_replay_with_matching_manifest_never_rewrites_inputs(
    tmp_path, monkeypatch
):
    subjective = tmp_path / "subjective.jsonl"
    subjective.write_text('{"id":"1","query":"q"}\n', encoding="utf-8")
    monkeypatch.setattr(remote, "submission_state", lambda *args: "SAME")
    monkeypatch.setattr(
        remote, "_write_remote_file", lambda *args: pytest.fail("rewrite")
    )
    monkeypatch.setattr(remote, "_scp", lambda *args: pytest.fail("rewrite"))
    launched = []
    monkeypatch.setattr(
        remote,
        "run_remote_script",
        lambda cfg, script: launched.append(script) or "ALREADY\n",
    )

    eval_remote.submit_eval(
        _cfg(),
        "20240101T000000Z-abcdef",
        eval_schema.EvalRequest(
            models=[eval_schema.ModelUnderTest(name="m", model_name_or_path="/m")],
            task_types=["subjective"],
        ),
        None,
        str(subjective),
    )

    assert len(launched) == 1
    assert "launch.lock" in launched[0]


def test_eval_submit_requires_durable_launch_ack(tmp_path, monkeypatch):
    subjective = tmp_path / "subjective.jsonl"
    subjective.write_text('{"id":"1","query":"q"}\n', encoding="utf-8")
    monkeypatch.setattr(remote, "submission_state", lambda *args: "SAME")
    monkeypatch.setattr(
        remote, "run_remote_script", lambda *args, **kwargs: "BUSY\n"
    )

    with pytest.raises(remote.RemoteError, match="launch acknowledgement"):
        eval_remote.submit_eval(
            _cfg(),
            "20240101T000000Z-abcdef",
            eval_schema.EvalRequest(
                models=[
                    eval_schema.ModelUnderTest(
                        name="m", model_name_or_path="/m"
                    )
                ],
                task_types=["subjective"],
            ),
            None,
            str(subjective),
        )
