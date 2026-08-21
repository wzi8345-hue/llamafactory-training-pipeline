"""Personal training-assistant dataset freezing and profiling tests."""

from __future__ import annotations

import json

import pytest

from .assistant_data import (
    estimate_tokens,
    holdout_count,
    prepare_generated_datasets,
)


def _write_json(path, records) -> None:
    path.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")


def _read_jsonl(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text("utf-8").splitlines() if line]


def _qa_records(count: int) -> list[dict]:
    return [
        {
            "conversations": [
                {"from": "human", "value": f"问题 {i}"},
                {"from": "gpt", "value": f"答案 {i}"},
            ]
        }
        for i in range(count)
    ]


def _tools() -> list[dict]:
    return [
        {
            "type": "function",
            "function": {"name": "plan", "parameters": {"type": "object"}},
        }
    ]


def test_token_estimator_and_holdout_boundaries():
    assert estimate_tokens("耐候钢 ABCD") == 4
    assert holdout_count(10, 0.1) == 2
    assert holdout_count(20, 0.1) == 4
    assert holdout_count(29, 0.1) == 6
    assert holdout_count(31, 0.1) == 20
    assert holdout_count(40, 0.1) == 20
    assert holdout_count(199, 0.1) == 20
    assert holdout_count(200, 0.1) == 20
    assert holdout_count(200, 0.2) == 40
    with pytest.raises(ValueError, match="2 records"):
        holdout_count(1, 0.1)


def test_sft_qa_holdout_becomes_subjective_eval(tmp_path):
    source = tmp_path / "qa.json"
    _write_json(source, _qa_records(40))

    prepared = prepare_generated_datasets(
        [source], "wf_test", 0, 0.1, 42, tmp_path / "out",
        validation_ratio=0.1, register=False,
    )

    train = json.loads(prepared.train_path.read_text("utf-8"))
    eval_rows = _read_jsonl(prepared.eval_paths["subjective"])
    assert len(train) == 20
    assert len(eval_rows) == 20
    assert set(eval_rows[0]) == {"id", "query", "reference", "tags"}
    assert eval_rows[0]["tags"]["task_type"] == "qa"
    assert eval_rows[0]["id"].startswith("wf_wf_test_it0_")
    assert prepared.profile.n_records == 20
    assert prepared.profile.holdout_records == 20
    assert prepared.profile.actual_holdout_ratio == 0.5
    assert prepared.profile.slice_counts["qa"] == 20
    assert prepared.profile.slice_counts["task_type=qa"] == 20
    assert prepared.profile.task_types == ["qa"]


def test_multi_value_source_documents_are_each_counted_in_frozen_holdout(tmp_path):
    records = _qa_records(40)
    for record in records:
        record["tags"] = {
            "task_type": "qa_multi",
            "source_doc": ["manual-a.pdf", "manual-b.pdf"],
        }
    source = tmp_path / "qa_multi.json"
    _write_json(source, records)

    prepared = prepare_generated_datasets(
        [source], "wf_multi", 0, 0.1, 42, tmp_path / "out",
        validation_ratio=0.1, register=False,
    )

    assert prepared.profile.slice_counts["source_doc=manual-a.pdf"] == 20
    assert prepared.profile.slice_counts["source_doc=manual-b.pdf"] == 20


def test_dpo_fc_holdout_preserves_tools_and_gold(tmp_path):
    tools = _tools()
    records = [
        {
            "conversations": [{"from": "human", "value": f"规划 {i}"}],
            "chosen": {
                "from": "function_call",
                "value": '{"name":"plan","arguments":{}}',
            },
            "rejected": {"from": "gpt", "value": "不需要"},
            "tools": json.dumps(tools, ensure_ascii=False),
            "tags": {"slice": "critical", "task_type": "qa"},
        }
        for i in range(40)
    ]
    source = tmp_path / "fc.json"
    _write_json(source, records)

    prepared = prepare_generated_datasets(
        [source], "wf_fc", 0, 0.1, 42, tmp_path / "out",
        validation_ratio=0.1, register=False,
    )

    eval_rows = _read_jsonl(prepared.eval_paths["function_call"])
    assert eval_rows[0]["gold"] == {"name": "plan", "arguments": {}}
    assert eval_rows[0]["tools"][0]["function"]["name"] == "plan"
    assert eval_rows[0]["tags"] == {"task_type": "fc", "slice": "critical"}
    assert prepared.profile.finetune_type == "dpo"
    assert prepared.profile.label_counts["plan"] == 20


def test_sft_fc_accepts_string_arguments_and_rejects_unknown_tool(tmp_path):
    tools = _tools()
    records = [
        {
            "conversations": [
                {"from": "human", "value": f"规划 {i}"},
                {
                    "from": "gpt",
                    "value": "",
                    "tool_calls": [
                        {"name": "plan", "arguments": json.dumps({"id": i})}
                    ],
                },
            ],
            "tools": tools,
        }
        for i in range(39)
    ]
    records.append(
        {
            "conversations": [
                {"from": "human", "value": "未知工具"},
                {
                    "from": "gpt",
                    "value": "",
                    "tool_calls": [{"name": "missing", "arguments": {}}],
                },
            ],
            "tools": tools,
        }
    )
    source = tmp_path / "fc.json"
    _write_json(source, records)

    prepared = prepare_generated_datasets(
        [source], "wf_invalid", 1, 0.1, 7, tmp_path / "out",
        validation_ratio=0.0, register=False,
    )

    train = json.loads(prepared.train_path.read_text("utf-8"))
    eval_rows = _read_jsonl(prepared.eval_paths["function_call"])
    assert len(train) + len(eval_rows) == 39
    assert all(row["gold"]["name"] == "plan" for row in eval_rows)
    assert prepared.profile.invalid_tool_call_count == 1


def test_split_is_reproducible(tmp_path):
    source = tmp_path / "data.json"
    _write_json(source, _qa_records(200))

    one = prepare_generated_datasets(
        [source], "wf", 0, 0.1, 42, tmp_path / "one",
        validation_ratio=0.1, register=False,
    )
    two = prepare_generated_datasets(
        [source], "wf", 0, 0.1, 42, tmp_path / "two",
        validation_ratio=0.1, register=False,
    )

    assert one.profile.sha256 == two.profile.sha256
    assert (
        one.eval_paths["subjective"].read_bytes()
        == two.eval_paths["subjective"].read_bytes()
    )


def test_mixed_qa_fc_train_file_starts_with_tools_record(tmp_path):
    tools = _tools()
    qa = _qa_records(40)
    fc = [
        {
            "conversations": [
                {"from": "human", "value": f"规划 {i}"},
                {
                    "from": "gpt",
                    "value": "",
                    "tool_calls": [{"name": "plan", "arguments": {}}],
                },
            ],
            "tools": tools,
        }
        for i in range(40)
    ]
    qa_path, fc_path = tmp_path / "qa.json", tmp_path / "fc.json"
    _write_json(qa_path, qa)
    _write_json(fc_path, fc)

    prepared = prepare_generated_datasets(
        [qa_path, fc_path], "wf/mix has spaces", 0, 0.1, 42, tmp_path / "out",
        validation_ratio=0.1, register=False,
    )

    train = json.loads(prepared.train_path.read_text("utf-8"))
    assert train[0].get("tools")
    assert prepared.dataset_name.startswith("assistant_wf_mix_has_spaces")
    assert set(prepared.profile.task_types) == {"qa", "fc"}
    assert set(prepared.eval_paths) == {"subjective", "function_call"}
    subjective = _read_jsonl(prepared.eval_paths["subjective"])
    function_call = _read_jsonl(prepared.eval_paths["function_call"])
    assert len(subjective) == 20
    assert len(function_call) == 20
    assert {row["tags"]["task_type"] for row in subjective} == {"qa"}
    assert {row["tags"]["task_type"] for row in function_call} == {"fc"}


def test_each_requested_task_needs_enough_records_for_auditable_holdout(tmp_path):
    qa_path, fc_path = tmp_path / "qa.json", tmp_path / "fc.json"
    _write_json(qa_path, _qa_records(40))
    fc = [
        {
            "conversations": [
                {"from": "human", "value": f"规划 {i}"},
                {
                    "from": "gpt",
                    "value": "",
                    "tool_calls": [{"name": "plan", "arguments": {}}],
                },
            ],
            "tools": _tools(),
        }
        for i in range(1)
    ]
    _write_json(fc_path, fc)

    with pytest.raises(ValueError, match="fc.*2 records"):
        prepare_generated_datasets(
            [qa_path, fc_path], "wf", 0, 0.1, 42, tmp_path / "out",
            validation_ratio=0.1, register=False,
        )


def test_profile_reads_generation_progress_and_registers_artifacts(tmp_path, monkeypatch):
    source_dir = tmp_path / "job"
    source_dir.mkdir()
    source = source_dir / "output.json"
    _write_json(source, _qa_records(40))
    (source_dir / "config.json").write_text(
        json.dumps({"task_type": "qa_multi"}), encoding="utf-8"
    )
    (source_dir / "progress.json").write_text(
        json.dumps(
            {
                "state": "done",
                "accepted": 40,
                "attempts": 50,
                "rejects": {"去重": 7, "生成": 3},
            }
        ),
        encoding="utf-8",
    )
    registrations = []

    def record_registration(src_path, name, kind, source):
        registrations.append((src_path, name, kind, source))
        return {"name": name}

    monkeypatch.setattr(
        "llamafactory_pipeline.assistant_data.dataset_store.register_dataset",
        record_registration,
    )

    prepared = prepare_generated_datasets(
        [source], "workflow-with-a-name-that-is-deliberately-long-1234567890", 2,
        0.1, 42, tmp_path / "out", validation_ratio=0.2, register=True,
    )

    assert prepared.profile.task_types == ["qa_multi"]
    assert prepared.profile.generation_acceptance_rate == 0.8
    assert prepared.profile.rejection_counts == {"去重": 7, "生成": 3}
    assert all(len(name) <= 64 for _, name, _, _ in registrations)
    assert [kind for _, _, kind, _ in registrations] == ["train", "eval"]
    assert all(source_name == "assistant" for _, _, _, source_name in registrations)


def test_invalid_validation_ratio_is_rejected_before_writing(tmp_path):
    source = tmp_path / "qa.json"
    _write_json(source, _qa_records(4))
    with pytest.raises(ValueError, match="validation_ratio"):
        prepare_generated_datasets(
            [source], "wf", 0, 0.1, 42, tmp_path / "out",
            validation_ratio=0.5, register=False,
        )
