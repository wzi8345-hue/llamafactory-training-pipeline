"""数据集注册纯逻辑自检: 校验/落盘/列表/删除。不依赖网络。

运行: python -m pytest llamafactory_pipeline/test_dataset.py -q
"""

from __future__ import annotations

import json

import pytest

from llamafactory_pipeline import dataset_store as ds


def _write_train_json(path, n=3):
    json.dump([{"conversations": [{"from": "human", "value": f"Q{i}?"},
                                  {"from": "gpt", "value": f"A{i}."}]}
               for i in range(n)], open(path, "w"), ensure_ascii=False)


def _write_eval_jsonl(path, n=2):
    with open(path, "w", encoding="utf-8") as f:
        for i in range(n):
            f.write(json.dumps({
                "id": f"fc{i}", "query": f"q{i}",
                "tools": [{"function": {"name": "f"}}],
                "gold": {"name": "f", "arguments": {}}}) + "\n")


# ── 名校验 ──

def test_validate_name():
    assert ds.validate_name("ds_v1") == "ds_v1"
    for bad in ["", "bad name", "a/b", "..", "a;b"]:
        with pytest.raises(ValueError):
            ds.validate_name(bad)


# ── 训练集注册 ──

def test_register_train_dataset(tmp_path):
    src = tmp_path / "in.json"
    _write_train_json(src, n=3)
    base = tmp_path / "store"
    meta = ds.register_dataset(str(src), "qa_v1", "train", base=base)
    assert meta["n_records"] == 3
    assert meta["format"] == "conversations"
    assert meta["kind"] == "train"
    assert ds.data_path("qa_v1", "train", base).exists()
    assert ds.dataset_meta("qa_v1", "train", base)["n_records"] == 3


def test_register_train_messages_format(tmp_path):
    src = tmp_path / "in.json"
    json.dump([{"messages": [{"role": "user", "content": "hi"}]}],
              open(src, "w"), ensure_ascii=False)
    meta = ds.register_dataset(str(src), "m1", "train", base=tmp_path)
    assert meta["format"] == "messages"


def test_register_dpo_train_dataset(tmp_path):
    src = tmp_path / "dpo.json"
    json.dump([{
        "conversations": [{"from": "human", "value": "Q?"}],
        "chosen": {"from": "gpt", "value": "good"},
        "rejected": {"from": "gpt", "value": "bad"},
    }], open(src, "w"), ensure_ascii=False)
    meta = ds.register_dataset(str(src), "dpo_v1", "train", base=tmp_path)
    assert meta["format"] == "conversations"
    assert meta["finetune_type"] == "dpo"


def test_register_rejects_empty(tmp_path):
    src = tmp_path / "in.json"
    json.dump([], open(src, "w"))
    with pytest.raises(ValueError):
        ds.register_dataset(str(src), "x", "train", base=tmp_path)


def test_register_rejects_bad_train_format(tmp_path):
    src = tmp_path / "in.json"
    json.dump([{"instruction": "x", "output": "y"}], open(src, "w"))  # 非 ShareGPT
    with pytest.raises(ValueError):
        ds.register_dataset(str(src), "x", "train", base=tmp_path)


# ── 评测集注册 ──

def test_register_eval_dataset_fc(tmp_path):
    src = tmp_path / "in.jsonl"
    _write_eval_jsonl(src, n=2)
    meta = ds.register_dataset(str(src), "fc_v1", "eval", base=tmp_path)
    assert meta["format"] == "function_call"
    assert meta["n_records"] == 2


def test_register_eval_subjective_fallback(tmp_path):
    src = tmp_path / "in.json"
    json.dump([{"id": "s1", "query": "q", "reference": "r"}], open(src, "w"),
              ensure_ascii=False)
    meta = ds.register_dataset(str(src), "subj_v1", "eval", base=tmp_path)
    assert meta["format"] == "subjective"


# ── 列表/删除 ──

def test_list_datasets(tmp_path):
    s1 = tmp_path / "a.json"; _write_train_json(s1)
    s2 = tmp_path / "b.jsonl"; _write_eval_jsonl(s2)
    ds.register_dataset(str(s1), "train_a", "train", base=tmp_path)
    ds.register_dataset(str(s2), "eval_b", "eval", base=tmp_path)
    all_ds = ds.list_datasets(base=tmp_path)
    names = {d["name"] for d in all_ds}
    assert "train_a" in names and "eval_b" in names
    train_only = ds.list_datasets(kind="train", base=tmp_path)
    assert all(d["kind"] == "train" for d in train_only)


def test_delete_dataset(tmp_path):
    src = tmp_path / "a.json"; _write_train_json(src)
    ds.register_dataset(str(src), "x", "train", base=tmp_path)
    assert ds.delete_dataset("x", "train", base=tmp_path) is True
    assert ds.dataset_meta("x", "train", base=tmp_path) is None
    assert ds.delete_dataset("x", "train", base=tmp_path) is False  # 已删


def test_register_overwrites(tmp_path):
    src = tmp_path / "a.json"; _write_train_json(src, n=3)
    ds.register_dataset(str(src), "x", "train", base=tmp_path)
    src2 = tmp_path / "b.json"; _write_train_json(src2, n=5)
    meta = ds.register_dataset(str(src2), "x", "train", base=tmp_path)
    assert meta["n_records"] == 5  # 覆盖
    assert len(ds.list_datasets(base=tmp_path)) == 1  # 不重复
