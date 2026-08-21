"""Reusable registered-evaluation service tests."""

from __future__ import annotations

import json

import pytest

from . import dataset_store, eval_judge, eval_remote, eval_service, remote
from .eval_schema import EvalRequest, ModelUnderTest


def _request() -> EvalRequest:
    return EvalRequest(
        models=[ModelUnderTest(name="base", model_name_or_path="/m")],
        task_types=["subjective"],
        gpus="0",
    )


def test_submit_registered_eval_uses_dataset_store(tmp_path, monkeypatch):
    base = tmp_path / "sft_data"
    eval_dir = base / "evalsets"
    eval_dir.mkdir(parents=True)
    data = eval_dir / "eval.data"
    data.write_text(
        '{"id":"1","query":"Q","reference":"A",'
        '"tags":{"source_doc":["doc-a","doc-b"]}}\n',
        encoding="utf-8",
    )
    (eval_dir / "eval.meta.json").write_text(
        '{"name":"eval","kind":"eval","ext":".jsonl",'
        '"format":"subjective","n_records":1}',
        encoding="utf-8",
    )
    captured = []
    monkeypatch.setattr(dataset_store, "_ROOT", base)
    monkeypatch.setattr(eval_service, "EVAL_RESULTS", tmp_path / "results")
    monkeypatch.setattr(
        eval_remote, "new_eval_id", lambda: "20260819T010203Z-a1b2c3"
    )
    monkeypatch.setattr(eval_remote, "submit_eval", lambda *args: captured.append(args))

    eval_id = eval_service.submit_registered_eval(
        remote.RemoteConfig("u@h", "/jobs", "/opt/LF"),
        _request(),
        fc_dataset_name=None,
        subjective_dataset_name="eval",
    )

    assert eval_id == "20260819T010203Z-a1b2c3"
    assert captured[0][4].endswith("subjective.items.jsonl")
    assert captured[0][2].gpus == "0"
    saved = json.loads(
        (tmp_path / "results" / eval_id / "config.json").read_text("utf-8")
    )
    assert saved["models"][0]["name"] == "base"
    normalized = [
        json.loads(line)
        for line in (tmp_path / "results" / eval_id / "subjective.items.jsonl")
        .read_text("utf-8")
        .splitlines()
    ]
    assert normalized[0]["tags"]["source_doc"] == ["doc-a", "doc-b"]
    manifest = json.loads(
        (tmp_path / "results" / eval_id / "frozen_eval_manifest.json").read_text(
            "utf-8"
        )
    )
    assert manifest["tasks"]["subjective"]["ids"] == ["1"]
    assert len(manifest["tasks"]["subjective"]["sha256"]) == 64


def test_submit_registered_eval_replays_preallocated_id(tmp_path, monkeypatch):
    base = tmp_path / "sft_data"
    eval_dir = base / "evalsets"
    eval_dir.mkdir(parents=True)
    (eval_dir / "eval.data").write_text(
        '{"id":"1","query":"Q","reference":"A"}\n', encoding="utf-8"
    )
    (eval_dir / "eval.meta.json").write_text(
        '{"name":"eval","kind":"eval","ext":".jsonl",'
        '"format":"subjective","n_records":1}',
        encoding="utf-8",
    )
    monkeypatch.setattr(dataset_store, "_ROOT", base)
    monkeypatch.setattr(eval_service, "EVAL_RESULTS", tmp_path / "results")
    submitted = []
    monkeypatch.setattr(
        eval_remote, "submit_eval", lambda *args: submitted.append(args)
    )
    monkeypatch.setattr(remote, "_write_remote_file", lambda *args: None)
    eval_id = "20260819T010203Z-a1b2c3"
    for _ in range(2):
        assert eval_service.submit_registered_eval(
            remote.RemoteConfig("u@h", "/jobs", "/opt/LF"),
            _request(),
            None,
            "eval",
            eval_id=eval_id,
        ) == eval_id
    assert [call[1] for call in submitted] == [eval_id, eval_id]


def test_score_registered_eval_reuses_normalized_items(tmp_path, monkeypatch):
    eval_id = "20260819T010203Z-a1b2c3"
    cfg = remote.RemoteConfig("u@h", "/jobs", "/opt/LF")
    req = _request()
    result_dir = tmp_path / eval_id
    result_dir.mkdir()
    (result_dir / "config.json").write_text(
        req.model_dump_json(), encoding="utf-8"
    )
    items = result_dir / "subjective.items.jsonl"
    items.write_text(
        '{"id":"1","query":"Q","reference":"A"}\n', encoding="utf-8"
    )
    digest = eval_remote.evaluation_submission_digest(
        cfg, eval_id, req, None, str(items)
    )
    eval_service._persist_frozen_eval_manifest(
        req, result_dir, None, str(items), digest
    )
    captured = {}

    def fake_scoring(
        cfg, eid, req, fc_items, subj_items, results_dir, **kwargs
    ):
        captured.update(
            eval_id=eid, fc=fc_items, subjective=subj_items, results_dir=results_dir
        )
        return {"eval_id": eid, "per_model": {}}

    monkeypatch.setattr(eval_judge, "run_scoring", fake_scoring)
    monkeypatch.setattr(remote, "submission_state", lambda *args: "SAME")
    result = eval_service.score_registered_eval(
        cfg,
        eval_id,
        results_root=tmp_path,
    )
    assert result["eval_id"] == eval_id
    assert captured["subjective"][0]["query"] == "Q"
    assert captured["fc"] == []


def test_scoring_rejects_tampered_frozen_items(tmp_path, monkeypatch):
    eval_id = "20260819T010203Z-a1b2c3"
    cfg = remote.RemoteConfig("u@h", "/jobs", "/opt/LF")
    req = _request()
    result_dir = tmp_path / eval_id
    result_dir.mkdir()
    items = result_dir / "subjective.items.jsonl"
    items.write_text(
        '{"id":"1","query":"Q","reference":"A"}\n', encoding="utf-8"
    )
    (result_dir / "config.json").write_text(
        req.model_dump_json(), encoding="utf-8"
    )
    digest = eval_remote.evaluation_submission_digest(
        cfg, eval_id, req, None, str(items)
    )
    eval_service._persist_frozen_eval_manifest(
        req, result_dir, None, str(items), digest
    )
    items.write_text(
        '{"id":"2","query":"favorable","reference":"A"}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        eval_judge,
        "run_scoring",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("tampered items reached scoring")
        ),
    )

    with pytest.raises(ValueError, match="frozen evaluation.*hash"):
        eval_service.score_registered_eval(
            cfg,
            eval_id,
            results_root=tmp_path,
        )


def test_scoring_rejects_prediction_ids_outside_frozen_manifest(
    tmp_path, monkeypatch
):
    eval_id = "20260819T010203Z-a1b2c3"
    cfg = remote.RemoteConfig("u@h", "/jobs", "/opt/LF")
    req = _request()
    result_dir = tmp_path / eval_id
    result_dir.mkdir()
    items = result_dir / "subjective.items.jsonl"
    items.write_text(
        '{"id":"1","query":"Q","reference":"A"}\n', encoding="utf-8"
    )
    (result_dir / "config.json").write_text(
        req.model_dump_json(), encoding="utf-8"
    )
    digest = eval_remote.evaluation_submission_digest(
        cfg, eval_id, req, None, str(items)
    )
    eval_service._persist_frozen_eval_manifest(
        req, result_dir, None, str(items), digest
    )

    def fetch(cfg, eid, model_name, local_path):
        with open(local_path, "w", encoding="utf-8") as handle:
            handle.write(
                '{"id":"1","task_type":"subjective","answer":"A"}\n'
            )
            handle.write(
                '{"id":"extra","task_type":"subjective","answer":"A"}\n'
            )

    monkeypatch.setattr(eval_remote, "fetch_predictions", fetch)
    monkeypatch.setattr(remote, "submission_state", lambda *args: "SAME")
    monkeypatch.setattr(
        eval_judge,
        "make_default_judge",
        lambda: (lambda system, user: '{"score":5,"reason":"ok"}'),
    )

    with pytest.raises(ValueError, match="prediction ID set"):
        eval_service.score_registered_eval(
            cfg,
            eval_id,
            results_root=tmp_path,
        )


def test_registered_eval_rejects_wrong_task_schema(tmp_path, monkeypatch):
    base = tmp_path / "sft_data"
    eval_dir = base / "evalsets"
    eval_dir.mkdir(parents=True)
    (eval_dir / "bad.data").write_text(
        '[{"id":"1","query":"Q"}]', encoding="utf-8"
    )
    (eval_dir / "bad.meta.json").write_text(
        '{"name":"bad","kind":"eval","ext":".json","n_records":1}',
        encoding="utf-8",
    )
    monkeypatch.setattr(dataset_store, "_ROOT", base)
    monkeypatch.setattr(eval_service, "EVAL_RESULTS", tmp_path / "results")
    request = EvalRequest(
        models=[ModelUnderTest(name="base", model_name_or_path="/m")],
        task_types=["function_call"],
    )
    try:
        eval_service.submit_registered_eval(
            remote.RemoteConfig("u@h", "/jobs", "/opt/LF"),
            request,
            fc_dataset_name="bad",
            subjective_dataset_name=None,
        )
    except ValueError as exc:
        assert "tools" in str(exc)
    else:
        raise AssertionError("wrong task schema should be rejected")
