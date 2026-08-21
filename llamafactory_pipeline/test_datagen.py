"""数据生成子系统纯逻辑自检: 闸门 / 去重 / 生成 / 记录组装 / 报告聚合。

运行: python -m pytest llamafactory_pipeline/test_datagen.py -q
LLM/embedder 用桩替身, 不打真实服务。
"""

from __future__ import annotations

import json
from io import StringIO

from . import datagen_job
from . import datagen_generate as gen
from . import datagen_quality as q
from . import datagen_run as run
from . import datagen_source as src
from .datagen_schema import (DEFAULT_FC_GEN_PROMPT, DEFAULT_QA_GEN_PROMPT,
                             DEFAULT_QA_MULTI_GEN_CONSENSUS,
                             DEFAULT_QA_MULTI_GEN_SYNTHESIS, DatagenConfig)


class FakeLLM:
    def __init__(self, answer: str):
        self._answer = answer

    def chat(self, **kwargs):
        return {"answer": self._answer}


class SequenceLLM:
    def __init__(self, answers):
        self._answers = list(answers)

    def chat(self, **kwargs):
        return {"answer": self._answers.pop(0)}


class FakeEmbedder:
    """A→[1,0], B→[0,1], 其他→对角。同类文本余弦=1, 触发近义去重。"""

    def embed(self, text: str):
        if "A" in text:
            return [1.0, 0.0]
        if "B" in text:
            return [0.0, 1.0]
        return [0.7071, 0.7071]


# ── 配置 / prompt 切换 ──

def test_config_prompt_switch():
    qa = DatagenConfig(task_type="qa", count=10)
    assert qa.resolved_gen_prompt() == DEFAULT_QA_GEN_PROMPT
    fc = DatagenConfig(task_type="fc", count=10)
    assert fc.resolved_gen_prompt() == DEFAULT_FC_GEN_PROMPT
    custom = DatagenConfig(task_type="qa", count=10, gen_prompt="我的模板")
    assert custom.resolved_gen_prompt() == "我的模板"


def test_config_defaults_to_sft_and_resolves_dpo_prompts():
    old = DatagenConfig.model_validate({"task_type": "qa", "count": 1})
    assert old.finetune_type == "sft"

    dpo = DatagenConfig(task_type="fc", count=1, finetune_type="dpo")
    assert "rejected" in dpo.resolved_rejected_prompt().lower()
    assert "chosen" in dpo.resolved_pair_judge_prompt().lower()


# ── FC 实体一致性 ──

def _tc(args: dict):
    return [{"name": "plan", "arguments": json.dumps(args, ensure_ascii=False)}]

def test_fc_entities_ok_missing_extra():
    tcs = _tc({"kw": ["Q345qENH 腐蚀速率"]})
    ok, _ = q.check_fc_entities("Q345qENH腐蚀速率多少", tcs)
    assert ok
    ok, why = q.check_fc_entities("腐蚀速率多少", tcs)  # 缺牌号
    assert not ok and "缺失" in why
    ok, why = q.check_fc_entities("Q345qENH和S355J2W呢", tcs)  # 臆造 S355J2W
    assert not ok and "臆造" in why


# ── FC schema ──

def test_fc_schema():
    names = {"plan", "ask"}
    ok, _ = q.check_fc_schema(_tc({"a": 1}), names)
    assert ok
    ok, _ = q.check_fc_schema([{"name": "unknown", "arguments": "{}"}], names)
    assert not ok
    ok, why = q.check_fc_schema([{"name": "plan", "arguments": "{bad"}], names)
    assert not ok and "JSON" in why


# ── 去重 ──

def test_deduper():
    dd = q.Deduper(FakeEmbedder(), threshold=0.9)
    assert dd.is_dup("x1 A") is False       # 首条
    assert dd.is_dup("x2 A") is True        # 近义 (余弦1)
    assert dd.is_dup("y B") is False        # 不同向量
    assert dd.is_dup("x1 A") is True        # 完全重复 (hash)


def test_deduper_prime():
    dd = q.Deduper(FakeEmbedder(), threshold=0.9)
    dd.prime("seed A")
    assert dd.is_dup("other A") is True      # 续跑后与已有近义命中


class FakeBatchEmbedder(FakeEmbedder):
    """支持 embed_batch: 复用 FakeEmbedder 的单条映射。"""
    def embed_batch(self, texts):
        return [self.embed(t) for t in texts]


def test_deduper_prime_batch():
    dd = q.Deduper(FakeBatchEmbedder(), threshold=0.9)
    dd.prime_batch(["seed A", "seed B"])
    assert dd.is_dup("other A") is True      # 与 A 近义
    assert dd.is_dup("other B") is True      # 与 B 近义
    assert dd.is_dup("fresh C") is False     # 全新


def test_deduper_prime_batch_fallback_no_embed_batch():
    """embedder 无 embed_batch 时 prime_batch 逐条 embed, 结果一致。"""
    dd = q.Deduper(FakeEmbedder(), threshold=0.9)  # FakeEmbedder 无 embed_batch
    dd.prime_batch(["seed A"])
    assert dd.is_dup("other A") is True


# ── judge ──

def test_judge_qa_gates():
    chunk = {"content": "片段"}
    item = {"question": "q", "answer": "a"}
    llm = FakeLLM('{"grounded": false, "score": 5}')
    ok, why, _ = q.judge_qa(llm, chunk, item, "p", 4, True)
    assert not ok and "接地" in why
    llm = FakeLLM('{"grounded": true, "score": 2}')
    ok, why, _ = q.judge_qa(llm, chunk, item, "p", 4, True)
    assert not ok and "分数" in why
    llm = FakeLLM('{"grounded": true, "score": 5}')
    ok, _, meta = q.judge_qa(llm, chunk, item, "p", 4, True)
    assert ok and meta["score"] == 5


# ── 生成 ──

def test_gen_qa():
    llm = FakeLLM('{"question": "腐蚀速率?", "answer": "约0.1mm/y"}')
    item = gen.gen_qa(llm, {"content": "..."}, "p", 0.9)
    assert item == {"question": "腐蚀速率?", "answer": "约0.1mm/y"}
    assert gen.gen_qa(FakeLLM("非JSON"), {"content": "x"}, "p", 0.9) is None


def test_gen_fc_preserves_label():
    seed = {"utterance": "原发话", "tool_calls": _tc({"kw": ["x"]})}
    llm = FakeLLM('{"utterance": "改写后的话"}')
    item = gen.gen_fc(llm, seed, "p", 0.9)
    assert item["utterance"] == "改写后的话"
    assert item["tool_calls"] == seed["tool_calls"]   # 标签沿用


def test_gen_qa_rejected_and_pair_judge():
    rejected = gen.gen_qa_rejected(
        FakeLLM('{"rejected": "把0.1误写成1.0。"}'),
        {"content": "腐蚀速率为0.1"},
        {"question": "速率？", "answer": "0.1"}, "p", 0.9)
    assert rejected == "把0.1误写成1.0。"

    ok, _, meta = q.judge_preference_pair(
        FakeLLM('{"chosen_better": true, "chosen_score": 5, "rejected_score": 2}'),
        "evidence", "0.1", rejected, "p", 4)
    assert ok and meta["chosen_score"] == 5


def test_preference_pair_rejects_identical_answers_without_llm_call():
    class MustNotCall:
        def chat(self, **kwargs):
            raise AssertionError("相同答案应在规则闸被拒绝")

    ok, why, _ = q.judge_preference_pair(
        MustNotCall(), "evidence", "same", " same ", "p", 4)
    assert not ok and "相同" in why


def test_preference_pair_rejects_judge_disagreement_and_no_score_gap():
    ok, why, _ = q.judge_preference_pair(
        FakeLLM('{"chosen_better": false, "chosen_score": 5, "rejected_score": 1}'),
        "ctx", "good", "bad", "p", 4)
    assert not ok and "偏好" in why

    ok, why, _ = q.judge_preference_pair(
        FakeLLM('{"chosen_better": true, "chosen_score": 5, "rejected_score": 5}'),
        "ctx", "good", "bad", "p", 4)
    assert not ok and "差距" in why


def test_fc_tool_message_uses_llamafactory_native_format():
    msg = gen.fc_tool_message(_tc({"kw": ["Q345"]}))
    assert msg["from"] == "function_call"
    assert json.loads(msg["value"]) == {
        "name": "plan", "arguments": {"kw": ["Q345"]}}


def test_gen_and_check_fc_rejected_three_error_types():
    chosen = _tc({"kw": ["Q345"]})
    seed = {"utterance": "查Q345", "tool_calls": chosen}
    item = {"utterance": "帮我查Q345", "tool_calls": chosen}
    cases = [
        ("wrong_tool", '{"name":"ask","arguments":{"q":"请补充"}}'),
        ("wrong_args", '{"name":"plan","arguments":{"kw":["S355"]}}'),
        ("direct_answer", '{"answer":"Q345的结果是……"}'),
    ]
    for error_type, answer in cases:
        rejected = gen.gen_fc_rejected(
            FakeLLM(answer), seed, item, "p", 0.9, error_type)
        ok, why = q.check_fc_rejected(
            chosen, rejected, error_type, {"plan", "ask"})
        assert ok, (error_type, why, rejected)


def test_fc_rejected_rejects_unchanged_or_unknown_tool():
    chosen = _tc({"kw": ["Q345"]})
    unchanged = gen.fc_tool_message(chosen)
    ok, why = q.check_fc_rejected(
        chosen, unchanged, "wrong_args", {"plan", "ask"})
    assert not ok and "未改变" in why

    unknown = {"from": "function_call", "value": '{"name":"missing","arguments":{}}'}
    ok, why = q.check_fc_rejected(
        chosen, unknown, "wrong_tool", {"plan", "ask"})
    assert not ok and "未知工具" in why


# ── 记录组装 / dedup key ──

def test_build_record_and_key():
    qa = run._build_record("sft", "qa", {"question": "q", "answer": "a"}, [])
    assert qa["conversations"][0] == {"from": "human", "value": "q"}
    assert qa["conversations"][1] == {"from": "gpt", "value": "a"}
    assert run._dedup_key("qa", qa) == "q"

    tools = [{"function": {"name": "plan"}}]
    fc = run._build_record(
        "sft", "fc", {"utterance": "u", "tool_calls": _tc({})}, tools)
    assert fc["conversations"][1]["tool_calls"] == _tc({})
    assert fc["tools"] == tools
    assert run._dedup_key("fc", fc) == "u"


def test_build_record_persists_deterministic_slice_tags():
    record = run._build_record(
        "sft",
        "fc",
        {"utterance": "u", "tool_calls": _tc({})},
        [{"function": {"name": "plan"}}],
        tags={"task_type": "fc", "tool_name": "plan"},
    )
    assert record["tags"] == {"task_type": "fc", "tool_name": "plan"}


def test_fc_seed_history_excludes_final_label(tmp_path):
    p = tmp_path / "fc.json"
    p.write_text(json.dumps([{
        "conversations": [
            {"from": "human", "value": "先查Q345"},
            {"from": "gpt", "value": "上一轮回答"},
            {"from": "human", "value": "那腐蚀呢"},
            {"from": "gpt", "value": "", "tool_calls": _tc({"kw": ["Q345"]})},
        ],
        "tools": [{"function": {"name": "plan"}}],
    }], ensure_ascii=False), encoding="utf-8")
    seeds, _ = src.load_fc_seeds(str(p))
    assert seeds[0]["history"] == [
        {"from": "human", "value": "先查Q345"},
        {"from": "gpt", "value": "上一轮回答"},
        {"from": "human", "value": "那腐蚀呢"},
    ]


def test_build_dpo_fc_record_preserves_history_and_native_messages():
    tools = [{"function": {"name": "plan"}}]
    item = {
        "utterance": "帮我看它的腐蚀",
        "tool_calls": _tc({"kw": ["Q345"]}),
        "history": [
            {"from": "human", "value": "先查Q345"},
            {"from": "gpt", "value": "上一轮回答"},
            {"from": "human", "value": "原末轮"},
        ],
    }
    rejected = {"from": "gpt", "value": "不需要检索"}
    record = run._build_record("dpo", "fc", item, tools, rejected)
    assert record["conversations"][-1] == {
        "from": "human", "value": "帮我看它的腐蚀"}
    assert record["chosen"]["from"] == "function_call"
    assert json.loads(record["chosen"]["value"])["name"] == "plan"
    assert record["rejected"] == rejected
    assert json.loads(record["tools"])[0]["function"]["name"] == "plan"
    assert run._dedup_key("fc", record) == "帮我看它的腐蚀"


def test_build_dpo_qa_record():
    item = {"question": "q", "answer": "good"}
    record = run._build_record("dpo", "qa", item, [], "bad")
    assert record == {
        "conversations": [{"from": "human", "value": "q"}],
        "chosen": {"from": "gpt", "value": "good"},
        "rejected": {"from": "gpt", "value": "bad"},
    }


class AlwaysUnique:
    def is_dup(self, text):
        return False


class DirectAnswerRng:
    def choice(self, values):
        assert "direct_answer" in values
        return "direct_answer"


def test_one_candidate_dpo_qa_writes_preference_record():
    cfg = DatagenConfig(task_type="qa", count=1, finetune_type="dpo")
    llm = SequenceLLM([
        '{"question":"速率？","answer":"0.1"}',
        '{"grounded":true,"score":5}',
        '{"rejected":"1.0"}',
        '{"chosen_better":true,"chosen_score":5,"rejected_score":2}',
    ])
    fout, fman = StringIO(), StringIO()
    ok, why, _ = run._one_candidate(
        cfg, llm, AlwaysUnique(), lambda: {
            "content": "速率为0.1", "doc_name": "manual-a.pdf"
        },
        "gen", "judge", "reject", "pair", set(), [], fout, fman, 0,
        DirectAnswerRng())
    assert ok, why
    record = json.loads(fout.getvalue())
    assert record["chosen"]["value"] == "0.1"
    assert record["rejected"]["value"] == "1.0"
    assert record["tags"]["source_doc"] == ["manual-a.pdf"]
    assert json.loads(fman.getvalue())["finetune_type"] == "dpo"


def test_slice_tags_preserve_every_source_document():
    tags = run._slice_tags(
        "qa_multi",
        {"question": "q", "answer": "a"},
        {"doc_ids": ["doc-a", "doc-b", "doc-c"]},
        None,
    )

    assert tags["source_doc"] == ["doc-a", "doc-b", "doc-c"]


def test_one_candidate_dpo_fc_writes_error_type_and_native_chosen():
    chosen = _tc({"kw": ["Q345"]})
    seed = {
        "utterance": "查Q345", "tool_calls": chosen,
        "history": [{"from": "human", "value": "查Q345"}],
    }
    cfg = DatagenConfig(task_type="fc", count=1, finetune_type="dpo")
    llm = SequenceLLM([
        '{"utterance":"帮我查Q345"}',
        '{"score":5}',
        '{"answer":"不需要调用工具"}',
        '{"chosen_better":true,"chosen_score":5,"rejected_score":1}',
    ])
    fout, fman = StringIO(), StringIO()
    tools = [{"function": {"name": "plan"}}, {"function": {"name": "ask"}}]
    ok, why, _ = run._one_candidate(
        cfg, llm, AlwaysUnique(), lambda: seed, "gen", "judge", "reject",
        "pair", {"plan", "ask"}, tools, fout, fman, 0, DirectAnswerRng())
    assert ok, why
    record = json.loads(fout.getvalue())
    assert record["chosen"]["from"] == "function_call"
    assert record["rejected"] == {"from": "gpt", "value": "不需要调用工具"}
    manifest = json.loads(fman.getvalue())
    assert manifest["finetune_type"] == "dpo"
    assert manifest["error_type"] == "direct_answer"


# ── 报告聚合 ──

def test_finalize(tmp_path):
    (tmp_path / "output.jsonl").write_text(
        json.dumps({"conversations": []}, ensure_ascii=False) + "\n", encoding="utf-8")
    cfg = DatagenConfig(task_type="qa", count=10)
    run._finalize(tmp_path, cfg, accepted=1, attempts=4, rejects={"judge": 2, "去重": 1})
    out = json.loads((tmp_path / "output.json").read_text("utf-8"))
    assert len(out) == 1
    report = (tmp_path / "report.md").read_text("utf-8")
    assert "接受率" in report and "judge" in report


# ── 原子写 ──

def test_atomic_write_produces_parseable_file(tmp_path):
    """_atomic_write 后目标文件可被 json.loads, 且不留 .tmp 残留。"""
    p = tmp_path / "progress.json"
    run._atomic_write(p, json.dumps({"state": "running", "accepted": 3}))
    assert json.loads(p.read_text("utf-8")) == {"state": "running", "accepted": 3}
    assert not (tmp_path / "progress.json.tmp").exists()  # tmp 已 replace 走


def test_atomic_write_replaces_existing(tmp_path):
    """覆盖写时旧内容被整体替换, 不会读到半新半旧。"""
    p = tmp_path / "progress.json"
    p.write_text('{"old": true}', encoding="utf-8")
    run._atomic_write(p, json.dumps({"state": "done", "accepted": 10}))
    obj = json.loads(p.read_text("utf-8"))
    assert obj == {"state": "done", "accepted": 10} and "old" not in obj


# ── QA 惰性采样 ──

def test_qa_source_lazy_sample(tmp_path):
    import random
    doc = tmp_path / "docA"
    doc.mkdir()
    (doc / "knowledge_blocks_vec.json").write_text(json.dumps([
        {"type": "text", "content": "x" * 300, "section": "s", "embedding": [0.1]},
        {"type": "title", "content": "短", "section": ""},          # 类型不符
        {"type": "text", "content": "短", "section": ""},           # 太短
    ], ensure_ascii=False), encoding="utf-8")
    qs = src.QASource(str(tmp_path), min_len=200)
    assert len(qs) == 1
    c = qs.sample(random.Random(0))
    assert c["content"] == "x" * 300 and "embedding" not in c   # 只留 content, 不留向量


def test_qa_source_cache_bounded(tmp_path):
    """加载 >_CACHE_MAX 个文件后, 缓存不超界, 最久未访问的被淘汰。"""
    import random
    for i in range(src._CACHE_MAX + 3):
        doc = tmp_path / f"doc{i}"
        doc.mkdir()
        (doc / "knowledge_blocks_vec.json").write_text(json.dumps([
            {"type": "text", "content": f"c{i}" * 200, "section": "s"}],
            ensure_ascii=False), encoding="utf-8")
    qs = src.QASource(str(tmp_path), min_len=200)
    rng = random.Random(0)
    # 逐个加载所有文件 (每个 sample 命中一个新文件)
    for _ in range(src._CACHE_MAX + 3):
        qs.sample(rng)
    assert len(qs._cache) <= src._CACHE_MAX


def test_qa_source_cache_lru_evicts_oldest(tmp_path):
    """连续访问文件 0 后再加载其他文件, 文件 0 应被淘汰 (除非再访问)。"""
    import random
    paths = []
    for i in range(src._CACHE_MAX + 1):
        doc = tmp_path / f"doc{i}"
        doc.mkdir()
        p = doc / "knowledge_blocks_vec.json"
        p.write_text(json.dumps([{"type": "text", "content": f"c{i}" * 200}],
                               ensure_ascii=False), encoding="utf-8")
        paths.append(p)
    qs = src.QASource(str(tmp_path), min_len=200)
    qs._load(paths[0])                       # 先加载文件0
    for i in range(1, src._CACHE_MAX + 1):   # 再加载 8 个新文件, 挤掉文件0
        qs._load(paths[i])
    assert paths[0] not in qs._cache         # 最久未访问的被淘汰
    assert len(qs._cache) == src._CACHE_MAX


# ── datagen 任务列表 ──

def test_list_datagen_jobs(tmp_path, monkeypatch):
    """list_jobs 扫 _BASE 目录, 按目录名格式过滤, 最新在前。"""
    import time as _time
    base = tmp_path / "generated"
    base.mkdir()
    # 两个合法 job 目录 + 一个非法名
    for jid in ["20260721T071950Z-c0e935", "20260721T071311Z-5bd111", "not-a-job"]:
        d = base / jid
        d.mkdir()
        (d / "config.json").write_text(
            json.dumps({"task_type": "qa", "count": 10}), encoding="utf-8")
        (d / "progress.json").write_text(
            json.dumps({"state": "done", "accepted": 5, "target": 10}),
            encoding="utf-8")
    from llamafactory_pipeline import datagen_job as dj
    monkeypatch.setattr(dj, "_BASE", base)
    jobs = dj.list_jobs()
    ids = [j["job_id"] for j in jobs]
    assert ids == ["20260721T071950Z-c0e935", "20260721T071311Z-5bd111"]  # 非法被过滤, 逆序
    assert jobs[0]["status"] == "failed"
    assert jobs[0]["task_type"] == "qa"
    assert jobs[0]["accepted"] == 5


# ── FC 种子解析 ──

def test_load_fc_seeds(tmp_path):
    data = [{
        "conversations": [
            {"from": "human", "value": "问Q345qENH"},
            {"from": "gpt", "value": "", "tool_calls": _tc({"kw": ["Q345qENH"]})},
        ],
        "tools": [{"function": {"name": "plan"}}],
    }]
    p = tmp_path / "seed.json"
    p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    seeds, tools = src.load_fc_seeds(str(p))
    assert seeds[0]["utterance"] == "问Q345qENH"
    assert tools == [{"function": {"name": "plan"}}]


# ── 多篇 QA (qa_multi) ──

def test_qa_multi_prompt_and_min_docs():
    syn = DatagenConfig(task_type="qa_multi", count=10, sub_mode="synthesis")
    assert syn.resolved_gen_prompt() == DEFAULT_QA_MULTI_GEN_SYNTHESIS
    assert syn.min_docs() == 2                      # 综合型强制跨篇
    con = DatagenConfig(task_type="qa_multi", count=10, sub_mode="consensus")
    assert con.resolved_gen_prompt() == DEFAULT_QA_MULTI_GEN_CONSENSUS
    assert con.min_docs() == 1
    assert DatagenConfig(task_type="qa", count=10).min_docs() == 1


def test_judge_qa_min_docs_gate():
    chunk = {"content": "多篇片段"}
    item = {"question": "q", "answer": "a"}
    llm = FakeLLM('{"grounded": true, "score": 5, "used_docs": [1]}')
    ok, why, _ = q.judge_qa(llm, chunk, item, "p", 4, True, min_docs=2)
    assert not ok and "覆盖不足" in why
    llm = FakeLLM('{"grounded": true, "score": 5, "used_docs": [1, 2]}')
    ok, _, _ = q.judge_qa(llm, chunk, item, "p", 4, True, min_docs=2)
    assert ok


def test_evidence_text_labels():
    txt = src.build_evidence_text([{"content": "甲"}, {"content": "乙"}])
    assert txt == "[文献1]\n甲\n\n[文献2]\n乙"


class FakeMilvus:
    """第一次 search 返回锚点 (带 embedding), 之后返回给定候选池。"""

    def __init__(self, pool):
        self.pool = pool
        self.calls = 0

    def search(self, **kw):
        self.calls += 1
        if self.calls == 1:
            return [[{"entity": {"embedding": [0.1, 0.2]}}]]
        return [self.pool]


def _hit(doc, content, typ="text"):
    return {"entity": {"doc_id": doc, "doc_name": doc, "content": content, "type": typ}}


def test_milvus_evidence_gather_distinct_docs():
    import random
    pool = [
        _hit("A", "a" * 300),
        _hit("A", "a2" * 300),            # 同 doc, 跳过
        _hit("B", "短"),                   # 太短, 跳过
        _hit("B", "b" * 300),
        _hit("C", "c" * 300, typ="image"),  # 类型不符, 跳过
        _hit("C", "c2" * 300),
        _hit("D", "d" * 300),
    ]
    s = src.MilvusEvidenceSource(FakeMilvus(pool), "col", dim=2, n_docs=3, min_len=200)
    picked = s.gather(random.Random(0))
    assert [c["doc_id"] for c in picked] == ["A", "B", "C"]   # 每 doc 一条, 凑够 3


def test_milvus_evidence_insufficient_returns_empty():
    import random
    pool = [_hit("A", "a" * 300), _hit("A", "a2" * 300)]      # 只有 1 篇不同文献
    s = src.MilvusEvidenceSource(FakeMilvus(pool), "col", dim=2, n_docs=3, min_len=200)
    assert s.gather(random.Random(0)) == []


# ── 源耗尽提前终止 ──

def test_run_job_early_terminate_on_empty_source(tmp_path, monkeypatch):
    """qa_multi 源永远凑不齐 >=2 篇 → 连续 20 次"召集: 文献不足"后提前终止,
    不应硬撑到 max_attempts (=count*multiplier=30)。"""
    import random as _random
    # 桩: build_milvus 返回总返回空池的 FakeMilvus → gather 永远返回 []
    empty_milvus = src.MilvusEvidenceSource(FakeMilvus([]), "col", dim=2,
                                            n_docs=3, min_len=200)
    monkeypatch.setattr(run, "build_milvus", lambda coll: (empty_milvus.client,
                                                           empty_milvus.dim))
    monkeypatch.setattr(run, "build_embedder", lambda: FakeEmbedder())
    # build_llm 在 run_job 内 `from rag_eval_plan.common import build_llm` 已绑定, 桩它
    monkeypatch.setattr("rag_eval_plan.common.build_llm",
                        lambda: FakeLLM('{"question":"q","answer":"a"}'))
    monkeypatch.setattr(datagen_job, "_process_identity", lambda pid: "test-start")

    cfg = DatagenConfig(task_type="qa_multi", count=10, attempt_multiplier=3.0)
    d = tmp_path / "job"
    d.mkdir()
    (d / "config.json").write_text(json.dumps(cfg.model_dump(), ensure_ascii=False),
                                   encoding="utf-8")
    run.run_job(str(d))

    prog = json.loads((d / "progress.json").read_text("utf-8"))
    assert prog["state"] == "error"
    assert prog["accepted"] == 0
    # 提前终止: attempts 应 <= 20 (致命阈值), 远小于 max_attempts=30
    assert prog["attempts"] <= 20, f"未提前终止: attempts={prog['attempts']}"
    assert prog["rejects"].get("召集", 0) >= 20


def test_datagen_launcher_does_not_duplicate_existing_job(tmp_path, monkeypatch):
    monkeypatch.setattr(datagen_job, "_BASE", tmp_path)
    job_id = "20260819T010203Z-a1b2c3"
    job_dir = tmp_path / job_id
    job_dir.mkdir()
    config = DatagenConfig(task_type="qa", count=10).model_dump(mode="json")
    (job_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (job_dir / "pid").write_text("123", encoding="utf-8")
    launched = []
    monkeypatch.setattr(
        datagen_job.subprocess,
        "Popen",
        lambda *args, **kwargs: launched.append((args, kwargs)),
    )
    monkeypatch.setattr(datagen_job, "_pid_alive", lambda directory: True)

    datagen_job.create_and_launch(job_id, config)

    assert launched == []


def test_datagen_launcher_recovers_stale_pid_with_same_job_id(tmp_path, monkeypatch):
    monkeypatch.setattr(datagen_job, "_BASE", tmp_path)
    job_id = "20260819T010203Z-a1b2c3"
    directory = tmp_path / job_id
    directory.mkdir()
    config = DatagenConfig(task_type="qa", count=10).model_dump(mode="json")
    (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (directory / "pid").write_text("123", encoding="utf-8")
    (directory / "progress.json").write_text(
        json.dumps({"state": "running", "accepted": 3}), encoding="utf-8"
    )
    launched = []
    monkeypatch.setattr(datagen_job, "_pid_alive", lambda path: False)
    class FakeProcess:
        pid = 456

    def fake_popen(*args, **kwargs):
        launched.append((args, kwargs))
        return FakeProcess()

    monkeypatch.setattr(datagen_job.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(datagen_job, "_process_identity", lambda pid: "start-456")

    datagen_job.create_and_launch(job_id, config)

    assert len(launched) == 1
    assert (directory / "pid").read_text("utf-8") == "456"
    assert (directory / "pid_identity").read_text("utf-8") == "start-456"
    assert json.loads((directory / "progress.json").read_text("utf-8"))["state"] == "starting"


def test_datagen_starting_marker_prevents_false_interruption_before_pid(tmp_path, monkeypatch):
    monkeypatch.setattr(datagen_job, "_BASE", tmp_path)
    job_id = "20260819T010203Z-a1b2c3"
    directory = tmp_path / job_id
    directory.mkdir()
    (directory / "config.json").write_text(
        json.dumps({"task_type": "qa", "finetune_type": "sft"}),
        encoding="utf-8",
    )
    (directory / "progress.json").write_text(
        json.dumps({"state": "starting", "updated_at": 1000.0}),
        encoding="utf-8",
    )
    monkeypatch.setattr(datagen_job.time, "time", lambda: 1010.0)

    assert datagen_job.status(job_id)["status"] == "running"


def test_datagen_pid_identity_rejects_reused_process(tmp_path, monkeypatch):
    directory = tmp_path / "job"
    directory.mkdir()
    (directory / "pid").write_text("123", encoding="utf-8")
    (directory / "pid_identity").write_text("original", encoding="utf-8")
    monkeypatch.setattr(datagen_job.os, "kill", lambda pid, signal: None)
    monkeypatch.setattr(datagen_job, "_process_identity", lambda pid: "reused")

    assert datagen_job._pid_alive(directory) is False


def test_datagen_pid_without_creation_identity_is_never_trusted(
    tmp_path, monkeypatch
):
    directory = tmp_path / "job"
    directory.mkdir()
    (directory / "pid").write_text("123", encoding="utf-8")
    monkeypatch.setattr(datagen_job.os, "kill", lambda pid, signal: None)
    monkeypatch.setattr(datagen_job, "_process_identity", lambda pid: "live")

    assert datagen_job._pid_alive(directory) is False


def test_datagen_pid_file_race_is_treated_as_not_alive(tmp_path, monkeypatch):
    directory = tmp_path / "job"
    directory.mkdir()
    pid_path = directory / "pid"
    pid_path.write_text("123", encoding="utf-8")
    original = type(pid_path).read_text

    def racing_read(path, *args, **kwargs):
        if path == pid_path:
            raise OSError("removed during replay")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(type(pid_path), "read_text", racing_read)

    assert datagen_job._pid_alive(directory) is False


def test_datagen_stop_fails_closed_without_creation_identity(tmp_path, monkeypatch):
    monkeypatch.setattr(datagen_job, "_BASE", tmp_path)
    job_id = "20260819T010203Z-a1b2c3"
    directory = tmp_path / job_id
    directory.mkdir()
    (directory / "pid").write_text("123", encoding="utf-8")
    kills = []
    monkeypatch.setattr(
        datagen_job.os, "kill", lambda pid, sig: kills.append((pid, sig))
    )

    result = datagen_job.stop(job_id)

    assert result["stopped"] is False
    assert kills == []


def test_datagen_status_marks_dead_running_process_interrupted(tmp_path, monkeypatch):
    monkeypatch.setattr(datagen_job, "_BASE", tmp_path)
    job_id = "20260819T010203Z-a1b2c3"
    directory = tmp_path / job_id
    directory.mkdir()
    (directory / "config.json").write_text(
        json.dumps({"task_type": "qa", "finetune_type": "sft"}),
        encoding="utf-8",
    )
    (directory / "progress.json").write_text(
        json.dumps({"state": "running", "accepted": 3}), encoding="utf-8"
    )
    monkeypatch.setattr(datagen_job, "_pid_alive", lambda path: False)

    observed = datagen_job.status(job_id)

    assert observed["status"] == "interrupted"
