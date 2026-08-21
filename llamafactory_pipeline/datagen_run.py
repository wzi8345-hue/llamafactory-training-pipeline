"""SFT 数据生成 job 主循环 (本地 detached 子进程运行)。

用法: python -m llamafactory_pipeline.datagen_run <job_dir>
job_dir 内需有 config.json (DatagenConfig)。产出写回同目录。
"""

from __future__ import annotations

import fcntl
import json
import os
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Any

# 允许作为脚本直接运行 (python -m 已在包内, 但补 sys.path 兜底)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llamafactory_pipeline import datagen_generate as gen  # noqa: E402
from llamafactory_pipeline import datagen_quality as q  # noqa: E402
from llamafactory_pipeline import datagen_source as src  # noqa: E402
from llamafactory_pipeline.datagen_schema import DatagenConfig  # noqa: E402


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _atomic_write(path: Path, content: str) -> None:
    """写临时文件再 os.replace, 保证读方永不看到半写状态。"""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, path)


def build_embedder():
    """按 local_api_config.embedding 建 embedder (归一化, 与库内向量同度量)。"""
    from rag_eval_plan.common import _pipeline_config, load_env
    from pipeline.clients.client_registry import get_global_registry

    load_env()
    emb = _pipeline_config().get("embedding", {})
    api_base = emb.get("api_base", "http://localhost:8002/v1")
    model = emb.get("model", "/Qwen3-Embedding-4B")
    api_key = os.environ.get("VLLM_API_KEY", "")
    if not api_key:
        raise ValueError("缺少 VLLM_API_KEY (应在 .env.local)")
    return get_global_registry().get_embedder(
        api_base=api_base, model=model, api_key=api_key, normalize=True)


def build_milvus(collection: str) -> tuple[Any, int]:
    """返回 (MilvusClient, embedding 维度)。dim 从 collection schema 读, 不写死。"""
    from rag_eval_plan.common import milvus_client
    client = milvus_client()
    dim = 0
    for f in client.describe_collection(collection).get("fields", []):
        if f.get("name") == "embedding":
            dim = int((f.get("params") or {}).get("dim") or 0)
    if dim <= 0:
        raise ValueError(f"无法确定 {collection} 的 embedding 维度")
    return client, dim


def _dedup_key(task_type: str, record: dict[str, Any]) -> str:
    if task_type == "fc":
        for turn in reversed(record["conversations"]):
            if turn.get("from") == "human":
                return turn["value"]
    return record["conversations"][0]["value"]


def _with_tags(record: dict[str, Any], tags: dict[str, str] | None) -> dict[str, Any]:
    if tags:
        record["tags"] = dict(tags)
    return record


def _build_record(finetune_type: str, task_type: str, item: dict[str, Any],
                  tools: list, rejected: Any = None,
                  tags: dict[str, str] | None = None) -> dict[str, Any]:
    if finetune_type == "sft" and task_type in ("qa", "qa_multi"):
        return _with_tags({"conversations": [
            {"from": "human", "value": item["question"]},
            {"from": "gpt", "value": item["answer"]},
        ]}, tags)
    if finetune_type == "sft":
        return _with_tags({
            "conversations": [
                {"from": "human", "value": item["utterance"]},
                {"from": "gpt", "value": "", "tool_calls": item["tool_calls"]},
            ],
            "tools": tools,
        }, tags)
    if task_type in ("qa", "qa_multi"):
        return _with_tags({
            "conversations": [{"from": "human", "value": item["question"]}],
            "chosen": {"from": "gpt", "value": item["answer"]},
            "rejected": {"from": "gpt", "value": rejected},
        }, tags)
    history = [dict(turn) for turn in item.get("history", [])]
    for turn in reversed(history):
        if turn.get("from") == "human":
            turn["value"] = item["utterance"]
            break
    else:
        history.append({"from": "human", "value": item["utterance"]})
    tools_value = tools if isinstance(tools, str) else json.dumps(tools, ensure_ascii=False)
    return _with_tags({
        "conversations": history,
        "chosen": gen.fc_tool_message(item["tool_calls"]),
        "rejected": rejected,
        "tools": tools_value,
    }, tags)


def _slice_tags(
    task_type: str,
    item: dict[str, Any],
    meta: dict[str, Any],
    error_type: str | None,
) -> dict[str, Any]:
    """Create the stable slice taxonomy carried into the frozen holdout."""
    tags = {"task_type": task_type}
    if task_type == "fc":
        calls = item.get("tool_calls") or []
        first = calls[0] if calls and isinstance(calls[0], dict) else {}
        function = first.get("function") if isinstance(first.get("function"), dict) else first
        if isinstance(function.get("name"), str) and function["name"]:
            tags["tool_name"] = function["name"]
    if error_type:
        tags["error_type"] = error_type
    doc_ids = meta.get("doc_ids") or meta.get("used_docs") or []
    if isinstance(doc_ids, (str, int)):
        doc_ids = [doc_ids]
    if isinstance(doc_ids, list):
        source_docs = list(
            dict.fromkeys(str(doc_id) for doc_id in doc_ids if doc_id not in (None, ""))
        )
        if source_docs:
            tags["source_doc"] = source_docs
    return tags


def run_job(job_dir: str) -> None:
    d = Path(job_dir)
    run_lock = (d / "run.lock").open("a+", encoding="utf-8")
    try:
        fcntl.flock(run_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        log("相同 job_id 已在运行，本次重放退出")
        return
    from llamafactory_pipeline.datagen_job import (
        _atomic_write_text,
        _process_identity,
    )

    identity = _process_identity(os.getpid())
    if not identity:
        raise RuntimeError("无法记录数据生成进程身份")
    _atomic_write_text(d / "pid", str(os.getpid()))
    _atomic_write_text(d / "pid_identity", identity)
    cfg = DatagenConfig.model_validate(json.loads((d / "config.json").read_text("utf-8")))
    out_jsonl = d / "output.jsonl"
    manifest = d / "manifest.jsonl"
    progress_path = d / "progress.json"

    rejects: dict[str, int] = {}

    def write_progress(
        state: str, accepted: int, attempts: int, error: str = ""
    ) -> None:
        # 原子写: 先写临时文件再 os.replace, 进程被 kill 在半写状态不会留截断 JSON
        # (datagen_job.status 会 json.loads 这个文件, 截断会静默吞掉状态)
        _atomic_write(progress_path, json.dumps({
            "state": state, "accepted": accepted, "target": cfg.count,
            "attempts": attempts, "rejects": rejects,
            "finetune_type": cfg.finetune_type,
            "error": error,
        }, ensure_ascii=False))

    try:
        log(f"启动: finetune={cfg.finetune_type} task={cfg.task_type} 目标={cfg.count}")
        from rag_eval_plan.common import build_llm
        llm = build_llm()
        deduper = q.Deduper(build_embedder(), cfg.dedup_threshold)

        # 载入源 (QA 惰性采样, 不全量读; FC 一次性载入种子)
        tools: list = []
        rng = random.Random()
        if cfg.task_type == "qa":
            qa_source = src.QASource(cfg.kb_source_dir, cfg.min_len)
            log(f"QA 源文件: {len(qa_source)}")
            if len(qa_source) == 0:
                raise ValueError("源为空, 无法生成")
            pick = lambda: qa_source.sample(rng)  # noqa: E731
        elif cfg.task_type == "qa_multi":
            client, dim = build_milvus(cfg.collection)
            ev_source = src.MilvusEvidenceSource(
                client, cfg.collection, dim, cfg.n_docs, cfg.neighbor_top_k, cfg.min_len)
            log(f"多篇证据源: collection={cfg.collection} dim={dim} n_docs={cfg.n_docs}")
            pick = lambda: ev_source.gather(rng)  # noqa: E731
        else:
            seeds, tools = src.load_fc_seeds(cfg.fc_seed_file)
            log(f"FC 种子: {len(seeds)}")
            if not seeds:
                raise ValueError("源为空, 无法生成")
            pick = lambda: src.sample(seeds, rng)  # noqa: E731
        tool_names = {t.get("function", {}).get("name") for t in tools}

        # 续跑: 已接受计数 + 灌入去重集
        accepted = 0
        if out_jsonl.exists():
            for line in out_jsonl.read_text("utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                deduper.prime(_dedup_key(cfg.task_type, rec))
                accepted += 1
            log(f"续跑: 已有 {accepted} 条")

        gen_prompt = cfg.resolved_gen_prompt()
        judge_prompt = cfg.resolved_judge_prompt()
        rejected_prompt = cfg.resolved_rejected_prompt()
        pair_judge_prompt = cfg.resolved_pair_judge_prompt()
        max_attempts = int(cfg.count * cfg.attempt_multiplier)
        attempts = 0

        fout = out_jsonl.open("a", encoding="utf-8")
        fman = manifest.open("a", encoding="utf-8")
        # 致命拒因 (源耗尽): 连续命中说明再试也没用, 提前终止省 LLM 调用
        _FATAL_REJECTS = {"召集", "源为空"}
        _FATAL_BREAK = 20
        fatal_streak = 0
        try:
            while accepted < cfg.count and attempts < max_attempts:
                attempts += 1
                try:
                    ok, reason, meta = _one_candidate(
                        cfg, llm, deduper, pick, gen_prompt, judge_prompt,
                        rejected_prompt, pair_judge_prompt, tool_names, tools,
                        fout, fman, accepted, rng)
                except Exception as e:  # 单条失败不终止整体
                    ok, reason = False, f"异常: {str(e)[:120]}"
                    meta = {}
                if ok:
                    accepted += 1
                    fatal_streak = 0
                    log(f"接受 {accepted}/{cfg.count} (尝试 {attempts})")
                else:
                    key = reason.split(":")[0]
                    rejects[key] = rejects.get(key, 0) + 1
                    if key in _FATAL_REJECTS:
                        fatal_streak += 1
                    else:
                        fatal_streak = 0
                    if fatal_streak >= _FATAL_BREAK:
                        log(f"连续 {fatal_streak} 次源耗尽 ({key}), 提前终止")
                        break
                if attempts % 5 == 0:
                    write_progress("running", accepted, attempts)
        finally:
            fout.close()
            fman.close()

        _finalize(d, cfg, accepted, attempts, rejects)
        if accepted < cfg.count:
            reason = f"可用数据不足: 接受 {accepted}/{cfg.count}"
            write_progress("error", accepted, attempts, reason)
            log(reason)
        else:
            write_progress("done", accepted, attempts)
            log(f"完成: 接受 {accepted}/{cfg.count}, 尝试 {attempts}")
    except Exception:
        log("任务失败:\n" + traceback.format_exc())
        write_progress("error", locals().get("accepted", 0), locals().get("attempts", 0))
        raise


def _one_candidate(cfg, llm, deduper, pick, gen_prompt, judge_prompt,
                   rejected_prompt, pair_judge_prompt, tool_names, tools,
                   fout, fman, accepted, rng) -> tuple[bool, str, dict]:
    """生成一条候选并过闸; 通过则落盘。返回 (是否接受, 拒因, judge meta)。"""
    seed = pick()
    rejected: Any = None
    error_type: str | None = None
    if cfg.task_type in ("qa", "qa_multi"):
        if cfg.task_type == "qa_multi":
            if not seed:  # gather 凑不齐 >=2 篇
                return False, "召集: 文献不足", {}
            chunk = {"content": src.build_evidence_text(seed)}
        else:
            chunk = seed
        source_docs = (
            [c.get("doc_id") or c.get("doc_name") for c in seed]
            if cfg.task_type == "qa_multi"
            else [seed.get("doc_id") or seed.get("doc_name")]
        )
        source_docs = [doc for doc in source_docs if doc not in (None, "")]
        item = gen.gen_qa(llm, chunk, gen_prompt, cfg.temperature)
        if not item:
            return False, "生成: 解析失败", {}
        ok, why = q.check_qa_schema(item)
        if not ok:
            return False, f"schema: {why}", {}
        ok, why, meta = q.judge_qa(llm, chunk, item, judge_prompt,
                                   cfg.judge_min_score, cfg.grounding_check,
                                   cfg.min_docs())
        if not ok:
            return False, why, meta
        if source_docs:
            meta = {**meta, "doc_ids": source_docs}
        if cfg.finetune_type == "dpo":
            rejected = gen.gen_qa_rejected(
                llm, chunk, item, rejected_prompt, cfg.temperature)
            if not rejected:
                return False, "rejected: 解析失败", meta
            ok, why, pair_meta = q.judge_preference_pair(
                llm, chunk["content"], item["answer"], rejected,
                pair_judge_prompt, cfg.judge_min_score)
            if not ok:
                return False, why, {**meta, "pair": pair_meta}
            meta = {**meta, "pair": pair_meta}
        key = item["question"]
    else:
        item = gen.gen_fc(llm, seed, gen_prompt, cfg.temperature)
        if not item:
            return False, "生成: 解析失败", {}
        ok, why = q.check_fc_schema(item["tool_calls"], tool_names)
        if not ok:
            return False, f"schema: {why}", {}
        ok, why = q.check_fc_entities(item["utterance"], item["tool_calls"])
        if not ok:
            return False, f"实体: {why}", {}
        ok, why, meta = q.judge_fc(llm, seed, item, judge_prompt, cfg.judge_min_score)
        if not ok:
            return False, why, meta
        item["history"] = seed.get("history", [])
        if cfg.finetune_type == "dpo":
            error_type = rng.choice(("wrong_tool", "wrong_args", "direct_answer"))
            rejected_seed = {**seed, "tool_names": sorted(tool_names)}
            rejected = gen.gen_fc_rejected(
                llm, rejected_seed, item, rejected_prompt, cfg.temperature, error_type)
            if not rejected:
                return False, "rejected: 解析失败", meta
            ok, why = q.check_fc_rejected(
                item["tool_calls"], rejected, error_type, tool_names)
            if not ok:
                return False, f"rejected: {why}", meta
            chosen_message = gen.fc_tool_message(item["tool_calls"])
            context = f"原发话: {seed['utterance']}\n改写发话: {item['utterance']}"
            ok, why, pair_meta = q.judge_preference_pair(
                llm, context, chosen_message, rejected,
                pair_judge_prompt, cfg.judge_min_score)
            if not ok:
                return False, why, {**meta, "pair": pair_meta, "error_type": error_type}
            meta = {**meta, "pair": pair_meta, "error_type": error_type}
        key = item["utterance"]

    if deduper.is_dup(key):
        return False, "去重: 命中", meta

    record = _build_record(
        cfg.finetune_type,
        cfg.task_type,
        item,
        tools,
        rejected,
        tags=_slice_tags(cfg.task_type, item, meta, error_type),
    )
    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
    fout.flush()
    fman.write(json.dumps({"index": accepted, "finetune_type": cfg.finetune_type,
                           "task_type": cfg.task_type, "error_type": error_type,
                           "score": meta.get("score"), "doc_ids": meta.get("doc_ids"),
                           "used_docs": meta.get("used_docs")}, ensure_ascii=False) + "\n")
    fman.flush()
    return True, "", meta


def _finalize(d: Path, cfg, accepted: int, attempts: int, rejects: dict) -> None:
    """汇总 output.jsonl → output.json (ShareGPT 数组) + report.md。"""
    records = []
    p = d / "output.jsonl"
    if p.exists():
        for line in p.read_text("utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
    _atomic_write(d / "output.json",
                  json.dumps(records, ensure_ascii=False, indent=2))

    rate = f"{accepted / attempts * 100:.1f}%" if attempts else "N/A"
    lines = [
        f"# 数据生成报告 ({cfg.finetune_type.upper()} / {cfg.task_type})", "",
        f"- 目标: {cfg.count}", f"- 接受: {accepted}", f"- 尝试: {attempts}",
        f"- 接受率: {rate}", "", "## 拒绝原因", "",
        "| 原因 | 次数 |", "|---|---|",
    ]
    for k, v in sorted(rejects.items(), key=lambda x: -x[1]):
        lines.append(f"| {k} | {v} |")
    (d / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("用法: python -m llamafactory_pipeline.datagen_run <job_dir>", file=sys.stderr)
        sys.exit(2)
    run_job(sys.argv[1])
