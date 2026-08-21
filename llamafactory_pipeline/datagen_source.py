"""生成源: QA 从本地 chunk 采样, FC 从现有 SFT 数据采种子。"""

from __future__ import annotations

import json
import random
from collections import OrderedDict
from pathlib import Path
from typing import Any, Optional

_QA_TYPES = {"text", "table"}
_CACHE_MAX = 8  # 同时驻留的 chunk 文件数; 大知识库跑久不膨胀


class QASource:
    """QA chunk 惰性采样器。

    知识库常有数千个 knowledge_blocks_vec.json (每个含 2560 维向量), 全量加载
    会吃光内存/耗时数分钟。生成只需随机取少量 chunk, 故按需采样: 随机选一个文件
    加载 (缓存), 从中随机取一条合格 chunk。只保留 content, 不留没用到的向量。
    缓存有界 (LRU, 默认 8 文件), 避免跑久后缓存累积膨胀。
    """

    def __init__(self, source_dir: str, min_len: int = 200):
        root = Path(source_dir)
        if not root.exists():
            raise FileNotFoundError(f"源目录不存在: {source_dir}")
        self.paths = list(root.rglob("knowledge_blocks_vec.json"))
        self.min_len = min_len
        self._cache: OrderedDict[Path, list[dict[str, Any]]] = OrderedDict()

    def __len__(self) -> int:
        return len(self.paths)

    def _load(self, path: Path) -> list[dict[str, Any]]:
        if path in self._cache:
            self._cache.move_to_end(path)  # LRU: 最近访问挪到尾
            return self._cache[path]
        chunks: list[dict[str, Any]] = []
        try:
            rows = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            rows = []
        for r in rows if isinstance(rows, list) else []:
            content = (r.get("content") or "").strip()
            if r.get("type") in _QA_TYPES and len(content) >= self.min_len:
                chunks.append({"content": content, "section": r.get("section", ""),
                               "doc_name": path.parent.name})
        self._cache[path] = chunks
        while len(self._cache) > _CACHE_MAX:  # LRU 淘汰最久未访问
            self._cache.popitem(last=False)
        return chunks

    def sample(self, rng: random.Random) -> dict[str, Any]:
        """随机取一条合格 chunk; 连续多个空文件则报错 (源无合格内容)。"""
        if not self.paths:
            raise ValueError("源目录下没有 knowledge_blocks_vec.json")
        for _ in range(min(100, len(self.paths) * 2)):
            chunks = self._load(rng.choice(self.paths))
            if chunks:
                return rng.choice(chunks)
        raise ValueError("采样不到合格 chunk (检查 min_len 或源内容)")


class MilvusEvidenceSource:
    """跨文献证据召集器 (qa_multi 用)。

    随机单位向量锚定一个主题簇 (无需 embedder), 取其 top1 真实 chunk 的向量做
    邻居检索, 从候选池里每个不同 doc 取一条合格 chunk, 凑够 n_docs 篇。
    凑不齐 >=2 篇不同文献则返回 [] (本次尝试作废)。
    """

    def __init__(self, client, collection: str, dim: int, n_docs: int = 3,
                 neighbor_top_k: int = 20, min_len: int = 200):
        self.client = client
        self.collection = collection
        self.dim = dim
        self.n_docs = n_docs
        self.neighbor_top_k = neighbor_top_k
        self.min_len = min_len
        self._sp = {"metric_type": "COSINE"}

    def _search(self, vec: list[float], limit: int, fields: list[str]):
        return self.client.search(
            collection_name=self.collection, data=[vec], anns_field="embedding",
            limit=limit, output_fields=fields, search_params=self._sp)[0]

    def gather(self, rng: random.Random) -> list[dict[str, Any]]:
        anchor = self._search(_rand_unit(rng, self.dim), 1, ["embedding"])
        if not anchor:
            return []
        pool = self._search(anchor[0]["entity"]["embedding"], self.neighbor_top_k,
                            ["doc_id", "doc_name", "content", "type"])
        picked: list[dict[str, Any]] = []
        seen: set[str] = set()
        for h in pool:
            e = h["entity"]
            content = (e.get("content") or "").strip()
            if e.get("type") not in _QA_TYPES or len(content) < self.min_len:
                continue
            if e.get("doc_id") in seen:
                continue
            seen.add(e.get("doc_id"))
            picked.append({"content": content, "doc_id": e.get("doc_id"),
                           "doc_name": e.get("doc_name", "")})
            if len(picked) >= self.n_docs:
                break
        return picked if len(picked) >= 2 else []


def _rand_unit(rng: random.Random, dim: int) -> list[float]:
    v = [rng.gauss(0, 1) for _ in range(dim)]
    n = sum(x * x for x in v) ** 0.5 or 1.0
    return [x / n for x in v]


def build_evidence_text(chunks: list[dict[str, Any]]) -> str:
    """把多篇 chunk 拼成带 [文献N] 标注的证据块 (复用 gen_qa/judge_qa 的 content)。"""
    return "\n\n".join(f"[文献{i + 1}]\n{c['content']}" for i, c in enumerate(chunks))


def load_fc_seeds(seed_file: str) -> tuple[list[dict[str, Any]], list[Any]]:
    """加载 FC 种子数据。返回 (种子记录列表, tools schema)。

    每条种子提取: 原发话 + 末轮 tool_calls (标签) + 标签之前的完整历史。
    tools 取首条记录的 tools (全数据集一致)。
    """
    data = json.loads(Path(seed_file).read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError("FC 种子文件应为非空 JSON 数组")
    tools = data[0].get("tools", [])
    seeds: list[dict[str, Any]] = []
    for rec in data:
        conv = rec.get("conversations", [])
        tool_calls = _last_tool_calls(conv)
        utterance = _last_human_before_tool(conv)
        if tool_calls and utterance:
            seeds.append({"utterance": utterance, "tool_calls": tool_calls,
                          "history": _history_before_last_tool(conv)})
    if not seeds:
        raise ValueError("FC 种子中未解析出任何 (发话, tool_calls) 对")
    return seeds, tools


def _last_tool_calls(conv: list[dict[str, Any]]) -> Optional[list[Any]]:
    for turn in reversed(conv):
        if turn.get("from") == "gpt" and turn.get("tool_calls"):
            return turn["tool_calls"]
    return None


def _last_human_before_tool(conv: list[dict[str, Any]]) -> str:
    """末轮 tool_call 对应的用户发话 = 最后一个 human turn 的 value。"""
    for turn in reversed(conv):
        if turn.get("from") == "human" and (turn.get("value") or "").strip():
            return turn["value"].strip()
    return ""


def _history_before_last_tool(conv: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """返回最终带 tool_calls 的 assistant 标签之前的所有对话轮。"""
    label_index = None
    for i in range(len(conv) - 1, -1, -1):
        turn = conv[i]
        if turn.get("from") == "gpt" and turn.get("tool_calls"):
            label_index = i
            break
    if label_index is None:
        return []
    return [dict(turn) for turn in conv[:label_index]]


def sample(items: list, rng: random.Random):
    return rng.choice(items)
