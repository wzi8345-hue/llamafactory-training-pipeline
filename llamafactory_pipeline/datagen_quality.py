"""质量四闸 + 语义去重。纯逻辑, LLM/embedder 由调用方注入 (便于单测)。"""

from __future__ import annotations

import json
import re
from typing import Any

from rag_eval_plan.common import extract_json_object

# 钢种牌号: Q345qENH / Q355NHB / Q450NQR1 / S355J2W 等。
_GRADE_RE = re.compile(r"[QS]\d{2,3}[A-Za-z0-9]*")


# ── schema 闸 ──

def check_qa_schema(item: dict[str, Any]) -> tuple[bool, str]:
    if (item.get("question") or "").strip() and (item.get("answer") or "").strip():
        return True, ""
    return False, "qa 字段缺失"


def check_fc_schema(tool_calls: list[Any], tool_names: set[str]) -> tuple[bool, str]:
    if not tool_calls:
        return False, "无 tool_calls"
    tc = tool_calls[0]
    if tc.get("name") not in tool_names:
        return False, f"未知工具 {tc.get('name')}"
    args = tc.get("arguments")
    if isinstance(args, str):
        try:
            json.loads(args)
        except json.JSONDecodeError:
            return False, "arguments 非合法 JSON"
    return True, ""


# ── 实体一致性闸 (FC) ──

def _grades(text: str) -> set[str]:
    return set(_GRADE_RE.findall(text or ""))


def check_fc_entities(utterance: str, tool_calls: list[Any]) -> tuple[bool, str]:
    """args 里的牌号必须原样出现在发话, 且发话不得臆造 args 没有的牌号 (双向一致)。"""
    tc = tool_calls[0] if tool_calls else {}
    args = tc.get("arguments", "")
    args_text = args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)
    want, got = _grades(args_text), _grades(utterance)
    missing = want - got
    if missing:
        return False, f"发话缺失实体 {sorted(missing)}"
    extra = got - want
    if extra:
        return False, f"发话臆造实体 {sorted(extra)}"
    return True, ""


# ── judge 闸 ──

def judge_qa(llm, chunk: dict[str, Any], item: dict[str, str], judge_prompt: str,
             min_score: int, grounding_check: bool, min_docs: int = 1) -> tuple[bool, str, dict]:
    user = (f"[文献片段]\n{chunk['content']}\n\n"
            f"[问题]\n{item['question']}\n\n[答案]\n{item['answer']}")
    raw = llm.chat(system=judge_prompt, user=user, temperature=0,
                   disable_thinking=True).get("answer", "")
    v = extract_json_object(raw)
    score = _as_int(v.get("score"))
    grounded = bool(v.get("grounded", True))
    if grounding_check and not grounded:
        return False, "judge: 未接地(幻觉)", v
    if min_docs > 1 and _distinct_docs(v.get("used_docs")) < min_docs:
        return False, f"judge: 覆盖不足(<{min_docs}篇)", v
    if score < min_score:
        return False, f"judge: 分数 {score}<{min_score}", v
    return True, "", v


def _distinct_docs(used: Any) -> int:
    if not isinstance(used, list):
        return 0
    return len({str(x).strip() for x in used if str(x).strip()})


def judge_fc(llm, seed: dict[str, Any], item: dict[str, Any], judge_prompt: str,
             min_score: int) -> tuple[bool, str, dict]:
    user = f"[原发话]\n{seed['utterance']}\n\n[改写发话]\n{item['utterance']}"
    raw = llm.chat(system=judge_prompt, user=user, temperature=0,
                   disable_thinking=True).get("answer", "")
    v = extract_json_object(raw)
    score = _as_int(v.get("score"))
    if score < min_score:
        return False, f"judge: 分数 {score}<{min_score}", v
    return True, "", v


def judge_preference_pair(llm, context: str, chosen: Any, rejected: Any,
                          prompt: str, min_score: int) -> tuple[bool, str, dict]:
    """确认 chosen 明显优于 rejected；相同内容先走本地规则闸。"""
    chosen_text = _preference_text(chosen)
    rejected_text = _preference_text(rejected)
    if chosen_text == rejected_text:
        return False, "pair judge: chosen/rejected 相同", {}
    user = (f"[上下文]\n{context}\n\n[chosen]\n{chosen_text}\n\n"
            f"[rejected]\n{rejected_text}")
    raw = llm.chat(system=prompt, user=user, temperature=0,
                   disable_thinking=True).get("answer", "")
    v = extract_json_object(raw)
    chosen_score = _as_int(v.get("chosen_score"))
    rejected_score = _as_int(v.get("rejected_score"))
    if v.get("chosen_better") is not True:
        return False, "pair judge: 未确认 chosen 偏好", v
    if chosen_score < min_score:
        return False, f"pair judge: chosen 分数 {chosen_score}<{min_score}", v
    if chosen_score <= rejected_score:
        return False, "pair judge: chosen/rejected 无分数差距", v
    return True, "", v


def _preference_text(message: Any) -> str:
    if isinstance(message, str):
        return message.strip()
    return json.dumps(message, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def check_fc_rejected(chosen_calls: list[Any], rejected_message: Any,
                      error_type: str, tool_names: set[str]) -> tuple[bool, str]:
    """验证 FC rejected 确实符合声明的错误类型，而不是伪劣空样本。"""
    if not isinstance(rejected_message, dict):
        return False, "rejected 不是消息对象"
    if error_type == "direct_answer":
        if rejected_message.get("from") != "gpt" or not (rejected_message.get("value") or "").strip():
            return False, "direct_answer 需要非空 gpt 文本"
        return True, ""
    if rejected_message.get("from") != "function_call":
        return False, f"{error_type} 需要 function_call 消息"
    rejected_call, why = _parse_function_message(rejected_message)
    if rejected_call is None:
        return False, why
    chosen_call = _normalized_call(chosen_calls[0] if chosen_calls else {})
    rejected_name = rejected_call.get("name")
    if rejected_name not in tool_names:
        return False, f"未知工具 {rejected_name}"
    if error_type == "wrong_tool":
        if rejected_name == chosen_call.get("name"):
            return False, "工具名未改变"
        return True, ""
    if error_type == "wrong_args":
        if rejected_name != chosen_call.get("name"):
            return False, "wrong_args 不得改变工具名"
        if rejected_call.get("arguments") == chosen_call.get("arguments"):
            return False, "参数未改变"
        return True, ""
    return False, f"未知错误类型 {error_type}"


def _parse_function_message(message: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
    try:
        value = json.loads(message.get("value", ""))
    except (TypeError, json.JSONDecodeError):
        return None, "function_call value 非合法 JSON"
    if isinstance(value, list):
        value = value[0] if len(value) == 1 else None
    if not isinstance(value, dict) or not value.get("name") or "arguments" not in value:
        return None, "function_call 字段缺失"
    return _normalized_call(value), ""


def _normalized_call(call: dict[str, Any]) -> dict[str, Any]:
    args = call.get("arguments")
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            pass
    return {"name": call.get("name"), "arguments": args}


def _as_int(x: Any) -> int:
    try:
        return int(round(float(x)))
    except (TypeError, ValueError):
        return 0


# ── 语义去重 ──

import numpy as _np


class Deduper:
    """完全重复走 hash; 近义走 embedding 余弦 (向量已归一化, 点积即余弦)。

    点积用 numpy 矩阵一次算, 替代纯 Python 双层循环 (5000 条省 ~1250 万次 Python 乘法)。
    """

    def __init__(self, embedder, threshold: float):
        self.embedder = embedder
        self.threshold = threshold
        self._hashes: set[str] = set()
        self._mat: _np.ndarray | None = None  # (n, dim) 已归一化向量堆叠
        self._dim: int | None = None

    def _append_vec(self, vec: list[float]) -> None:
        v = _np.asarray(vec, dtype=_np.float32)
        self._dim = v.shape[0]
        if self._mat is None:
            self._mat = v.reshape(1, -1)
        else:
            self._mat = _np.vstack([self._mat, v.reshape(1, -1)])

    def prime(self, text: str) -> None:
        """续跑时把已接受文本灌入去重集 (不判定)。"""
        self._hashes.add(text.strip())
        self._append_vec(self.embedder.embed(text))

    def prime_batch(self, texts: list[str]) -> None:
        """批量 prime: 若 embedder 支持 embed_batch 则一次调用, 否则逐条 (兼容旧 embedder)。"""
        texts = [t.strip() for t in texts]
        self._hashes.update(texts)
        batch_fn = getattr(self.embedder, "embed_batch", None)
        if batch_fn is not None:
            vecs = batch_fn(texts)  # list[list[float]]
            for v in vecs:
                self._append_vec(v)
        else:
            for t in texts:
                self._append_vec(self.embedder.embed(t))

    def is_dup(self, text: str) -> bool:
        t = text.strip()
        if t in self._hashes:
            return True
        vec = _np.asarray(self.embedder.embed(text), dtype=_np.float32)
        if self._mat is not None:
            # 矩阵乘一次算完与所有已有向量的点积; 向量归一化时即余弦
            sims = self._mat @ vec
            if float(sims.max()) > self.threshold:
                return True
        self._hashes.add(t)
        self._append_vec(vec.tolist())
        return False
