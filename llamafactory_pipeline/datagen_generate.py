"""生成策略: qa / fc。仅调用 LLM 产候选, 校验交给 datagen_quality。"""

from __future__ import annotations

import json
from typing import Any, Optional

from rag_eval_plan.common import extract_json_object


def gen_qa(llm, chunk: dict[str, Any], gen_prompt: str, temperature: float) -> Optional[dict[str, str]]:
    """依据 chunk 生成 {question, answer}; 解析失败或字段缺失返回 None。"""
    user = f"[文献片段]\n{chunk['content']}"
    raw = llm.chat(system=gen_prompt, user=user, temperature=temperature,
                   disable_thinking=True).get("answer", "")
    obj = extract_json_object(raw)
    q = (obj.get("question") or "").strip()
    a = (obj.get("answer") or "").strip()
    if not q or not a:
        return None
    return {"question": q, "answer": a}


def gen_fc(llm, seed: dict[str, Any], gen_prompt: str, temperature: float) -> Optional[dict[str, Any]]:
    """把种子发话改写成多样发话; tool_calls 直接沿用种子标签。"""
    label = _label_hint(seed["tool_calls"])
    user = f"[原始发话]\n{seed['utterance']}\n\n[意图标签]\n{label}"
    raw = llm.chat(system=gen_prompt, user=user, temperature=temperature,
                   disable_thinking=True).get("answer", "")
    obj = extract_json_object(raw)
    utterance = (obj.get("utterance") or "").strip()
    if not utterance:
        return None
    return {"utterance": utterance, "tool_calls": seed["tool_calls"]}


def gen_qa_rejected(llm, chunk: dict[str, Any], item: dict[str, str],
                    prompt: str, temperature: float) -> Optional[str]:
    """为已通过 chosen 闸的 QA 生成一个非空 rejected 回答。"""
    user = (f"[文献片段]\n{chunk['content']}\n\n[问题]\n{item['question']}\n\n"
            f"[chosen]\n{item['answer']}")
    raw = llm.chat(system=prompt, user=user, temperature=temperature,
                   disable_thinking=True).get("answer", "")
    rejected = (extract_json_object(raw).get("rejected") or "").strip()
    return rejected or None


def fc_tool_message(tool_calls: list[Any]) -> dict[str, str]:
    """把 OpenAI 风格 tool_calls 转成 LLaMAFactory 原生 function_call 消息。"""
    calls = []
    for tc in tool_calls:
        args = tc.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                pass
        calls.append({"name": tc.get("name"), "arguments": args})
    value: Any = calls[0] if len(calls) == 1 else calls
    return {"from": "function_call", "value": json.dumps(value, ensure_ascii=False)}


def gen_fc_rejected(llm, seed: dict[str, Any], item: dict[str, Any], prompt: str,
                    temperature: float, error_type: str) -> Optional[dict[str, str]]:
    """生成一种声明错误类型的 FC rejected，并转成原生消息结构。"""
    user = (f"[原始发话]\n{seed['utterance']}\n\n[改写发话]\n{item['utterance']}\n\n"
            f"[chosen]\n{fc_tool_message(item['tool_calls'])['value']}\n\n"
            f"[可用工具]\n{json.dumps(seed.get('tool_names', []), ensure_ascii=False)}\n\n"
            f"[错误类型]\n{error_type}")
    raw = llm.chat(system=prompt, user=user, temperature=temperature,
                   disable_thinking=True).get("answer", "")
    obj = extract_json_object(raw)
    if error_type == "direct_answer":
        answer = (obj.get("answer") or "").strip()
        return {"from": "gpt", "value": answer} if answer else None
    name = (obj.get("name") or "").strip()
    if not name or "arguments" not in obj:
        return None
    return fc_tool_message([{"name": name, "arguments": obj["arguments"]}])


def _label_hint(tool_calls: list[Any]) -> str:
    """把 tool_calls 压成简短标签提示 (工具名 + args), 供改写时保持意图。"""
    tc = tool_calls[0] if tool_calls else {}
    name = tc.get("name", "")
    args = tc.get("arguments", "")
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            pass
    return f"{name} {json.dumps(args, ensure_ascii=False)}"
