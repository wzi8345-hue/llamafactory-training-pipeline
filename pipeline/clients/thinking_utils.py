"""从 LLM 原始输出中剥离思考块, 供 JSON / tool_call 解析前使用。

Qwen / DeepSeek 等推理模型可能输出:
  - `` (vLLM reasoning_parser)
  - `` (部分后端/截断场景)

支持三种边界情况 (与 agentic 历史行为对齐):
  1. 完整闭合块 — 整段删除
  2. 仅有闭合标签 (开头被 max_tokens 截断) — 保留闭合标签之后的内容
  3. 仅有开始标签 (未闭合) — 保留开始标签之前的内容
"""

from __future__ import annotations

import re
from typing import Iterable, Iterator

# 完整块: 支持 open/close 标签混用 (部分模型输出 <think>...</think>)
_COMPLETE_BLOCK_RES = (
    re.compile(
        r"<redacted_thinking\b[^>]*>[\s\S]*?</think>",
        re.IGNORECASE,
    ),
    re.compile(
        r"<think\b[^>]*>[\s\S]*?</think\s*>",
        re.IGNORECASE,
    ),
    re.compile(
        r"<think\b[^>]*>[\s\S]*?</think>",
        re.IGNORECASE,
    ),
    re.compile(
        r"<redacted_thinking\b[^>]*>[\s\S]*?</think\s*>",
        re.IGNORECASE,
    ),
)

_CLOSE_RES = (
    re.compile(r"</think>", re.IGNORECASE),
    re.compile(r"</think\s*>", re.IGNORECASE),
)

_OPEN_RES = (
    re.compile(r"<redacted_thinking\b[^>]*>", re.IGNORECASE),
    re.compile(r"<think\b[^>]*>", re.IGNORECASE),
)


def strip_think_blocks(text: str) -> str:
    """剥离思考块; 截断场景下尽量 salvage 后面的 JSON / tool_call。"""
    if not text:
        return text

    cleaned = text
    # 多轮替换, 处理嵌套或连续多块
    while True:
        prev = cleaned
        for pat in _COMPLETE_BLOCK_RES:
            cleaned = pat.sub("", cleaned)
        if cleaned == prev:
            break

    # 开头被截断: 第一个闭合标签之后才是有效 payload
    for close_pat in _CLOSE_RES:
        close_match = close_pat.search(cleaned)
        if close_match:
            cleaned = cleaned[close_match.end():]
            break

    # 末尾未闭合: 开始标签之前才是有效 payload
    for open_pat in _OPEN_RES:
        open_match = open_pat.search(cleaned)
        if open_match:
            cleaned = cleaned[:open_match.start()]
            break

    return cleaned.strip()


# ---------------------------------------------------------------------------
# 流式版本
# ---------------------------------------------------------------------------

_OPEN_ANY_RE = re.compile(r"<(?:think|redacted_thinking)\b[^>]*>", re.IGNORECASE)
_CLOSE_TAG = "</think>"
# 未闭合的 "<" 最多缓冲这么多字符; 超过就判定它只是正文里的小于号, 直接放行,
# 免得 "a < b" 这类文本把后续输出一直卡在缓冲区里。
_MAX_TAG_LOOKAHEAD = 64


def strip_think_stream(pieces: Iterable[str]) -> Iterator[str]:
    """``strip_think_blocks`` 的流式版: 边收边吐正文, 吞掉 <think>...</think>。

    必须逐块做状态机而不能对单块调 ``strip_think_blocks``: 标签会被切在两个 chunk
    之间 (``"<thi"`` + ``"nk>"``), 单块正则一个也匹配不上, 标签会原样漏给前端。
    未闭合的 think 块 (被 max_tokens 截断) 整段丢弃, 与非流式行为一致。
    """
    buf = ""
    in_think = False
    for piece in pieces:
        if not piece:
            continue
        buf += piece
        while buf:
            if in_think:
                i = buf.find(_CLOSE_TAG)
                if i < 0:
                    # 只保留可能是半个闭合标签的尾巴
                    buf = buf[-(len(_CLOSE_TAG) - 1):]
                    break
                buf = buf[i + len(_CLOSE_TAG):]
                in_think = False
                continue
            m = _OPEN_ANY_RE.search(buf)
            if m:
                if m.start():
                    yield buf[:m.start()]
                buf = buf[m.end():]
                in_think = True
                continue
            # 尾部可能是被切开的开始标签: 从最后一个未闭合的 "<" 起留到下一块。
            lt = buf.rfind("<")
            hold = lt if (lt >= 0 and ">" not in buf[lt:]
                          and len(buf) - lt <= _MAX_TAG_LOOKAHEAD) else len(buf)
            if hold:
                yield buf[:hold]
            buf = buf[hold:]
            break
    if buf and not in_think:
        yield buf
