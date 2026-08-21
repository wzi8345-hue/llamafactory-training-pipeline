"""LLM Chat Completions 客户端: OpenAI 兼容接口。

从原始 llm_client.py 搬入, 逻辑完全保留。
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, Iterator, List, Optional, Tuple

import requests

from ..models import LLMChatResponse
from .thinking_utils import strip_think_blocks, strip_think_stream

logger = logging.getLogger(__name__)


def _safe_resp_text(resp: "requests.Response", limit: int = 300) -> str:
    """按 UTF-8 解码响应体, 失败时回退到 latin-1, 避免中文报错乱码。"""
    try:
        text = resp.content.decode("utf-8", errors="replace")
    except Exception:
        text = resp.text
    return text[:limit]


def _flatten_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if isinstance(item.get("text"), str):
                    parts.append(item["text"])
                elif item.get("type") == "text" and isinstance(item.get("content"), str):
                    parts.append(item["content"])
        return "".join(parts)
    return str(content)


def _extract_message_text(data: Dict[str, Any]) -> str:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""

    first = choices[0] if isinstance(choices[0], dict) else {}
    message = first.get("message") if isinstance(first.get("message"), dict) else {}
    content = message.get("content")
    answer = _flatten_content(content)
    if answer:
        return answer

    delta = first.get("delta") if isinstance(first.get("delta"), dict) else {}
    answer = _flatten_content(delta.get("content"))
    if answer:
        return answer

    text = first.get("text")
    return _flatten_content(text)


def _extract_message_reasoning(data: Dict[str, Any]) -> str:
    """抽取推理/思考增量 (delta.reasoning / delta.reasoning_content)。

    本地 vLLM 等推理后端开启 thinking 后, 会把"思考过程"放在 reasoning 字段,
    与正文 content 分流下发 (一帧通常只有其一)。用于把思考与真正输出分开展示。
    """
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0] if isinstance(choices[0], dict) else {}
    delta = first.get("delta") if isinstance(first.get("delta"), dict) else {}
    for key in ("reasoning", "reasoning_content"):
        val = delta.get(key)
        if isinstance(val, str) and val:
            return val
    message = first.get("message") if isinstance(first.get("message"), dict) else {}
    for key in ("reasoning", "reasoning_content"):
        val = message.get(key)
        if isinstance(val, str) and val:
            return val
    return ""


def _extract_finish_reason(data: Dict[str, Any]) -> str:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0] if isinstance(choices[0], dict) else {}
    return str(first.get("finish_reason") or "")


def _request_id(resp: "requests.Response") -> str:
    headers = getattr(resp, "headers", {}) or {}
    for key in ("x-request-id", "request-id", "x-correlation-id"):
        value = headers.get(key)
        if value:
            return str(value)
    return ""


def _is_retryable_status(status_code: int) -> bool:
    """只重试限流和服务端瞬态错误；配置/鉴权等 4xx 立即失败。"""
    return status_code == 429 or status_code >= 500


# 默认配置
DEFAULT_LLM_API_BASE = "https://api.gpugeek.com/v1"
DEFAULT_LLM_MODEL = "DeepSeek/DeepSeek-V3-0324"
# 不在代码里硬编码 api_key, 必须通过 config / 环境变量传入
DEFAULT_LLM_API_KEY = ""
DEFAULT_TEMPERATURE = 0
DEFAULT_MAX_TOKENS = 2048


def _apply_thinking_control(
    payload: Dict[str, Any], messages: List[Dict[str, str]],
    *,
    disable_thinking: bool = False,
    use_extra_body: bool = False,
) -> List[Dict[str, str]]:
    """控制推理模型的"思考模式"。

    当 use_extra_body=True (vLLM 启动的本地模型服务):
      - 在请求体顶层下发 chat_template_kwargs.enable_thinking = not disable_thinking
      - 重要: 这里用裸 HTTP, 必须把 chat_template_kwargs 直接放在 payload 根上;
        OpenAI Python SDK 风格的 ``extra_body`` 包装服务端不会自动解包, 等于没下发.
      - 不再追加 /no_think 文本, 避免污染 prompt.

    当 use_extra_body=False (阿里云 / DeepSeek / GPUGeek 等云平台):
      - 不下发任何 vLLM 专属字段.
      - 仅当 disable_thinking=True 时, 在最后一条 user message 末尾追加 /no_think,
        兼容 Qwen3 / DeepSeek-R1 等支持该指令的推理模型.

    返回新的 messages 列表 (拷贝, 不修改原引用)。
    """
    # vLLM 模式: chat_template_kwargs 必须放在请求体顶层
    if use_extra_body:
        ctk = payload.get("chat_template_kwargs")
        if not isinstance(ctk, dict):
            ctk = {}
        ctk["enable_thinking"] = not disable_thinking
        payload["chat_template_kwargs"] = ctk

    new_messages: List[Dict[str, str]] = []
    last_user_idx = -1
    for i, m in enumerate(messages):
        new_messages.append(dict(m))
        if m.get("role") == "user":
            last_user_idx = i

    # /no_think 文本: 仅在云平台模式 + 显式 disable_thinking=True 时使用
    if disable_thinking and not use_extra_body:
        if last_user_idx >= 0:
            content = new_messages[last_user_idx].get("content", "")
            if isinstance(content, str) and "/no_think" not in content:
                new_messages[last_user_idx]["content"] = content.rstrip() + " /no_think"

    return new_messages


class LLMClient:
    """GPUGeek / 任意 OpenAI 兼容服务的 chat completions 客户端。"""

    def __init__(
        self,
        api_base: str = DEFAULT_LLM_API_BASE,
        model: str = DEFAULT_LLM_MODEL,
        api_key: str = DEFAULT_LLM_API_KEY,
        timeout: int = 120,
        max_retries: int = 3,
        disable_thinking_extra_body: bool = False,
    ) -> None:
        if not api_key:
            raise ValueError(
                "未提供 LLM API key, 请设置配置或传入 api_key 参数"
            )
        self.api_base = api_base.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        # 仅 vLLM 后端需要 extra_body / enable_thinking; 其他后端 (GPUGeek 等) 不支持
        self._disable_thinking_extra_body = disable_thinking_extra_body
        # ClientRegistry 会复用 LLMClient；Session 因而可以稳定复用 HTTP 连接池。
        self._session = requests.Session()

    def close(self) -> None:
        self._session.close()

    def _headers(self) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def chat(
        self,
        system: str,
        user: str,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        stream: bool = False,
        disable_thinking: bool = False,
        **extra: Any,
    ) -> Dict[str, Any]:
        """非流式: 一次返回完整答案。

        Args:
            disable_thinking: 关闭"思考模式" (适用于 Qwen3 / DeepSeek-R1 等推理模型),
                避免 <think>...</think> 占满 max_tokens 导致输出被截断。

        返回: {"answer": str, "raw": dict, "usage": dict | None}
        """
        url = f"{self.api_base}/chat/completions"
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        payload.update(extra)
        if disable_thinking or self._disable_thinking_extra_body:
            payload["messages"] = _apply_thinking_control(
                payload, messages,
                disable_thinking=disable_thinking,
                use_extra_body=self._disable_thinking_extra_body,
            )

        last_err: Optional[str] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self._session.post(
                    url, headers=self._headers(), json=payload, timeout=self.timeout,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    answer = strip_think_blocks(_extract_message_text(data))
                    return {
                        "answer": answer,
                        "raw": data,
                        "usage": data.get("usage"),
                        "finish_reason": _extract_finish_reason(data),
                        "request_id": _request_id(resp),
                    }
                last_err = f"HTTP {resp.status_code}: {_safe_resp_text(resp)}"
                if not _is_retryable_status(resp.status_code):
                    break
            except Exception as e:
                last_err = str(e)
            if attempt < self.max_retries:
                wait = 2 ** attempt
                logger.debug(
                    f"  [retry {attempt}/{self.max_retries}] {last_err} -> wait {wait}s"
                )
                time.sleep(wait)
        raise RuntimeError(f"LLM 请求失败: {last_err}")

    def chat_stream(
        self,
        system: str,
        user: str,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        disable_thinking: bool = False,
        **extra: Any,
    ) -> Iterator[str]:
        """流式: yield 文本增量片段 (SSE), 已剥离 <think> 块。"""
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        return self.chat_messages_stream(
            messages, temperature=temperature, max_tokens=max_tokens,
            disable_thinking=disable_thinking, **extra,
        )

    def _iter_sse_text(self, payload: Dict[str, Any]) -> Iterator[str]:
        """POST 一次流式请求, yield content 增量 (不做任何清洗)。"""
        finish_reason = ""
        malformed_chunks = 0
        with self._session.post(
            f"{self.api_base}/chat/completions",
            headers=self._headers(), json=payload,
            timeout=self.timeout, stream=True,
        ) as resp:
            if resp.status_code != 200:
                raise RuntimeError(
                    f"LLM 流式请求失败 HTTP {resp.status_code}: {_safe_resp_text(resp)}"
                )
            # SSE 流的 Content-Type 通常是 text/event-stream 不带 charset, requests
            # 会回退到 ISO-8859-1 导致中文乱码. 强制按 UTF-8 解码.
            resp.encoding = "utf-8"
            for raw_line in resp.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue
                line = raw_line.strip()
                if not line.startswith("data:"):
                    continue
                data_str = line[len("data:"):].strip()
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    malformed_chunks += 1
                    continue
                finish_reason = _extract_finish_reason(chunk) or finish_reason
                yield_text = _extract_message_text(chunk)
                if yield_text:
                    yield yield_text
        if malformed_chunks:
            logger.warning(f"LLM SSE 忽略了 {malformed_chunks} 个非法 JSON chunk")
        if finish_reason == "length":
            raise RuntimeError("LLM 流式输出被 max_tokens 截断")

    def chat_stream_events(
        self,
        system: str,
        user: str,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        disable_thinking: bool = False,
        **extra: Any,
    ) -> Iterator[Tuple[str, str]]:
        """流式: 把"思考过程"与"正文"分两路 yield。

        产出 ("thinking", piece) 表示推理增量 (reasoning), ("answer", piece) 表示正文增量
        (content)。供上层将专家模式的思考过程与真正输出分开展示。

        注: 需开启思考模式 (disable_thinking=False) 才会有 thinking 增量; vLLM 后端
        要求启动时配 --reasoning-parser, 否则思考会混在 content 里。
        """
        url = f"{self.api_base}/chat/completions"
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        payload.update(extra)
        if disable_thinking or self._disable_thinking_extra_body:
            payload["messages"] = _apply_thinking_control(
                payload, messages,
                disable_thinking=disable_thinking,
                use_extra_body=self._disable_thinking_extra_body,
            )

        finish_reason = ""
        with self._session.post(
            url, headers=self._headers(), json=payload,
            timeout=self.timeout, stream=True,
        ) as resp:
            if resp.status_code != 200:
                raise RuntimeError(
                    f"LLM 流式请求失败 HTTP {resp.status_code}: {_safe_resp_text(resp)}"
                )
            resp.encoding = "utf-8"
            for raw_line in resp.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue
                line = raw_line.strip()
                if not line.startswith("data:"):
                    continue
                data_str = line[len("data:"):].strip()
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                finish_reason = _extract_finish_reason(chunk) or finish_reason
                reasoning = _extract_message_reasoning(chunk)
                if reasoning:
                    yield ("thinking", reasoning)
                answer = _extract_message_text(chunk)
                if answer:
                    yield ("answer", answer)
        if finish_reason == "length":
            raise RuntimeError("LLM 流式输出被 max_tokens 截断")

    def chat_messages(
        self,
        messages: List[Dict[str, str]],
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        disable_thinking: bool = False,
        **extra: Any,
    ) -> Dict[str, Any]:
        """非流式: 接受完整 messages 列表 (支持多轮对话)。

        Args:
            messages: OpenAI 格式消息列表, 如 [{"role":"system","content":...}, {"role":"user","content":...}, ...]
            disable_thinking: 关闭"思考模式" (适用于 Qwen3 / DeepSeek-R1 等推理模型),
                避免 <think>...</think> 占满 max_tokens 导致输出被截断。

        返回: {"answer": str, "raw": dict, "usage": dict | None}
        """
        url = f"{self.api_base}/chat/completions"
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        payload.update(extra)
        if disable_thinking or self._disable_thinking_extra_body:
            payload["messages"] = _apply_thinking_control(
                payload, messages,
                disable_thinking=disable_thinking,
                use_extra_body=self._disable_thinking_extra_body,
            )

        last_err: Optional[str] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self._session.post(
                    url, headers=self._headers(), json=payload, timeout=self.timeout,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    answer = strip_think_blocks(_extract_message_text(data))
                    return {
                        "answer": answer,
                        "raw": data,
                        "usage": data.get("usage"),
                        "finish_reason": _extract_finish_reason(data),
                        "request_id": _request_id(resp),
                    }
                last_err = f"HTTP {resp.status_code}: {_safe_resp_text(resp)}"
                if not _is_retryable_status(resp.status_code):
                    break
            except Exception as e:
                last_err = str(e)
            if attempt < self.max_retries:
                wait = 2 ** attempt
                logger.debug(
                    f"  [retry {attempt}/{self.max_retries}] {last_err} -> wait {wait}s"
                )
                time.sleep(wait)
        raise RuntimeError(f"LLM 请求失败: {last_err}")

    def chat_messages_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        disable_thinking: bool = False,
        **extra: Any,
    ) -> Iterator[str]:
        """流式: 接受完整 messages 列表 (支持多轮对话), yield 文本增量片段。

        推理模型 (Qwen3.x / DeepSeek-R1) 在服务端未配 ``--reasoning-parser`` 时,
        会把 <think>...</think> 直接写进 content —— 这里统一剥掉, 免得每个调用方
        各自补 (漏一个就是整段思考过程流到前端)。
        """
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        payload.update(extra)
        if disable_thinking or self._disable_thinking_extra_body:
            payload["messages"] = _apply_thinking_control(
                payload, messages,
                disable_thinking=disable_thinking,
                use_extra_body=self._disable_thinking_extra_body,
            )
        return strip_think_stream(self._iter_sse_text(payload))

    def chat_validated(
        self,
        system: str,
        user: str,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        **extra: Any,
    ) -> LLMChatResponse:
        """非流式, 返回 Pydantic 验证后的 LLMChatResponse。"""
        raw = self.chat(
            system=system, user=user,
            temperature=temperature, max_tokens=max_tokens,
            **extra,
        )
        return LLMChatResponse(answer=raw["answer"], usage=raw.get("usage"))

    # ── Function Calling 支持 (供 pipeline.routing 模块使用) ─────────────

    def chat_with_tools(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict[str, Any]],
        *,
        tool_choice: Any = "required",
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        disable_thinking: bool = False,
        parallel_tool_calls: Optional[bool] = None,
        **extra: Any,
    ) -> Dict[str, Any]:
        """非流式 chat completions, 携带 tools 与 tool_choice 走 function calling。

        Args:
            messages: 完整 OpenAI 格式 messages 列表 (含 system/user/history)
            tools: OpenAI 兼容的 tools 数组, 每项形如
                {"type": "function", "function": {"name": "...", "description": "...",
                "parameters": <JSON Schema>}}
            tool_choice: "auto" / "required" / "none" / {"type":"function","function":{"name":"..."}}.
                "required" 强制 LLM 必须调用 tools 之一 (本项目的默认选择).
            parallel_tool_calls: 仅 OpenAI / 部分兼容后端支持; 单工具决策场景建议显式设 False 节省 token.
                None 表示不下发该字段, 使用后端默认.
            disable_thinking: 关闭推理模型 (Qwen3 / DeepSeek-R1) 的思考模式, 避免 <think/> 吃掉 max_tokens.

        返回:
            {
              "answer": "<message.content 文本, 通常为空>",
              "raw":    <OpenAI 完整响应字典>,
              "usage":  <token 统计>,
              "tool_calls": [<message.tool_calls 数组原样>],  # 如果存在
              "message": <message 字典原样>,                  # 方便 fc_parser 直接消费
              "reasoning_content": "<vLLM reasoning parser 抽出的 <think> 块, 可能为空>",
            }

        不在此层做 tool_calls 解析; 解析交给 pipeline.routing.fc_parser, 因为不同后端
        (OpenAI / vLLM / Qwen <tool_call> 文本块) 的格式差异大, 解析逻辑集中维护更清晰.

        reasoning_content 用于 **隐式 CoT**: 当 vLLM 启动时配了 --reasoning-parser, 模型在
        <think>...</think> 块里的隐式思考会被剥到 message.reasoning_content (OpenAI 标准
        响应里没有该字段, 但 vLLM/DeepSeek API/部分云厂商提供). 调用方可把该字段打到日志便于审计,
        但**不要**让它影响业务决策 (业务决策只从 tool_calls 里来).
        """
        url = f"{self.api_base}/chat/completions"
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
            "tools": tools,
            "tool_choice": tool_choice,
        }
        if parallel_tool_calls is not None:
            payload["parallel_tool_calls"] = bool(parallel_tool_calls)
        payload.update(extra)
        if disable_thinking or self._disable_thinking_extra_body:
            payload["messages"] = _apply_thinking_control(
                payload, messages,
                disable_thinking=disable_thinking,
                use_extra_body=self._disable_thinking_extra_body,
            )

        last_err: Optional[str] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self._session.post(
                    url, headers=self._headers(), json=payload, timeout=self.timeout,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    answer = _extract_message_text(data)
                    message: Dict[str, Any] = {}
                    tool_calls: List[Any] = []
                    reasoning_content: str = ""
                    finish_reason: str = ""
                    choices = data.get("choices") or []
                    if isinstance(choices, list) and choices:
                        first = choices[0] if isinstance(choices[0], dict) else {}
                        finish_reason = str(first.get("finish_reason") or "")
                        msg = first.get("message") if isinstance(first.get("message"), dict) else {}
                        if isinstance(msg, dict):
                            message = msg
                            tcs = msg.get("tool_calls")
                            if isinstance(tcs, list):
                                tool_calls = tcs
                            # vLLM (reasoning_parser 启用时) / DeepSeek API 会把 <think> 块
                            # 剥到 message.reasoning_content; 没启用就是空, 不影响业务。
                            rc = msg.get("reasoning_content")
                            if isinstance(rc, str):
                                reasoning_content = rc.strip()
                    return {
                        "answer": answer,
                        "raw": data,
                        "usage": data.get("usage"),
                        "tool_calls": tool_calls,
                        "message": message,
                        "reasoning_content": reasoning_content,
                        "finish_reason": finish_reason,
                        "request_id": _request_id(resp),
                    }
                last_err = f"HTTP {resp.status_code}: {_safe_resp_text(resp)}"
                if not _is_retryable_status(resp.status_code):
                    break
            except Exception as e:
                last_err = str(e)
            if attempt < self.max_retries:
                wait = 2 ** attempt
                logger.debug(
                    f"  [retry {attempt}/{self.max_retries}] {last_err} -> wait {wait}s"
                )
                time.sleep(wait)
        raise RuntimeError(f"LLM 请求 (tools) 失败: {last_err}")
