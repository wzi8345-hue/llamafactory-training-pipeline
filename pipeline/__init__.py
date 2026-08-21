"""DP-RAG Pipeline: 从 PDF 解析到 RAG 生成的端到端流水线。

模块结构:
- clients/    外部服务客户端 (MinerU, Embedding, LLM, Milvus)
- processors/  数据处理 (chunker, vectorizer)
- retrieval/   检索系统 (retrievers, context_builder, agentic)
- steps/       Pipeline 步骤 (parse, chunk, embed, store, retrieve, generate)
- flows/       高级流程 (ingest, query)
- models/      Pydantic 数据模型 (仅验证大模型输出)
- config/      配置管理
"""

from __future__ import annotations

import os as _os

# ---------------------------------------------------------------------------
# 抑制 milvus-lite 的 gRPC keepalive ping 噪音
#   症状: 长时间 idle (例如 LLM 生成 60s+) 后, milvus-lite 服务端会发
#         GOAWAY ENHANCE_YOUR_CALM "too_many_pings", grpc client 被迫重连,
#         产生告警日志 + 几百 ms 重连开销.
#   修复: 把 grpc 客户端 keepalive_time 调到 5min, 远超任何 LLM 生成时间;
#         禁用 idle 状态的 ping (PERMIT_WITHOUT_CALLS=0).
# 必须在任何 grpc/pymilvus import 之前设置, 故放在包 __init__ 顶部.
_os.environ.setdefault("GRPC_KEEPALIVE_TIME_MS", "300000")
_os.environ.setdefault("GRPC_KEEPALIVE_TIMEOUT_MS", "20000")
_os.environ.setdefault("GRPC_KEEPALIVE_PERMIT_WITHOUT_CALLS", "0")

__version__ = "0.2.1"

# 顶层便利导入
from .config import Config, load_config, get_config
from .pipeline import Pipeline

# Pydantic 模型: 仅用于验证大模型输出
from .models import RouteDecision, LLMChatResponse, QueryResult

# 非模型输出的数据结构 (dataclass)
from .flows.ingest import IngestResult
from .flows.query import ChatSession
from .retrieval.retrievers import Hit
from .retrieval.agentic import LocalRetrieveResult

# 流程类
from .flows import IngestFlow, QueryFlow

__all__ = [
    "Pipeline",
    "Config",
    "load_config",
    "get_config",
    "IngestFlow",
    "QueryFlow",
    # Pydantic (大模型输出)
    "RouteDecision",
    "LLMChatResponse",
    "QueryResult",
    # Dataclass (非模型输出)
    "IngestResult",
    "ChatSession",
    "Hit",
    "LocalRetrieveResult",
]
