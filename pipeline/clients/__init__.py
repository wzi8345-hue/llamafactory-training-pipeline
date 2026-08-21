"""API 客户端包: 封装与外部服务的通信。"""

from .mineru import MinerUClient
from .uniparser import UniParserClient
from .embedding import EmbeddingClient
from .llm import LLMClient
from .milvus import MilvusIngester
from .client_registry import (
    ClientRegistry,
    get_global_registry,
    set_global_registry,
    reset_global_registry,
)

__all__ = [
    "MinerUClient",
    "UniParserClient",
    "EmbeddingClient",
    "LLMClient",
    "MilvusIngester",
    "ClientRegistry",
    "get_global_registry",
    "set_global_registry",
    "reset_global_registry",
]
