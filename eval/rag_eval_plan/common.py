"""共享工具: 环境加载、LLM/Milvus 客户端、JSON 提取。"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_COLLECTION = "literature_chunks"

# 生成/评测目标 doc: 默认取 Milvus 在库文献 (可用 --doc-ids 覆盖)。
CHUNK_FIELDS = [
    "pk", "chunk_id", "doc_id", "doc_name", "type", "section",
    "page_start", "publication_year", "content", "context",
]


def load_env(env_file: str = ".env.local") -> None:
    """把 .env.local 里的 export KEY=VAL 灌进 os.environ (仅补缺失项)。"""
    path = PROJECT_ROOT / env_file
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        line = line[len("export "):].strip() if line.startswith("export ") else line
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        os.environ.setdefault(key, val)


def _pipeline_config() -> Dict[str, Any]:
    import yaml
    p = PROJECT_ROOT / "local_api_config.yaml"
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_llm(timeout: int = 180, max_retries: int = 3):
    """按 local_api_config.generation 建立当前最终生成 LLM 客户端。"""
    load_env()
    sys.path.insert(0, str(PROJECT_ROOT))
    from pipeline.clients.llm import LLMClient

    gen = _pipeline_config().get("generation", {})
    api_base = gen.get("api_base", "http://localhost:8003/v1")
    final = gen.get("final") or {}
    api_base = final.get("api_base") or api_base
    model = final.get("model") or gen.get("model", "/model")
    api_key = os.environ.get("VLLM_API_KEY", "")
    if not api_key:
        raise ValueError("缺少 VLLM_API_KEY (应在 .env.local)")
    return LLMClient(
        api_base=api_base, model=model, api_key=api_key,
        timeout=timeout, max_retries=max_retries,
        disable_thinking_extra_body=True,
    )


def milvus_client(uri: str = "http://localhost:19530", token: str = ""):
    from pymilvus import MilvusClient
    kwargs: Dict[str, Any] = {"uri": uri}
    if token:
        kwargs["token"] = token
    return MilvusClient(**kwargs)


def fetch_corpus_chunks(
    client,
    collection: str = DEFAULT_COLLECTION,
    doc_ids: Optional[List[str]] = None,
    limit: int = 16000,
) -> List[Dict[str, Any]]:
    """拉取集合内 chunk 行 (可按 doc_id 过滤)。"""
    flt = ""
    if doc_ids:
        ors = " or ".join(f'doc_id == "{d}"' for d in doc_ids)
        flt = f"({ors})"
    return client.query(
        collection_name=collection, filter=flt,
        output_fields=CHUNK_FIELDS, limit=limit,
    )


def list_corpus_doc_ids(client, collection: str = DEFAULT_COLLECTION) -> List[str]:
    """枚举集合内全部 doc_id (大库用 query_iterator 绕开 16384 窗口限制)。"""
    seen: set[str] = set()
    try:
        it = client.query_iterator(collection_name=collection, filter="",
                                   output_fields=["doc_id"], batch_size=2000)
        while True:
            batch = it.next()
            if not batch:
                break
            for r in batch:
                if r.get("doc_id"):
                    seen.add(r["doc_id"])
        it.close()
    except Exception:
        rows = client.query(collection_name=collection, filter="",
                            output_fields=["doc_id"], limit=16000)
        seen = {r["doc_id"] for r in rows if r.get("doc_id")}
    return sorted(seen)


# ---------------------------------------------------------------------------
# LLM JSON 输出解析
# ---------------------------------------------------------------------------

def _strip_fences(raw: str) -> str:
    s = re.sub(r"```(?:json)?\s*", "", raw or "")
    return re.sub(r"```\s*", "", s)


def extract_json_array(raw: str) -> List[Any]:
    s = _strip_fences(raw)
    m = re.search(r"\[[\s\S]*\]", s)
    if not m:
        return []
    try:
        val = json.loads(m.group())
        return val if isinstance(val, list) else []
    except json.JSONDecodeError:
        return []


def extract_json_object(raw: str) -> Dict[str, Any]:
    s = _strip_fences(raw)
    m = re.search(r"\{[\s\S]*\}", s)
    if not m:
        return {}
    try:
        val = json.loads(m.group())
        return val if isinstance(val, dict) else {}
    except json.JSONDecodeError:
        return {}
