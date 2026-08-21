"""Pydantic 模型: 仅用于验证大模型 (LLM) 的输出结果。

包括:
- RouteDecision: LLM router 输出的结构化决策
- LLMChatResponse: LLM chat completions 接口返回
- QueryResult: 最终查询输出 (包含 LLM 生成的 answer)
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# LLM Router 输出验证
# ---------------------------------------------------------------------------

class RouteDecision(BaseModel):
    """LLM router 输出的结构化决策。

    rewrites: 扁平结构, 每条路径一个改写字符串。
    time: 时间表达式字符串, 格式 "xxxx-xxxx" 或 "xxxx" 或 ""。
    chunk_type: chunk 类型过滤 (image/table/equation/references); references 表示
                 "参考文献意图", 配合 progressive/local 全量召回 references chunk,
                 不需要给条目编号 (精确编号场景占比低, 已简化).
    target_doc_ids: 由 router 通过 doc_registry 解析得到的规范 doc_id 列表;
                    与 target_docs 一一对应 (但仅在 registry 命中时才填). 检索层
                    优先用 doc_id 短路定位, 避免 doc_name 拼写/翻译差异导致 BM25
                    兜底扩散到无关文献.
    fig_refs / table_refs: 提升为顶层, 用于 metadata 路径的 type 过滤。
    page_refs: 用户在问题中明确指定的页码列表 (1-based; 检索时会减 1 转 page_start).
    paragraph_refs: 用户明确指定的段落序号列表 (1-based, 与 chunk.paragraph_index 对应).
    entities: 用户明确指定要在 chunk.content 中查找的实体名/术语列表;
              metadata 路径会用 LIKE 在 content 上做精确子串匹配。
    expand_neighbors: 依赖图谱式邻域扩展模式列表 (空=不扩展); 取值 ∈
              {"assets","adjacent","page","similar"}。检索完成后由 NeighborExpander
              对种子 hit 沿对应的边各扩 1 跳, 回填到 results["neighbor"]。
              典型: "图N附近的文字"→["page","assets"]; "还研究了什么/相关内容"→
              ["adjacent","assets"]; "其他类似方法"→["similar"]。
    """
    routes: List[str] = []
    rewrites: Dict[str, str] = {}
    time: str = ""
    chunk_type: Optional[str] = None
    target_docs: List[str] = []
    target_doc_ids: List[str] = []
    fig_refs: List[str] = []
    table_refs: List[str] = []
    page_refs: List[int] = []
    paragraph_refs: List[int] = []
    entities: List[str] = []
    retrieve_bias: Optional[str] = None
    rerank_mode: Optional[bool] = None
    # KG 开关 (router 产出): True=本 query 适合走 KG 支路; False=明确不走; None=模型未表态
    # (老 SFT 模型不产出该字段) → 沿用旧行为 (KG 照常按配置跑)。仅 False 会关掉 KG 补召回。
    want_kg: Optional[bool] = None
    # Router 给 KG 图召回的最大授权范围；None=legacy/heuristic 未表态，继续使用链路配置。
    # 最终是否能召回仍由本地实体链接、单位、指标与未知牌号守卫裁决。
    kg_scope: Optional[Literal["entity_only", "allow_global_metric", "off"]] = None
    # KG 数值类目意图 (router LLM 输出; 消费侧在 graph_link 的 chunks_for_metrics/category_clause)
    # None=模型未表态 (走规则 fallback); 空 list=模型明确说"无类目约束" (放行全部)。
    # 由 cats 数组解析而来, 在 decision_builder._apply_top_level_extras 写入。
    kg_categories: Optional[List[str]] = None
    # rerank 精排 query (router LLM 输出): 非空时优先于规则 (user_query + hints) 直接用作
    # reranker 的 query 文本; None=模型未表态, 走 synthesize_rerank_query 的规则 fallback。
    rerank_query: Optional[str] = None
    # Router 对“用户是否明确索取参考文献列表”的结构化语义判断。
    # None 仅用于 legacy/heuristic 兼容；FC schema 要求显式 true/false。
    want_reference_list: Optional[bool] = None
    expand_neighbors: List[str] = []
    reasoning: str = ""
    raw_response: str = ""

    def has(self, route: str) -> bool:
        return route in self.routes

    def get_rewrite(self, route: str, fallback: str = "") -> str:
        return self.rewrites.get(route) or fallback

    def has_metadata_filters(self) -> bool:
        """是否带有任意可用于 metadata 路径的硬过滤条件。"""
        return bool(
            self.fig_refs or self.table_refs
            or self.page_refs or self.paragraph_refs
            or self.entities
        )

    def to_time_filter(self) -> Optional[str]:
        """解析 time 字段 "2015-2025" / "2018" / "" → Milvus filter。"""
        import re as _re
        t = self.time.strip()
        if not t:
            return None
        m = _re.match(r"^(\d{4})\s*[-~到至]\s*(\d{4})$", t)
        if m:
            return f"publication_year >= {m.group(1)} and publication_year <= {m.group(2)}"
        m = _re.match(r"^(\d{4})$", t)
        if m:
            return f"publication_year >= {m.group(1)} and publication_year <= {m.group(1)}"
        return None


# ---------------------------------------------------------------------------
# LLM Chat 输出验证
# ---------------------------------------------------------------------------

class LLMChatResponse(BaseModel):
    """LLM chat completions 接口返回的结构化结果。"""
    answer: str
    usage: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# 最终查询输出 (包含 LLM answer)
# ---------------------------------------------------------------------------

class QueryResult(BaseModel):
    """query 流程的输出, 包含 LLM 生成的 answer。"""
    query: str
    answer: str = ""
    hits: List[Dict[str, Any]] = []
    context: str = ""
    usage: Optional[Dict[str, Any]] = None
    latency_s: float = 0.0
    error: Optional[str] = None
    session_meta: Dict[str, Any] = Field(default_factory=dict)
    # 智能体执行信号 (仅 LangGraph 路径填充; 其它路径保持默认)
    needs_clarify: bool = False   # 本轮是 router 反问 (answer 即反问文本), 未做检索/生成
    needs_reuse: bool = False     # 本轮复用上轮材料直接作答, 未做新检索
    no_answer: bool = False       # 检索/重试后仍无可靠证据, answer 为保守说明
    retry_count: int = 0          # 反思循环触发的重试次数
    correlation_id: str = ""      # 本轮检索的关联 ID, 便于对齐日志
    # 专业研究模式 (professional) 的执行概要; 普通模式为 None
    research: Optional[Dict[str, Any]] = None
