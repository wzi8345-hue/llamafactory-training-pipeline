"""SFT 数据生成配置模型 + 默认 prompt 模板。"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# QA 生成: 依据给定 chunk 出一问一答, 硬约束只用给定内容。
DEFAULT_QA_GEN_PROMPT = """你是耐候钢知识库的 SFT 数据构造助手。下面给你一段文献片段。
请基于**且仅基于**该片段内容, 生成一个高质量的中文问答对:
- question: 一个具体、自然的用户问题, 答案必须能在片段中找到依据;
- answer: 准确、完整的回答, 只使用片段中的信息, 不得编造片段外的数据或结论。
只输出 JSON 对象, 形如 {"question": "...", "answer": "..."}, 不要额外文字。"""

# QA judge: 一次调用同时判接地(不幻觉) + 打分。
DEFAULT_QA_JUDGE_PROMPT = """你是严格的 SFT 数据质检员。给你[文献片段]、[问题]、[答案]。
判断:
1. grounded: 答案是否完全由片段支撑、无幻觉 (true/false);
2. score: 综合质量 1-5 分 (问题是否具体有价值、答案是否准确完整)。
只输出 JSON: {"grounded": true/false, "score": 1-5, "reason": "简述"}。"""

# FC 生成: 把种子发话改写成一句多样口语发话, 保持意图与全部实体。
DEFAULT_FC_GEN_PROMPT = """你在为"意图路由"节点扩充 SFT 训练发话。给你一条原始用户发话与其意图标签。
请改写出**一句**语义等价但表达更口语、更多样的中文发话:
- 必须保留原发话涉及的全部实体 (钢种牌号、文献名/编号、图表号、页码等), 不增不减;
- 可用省略、口语词、简称、疑问变体, 但不得改变查询意图;
- 不要输出标签或解释。
只输出 JSON: {"utterance": "改写后的发话"}。"""

# FC judge: 打分改写质量 (是否自然、意图是否保持)。
DEFAULT_FC_JUDGE_PROMPT = """你是 SFT 数据质检员。给你[原发话]和[改写发话]。
判断改写是否自然通顺且**意图完全一致**, 打 1-5 分。
只输出 JSON: {"score": 1-5, "reason": "简述"}。"""

# 多篇 QA 生成 — 综合型: 一个问题必须拼多篇才能答全。
DEFAULT_QA_MULTI_GEN_SYNTHESIS = """你是耐候钢/材料知识库的 SFT 数据构造助手。下面给你**多段来自不同文献**的片段, 用 [文献1]/[文献2]... 标注。
请构造一个需要**综合其中至少两篇**才能完整回答的中文问答对:
- question: 一个具体问题, 单独看任何一篇都答不全, 需要跨文献拼接/对比信息;
- answer: 综合多篇给出完整回答, 只使用给定片段的信息, 不得编造; 说明结论分别来自哪些文献。
只输出 JSON: {"question": "...", "answer": "..."}, 不要额外文字。"""

# 多篇 QA 生成 — 共识型: 多篇都能答, 答案融合并标注一致/差异。
DEFAULT_QA_MULTI_GEN_CONSENSUS = """你是耐候钢/材料知识库的 SFT 数据构造助手。下面给你**多段来自不同文献**的片段, 用 [文献1]/[文献2]... 标注, 它们主题相近。
请构造一个这几篇都涉及的共性中文问答对:
- question: 一个多篇文献都能提供依据的具体问题;
- answer: 融合多篇内容作答, 只用给定片段信息, 不得编造; 若各篇存在一致/差异, 明确指出。
只输出 JSON: {"question": "...", "answer": "..."}, 不要额外文字。"""

# 多篇 QA judge: 接地在片段整体 + 报告实际用到的文献编号 (用于防退化成单篇)。
DEFAULT_QA_MULTI_JUDGE_PROMPT = """你是严格的 SFT 数据质检员。给你多段带 [文献N] 标注的片段、[问题]、[答案]。
判断:
1. grounded: 答案是否完全由这些片段整体支撑、无幻觉 (true/false);
2. used_docs: 答案实际依据了哪些文献编号的列表, 如 [1,2];
3. score: 综合质量 1-5 分 (问题是否有价值、答案是否准确完整)。
只输出 JSON: {"grounded": true/false, "used_docs": [1,2], "score": 1-5, "reason": "简述"}。"""

DEFAULT_QA_DPO_REJECTED_PROMPT = """你在构造 DPO 偏好数据。给你证据、问题和正确的 chosen 回答。
请生成一个自然、相关、看似合理，但包含明确事实错误、证据误读或关键遗漏的 rejected 回答。
不得输出乱码、空话或声明自己在故意犯错，也不得照抄 chosen。
只输出 JSON: {"rejected": "较差回答"}。"""

DEFAULT_FC_DPO_REJECTED_PROMPT = """你在构造函数调用路由的 DPO 偏好数据。给你用户发话、正确的 chosen 工具调用和错误类型。
生成一个结构合法但决策错误的 rejected 消息。错误类型只会是 wrong_tool、wrong_args、direct_answer。
wrong_tool 输出另一个已提供工具；wrong_args 保持工具名但修改、遗漏或臆造参数；direct_answer 直接给出不应给出的文本回答。
只输出 JSON；工具错误用 {"name": "工具名", "arguments": {...}}，文本错误用 {"answer": "文本"}。"""

DEFAULT_DPO_PAIR_JUDGE_PROMPT = """你是严格的 DPO 偏好对质检员。比较同一输入下的 chosen 和 rejected。
chosen 必须正确、接地且明显优于 rejected；rejected 应自然相关但确有缺陷，不能只是乱码或空值。
只输出 JSON: {"chosen_better": true/false, "chosen_score": 1-5, "rejected_score": 1-5, "reason": "简述"}。"""


class DatagenConfig(BaseModel):
    finetune_type: Literal["sft", "dpo"] = "sft"
    task_type: Literal["qa", "fc", "qa_multi"]
    count: int = Field(gt=0, le=100000)

    # 源
    kb_source_dir: str = "uploads"                                   # QA: 扫 knowledge_blocks_vec.json
    fc_seed_file: str = "sft_data/router_fc/sft_router_fc_shougang_1160.json"  # FC 种子
    min_len: int = 200                                              # QA chunk 最短内容

    # 多篇 QA (qa_multi): 从 Milvus 跨文献召集证据
    collection: str = "kb_c2261879"                                 # 证据来源 collection
    n_docs: int = Field(default=3, ge=2, le=8)                       # 每题目标文献数
    neighbor_top_k: int = Field(default=20, ge=2, le=200)           # 邻居候选池大小
    sub_mode: Literal["synthesis", "consensus"] = "synthesis"       # 综合型 / 共识型

    # 生成
    temperature: float = 0.9

    # 质量闸门
    judge_min_score: int = Field(default=4, ge=1, le=5)
    grounding_check: bool = True

    # 去重
    dedup_threshold: float = Field(default=0.9, ge=0.0, le=1.0)

    # 循环
    attempt_multiplier: float = Field(default=3.0, ge=1.0, le=50.0)

    # 可编辑 prompt
    gen_prompt: str = ""
    judge_prompt: str = ""
    rejected_prompt: str = ""
    pair_judge_prompt: str = ""

    def min_docs(self) -> int:
        """答案至少需覆盖的文献数: 综合型强制 >=2 防退化成单篇。"""
        return 2 if self.task_type == "qa_multi" and self.sub_mode == "synthesis" else 1

    def resolved_gen_prompt(self) -> str:
        if self.gen_prompt.strip():
            return self.gen_prompt
        if self.task_type == "qa":
            return DEFAULT_QA_GEN_PROMPT
        if self.task_type == "fc":
            return DEFAULT_FC_GEN_PROMPT
        return (DEFAULT_QA_MULTI_GEN_SYNTHESIS if self.sub_mode == "synthesis"
                else DEFAULT_QA_MULTI_GEN_CONSENSUS)

    def resolved_judge_prompt(self) -> str:
        if self.judge_prompt.strip():
            return self.judge_prompt
        if self.task_type == "qa":
            return DEFAULT_QA_JUDGE_PROMPT
        if self.task_type == "fc":
            return DEFAULT_FC_JUDGE_PROMPT
        return DEFAULT_QA_MULTI_JUDGE_PROMPT

    def resolved_rejected_prompt(self) -> str:
        if self.rejected_prompt.strip():
            return self.rejected_prompt
        if self.task_type == "fc":
            return DEFAULT_FC_DPO_REJECTED_PROMPT
        return DEFAULT_QA_DPO_REJECTED_PROMPT

    def resolved_pair_judge_prompt(self) -> str:
        return self.pair_judge_prompt.strip() or DEFAULT_DPO_PAIR_JUDGE_PROMPT
