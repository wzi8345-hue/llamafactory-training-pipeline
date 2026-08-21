# SFT 数据生成子系统 设计

日期: 2026-07-21 · 归属: `llamafactory_pipeline`

## 目标

在训练/评测之外，新增第三条能力：**用 LLM 生成 SFT 数据**，规则前端可配，
重点保证**生成质量**与**不重复**，数量可设。移植/取代 `sft_data/router_fc` 的
纯规则生成，统一到 pipeline 内。

## 关键决策 (已与用户确认)

- 数据类型：**两者都要**，通用框架按 `task_type` 切换（`qa` / `fc`）。
- 通用问答接地：复用本项目检索库（本地 `knowledge_blocks_vec.json` 的 chunk，边采边生成）。
- 质量闸门：**schema 校验 + 接地校验 + LLM 打分门槛 + 实体一致性(FC)**。
- 去重：**语义去重**（EmbeddingClient 余弦 > 阈值即弃）+ 完全重复 hash 快速路径。
- Judge 模型：与生成**同一个模型**（`rag_eval_plan.common.build_llm`，读 `local_api_config.generation`）。
- FC 标签模板：**先复用现有 `sft_data` 数据作种子**（后续用户可自配模板）。

## 执行模型

生成是 LLM-API + embedding 密集、**不吃 GPU**，跑在**本机后端**（不上训练服务器）。
以**本地 detached 子进程**运行 `python -m llamafactory_pipeline.datagen_run <job_dir>`
（`start_new_session=True`，stdout/err 重定向到 `run.log`），浏览器/后端重启不影响。
**可续跑**：accepted 记录 append 到 `output.jsonl`，重启后按已有条数续补到目标。

## 数据流 (每条候选的漏斗)

```
采样源 → LLM 生成候选 → schema → [FC:实体一致性] → 接地+打分 judge → 语义去重 → accept++
```

循环直到 `accepted == count`，或尝试数 > `count × attempt_multiplier`（防死循环）。
每弃一条记原因，产出报告（接受率 / 各闸拒绝数 / 去重命中数）。

## 两条生成线

- **QA 线**：从 `uploads/**/knowledge_blocks_vec.json` 采一个 chunk（`type∈{text,table}`
  且 `len(content)≥min_len`，自带预算 embedding）→ LLM 依据 chunk 出 `{question, answer}`
  （prompt 硬约束"只用给定内容、中文、具体"）→ judge 一次调用同时判**接地(不幻觉)+打分**
  → 去重（对 question 向量）。输出 ShareGPT `human/gpt`。
- **FC 线**：从现有 `sft_router_fc_*.json` 采一条种子（其末轮 `tool_calls` = 保证正确的标签）
  → LLM 把原发话改写成一句多样口语发话（保持意图与全部实体）→ schema 校验 +
  **实体一致性**（args 里的牌号/编号必须出现在新发话，且不臆造）→ judge 打分（改写质量）
  → 去重（对 utterance 向量）。输出 ShareGPT `tool_calls` + 复用种子的 `tools` schema。
  标签由构造保证正确，LLM 只负责多样性，规避 tool_call 出错。

## 模块 (各文件单一职责)

| 文件 | 职责 |
|---|---|
| `datagen_schema.py` | Pydantic 配置 `DatagenConfig` + 默认生成/judge prompt |
| `datagen_source.py` | QA chunk 采样 (读本地 vec.json) / FC 种子加载采样 |
| `datagen_generate.py` | `gen_qa` / `gen_fc` 两个生成策略 (用 LLMClient) |
| `datagen_quality.py` | 四闸 (schema/实体/judge) + `Deduper` (embedding 余弦) |
| `datagen_run.py` | job 主循环 + 续跑 + 写 progress/manifest/output/report |
| `app.py` (+) | `/api/datagen/*` 接口 + 本地子进程启动 + 本地日志 SSE |
| `static/index.html` (+) | 「数据生成」Tab |

## 前端可配字段

`task_type`、`count`(目标接受数)、源 (QA: `kb_source_dir`; FC: `fc_seed_file`)、
`temperature`、`judge_min_score`(1-5)、`grounding_check`、`dedup_threshold`(余弦)、
`attempt_multiplier`、可编辑 `gen_prompt` / `judge_prompt`。

## 输出

`sft_data/generated/<job_id>/`：`config.json` / `run.log` / `progress.json` /
`output.jsonl`(增量) / `output.json`(ShareGPT 数组, 可直接进训练上传) /
`manifest.jsonl`(每条 meta + 拒因) / `report.md`。
`sft_data/generated/` 加入 `.gitignore`（生成产物勿入库）。

## 质量/去重细节

- 闸门顺序：先便宜 (schema/实体) 后贵 (judge)，早弃省钱。
- 去重：候选先 hash 挡完全重复；再 embed 与已接受集算**最大余弦**，`>阈值(默认0.9)` 弃。
  已接受向量增量维护在内存，重启从 `output.jsonl` 重算。
- 成本：QA 用 chunk 现成 embedding 接地，仅问句现算向量；judge 与生成同模型。

## 默认值 (可前端覆盖)

`temperature=0.9`、`judge_min_score=4`、`grounding_check=true`、`dedup_threshold=0.9`、
`attempt_multiplier=3`、`min_len=200`、输出 `sft_data/generated/`。

## 测试 (纯逻辑, 不打真实 LLM)

`test_datagen.py`：schema 校验、FC 实体一致性、`Deduper`(构造近义/重复验证命中)、
报告聚合、续跑计数、ShareGPT 记录组装。LLM/embedder 用桩替身。
