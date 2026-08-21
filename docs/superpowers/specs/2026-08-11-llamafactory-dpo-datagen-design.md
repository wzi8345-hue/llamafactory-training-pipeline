# LlamaFactory SFT / DPO 数据生成改造设计

日期：2026-08-11
范围：`llamafactory_pipeline` 数据生成页、生成任务后端和训练数据导入兼容性

## 目标

在现有 `QA / 多篇 QA / FC` 三类生成任务之上增加独立的微调数据类型：
`SFT` 和 `DPO`。用户创建生成任务时先选择微调数据类型，再选择业务任务类型。
现有 SFT 行为保持兼容；DPO 产物采用 LLaMAFactory 官方 ShareGPT 偏好数据格式，
可从训练页直接导入。

## 方案选择

采用正交配置 `finetune_type: sft | dpo`，保留现有
`task_type: qa | qa_multi | fc`。生成来源和任务语义由 `task_type` 决定，记录格式、
负样本生成和成对质量闸由 `finetune_type` 决定。

未采用的方案：

- 新增 `dpo_qa / dpo_qa_multi / dpo_fc`：会复制任务分支，并让前端条件组合持续膨胀。
- 先生成 SFT 再离线转换为 DPO：不能在同一证据和同一次质量判断中保证偏好对有效。

## 数据契约

`DatagenConfig` 新增 `finetune_type`，默认值为 `sft`，从而兼容历史任务配置。

SFT 输出保持现状：

```json
{
  "conversations": [
    {"from": "human", "value": "问题"},
    {"from": "gpt", "value": "回答"}
  ]
}
```

DPO QA / 多篇 QA 输出：

```json
{
  "conversations": [{"from": "human", "value": "问题"}],
  "chosen": {"from": "gpt", "value": "有证据支撑的高质量回答"},
  "rejected": {"from": "gpt", "value": "自然但存在明确缺陷的回答"}
}
```

DPO FC 输出保留工具定义，并用正确与错误 assistant 消息形成偏好对：

```json
{
  "conversations": [{"from": "human", "value": "用户发话"}],
  "chosen": {"from": "function_call", "value": "{\"name\":\"plan\",\"arguments\":{...}}"},
  "rejected": {"from": "gpt", "value": "不应直接回答的文本"},
  "tools": "[...]"
}
```

FC 负样本混合三类错误：选错工具/意图、工具正确但参数错误、错误地直接文本回答。
负样本必须结构合法且非空，但业务决策错误，避免用乱码、空对象等捷径训练模型。
工具调用消息直接使用 LLaMAFactory 原生 `function_call` 角色；若种子含多轮历史，
`conversations` 保留末轮正确标签之前的全部历史，只把末轮正确标签放入 `chosen`。

生成任务的 `progress.json`、列表接口和创建响应均带 `finetune_type`。下载文件名改为
`sft_<job_id>.json` 或 `dpo_<job_id>.json`，历史任务缺少字段时按 SFT 处理。

## 生成与质量流程

### SFT

保持现有流程：采样来源 → 生成候选 → schema/实体/接地/评分 → 语义去重 → 接受。

### DPO QA / 多篇 QA

1. 复用现有来源和生成逻辑得到问题与高质量答案。
2. 复用现有 QA Judge，先确认 chosen 接地且达到分数门槛。
3. 针对同一问题、答案和证据生成一个自然但有明确缺陷的 rejected。
4. 成对 Judge 检查 chosen 明显优于 rejected、两者不相同，且 rejected 的缺陷真实存在。
5. 以问题作为去重键，组装 ShareGPT 偏好记录。

### DPO FC

1. 复用 FC 种子采样和发话改写；种子的正确 `tool_calls` 转换为 LLaMAFactory 原生
   `function_call` chosen 消息。
2. 随机选择三类错误之一，生成结构合法的 rejected assistant 消息。
3. 规则闸检查 chosen 与种子一致、rejected 与 chosen 不相同；对于错误参数样本，确认确有
   参数缺失、变更或臆造；对于错误工具样本，确认工具名/意图不同；对于文本回答样本，
   确认没有复刻正确 tool call。
4. 成对 Judge 检查 chosen 明显优于 rejected，随后按发话去重并组装记录。

DPO 新增可编辑 `rejected_prompt` 和 `pair_judge_prompt`。现有 `gen_prompt`、
`judge_prompt` 继续分别控制 chosen 生成和 chosen 质量；SFT 时隐藏 DPO 专属字段。

## 后端边界

- `datagen_schema.py`：新增微调类型、DPO 默认 prompt 与配置解析。
- `datagen_generate.py`：新增 QA/FC rejected 生成函数和 FC 原生消息转换，不负责闸门。
- `datagen_quality.py`：新增偏好对 Judge 解析和 FC rejected 规则校验。
- `datagen_run.py`：按 `finetune_type` 编排候选漏斗、记录组装、续跑和报告。
- `datagen_job.py` / `app.py`：状态和列表暴露微调类型，下载使用匹配的文件名。
- `schema.py`：识别偏好记录并为 `dataset_info.json` 写入 `ranking: true`、
  `chosen/rejected` 列映射。
- `app.py` 训练入口：从生成任务导入 DPO 数据时校验训练 `stage=dpo`；SFT 数据要求
  `stage=sft`，避免格式与训练阶段错配。

## 前端交互

数据生成页在“任务类型”之前增加“微调数据类型”下拉框，选项为 SFT、DPO。
选择 DPO 时显示 rejected 生成 prompt 和成对 Judge prompt，并保留 QA / 多篇 QA / FC
及其原有来源字段。任务状态、历史任务和下载按钮展示微调数据类型。

训练页的“从生成任务导入”选项显示 `[SFT]` / `[DPO]` 标签；选中任务时自动把训练
`stage` 切换为对应值，后端仍进行最终校验，防止绕过前端。

## 错误处理与兼容性

- 历史配置没有 `finetune_type` 时默认为 `sft`。
- DPO rejected 解析失败、与 chosen 相同、错误类型不成立或成对 Judge 不通过时，只拒绝
  当前候选并记录原因，不终止整个任务。
- DPO 专属 prompt 为空时使用后端默认值；前端从 schema 接口加载默认模板。
- 生成任务达到尝试上限时仍输出已接受数据和报告，报告标明微调类型与各拒绝原因。
- 不改变历史 SFT `output.json` 结构和已有 API 路径。

## 测试与验收

纯逻辑测试不调用真实 LLM/embedding：

- 旧配置默认解析为 SFT，DPO 配置和默认 prompt 切换正确。
- QA DPO 与三类 FC DPO rejected 均能解析、验证和组装为官方格式。
- chosen/rejected 相同、错误类型不成立、pair Judge 不通过会被拒绝。
- DPO 续跑可从既有 `output.jsonl` 恢复去重状态和计数。
- DPO 首条记录生成的 `dataset_info.json` 包含 `ranking: true` 及列映射。
- 训练入口拒绝 SFT/DPO 数据类型与 `stage` 不匹配。
- schema、任务列表、下载文件名和前端配置提交均包含微调类型。

最终运行后端针对性测试和完整 `llamafactory_pipeline` 测试，并对静态前端脚本做语法检查。
若本地生成依赖的模型服务可用，再执行一条 QA DPO 和一条 FC DPO 的最小真实生成；若服务
不可用，则明确区分“代码测试通过”和“真实数据生成未激活”。
