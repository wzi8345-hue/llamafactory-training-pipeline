# LlamaFactory 训练后批量评测 设计

## 目标

在训练完成后，对**基座模型**和**微调模型**批量评测两类任务：function call 与主观任务。
function call 评测工具名正确率与参数正确性；主观任务由 LLM 按用户 prompt 打分。支持多模型对比或
单独评测微调模型，每次评测结果落盘记录。本子系统并入现有 `llamafactory_pipeline/`（新增“评测”页），
不改动 RAG 主链路。

## 关键决策（已确认）

- 推理方式：每个被测模型在容器内起 OpenAI 兼容端点（`llamafactory-cli api`，vLLM 后端），
  function call 用原生 tool-calling，主观用 chat。
- function call 打分：工具名精确匹配算正确率；参数由用户 prompt 交 LLM judge 打分（混合）。
- judge LLM：复用项目生成模型（`rag_eval_plan/common.build_llm()`，当前 Qwen3.6-27B）。
- 执行拆两阶段：推理在服务器持久跑，打分在本地可重跑。
- 模型选择：微调模型=选一个已完成训练任务（取其 base+adapter）；基座=服务器上模型绝对路径。
- 评测集格式由本设计定义。

## 两阶段架构

### 阶段一 · 推理（服务器任务，持久）

沿用训练 pipeline 的 SSH/docker/nohup 机制。一个评测任务生成 `run_eval.sh`，容器内对每个被测模型依次：

1. 生成该模型的推理配置 `infer.yaml`（`model_name_or_path`、可选 `adapter_name_or_path`、
   `template`、`infer_backend: vllm`、`trust_remote_code: true`）。
2. `API_PORT=<port> CUDA_VISIBLE_DEVICES=<gpus> llamafactory-cli api infer.yaml` 起端点。
3. 轮询 `/v1/models` 就绪（带超时）。
4. 运行 `runner.py` 遍历评测集：FC 调 `chat/completions` 带 `tools`，主观调 `chat/completions`；
   写 `<model_name>.predictions.jsonl`。
5. 杀掉端点，进入下一模型。

整个脚本用 `nohup` 脱离 SSH 会话；结束写 `exit_code`，全程日志进 `eval.log`。状态/日志判定完全复用
`remote.py` 既有逻辑（pid/exit_code/tail）。

### 阶段二 · 打分（本地，可重跑）

`scp` 拉回各模型 `predictions.jsonl` → 复用 `common.build_llm()` 按用户 prompt 打分 →
聚合写报告。打分只读 predictions，失败可直接重跑，不依赖服务器可达 judge 端点。

## 评测集格式

两类均为 JSON 数组或 JSONL，逐条一个对象。

### function call

```json
{
  "id": "fc_001",
  "query": "把明天10点的会议改到下午3点",
  "system": "可选 system 提示",
  "tools": [
    {"type": "function", "function": {"name": "update_event",
      "parameters": {"type": "object", "properties": {"time": {"type": "string"}}}}}
  ],
  "gold": {"name": "update_event", "arguments": {"time": "15:00"}}
}
```

- `tools` 为 OpenAI tools 结构，直接传给模型。
- `gold.name` 为期望工具名（精确匹配用）；`gold.arguments` 为期望参数（judge 参考）。

### 主观任务

```json
{"id": "subj_001", "query": "…", "system": "可选", "reference": "可选参考答案"}
```

## 打分口径

- FC 工具名：`pred.name == gold.name` → `tool_name_accuracy`（模型级命中率）。
- FC 参数：judge(prompt, query, gold.arguments, pred.arguments) → `param_score`(1–5) + `reason`。
- 主观：judge(prompt, query, reference?, answer) → `score`(1–5) + `reason`。
- judge 输出 JSON，用 `common.extract_json_object` 解析；解析失败记为无效并计入统计。
- prompt 有内置默认值，前端可编辑；两类各一套。

## 模型选择

- 微调模型：选一个已完成训练任务 `job_id` → 从该任务的 `qwen3_lora_sft.yaml` 读
  `model_name_or_path`(base) 与 `output_dir`(adapter)、`template`。
- 基座模型：一个或多个服务器上的模型绝对路径；template 默认沿用训练模板，可改。
- 每个被测模型有一个 `name` 标签（报告与产物按它命名，校验为安全字符）。
- 至少选一个模型；可只选微调，也可基座+微调混选。

## API

并入现有 app：

- `GET /api/eval/schema`：返回默认 judge prompt 与评测集字段说明。
- `POST /api/eval/jobs`：multipart 提交 —— 评测配置 JSON（模型列表、任务类型、gpus、prompts）
  + 评测集文件（FC/主观各一，按需）。校验后生成 `run_eval.sh` 与 runner，上传并启动推理任务，
  返回 `eval_run_id`。
- `GET /api/eval/jobs/{id}`：推理任务状态（复用训练状态判定）。
- `GET /api/eval/jobs/{id}/logs`：SSE 日志（复用）。
- `POST /api/eval/jobs/{id}/score`：拉回 predictions → 本地打分 → 写报告，返回汇总。
- `GET /api/eval/jobs/{id}/report`：返回已生成报告（markdown/JSON）。

`eval_run_id` 与训练 `job_id` 同构（时间戳+随机），只允许安全字符。

## 目录与产物

服务器任务目录：

```
<TRAIN_REMOTE_ROOT>/<eval_run_id>/
├── evalset/{fc.jsonl, subjective.jsonl}
├── models/<model_name>.infer.yaml
├── runner.py
├── run_eval.sh
├── <model_name>.predictions.jsonl
├── eval.log, pid, exit_code
```

本地结果：

```
llamafactory_pipeline/eval_results/<eval_run_id>/
├── <model_name>.predictions.jsonl   # 拉回
├── <model_name>.scores.jsonl        # 逐条得分+reason
├── config.json                      # 本次模型/任务/prompt 快照
└── report.md                        # 多模型对比: tool_name_accuracy / param_score 均值 / 主观 score 均值
```

结果目录不进 git（数据产物）。

## 文件组织

- `eval_schema.py`：评测请求模型（Pydantic）、评测集逐条校验、默认 judge prompt、模型名/参数校验。
- `eval_remote.py`：生成 `infer.yaml` / `runner.py` / `run_eval.sh`、提交推理任务、拉回 predictions。
  复用 `remote.py` 的 ssh/scp/job_dir/状态/日志，不重写连接层。
- `eval_judge.py`：本地打分 + 聚合 + 报告；复用 `rag_eval_plan.common.build_llm/extract_json_object`。
- `app.py`：新增上述评测接口。
- `static/index.html`：新增“评测”页签（模型选择、评测集上传、prompt 编辑、状态/日志、打分与报告）。
- `test_eval.py`：纯逻辑自检。

## 错误处理

- 评测配置或评测集字段不合法：提交阶段拒绝，不建远端任务。
- 某模型端点起不来/就绪超时：`run_eval.sh` 跳过该模型并在 `eval.log` 标注，继续其余模型，
  不整体失败（缺失的 predictions 在打分阶段体现为该模型缺数据）。
- 打分阶段某条 judge 解析失败：该条标记 invalid，计入统计，不中断整体。
- 服务器不可达/scp 失败：返回明确错误；已生成的 predictions 保留供重试打分。
- 报错内容不含私钥、完整 SSH 命令或敏感环境变量。

## 复用清单

| 复用 | 来源 |
|------|------|
| ssh/scp、job 目录、状态、日志 | `llamafactory_pipeline/remote.py` |
| 训练任务读取（base/adapter/template） | 远端 `<job_id>/qwen3_lora_sft.yaml` |
| judge 客户端 + JSON 解析 | `rag_eval_plan/common.py` |
| judge prompt/打分模式 | `rag_eval_plan/score_answers.py` 的 `judge()` |
| 增量写盘、MD 报告模式 | `rag_eval_plan/score_answers.py` / `evaluate.py` |
| 数据文件大小/扩展名校验 | `llamafactory_pipeline/app.py` 上传逻辑 |

## 验证

纯逻辑自检（不依赖真实服务器/GPU）：

- FC / 主观评测集逐条校验：合法通过、缺字段/结构错被拒。
- 工具名精确匹配与 `tool_name_accuracy` 聚合正确。
- judge 输出 JSON 解析：正常、带 ```json 围栏、非法各一。
- `runner.py` 通过 `py_compile`；`run_eval.sh` 通过 `sh -n`；路径经 shlex.quote（注入用例）。
- 模型名/gpus 非法输入被拒。
- 聚合报告在多模型/缺失模型下数值正确。

真实验收：选一个已完成训练任务 + 一个基座，跑小评测集 → 观察 eval.log → 拉回打分 → 报告对比。
