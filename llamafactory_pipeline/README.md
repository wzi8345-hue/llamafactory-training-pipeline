# LlamaFactory 远程训练 Pipeline

本地 Web 配置参数 + 上传数据 → 经 SSH 在服务器启动 LlamaFactory 训练 → 浏览器实时看日志。
训练用 `nohup` 脱离 SSH 会话, 关闭浏览器或重启本地服务都不会终止远端训练。

全流程闭环：**数据管理 → 训练 → 评测 → 模型部署**，四个 tab。

## 前置条件

- 本机可用 `ssh` / `scp`, 且已配置到训练服务器的免密登录 (密钥或 ssh-agent)。
- 服务器已安装 LlamaFactory, 可直接执行 `llamafactory-cli train ...`。
- 模型权重已在服务器 (本地不传模型)。

## 服务器配置

连接信息写在 `server_config.yaml` (已 gitignore)。首次使用:

```bash
cp llamafactory_pipeline/server_config.example.yaml llamafactory_pipeline/server_config.yaml
# 编辑填入 ssh_target / remote_root / llamafactory_dir 等
```

| 配置项 (yml) / 环境变量 | 必填 | 说明 |
|------|------|------|
| `ssh_target` / `TRAIN_SSH_TARGET` | 是 | `user@host` |
| `remote_root` / `TRAIN_REMOTE_ROOT` | 是 | 远端任务根目录, 如 `/data/lf_jobs` |
| `llamafactory_dir` / `LLAMAFACTORY_DIR` | 是 | 服务器 LlamaFactory 目录 |
| `ssh_port` / `TRAIN_SSH_PORT` | 否 | 默认 `22` |
| `ssh_identity` / `TRAIN_SSH_IDENTITY` | 否 | 私钥路径; 不填走 ssh-agent/默认配置 |
| `docker_container` / `TRAIN_DOCKER_CONTAINER` | 否 | 设置后走 `docker exec <容器>`; 不设则在宿主机跑 |

同名环境变量若设置会**覆盖** yml。配置文件路径可用 `TRAIN_CONFIG` 指定 (默认包目录下)。

## 启动

```bash
python -m uvicorn llamafactory_pipeline.app:app --host 127.0.0.1 --port 8899
```

浏览器打开 http://127.0.0.1:8899 。

## 用容器训练 (推荐)

服务器上起一个常驻官方容器, 把 LlamaFactory 目录、任务根目录、模型目录用**相同路径**挂进去
(路径一致才能免去宿主机↔容器的路径转换):

```yaml
# docker-compose.yml
services:
  llamafactory:
    image: hiyouga/llamafactory:latest
    container_name: llamafactory
    command: sleep infinity
    ipc: host
    gpus: all
    volumes:
      - /data/wangzhengyan/LLaMA-Factory:/data/wangzhengyan/LLaMA-Factory
      - /data/wangzhengyan/sft_data:/data/wangzhengyan/sft_data
      - /data/wangzhengyan/Qwen3.5-9B:/data/wangzhengyan/Qwen3.5-9B
    restart: unless-stopped
```

```bash
docker compose up -d
docker exec llamafactory llamafactory-cli version   # 验证
```

本地启动时加一行:

```bash
export TRAIN_DOCKER_CONTAINER=llamafactory
```

pipeline 会自动把训练命令换成 `docker exec ... llamafactory-cli train`, 而 SSH 上传、
状态查询、日志 tail 全部不变 (文件在宿主机, 容器通过挂载卷读写同一份)。

## 指定显卡

前端表单 "GPU (CUDA_VISIBLE_DEVICES)" 字段:

- 留空 → 用容器/环境默认全部可见卡。
- `0` → 只用 0 号卡。
- `0,1` → 用 0、1 两张卡, 并自动设置 `FORCE_TORCHRUN=1` 启用分布式 (否则 LlamaFactory 只用一张)。

## 数据格式

ShareGPT SFT, 支持两种:

- `conversations` + `from`/`value`
- `messages` + `role`/`content` (OpenAI 风格)

系统读取首条记录自动识别并生成 `dataset_info.json`, 训练 YAML 的 `dataset` 自动指向本次上传。

## 远端任务布局

```
<TRAIN_REMOTE_ROOT>/<job_id>/
├── data/{train.json[l], dataset_info.json}
├── qwen3_lora_sft.yaml
├── train.log   # SSE 实时读取
├── pid
└── exit_code   # 0=成功, 非0=失败
```

## 评测 (训练后批量评测)

前端 "评测" 页, 两阶段:

1. **推理 (服务器, 持久)**: 选被测模型 (微调模型填训练 job_id 自动解析 base+adapter+template;
   基座模型每行 `name=/绝对路径`) + 上传评测集 → 容器内对每个模型依次起 `llamafactory-cli api`
   (vLLM) 端点, runner 遍历评测集出 `<model>.predictions.jsonl`。同样 `nohup` 脱离会话,
   状态/日志复用训练那套 (日志文件为 `eval.log`)。
2. **打分 (本地, 可重跑)**: 点 "拉回并打分" → `scp` 拉回 predictions → 复用项目生成模型做
   LLM judge → 生成对比报告。只读 predictions, 失败可直接重跑。

评测集格式 (`.json` 数组或 `.jsonl`):

```jsonc
// function call: 工具名精确匹配 + 参数交 judge 打分
{"id":"fc1","query":"...","tools":[{"type":"function","function":{"name":"f","parameters":{}}}],
 "gold":{"name":"f","arguments":{"x":1}}}
// 主观: judge 按 prompt 打分
{"id":"s1","query":"...","reference":"可选参考答案"}
```

Judge prompt 前端可编辑 (FC 参数 / 主观各一套)。结果落
`llamafactory_pipeline/eval_results/<eval_id>/` (已 gitignore): 各模型 `*.scores.jsonl`、
`report.md`、`config.json`。设计文档见 `docs/superpowers/specs/2026-07-20-llamafactory-eval-design.md`。

评测复用训练的 SSH/docker/GPU 配置 (同样的环境变量), 无需额外设置。

## SFT / DPO 数据生成

前端“数据生成”页先选择微调数据类型 `SFT` 或 `DPO`，再选择任务类型：单篇 QA、
多篇 QA（综合/共识）或 FC 路由。SFT 保持原有生成、接地、Judge 和语义去重流程；
DPO 在通过 chosen 质量闸后继续生成 rejected，并用规则闸和成对 Judge 确认 chosen
明显更优。FC rejected 会混合选错工具、参数错误和错误直接回答三种负样本。

QA DPO 产物使用 LLaMAFactory ShareGPT 偏好格式：

```json
{
  "conversations": [{"from": "human", "value": "问题"}],
  "chosen": {"from": "gpt", "value": "正确回答"},
  "rejected": {"from": "gpt", "value": "自然但有明确缺陷的回答"}
}
```

FC DPO 的工具调用采用 LLaMAFactory 原生消息格式：

```json
{
  "conversations": [{"from": "human", "value": "用户发话"}],
  "chosen": {"from": "function_call", "value": "{\"name\":\"plan\",\"arguments\":{}}"},
  "rejected": {"from": "gpt", "value": "不应直接给出的回答"},
  "tools": "[...]"
}
```

下载文件名按类型为 `sft_<job_id>.json` 或 `dpo_<job_id>.json`。训练页选择“从生成
任务导入”时会显示 `[SFT]` / `[DPO]` 并自动同步 `stage`；后端也会强制校验：SFT
产物只能用于 `stage=sft`，DPO 产物只能用于 `stage=dpo`。生成任务配置、进度、
`manifest.jsonl` 和报告均记录 `finetune_type`，历史任务缺少该字段时按 SFT 处理。

## 数据管理

训练数据集/评测集本地注册、可复用 (不再每次上传)：

- 存 `sft_data/datasets/` (训练, ShareGPT) 与 `sft_data/evalsets/` (评测, FC/主观), 各带 `.meta.json` (格式/条数/大小)。
- `POST /api/datasets` 上传注册; `GET /api/datasets?kind=train|eval` 列表; `DELETE /api/datasets/{name}?kind=` 删除。
- 训练时 `create_job` 传 `dataset_name` 直接引用本地训练集 (跳过上传); 评测时 `create_eval_job` 配置里传 `fc_dataset_name`/`subj_dataset_name` 引用本地评测集。
- 前端"数据"tab 可视化注册/列表/删除; 训练页数据来源下拉新增"从数据集库选"。

## 个人训练助手

前端“助手”页面把需求澄清、数据构建、训练、监控、A/B 评测和下一轮迭代串成一个可恢复的单人工作流。它是个人工具，不包含账号、租户、权限隔离或多人审批。

### 需求与审批门

标准顺序是：

```text
需求收集 -> 需求确认 -> 数据方案确认 -> 数据构建 -> 数据验收
-> 训练预检 -> 训练 -> A/B -> 诊断/下一轮
```

助手会先收集目标、任务类型（QA、多文档 QA 或 FC）、基座模型路径、模板、基线、数据源、成功标准、数据量上限和可选时间预算。首次输入信息不完整时，助手先给出一张不可执行的“暂定训练方案框架”，同时一次性追问最多三个核心缺口，不调用生成、训练或评测工具。

需求卡的业务场景、当前问题和期望行为必须引用真实用户消息 ID；每个字段都标为“用户明确”、“系统默认”或“助手假设”。不能将助手推断冒充用户目标。只有证据齐全时才进入 `requirements_review`，并创建 `confirm_requirements` 审批。用户确认后才固化 `confirmed_objective` / `objective_hash`，进入数据方案准备；数据构建仍需另一个 `start_datagen` 审批，两个 hash 和动作不可合并。

以下操作都有独立审批门：

- 开始数据生成；
- 开始训练；
- 开始或跳过 A/B 评测；
- 保留已有 A/B 推理产物，只重试本地 judge 评分；
- 训练失败后检查 checkpoint，并在重新预检后创建续训审批；
- 通过全部门槛后接受候选模型；未通过时可结束且不接受，或批准诊断生成的下一轮方案。

每个审批都绑定当时展示计划的 SHA-256。计划、数据集、预检告警或参数发生变化后，旧审批会失效，需重新确认。助手不会自动删除训练、评测、checkpoint 或数据集。

### 八阶段可视化与手动中止

后端 `workflow_steps` 固定投影需求、数据策略、数据构建、数据复核、训练方案/预检、训练、A/B 和诊断八个阶段。前端不猜事件语义；每张卡直接展示后端给出的状态、摘要、数字进度、ETA、问题、参数决策和产物链接。数据构建只在数值变化或五分钟心跳时落 `datagen_progress`；训练进度包含 step/loss/epoch/学习率/ETA；评测在远程推理完成并开始评分时落 `evaluation_progress`。原始 JSON 仅保留在“查看技术详情”折叠区。

任一非终态页面都可显式调用：

```http
POST /api/assistant/workflows/{workflow_id}/cancel
Content-Type: application/json

{"reason":"用户手动中止"}
```

该请求先持久化 `cancelling`，使当前审批和普通 monitor 失效，再为每个数据构建/训练/评测任务创建 `cancel_external_job` 动作。停止器必须校验 PID 身份；只有确认已停止或已终态后才转 `cancelled`。重复点击使用同一 `cancel_request_id`，进程重启后继续同一停止意图。所有已有数据、日志、指标、checkpoint 和 A/B 报告链接均保留；中止不是删除。

### 数据冻结、参数和预检

生成完成后，助手使用固定 `split_seed` 按 `qa` / `qa_multi` / `fc` 分层切分训练集和 holdout。每个任务至少 2 条：少于 30 条时 holdout 取 20%，30–199 条时取 `max(20, round(n * ratio))`，不少于 200 条时取 `round(n * ratio)`，并始终保留训练样本；成对评测少于 30 条时绝不自动接受候选。holdout 不会进入训练产物。`critical_slices` 支持已请求任务标签，以及生成链路可审计的 `task_type=...`、`tool_name=...`、`source_doc=...` 和 DPO `error_type=...` 选择器；多文档样本会保留全部来源文档选择器，与任务不兼容或无法生成的别名会在数据方案前被拒绝。若生成进程中断、未达目标数量或冻结时格式/覆盖校验失败，工作流会回到数据方案并只生成失败任务的恢复审批。数据血缘可由 `workflow_id`、`iteration`、各 job ID、数据集名、文件 SHA-256、`split_seed`、实际 holdout 比例、`train_job_id` 和 `eval_id` 追溯。

参数建议是可重现的确定性策略：用 token P95 选 `cutoff_len`，按数据量选 epoch，按 SFT/DPO 选学习率和目标 global batch，结合模型参数量与当前可用显存估算 LoRA/4-bit QLoRA。多卡默认使用模型在每张卡上复制的 DDP，因此按所选 GPU 中“最小单卡空闲显存”判定，不把多卡显存相加。页面会同时展示完整配置、自动值、理由、风险、预计步数、单卡显存、ETA 依据和时长。

诊断后的下一轮方案可携带结构化 `training_adjustments`，目前只允许在安全边界内调整学习率、epoch、LoRA rank 和 dropout。迭代必须继承上一轮 SFT/DPO 类型，每个参数只能出现一次，且仅在确定性诊断明确建议时按欠拟合/过拟合方向校验。它们会叠加到确定性参数策略上，并记录旧值、新值和证据理由；自由文本不会直接改写训练配置。

置信度的含义：

- `high`：来自确定性规则，或至少 3 次兼容历史训练的 P25–P75 速度区间；
- `medium`：只有 1–2 次兼容历史训练，速度区间已放宽；
- `low`：还没有兼容历史，ETA 使用冷启动吞吐上下界。

冷启动默认使用 `0.05–1.0 steps/s`，可通过 `ASSISTANT_COLD_START_STEPS_PER_SECOND_LOW` 和 `ASSISTANT_COLD_START_STEPS_PER_SECOND_HIGH` 调整。训练结束后会记录实际速度；后续只使用阶段、模型规模、GPU 型号/数量、`cutoff_len` 和量化配置兼容的成功运行更新 ETA。

训练前预检会读取 SSH/容器可达性、远端任务目录写权限、模型文件、GPU 显存与占用、BF16、bitsandbytes、预计截断率和训练时间预算。CLI、Python 与 bitsandbytes 探针使用和训练任务一致的远端前缀及 LlamaFactory 工作目录；容器模式会在容器内读取模型 inventory、GPU/BF16、checkpoint 和训练产物。checkpoint 需包含带合法 `global_step` 的 trainer state、非空权重、optimizer 和 scheduler 文件；分片权重索引必须可解析、具有非空 `weight_map`，且所有引用 shard 均存在并非空。训练即使退出码为 0，也只有在实际运行环境确认可解析的 `adapter_config.json` 及非空 adapter 权重后才算成功。预检会重算真实训练文件 SHA-256，并分别检查任务 staging 盘的“`10GB + 2 × 数据集大小`”和实际 `output_dir` 文件系统的写权限/空间。`pass` 允许审批，`warn` 需用户看到并确认当前告警指纹，`block` 不创建训练审批。点击训练或续训审批时会再做一次预检。

A/B 审批会绑定完整的执行计划：基线/候选的 base、adapter、template、生成参数、冻结评测文件 SHA-256 和基线来源指纹。执行前任一指纹变化都会使本次执行失败并要求重新确认；runner 还会检查评测端口独占性，绑定本次 API PID 和基于完整执行计划生成的唯一服务 ID，不会仅按模型目录名连到旧端口。A/B 使用同一份冻结 holdout 按 ID 成对对比；评分前会再校验本地冻结文件摘要、完整 ID 清单、远端不可变提交摘要和 predictions ID 集。每个请求任务都会自动成为非回归关键切片；多来源文档标签以严格的非空字符串列表贯穿注册、规范化和评分。`report.md` 包含样本数、缺失率、无效率、P50/P95 延迟、win/tie/loss、McNemar、paired bootstrap 区间、关键切片、聚合失败原因和有效但低分/错工具的代表样例。subjective 以 4 分作为可审计正确阈值，missing/invalid 以 0 分进入切片对比，不会从分母删除。judge 分数及缓存证据必须满足完整对象合同和 `1..5` 范围；畸形缓存会被删除并重新评分。judge 逐条加锁落盘，可修复截断尾行并重试 judge 无效行；judge 最终失败时会保留同一 `eval_id` 的 predictions/scores，审批后只重试本地评分。缺少任一模型汇总、请求任务切片、主指标或完整成对业务分数时，证据门会阻止接受候选。接受候选模型还必须同时满足主指标、`non_regression_metrics`、无效率和各关键切片的最坏回归门槛。评测证据质量问题只会建议修复/重跑评测，不会误触发新一轮训练。

如果未达标，确定性诊断会先给出数据覆盖、标注/偏好对、协议、欠拟合或过拟合类别，再由规划器根据证据产生新 `DataPlan`。新方案必须再次批准才会生成数据；只有不同 `train_job_id` 的两轮主指标连续低于 `min_improvement` 才会停止自动迭代，同一候选的重复评分不会重复计数；达到最大迭代数时同样只提示人工复盘。

### 安全工作流 A/B 验收

以下回放只使用脱敏旧版状态/动作事实、临时 SQLite、固定 planner 和假 tools；不连接 SSH，不启动数据生成、GPU 训练或远程评测：

```bash
PYTHONPATH=.:eval .venv-api/bin/python -m llamafactory_pipeline.assistant_ab \
  --output-dir eval/rag_eval_plan/results/assistant_safe_workflow_ab_20260821
PYTHONPATH=.:eval .venv-api/bin/python -m pytest \
  llamafactory_pipeline/test_assistant_ab.py -q
```

验收门槛：B 组未确认需求的提前执行率必须为 0；三个核心需求的用户证据覆盖、`confirm_requirements -> start_datagen` 审批顺序、八阶段投影、取消成功/幂等/产物保留均必须为 1.0；真实外部副作用调用必须为 0。机器报告为 `report.json`，逐场景对照为 `report.md`。

### 独立监控任务

手动执行一次到期动作（最多 20 个）：

```bash
PYTHONPATH=. .venv-api/bin/python -m llamafactory_pipeline.assistant_worker --once --limit 20
```

`--once` 只领取当前到期的持久化动作，每次在真正准备执行时才单条租用，不会让后续动作在队列等待时过期。更新工作流后退出；不保持聊天请求、浏览器或 Uvicorn 进程。launchd 可每 60 秒调用它，worker 内部会根据阶段将下次检查延后 60/120/180 秒；正常轮询在同一条 leased action 上原地重排，不派生多条监控链，过期租约可被下次执行恢复。

所有审批都会先把“待执行动作”原子写入 SQLite；数据生成、训练和评测还会预分配 job ID。外部输入先上传到唯一 staging 目录，再按 submission hash 原子发布；同 hash 重放只检查/启动，不重写正在运行的 YAML、数据或脚本，不同 hash 直接拒绝。远端启动必须回传 `STARTED/ALREADY`：新任务使用单个原子 `launch_identity` 绑定 PID 与 `/proc` starttime；历史多文件标记若不完整或身份已失效会被安全终态化为启动失败，不会永久卡在重放或重复启动。拿不到启动锁时只会延后对账，不会提前提交本地状态。动作租约使用 owner token、定时续租和 compare-and-set fencing；任何 stop 操作在缺少或不匹配创建身份时都会 fail closed，不按命令名猜测 PID。已可能启动远端任务的意图不会因重试次数而放弃或分配新 ID。若 API/worker 在执行中断，下一轮会使用同一意图幂等重放；终态、后续审批和调度动作在同一事务内提交。拒绝初始数据方案会返回需求收集状态，可在同一工作流继续修改。

系统状态默认保存在 `llamafactory_pipeline/assistant_state/assistant.sqlite`，可用 `ASSISTANT_DB_PATH` 覆盖。`assistant_state/*.sqlite*` 和 `assistant_state/assistant_artifacts/` 不进入 Git，SQLite 启用 WAL，API 和 monitor 必须指向同一文件。

如需启用 launchd，在代码合并到主目录后由用户执行：

```bash
cp scripts/com.dprag.llamafactory-assistant-monitor.plist \
  "$HOME/Library/LaunchAgents/com.dprag.llamafactory-assistant-monitor.plist"
launchctl bootstrap "gui/$UID" \
  "$HOME/Library/LaunchAgents/com.dprag.llamafactory-assistant-monitor.plist"
launchctl print "gui/$UID/com.dprag.llamafactory-assistant-monitor"
```

更新 plist 时先执行 `launchctl bootout "gui/$UID/com.dprag.llamafactory-assistant-monitor"`，再重新 `bootstrap`。日志在 `/tmp/dprag-llamafactory-assistant-monitor.log`。本文档中的命令只是安装说明；代码实施本身不会自动安装或启动 plist。

### 恢复与排错

- API 重启：工作流、审批、外部 job ID、事件和调度动作均从 SQLite 恢复；重新打开“助手”页即可继续。
- monitor 中断：手动重跑上述 `--once` 命令或重新加载 plist；过期 lease 会自动回收。
- SSH/GPU 短时不可用：连续 5 次探测失败后标记需关注，仍每 15 分钟重试，不会把远端训练误判为失败。
- 工作流长时间留在 `cancelling`：先看 `cancellation_needs_attention` 事件和 monitor 日志。连续五次 SSH/身份校验失败后会转为 15 分钟重试，但不会绕过 PID 身份门槛或误报 `cancelled`。修复 SSH/远程标记后重跑 worker 即可对账。
- 远程训练中断：助手保留并展示绝对 checkpoint 路径、最近日志摘要和 `oom` / `distributed` / `tokenizer` / `dataset` / `runtime` 等失败分类。若存在 checkpoint，会生成“检查并恢复训练”审批；确认后先重新预检，再生成新的训练审批。两次确认之间不会自动续训、重启、停止或删除任何任务。
- 评测或本地打分失败：远程 predictions 保留，工作流回到 A/B 方案已就绪状态，并生成新的“重试评测”和“跳过评测”审批；修复 judge/配置后可重跑，无需重新训练。

建议排查顺序是先看助手事件流，再看 monitor 日志、对应数据生成/训练/评测日志，最后核对 `server_config.yaml` 与环境变量。

## 模型部署

把训练产物或基座用 vLLM (docker) 部署到服务器, 参数全可配置：

- `deploy_schema.DeployConfig` 把 vLLM 启动参数建模 (max_model_len / gpu_memory_utilization / max_num_seqs / reasoning_parser / tool_call_parser / speculative_config / LoRA 等), 容器名自动补 `vllm-` 前缀。
- `POST /api/deploy` 生成 `docker run -d` 命令远端执行; `GET /api/deploy` 用 `docker ps -a` 查状态 (容器即真相); `POST /api/deploy/{name}/stop` 停止+删容器; `DELETE /api/deploy/{name}` 连本地配置一起删; `GET /api/deploy/{name}/logs` SSE 日志。
- 本地存一份配置 `deploy_configs/<name>.json` 便于重建, 但状态不依赖它 (容器在不在看 docker ps)。
- 前端"部署"tab: 表单填参数 → 启动 → 列表看状态/端口 → 日志/停止/删除。

### 续训与产物管理

- **续训**：训练表单选数据来源后, 在 "Checkpoint / 产物管理" 面板点"刷新列表"加载某训练任务的 `checkpoint-*`, 点"续训"把绝对路径填入 `resume_from_checkpoint`。提交时若带该字段, `output_dir` 会复用原目录 (不追加 job_id), 使 checkpoint 可被正确加载。
- **产物管理**：`GET /api/jobs/{job_id}/checkpoints` 列出某训练任务的 checkpoint 及大小; `POST /api/jobs/{job_id}/cleanup` 删除指定 checkpoint (名必须形如 `checkpoint-N`, 防 path traversal); `GET /api/jobs/{job_id}/download?name=checkpoint-N` 流式 tar 下载。
- **生成直送训练**：训练页数据来源选"从生成任务导入", 直接用已完成生成任务的 `output.json` 送训, 省去下载-上传往返。`GET /api/datagen/jobs` 列出所有生成任务。

### 评测多 adapter 合并

被测模型若**同基座不同 LoRA adapter**, 自动合并到一个 vLLM 端点: 一次冷启动加载基座权重, 组内每模型用 `--adapter` 切换 (省 N-1 次权重加载)。不同基座仍各自起端点。

## 测试

```bash
python -m pytest llamafactory_pipeline/test_remote.py llamafactory_pipeline/test_eval.py \
  llamafactory_pipeline/test_datagen.py llamafactory_pipeline/test_app.py -q
```

纯逻辑测试, 不需要真实服务器/GPU/LLM。

## 未包含

多服务器调度、任务队列、鉴权、超参搜索。多任务靠 GPU 占用软提醒 (提交前查 `/api/gpus`), 不自动排队。
