# LlamaFactory 远程训练 Pipeline 设计

## 目标

在仓库根目录新增独立的 `llamafactory_pipeline/`，通过本地 Web 页面配置并启动服务器上的 LlamaFactory Qwen3 LoRA SFT 任务。训练参数严格来自官方 `examples/train_lora/qwen3_lora_sft.yaml`，本地 ShareGPT 数据通过 SSH 上传，训练日志实时返回浏览器。浏览器关闭或本地服务重启不能终止远端训练。

## 初版范围

- 单台已安装 LlamaFactory 的 Linux 训练服务器。
- 使用本机 `ssh`、`scp` 和已有 SSH 密钥或 ssh-agent。
- 上传一个小于 500 MB 的 JSON 或 JSONL 数据文件。
- 支持 `conversations/from/value` 和 `messages/role/content` 两种 ShareGPT SFT 格式。
- 展示官方 `qwen3_lora_sft.yaml` 的全部字段，包括默认未启用的 eval 字段。
- 创建、查询和恢复任务，使用 SSE 实时显示训练日志。

初版不包含多服务器调度、并发队列、浏览器断点上传、模型或训练产物下载、任务取消、用户系统和权限管理。

## 目录

```text
llamafactory_pipeline/
├── __init__.py
├── app.py
├── schema.py
├── remote.py
├── defaults.yaml
├── static/
│   └── index.html
├── test_remote.py
└── README.md
```

该模块复用仓库已有的 FastAPI、Uvicorn、Pydantic 和 PyYAML，不修改现有 RAG API 或 React 前端，不新增第三方依赖。

## 配置

本地服务只从环境变量读取基础设施配置：

- `TRAIN_SSH_TARGET`：`user@host`，必填。
- `TRAIN_SSH_PORT`：SSH 端口，默认 `22`。
- `TRAIN_SSH_IDENTITY`：可选私钥路径；未设置时使用 ssh-agent 或 SSH 默认配置。
- `TRAIN_REMOTE_ROOT`：远端任务根目录，必填。
- `LLAMAFACTORY_DIR`：远端 LlamaFactory 安装目录，必填。

浏览器不接收或保存 SSH 凭据。服务默认只监听 `127.0.0.1`。

## 参数模型

`schema.py` 使用一个禁止额外字段的 Pydantic 模型描述官方 YAML 参数。字段按 `model`、`method`、`dataset`、`output`、`train` 和 `eval` 分组，并保留官方默认值、布尔类型、数值范围和枚举选项。

前端从 `GET /api/schema` 获取 JSON Schema 并生成表单，避免维护第二份参数清单。上传数据后，系统自动把 `dataset` 设置为当前任务的数据集名称。

系统会在最终 YAML 中增加不可由前端编辑的 `dataset_dir`，使每个任务使用独立数据目录。这是运行隔离信息，不属于用户可配置训练参数。

## API

- `GET /`：返回静态训练页面。
- `GET /api/schema`：返回参数 Schema 和默认值。
- `POST /api/jobs`：multipart 请求同时提交参数 JSON 和数据文件；校验、上传并启动训练，返回任务 ID。
- `GET /api/jobs/{job_id}`：查询远端任务状态。
- `GET /api/jobs/{job_id}/logs`：以 SSE 返回日志；支持 `Last-Event-ID` 从指定行恢复。

任务 ID 由 UTC 时间和随机后缀组成，只允许安全字符。所有任务路径都由服务端生成，用户输入不能成为 shell 命令或远端路径。

## 数据处理

1. FastAPI 使用 `UploadFile` 流式写入本地临时文件，并在复制过程中限制文件大小为 500 MB。
2. 仅接受 `.json` 和 `.jsonl`。
3. 读取首条记录识别 ShareGPT 结构，并校验消息列表、角色键和内容键。
4. 为任务生成唯一数据集名称和对应的 `dataset_info.json`。
5. 使用 `scp` 把数据、`dataset_info.json` 和最终 YAML 上传到远端任务目录。
6. 无论成功或失败，提交流程结束后删除本地临时文件。

对 JSONL 逐行完成 JSON 语法校验。对 JSON 数组只校验顶层数组和首条记录，避免把接近 500 MB 的文件整体载入内存；完整语法和数据语义继续由 LlamaFactory 负责。

## 远端任务布局

```text
<TRAIN_REMOTE_ROOT>/<job_id>/
├── data/
│   ├── train.json 或 train.jsonl
│   └── dataset_info.json
├── qwen3_lora_sft.yaml
├── train.log
├── pid
└── exit_code
```

远端启动器使用固定命令模板：

```text
cd <LLAMAFACTORY_DIR>
nohup sh -c 'llamafactory-cli train <config>; printf "%s" "$?" > <exit_code>' \
  > <train.log> 2>&1 < /dev/null &
printf "%s" "$!" > <pid>
```

路径使用 `shlex.quote`，参数值只写入 `yaml.safe_dump` 生成的配置，不插入 shell 命令。

## 状态与日志

远端目录是任务状态的唯一事实来源，不使用本地数据库：

- 存在 `exit_code` 且值为 `0`：`succeeded`。
- 存在 `exit_code` 且值非 `0`：`failed`。
- 无退出码且 `kill -0 <pid>` 成功：`running`。
- 只有任务目录或 PID 不可用：`unknown`。
- 任务目录不存在：`not_found`。

日志接口通过 SSH 执行 GNU `tail`，逐行转换为 SSE，并给每行分配递增事件 ID。浏览器重连时使用 `Last-Event-ID` 跳过已显示日志。日志流附带心跳；SSH 中断时关闭当前流，由浏览器自动重连。训练任务不依赖日志连接存活。

## 前端

单页使用原生 HTML、CSS 和 JavaScript，不新建第二套 Node 工程：

- 六组折叠参数表单。
- 数据文件选择器及格式、大小提示。
- 启动按钮和当前任务 ID。
- `submitting`、`running`、`succeeded`、`failed`、`unknown` 状态展示。
- 使用 `EventSource` 的日志终端，自动滚动并支持手动重连。
- 使用 `localStorage` 保存最近任务 ID，本地页面重开后恢复查询。

## 错误处理

- 环境变量不完整或本机找不到 `ssh/scp`：启动时直接报错。
- SSH 主机验证、认证或连接失败：返回明确错误，不关闭主机密钥检查。
- 文件超限、扩展名或 ShareGPT 结构错误：上传阶段拒绝，不创建远端任务。
- 上传部分失败：保留远端任务目录供排查，但不启动训练。
- 启动命令失败：记录到 `train.log` 并写入非零退出码。
- 状态查询 SSH 失败：返回 `unknown` 和错误摘要，不把训练标记为失败。
- HTTP 错误内容不包含私钥、完整 SSH 命令或服务器敏感环境变量。

## 验证

最小自动化测试覆盖：

- 官方默认 YAML 能通过参数模型。
- 额外训练字段会被拒绝。
- 两种 ShareGPT 首条记录均能生成正确 `dataset_info.json`。
- 不合法文件、超限文件和不安全任务 ID 被拒绝。
- SSH 命令中的目录经过安全引用。
- 根据 PID 和退出码解析出正确状态。

使用替代 `ssh/scp` 可执行文件完成本地集成测试，不依赖真实服务器。真实服务器验收流程为：上传小型 ShareGPT 样例、启动任务、观察日志、关闭并重启本地服务、按任务 ID 恢复状态和日志。
