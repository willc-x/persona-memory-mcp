# 聊天记录蒸馏与本地人物记忆镜像（通用版）

这是一个面向**本机、单用户、可审计**的聊天记录处理工程。它把导出的聊天文本准备成带来源溯源的本地语料，可选地对训练段落做证据约束蒸馏，把结果接入本地向量检索和人物运行目录，并通过本地 Web UI 或 stdio MCP 提供检索与（另行授权的）AI 模拟聊天。

本项目只附带通用代码、合成测试、空配置模板和第三方许可。

聊天记录不限定来源客户端；只要按第 4 节导出为受支持的 UTF-8 行式文本或 JSON，即可进入同一套本地处理流程。仓库内不应包含真实聊天、真实人物画像、密钥、运行数据库或模型权重。

> **完成状态不是由“跑过一次命令”推断的。** 只有 `meta.json.distillation_complete=true`、计划中的全部训练分段均有完成记录，且 `integrate_persona.py` 写入 `status=complete` 后，该项目实例才可称为“全量蒸馏完成”。仅导入历史、只跑 baseline 或 trial，都必须明确标为未完成。**本次通用代码交付没有执行任何真实聊天语料的全量蒸馏，也不声称完成了任何具体人物的全量蒸馏。**

---

## 1. 能力与边界

流程分为四层，每一层都必须显式触发：

1. **本地准备**：把 UTF-8 聊天文本规范为带 `message_id`、源文件 SHA-256、源行号、发送者和角色 A/B 的 `messages.json`。
2. **本地导入**：写入本地真实历史 SQLite 和向量索引。检索不调用大模型。
3. **有授权的蒸馏（可选）**：先生成本地 `plan.json`；试跑或全量运行必须额外生成精确绑定服务、模型、语料 SHA、计划 SHA、调用次数和预算的授权。只有被选中的聊天内容会发送给所配置的服务。
4. **本地运行**：Web UI 的搜索不调用模型；对话调用必须另有 `chat` 授权，并使用与真实历史隔离的 AI 会话库。

必须遵守的限制：

- 只监听 loopback；不支持远程多用户部署，不把 API 暴露到公网。
- 不内置任何供应商、模型、端点或密钥。`config/provider.example.json` 只有占位符。
- 只读取项目专用的 `CHAT_LLM_API_KEY`，不读取其他软件的 Keychain、dotenv 或环境变量。
- MCP 的检索工具不调用模型；模拟聊天必须同时开启显式开关并具有 `chat` 授权。
- 不提供“自动绕过登录”“自动抓取聊天”“安全抹除磁盘”等能力。
- 当前启停脚本面向 macOS/Linux（启停依赖 POSIX shell 与 `lsof`；用量账本还使用 `fcntl` 文件锁）；Windows 未验证。

---

## 2. 环境要求

- Python **3.10+**，推荐 3.11/3.12；以 `config/requirements-local.txt` 的锁定版本为准。
- 约 2 GB 以上可用磁盘（PyTorch、本地 embedding 模型、向量库、聊天备份都会占空间）。
- macOS/Linux。启动、停止脚本优先使用 `lsof -n0`；不支持 NUL 字段的旧版 lsof 会回退到换行字段解析，仍严格核对进程 cwd 与命令行。
- 首次安装依赖需要访问包索引；embedding 模型可以从公开 Hugging Face 固定 revision 下载。除用户明确发起的蒸馏/聊天外，程序不调用模型。
- 不强依赖 GPU；本地 embedding 固定 CPU 推理。

真实聊天数据只能写入**项目源码目录之外的私有 `DATA_ROOT`**。仓库本身只放代码、文档、样例配置和合成测试，绝不能在仓库内创建真实 `data/`、`prepared/`、`reports/` 或人物产物。先创建并固定一个私有目录：

```bash
cd /path/to/persona-memory-mcp
export CHAT_DATA_DIR="/absolute/private/path/outside/repository"
export DATA_ROOT="$CHAT_DATA_DIR"
install -d -m 700 "$DATA_ROOT"
```

`CHAT_DATA_DIR` 必须在启动服务、导入、蒸馏、配置授权和 MCP 的每个进程中保持一致。不要使用相对路径、项目根、`/tmp`、公共网盘或被 Git 同步的目录；不要把真实数据目录放在源码树内。若省略 `CHAT_DATA_DIR`，程序出于合成测试/演示目的会回退到项目根，但该回退路径**不得写入任何真实聊天或人物数据**。

---

## 3. 安装

### 3.1 创建虚拟环境并安装锁定依赖

```bash
cd /path/to/persona-memory-mcp
export CHAT_DATA_DIR="${CHAT_DATA_DIR:?请先按第 2 节设置源码树之外的私有数据目录}"
./scripts/setup.sh
```

`scripts/setup.sh` 会：

- 创建项目内 `.venv`；
- 安装根 `requirements.txt`（锁定运行时依赖并加入 `mcp>=1.10,<2`）；
- 调用 `setup.py` 创建数据目录和**不含密钥**的配置示例。

它不会读取聊天、不会启动服务、不会发起模型请求。若系统 Python 不是 3.10+，先安装合适版本；脚本不会 `sudo`，也不会修改全局 Python。

已自行准备好虚拟环境时可只执行：

```bash
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python setup.py
```

注意：`requirements.txt` 引用 `config/requirements-local.txt`，不要再叠加引用 `ex-memory/requirements-mcp.txt`；后者会重新选择另一套 `uvicorn` 版本。

### 3.2 下载固定版本的本地 embedding

```bash
.venv/bin/python scripts/download_embedding.py --yes-public-download
.venv/bin/python scripts/download_embedding.py --verify
```

固定身份：

- 模型：`BAAI/bge-small-zh-v1.5`
- revision：`7999e1d3359715c523056ef9478215996d62a620`
- 主权重 SHA-256：`354763b9b1357bc9c44f62c6be2276321081ed2567773608c0d0785b61d5a026`
- 主权重大小：`95827648` bytes

下载器只访问固定公开 Hugging Face 文件 URL，固定 revision 和文件清单，不使用匿名 token；HTTP 客户端可能跟随重定向，但最终内容必须逐文件满足完整的 SHA-256 与字节数校验，权重响应还必须匹配固定 ETag/长度，任何不符即拒绝发布。下载先进入临时 staging，逐文件校验后再原子发布。`--verify` 完全离线，用于断网环境复核已有目录和 manifest。

模型缓存在 `$CHAT_DATA_DIR/data/models/bge-small-zh-v1.5`，manifest 在 `$CHAT_DATA_DIR/reports/embedding-model-manifest.json`。不要在下载或校验时手工修改这些文件。

### 3.3 配置模型服务（仅在使用蒸馏或聊天时需要）

```bash
cd /path/to/persona-memory-mcp
cp config/provider.example.json "$DATA_ROOT/config/provider.json"
cp config/provider.env.example "$DATA_ROOT/config/provider.env"
chmod 600 "$DATA_ROOT/config/provider.json" "$DATA_ROOT/config/provider.env"
```

编辑 `provider.json`，至少填写：

| 字段 | 要求 |
| --- | --- |
| `service_name` | 你自己的服务标识，会进入授权指纹 |
| `base_url` | 本机回环 `http://127.0.0.1:.../v1`，或经 HTTPS 的远端端点；禁止 URL 内嵌账号、查询参数 |
| `model` | 精确模型名 |
| `context_tokens` | 实际上下文预算，至少 4096 |
| `prices.input_rmb_per_million` / `prices.output_rmb_per_million` | 全量蒸馏做固定预算时必须填写非负单价；未知价格不能被当作 0 |
| `billing_mode` | `priced` 或 `usage_only`；usage-only 路由仍需显式确认 |

在 `provider.env` 中填入自己的 `CHAT_LLM_API_KEY=...`，或只在当前终端 `export CHAT_LLM_API_KEY=...`。**不要把密钥粘到聊天、issue、日志或测试代码里。** 对允许无密钥的本机回环服务，适配层才可能使用明确标记的本地占位值。

`provider.json` 的任何绑定字段（服务名、URL、模型、上下文、价格）变化都会改变指纹；旧授权随即失效，必须重新确认。

兼容说明：当 `base_url` 的 hostname 恰为 `api.deepseek.com` 时，受控客户端会附加该服务约定的“关闭 thinking”请求字段，以避免推理内容耗尽 JSON 输出预算。该字段不读取密钥；hostname 本身已属于授权指纹，不能在不重新授权的情况下替换端点。

---

## 4. 准备真实数据

### 4.1 先脱敏与留存原件

聊天记录属于高敏感数据。建议：

1. 从导出工具得到纯文本，保留原文件只读副本，不放入 Git；
2. 删除与蒸馏目标无关的第三方隐私、验证码、住址、账号和支付信息；
3. 以 UTF-8 保存；如原始编码为 GB18030，可交给 `prepare --encoding gb18030` 读取；
4. 先确认目标人物本人和对话参与者的授权范围。

### 4.2 文本格式

`prepare` 按**一行一条消息**读取，格式必须是：

```text
[2025-01-02 03:04:05] 发送者: 消息正文
[2025-01-02 03:05:10] 我: 另一条消息
```

同一发送者在导出中的别名要在准备时显式映射，避免角色 A/B 被错误交换。目标人物（B）必须由 `--target` 指定；本人（A）默认“我”。

### 4.3 生成通用语料包

```bash
.venv/bin/python scripts/chat_memory_tool.py prepare \
  --input /path/to/private/chat-export.txt \
  --target "目标人物" \
  --target-alias "目标人物别名1" \
  --target-alias "目标人物别名2" \
  --self-name "我" \
  --out "$DATA_ROOT/prepared/persona"
```

角色别名映射在**规划之前**固定。建议先加 `--expect-lines N` 核对行数，再查看：

- `audit.json`：行数、跳过/隔离、源 SHA-256；
- `source_ledger.jsonl`：逐行来源账本；
- `suhui/messages.json`：后续规划和导入使用的公开消息 schema；
- `ex_memory.json`：本地导入的规范化语料；
- 异常、重复和时间异常账本。

默认遇到无法解析的行会失败；只有确认可隔离时才加 `--allow-quarantine`。`prepare` 不调用网络，也不调用模型。

### 4.4 导入本地真实历史

```bash
.venv/bin/python scripts/import_history.py \
  --prepared "$DATA_ROOT/prepared/persona" \
  --slug persona
```

导入要求 prepared 位于 `CHAT_DATA_DIR` 内、`audit.json` 的源 SHA 未变；默认 slug 为 `persona`。它会：

- 在 `$DATA_ROOT/data/real_history.sqlite` 建立带 `origin=real_chat` 的真实历史；
- 在 `$DATA_ROOT/exes/...`（或 installation 已记录的自定义镜像）建人物镜像骨架；
- 在 `$DATA_ROOT/runtime/installation.json` 记录本地历史状态和人物目录；
- 生成仅本机使用的 `$DATA_ROOT/config/local-login.json`（0600）并写入 Web 登录账号。

**不要展示、复制或提交 `local-login.json` 中的密码。** 首次登录使用该文件里生成的本机账号；若忘记，只能停服后备份并重新生成本地账号，不能恢复明文。

重复导入完全相同的源不重置已有的人物状态；导入新源会保留原 `persona_directory`（已完成的画像不会被覆盖），同时在报告和 installation 中标记 `stale_after_history_append=true`。这只说明旧画像尚未覆盖新增历史，**不得声称旧画像已经覆盖新记录**；需要按第 10 节重新规划、蒸馏并整合。导入后的报告始终标注 `persona_distillation_completed=false`，只完成“本地历史可用”不等于蒸馏完成。

---

## 5. 生成全量计划

规划纯本地，不需要服务、密钥或授权。身份字段、打包/holdout、费用上界、checkpoint 和产物校验的完整说明见 [distillation/README.md](distillation/README.md)。

规划命令示例：

```bash
.venv/bin/python -m distillation.pipeline plan \
  --messages "$DATA_ROOT/prepared/persona/suhui/messages.json" \
  --out "$DATA_ROOT/artifacts/persona/run" \
  --target-name "目标人物" \
  --slug persona \
  --user-alias "我" \
  --context-tokens 16000 \
  --reserved-tokens 4000 \
  --pack-window week \
  --input-price 1.00 \
  --output-price 2.00
```

- 默认上下文是 **16000** tokens；`--reserved-tokens` 当前必须与模型输出上限一致，默认 **4000**。
- 输出严格写到 `--out/plan.json`；若该目录已经存在 `segments/` 或 `merges/` 检查点，程序拒绝覆盖，请换新的工作目录。
- `--slug` 必须是通用小写标识（如 `persona`），显示名 `--target-name` 不参与源文件哈希；本次版本的新计划会把身份元数据纳入自身 `plan_hash`，不要混用旧版本计划。
- 不传 `--input-price/--output-price` 时，计划仍可用于试跑规划和次数授权，但全量固定预算无法得到保守费用，`distill_full` 固定费用授权会被拒绝。
- `plan.json` 包含训练段、holdout 会话、token 上界、调用次数和费用上界。holdout 只隔离不发送，当前没有自动质量评估；不能把“计划完成”写成“质量合格”。

把 `--out` 与后续产物放在私有目录，不要放在源码目录；默认 `.gitignore` 已忽略 `artifacts/`。

---

## 6. 授权、试跑与全量蒸馏

授权是**精确绑定**，不是“全局同意”。每次 trial/full/chat 授权都必须由用户在阅读计划、来源哈希、费用上界和发送范围后**亲自确认**；自动化脚本或 AI Agent 只能准备并展示待执行命令，不得代替用户作出同意，也不得自行添加 `--confirm`、`--confirm-full-cost`、`--confirm-route-usage` 或 `--allow-usage-trial-or-chat`。检查 `provider.json` 和计划后，再分别授权：

### 6.1 少量试跑

```bash
.venv/bin/python scripts/configure.py \
  --purpose distill_trial \
  --plan "$DATA_ROOT/artifacts/persona/run/plan.json" \
  --grant-id persona-trial-001 \
  --max-calls 8 \
  --max-cost-rmb 10 \
  --confirm
```

试跑次数不能超过计划前两个训练段的保守上限。`configure.py` 只写 `$DATA_ROOT/config/privacy-consent.json`，不创建客户端、不读取密钥、不发送任何内容。

执行试跑：

```bash
.venv/bin/python -m distillation.pipeline run \
  --plan "$DATA_ROOT/artifacts/persona/run/plan.json" \
  --limit 2
```

试跑结果状态是 `trial`，即使产物完整也不能标为全量完成。

### 6.2 全量固定预算授权

先从计划中核对 `calls.total_planned`、`cost.estimated_cost_rmb`，并把预算设为不低于计划保守上界：

```bash
.venv/bin/python scripts/configure.py \
  --purpose distill_full \
  --plan "$DATA_ROOT/artifacts/persona/run/plan.json" \
  --grant-id persona-full-001 \
  --max-calls 200 \
  --max-cost-rmb 50 \
  --confirm-full-cost \
  --confirm
```

执行：

```bash
.venv/bin/python -m distillation.pipeline run \
  --plan "$DATA_ROOT/artifacts/persona/run/plan.json" \
  --full
```

只有全部训练段完成、没有中途失败时，产物状态才是 `complete`。运行节点永久最多两次尝试，失败后保留已核验 claim 和 `incomplete` 状态，不会把失败伪装成成功。重试前核对检查点和计划哈希，必要时换新的工作目录重新规划。

### 6.3 usage-only 路由

如果服务按路由计费、无法给出固定人民币单价，可以在 `provider.json` 设置 `billing_mode: "usage_only"`，并使用：

```bash
.venv/bin/python scripts/configure.py \
  --purpose distill_full \
  --plan "$DATA_ROOT/artifacts/persona/run/plan.json" \
  --grant-id persona-route-full-001 \
  --max-calls 200 \
  --confirm-route-usage \
  --confirm
```

usage-only 与固定费用预算是互斥模式，不能互相冒充。试用或聊天若确需走 usage-only，还必须额外加 `--allow-usage-trial-or-chat`；默认策略是拒绝。使用前应自行确认该路由的账单与数据政策。

### 6.4 允许 AI 聊天

```bash
.venv/bin/python scripts/configure.py \
  --purpose chat \
  --grant-id persona-chat-001 \
  --max-calls 1000 \
  --max-cost-rmb 20 \
  --confirm
```

聊天授权同样绑定当前来源 SHA、提供商指纹和次数/预算。没有聊天授权时，Web/UI 的模拟对话和 MCP `simulate_persona` 都必须拒绝发送。

### 6.5 从已保存的失败响应恢复节点

`distillation.pipeline run` 对每个节点最多请求两次。若两次都只得到可解析性错误（`invalid_response`），可把已经落盘的原始响应恢复为本地验证过的 checkpoint，而**不重新调用模型**：

```bash
.venv/bin/python scripts/recover_failed_node.py \
  --workdir "$DATA_ROOT/artifacts/persona/run" \
  --segment-id segment_000000 \
  --kind persona \
  --attempt 2 \
  --usage-event 17 \
  --grant persona-full-001
```

脚本只接受以下同时成立的情况：计划、consent、usage、checkpoint 与诊断文件都在 `CHAT_DATA_DIR` 内；有唯一且计划哈希匹配的 `distill_full` 授权；指定 usage event 是该授权下已完成的真实请求；诊断响应 SHA 与文件一致；节点仍处于失败、且只能恢复其最后一次尝试。恢复后写 `provider_recall=false`、`network_calls=0`，不会增加第三次尝试。若整段 JSON 在输出上限处被截断，可追加 `--salvage`：只解析已经闭合的独立 claim，再逐条执行同一套原话、角色、时间和来源校验；不会补全截断文本，也不会猜造缺失字段。

如需自动驱动全量运行并只对 `invalid_response` 走上述恢复，可使用：

```bash
.venv/bin/python scripts/drive_full_run.py \
  --plan-dir "$DATA_ROOT/artifacts/persona/run" \
  --grant persona-full-001 \
  --max-recoveries 10
```

`provider_error`、授权缺失、usage 事件不存在或诊断不唯一时都会停止，不会靠重发聊天内容掩盖问题。

---

## 7. 接入人物运行目录并启动

### 7.1 整合蒸馏产物

```bash
.venv/bin/python scripts/integrate_persona.py \
  --source "$DATA_ROOT/artifacts/persona/run"
```

整合器会重新核对 artifact 完整性、claim 的原话/发送者/时间/来源、canonical 范围以及完成状态。只有状态为 `complete` 的产物才写入完成 manifest；非 complete 不伪装成全量。它会先备份原生 `SKILL.md` 与 `meta.json`，再原子切换运行目录。

### 7.2 启动本地服务

```bash
./start.sh
```

默认只监听 `127.0.0.1:8765`：

- UI：<http://127.0.0.1:8765/chat>
- 健康检查：<http://127.0.0.1:8765/chat/health>

`start.sh` 使用 `.venv/bin/python`，也可用 `CHAT_PYTHON` 显式覆盖。它只启动一个由本项目 cwd/命令匹配的 uvicorn，等待健康检查；失败时只回收刚启动且仍匹配的 PID。PID/状态写入项目 `runtime/`，日志写入项目 `logs/`；不要把其他项目或端口上的服务当作本项目服务管理。

访问 UI 时使用本地账号和 `local-login.json` 里的密码。登录后搜索默认不调用模型；聊天必须已有 `chat` 授权。

### 7.3 停止

```bash
./stop.sh
```

`stop.sh` 不按端口猜测进程，也不会向 cwd 或命令行不匹配的 PID 发信号。它先 TERM，等待 15 秒；**不会强杀**不退出或不匹配的进程。状态文件异常时只会清理状态文件并提示人工检查。

---

## 8. MCP 接入

MCP 是前台 stdio 服务，不是另一个 HTTP 服务。详见 [docs/MCP.md](docs/MCP.md)。

```bash
export CHAT_DATA_DIR="/path/to/private-data"
export CHAT_MCP_OWNER=1
export CHAT_MCP_SLUG=persona
./run_mcp.sh
```

- stdout 只输出 JSON-RPC，运行状态走 stderr；
- `memory_search`、`persona_status`、`list_periods` 只读本地状态，不调用 LLM；
- `simulate_persona` 需要 `DSH_MCP_CHAT_AUTH=true` **且**已有 `chat` 授权，回复正文不带可见模拟/AI 标签，会话与真实历史分开存储，内部标为 AI 模拟；
- 工具不返回原始文件路径、完整 prompt、密钥或任意写接口；
- owner/slug 必须与安装/导入时一致，且由环境变量配置，不写死在代码里；
- 默认 `CHAT_MCP_SLUG=persona`，owner 默认 `1`；生产环境应显式设置，不要依赖默认值。

MCP 客户端配置示例（路径按实际修改，stdout 必须是 JSON-RPC）：

```json
{
  "mcpServers": {
    "local-persona-memory": {
      "command": "/path/to/persona-memory-mcp/run_mcp.sh",
      "env": {
        "CHAT_DATA_DIR": "/path/to/private-data",
        "CHAT_MCP_OWNER": "1",
        "CHAT_MCP_SLUG": "persona",
        "DSH_MCP_CHAT_AUTH": "false"
      }
    }
  }
}
```

---

## 9. 备份、恢复、删除

### 9.1 备份

先停止服务，再备份**数据根**，不要备份源码中的示例和 Git：

```bash
./stop.sh
export CHAT_DATA_DIR="${CHAT_DATA_DIR:?请先设置私有 DATA_ROOT}"
export DATA_ROOT="$CHAT_DATA_DIR"
umask 077
.venv/bin/python scripts/manage_data.py --list
.venv/bin/python scripts/manage_data.py --backup
```

`--list` 只读展示将要归档的目录，不创建文件；`--backup` 在 `$DATA_ROOT/backups/` 下原子创建 `0600` 的 gzip 包，目录权限为 `0700`。默认归档固定目录 `config`、`data`、`prepared`、`reports`、`artifacts`、`runtime`，并读取本机安装记录 `runtime/installation.json` 中的 `mirror_directory` 与独立的 `persona_directory`。因此即使镜像位于自定义的 `$DATA_ROOT/ex-memory/exes/...`（而不是默认 `exes/`），Chroma 向量库和 AI 会话也会进入备份；被父目录已经覆盖的人物目录不会重复归档。所有安装路径都必须是 `DATA_ROOT` 内的绝对路径，否则备份失败关闭。`logs/` 和 `backups/` 自身不在归档集内。

需要指定文件名时使用绝对路径，且必须直接位于 `$DATA_ROOT/backups/`：

```bash
.venv/bin/python scripts/manage_data.py --backup \
  --output "$DATA_ROOT/backups/persona-$(date +%Y%m%d-%H%M%S).tgz"
```

备份包含真实聊天、向量库、登录密码、授权、可能的 provider 凭据和模型 manifest，必须按最高敏感级保存。`config/requirements-local.txt`、`provider.example.json` 和源码不随数据一起复制。不要手工改成只打包固定的 `exes/`：自定义镜像会漏掉，恢复后可能只剩索引而丢失人物与 AI 记忆。

### 9.2 恢复

恢复前停止服务，解压到新的临时目录并先检查 manifest、SHA 和文件所有权；确认后再替换数据根中的对应目录。**不要**直接覆盖正在运行的 SQLite/向量库，也不要恢复不属于本机的 `local-login.json` 或授权文件。恢复后重新运行离线校验：

```bash
.venv/bin/python scripts/download_embedding.py --verify
```

并逐项核对 `runtime/installation.json`、计划哈希、来源哈希和授权指纹。只恢复数据库而不恢复相应配置，会导致检索或聊天门禁拒绝，这是预期的 fail-closed 行为。

### 9.3 删除

删除是破坏性操作。先停止服务、完成独立备份，并确认路径变量确实指向目标数据根：

```bash
./stop.sh
printf 'DATA_ROOT=%s\n' "$DATA_ROOT"
.venv/bin/python scripts/manage_data.py --list
# 仅在逐项确认、且 DATA_ROOT 非 / 或空字符串后执行
rm -rf "$DATA_ROOT/data" "$DATA_ROOT/prepared" "$DATA_ROOT/artifacts" \
       "$DATA_ROOT/reports" "$DATA_ROOT/runtime" \
       "$DATA_ROOT/logs" "$DATA_ROOT/backups"
rm -f "$DATA_ROOT/config/provider.json" "$DATA_ROOT/config/provider.env" \
      "$DATA_ROOT/config/privacy-consent.json" "$DATA_ROOT/config/local-login.json"
```

`--list` 还会显示 `runtime/installation.json` 中实际的人物 `mirror_directory`（默认实现位于 `$DATA_ROOT/exes/...`，也可在 `$DATA_ROOT/ex-memory/exes/...` 等自定义位置）。如果它不在上面的固定删除列表中，必须另行确认目标后删除；只删除固定 `exes/` 会漏掉自定义镜像中的 Chroma 与 AI 会话。不要按未经校验的变量值批量删除。

这不会删除源码、`config/requirements-local.txt` 或示例配置。普通文件删除不等于安全擦除；磁盘快照、备份和文件系统日志可能保留副本。需要合规擦除时使用磁盘/备份平台提供的能力，并在项目外单独评估。

---

## 10. 追加真实记录

新增来源会改变语料范围和授权边界。推荐完整重建一份可审计语料，而不是直接修改已导入的 SQLite：

1. 保留原导出，合并新增消息到一个**完整、去重、UTF-8**的文本；
2. 用新的输出目录重新准备，例如 `--out "$DATA_ROOT/prepared/persona-v2"`；
3. 用同一 slug 运行 `import_history.py`；检查导入报告、来源账本和重复账本。新源会保留原 `persona_directory` 并标记 `stale_after_history_append=true`，旧画像不会自动覆盖新增历史；
4. 重新生成本地计划到全新目录（不要复用旧 `segments/merges`）；
5. 由于来源 SHA 和计划哈希已变化，旧的 trial/full/chat 授权不会继续匹配；按第 6 节重新授权；
6. 跑试跑/全量，再运行 `integrate_persona.py` 更新运行目录；
7. 重启服务并重新验证搜索、健康检查和登录。

只有新增少量消息不能证明“全量完成”；状态仍以最新计划的全部训练段完成情况和 `meta.json` 为准。人物显示名或 slug 改变应视为新的运行目录，不要混用旧授权或旧向量库。

---

## 11. 测试与验证

所有测试只使用临时目录、合成文本、假 embedding/假服务；不读取真实聊天，不访问网络，不调用模型。

```bash
cd /path/to/persona-memory-mcp
.venv/bin/python -m pytest -q
```

根 `pytest.ini` 只收集本项目 `tests/`，不递归收集 vendored `ex-memory`。上游适配相关测试从上游目录运行，例如：

```bash
cd ex-memory
../../.venv/bin/python -m pytest -q \
  tests/test_first_import.py tests/test_privacy_wiring.py tests/test_local_json_provenance.py
```

测试通过只能说明合成路径和离线门禁正确，不能替代真实数据的人工脱敏、费用核对、模型供应商条款审查或质量评估。

---

## 12. 目录结构

```text
.
├── adapters/              # 私有 HTTP 服务、MCP、上游隔离、授权出口、本地记忆
├── distillation/          # 本地计划、证据校验、串行运行、产物写出
├── scripts/               # 准备、导入、整合、授权、模型下载、备份、启停守卫
├── tests/                 # 通用合成测试（禁止真实数据）
├── config/                # 锁定依赖与无密钥示例
├── docs/                  # MCP 与隐私说明
├── third_party/           # 第三方许可原文
├── ex-memory/             # 上游代码（MIT）；其运行数据不写回源码目录
├── setup.py               # 离线创建目录和配置模板
├── start.sh / stop.sh     # 只管理匹配本项目的 uvicorn
├── run_mcp.sh             # 前台 stdio MCP
└── requirements.txt       # 锁定运行时依赖 + MCP 客户端
```

`CHAT_DATA_DIR` 下的运行目录：

```text
data/           真实历史 SQLite、用户数据库、本地模型
prepared/       准备后的敏感语料和审计账本
artifacts/      蒸馏计划、检查点和人物产物
reports/        导入摘要、embedding manifest
exes/           默认人物镜像；实际位置以 installation.json 的 mirror_directory 为准
runtime/        安装状态、PID、用量账本
logs/           运行日志
backups/        用户自行创建的备份
config/         provider.json、provider.env、privacy-consent.json、local-login.json
```

---

## 13. 已知限制

- 不内置真实数据、真实人物卡、密钥或预训练产物；首次启动前必须自行准备。
- 聊天文本解析器只支持文档中的逐行纯文本格式；不保证兼容所有导出工具。
- 角色映射依赖 `--target/--target-alias/--self-name`，错误映射会被 fail-closed 拒绝，但无法自动推断事实真相。
- 检索结果和蒸馏结论只可回溯到已导入语料；不能证明未导入内容或人物内心。
- holdout 只做隔离，当前没有统计意义上的质量评估；trial 与 complete 状态不能互相冒充。
- 本地 embedding 固定公开 BGE revision；更换模型会改变索引身份，必须重新导入并重新校验。
- 启停仅支持本机 UNIX；不对其他目录或端口上的服务负责。
- 无多用户隔离、远程暴露、集群调度、自动安全擦除或灾难恢复保证。
- 任何外部模型服务都可能记录请求。启用蒸馏/聊天前，必须自行审查供应商数据保留、地域和训练政策，并完成参与者授权。

---

## 14. 许可与来源说明

本项目原创适配代码以根 [LICENSE](LICENSE) 的 MIT 条款发布（Copyright © 2026 persona-memory-mcp contributors）。

- `ex-memory/` 是随项目分发的第三方组件，遵循 MIT；原文见 [third_party/EX_MEMORY_LICENSE](third_party/EX_MEMORY_LICENSE)，Copyright © 2026 ex-memory contributors。
- 本项目的通用化封装和流程设计**基于 Suhui/ex-memory 的概念与接口思路**，但**不是对上游项目的直接复制**；上游代码保留其原始版权与许可。
- `BAAI/bge-small-zh-v1.5` 遵循其公开模型许可；本仓库不分发模型权重，下载器只获取固定公开 revision。
- 使用本项目处理任何人的聊天记录前，请获得合法授权，并遵守所在地法律及服务条款。

更多边界见 [NOTICE.md](NOTICE.md)、[docs/PRIVACY.md](docs/PRIVACY.md) 与 [docs/MCP.md](docs/MCP.md)；蒸馏执行层的细节见 [distillation/README.md](distillation/README.md)。