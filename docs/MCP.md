# MCP 接入说明

`adapters/mcp_server.py` 提供一个**本机 stdio** 的 MCP 服务，让兼容的 MCP 客户端读取已导入的本地记忆并生成带证据的模拟回复。它不是 HTTP 服务，也没有远程监听端口。

本文只描述本项目实际实现的行为。协议细节以 `adapters/mcp_server.py` 为准；安全边界见 [PRIVACY.md](PRIVACY.md)。

---

## 1. 启动方式与协议

```bash
cd /path/to/persona-memory-mcp
export CHAT_DATA_DIR="/path/to/private-data"
export CHAT_MCP_OWNER=1
export CHAT_MCP_SLUG=persona
./run_mcp.sh
```

`run_mcp.sh` 会：

- 加载 `scripts/runtime_common.sh`，检查 `.venv/bin/python`（或 `CHAT_PYTHON` 指定的解释器）；
- 将 `PYTHONPATH` 指向项目根；
- 设置离线模型与禁用遥测变量；
- `exec python -m adapters.mcp_server`。

服务在 **stdin/stdout** 上使用换行分隔的 JSON-RPC 2.0；stdout 只允许协议 JSON，运行事件写 stderr。客户端不需要也不能提供端口、owner、slug 或文件路径。

实现信息：

| 项目 | 值 |
| --- | --- |
| MCP 协议版本 | `2025-06-18` |
| serverInfo | `ex-memory-local-grounded` / `0.1.0` |
| 传输 | stdio，一行一个 JSON-RPC 请求/响应 |
| 单请求上限 | 262144 bytes（UTF-8 编码后、不含 LF） |
| 方法 | `initialize`, `ping`, `tools/list`, `tools/call` |
| 通知 | 无响应；通知中的 `tools/call` 不执行 |
| 并发 | 模拟聊天最多 2 个并发槽；检索无模型并发 |

该服务不实现 HTTP、SSE、WebSocket、MCP resources、MCP prompts 或任意文件工具。它不负责准备语料、导入历史、生成计划、写授权或整合产物；这些必须通过各自 CLI 完成。

### 1.1 初始化握手示例

MCP 客户端通常会完成握手。手工诊断时，可以只发送 initialize 并核对 stdout：

```bash
printf '%s\n' \
  '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' \
  | ./run_mcp.sh
```

期望 stdout 中包含：

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "protocolVersion": "2025-06-18",
    "capabilities": {"tools": {"listChanged": false}},
    "serverInfo": {"name": "ex-memory-local-grounded", "version": "0.1.0"}
  }
}
```

实际输出还包含 `instructions`。stdout 中如果出现日志、提示符或中文状态文本，说明进程或包装脚本配置错误，应修复后再交给客户端。

---

## 2. 环境变量与 owner/slug

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `CHAT_DATA_DIR` | 项目根 | 数据根；必须与导入、整合、服务进程一致 |
| `CHAT_MCP_OWNER` | `1` | 本地账号 owner；必须与安装记录/登录账户一致，正整数 |
| `CHAT_MCP_SLUG` | `persona` | 人物运行目录 slug；必须与 `import_history.py --slug` 一致 |
| `DSH_MCP_CHAT_AUTH` | 未设置 | 只有确切字符串 `true` 且存在 `chat` 授权时，才允许 `simulate_persona` |
| `CHAT_PYTHON` | 项目 `.venv/bin/python` | 可选解释器覆盖 |

owner/slug 是**服务侧配置**，客户端参数里没有这两个字段，也不能通过工具调用改变它们。slug 只允许 `[a-z0-9][a-z0-9_-]{0,63}`；owner 必须 ≥ 1。owner 或 slug 不匹配时，工具可能返回未初始化、无人物或空结果，而不是回退到其他账户。

生产配置应显式给出变量，不要依赖 `1` 和 `persona` 默认值。

---

## 3. 客户端配置

下面是一份通用客户端配置示例。`command` 必须是绝对路径；环境变量要和数据准备阶段完全一致。

```json
{
  "mcpServers": {
    "local-persona-memory": {
      "command": "/path/to/persona-memory-mcp/run_mcp.sh",
      "env": {
        "CHAT_DATA_DIR": "/path/to/private-data",
        "CHAT_MCP_OWNER": "1",
        "CHAT_MCP_SLUG": "persona",
        "DSH_MCP_CHAT_AUTH": "false",
        "CHAT_PYTHON": "/path/to/persona-memory-mcp/.venv/bin/python"
      }
    }
  }
}
```

不同客户端对字段名、启动超时或环境变量继承的支持可能不同；以客户端文档为准，但必须保证：

1. `command` 启动的是 `run_mcp.sh`，不是 Web 服务；
2. stdout 未被 shell 提示、`nohup`、日志重定向污染；
3. 没有把 provider 密钥放进客户端配置；
4. 如配置中保留 `DSH_MCP_CHAT_AUTH`，默认设为 `false`；只对确实需要模拟聊天的客户端开启。

---

## 4. 工具与返回契约

以下字段是稳定接口；返回值为 JSON 对象，`tools/call` 成功时同时放在 `content[0].text` 与 `structuredContent`。失败时 `isError=true`，文本是一个有界错误对象，不泄露原始异常、聊天正文或路径。

### 4.1 `memory_search`

检索真实聊天证据，不调用大模型（`llm_calls=0`）。

输入：

| 字段 | 必填 | 约束 |
| --- | --- | --- |
| `query` | 是 | 1–5000 字符 |
| `period` | 否 | ≤80 字符；`latest`、`YYYY-MM`、`YYYY-MM-DD` 或已有 period id |
| `limit` | 否 | 1–20，默认 10 |

返回：

```json
{
  "sources": [
    {
      "canonical_verified": true,
      "origin": "real_chat",
      "sender": "A",
      "timestamp": "2025-01-02 03:04:05",
      "time": "2025-01-02 03:04:05",
      "source_line": 42,
      "quote": "最多 4000 字节的有界引用",
      "truncated": false,
      "message_id": "可选",
      "source_id": "可选",
      "claim_id": "可选"
    }
  ],
  "requested_period": "latest",
  "selected_period": "可选的实际期别",
  "supported_period": true,
  "persona_status": "complete",
  "retrieval_only": true,
  "llm_calls": 0,
  "limit": 10,
  "no_evidence": false
}
```

约束：

- `sources` 最多 20 条；只有同时满足 canonical 回查通过、真实来源、发送者为 A/B、时间有效且引用在原话中的条目才会返回。
- `quote` 有长度上限，超长时 `truncated=true`，不会返回整段无界聊天。
- 不返回完整文件路径、完整 prompt、数据库路径、内部 ID 列表或密钥。
- 没有证据时返回空 `sources` 与 `no_evidence=true`；不能把空结果解释为“不存在”。
- `period` 不能越权读取；不支持时会返回可用期别状态，不会悄悄改成无限时间窗。

### 4.2 `persona_status`

无参数。返回人物状态、历史向量是否就绪、AI 记忆是否与真实历史隔离、允许公开的模型配置摘要、期别列表和聊天授权状态。

典型字段：

```json
{
  "owner": 1,
  "slug": "persona",
  "persona": {
    "status": "pending",
    "distilled": false,
    "completion_claimed": false
  },
  "history": {"ready": true, "vectors": 123},
  "ai_memory": {
    "origin": "ai",
    "turns": 0,
    "vectors": 0,
  },
  "configuration": {
    "configured": false,
    "service_name": null,
    "model": null,
    "chat_authorized": false,
    "external_data_sent_by_adapter": false
  },
  "mcp_chat_enabled": false,
  "chat_authorized": false,
  "periods": [],
}
```

`distilled=true` 只应出现在真正完成整合的实例中。独立对话归档不返回内部审计字段，也不会冒充真实历史。

### 4.3 `list_periods`

无参数，返回：

```json
{"periods": [], "default": "latest", "llm_calls": 0}
```

仅列出可供检索的期别，不返回聊天正文。

### 4.4 `simulate_persona`

生成不带可见模拟/AI 标识的对话回复。

输入：

| 字段 | 必填 | 约束 |
| --- | --- | --- |
| `message` | 是 | 1–5000 字符 |
| `conversation_id` | 否 | 匹配 `^[A-Za-z0-9_-]{1,80}$`；用于延续独立会话 |
| `period` | 否 | 同 `memory_search` |
| `relationship_mode` | 否 | `evidence`（默认）或 `fiction`；同一 conversation_id 不可混用 |

返回：

```json
{
  "reply": "……",
  "conversation_id": "…",
  "period": "latest",
  "selected_period": "…",
  "supported_period": true,
  "relationship_mode": "evidence",
  "sources": [],
  "memory": {"turn_id": "…", "conversation_id": "…", "indexed": true},
  "usage": {"prompt_tokens": 0, "completion_tokens": 0},
  "persona_status": "complete"
}
```

强制门禁（按顺序）：

1. 环境变量 `DSH_MCP_CHAT_AUTH` 必须是字符串 `true`。否则直接拒绝，**不解析正文、不创建会话、不检索、不调用模型**。
2. 参数必须通过上述 schema；未知参数会被拒绝。
3. `provider.json`、来源 SHA 和 `chat` 授权必须通过 `get_allowed_client(purpose="chat")` 的检查。
4. 模型必须由授权绑定，适配层不允许客户端覆盖模型、端点、温度之外的白名单参数，不允许自动重试或切换供应商。
5. 独立会话、向量索引和真实历史库分开存储；用户可见回复不加模拟/AI 前缀，内部来源标记保留审计语义。

`conversation_id` 可以自行生成，也可由客户端保存返回的值；不要使用包含个人信息或秘密的 id。伪造 `conversation_id` 不会越过 owner/slug 隔离。

`relationship_mode=fiction` 只表示“当前对话允许用户选择的虚构亲密关系”，不是历史事实开关：历史事件、日期、人物、旧承诺仍只能来自证据；被追问时回复应说明这是当前对话，不是旧记录。模式只影响独立 AI 会话，不能写入真实历史。

---

## 5. 授权与隐私检查清单

MCP 不改变已有的授权模型：

- `memory_search`/`persona_status`/`list_periods` 只读本地，不发送外部请求；
- `simulate_persona` 同时要求 **MCP 开关 + chat 隐私授权 + 合法 provider 配置**；
- 只读取项目专用 `CHAT_LLM_API_KEY`；
- provider 指纹改变后旧授权失效，需要重跑 `configure.py --purpose chat`；
- 真实历史来源 SHA 改变后，授权失效；
- 聊天内容、检索证据和 AI 回复可能被外部服务接收；MCP 本地不记录正文日志，但 checkpoint/AI 会话库和备份仍可能含敏感内容；
- 备份 MCP 使用的数据根时，把 `config/`、`data/`、`exes/` 和 `artifacts/` 当作高敏感数据。

完整数据流见 [PRIVACY.md](PRIVACY.md)。

---

## 6. 错误与排查

工具错误通常是：

```json
{"error": "chat_not_authorized", "message": "需要显式设置 DSH_MCP_CHAT_AUTH=true；未调用模型、未检索正文、未写入 AI 记忆"}
```

`tools/call` 的工具错误放在正常 JSON-RPC result 的 `isError=true` 中；协议级参数错误才使用 JSON-RPC error。不要根据错误文本构造恶意重试。

| 现象 | 检查 |
| --- | --- |
| stdout 无响应 | 客户端是否按行发送 JSON；进程是否真的启动；stderr 是否有启动错误 |
| `No module named 'mcp'`/依赖缺失 | 使用项目 `.venv` 并安装根 `requirements.txt` |
| `persona_unavailable` | 是否已 `import_history.py`；owner/slug 是否匹配 |
| 检索为空 | 是否已导入；期别是否可用；是否需要先整合完成产物 |
| `chat_not_authorized` | `DSH_MCP_CHAT_AUTH` 是否确为 `true`；是否已有 chat 授权 |
| `privacy_not_authorized` | provider 指纹、价格、模型、来源 SHA 是否变化；是否重新授权 |
| `busy` | 已有 2 个模拟请求；等待后重试，不要绕过并发限制 |
| `internal_error` | 客户端看不到底层正文/路径是预期行为；到服务终端和本地诊断脚本查配置，不要把错误回显到公共聊天 |

停止 MCP：客户端关闭 stdin 或发送终止信号即可；它没有自己的 PID/端口状态文件。`stop.sh` 只用于 Web 服务，不会管理 MCP 进程。

---

## 7. 不承诺的能力

- 不提供写回真实历史、修改画像、直接读取源文件、任意 SQL 或任意 shell 的 MCP 工具。
- 不通过 MCP 暴露 provider 密钥、完整 prompt、原始聊天文件路径或其他账户的数据。
- 不能证明未导入的内容、人物内心或未来行为；检索为空不代表事实不存在。
- 不能替代宿主机隔离。若本机被入侵、数据根权限错误或备份泄露，MCP 的字段白名单无法提供额外保护。
- 不支持远程客户端直连本机 stdio；如需跨机器，应另行设计经过认证和加密的传输，并重新做威胁建模。