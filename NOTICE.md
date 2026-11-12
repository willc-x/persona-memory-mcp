# NOTICE 与第三方来源

本文件补充根 [LICENSE](LICENSE) 和 [README](README.md) 的许可说明。它不改变任何第三方许可证的条款，也不把第三方代码重新授权。

## 1. 本项目

本项目的原创适配代码、文档、脚本与蒸馏执行层以根 `LICENSE` 的 MIT 条款发布：

Copyright © 2026 persona-memory-mcp contributors

## 2. 随仓库分发的 ex-memory

`ex-memory/` 目录包含第三方 `ex-memory` 代码，按 MIT License 分发。许可证原文保存在 [third_party/EX_MEMORY_LICENSE](third_party/EX_MEMORY_LICENSE)：

Copyright © 2026 ex-memory contributors

该组件的版权、免责声明和许可条件均以其原始 LICENSE 为准。本项目没有删除或替换其版权声明。若未来更新 vendored 版本，应同步保留并复核对应许可证和上游 NOTICE。

### 2.1 vendored 来源与本地改动

`ex-memory/` 取自公开仓库 <https://github.com/Meteorkid/ex-memory> 的公开提交 `cc59455fe0e93934eba5d4216bf7ed62992316c1`。为让父仓库直接包含可审计源码、避免用户只得到 submodule 占位，本项目移除了嵌套的 `ex-memory/.git` 元数据，保留上游 `LICENSE`、`README` 和其他随源码发布的文件。

vendored 内容相对该公开提交包含以下本地改动，均无真实聊天、密钥或人物数据：

- `config.py`：增加 `EX_MEMORY_LOAD_DOTENV`、`EX_MEMORY_USE_KEYCHAIN` 环境开关；
- `memory/ingest.py`：本地 embedding identity 先校验、失败前不归档，并调用历史向量库的切片接口；
- `parsers/chat_parser.py`：校验 A/B 角色、`is_target` 与 `origin=real_chat` 一致，并保留来源字段；
- `tests/test_first_import.py`、`tests/test_privacy_wiring.py`：只调整测试替身，提供合成 chunks；
- `tests/test_local_json_provenance.py`：新增全合成来源/失败顺序测试。
- 仓库内的平台特定名称、环境变量、路由和文件路径已统一中性化；未使用的第三方导出适配及其子模块声明已移除。
- 上述中性化不改变数据格式校验、来源校验或证据约束。

这些本地改动仍适用上游 MIT 许可；上游来源与许可证优先于本项目的适配说明。

## 3. 概念与设计来源

本项目的通用封装、数据流和部分接口思路借鉴了 **Suhui / ex-memory** 系列关于“真实历史证据优先、来源可回溯、AI 与真实历史隔离”的概念。项目维护者认为这种借鉴属于通用架构与接口层面的独立实现。

本仓库：

- 不是 Suhui 项目的直接拷贝；
- 不以 Suhui 的名称、人物数据、提示词、私有运行产物或密钥作为本项目内容；
- 不声称得到 Suhui、ex-memory 或任何模型供应商的额外背书；
- 上游 ex-memory 代码仍完整适用其自己的 MIT 许可，而不是被本项目的 MIT 声明覆盖。

如果某个文件或片段实际复制了第三方代码，应以该代码附带的版权和许可证为准；发现遗漏时，应补充 NOTICE 或许可证，而不是用本说明掩盖。

## 4. 模型与数据

本仓库不包含、不再分发：

- 真实聊天记录或由真实聊天派生的人物画像；
- API 密钥、访问令牌、账号密码或本机登录文件；
- 训练好的 embedding 权重、模型 checkpoint 或私有蒸馏产物。

`scripts/download_embedding.py` 只从公开 Hugging Face 固定 revision 获取 `BAAI/bge-small-zh-v1.5`。模型权重适用其发布方提供的许可；使用者应在下载和使用前自行阅读该许可，并遵守模型卡和适用法律。下载器的完整性校验只证明字节与固定清单一致，不构成许可授权或安全审计。

## 5. 免责声明

本项目和第三方组件均按“现状”提供，不附带任何明示或暗示担保。使用者对以下事项自行负责：

- 是否拥有处理聊天记录的合法授权；
- 是否取得对话参与者对蒸馏和模拟的同意；
- 对模型服务的数据保留、训练、跨境传输和费用政策的审查；
- 备份、删除、访问控制以及任何合规或安全擦除要求；
- 不将模拟内容冒充真人陈述或用于欺骗、骚扰或高风险决策。

完整的数据流与威胁模型见 [docs/PRIVACY.md](docs/PRIVACY.md)。