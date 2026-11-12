# 证据约束蒸馏执行层

本目录是通用、可离线规划、可断点续跑的聊天蒸馏执行层。它只负责**把已准备的聊天记录变成有来源约束的候选人物画像**，不附带任何真实人物、真实聊天、密钥或预训练产物，也不直接复制任何外部项目代码。

其设计借鉴了 ex-memory / Suhui 系列的“真实历史优先、证据可回查、AI 与真实历史隔离”概念；实现、schema 与校验逻辑属于本工程，上游 ex-memory 以 MIT 单独分发，详见根 [README](../README.md) 与 [third_party/EX_MEMORY_LICENSE](../third_party/EX_MEMORY_LICENSE)。

> 默认上下文为 **16000 tokens**，默认输出预留为 **4000 tokens**。`reserved_tokens` 必须与当前 prompt 的输出上限一致；调高模型输出上限时，必须同时提高 `context_tokens`。
>
> 本仓库验证只使用合成消息和 fake provider。**没有执行也没有声称执行任何真实语料的全量蒸馏。**

---

## 1. 边界

- `plan` 只读 `messages.json`，在本地完成分段、holdout 隔离、token 与费用上界；不初始化 API 客户端。
- `run` 通过 `adapters.privacy.get_allowed_client` 取唯一授权客户端；调用方不能自带凭据、URL、模型覆盖或任意 SDK 参数。
- 模型只在已有精确授权时被请求；授权绑定 provider 指纹、来源 SHA、计划哈希、调用次数与费用/路由模式。授权细节见根 README 第 6 节和 [PRIVACY.md](../docs/PRIVACY.md)。
- holdout 会话不进入 prompt、claims、merge 或质量结论；当前也没有自动 holdout 评估，不能把计划隔离称为质量已证明。
- 只有 `--full` 且全部训练段完成，状态才可能是 `complete`；`trial`、`local_baseline`、`incomplete` 均不是全量完成。

## 2. 输入与身份

输入是 `prepare` 生成的公开消息 schema，通常为：

```text
$DATA_ROOT/prepared/persona/suhui/messages.json
```

消息至少包含唯一、递增的 `source_line`、ISO 时间 `ts`、角色 `sender`（A=本人，B=目标人物）和非空 `text`。角色映射应在 `prepare` 阶段固定，规划阶段不得交换 A/B。

身份不是写死的。计划会公开保存：

- `target_name`：目标人物显示名，默认 `目标人物`；
- `slug`：产物用的小写标识，默认 `persona`，正则 `[a-z0-9][a-z0-9_-]{0,63}`；
- `user_alias`：本人 A 的显示名，默认 `我`。

命令行优先于环境变量。`plan` 可用 `--target-name/--slug/--user-alias`，也可用 `CHAT_TARGET_NAME`、`CHAT_DISTILL_SLUG`、`CHAT_USER_ALIAS`。这些展示元数据会进入当前版本的 `plan.json` 与 `plan_hash`，因此不要混用不同版本生成的旧计划。

产物写出时还会做身份校验：显示名必须是非空单行、最多 80 字符；slug 必须为小写。输出的 `merged.json`、`meta.json`、`events.json`、`timeline.json`、`persona_core.json`、三份 markdown 与 `SKILL.md` 均使用同一身份，不再存在任何固定人物名。

## 3. 命令

以下命令从项目根执行。虚拟环境按安装方式为 `.venv/bin/python`；只做测试的开发者可用自己的解释器。

### 3.1 规划

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

输出 `--out/plan.json`。规划的本地行为包括：

- 相邻消息间隔超过 `--gap-minutes`（默认 30）切分会话；
- `--pack-window day|week|all` 控制训练会话打包，默认 `week`；每行保留 `session_id`，不同会话不会被模型误认为连续对话；
- 会话超过消息数或上下文硬上界才会拆分；单条消息超限直接失败，不截断原文；
- 按每月稳定 hash 抽取约 `--holdout-rate`（默认 `0.02`）的**完整会话**作为 holdout；
- 保守估算每段两次抽取、层次 merge、以及每节点第二次尝试的 token/费用上界；
- 拒绝覆盖已有 `segments/` 或 `merges/` 检查点的工作目录，请为新计划使用新的 `--out`。

不指定单价时，计划仍可用于试跑和调用次数授权，但状态会提示价格待确认，固定费用的全量授权会被拒绝。UTF-8 字节上界不等于供应商 token 账单，预算审核仍需结合实际服务价格。

### 3.2 试跑

```bash
.venv/bin/python -m distillation.pipeline run \
  --plan "$DATA_ROOT/artifacts/persona/run/plan.json" \
  --limit 2
```

`--limit N` 只执行计划前 N 个训练段，产物状态为 `trial`。运行前必须有 `distill_trial` 授权；无授权时不会初始化客户端。

### 3.3 全量

```bash
.venv/bin/python -m distillation.pipeline run \
  --plan "$DATA_ROOT/artifacts/persona/run/plan.json" \
  --full
```

`--full` 执行全部训练段。只有全部计划段都有完成记录时状态为 `complete`。失败、缺段、校验不通过或超预算时输出 `incomplete`，不能手动改 meta 冒充完成。

`run` 必须且只能选择 `--limit N` 或 `--full`。

### 3.4 本地骨架

```bash
.venv/bin/python -m distillation.pipeline baseline \
  --plan "$DATA_ROOT/artifacts/persona/run/plan.json"
```

只生成本地空骨架，`api_calls=0`，状态 `local_baseline`，不能作为人事画像或全量结果使用。

## 4. 证据与校验

每个训练段对 `persona` 和 `memories` 各发一次抽取请求。模型原始 claim 必须**恰好**包含七个字段：

```text
id, category, statement, evidence, time_start, time_end, confidence
```

- `category`：`fact / emotion_expression / inference / unknown / style / relationship`；
- `evidence` 项只能含 `source_line, sender, quote`；
- `quote` 必须逐字命中指定 `source_line`，且 `sender` 与源消息一致；
- 模型给出的占位 `id` 会被忽略；通过校验后由本地内容生成稳定 `claim_...` ID；
- 跨段证据、额外字段、伪造时间、无法定位的引文、角色不符、非法类型均使整个响应失败，不做局部“修复后成功”；
- `inference/unknown` 不得升级为事实；无时间的 claim 不进入 `persona_core`；
- `persona_core` 只含目标人物 B 的有时间 `fact/style` 候选，并标 `candidate_only: true`，不是永久人格真相。

内部缓存复核允许在七字段之外附加三个**本地派生时间元数据**：`derived_time_start`、`derived_time_end`、`derived_time_source`。这三个字段只能由本地迁移/复核生成，模型原始输出仍严格七键；额外字段会被拒绝。

## 5. checkpoint、重试与 merge

- 每个节点在请求前先写 `started`，因此中断也计一次尝试；
- 每个节点永久最多两次尝试；失败后不自动启动第三次；
- checkpoint 绑定 `plan_hash`、输入 hash、节点身份与校验和；缺失或篡改不会被跳过；
- 每次成功 claim 立即持久化，后续失败不抹掉已验证结果；
- 层次 merge 只生成摘要视图和已有 `claim_ids` 的引用，完整 claim 由本地 deterministic union 保留；
- merge 超限时，孩子原样带到上层；没有可合并 pair 时使用确定性 fallback，不截断不重试，最终记录 `claim_union_verified` 与未合并节点；
- 任何失败都不执行成功 merge，不生成假的 `complete`。

## 6. 产物

`run` 和 `baseline` 在 plan 同目录写出：

- `persona.md`、`memories.md`、`user_profile.md`
- `SKILL.md`
- `claims.json`、`timeline.json`、`events.json`、`quality.json`
- `meta.json`、`merged.json`
- `persona_core.json`、`persona_core.md`

`merged.json` 是自包含的通用合并结构，不嵌入原始 corpus 或 holdout；如将合并结果交给外部构建器，应显式提供目标名与 slug，不能复用未授权语料。

所有 JSON 和 markdown 产物按私密文件写出（目录 0700、文件 0600），原子替换。`meta.json` 至少记录状态、运行模式、计划哈希、覆盖情况、身份、模型、调用次数和 `distillation_complete`。

状态含义：

- **complete**：计划内全部训练段完成；holdout 未参与、未评价；
- **trial**：仅选定的前缀训练段完成，不是全量；
- **local_baseline**：仅本地骨架，无模型蒸馏；
- **incomplete**：有失败或缺段，没有成功合并。

把产物接入本地运行目录：

```bash
.venv/bin/python scripts/integrate_persona.py \
  --source "$DATA_ROOT/artifacts/persona/run"
```

整合器重新核验 11 个必要 artifact、claim 的 canonical 来源和完成状态；只有 `complete` 才写入完成 manifest。非 complete 不会覆盖已有完成人物的 manifest。

## 7. 测试

从项目根运行：

```bash
../.venv-native/bin/python -m pytest -q tests/test_distillation.py
# 或安装后：
.venv/bin/python -m pytest -q tests/test_distillation.py
```

测试全部使用临时目录、合成消息、fake embedding 和 fake provider；不读取真实聊天，不读取凭据，不访问网络。测试覆盖规划/hash/费用、会话打包、整会话 holdout、精确证据、推断隔离、checkpoint 结构与篡改、永久重试上限、中断失败、claim 持续保存、异常文本不回显、超预算无损 union、显式 baseline、身份与缓存字段边界。

测试通过不等于真实 provider API 已验证、费用已核实或真实全量蒸馏已完成。全量完成仍只以最新计划的全部分段完成记录和 complete manifest 为准。