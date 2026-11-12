"""供本机 DSH 使用的 stdio MCP 适配层。

复用 service.RuntimeState、GroundedChatEngine、CanonicalVerifier 和独立的
SessionMemory。stdio 只输出 JSON-RPC；真实聊天文件、完整 prompt、凭据和
任意路径写参数均不进入本协议。只有 simulate_persona 可以经双重授权调用模型。

运行：DSH_MCP_CHAT_AUTH=true python -m adapters.mcp_server
（该环境变量只是 MCP 开关，仍需通过 adapters.privacy 的 chat 授权。）
"""
from __future__ import annotations

import json
import os
import re
import sys
import threading
from collections import Counter
from typing import Any, Callable, Optional

PROTOCOL_VERSION = "2025-06-18"
SERVER_NAME = "ex-memory-local-grounded"
SERVER_VERSION = "0.1.0"
OWNER = int(os.environ.get("CHAT_MCP_OWNER", "1"))
SLUG = os.environ.get("CHAT_MCP_SLUG", "persona")
if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", SLUG) or OWNER < 1:
    raise ValueError("CHAT_MCP_OWNER/SLUG 必须是本机既有镜像的合法标识")
MAX_REQUEST_BYTES = 262_144
MAX_INPUT_BYTES = 15_000
MAX_REPLY_BYTES = 65_536
MAX_EXCERPT_BYTES = 4_000
MAX_SOURCES = 20

INSTRUCTIONS = (
    f"本服务只访问 owner={OWNER}、slug={SLUG} 的既有本地 ex-memory 实例。"
    "memory_search 仅检索 canonical 本地真实历史并返回原话片段、原始时间和行号，绝不自动调用 LLM。"
    "simulate_persona 依据本地历史证据生成角色回复；必须同时设置 DSH_MCP_CHAT_AUTH=true "
    "并满足项目的 chat 隐私授权，成功后只写独立对话 SessionMemory。"
    "人格状态不代表已经完成蒸馏；本服务不提供源文件、raw txt、完整 prompt、凭据或任意路径写入。"
)


class MCPError(Exception):
    """可安全返回给 MCP 客户端的固定错误，不携带底层路径或正文。"""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def _default_state(owner: int, slug: str):
    # 延迟导入：tools/list 与 initialize 不加载 Chroma、配置或任何模型客户端。
    from adapters.service import _state

    return _state(slug, owner)


def _configuration_status() -> dict:
    from adapters.privacy import configuration_status

    return configuration_status()


def _text(value: Any, field: str, *, max_bytes: int = MAX_INPUT_BYTES) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise MCPError("invalid_arguments", f"{field} 必须是非空文本")
    if len(value.encode("utf-8")) > max_bytes:
        raise MCPError("invalid_arguments", f"{field} 超过 {max_bytes} UTF-8 字节上限")
    return value


def _period(value: Any) -> tuple[Optional[str], str]:
    if value in (None, "", "latest"):
        return None, "latest"
    value = _text(value, "period", max_bytes=80 * 4)
    if len(value) > 80:
        raise MCPError("invalid_arguments", "period 最长 80 个字符")
    return value, value


def _visible_reply(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MCPError("generation_failed", "模型未返回有效文本；未保存本轮记忆")
    reply = re.sub(r"^\s*(?:【(?:AI)?(?:模拟回复|生成|模拟|分身)】\s*)+", "", value).strip()
    if not reply:
        raise MCPError("generation_failed", "模型未返回可展示正文；未保存本轮记忆")
    if len(reply.encode("utf-8")) > MAX_REPLY_BYTES:
        raise MCPError("generation_failed", "模拟回复超过安全上限；未保存本轮记忆")
    return reply


def _excerpt(value: Any) -> tuple[str, bool]:
    if not isinstance(value, str) or not value:
        raise MCPError("evidence_rejected", "canonical 证据缺少原话片段")
    encoded = value.encode("utf-8")
    truncated = len(encoded) > MAX_EXCERPT_BYTES
    return encoded[:MAX_EXCERPT_BYTES].decode("utf-8", "ignore"), truncated


def _public_sources(sources: Any, limit: int = MAX_SOURCES) -> list[dict]:
    """只输出 verifier 回查过的证据定位与有界片段，不透传原始 metadata。"""
    public = []
    for item in sources if isinstance(sources, list) else []:
        if len(public) >= limit:
            break
        if not isinstance(item, dict) or item.get("canonical_verified") is not True:
            continue
        if item.get("origin") != "real_chat" or item.get("sender") not in {"A", "B"}:
            continue
        line, stamp = item.get("source_line"), item.get("timestamp") or item.get("time")
        if not isinstance(line, int) or isinstance(line, bool) or line < 1 or not isinstance(stamp, str) or not stamp:
            continue
        try:
            quote, truncated = _excerpt(item.get("raw_quote", item.get("quote")))
        except MCPError:
            continue
        source = {
            "origin": "real_chat",
            "canonical_verified": True,
            "timestamp": stamp,
            "time": stamp,
            "source_line": line,
            "sender": item["sender"],
            "quote": quote,
            "truncated": truncated,
        }
        # ID 是 canonical 定位符；source_file、raw_sender、raw_quote 等绝不透传。
        for key in ("source_id", "message_id", "claim_id"):
            value = item.get(key)
            if isinstance(value, str) and value and len(value.encode("utf-8")) <= 512:
                source[key] = value
        public.append(source)
    return public


def _safe_artifact_status(value: Any) -> Any:
    if isinstance(value, str):
        return value[:80]
    if isinstance(value, dict):
        return {str(key)[:40]: str(item)[:80] for key, item in value.items()
                if isinstance(item, (str, int, float, bool)) or item is None}
    return "unknown"


def _usage(value: Any) -> dict:
    result = {}
    for key in ("prompt_tokens", "completion_tokens"):
        item = getattr(value, key, None)
        result[key] = item if isinstance(item, int) and not isinstance(item, bool) and item >= 0 else None
    return result


def _tool_definitions() -> list[dict]:
    period = {"type": "string", "maxLength": 80,
              "description": "latest、YYYY-MM、YYYY-MM-DD 或已有证据的 period id；默认 latest"}
    return [
        {
            "name": "memory_search",
            "description": "只读检索本地真实聊天 canonical 证据；返回原始时间/行号和有界原话片段，不调用 LLM。",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "minLength": 1, "maxLength": 5000},
                    "period": period,
                    "limit": {"type": "integer", "minimum": 1, "maximum": MAX_SOURCES, "default": 10},
                },
                "required": ["query"], "additionalProperties": False,
            },
        },
        {
            "name": "persona_status",
            "description": "返回非秘密的配置、persona 完成度、隐私授权和记忆计数状态；不返回正文、路径或凭据。",
            "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
        },
        {
            "name": "simulate_persona",
            "description": "调用 GroundedChatEngine.chat_with_sources 生成角色答复；双授权后写独立对话记忆。",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "message": {"type": "string", "minLength": 1, "maxLength": 5000},
                    "conversation_id": {"type": "string", "pattern": "^[A-Za-z0-9_-]{1,80}$"},
                    "period": period,
                },
                "required": ["message"], "additionalProperties": False,
            },
        },
        {
            "name": "list_periods",
            "description": "列出本地时间线可用的 period id 和日期范围，不调用 LLM。",
            "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    ]


class MCPServer:
    """纯 stdlib 的 MCP 2025-06-18 换行 JSON-RPC 服务。"""

    def __init__(self, state_factory: Optional[Callable[[int, str], Any]] = None):
        self._state_factory = state_factory or _default_state
        self._counts = Counter()
        self._count_lock = threading.Lock()

    def _state(self):
        # owner/slug 固定在服务端，客户端不能选择其他镜像或传文件路径。
        return self._state_factory(OWNER, SLUG)

    @staticmethod
    def _arguments(arguments: Any, allowed: set[str], required: set[str] = frozenset()) -> dict:
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, dict):
            raise MCPError("invalid_arguments", "arguments 必须是 object")
        if not required.issubset(arguments) or set(arguments) - allowed:
            raise MCPError("invalid_arguments", "arguments 缺少必填项或含未允许字段")
        return arguments

    @staticmethod
    def _limit(value: Any) -> int:
        if value is None:
            return 10
        if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= MAX_SOURCES:
            raise MCPError("invalid_arguments", f"limit 必须是 1 到 {MAX_SOURCES} 的整数")
        return value

    @staticmethod
    def _periods(status: dict) -> list[dict]:
        periods = []
        for era in status.get("eras", []) if isinstance(status, dict) else []:
            if not isinstance(era, dict) or not isinstance(era.get("id"), str):
                continue
            item = {"id": era["id"][:80]}
            for key in ("start", "end", "label"):
                value = era.get(key)
                if isinstance(value, str):
                    item[key] = value[:200]
            periods.append(item)
        return periods[:100]

    def memory_search(self, arguments: dict) -> dict:
        args = self._arguments(arguments, {"query", "period", "limit"}, {"query"})
        query = _text(args.get("query"), "query")
        era, requested = _period(args.get("period"))
        limit = self._limit(args.get("limit"))
        state = self._state()
        # RuntimeState.engine 每次创建 engine + store；绝不把 cutoff 写到共享 store。
        engine = state.engine(era)
        try:
            _messages, sources = engine._prepare_turn(query, [])
            verified = state.verifier.enrich_sources(sources)
        except (ValueError, TypeError) as exc:
            raise MCPError("invalid_period_or_evidence", "period 不合法或本地证据格式不一致；未调用模型") from None
        except FileNotFoundError:
            raise MCPError("persona_unavailable", "本地人物候选包尚未初始化；历史库仍保留") from None
        selected = getattr(engine, "selected_era", None)
        return {
            "sources": _public_sources(verified, limit),
            "requested_period": requested,
            "selected_period": selected.get("id") if isinstance(selected, dict) else None,
            "supported_period": bool(getattr(engine, "supported_era", False)),
            "persona_status": _safe_artifact_status(getattr(engine, "artifact_status", "unknown")),
            "retrieval_only": True,
            "llm_calls": 0,
            "limit": limit,
            "no_evidence": not bool(verified),
        }

    def persona_status(self, arguments: dict) -> dict:
        self._arguments(arguments, set())
        status = self._state().status()
        model = status.get("model", {}) if isinstance(status, dict) else {}
        # configuration_status 是项目已有的非秘密视图；若 fake/state 已提供，则仍逐项白名单。
        if not isinstance(model, dict) or not model:
            model = _configuration_status()
        safe_model = {key: model.get(key) for key in (
            "configured", "service_name", "model", "chat_authorized", "external_data_sent_by_adapter")}
        mcp_enabled = os.environ.get("DSH_MCP_CHAT_AUTH") == "true"
        memory = status.get("session_memory", {}) if isinstance(status.get("session_memory"), dict) else {}
        return {
            "owner": OWNER,
            "slug": SLUG,
            "persona": {
                "status": str(status.get("persona_status", "pending"))[:80],
                "distilled": status.get("persona_distilled") is True,
                "completion_claimed": status.get("persona_distilled") is True,
            },
            "history": {
                "ready": status.get("history_ready") is True,
                "vectors": status.get("history_vectors", 0) if isinstance(status.get("history_vectors"), int) else 0,
            },
            "session_memory": {key: memory.get(key) for key in (
                "origin", "turns", "vectors", "pending_index_turns", "verified_real_history")},
            "configuration": safe_model,
            "mcp_chat_enabled": mcp_enabled,
            "chat_authorized": mcp_enabled and safe_model.get("chat_authorized") is True,
            "periods": self._periods(status),
        }

    def list_periods(self, arguments: dict) -> dict:
        self._arguments(arguments, set())
        periods = self._periods(self._state().status())
        return {"periods": periods, "default": "latest", "llm_calls": 0}

    def simulate_persona(self, arguments: dict) -> dict:
        # 必须在解析正文、构造状态/会话或检索之前检查，未授权时连空会话都不创建。
        if os.environ.get("DSH_MCP_CHAT_AUTH") != "true":
            raise MCPError("chat_not_authorized",
                           "需要显式设置 DSH_MCP_CHAT_AUTH=true；未调用模型、未检索正文、未写入会话记忆")
        args = self._arguments(arguments, {"message", "conversation_id", "period"}, {"message"})
        message = _text(args.get("message"), "message")
        era, requested = _period(args.get("period"))
        conversation_id = args.get("conversation_id")
        if conversation_id is not None and (
            not isinstance(conversation_id, str) or re.fullmatch(r"[A-Za-z0-9_-]{1,80}", conversation_id) is None):
            raise MCPError("invalid_arguments", "conversation_id 只能含 1-80 个字母、数字、下划线或连字符")
        state = self._state()
        try:
            # 唯一的模型调用入口是该方法内部的 privacy.get_allowed_client(purpose='chat')。
            engine = state.engine(era)
            conversation_id = state.sessions.ensure_conversation(conversation_id, era)
        except (ValueError, TypeError):
            raise MCPError("invalid_period_or_conversation", "period/conversation_id 无效或与既有会话不一致") from None
        except FileNotFoundError:
            raise MCPError("persona_unavailable", "本地人物候选包尚未初始化；未调用模型") from None
        if not state.chat_slots.acquire(blocking=False):
            raise MCPError("busy", "已有两个模拟请求运行，请稍后再试")
        try:
            with state.conversation_lock(conversation_id):
                history = state.sessions.history(conversation_id, era)
                # 每轮独立 engine 实例持有 session_memories，避免跨请求污染与 cutoff race。
                engine.session_memories = state.sessions.search(message, era=era)
                try:
                    from adapters.privacy import PrivacyError
                    reply, _stickers, usage, sources = engine.chat_with_sources(message, history, conversation_id)
                except PrivacyError:
                    raise MCPError("privacy_not_authorized",
                                   "项目 chat 隐私授权/模型配置未通过；未自动重试或切换供应商") from None
                except (ValueError, TypeError):
                    raise MCPError("chat_rejected", "会话状态或模型预算校验未通过；未保存本轮记忆") from None
                reply = _visible_reply(reply)
                verified = state.verifier.enrich_sources(sources)
                public_sources = _public_sources(verified, MAX_SOURCES)
                # 只把脱敏后的 canonical 引用存入独立对话库，不写真实历史 collection。
                saved = state.sessions.append_turn(conversation_id, message, reply, public_sources, era, ai_generated=True)
            selected = getattr(engine, "selected_era", None)
            return {
                "reply": reply,
                "conversation_id": conversation_id,
                "period": requested,
                "selected_period": selected.get("id") if isinstance(selected, dict) else None,
                "supported_period": bool(getattr(engine, "supported_era", False)),
                "sources": public_sources,
                "memory": {key: saved[key] for key in ("turn_id", "conversation_id", "indexed")},
                "usage": _usage(usage),
                "persona_status": _safe_artifact_status(getattr(engine, "artifact_status", "unknown")),
            }
        finally:
            state.chat_slots.release()

    def call_tool(self, name: str, arguments: dict) -> dict:
        handlers = {
            "memory_search": self.memory_search,
            "persona_status": self.persona_status,
            "simulate_persona": self.simulate_persona,
            "list_periods": self.list_periods,
        }
        if name not in handlers:
            raise MCPError("unknown_tool", "未知工具")
        return handlers[name](arguments)

    @staticmethod
    def _tool_result(payload: dict, *, error: bool = False) -> dict:
        result = {"content": [{"type": "text", "text": json.dumps(payload, ensure_ascii=False, allow_nan=False)}]}
        if not error:
            result["structuredContent"] = payload
        else:
            result["isError"] = True
        return result

    def handle_request(self, request: Any) -> Optional[dict]:
        if not isinstance(request, dict) or request.get("jsonrpc") != "2.0" or not isinstance(request.get("method"), str):
            return {"jsonrpc": "2.0", "id": None, "error": {"code": -32600, "message": "Invalid Request"}}
        request_id = request.get("id")
        method, params = request["method"], request.get("params", {})
        notification = "id" not in request
        if notification:
            # initialized/cancelled 等通知一律无响应；通知中的工具调用也不执行。
            return None
        try:
            if params is None:
                params = {}
            if not isinstance(params, dict):
                raise ValueError("params")
            if method == "initialize":
                result = {
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
                    "instructions": INSTRUCTIONS,
                }
            elif method == "ping":
                result = {}
            elif method == "tools/list":
                result = {"tools": _tool_definitions()}
            elif method == "tools/call":
                name = params.get("name")
                arguments = params.get("arguments", {})
                if not isinstance(name, str) or set(params) - {"name", "arguments"}:
                    return {"jsonrpc": "2.0", "id": request_id,
                            "error": {"code": -32602, "message": "Invalid tools/call params"}}
                # 工具名不受信任；只使用预定义事件，避免日志注入或回显正文/路径。
                event_name = name if name in {"memory_search", "persona_status", "simulate_persona", "list_periods"} else "unknown"
                self._log(f"tool.{event_name}.started")
                try:
                    result = self._tool_result(self.call_tool(name, arguments))
                    self._log(f"tool.{event_name}.ok")
                except MCPError as exc:
                    result = self._tool_result({"error": exc.code, "message": exc.message}, error=True)
                    self._log(f"tool.{event_name}.rejected")
                except Exception:
                    # 底层异常可能含正文/路径，不进入响应或日志。
                    result = self._tool_result({"error": "internal_error", "message": "本地操作失败；请检查服务配置"}, error=True)
                    self._log(f"tool.{event_name}.failed")
            else:
                return {"jsonrpc": "2.0", "id": request_id, "error": {"code": -32601, "message": "Method not found"}}
            return {"jsonrpc": "2.0", "id": request_id, "result": result}
        except (ValueError, TypeError):
            return {"jsonrpc": "2.0", "id": request_id, "error": {"code": -32602, "message": "Invalid params"}}
        except Exception:
            self._log("request.failed")
            return {"jsonrpc": "2.0", "id": request_id, "error": {"code": -32603, "message": "Internal error"}}

    def _log(self, event: str) -> None:
        # event 全部来自代码内固定字符串，绝不拼接 query/message/异常/path。
        with self._count_lock:
            self._counts[event] += 1
            count = self._counts[event]
        sys.stderr.write(f"mcp event={event} count={count}\n")
        sys.stderr.flush()

    @staticmethod
    def _write_json(stream, message: dict) -> None:
        data = json.dumps(message, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
        stream.write(data + b"\n")
        stream.flush()

    def serve(self, stdin=None, stdout=None) -> None:
        stdin = stdin if stdin is not None else sys.stdin.buffer
        stdout = stdout if stdout is not None else sys.stdout.buffer
        while True:
            line = stdin.readline(MAX_REQUEST_BYTES + 3)
            if not line:
                break
            # LF 不计入请求体上限；EOF 前没有 LF 时也不能放过 MAX+1 字节。
            oversized = len(line) - int(line.endswith(b"\n")) > MAX_REQUEST_BYTES
            if oversized and not line.endswith(b"\n"):
                # 丢弃本条剩余内容，避免下一条 JSON 被污染，同时不分配整段消息。
                while line and not line.endswith(b"\n"):
                    line = stdin.readline(MAX_REQUEST_BYTES + 3)
            if oversized:
                self._log("request.oversized")
                self._write_json(stdout, {"jsonrpc": "2.0", "id": None,
                                          "error": {"code": -32600, "message": "Request exceeds byte limit"}})
                continue
            if not line.strip():
                continue
            try:
                request = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                self._log("request.invalid_json")
                self._write_json(stdout, {"jsonrpc": "2.0", "id": None,
                                          "error": {"code": -32700, "message": "Parse error"}})
                continue
            response = self.handle_request(request)
            if response is not None:
                self._write_json(stdout, response)
        self._log("server.stopped")


def main() -> int:
    try:
        MCPServer().serve()
    except BrokenPipeError:
        return 0
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
