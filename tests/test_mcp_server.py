"""MCP 合成回归：tempdir、本地真实 SessionMemory、fake engine/verifier。

不读取任何真实聊天、provider 配置或凭据；禁止网络，所有模型响应均为 fixture。
运行：./.venv-native/bin/python -m pytest -q tests/test_mcp_server.py
"""
import io
import json
import socket
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from adapters.mcp_server import (MAX_REQUEST_BYTES, MCPServer,
                                 PROTOCOL_VERSION, SLUG, OWNER, _public_sources)
from adapters.session_memory import SessionMemory


class FakeEmbedder:
    is_local = True

    def split_document(self, text):
        return [text]

    def embed(self, texts):
        return [[float(len(text) % 11), 1.0, 0.0] for text in texts]

    def embed_one(self, text):
        return self.embed([text])[0]


class FakeEngine:
    supported_era = True
    selected_era = {"id": "synthetic-era"}
    artifact_status = {"claims": "complete", "timeline": "complete"}

    def __init__(self, owner, **kwargs):
        self.owner = owner
        self.kwargs = kwargs
        self.owner.engines.append(self)
        self.prepare_calls = 0
        self.chat_calls = 0
        self.session_memories = None

    def _prepare_turn(self, message, history, conversation_id=None):
        self.prepare_calls += 1
        if self.owner.fail_prepare:
            raise AssertionError("fake engine 不得意外调用 _prepare_turn")
        self.owner.calls.append(("prepare", message, conversation_id))
        if self.owner.llm_may_not_run:
            raise AssertionError("LLM 不得进入 _call_api")
        return [], []

    def chat_with_sources(self, message, history, conversation_id=None):
        self.chat_calls += 1
        self.owner.calls.append(("chat", message, conversation_id, history))
        sources = [self.owner.verified_source()] if self.owner.return_evidence else []
        return "合成 AI 回答，不代表真人", [], SimpleNamespace(prompt_tokens=7, completion_tokens=3), sources


class FakeVerifier:
    def enrich_sources(self, sources):
        # 模拟 CanonicalVerifier：只有回查成功的真实记录才会带有这些白名单标记。
        return [{**item, "canonical_verified": True, "origin": "real_chat"} for item in sources]


class FakeState:
    def __init__(self, owner, tmp_path):
        self.owner = owner
        self.sessions = SessionMemory(tmp_path, owner=OWNER, slug=SLUG, embedder=FakeEmbedder())
        self.verifier = FakeVerifier()
        self.chat_slots = threading.BoundedSemaphore(2)

    def engine(self, era=None):
        self.owner.calls.append(("engine", era))
        return FakeEngine(self.owner, era=era)

    def conversation_lock(self, conversation_id):
        return self.owner.conversation_locks.setdefault(conversation_id, threading.Lock())

    def status(self):
        return {
            "persona_status": "candidate",
            "persona_distilled": False,
            "history_ready": True,
            "history_vectors": 2,
            "session_memory": self.sessions.stats(),
            "eras": [{"id": "synthetic-era", "start": "2021-01-01", "end": "2021-01-31", "label": "合成期别"}],
            "model": {"configured": False, "service_name": None, "model": None,
                      "chat_authorized": False, "external_data_sent_by_adapter": False},
        }


class Harness:
    def __init__(self, tmp_path):
        self.calls = []
        self.engines = []
        self.conversation_locks = {}
        self.fail_prepare = False
        self.llm_may_not_run = False
        self.return_evidence = True
        self.states = []
        self.tmp_path = tmp_path
        # 与 RuntimeState 的 _states 缓存一致，避免重复初始化同一 Chroma 路径。
        self.state = FakeState(self, tmp_path)

    def state_factory(self, owner, slug):
        assert (owner, slug) == (OWNER, SLUG)
        self.states.append(self.state)
        return self.state

    @staticmethod
    def verified_source():
        return {
            "source_id": "synthetic-source", "message_id": "synthetic-message",
            "source_line": 17, "sender": "B", "timestamp": "2021-01-02 03:04:05",
            "quote": "一段完全合成的 canonical 原话", "raw_quote": "一段完全合成的 canonical 原话",
            "source_file": "PRIVATE_PATH_MUST_NOT_LEAVE", "raw_sender": "PRIVATE_SENDER",
        }


@pytest.fixture
def harness(tmp_path):
    return Harness(tmp_path)


@pytest.fixture
def server(harness):
    return MCPServer(state_factory=harness.state_factory)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("测试禁止网络")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)


def rpc(server, method, params=None, request_id=1):
    request = {"jsonrpc": "2.0", "id": request_id, "method": method}
    if params is not None:
        request["params"] = params
    return server.handle_request(request)


def tool(server, name, arguments=None):
    return rpc(server, "tools/call", {"name": name, "arguments": arguments or {}})


def structured(response):
    return response["result"]["structuredContent"]


def test_mcp_initialize_list_and_notification(server):
    initialized = rpc(server, "initialize", {"protocolVersion": PROTOCOL_VERSION,
                                             "capabilities": {}, "clientInfo": {"name": "fixture", "version": "1"}})
    assert initialized["result"]["protocolVersion"] == "2025-06-18"
    assert initialized["result"]["capabilities"]["tools"] == {"listChanged": False}
    assert initialized["result"]["serverInfo"]["name"]
    assert "【AI模拟回复】" not in initialized["result"]["instructions"]
    listing = rpc(server, "tools/list")
    names = {item["name"] for item in listing["result"]["tools"]}
    assert {"memory_search", "persona_status", "simulate_persona", "list_periods"} <= names
    search = next(item for item in listing["result"]["tools"] if item["name"] == "memory_search")
    assert search["inputSchema"]["properties"]["limit"]["maximum"] == 20
    assert server.handle_request({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None
    assert rpc(server, "does/not/exist")["error"]["code"] == -32601


def test_search_uses_local_sources_without_llm_or_private_paths(server, harness, monkeypatch):
    harness.llm_may_not_run = True
    # _prepare_turn 是检索步骤，不调用 chat_with_sources；直接返回合成 RAG 证据。
    def prepare(engine, message, history, conversation_id=None):
        harness.calls.append(("prepare", message, conversation_id))
        return [{"role": "system", "content": "PRIVATE_PROMPT"}], [harness.verified_source()]

    monkeypatch.setattr(FakeEngine, "_prepare_turn", prepare)
    monkeypatch.setattr(FakeEngine, "_call_api", lambda *args: (_ for _ in ()).throw(AssertionError("LLM called")), raising=False)
    response = tool(server, "memory_search", {"query": "合成检索", "period": "2021-01", "limit": 1})
    assert "error" not in response
    payload = structured(response)
    assert payload["llm_calls"] == 0 and payload["retrieval_only"] is True
    assert payload["sources"] == [{
        "origin": "real_chat", "canonical_verified": True,
        "timestamp": "2021-01-02 03:04:05", "time": "2021-01-02 03:04:05",
        "source_line": 17, "sender": "B", "quote": "一段完全合成的 canonical 原话", "truncated": False,
        "source_id": "synthetic-source", "message_id": "synthetic-message",
    }]
    assert not any(kind == "chat" for kind, *rest in harness.calls)
    encoded = json.dumps(response, ensure_ascii=False)
    assert "PRIVATE_PATH_MUST_NOT_LEAVE" not in encoded and "raw_quote" not in encoded
    assert tool(server, "memory_search", {"query": "x", "limit": 21})["result"]["isError"] is True


def test_chat_auth_blocks_before_state_retrieval_or_memory(server, harness, monkeypatch):
    monkeypatch.delenv("DSH_MCP_CHAT_AUTH", raising=False)
    response = tool(server, "simulate_persona", {"message": "秘密合成正文", "conversation_id": "blocked"})
    assert response["result"]["isError"] is True
    assert "DSH_MCP_CHAT_AUTH=true" in response["result"]["content"][0]["text"]
    assert harness.calls == [] and harness.states == []
    assert not (harness and any(state.sessions.conversations() for state in harness.states))


def test_chat_authorized_persists_separate_session_memory_and_sources(server, harness, tmp_path, monkeypatch):
    monkeypatch.setenv("DSH_MCP_CHAT_AUTH", "true")
    # 不读取真实 provider/env；fake chat 沿用 runtime_adapter 的 privacy 出口。
    allowed = []
    monkeypatch.setattr("adapters.privacy.get_allowed_client", lambda purpose="chat": allowed.append(purpose) or SimpleNamespace())
    base_chat = FakeEngine.chat_with_sources
    def gated_chat(engine, message, history, conversation_id=None):
        from adapters.privacy import get_allowed_client
        get_allowed_client(purpose="chat")
        return base_chat(engine, message, history, conversation_id)
    monkeypatch.setattr(FakeEngine, "chat_with_sources", gated_chat)
    response = tool(server, "simulate_persona", {"message": "本轮合成用户声明", "conversation_id": "mcp-test", "period": "2021-01"})
    assert "error" not in response
    result = structured(response)
    assert "【AI模拟回复】" not in result["reply"] and "ai_generated" not in result
    assert allowed == ["chat"]
    assert set(result["memory"]) == {"turn_id", "conversation_id", "indexed"}
    memory = harness.states[0].sessions
    with memory.connect() as db:
        row = db.execute("SELECT * FROM turns WHERE conversation_id=?", ("mcp-test",)).fetchone()
    assert row["origin"] == "ai_simulation" and row["verified_real_history"] == 0
    assert "【AI模拟回复】" not in row["ai_text"]
    assert "source_file" not in row["sources_json"] and "raw_quote" not in row["sources_json"]
    assert row["sources_json"].find("source_line") >= 0
    history = memory.history("mcp-test", "2021-01")
    assert history[-1]["role"] == "assistant" and "【AI模拟回复】" not in history[-1]["content"]
    assert history[0]["content"] == "本轮合成用户声明"
    assert not (tmp_path / "chroma_db").exists()


def tool_error(response):
    return json.loads(response["result"]["content"][0]["text"])


def test_chat_privacy_denial_does_not_persist_or_export_prompt(server, harness, monkeypatch):
    monkeypatch.setenv("DSH_MCP_CHAT_AUTH", "true")
    from adapters.privacy import PrivacyError
    checked = []

    def denied_chat(engine, message, history, conversation_id=None):
        from adapters.privacy import get_allowed_client
        checked.append(conversation_id)
        get_allowed_client(purpose="chat")
        raise AssertionError("fake 未阻止模型调用")

    monkeypatch.setattr("adapters.privacy.get_allowed_client",
                        lambda purpose="chat": (_ for _ in ()).throw(PrivacyError("fake 拒绝")))
    monkeypatch.setattr(FakeEngine, "chat_with_sources", denied_chat)
    response = tool(server, "simulate_persona", {"message": "合成隐私拒绝正文", "conversation_id": "privacy-denied"})
    assert response["result"]["isError"] is True
    assert tool_error(response)["error"] == "privacy_not_authorized"
    assert checked == ["privacy-denied"]
    memory = harness.state.sessions
    assert memory.history("privacy-denied", None) == []
    with memory.connect() as db:
        assert db.execute("SELECT count(*) FROM turns").fetchone()[0] == 0
    encoded = json.dumps(response, ensure_ascii=False)
    assert "合成隐私拒绝正文" not in encoded and "fake 拒绝" not in encoded
    assert not any(call[0] == "chat" for call in harness.calls)


def test_status_periods_and_privacy_are_non_secret(server, harness):
    status = structured(tool(server, "persona_status"))
    assert status["persona"]["status"] == "candidate"
    assert status["persona"]["completion_claimed"] is False
    assert status["chat_authorized"] is False and status["mcp_chat_enabled"] is False
    assert status["configuration"]["chat_authorized"] is False
    periods = structured(tool(server, "list_periods"))
    assert periods["periods"][0]["id"] == "synthetic-era" and periods["llm_calls"] == 0
    assert harness.engines == []  # 状态和时间线不构造 engine，也不触发 LLM


@pytest.mark.parametrize("manifest_status,distilled", [
    ("trial", False), ("full_in_progress", False), ("complete", True),
])
def test_persona_status_mirrors_full_trial_progress(server, harness, manifest_status, distilled):
    status = harness.state.status()
    status["persona_status"] = manifest_status
    status["persona_distilled"] = distilled
    harness.state.status = lambda: status
    result = structured(tool(server, "persona_status"))
    assert result["persona"]["status"] == manifest_status
    assert result["persona"]["distilled"] is distilled
    assert result["persona"]["completion_claimed"] is distilled


def test_stdio_strict_json_stderr_counts_and_byte_limit(server):
    class CountingStderr:
        def __init__(self):
            self.lines = []
        def write(self, text):
            self.lines.append(text)
        def flush(self):
            pass

    stdin = io.BytesIO(
        b'{"jsonrpc":"2.0","id":1,"method":"initialize"}\n'
        b'{bad json}\n'
        + b' ' * (MAX_REQUEST_BYTES + 1) + b'\n'
        + b'{"jsonrpc":"2.0","id":2,"method":"tools/list"}\n'
        + b' ' * (MAX_REQUEST_BYTES + 1)  # EOF 超限，无换行
    )
    stdout = io.BytesIO()
    stderr = CountingStderr()
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(sys, "stderr", stderr)
    try:
        server.serve(stdin, stdout)
    finally:
        monkeypatch.undo()
    lines = stdout.getvalue().splitlines()
    assert len(lines) == 5
    decoded = [json.loads(line) for line in lines]
    assert decoded[0]["result"]["protocolVersion"] == PROTOCOL_VERSION
    assert decoded[1]["error"]["code"] == -32700
    assert decoded[2]["error"]["code"] == -32600
    assert "tools" in decoded[3]["result"]
    assert decoded[4]["error"]["code"] == -32600
    assert all("event=" in line and "count=" in line for line in stderr.lines)


def test_real_subprocess_stdout_is_only_json_rpc():
    requests = b"\n".join([
        json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}).encode(),
        json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}).encode(),
        json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}).encode(),
        json.dumps({"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                    "params": {"name": "simulate_persona", "arguments": {"message": "不读取本地状态的合成消息"}}}).encode(),
    ]) + b"\n"
    completed = subprocess.run(
        [sys.executable, "-m", "adapters.mcp_server"], cwd=ROOT, input=requests,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=15, check=True,
        env={"PATH": "/usr/bin:/bin", "PYTHONIOENCODING": "utf-8"},
    )
    responses = [json.loads(line) for line in completed.stdout.splitlines()]
    assert len(responses) == 3 and [item["id"] for item in responses] == [1, 2, 3]
    assert responses[0]["result"]["protocolVersion"] == PROTOCOL_VERSION
    assert len(responses[1]["result"]["tools"]) == 4
    assert responses[2]["result"]["isError"] is True
    assert tool_error(responses[2])["error"] == "chat_not_authorized"
    assert b"initialize" not in completed.stderr and b"source_file" not in completed.stderr
    assert all(line.startswith(b"mcp event=") and b" count=" in line for line in completed.stderr.splitlines())


def test_no_history_leak_invalid_params_and_source_allowlist(server):
    assert rpc(server, "tools/call", {"name": "memory_search", "arguments": {"query": "x", "path": "/tmp/no"}})["result"]["isError"]
    rejected = tool(server, "simulate_persona", {"message": "x", "conversation_id": "../escape"})
    assert rejected["result"]["isError"] is True
    # 缺 origin/canonical 或含私有路径的条目不得透传。
    assert _public_sources([{"canonical_verified": True, "origin": "real_chat", "source_line": 1,
                             "timestamp": "2021-01-01", "sender": "A", "quote": "合成",
                             "source_file": "/private/source.txt"}]) == [{
        "origin": "real_chat", "canonical_verified": True, "timestamp": "2021-01-01", "time": "2021-01-01",
        "source_line": 1, "sender": "A", "quote": "合成", "truncated": False,
    }]
    assert _public_sources([{"source_line": 1, "quote": "unverified"}]) == []


def test_concurrent_chats_have_independent_engines_and_periods(server, harness, monkeypatch):
    monkeypatch.setenv("DSH_MCP_CHAT_AUTH", "true")
    started = threading.Barrier(2)

    def chat(engine, message, history, conversation_id=None):
        engine.session_memories = []
        started.wait(timeout=3)
        return "【AI模拟回复】合成并发回复", [], None, []

    monkeypatch.setattr(FakeEngine, "chat_with_sources", chat)
    errors = []

    def invoke(cid, period):
        try:
            response = tool(server, "simulate_persona", {"message": cid, "conversation_id": cid, "period": period})
            assert not response["result"].get("isError")
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=invoke, args=(f"concurrent-{index}", f"2021-0{index + 1}")) for index in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)
    assert not errors
    assert len(harness.engines) == 2
    assert {engine.kwargs["era"] for engine in harness.engines} == {"2021-01", "2021-02"}
    assert all(id(engine) != id(harness.engines[0]) for engine in harness.engines[1:])
