"""纯合成运行测试：执行真实父 ChatEngine，隔离配置、凭据、网络与镜像 I/O。

不加载项目的聊天/准备数据或账户。模型 create 全部是本文件的 fixture；
权限测试使用真实 privacy 代码，但配置和授权也只来自 pytest tmp_path。
"""

import copy
import importlib.util
import json
import socket
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "ex-memory"
CFG = {"model": "explicit-fixture", "temperature": 0, "top_p": 1, "frequency_penalty": 0, "max_tokens": 100}


def _claim(claim_id, statement, stamp="2021-06-03", category="fact", sender="A", line=10, canonical=True, **extra):
    claim = {
        "id": claim_id, "category": category, "statement": statement,
        "evidence": [{"source_line": line, "sender": sender, "quote": statement}],
        "time_start": stamp, "time_end": stamp, "confidence": 0.9, **extra,
    }
    return _canonical(claim) if canonical else claim


def _era(claim_ids, era_id="june", start="2021-06-01", end="2021-06-30"):
    return {"id": era_id, "start": start, "end": end, "label": "FUTURE_LABEL_NOT_FOR_PROMPT", "claim_ids": claim_ids}


def _rag(text, stamp="2021-06-03 12:00:00", sender="A", line=40, score=0.95, record_id="rag", **metadata):
    return {
        "id": record_id, "display_text": text, "score": score,
        "metadata": {
            "source": "chat", "origin": "real_chat", "sender": sender,
            "dominant_speaker": "target" if sender == "B" else "other",
            "start_ts": stamp, "end_ts": stamp, "source_line": line, **metadata,
        },
    }


@pytest.fixture
def isolated_runtime(monkeypatch, tmp_path):
    """装载实际 engine.py 的类和方法，不装载实际 config/default clients。"""
    def forbidden(*args, **kwargs):
        raise AssertionError("禁止配置默认值、凭据、网络或上游副作用")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    config = ModuleType("config")
    for key, value in {
        "RECENT_SESSIONS": 3, "DEFAULT_TOP_K": 10, "RAG_THRESHOLD": 0.6,
        "LLM_MAX_CONTEXT_CHARS": 50000, "LLM_TOTAL_TOKEN_BUDGET": 12000,
        "get_llm_config": forbidden,
    }.items():
        setattr(config, key, value)
    monkeypatch.setitem(sys.modules, "config", config)
    core = ModuleType("core")
    core.__path__ = [str(UPSTREAM / "core")]
    monkeypatch.setitem(sys.modules, "core", core)
    adapters = ModuleType("adapters")
    adapters.__path__ = [str(ROOT / "adapters")]
    monkeypatch.setitem(sys.modules, "adapters", adapters)
    bootstrap = ModuleType("adapters.upstream")
    bootstrap.ROOT = tmp_path
    bootstrap.prepare_upstream = lambda: UPSTREAM
    monkeypatch.setitem(sys.modules, "adapters.upstream", bootstrap)
    retry = ModuleType("core.retry")
    retry.retry_api = forbidden
    monkeypatch.setitem(sys.modules, "core.retry", retry)
    stickers = ModuleType("core.sticker_selector")
    stickers.select_stickers = forbidden
    stickers.IMAGE_STICKERS = {}
    monkeypatch.setitem(sys.modules, "core.sticker_selector", stickers)
    mirror_calls = []

    class Mirror:
        base_dir = tmp_path
        exists = path = read_text = read_json = forbidden

    def mirror_store(slug, owner=None):
        mirror_calls.append((slug, owner))
        return Mirror()

    mirror = ModuleType("core.mirror_store")
    mirror.mirror_store = mirror_store
    monkeypatch.setitem(sys.modules, "core.mirror_store", mirror)
    for name in ("llm_router", "wallet_manager", "psych_state", "api_clients"):
        module = ModuleType(f"core.{name}")
        module.get_router = module.create_redpacket = module.detect_redpacket_trigger = forbidden
        monkeypatch.setitem(sys.modules, f"core.{name}", module)

    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        return module

    for name in ("path_safety", "validation", "privacy"):
        load(f"core.{name}", UPSTREAM / "core" / f"{name}.py")
    parent = load("core.engine", UPSTREAM / "core" / "engine.py")
    runtime = load("adapters.runtime_adapter", ROOT / "adapters" / "runtime_adapter.py")
    calls, permissions = [], []
    usage = SimpleNamespace(prompt_tokens=17, completion_tokens=9)
    api = SimpleNamespace(text="fixture response", failure=None)

    def create(**kwargs):
        calls.append(kwargs)
        if api.failure is not None:
            raise api.failure
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=api.text))], usage=usage)

    client = SimpleNamespace(
        model="allowed-fixture", provider={"service_name": "fixture-service"},
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
    )
    privacy = ModuleType("adapters.privacy")

    class PrivacyError(RuntimeError):
        pass

    def allowed(purpose="chat"):
        permissions.append(purpose)
        return client

    privacy.get_allowed_client, privacy.PrivacyError = allowed, PrivacyError
    monkeypatch.setitem(sys.modules, "adapters.privacy", privacy)

    def persona(claims=None, eras=None, core_ids=(), claims_status="complete", timeline_status="complete", list_claims=False, list_timeline=False):
        claims = claims if claims is not None else [_claim("coffee", "coffee preference from A")]
        eras = eras if eras is not None else [_era([claim["id"] for claim in claims])]
        directory = tmp_path / "persona"
        directory.mkdir(exist_ok=True)
        claims_doc = claims if list_claims else {"schema_version": 1, "status": claims_status, "claims": claims}
        eras_doc = eras if list_timeline else {"schema_version": 1, "status": timeline_status, "eras": eras}
        (directory / "claims.json").write_text(json.dumps(claims_doc, ensure_ascii=False), encoding="utf-8")
        (directory / "timeline.json").write_text(json.dumps(eras_doc, ensure_ascii=False), encoding="utf-8")
        (directory / "persona_core.json").write_text(json.dumps({"schema_version": 1, "status": claims_status, "slug": "fixture", "candidate_only": True, "claim_ids": list(core_ids)}), encoding="utf-8")
        (directory / "persona.md").write_text("FUTURE_FULL_PERSONA_NEVER_LOAD", encoding="utf-8")
        return directory

    def engine(directory=None, era="2021-06-10", **kwargs):
        return runtime.GroundedChatEngine("fixture", persona_dir=directory or persona(), era=era, model_config=dict(CFG), **kwargs)

    return SimpleNamespace(
        runtime=runtime, parent=parent, privacy=privacy, calls=calls, permissions=permissions,
        api=api, usage=usage, persona=persona, engine=engine, mirror_calls=mirror_calls,
        load=load, forbidden=forbidden,
    )


def _canonical(claim, stamp=None):
    """模拟 service 的 SQLite 回查，不代表本测试读取任何原始聊天。"""
    claim = copy.deepcopy(claim)
    claim["canonical_verified"] = True
    start = claim.get("time_start")
    actual_time = stamp or (start + "T12:00:00" if isinstance(start, str) and len(start) == 10 else start)
    for item in claim["evidence"]:
        item.update(
            canonical_verified=True, time=actual_time, timestamp=actual_time,
            source_id="synthetic.sqlite", message_id=f"message-{item['source_line']}",
            source_file="/synthetic/fixture.txt", raw_sender=item["sender"],
        )
    return claim


def _verified_verifier(claim):
    return _canonical(claim)


def test_actual_parent_and_explicit_model_config(isolated_runtime):
    t = isolated_runtime
    with pytest.raises(ValueError, match="显式"):
        t.runtime.GroundedChatEngine("fixture", owner=99, persona_dir=t.persona())
    with pytest.raises(ValueError, match="配置"):
        t.runtime.GroundedChatEngine("fixture", persona_dir=t.persona(), model_config={"model": "fixture"})
    engine = t.engine(owner=99)
    assert isinstance(engine, t.parent.ChatEngine)
    assert engine._llm_config == CFG
    assert t.mirror_calls == [("fixture", 99)]
    assert not t.calls and not t.permissions
    assert engine.relationship_stage == "source" and engine.source_grounded


@pytest.mark.parametrize("list_claims,list_timeline", [(True, True), (True, False), (False, True), (False, False)])
def test_artifact_schema_variants(isolated_runtime, list_claims, list_timeline):
    t = isolated_runtime
    directory = t.persona([_claim("coffee", "coffee preference from A", canonical=False)], list_claims=list_claims, list_timeline=list_timeline)
    engine = t.engine(directory)
    assert not engine.supported_era
    assert engine.verify_claims(_verified_verifier) == {"verified_claim_ids": ["coffee"], "rejected_claim_ids": []}
    messages, sources = engine._prepare_turn("coffee", [])
    assert engine.supported_era
    assert engine._candidate_claims["coffee"]["canonical_verified"] is True
    assert "coffee preference from A" in messages[0]["content"]
    assert sources[0]["sender"] == "A" and sources[0]["source_line"] == 10


@pytest.mark.parametrize("selector", ["2021-06", "2021-06-10", "june"])
def test_supported_era_selectors(isolated_runtime, selector):
    engine = isolated_runtime.engine(era=selector)
    engine.verify_claims(_verified_verifier)
    assert engine.supported_era and engine.selected_era["id"] == "june"


@pytest.mark.parametrize("selector", ["missing-id", "2021", "2021-13", "2021-02-31"])
def test_invalid_or_unsupported_era_fails_before_api(isolated_runtime, selector):
    t = isolated_runtime
    with pytest.raises(t.runtime.UnsupportedEraError, match="不支持") as error:
        t.engine(era=selector)
    assert error.value.status_code == 400
    assert not t.calls and not t.permissions


def test_era_id_without_canonical_evidence_is_400_not_paid_unknown(isolated_runtime):
    t = isolated_runtime
    directory = t.persona([_claim("coffee", "coffee candidate", canonical=False)])
    with pytest.raises(t.runtime.UnsupportedEraError):
        t.engine(directory, era="june")
    assert not t.calls and not t.permissions


@pytest.mark.parametrize("era", [None, "latest", "2040-01", "2021-06-01"])
def test_unknown_stage_still_retrieves_history_without_fixed_unknown(isolated_runtime, era):
    t = isolated_runtime
    history = HistoryFixture([_rag("TRUE_PAST_HISTORY", stamp="2021-05-02 12:00:00")])
    engine = t.engine(era=era, vector_store=history, embedder=object())
    t.api.text = "answer grounded in history"
    reply, stickers, usage, sources = engine.chat_with_sources("coffee", [], "cid")
    assert not engine.supported_era and engine.selected_era is None
    assert reply == "answer grounded in history"
    assert stickers == [] and usage is t.usage
    assert [s["text"] for s in sources if s["source"] == "chat"] == ["TRUE_PAST_HISTORY"]
    prompt = t.calls[0]["messages"][0]["content"]
    assert "未确定关系阶段" in prompt and "仍允许" in prompt
    assert "不注入未经证实的人格阶段" in prompt and len(history.queries) == 1


@pytest.mark.parametrize("status", ["trial", "local_baseline", "incomplete"])
def test_noncomplete_artifacts_only_candidates(isolated_runtime, status):
    t = isolated_runtime
    directory = t.persona(claims_status=status)
    engine = t.engine(directory)
    messages, sources = engine._prepare_turn("coffee", [])
    assert len(engine.claims) == 1 and len(engine.timeline) == 1
    assert not engine.supported_era and not sources
    assert "候选/本地统计" in messages[0]["content"]
    assert "coffee preference from A" not in messages[0]["content"]


def test_time_cutoff_no_future_no_full_persona_and_relevant_only(isolated_runtime):
    t = isolated_runtime
    claims = [
        _claim("past", "coffee current fact", sender="A", line=11),
        _claim("future", "coffee FUTURE_CLAIM", stamp="2021-06-25", sender="B", line=12),
        _claim("later", "coffee NEXT_ERA", stamp="2021-07-03", line=13),
        _claim("unrelated", "satellite UNRELATED_CLAIM", line=14),
        _claim("span", "coffee CROSS_CUTOFF", time_end="2021-06-20", line=15),
        _claim("inference", "coffee INFERENCE_NOT_FACT", category="inference", line=16),
        _claim("unknown", "coffee UNKNOWN_NOT_FACT", category="unknown", line=17),
    ]
    directory = t.persona(claims, [_era([c["id"] for c in claims]), _era(["later"], "july", "2021-07-01", "2021-07-31")], core_ids=["future", "inference"])
    engine = t.engine(directory)
    messages, sources = engine._prepare_turn("coffee", [])
    prompt = messages[0]["content"]
    assert len(engine.claims) == len(claims) and len(engine.timeline) == 2
    assert "coffee current fact" in prompt
    for forbidden in ("FUTURE_CLAIM", "NEXT_ERA", "UNRELATED_CLAIM", "CROSS_CUTOFF", "INFERENCE_NOT_FACT", "UNKNOWN_NOT_FACT", "FUTURE_FULL_PERSONA", "FUTURE_LABEL"):
        assert forbidden not in prompt
    assert {s["claim_id"] for s in sources} == {"past"}
    month_engine = t.engine(directory, era="2021-06")
    _, month_sources = month_engine._prepare_turn("coffee", [])
    assert "future" in {s["claim_id"] for s in month_sources}
    assert "later" not in {s["claim_id"] for s in month_sources}


def test_stable_core_candidate_filtered_and_priority(isolated_runtime):
    t = isolated_runtime
    claims = [
        _claim("explicit", "coffee user fact", claim_kind="user_fact", line=21),
        _claim("stable", "coffee stable fact", stamp="2020-05-03", line=22),
        _claim("style", "coffee concise style", category="style", line=23),
        _claim("emotion", "coffee temporary emotion", category="emotion_expression", line=24),
        _claim("future_core", "coffee FUTURE_CORE", stamp="2021-06-29", line=25),
    ]
    directory = t.persona(claims, [_era(["explicit", "style", "emotion"])], core_ids=["stable", "future_core", "emotion"])
    engine = t.engine(directory)
    assert [c["id"] for c in engine._relevant_claims("coffee")] == ["explicit", "stable", "style", "emotion"]
    _, sources = engine._prepare_turn("coffee", [])
    assert "future_core" not in {s["claim_id"] for s in sources}
    assert "emotion" not in engine.persona_core_ids


@pytest.mark.parametrize("patch", [
    {"category": "inference"}, {"category": "unknown"}, {"evidence": []},
    {"evidence": [{"source_line": 0, "sender": "A", "quote": "coffee"}]},
    {"evidence": [{"source_line": True, "sender": "A", "quote": "coffee"}]},
    {"evidence": [{"source_line": "0", "sender": "A", "quote": "coffee"}]},
    {"evidence": [{"source_line": "not-a-line", "sender": "A", "quote": "coffee"}]},
    {"origin": "ai_chat"},
    {"evidence": [{"source_line": 1, "sender": "assistant", "quote": "coffee"}]},
    {"evidence": [{"source_line": 1, "sender": "A", "quote": " "}]},
    {"confidence": float("nan")}, {"confidence": True}, {"confidence": -1},
    {"time_start": None}, {"time_start": "2021-07-01", "time_end": "2021-06-01"},
    {"claim_kind": "inference"}, {"source": "ai_chat"}, {"validated": False},
])
def test_invalid_and_ai_claims_never_promoted(isolated_runtime, patch):
    claim = _claim("invalid", "coffee INVALID_CLAIM", **patch)
    assert isolated_runtime.runtime.GroundedChatEngine._validated_claim(claim) is None


def test_duplicate_claim_ids_fail_loudly(isolated_runtime):
    t = isolated_runtime
    directory = t.persona([_claim("same", "coffee one"), _claim("same", "coffee two")])
    with pytest.raises(ValueError, match="重复 claim id"):
        t.engine(directory)


def test_artifact_symlink_cannot_escape(isolated_runtime, tmp_path):
    t = isolated_runtime
    directory = t.persona()
    outside = tmp_path / "outside-synthetic.json"
    outside.write_text("[]", encoding="utf-8")
    path = directory / "claims.json"
    assert path.resolve() == directory.resolve() / "claims.json"
    assert path.is_relative_to(tmp_path.resolve())
    path.unlink()
    path.symlink_to(outside)
    with pytest.raises(ValueError, match="路径越界"):
        t.engine(directory)


def test_persona_and_legacy_mirror_are_never_read(isolated_runtime, monkeypatch):
    t = isolated_runtime
    directory = t.persona()
    original = Path.read_text
    allowed = {directory / "claims.json", directory / "timeline.json", directory / "persona_core.json"}

    def guarded(path, *args, **kwargs):
        assert path in allowed, f"unexpected artifact read: {path.name}"
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded)
    engine = t.engine(directory)
    engine._prepare_turn("coffee", [])
    assert engine.session_summaries == [] and engine.corrections == ""


class HistoryFixture:
    def __init__(self, rows):
        self.rows, self.queries = rows, []

    def search(self, **kwargs):
        self.queries.append(kwargs)
        return copy.deepcopy(self.rows)

    def search_target_only(self, **kwargs):
        raise AssertionError("不能丢弃 A/other 的事实")

    def add_session_summary(self, **kwargs):
        raise AssertionError("AI 摘要不能写到历史库")

    def count(self):
        return len(self.rows)


def test_parent_rag_retained_without_target_or_romance_filter(isolated_runtime, monkeypatch):
    t = isolated_runtime
    history = HistoryFixture([_rag("RAG_A_FACT", sender="A", line=41), _rag("RAG_B_FACT", sender="B", line=42, record_id="b")])
    engine = t.engine(vector_store=history, embedder=object(), owner=77)
    parent_calls = []
    original = t.parent.ChatEngine._rag_search

    def spy(self, user_input):
        parent_calls.append(user_input)
        return original(self, user_input)

    monkeypatch.setattr(t.parent.ChatEngine, "_rag_search", spy)
    messages, sources = engine._prepare_turn("宝贝 coffee", [])
    assert parent_calls == ["宝贝 coffee"]
    assert history.queries[0]["query"] == "宝贝 coffee"
    assert history.queries[0]["where"] is None and history.queries[0]["top_k"] == 10
    rag_sources = [s for s in sources if s["source"] == "chat"]
    assert {(s["sender"], s["source_line"]) for s in rag_sources} == {("A", 41), ("B", 42)}
    assert all(s["time"] == "2021-06-03 12:00:00" for s in rag_sources)
    assert "原 RAG 检索参考" in messages[0]["content"] and "RAG_A_FACT" in messages[0]["content"]
    assert not hasattr(engine.vector_store, "add_session_summary")
    assert engine.vector_store.count() == 2


def test_rag_threshold_dedupe_does_not_misalign_sources(isolated_runtime):
    t = isolated_runtime
    low = _rag("LOW_SCORE", line=51, score=0.2, record_id="low")
    good = _rag("GOOD_SOURCE", sender="B", line=52, record_id="good")
    history = HistoryFixture([low, good, good, _rag("NAN_SCORE", score=float("nan"), line=53)])
    messages, sources = t.engine(vector_store=history, embedder=object())._prepare_turn("coffee", [])
    rag_sources = [s for s in sources if s["source"] == "chat"]
    assert len(rag_sources) == 1
    assert rag_sources[0]["source_line"] == 52 and rag_sources[0]["sender"] == "B"
    assert "GOOD_SOURCE" in messages[0]["content"] and "LOW_SCORE" not in messages[0]["content"]


def test_rag_no_future_inference_ai_or_missing_provenance(isolated_runtime):
    t = isolated_runtime
    rows = [
        _rag("VALID_PAST", record_id="past"),
        _rag("FUTURE_RAG", stamp="2021-06-11 00:00:00", record_id="future"),
        _rag("SPAN_RAG", end_ts="2021-06-11 00:00:00", record_id="span"),
        _rag("AI_RAG", source="ai_chat", record_id="ai"),
        _rag("SUMMARY_RAG", source="session_summary", record_id="summary"),
        _rag("INFERENCE_RAG", category="inference", record_id="inferred"),
        _rag("MISSING_TIME", start_ts="", end_ts="", record_id="missing"),
        _rag("MISSING_LINE", line=None, record_id="line"),
    ]
    engine = t.engine(vector_store=HistoryFixture(rows), embedder=object())
    messages, sources = engine._prepare_turn("coffee", [])
    assert [s["text"] for s in sources if s["source"] == "chat"] == ["VALID_PAST"]
    for marker in ("FUTURE_RAG", "SPAN_RAG", "AI_RAG", "SUMMARY_RAG", "INFERENCE_RAG", "MISSING_TIME", "MISSING_LINE"):
        assert marker not in messages[0]["content"]


def test_mixed_chunk_evidence_preserves_both_speakers_and_exact_lineage(isolated_runtime):
    t = isolated_runtime
    chunk = _rag("chunk normalized display", sender="B", line=61, end_ts="2021-06-03 12:05:00", source_lines="61,62")
    chunk["evidence"] = [
        {"sender": "A", "source_line": 61, "quote": "EXACT_A_QUOTE", "timestamp": "2021-06-03 12:00:00"},
        {"sender": "B", "source_line": 62, "quote": "EXACT_B_QUOTE", "time": "2021-06-03 12:05:00"},
    ]
    messages, sources = t.engine(vector_store=HistoryFixture([chunk]), embedder=object())._prepare_turn("coffee", [])
    rag_sources = [s for s in sources if s["source"] == "chat"]
    assert [(s["sender"], s["source_line"], s["text"]) for s in rag_sources] == [("A", 61, "EXACT_A_QUOTE"), ("B", 62, "EXACT_B_QUOTE")]
    assert "chunk normalized display" not in messages[0]["content"]


def test_future_evidence_cannot_hide_behind_past_chunk_metadata(isolated_runtime):
    t = isolated_runtime
    chunk = _rag("bad chunk")
    chunk["evidence"] = [{"sender": "B", "source_line": 71, "quote": "HIDDEN_FUTURE", "time": "2021-06-11 00:00:00"}]
    messages, sources = t.engine(vector_store=HistoryFixture([chunk]), embedder=object())._prepare_turn("coffee", [])
    assert "HIDDEN_FUTURE" not in messages[0]["content"]
    assert not [s for s in sources if s["source"] == "chat"]


def test_timezone_calendar_cutoff_and_end_of_month(isolated_runtime):
    t = isolated_runtime
    claims = [
        _claim("past", "coffee local past", stamp="2021-06-10T23:55:00+08:00"),
        _claim("future", "coffee NEXT_LOCAL_DAY", stamp="2021-06-11T00:05:00+08:00", line=81),
    ]
    era = _era(["past", "future"], start="2021-06-01T00:00:00+08:00", end="2021-06-30T23:59:59+08:00")
    engine = t.engine(t.persona(claims, [era]), era="2021-06-10")
    messages, sources = engine._prepare_turn("coffee", [])
    assert {s["claim_id"] for s in sources} == {"past"}
    assert "NEXT_LOCAL_DAY" not in messages[0]["content"]
    assert t.runtime._instant("2020-02", end=True).day == 29
    assert t.runtime._instant("2021-12", end=True).month == 12


@pytest.mark.parametrize("method", ["chat", "chat_with_sources", "chat_stream", "chat_stream_with_sources"])
@pytest.mark.parametrize("text", ["unlabelled", "【AI模拟回复】already", "  【AI模拟回复】【AI模拟回复】twice"])
def test_all_public_outputs_have_no_visible_label_or_side_effects(isolated_runtime, method, text):
    t = isolated_runtime
    t.api.text = text
    result = getattr(t.engine(), method)("coffee", [])
    if "stream" in method:
        events = list(result)
        assert {event["type"] for event in events} <= {"text", "usage", "sources"}
        reply = events[0]["content"]
        assert events[0]["streaming"] is False
    else:
        reply = result[0]
        assert result[1] == []
    expected = {'unlabelled': 'unlabelled', '【AI模拟回复】already': 'already', '  【AI模拟回复】【AI模拟回复】twice': 'twice'}[text]
    assert reply == expected
    assert len(t.calls) == 1 and t.permissions == ["chat"]
    assert t.calls[0]["model"] == "allowed-fixture" and t.calls[0]["stream"] is False


def test_no_retry_router_fallback_or_legacy_stream(isolated_runtime):
    t = isolated_runtime
    engine = t.engine()
    t.api.failure = RuntimeError("synthetic client failure")
    with pytest.raises(RuntimeError, match="synthetic client failure"):
        engine.chat_with_sources("coffee", [])
    assert len(t.calls) == 1 and t.permissions == ["chat"]
    with pytest.raises(NotImplementedError, match="非流式"):
        engine._call_stream([])
    assert len(t.calls) == 1


@pytest.mark.parametrize("selector", ["2021-06-10", "2021-06-01"])
def test_unauthorized_raises_instead_of_fake_reply(isolated_runtime, selector):
    t = isolated_runtime

    def denied(purpose="chat"):
        assert purpose == "chat"
        raise t.privacy.PrivacyError("synthetic consent denied")

    t.privacy.get_allowed_client = denied
    engine = t.engine(era=selector)
    with pytest.raises(t.privacy.PrivacyError, match="consent denied"):
        engine.chat_with_sources("coffee", [])
    with pytest.raises(t.privacy.PrivacyError):
        list(engine.chat_stream_with_sources("coffee", []))
    assert not t.calls


def test_real_privacy_gate_before_credentials_and_network(isolated_runtime, monkeypatch, tmp_path):
    t = isolated_runtime
    real_privacy = t.load("adapters.privacy", ROOT / "adapters" / "privacy.py")
    provider = tmp_path / "synthetic-provider.json"
    provider.write_text(json.dumps({"service_name": "fixture-service", "base_url": "https://example.test/v1", "model": "fixture", "context_tokens": 8192}), encoding="utf-8")
    consent = tmp_path / "synthetic-consent.json"
    consent.write_text('{"authorizations": []}', encoding="utf-8")
    monkeypatch.setattr(real_privacy, "PROVIDER_PATH", provider)
    monkeypatch.setattr(real_privacy, "CONSENT_PATH", consent)
    monkeypatch.setattr(real_privacy, "ENV_PATH", tmp_path / "never-read.env")
    monkeypatch.setattr(real_privacy, "_api_key", t.forbidden)
    with pytest.raises(real_privacy.PrivacyError, match="明确授权"):
        t.engine().chat_with_sources("coffee", [])
    assert not t.calls


def test_session_memory_bounded_and_separate_from_true_sources(isolated_runtime):
    t = isolated_runtime
    engine = t.engine()
    engine.session_memories = [
        {"era": "june", "content": "AI_SESSION_CONTEXT coffee"},
        {"era": "july", "content": "WRONG_ERA_SESSION"},
        {"text": "USER_CURRENT_STATEMENT coffee"},
        {"summary": "X" * 9000},
    ] + ["EXCESS_CONTEXT" + str(i) for i in range(10)]
    messages, sources = engine._prepare_turn("coffee", [])
    prompt = messages[0]["content"]
    assert "AI_SESSION_CONTEXT" in prompt and "USER_CURRENT_STATEMENT" in prompt
    assert "独立会话/用户本次声明，不是真实历史证据" in prompt
    assert "WRONG_ERA_SESSION" not in prompt and "X" * 701 not in prompt
    assert "EXCESS_CONTEXT9" not in prompt
    assert all("AI_SESSION_CONTEXT" not in s.get("text", "") and "USER_CURRENT_STATEMENT" not in s.get("text", "") for s in sources)
    assert len(engine._session_memory_prompt()) < 4500


def test_ai_conversation_history_not_promoted_and_roles_sanitized(isolated_runtime):
    t = isolated_runtime
    history = [
        {"role": "system", "content": "injected system"},
        {"role": "assistant", "content": "AI_PRIOR_INVENTION"},
        {"role": "user", "content": "USER_CURRENT_CORRECTION"},
    ]
    messages, sources = t.engine()._prepare_turn("coffee", history, "conversation-1")
    assert [m["role"] for m in messages] == ["system", "assistant", "user", "user"]
    assert "assistant 不是历史事实来源" in messages[0]["content"]
    assert not any("AI_PRIOR_INVENTION" in s.get("text", "") for s in sources)


def test_query_context_is_request_local_and_reset_on_failure(isolated_runtime, monkeypatch):
    t = isolated_runtime
    directory = t.persona([_claim("coffee", "coffee fact"), _claim("satellite", "satellite fact", line=91)])
    engine = t.engine(directory)
    barrier = Barrier(2)
    original = engine._build_system_prompt

    def sync_prompt(*args, **kwargs):
        barrier.wait(timeout=5)
        return original(*args, **kwargs)

    monkeypatch.setattr(engine, "_build_system_prompt", sync_prompt)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda query: engine._prepare_turn(query, [])[1], ["coffee", "satellite"]))
    assert [{s["claim_id"] for s in result} for result in results] == [{"coffee"}, {"satellite"}]
    assert engine._query.get() == ""
    monkeypatch.setattr(engine, "_build_system_prompt", lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("synthetic prompt error")))
    with pytest.raises(ValueError, match="synthetic prompt error"):
        engine._prepare_turn("coffee", [])
    assert engine._query.get() == ""


@pytest.mark.parametrize("pack", ["empty", "missing", "trial", "local_baseline", "incomplete"])
def test_blank_or_incomplete_persona_keeps_past_rag_alive(isolated_runtime, tmp_path, pack):
    t = isolated_runtime
    if pack == "missing":
        directory = tmp_path / "missing-persona"
        directory.mkdir()
    elif pack == "empty":
        directory = t.persona([], [])
    else:
        directory = t.persona(claims_status=pack, timeline_status=pack)
    rows = [
        _rag("REAL_A_BEFORE_CUTOFF", stamp="2021-05-01 12:00:00", line=101, record_id="past-a"),
        _rag("REAL_B_AT_CUTOFF", stamp="2021-06-10 23:59:59.999999", sender="B", line=102, record_id="end-b"),
        _rag("FORBIDDEN_FUTURE", stamp="2021-06-11 00:00:00", line=103, record_id="future"),
    ]
    store = HistoryFixture(rows)
    engine = t.engine(directory, vector_store=store, embedder=object())
    messages, sources = engine._prepare_turn("coffee", [])
    assert not engine.supported_era and engine.selected_era is None
    assert [s["text"] for s in sources] == ["REAL_A_BEFORE_CUTOFF", "REAL_B_AT_CUTOFF"]
    assert "FORBIDDEN_FUTURE" not in messages[0]["content"]
    assert "未确定关系阶段" in messages[0]["content"] and "仍允许" in messages[0]["content"]
    assert store.queries[0]["where"] is None


@pytest.mark.parametrize("era", [None, "latest"])
def test_latest_uses_today_shanghai_not_last_timeline_end(isolated_runtime, monkeypatch, era):
    t = isolated_runtime
    monkeypatch.setattr(t.runtime, "_today", lambda: "2021-07-09")
    rows = [
        _rag("AFTER_LAST_ERA", stamp="2021-07-08 08:00:00", record_id="after-era"),
        _rag("TODAY_LAST_INSTANT", stamp="2021-07-09 23:59:59.999999", record_id="today"),
        _rag("TOMORROW_NOT_ALLOWED", stamp="2021-07-10 00:00:00", record_id="tomorrow"),
    ]
    engine = t.engine(era=era, vector_store=HistoryFixture(rows), embedder=object())
    assert engine.requested_era == "latest"
    assert engine._cutoff == t.runtime._instant("2021-07-09", end=True)
    assert engine._cutoff.isoformat() == "2021-07-09T15:59:59.999999"
    assert engine.cutoff_epoch == engine._cutoff.replace(tzinfo=t.runtime.timezone.utc).timestamp()
    messages, sources = engine._prepare_turn("coffee", [])
    assert not engine.supported_era
    assert [s["text"] for s in sources] == ["AFTER_LAST_ERA", "TODAY_LAST_INSTANT"]
    assert "TOMORROW_NOT_ALLOWED" not in messages[0]["content"]


def test_supported_month_does_not_shrink_history_cutoff_to_era_end(isolated_runtime):
    t = isolated_runtime
    directory = t.persona(eras=[_era(["coffee"], end="2021-06-05")])
    rows = [
        _rag("AFTER_SUPPORTED_ERA", stamp="2021-06-28 10:00:00", record_id="after-era"),
        _rag("NEXT_MONTH_FORBIDDEN", stamp="2021-07-01 00:00:00", record_id="next-month"),
    ]
    engine = t.engine(directory, era="2021-06", vector_store=HistoryFixture(rows), embedder=object())
    assert engine.supported_era and engine._cutoff == t.runtime._instant("2021-06", end=True)
    messages, sources = engine._prepare_turn("coffee", [])
    assert [s["text"] for s in sources if s["source"] == "chat"] == ["AFTER_SUPPORTED_ERA"]
    assert "NEXT_MONTH_FORBIDDEN" not in messages[0]["content"]


def test_month_boundary_is_shanghai_even_without_supported_era(isolated_runtime):
    t = isolated_runtime
    rows = [
        _rag("LOCAL_JUNE", stamp="2021-06-30T23:59:59+08:00", record_id="local"),
        _rag("UTC_JUNE", stamp="2021-06-30T15:30:00Z", record_id="utc"),
        _rag("LOCAL_JULY_FORBIDDEN", stamp="2021-06-30T16:00:00Z", record_id="future"),
    ]
    engine = t.engine(t.persona([], []), era="2021-06", vector_store=HistoryFixture(rows), embedder=object())
    _, sources = engine._prepare_turn("coffee", [])
    assert [s["text"] for s in sources] == ["LOCAL_JUNE", "UTC_JUNE"]


def test_legacy_core_is_candidate_not_source_or_certified_fact(isolated_runtime):
    t = isolated_runtime
    claims = [
        _claim("core", "coffee LEGACY_CORE_CANDIDATE", stamp="2021-05-03", canonical=False),
        _claim("future", "coffee FUTURE_CORE_CANDIDATE", stamp="2021-06-28", canonical=False),
    ]
    directory = t.persona(claims, [], core_ids=["core", "future"], list_claims=True, list_timeline=True)
    engine = t.engine(directory, vector_store=HistoryFixture([_rag("REAL_COFFEE_TRUTH")]), embedder=object())
    messages, sources = engine._prepare_turn("coffee", [{"role": "user", "content": "USER_UNVERIFIED_CORRECTION"}])
    prompt = messages[0]["content"]
    assert not engine.supported_era and not engine._validated_claims
    assert "未经 canonical 核验的 core 候选" in prompt
    assert "LEGACY_CORE_CANDIDATE" not in prompt and "FUTURE_CORE_CANDIDATE" not in prompt
    assert "REAL_COFFEE_TRUTH" in prompt
    assert "绝不能覆盖真实历史事实" in prompt
    assert all("claim_id" not in source for source in sources)


def test_verified_core_is_relevant_and_timed_independent_of_stage(isolated_runtime):
    t = isolated_runtime
    core = _claim("core", "coffee PAST_CORE", stamp="2021-05-03", sender="B", line=114)
    directory = t.persona([core], [], core_ids=["core"])
    engine = t.engine(directory, vector_store=HistoryFixture([
        _rag("coffee PAST_CORE", stamp="2021-05-03 12:00:00", sender="B", line=114,
             source_id="synthetic.sqlite", message_id="message-114",
             source_file="/synthetic/fixture.txt", raw_sender="B")
    ]), embedder=object())
    _, sources = engine._prepare_turn("coffee", [])
    assert not engine.supported_era
    claim_sources = [source for source in sources if "claim_id" in source]
    assert {s["claim_id"] for s in claim_sources} == {"core"}
    source = claim_sources[0]
    assert source["time"] == source["time_start"] == source["time_end"] == "2021-05-03T12:00:00"
    assert source["time"] != source["claim_time_start"]
    assert source["source_id"] == "synthetic.sqlite" and source["message_id"] == "message-114"
    assert source["source_file"] == "/synthetic/fixture.txt" and source["raw_sender"] == "B"
    assert not any("claim_id" in source for source in engine._prepare_turn("satellite", [])[1])


@pytest.mark.parametrize("mutation", ["top_false", "evidence_false", "top_numeric", "missing_actual_time"])
def test_flags_or_claim_range_cannot_certify_canonical_evidence(isolated_runtime, mutation):
    t = isolated_runtime
    claim = _claim("core", "coffee UNVERIFIED_FLAG")
    if mutation == "top_false":
        claim["canonical_verified"] = False
    elif mutation == "top_numeric":
        claim["canonical_verified"] = 1
    elif mutation == "evidence_false":
        claim["evidence"][0]["canonical_verified"] = False
    else:
        claim["evidence"][0].pop("time")
        claim["evidence"][0].pop("timestamp")
    engine = t.engine(t.persona([claim], core_ids=["core"]))
    messages, sources = engine._prepare_turn("coffee", [])
    assert not engine.supported_era and not sources
    assert "coffee UNVERIFIED_FLAG" not in messages[0]["content"]
    assert "未经 canonical 核验" in messages[0]["content"]


def test_file_self_asserted_flags_are_rechecked_and_false_claim_removed(isolated_runtime):
    t = isolated_runtime
    engine = t.engine(vector_store=HistoryFixture([_rag("REAL_TRUTH_AFTER_RECHECK")]), embedder=object())
    assert engine.supported_era
    original = copy.deepcopy(engine.claims)
    called = []

    def verifier(claim):
        called.append(claim["id"])
        claim["statement"] = "MUTATION_MUST_NOT_REACH_STORED_CLAIMS"
        return False

    assert engine.verify_claims(verifier) == {"verified_claim_ids": [], "rejected_claim_ids": ["coffee"]}
    assert called == ["coffee"] and engine.claims == original
    assert not engine.supported_era and not engine._candidate_claims and not engine._validated_claims
    messages, sources = engine._prepare_turn("coffee", [])
    assert "coffee preference from A" not in messages[0]["content"]
    assert [s["text"] for s in sources] == ["REAL_TRUTH_AFTER_RECHECK"]


def test_verifier_preserves_full_raw_quote_actual_time_and_provenance(isolated_runtime):
    t = isolated_runtime
    claim = _claim("core", "coffee MODEL_CANDIDATE", stamp="2021-06-01", time_end="2021-06-10", canonical=False)
    engine = t.engine(t.persona([claim], core_ids=["core"]))
    full_raw = "  CANONICAL_FULL_RAW coffee body  "

    def verifier(candidate):
        result = _canonical(candidate, stamp="2021-06-05T09:30:00+08:00")
        result["evidence"][0].update(quote=full_raw, source_id="actual-synthetic-source", message_id="actual-synthetic-id", source_file="/synthetic/canonical.txt", raw_sender="合成A")
        return result

    engine.verify_claims(verifier)
    assert engine._validated_claims["core"]["evidence"][0]["quote"] == full_raw
    messages, sources = engine._prepare_turn("coffee", [])
    source = sources[0]
    assert source["time"] == source["start_ts"] == source["end_ts"] == "2021-06-05T09:30:00+08:00"
    assert source["claim_time_start"] == "2021-06-01" and source["claim_time_end"] == "2021-06-10"
    assert source["source_id"] == "actual-synthetic-source" and source["message_id"] == "actual-synthetic-id"
    assert source["source_file"] == "/synthetic/canonical.txt" and source["raw_sender"] == "合成A"
    assert "statement 仍是抽象候选" in messages[0]["content"]
    assert "文件标记不构成独立防篡改认证" in messages[0]["content"]


def test_verifier_true_requires_actual_quote_times_not_claim_date_range(isolated_runtime):
    t = isolated_runtime
    legacy = _claim("legacy", "coffee legacy range only", canonical=False)
    engine = t.engine(t.persona([legacy], core_ids=["legacy"]))
    report = engine.verify_claims(lambda claim: True)
    assert report == {"verified_claim_ids": [], "rejected_claim_ids": ["legacy"]}
    assert not engine._validated_claims


@pytest.mark.parametrize("change", ["inference", "future_time", "missing_evidence_flag", "different_id"])
def test_verifier_cannot_promote_invalid_or_future_evidence(isolated_runtime, change):
    t = isolated_runtime
    engine = t.engine()

    def verifier(claim):
        if change == "inference":
            claim["category"] = "inference"
        elif change == "future_time":
            claim["evidence"][0]["time"] = "2021-06-11T00:00:00+08:00"
        elif change == "missing_evidence_flag":
            claim["evidence"][0].pop("canonical_verified")
        else:
            claim["id"] = "different"
        return claim

    assert engine.verify_claims(verifier)["rejected_claim_ids"] == ["coffee"]
    assert not engine._validated_claims


def test_relationship_timeline_filename_preferred_with_legacy_fallback(isolated_runtime):
    t = isolated_runtime
    directory = t.persona()
    preferred = directory / "relationship_timeline.json"
    preferred.write_text(json.dumps({"status": "complete", "eras": [_era(["coffee"], era_id="preferred")] }), encoding="utf-8")
    engine = t.engine(directory)
    assert engine.selected_era["id"] == "preferred"
    assert engine.artifact_status["timeline_file"] == "relationship_timeline.json"


def test_rag_same_line_in_different_sources_keeps_identity_and_provenance(isolated_runtime):
    t = isolated_runtime
    chunks = []
    for suffix in ("one", "two"):
        chunk = _rag("normalized display", record_id="same-chunk-id", line=121, sender="B")
        chunk["evidence"] = [{"sender": "B", "source_line": 121, "quote": "SAME_ACTUAL_QUOTE", "time": "2021-06-03 12:00:00", "source_id": suffix, "message_id": "same-message-id", "source_file": f"/synthetic/{suffix}.txt", "raw_sender": "合成B"}]
        chunks.append(chunk)
    engine = t.engine(vector_store=HistoryFixture(chunks), embedder=object())
    _, sources = engine._prepare_turn("coffee", [])
    rag_sources = [source for source in sources if source["source"] == "chat"]
    assert len(rag_sources) == 2 and len({source["id"] for source in rag_sources}) == 2
    assert {source["source_id"] for source in rag_sources} == {"one", "two"}
    assert {source["source_file"] for source in rag_sources} == {"/synthetic/one.txt", "/synthetic/two.txt"}
    assert all(source["message_id"] == "same-message-id" and source["raw_sender"] == "合成B" for source in rag_sources)


def test_overlap_chunks_dedupe_same_canonical_message(isolated_runtime):
    t = isolated_runtime
    chunks = []
    for record_id in ("chunk-one", "chunk-two"):
        chunk = _rag("normalized", record_id=record_id)
        chunk["evidence"] = [{"sender": "A", "source_line": 131, "quote": "ONE_CANONICAL_MESSAGE", "time": "2021-06-03 12:00:00", "source_id": "same-source", "message_id": "same-id"}]
        chunks.append(chunk)
    _, sources = t.engine(vector_store=HistoryFixture(chunks), embedder=object())._prepare_turn("coffee", [])
    assert len([source for source in sources if source["source"] == "chat"]) == 1


def test_corpus_style_stats_inject_short_reply_guidance_without_quotes(isolated_runtime):
    t = isolated_runtime
    engine = t.engine()
    engine.style_profile = SimpleNamespace(
        avg_message_chars=9.03, messages_per_reply=1.03, no_end_punct_rate=0.95,
        emoji_rate=0.0006, top_fillers=["哈哈", "啊", "吧"], sample_size=16912,
    )
    messages, _sources = engine._prepare_turn("在吗", [])
    prompt = messages[0]["content"]
    assert "## 目标人物真实表达习惯" in prompt
    assert "平均每条9字" in prompt and "句末经常不打标点" in prompt
    assert "优先回一条很短的消息" in prompt
    assert "不要套用" in prompt
    assert "【AI模拟回复】" not in prompt and "PRIVATE_RAW_MESSAGE" not in prompt


def test_fiction_mode_prompt_is_explicit_and_keeps_history_boundary(isolated_runtime):
    t = isolated_runtime
    fiction = t.engine(t.persona(), relationship_mode="fiction")
    messages, _sources = fiction._prepare_turn("好想你", [])
    prompt = messages[0]["content"]
    assert "当前虚构亲密模式" in prompt
    assert "不得说成真实聊天记录中的历史" in prompt
    assert "绝不能覆盖真实历史事实" in prompt
    evidence = t.engine(t.persona())
    evidence_prompt = evidence._prepare_turn("好想你", [])[0][0]["content"]
    assert "当前虚构亲密模式" not in evidence_prompt
    assert "不自动假设浪漫关系" in evidence_prompt


class KeywordFixture:
    def __init__(self, rows):
        self.rows = rows

    def keyword_results(self, query, limit=4, source_tag="chat"):
        return copy.deepcopy(self.rows[:limit])


def test_lexical_fallback_is_merged_before_vector_fragments(isolated_runtime):
    t = isolated_runtime
    store = HistoryFixture([])
    store.history = KeywordFixture([_rag("我不喜欢吃蛋糕 但是我喜欢吃炸鸡柳 炸蘑菇这些", line=6440)])
    engine = t.engine(vector_store=store, embedder=object(), owner=77, relationship_mode="fiction")
    messages, sources = engine._prepare_turn("你记得咱俩以前喜欢吃什么吗", [])
    assert [source["text"] for source in sources if source.get("source") == "chat"] == [
        "我不喜欢吃蛋糕 但是我喜欢吃炸鸡柳 炸蘑菇这些"
    ]
    assert "我不喜欢吃蛋糕 但是我喜欢吃炸鸡柳 炸蘑菇这些" in messages[0]["content"]
