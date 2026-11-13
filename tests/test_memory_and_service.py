"""隔离记忆、门禁与私有 HTTP 路由的纯合成回归。"""
import hashlib
import json
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from starlette.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from adapters.privacy import AllowedClient, PrivacyError, provider_fingerprint
from adapters.session_memory import SessionMemory


class StubEmbedder:
    is_local = True

    def split_document(self, text):
        return [text]

    def embed(self, texts):
        return [[float(len(t) % 7), 1.0, 0.0] for t in texts]

    def embed_one(self, text):
        return self.embed([text])[0]


def test_session_memory_is_separate_and_never_real(tmp_path):
    history = tmp_path / 'real-chroma'
    history.mkdir()
    history_file = history / 'unchanged'
    history_file.write_text('real')
    memory = SessionMemory(tmp_path, owner=4, slug='synthetic', embedder=StubEmbedder())
    cid = memory.ensure_conversation(era='2021-01')
    result = memory.append_turn(cid, '合成用户声明', '合成答复', [], '2021-01', ai_generated=True)
    assert result['origin'] == 'ai_simulation' and not result['verified_real_history']
    assert memory.history(cid, '2021-01')[-1]['content'] == '合成答复'
    assert memory.stats()['verified_real_history'] is False
    assert memory.collection.get()['metadatas'][0]['origin'] == 'ai_simulation'
    with pytest.raises(ValueError):
        memory.append_turn(cid, 'bad', 'no trusted caller flag', [], '2021-01')
    with pytest.raises(ValueError):
        memory.ensure_conversation(cid, '2022-01')
    assert history_file.read_text() == 'real'


def test_gateway_recap_and_no_retry(monkeypatch, tmp_path):
    from adapters import privacy
    monkeypatch.setattr(privacy, 'LEDGER_PATH', tmp_path / 'ledger.json')
    monkeypatch.setattr(privacy, 'PROVIDER_PATH', tmp_path / 'provider.json')
    monkeypatch.setattr(privacy, 'ENV_PATH', tmp_path / 'provider.env')
    monkeypatch.setattr(privacy, '_source_sha', lambda: 'irrelevant')
    provider = {'service_name': 'fixture', 'base_url': 'http://127.0.0.1:9/v1', 'model': 'fixture',
                'context_tokens': 4096, 'prices': {'input_rmb_per_million': 2, 'output_rmb_per_million': 8}}
    grant = {'id': 'test', 'purposes': ['chat'], 'provider_fingerprint': provider_fingerprint(provider),
             'privacy_confirmed': True, 'confirmed_by_user': True, 'source_sha256': 'irrelevant',
             'max_calls': 2, 'max_cost_rmb': 0.00015}
    monkeypatch.setattr(privacy, '_find_grant', lambda *args, **kwargs: grant)
    privacy.PROVIDER_PATH.write_text(json.dumps(provider), encoding='utf-8')
    privacy.ENV_PATH.write_text('CHAT_LLM_API_KEY' + '=synthetic-fixture-only', encoding='utf-8')
    client = AllowedClient(provider, 'chat')
    calls = []
    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1))
    monkeypatch.setattr(client._client.chat.completions, 'create', create)
    client._create(model='fixture', messages=[{'role': 'user', 'content': 'synthetic'}], max_tokens=5)
    assert len(calls) == 1
    with pytest.raises(PrivacyError):
        client._create(model='fixture', messages=[{'role': 'user', 'content': 'next'}], max_tokens=16384)
    assert len(calls) == 1


def test_private_http_auth_and_ai_isolation(tmp_path):
    from adapters.service import attach_routes
    from fastapi import FastAPI
    synthetic_reply = '合成测试答复'
    class Engine:
        supported_era = False
        selected_era = None
        artifact_status = 'local_baseline'
        def _prepare_turn(self, message, history):
            return [{'role': 'user', 'content': message}], [{'source_line': 1, 'sender': 'B', 'quote': '合成证据'}]
        def chat_with_sources(self, message, history, conversation_id=None):
            return synthetic_reply, [], SimpleNamespace(prompt_tokens=1, completion_tokens=1), []
    class State:
        chat_slots = threading.BoundedSemaphore(2)
        def __init__(self):
            self.sessions = SessionMemory(tmp_path, owner=1, slug='synthetic', embedder=StubEmbedder())
            self.verifier = SimpleNamespace(enrich_sources=lambda rows: [{**row, 'canonical_verified': True} for row in rows])
        def engine(self, era=None):
            assert era in (None, '2021-01')
            return Engine()
        def status(self):
            return {'name': '合成人物', 'slug': 'synthetic', 'history_vectors': 1,
                    'history': {'first_message_at': '2021-01-01', 'last_message_at': '2021-02-01'},
                    'session_memory': self.sessions.stats(), 'persona_distilled': False, 'eras': [],
                    'model': {'chat_authorized': False}, 'embedding': '合成'}
        def conversation_lock(self, cid):
            return threading.Lock()
    app = FastAPI()
    # 合成测试不读取本机任何 token，只验证路由业务。
    import server.middleware as middleware
    app.dependency_overrides[middleware.require_auth] = lambda: 1
    app.add_middleware(__import__('starlette.middleware.trustedhost', fromlist=['TrustedHostMiddleware']).TrustedHostMiddleware, allowed_hosts=['testserver'])
    app = attach_routes(app, lambda slug, user_id: State())
    with TestClient(app) as client:
        headers = {'Authorization': 'Bearer fixture'}
        assert client.post('/chat/api/synthetic/chat', headers=headers, json={'message': 'x', 'unexpected': 1}).status_code == 422
        response = client.post('/chat/api/synthetic/chat', headers=headers, json={'message': 'x', 'era': '2021-01'})
        assert response.status_code == 200
        result = response.json()
        assert result['reply'] == synthetic_reply
        assert set(result['memory']) == {'turn_id', 'conversation_id', 'indexed'}
        history = client.get('/chat/api/synthetic/conversations/' + result['conversation_id'] + '?era=2021-01', headers=headers).json()['history']
        assert history[-1]['content'] == synthetic_reply
        assert client.post('/chat/api/synthetic/search', headers=headers, json={'query': 'x', 'era': '2021-01'}).json()['llm_calls'] == 0


def test_relationship_mode_persists_and_isolates_vectors(tmp_path):
    memory = SessionMemory(tmp_path, owner=9, slug='synthetic', embedder=StubEmbedder())
    evidence = memory.ensure_conversation('evidence-chat', '2021-01')
    fiction = memory.ensure_conversation('fiction-chat', '2021-01', 'fiction')
    memory.append_turn(evidence, '合成史实问题', '合成史实回答', [], '2021-01', ai_generated=True)
    memory.append_turn(fiction, '合成虚构问题', '合成虚构回答', [], '2021-01', ai_generated=True,
                       relationship_mode='fiction')
    assert memory.history(evidence, '2021-01', relationship_mode='evidence')[-1]['content'] == '合成史实回答'
    assert memory.history(fiction, '2021-01', relationship_mode='fiction')[-1]['content'] == '合成虚构回答'
    with pytest.raises(ValueError):
        memory.ensure_conversation('fiction-chat', '2021-01', 'evidence')
    with pytest.raises(ValueError):
        memory.history(fiction, '2021-01', relationship_mode='evidence')
    stored = memory.collection.get(where={'relationship_mode': 'fiction'}, include=['metadatas'])
    assert stored['ids'] and all(row['relationship_mode'] == 'fiction' for row in stored['metadatas'])


def test_duplicate_retry_returns_same_turn_without_new_memory(tmp_path):
    memory = SessionMemory(tmp_path, owner=9, slug='synthetic', embedder=StubEmbedder())
    cid = memory.ensure_conversation('retry-chat', '2021-01', 'fiction')
    first = memory.append_turn(cid, '合成重试', '合成首次回复', [{'source_line': 3}], '2021-01',
                               ai_generated=True, relationship_mode='fiction')
    duplicate = memory.duplicate_turn(cid, '合成重试', relationship_mode='fiction', within_seconds=45)
    assert duplicate['turn_id'] == first['turn_id'] and duplicate['reply'] == '合成首次回复'
    assert duplicate['sources'] == [{'source_line': 3}]
    assert memory.duplicate_turn(cid, '合成重试', relationship_mode='evidence') is None
    assert memory.duplicate_turn(cid, '另一句', relationship_mode='fiction') is None
    assert memory.stats()['turns'] == 1
