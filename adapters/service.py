"""本机 ex-memory FastAPI 扩展：原生认证/ChatEngine/RAG，独立AI记忆与单一模型门禁。"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import threading
from typing import Optional

from adapters.upstream import ROOT, DATA_ROOT, prepare_upstream
prepare_upstream()

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ConfigDict
from starlette.middleware.trustedhost import TrustedHostMiddleware

from config import get_collection_name, resolve_ex_dir
from core.exe_access import assert_exe_access
from core.validation import validate_slug
from server.middleware import require_auth
from adapters.local_memory import LocalEmbedder, RealHistory, HistoricalVectorStore
from adapters.session_memory import SessionMemory, validate_conversation_id
from adapters.evidence import CanonicalVerifier
from adapters.privacy import PrivacyError, configuration_status, _read_json, PROVIDER_PATH

STATIC = ROOT / 'web'
INSTALLATION = DATA_ROOT / 'runtime' / 'installation.json'


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    query: str = Field(min_length=1, max_length=5000)
    era: Optional[str] = Field(default=None, max_length=80)


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    message: str = Field(min_length=1, max_length=5000)
    era: Optional[str] = Field(default=None, max_length=80)
    conversation_id: Optional[str] = Field(default=None, max_length=80)


class RuntimeState:
    def __init__(self, installation):
        self.installation = installation
        self.owner, self.slug = int(installation['owner']), installation['slug']
        self.mirror = resolve_ex_dir(self.slug, self.owner)
        if self.mirror.resolve() != Path(installation['mirror_directory']).resolve():
            raise ValueError('安装镜像位置与上游属主目录不一致')
        self.history = RealHistory()
        self.embedder = LocalEmbedder.shared()
        self.verifier = CanonicalVerifier(self.history)
        self.history_store = HistoricalVectorStore(self.mirror / 'chroma_db', get_collection_name(self.slug),
                                slug=self.slug, owner=self.owner, history=self.history)
        self.sessions = SessionMemory(self.mirror, owner=self.owner, slug=self.slug, embedder=self.embedder)
        self.chat_slots = threading.BoundedSemaphore(2)
        self._conversation_locks = {}
        self._lock = threading.Lock()

    def conversation_lock(self, conversation_id):
        with self._lock:
            return self._conversation_locks.setdefault(conversation_id, threading.Lock())

    def engine(self, era=None):
        from adapters.runtime_adapter import GroundedChatEngine
        provider = _read_json(PROVIDER_PATH)
        # 显式传配置；模型请求仍由privacy client选定，绝不回落到上游默认密钥。
        model_config = {'model': provider.get('model') or 'unconfigured', 'temperature': 0.55,
                        'top_p': 0.9, 'frequency_penalty': 0.0, 'max_tokens': 1024}
        # cutoff属于本轮对象，不可在共享store上修改以免两个时期并发串库。
        store = HistoricalVectorStore(self.mirror / 'chroma_db', get_collection_name(self.slug),
                             slug=self.slug, owner=self.owner, history=self.history)
        persona_dir = Path(self.installation['persona_directory']) / 'runtime'
        engine = GroundedChatEngine(self.slug, owner=self.owner, vector_store=store, embedder=self.embedder,
                                   persona_dir=persona_dir, era=era, model_config=model_config)
        if getattr(engine, '_cutoff', None) is not None:
            cutoff = engine._cutoff
            store.cutoff = int(cutoff.replace(tzinfo=cutoff.tzinfo or timezone.utc).timestamp())
        # 磁盘文件里的verified标记不足以代替真实库；每次构造重新回查。
        if hasattr(engine, 'verify_claims'):
            engine.verify_claims(self.verifier.verify)
        else:
            checked = {key: self.verifier.verify(claim) for key, claim in engine._validated_claims.items()}
            engine._validated_claims = {key: claim for key, claim in checked.items() if claim is not None}
            engine._era_claim_ids &= set(engine._validated_claims)
            engine.persona_core_ids &= set(engine._validated_claims)
        return engine

    def status(self):
        persona_dir = Path(self.installation['persona_directory']) / 'runtime'
        manifest = _read_json(persona_dir / 'manifest.json')
        timeline = _read_json(persona_dir / 'timeline.json', {'eras': []})
        eras = timeline.get('eras', []) if isinstance(timeline, dict) else timeline
        return {'slug': self.slug, 'name': self.installation['role_mapping']['B'],
                'history_ready': self.history_store.count() > 0, 'history_vectors': self.history_store.count(),
                'history': self.history.stats(), 'session_memory': self.sessions.stats(),
                'embedding': '本机 BAAI/bge-small-zh-v1.5；不发送到Embedding API',
                'persona_status': manifest.get('status', 'pending'),
                'persona_distilled': manifest.get('status') == 'complete',
                'eras': [{'id': era['id'], 'start': era['start'], 'end': era['end'],
                         'label': era.get('label', '有证据的时期')} for era in eras if isinstance(era, dict)],
                'model': configuration_status(), 'reply_separation': 'independent_session'}


_states = {}
_state_lock = threading.Lock()


def _installation():
    if not INSTALLATION.is_file():
        raise HTTPException(status_code=503, detail='真实历史尚未导入，请先运行本项目导入脚本')
    return json.loads(INSTALLATION.read_text(encoding='utf-8'))


def _access(slug, user_id):
    try:
        valid = validate_slug(slug)
        if valid != slug:
            raise ValueError('镜像名称不能隐式转换')
    except ValueError:
        raise HTTPException(status_code=400, detail='镜像名称不合法') from None
    installation = _installation()
    if slug != installation['slug']:
        raise HTTPException(status_code=404, detail='镜像不存在')
    if user_id != installation['owner']:
        raise HTTPException(status_code=403, detail='该镜像不属于当前账户')
    try:
        assert_exe_access(slug, user_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail='镜像不存在') from None
    except PermissionError:
        raise HTTPException(status_code=403, detail='无权访问该镜像') from None
    return installation


def _state(slug, user_id):
    installation = _access(slug, user_id)
    key = (user_id, slug)
    with _state_lock:
        if key not in _states:
            _states[key] = RuntimeState(installation)
        return _states[key]


def _era(value):
    return None if value in {None, '', 'latest'} else value


def _input(value):
    if not value.strip() or len(value.encode('utf-8')) > 15000:
        raise HTTPException(status_code=400, detail='消息不能为空且不能超过15000 UTF-8字节')
    return value


def attach_routes(app, state_factory=None):
    factory = state_factory or _state
    router = APIRouter(prefix='/chat/api')

    @router.get('/me')
    def me(user_id: int = Depends(require_auth)):
        installation = _installation()
        state = factory(installation['slug'], user_id)
        return state.status()

    @router.get('/{slug}/status')
    def status(slug: str, user_id: int = Depends(require_auth)):
        return factory(slug, user_id).status()

    @router.get('/{slug}/conversations')
    def conversations(slug: str, user_id: int = Depends(require_auth)):
        return {'conversations': factory(slug, user_id).sessions.conversations()}

    @router.get('/{slug}/conversations/{conversation_id}')
    def history(slug: str, conversation_id: str, era: Optional[str] = None, user_id: int = Depends(require_auth)):
        state = factory(slug, user_id)
        try:
            validate_conversation_id(conversation_id)
            return {'conversation_id': conversation_id, 'era': era or 'latest',
                    'history': state.sessions.history(conversation_id, _era(era), max_turns=100, max_bytes=1000000)}
        except ValueError:
            raise HTTPException(status_code=400, detail='会话ID或时期不匹配') from None

    @router.post('/{slug}/search')
    def search(slug: str, req: SearchRequest, user_id: int = Depends(require_auth)):
        state = factory(slug, user_id)
        query = _input(req.query)
        try:
            engine = state.engine(_era(req.era))
            _messages, sources = engine._prepare_turn(query, [])
            sources = state.verifier.enrich_sources(sources)
            return {'sources': sources, 'requested_era': req.era or 'latest',
                    'supported_relationship_era': engine.supported_era,
                    'relationship_era': engine.selected_era, 'llm_calls': 0,
                    'retrieval_only': True, 'no_evidence': not bool(sources),
                    'persona_status': engine.artifact_status}
        except ValueError as exc:
            # 不返回可能含原文的异常；非法时期可安全映射400。
            raise HTTPException(status_code=400, detail='时期选择不合法或证据资料格式不一致') from None
        except FileNotFoundError:
            raise HTTPException(status_code=503, detail='本地人格候选包未初始化；历史库仍保留') from None

    @router.post('/{slug}/chat')
    def chat(slug: str, req: ChatRequest, user_id: int = Depends(require_auth)):
        state = factory(slug, user_id)
        message = _input(req.message)
        era = _era(req.era)
        try:
            engine = state.engine(era)  # 日期校验应在API请求前完成。
            conversation_id = state.sessions.ensure_conversation(req.conversation_id, era)
        except ValueError:
            raise HTTPException(status_code=400, detail='时期/会话ID无效或与既有会话不一致') from None
        except FileNotFoundError:
            raise HTTPException(status_code=503, detail='本地人格候选包尚未初始化') from None
        if not state.chat_slots.acquire(blocking=False):
            raise HTTPException(status_code=429, detail='已有两个请求运行，请稍后再试')
        try:
            with state.conversation_lock(conversation_id):
                history = state.sessions.history(conversation_id, era)
                engine.session_memories = state.sessions.search(message, era=era)
                reply, _stickers, usage, sources = engine.chat_with_sources(message, history, conversation_id)
                sources = state.verifier.enrich_sources(sources)
                saved = state.sessions.append_turn(conversation_id, message, reply, sources, era, ai_generated=True)
                return {'reply': reply, 'sources': sources, 'conversation_id': conversation_id,
                        'era': req.era or 'latest', 'relationship_era': engine.selected_era,
                        'supported_relationship_era': engine.supported_era, 'memory': {'turn_id': saved['turn_id'], 'conversation_id': saved['conversation_id'], 'indexed': saved['indexed']},
                        'persona_status': engine.artifact_status,
                        'usage': {'prompt_tokens': getattr(usage, 'prompt_tokens', None),
                                  'completion_tokens': getattr(usage, 'completion_tokens', None)}}
        except PrivacyError as exc:
            raise HTTPException(status_code=412, detail=str(exc)) from None
        except ValueError:
            raise HTTPException(status_code=400, detail='会话状态或模型预算校验未通过；未覆盖真实历史') from None
        finally:
            state.chat_slots.release()

    app.include_router(router)

    @app.get('/')
    def index():
        return RedirectResponse('/chat', status_code=307)

    @app.get('/chat')
    def chat_page():
        return FileResponse(STATIC / 'index.html', headers={'Cache-Control': 'no-store'})

    @app.get('/chat/health')
    def private_health():
        return {'status': 'running', 'mode': 'local-embedding-guarded-llm',
                'real_history_separate': True}

    if STATIC.is_dir():
        app.mount('/chat-static', StaticFiles(directory=str(STATIC)), name='chat-static')

    @app.middleware('http')
    async def privacy_headers(request: Request, call_next):
        # 不让真实查询通过URL进入访问日志。检索/chat仅支持POST JSON。
        response = await call_next(request)
        response.headers['Cache-Control'] = 'no-store'
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['Referrer-Policy'] = 'no-referrer'
        response.headers['X-Frame-Options'] = 'DENY'
        response.headers['Content-Security-Policy'] = "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; form-action 'self'; base-uri 'none'"
        return response
    return app


def create_private_app(test_mode=False, state_factory=None):
    from server.app import app as native_app
    if test_mode:
        app = FastAPI(title='ex-memory 聊天适配测试', docs_url=None, redoc_url=None)
        app.router.routes.extend(route for route in native_app.router.routes
                                if getattr(route, 'path', None) in {'/api/auth/login', '/api/auth/logout', '/health', '/health/ready'})
    else:
        app = native_app
        # 关闭本次安装的其他原生云/付费/修改端点，避免绕开唯一privacy出口。
        retained = {'/api/auth/login', '/api/auth/logout', '/health', '/health/ready', '/openapi.json'}
        app.router.routes[:] = [route for route in app.router.routes
                               if getattr(route, 'path', None) in retained or not str(getattr(route, 'path', '')).startswith(('/api/', '/static', '/metrics'))]
        app.title = 'ex-memory · 真实聊天历史与 AI 模拟'
        app.description = '本地证据检索、分时期画像、独立AI会话；所有模拟回复不代表真人思想。'
        app.openapi_schema = None
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=['localhost', '127.0.0.1', '[::1]'] + (['testserver'] if test_mode else []))
    return attach_routes(app, state_factory)


app = create_private_app()
