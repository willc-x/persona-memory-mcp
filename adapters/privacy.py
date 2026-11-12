"""唯一允许发送模型请求的出口：明确服务、用途、语料哈希、次数和费用授权。"""
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit

from adapters.upstream import ROOT
try:
    from adapters.upstream import DATA_ROOT
except ImportError:
    # 测试桩和旧版 ex-memory 不感知数据目录；显式环境变量仍优先。
    DATA_ROOT = Path(os.environ.get('CHAT_DATA_DIR', str(ROOT))).expanduser().resolve()

PROVIDER_PATH = DATA_ROOT / 'config' / 'provider.json'
ENV_PATH = DATA_ROOT / 'config' / 'provider.env'
CONSENT_PATH = DATA_ROOT / 'config' / 'privacy-consent.json'
LEDGER_PATH = DATA_ROOT / 'runtime' / 'api-usage.json'
PURPOSES = {'distill_trial', 'distill_full', 'chat'}


class PrivacyError(RuntimeError):
    """配置或明确授权不足；禁止请求，不能自动切换其他供应商。"""


def _read_json(path, default=None):
    if not path.is_file():
        return {} if default is None else default
    return json.loads(path.read_text(encoding='utf-8'))


def _atomic_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        f.write(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
        f.flush()
        os.fsync(f.fileno())
    tmp.chmod(0o600)
    os.replace(tmp, path)


@contextmanager
def _locked_ledger():
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    lock_path = LEDGER_PATH.with_suffix('.lock')
    with lock_path.open('a+', encoding='utf-8') as lock:
        lock_path.chmod(0o600)
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        data = _read_json(LEDGER_PATH, {'grants': {}, 'events': []})
        try:
            yield data
        finally:
            _atomic_json(LEDGER_PATH, data)
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def provider_fingerprint(provider):
    value = {k: provider.get(k) for k in ('service_name', 'base_url', 'model', 'context_tokens', 'prices')}
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _validate_provider(provider):
    base = provider.get('base_url', '').rstrip('/')
    model = provider.get('model', '').strip()
    if not base or not model or not provider.get('service_name'):
        raise PrivacyError('尚未选择模型服务。请填写本项目 config/provider.json；不要提供其他软件的密钥。')
    parsed = urlsplit(base)
    loopback = parsed.hostname in {'127.0.0.1', 'localhost', '::1'}
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise PrivacyError('模型 URL 禁止携带凭据、查询参数或片段。')
    if parsed.scheme != 'https' and not (parsed.scheme == 'http' and loopback):
        raise PrivacyError('非本机模型服务必须使用 HTTPS。')
    if not isinstance(provider.get('context_tokens'), int) or provider['context_tokens'] < 4096:
        raise PrivacyError('必须配置有效模型上下文预算（至少4096 tokens）。')
    return base, model, loopback


def _source_sha():
    # 新增真实来源会改变授权范围；原历史不变也不能默许把新文件发给旧供应商。
    history_db = DATA_ROOT / 'data' / 'real_history.sqlite'
    if history_db.is_file():
        import sqlite3
        with sqlite3.connect(f'file:{history_db}?mode=ro', uri=True) as db:
            hashes = [row[0] for row in db.execute('SELECT sha256 FROM sources ORDER BY sha256')]
        if hashes:
            return hashes[0] if len(hashes) == 1 else hashlib.sha256(json.dumps(hashes).encode()).hexdigest()
    return _read_json(DATA_ROOT / 'reports' / 'source_manifest.json').get('sha256')


def _find_grant(purpose, provider, plan_hash=None, estimated_cost=None):
    if purpose not in PURPOSES:
        raise PrivacyError('未知模型调用用途。')
    fingerprint = provider_fingerprint(provider)
    authorizations = _read_json(CONSENT_PATH, {'authorizations': []}).get('authorizations', [])
    for grant in reversed(authorizations):
        if grant.get('revoked') or purpose not in grant.get('purposes', []):
            continue
        if grant.get('provider_fingerprint') != fingerprint:
            continue
        if not grant.get('privacy_confirmed') or not grant.get('confirmed_by_user'):
            continue
        if grant.get('source_sha256') != _source_sha():
            continue
        if purpose.startswith('distill') and (not plan_hash or grant.get('plan_hash') != plan_hash):
            continue
        if not isinstance(grant.get('max_calls'), int) or grant['max_calls'] < 1:
            continue
        budget = grant.get('max_cost_rmb')
        if budget is not None:
            try:
                if not math.isfinite(float(budget)) or float(budget) < 0:
                    continue
            except (ValueError, TypeError):
                continue
        if estimated_cost is not None:
            try:
                if not math.isfinite(float(estimated_cost)) or float(estimated_cost) < 0:
                    continue
            except (ValueError, TypeError):
                continue
        if purpose == 'distill_full':
            # 费用预算与用户指定的本地路由授权是两种独立模式，不能互相冒充。
            usage_only = grant.get('billing_mode') == 'usage_only'
            if not usage_only:
                if estimated_cost is None or not grant.get('full_cost_confirmed'):
                    continue
                if budget is None or float(estimated_cost) > float(budget):
                    continue
            elif not grant.get('route_usage_confirmed') or budget is not None:
                continue
            if grant.get('allowed_base_url') and grant['allowed_base_url'] != provider.get('base_url', '').rstrip('/'):
                continue
            if grant.get('allowed_model') and grant['allowed_model'] != provider.get('model'):
                continue
        if not grant.get('id'):
            continue
        return grant
    label = {'distill_trial': '少量试蒸馏的服务及隐私', 'distill_full': '全量蒸馏的隐私及预计费用', 'chat': '日常AI聊天的服务及隐私'}[purpose]
    raise PrivacyError(f'未收到{label}明确授权；已阻止发送任何聊天正文。')


def _api_key(provider, loopback):
    name = provider.get('api_key_env', 'CHAT_LLM_API_KEY')
    if name != 'CHAT_LLM_API_KEY':
        raise PrivacyError('只允许读取本项目专用 CHAT_LLM_API_KEY，禁止读取无关软件凭据。')
    key = os.environ.get(name, '')
    if not key and ENV_PATH.is_file():
        for line in ENV_PATH.read_text(encoding='utf-8').splitlines():
            if line.strip().startswith(name + '='):
                key = line.split('=', 1)[1].strip().strip('\"\'')
                break
    if not key and loopback:
        return 'local-only-not-a-secret'
    if not key or key in {'TOKEN', 'YOUR_API_KEY'}:
        raise PrivacyError('请在 config/provider.env 填写 CHAT_LLM_API_KEY；不要将密钥发送到聊天窗口或写入源码。')
    return key


def _token_upper(messages):
    """用于硬费用/上下文门禁的保守上界：UTF-8字节数+消息封装开销。"""
    return sum(len(str(m.get('content', '')).encode('utf-8')) + 12 for m in messages) + 32


def _prices(provider):
    values = provider.get('prices') or {}
    a, b = values.get('input_rmb_per_million'), values.get('output_rmb_per_million')
    if a is None or b is None:
        return None
    a, b = float(a), float(b)
    if not (math.isfinite(a) and math.isfinite(b)) or min(a, b) < 0:
        raise PrivacyError('模型单价必须为非负有限数，不能用未知单价假装零费用。')
    return a, b


class AllowedClient:
    def __init__(self, provider, purpose, plan_hash=None, estimated_cost=None):
        self.provider = provider
        self.base_url, self.model, loopback = _validate_provider(provider)
        self.purpose, self.plan_hash, self.estimated_cost = purpose, plan_hash, estimated_cost
        self.grant = _find_grant(purpose, provider, plan_hash, estimated_cost)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))
        # 只有上面全部门禁通过后才读本项目密钥并初始化客户端。
        from openai import OpenAI
        import httpx
        timeout = float(provider.get('request_timeout_seconds', 90))
        if not math.isfinite(timeout) or timeout <= 0 or timeout > 900:
            raise PrivacyError('模型超时必须是0到900秒的有限值。')
        self._client = OpenAI(api_key=_api_key(provider, loopback), base_url=self.base_url,
                              max_retries=0, timeout=timeout,
                              http_client=httpx.Client(follow_redirects=False, trust_env=False, timeout=timeout))

    def _reserve(self, messages, max_tokens):
        prices = _prices(self.provider)
        tokens = _token_upper(messages)
        if tokens + max_tokens > self.provider['context_tokens']:
            raise PrivacyError('保守上下文预算不足，必须缩短段或分层合并；禁止直接截断证据/无限重试。')
        reserve = None if prices is None else (tokens * prices[0] + max_tokens * prices[1]) / 1_000_000
        grant = self.grant
        with _locked_ledger() as ledger:
            record = ledger['grants'].setdefault(grant['id'], {'attempts': 0, 'reserved_cost_rmb': 0.0, 'actual_cost_rmb': 0.0})
            if record['attempts'] >= grant['max_calls']:
                raise PrivacyError('已达到明确授权的模型请求次数上限。')
            budget = grant.get('max_cost_rmb')
            if budget is not None:
                if reserve is None:
                    raise PrivacyError('单价未知，无法遵守已授权费用上限，已阻止调用。')
                if record['reserved_cost_rmb'] + reserve > float(budget) + 1e-12:
                    raise PrivacyError('下一请求可能超过已授权费用上限，已阻止调用。')
            record['attempts'] += 1
            record['reserved_cost_rmb'] += reserve or 0.0
            event_id = len(ledger['events'])
            ledger['events'].append({'id': event_id, 'grant_id': grant['id'], 'purpose': self.purpose,
                                     'model': self.model, 'provider_fingerprint': provider_fingerprint(self.provider),
                                     'at': datetime.now(timezone.utc).isoformat(), 'status': 'reserved',
                                     'input_token_upper': tokens, 'output_token_limit': max_tokens,
                                     'reserved_cost_rmb': reserve})
        return event_id

    def _create(self, *, model, messages, **kwargs):
        # 调用时重新检查撤销/配置变化；创建过客户端不意味着长期放行。
        current = _read_json(PROVIDER_PATH)
        if provider_fingerprint(current) != provider_fingerprint(self.provider):
            raise PrivacyError('服务配置已变化，原授权不能用于新服务。')
        self.grant = _find_grant(self.purpose, current, self.plan_hash, self.estimated_cost)
        if model != self.model or kwargs.get('stream'):
            raise PrivacyError('禁止切换未授权模型或绕过非流式费用记账。')
        if not isinstance(messages, list) or not all(isinstance(m, dict) and isinstance(m.get('content'), str) for m in messages):
            raise PrivacyError('只允许明确的文本对话请求，不接受文件上传或非文本payload。')
        max_tokens = int(kwargs.get('max_tokens', kwargs.get('max_completion_tokens', 4096)))
        if not 1 <= max_tokens <= 65536:
            raise PrivacyError('输出token上限必须介于1和65536。')
        if kwargs.get('n', 1) != 1:
            raise PrivacyError('每个受控请求只允许生成一个结果，禁止绕过输出预算。')
        allowed = {'max_tokens', 'max_completion_tokens', 'temperature', 'top_p', 'frequency_penalty', 'presence_penalty', 'response_format', 'seed', 'n', 'stream'}
        if set(kwargs) - allowed:
            raise PrivacyError('请求含未允许参数，禁止用额外payload覆盖模型/消息/费用限制。')
        kwargs.pop('max_completion_tokens', None)
        kwargs['max_tokens'] = max_tokens
        if urlsplit(self.base_url).hostname == 'api.deepseek.com':
            # 默认思考模式可能耗尽 JSON 输出预算；蒸馏和模拟聊天统一非思考模式。
            kwargs['extra_body'] = {'thinking': {'type': 'disabled'}}
        event_id = self._reserve(messages, max_tokens)
        try:
            result = self._client.chat.completions.create(model=model, messages=messages, **kwargs)
        except Exception as exc:
            with _locked_ledger() as ledger:
                ledger['events'][event_id].update(status='failed', error_type=type(exc).__name__)
            # 不记录可能包含正文/密钥的错误体；失败的预留费用不退款，防止重试超预算。
            raise PrivacyError(f'模型请求失败（{type(exc).__name__}）；未自动重试，次数与预算已记账。') from None
        usage = getattr(result, 'usage', None)
        incoming = int(getattr(usage, 'prompt_tokens', 0) or 0)
        outgoing = int(getattr(usage, 'completion_tokens', 0) or 0)
        prices = _prices(self.provider)
        actual = None if prices is None or usage is None else (incoming * prices[0] + outgoing * prices[1]) / 1_000_000
        with _locked_ledger() as ledger:
            ledger['events'][event_id].update(status='completed', prompt_tokens=incoming,
                                              completion_tokens=outgoing, actual_cost_rmb=actual)
            ledger['grants'][self.grant['id']]['actual_cost_rmb'] += actual or 0.0
        return result


def get_allowed_client(purpose='chat', estimated_cost=None, plan_hash=None):
    """没有明确授权时立即失败；既不读密钥也不发送任何模型请求。"""
    provider = _read_json(PROVIDER_PATH)
    return AllowedClient(provider, purpose, plan_hash, estimated_cost)


def configuration_status():
    """安全状态概览，不返回任何密钥或聊天正文。"""
    provider = _read_json(PROVIDER_PATH)
    result = {'configured': False, 'service_name': provider.get('service_name') or None,
              'model': provider.get('model') or None, 'chat_authorized': False,
              'external_data_sent_by_adapter': len(_read_json(LEDGER_PATH, {'events': []}).get('events', [])) > 0}
    try:
        _validate_provider(provider)
        result['configured'] = True
        _find_grant('chat', provider)
        result['chat_authorized'] = True
    except PrivacyError:
        pass
    return result
