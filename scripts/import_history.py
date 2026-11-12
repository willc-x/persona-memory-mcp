#!/usr/bin/env python3
"""走 ex-memory 的 JSON 聊天导入接口，附加本地证据账本/稳定ID/断点重导。"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import secrets
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from adapters.upstream import DATA_ROOT, prepare_upstream
prepare_upstream()
from adapters.local_memory import RealHistory, HistoricalVectorStore, LocalEmbedder, atomic_json


def ensure_local_account():
    from server.auth import init_db, register_user, login_user, validate_token, revoke_token
    init_db()
    path = DATA_ROOT / 'config' / 'local-login.json'
    if path.is_file():
        login = json.loads(path.read_text())
    else:
        login = {'username': 'chat-local', 'password': secrets.token_urlsafe(18)}
        error = register_user(login['username'], login['password'])
        if error:
            raise ValueError('本机账号已存在且没有本项目登录文件；请勿重置已有密码')
        atomic_json(path, login)
    token = login_user(login['username'], login['password'])
    if not token:
        raise ValueError('本机账号验证失败，拒绝分配记忆所属者')
    owner = validate_token(token)
    revoke_token(token)
    if not owner:
        raise ValueError('本机账号所属者不存在')
    return int(owner)


def _verified_source(audit, data_root=DATA_ROOT):
    """Resolve and verify that the imported source is a stable file below data_root."""
    raw = audit.get('input_file')
    expected = audit.get('source_sha256')
    if not isinstance(raw, str) or not raw or not isinstance(expected, str) or len(expected) != 64:
        raise ValueError('导入审计缺少有效的源路径或 SHA-256')
    data_root = Path(data_root).expanduser().resolve()
    source = Path(raw).resolve()
    if source == data_root or not source.is_relative_to(data_root) or not source.is_file():
        raise ValueError('真实源路径不在数据目录内或不是普通文件')
    if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
        raise ValueError('真实源文件在准备后发生变化，拒绝导入')
    return source


def _source_transition(meta, installation, source_sha256):
    """Return (source_changed, previous_source_sha256) without mutating state."""
    previous = installation.get('real_source_sha256') or meta.get('original_source_sha256')
    previous = previous if isinstance(previous, str) and len(previous) == 64 else None
    return bool(previous and previous != source_sha256), previous


def _persona_directory(installation, slug, data_root=DATA_ROOT):
    """保留已安装画像路径；仅首次导入回退到 artifacts/<slug>。"""
    data_root = Path(data_root).expanduser().resolve()
    raw = installation.get('persona_directory')
    if isinstance(raw, str) and raw.strip():
        candidate = Path(raw).expanduser()
        if candidate.is_absolute():
            candidate = candidate.resolve()
            if candidate != data_root and candidate.is_relative_to(data_root):
                return candidate
    return data_root / 'artifacts' / slug


def _prepare_import_state(meta, installation, audit, owner, slug, *, mirror=None,
                          data_root=DATA_ROOT):
    """Pure state transition before ingest; preserves an existing persona."""
    data_root = Path(data_root).expanduser().resolve()
    source_sha256 = audit['source_sha256']
    source_changed, previous_source = _source_transition(meta, installation, source_sha256)
    persona_directory = _persona_directory(installation, slug, data_root)
    if mirror is None:
        raw_mirror = installation.get('mirror_directory')
        if isinstance(raw_mirror, str) and raw_mirror.strip():
            candidate = Path(raw_mirror).expanduser().resolve()
            mirror = candidate if candidate.is_relative_to(data_root) and candidate != data_root else None
        if mirror is None:
            mirror = data_root / 'exes' / str(owner) / slug
    stale = bool(meta.get('stale_after_history_append')) or source_changed
    next_meta = dict(meta)
    next_meta.update(
        slug=slug, name=audit['target'], target_name=audit['target'], owner_user_id=owner,
        creation_source='chat_import', stage='source',
        privacy_mode='local-embedding-guarded-llm', role_mapping=audit['role_mapping'],
        personality_distilled=bool(meta.get('personality_distilled', False)),
        stale_after_history_append=stale, persona_directory=str(persona_directory),
        import_source_changed=source_changed, original_source_sha256=source_sha256,
        created_at=meta.get('created_at') or datetime.now().isoformat(),
    )
    if source_changed and previous_source:
        next_meta['previous_source_sha256'] = previous_source
    # 同源重导不把已经完成的画像退回 pending；只有新来源进入索引重建状态。
    next_meta.setdefault('pipeline_state', 'index_ready_persona_pending')
    if source_changed:
        next_meta['pipeline_state'] = 'index_building'
    next_installation = dict(installation)
    next_installation.update(
        owner=owner, slug=slug, mirror_directory=str(Path(mirror).resolve()),
        persona_directory=str(persona_directory), role_mapping=audit['role_mapping'],
        real_source_sha256=source_sha256, stale_after_history_append=stale,
        import_source_changed=source_changed,
    )
    if source_changed and previous_source:
        next_installation['previous_source_sha256'] = previous_source
    return next_meta, next_installation


def _finalize_import_state(meta, installation, *, message_count, vector_count):
    """Pure end-of-ingest transition; never claims the old persona covers new data."""
    stale = bool(meta.get('stale_after_history_append'))
    if stale:
        status = 'history_ready_persona_stale'
        pipeline_state = 'index_ready_persona_stale'
    elif installation.get('persona_integrated'):
        status = 'history_ready_persona_existing'
        pipeline_state = 'index_ready_persona_integrated'
    else:
        status = 'history_ready_persona_pending'
        pipeline_state = 'index_ready_persona_pending'
    next_meta = dict(meta)
    next_meta.update(pipeline_state=pipeline_state, history_vector_count=vector_count,
                     history_messages=message_count, history_ready=True,
                     personality_distilled=bool(meta.get('personality_distilled', False)),
                     stale_after_history_append=stale)
    next_installation = dict(installation)
    next_installation.update(history_ready=True, history_vector_count=vector_count,
                             history_messages=message_count,
                             stale_after_history_append=stale,
                             persona_distilled=bool(installation.get('persona_distilled', False)))
    return next_meta, next_installation, status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, default=DATA_ROOT / 'prepared')
    parser.add_argument('--slug', default='persona')
    args = parser.parse_args()
    from config import ensure_ex_dirs_owned, get_collection_name
    from core.validation import validate_slug
    from memory.ingest import ingest_chat_file
    from core.corpus_store import count_messages
    slug = validate_slug(args.slug)
    if slug != args.slug:
        raise ValueError('镜像名称不能发生隐式路径归一化')
    prepared = args.prepared.resolve()
    if prepared != DATA_ROOT and DATA_ROOT not in prepared.parents:
        raise ValueError('导入材料必须放在当前数据目录内')
    audit = json.loads((prepared / 'audit.json').read_text())
    source = _verified_source(audit, DATA_ROOT)
    history = RealHistory()
    ledger_result = history.import_ledger(prepared / 'source_ledger.jsonl', audit)
    owner = ensure_local_account()
    mirror = ensure_ex_dirs_owned(slug, owner)
    mirror.chmod(0o700)
    meta_path = mirror / 'meta.json'
    meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
    if meta.get('owner_user_id') not in {None, owner}:
        raise ValueError('镜像已有不同所属者，拒绝覆盖')
    if meta.get('name') and meta['name'] != audit['target']:
        raise ValueError('镜像绑定了不同人物，拒绝把双方混在同一镜像')
    installation_path = DATA_ROOT / 'runtime' / 'installation.json'
    installation = (json.loads(installation_path.read_text(encoding='utf-8'))
                    if installation_path.is_file() else {})
    meta, installation = _prepare_import_state(
        meta, installation, audit, owner, slug, mirror=mirror, data_root=DATA_ROOT)
    atomic_json(meta_path, meta)
    skill_path = mirror / 'SKILL.md'
    if not skill_path.is_file():
        skill_path.write_text('# ' + audit['target'] + ' · 人格对话镜像\n\n'
                              '真实聊天历史只作证据；不得代言真人思想。\n'
                              '人格蒸馏尚未执行，不能宣称已完成画像。\n'
                              '稳定/时期画像由 adapters.runtime_adapter 动态加载，用户可见回复不加模拟/AI标识；独立会话不写回真实历史。\n', encoding='utf-8')
        skill_path.chmod(0o600)
    embedder = LocalEmbedder.shared()
    store = HistoricalVectorStore(mirror / 'chroma_db', get_collection_name(slug), slug=slug, owner=owner, history=history)
    import_id = 'real-chat-' + audit['source_sha256']
    messages, chunk_count = ingest_chat_file(str(prepared / 'ex_memory.json'), slug, audit['target'],
                                            store, embedder, owner, import_id=import_id, require_target=True)
    if len(messages) != audit['ex_memory_index_messages'] or store.count() < chunk_count:
        raise ValueError('导入消息数或已落盘向量数校验失败')
    meta, installation, import_status = _finalize_import_state(
        meta, installation, message_count=len(messages), vector_count=store.count())
    atomic_json(meta_path, meta)
    atomic_json(installation_path, installation)
    report = {'status': import_status, 'source_sha256': audit['source_sha256'],
              'source_changed': bool(meta.get('import_source_changed')),
              'previous_source_sha256': meta.get('previous_source_sha256'),
              'stale_after_history_append': bool(meta.get('stale_after_history_append')),
              'persona_directory': meta.get('persona_directory'),
              'persona_current_source_verified': False,
              'owner': owner, 'slug': slug, 'parser_route': 'memory.ingest.ingest_chat_file(JSON)',
              'history_ledger': ledger_result, 'canonical_statistics': history.stats(),
              'indexable_messages': len(messages), 'corpus_archive_messages': count_messages(slug, owner),
              'expected_chunks_this_source': chunk_count, 'actual_vector_count': store.count(),
              'embedding_identity': embedder.identity, 'ingest': store.stats_last_ingest,
              'remote_embedding_calls': 0, 'llm_calls': 0, 'persona_distillation_completed': False,
              'original_unchanged': hashlib.sha256(source.read_bytes()).hexdigest() == audit['source_sha256']}
    atomic_json(DATA_ROOT / 'reports' / 'history-import.json', report)
    print(json.dumps({k: report[k] for k in ('status', 'indexable_messages', 'actual_vector_count',
                                             'corpus_archive_messages', 'remote_embedding_calls', 'llm_calls', 'original_unchanged')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
