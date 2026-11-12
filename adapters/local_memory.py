"""本机中文 Embedding、不可变真实历史账本与有证据/时期过滤的 Chroma 适配。"""
from collections import Counter
from contextlib import contextmanager
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import threading
from zoneinfo import ZoneInfo

from adapters.upstream import ROOT, DATA_ROOT, prepare_upstream
prepare_upstream()
from memory.vector_store import VectorStore
from memory.chunker import Chunker

LOCAL_MODEL = DATA_ROOT / 'data' / 'models' / 'bge-small-zh-v1.5'
HISTORY_DB = DATA_ROOT / 'data' / 'real_history.sqlite'
CHUNK_VERSION = 'chat-3msg-overlap1-gap30m-day-token420-v1'
QUERY_PREFIX = '为这个句子生成表示以用于检索相关文章：'
TIMEZONE = ZoneInfo('Asia/Shanghai')


def atomic_json(path, value):
    import os
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temporary.open('w', encoding='utf-8') as f:
        f.write(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
        f.flush()
        os.fsync(f.fileno())
    temporary.chmod(0o600)
    os.replace(temporary, path)


def epoch(timestamp):
    return int(datetime.strptime(timestamp[:19].replace('T', ' '), '%Y-%m-%d %H:%M:%S').replace(tzinfo=TIMEZONE).timestamp())


class LocalEmbedder:
    """所有 encode/search 在本机执行，运行阶段禁止 Hub 网络和远程代码。"""
    is_local = True
    _instance = None
    _init_lock = threading.Lock()

    def __init__(self, model_path=LOCAL_MODEL):
        self.model_path = Path(model_path).resolve()
        if not self.model_path.is_dir():
            raise FileNotFoundError('本地 Embedding 权重未下载完成')
        import torch
        torch.set_num_threads(4)
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(str(self.model_path), device='cpu', local_files_only=True, trust_remote_code=False)
        self.model.max_seq_length = 512
        self.dimension = int(self.model.get_sentence_embedding_dimension())
        self._lock = threading.RLock()
        manifest_path = DATA_ROOT / 'reports' / 'embedding-model-manifest.json'
        self.receipt = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
        self.identity = {'backend': 'sentence-transformers-local', 'model': 'BAAI/bge-small-zh-v1.5',
                         'revision': self.receipt.get('revision', 'unknown'),
                         'weights_sha256': self.receipt.get('weights_sha256', 'unknown'),
                         'dimension': self.dimension, 'normalized': True, 'metric': 'cosine',
                         'query_prefix': QUERY_PREFIX, 'document_prefix': '', 'max_sequence_tokens': 512,
                         'chunk_version': CHUNK_VERSION}

    @classmethod
    def shared(cls):
        with cls._init_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def embed(self, texts):
        if not texts:
            return []
        with self._lock:
            vectors = self.model.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
        return vectors.tolist()

    def embed_one(self, text):
        return self.embed([QUERY_PREFIX + text])[0]

    def split_document(self, text, size=420, overlap=48):
        with self._lock:
            tokens = self.model.tokenizer.encode(text, add_special_tokens=False)
            if len(tokens) <= size:
                return [text]
            return [self.model.tokenizer.decode(tokens[start:start + size], skip_special_tokens=True)
                    for start in range(0, len(tokens), size - overlap)]


class RealHistory:
    """原始历史独立 SQLite：系统/空行/隔离行也记账，任何AI产物不可写入。"""
    def __init__(self, path=HISTORY_DB):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with self.connect() as db:
            db.executescript('''
            CREATE TABLE IF NOT EXISTS sources(source_id TEXT PRIMARY KEY, sha256 TEXT NOT NULL,
              source_file TEXT NOT NULL, physical_lines INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS records(message_id TEXT PRIMARY KEY, source_id TEXT NOT NULL,
              source_line INTEGER NOT NULL, record_type TEXT NOT NULL, timestamp TEXT,
              sender TEXT, role TEXT, content TEXT, event_type TEXT,
              raw_json TEXT NOT NULL, FOREIGN KEY(source_id) REFERENCES sources(source_id));
            CREATE INDEX IF NOT EXISTS real_record_time ON records(timestamp);
            CREATE INDEX IF NOT EXISTS real_source_line ON records(source_id,source_line);
            ''')
        self.path.chmod(0o600)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA foreign_keys=ON')
        db.execute('PRAGMA journal_mode=WAL')
        try:
            with db:
                yield db
        finally:
            db.close()

    def import_ledger(self, ledger_path, audit):
        if not audit.get('all_lines_accounted') or not audit.get('original_unchanged'):
            raise ValueError('必须先通过完整原始行记账与源文件不变校验')
        with self.connect() as db:
            old = db.execute('SELECT sha256 FROM sources WHERE source_id=?', (audit['source_id'],)).fetchone()
            if old and old['sha256'] != audit['source_sha256']:
                raise ValueError('已记录的真实源文件发生变化；禁止覆盖原历史，请作为新增独立来源导入')
            db.execute('INSERT OR IGNORE INTO sources VALUES(?,?,?,?)',
                       (audit['source_id'], audit['source_sha256'], audit['input_file'], audit['physical_lines']))
            before = db.total_changes
            count = 0
            with Path(ledger_path).open(encoding='utf-8') as f:
                for line in f:
                    row = json.loads(line)
                    if row.get('origin') != 'real_chat' or row['source_sha256'] != audit['source_sha256']:
                        raise ValueError('历史账本来源/哈希不一致；拒绝AI或不明材料')
                    db.execute('INSERT OR IGNORE INTO records VALUES(?,?,?,?,?,?,?,?,?,?)',
                               (row['message_id'], row['source_id'], row['source_line'], row['record_type'],
                                row.get('timestamp'), row.get('sender'), row.get('role'), row.get('content'),
                                row.get('header_tag') or 'text', json.dumps(row, ensure_ascii=False)))
                    count += 1
            if count != audit['physical_lines']:
                raise ValueError('历史账本行数不完整，事务回滚')
            inserted = db.total_changes - before
            total = db.execute('SELECT count(*) FROM records WHERE source_id=?', (audit['source_id'],)).fetchone()[0]
            if total != count:
                raise ValueError('历史源行数量不一致，事务回滚')
        return {'ledger_rows': count, 'new_records': inserted, 'source_id': audit['source_id']}

    def evidence(self, message_ids):
        if not message_ids:
            return []
        with self.connect() as db:
            placeholders = ','.join('?' for _ in message_ids)
            rows = db.execute(f'SELECT * FROM records WHERE message_id IN ({placeholders})', message_ids).fetchall()
        mapping = {r['message_id']: r for r in rows}
        result = []
        for message_id in message_ids:
            row = mapping.get(message_id)
            if row is None or row['record_type'] != 'message' or row['role'] not in {'A', 'B'}:
                raise ValueError('向量引用不存在或不是明确归属的真实消息')
            raw = json.loads(row['raw_json'])
            result.append({'sender': row['role'], 'raw_sender': row['sender'], 'source_line': row['source_line'],
                           'quote': row['content'] or '', 'timestamp': row['timestamp'], 'time': row['timestamp'],
                           'source_file': raw['source_file'], 'source_id': row['source_id'],
                           'message_id': message_id, 'origin': 'real_chat'})
        return result

    def stats(self):
        with self.connect() as db:
            by_kind = dict(db.execute('SELECT record_type,count(*) FROM records GROUP BY record_type').fetchall())
            by_sender = dict(db.execute("SELECT sender,count(*) FROM records WHERE record_type='message' GROUP BY sender").fetchall())
            sources = db.execute('SELECT count(*) FROM sources').fetchone()[0]
        return {'record_types': by_kind, 'senders': by_sender, 'sources': sources}


class HistoricalVectorStore(VectorStore):
    """复用上游 ingest/search；真实历史禁止混入 AI 摘要，引用回查独立原始账本。"""
    def __init__(self, persist_dir, collection_name, *, slug, owner, history=None, cutoff=None):
        super().__init__(str(persist_dir), collection_name)
        self.slug, self.owner, self.cutoff = slug, owner, cutoff
        self.history = history or RealHistory()
        self.index_manifest = Path(persist_dir).parent / 'history-index.json'
        self._ingest_embedder = None
        self.relevance_threshold = 0.45
        self.stats_last_ingest = {}

    def validate_embedding_identity(self, embedder):
        if not getattr(embedder, 'is_local', False):
            raise ValueError('真实聊天索引仅支持当前明确配置的本地 Embedding')
        identity = embedder.identity
        if self.index_manifest.is_file():
            current = json.loads(self.index_manifest.read_text())
            if current.get('embedding_identity') != identity:
                raise ValueError('索引模型/维度/分块身份改变；必须新建staging索引，不能混写或先删原库')
        elif self.count():
            raise ValueError('已有非空索引缺少可验证模型身份，禁止混写')
        else:
            atomic_json(self.index_manifest, {'status': 'building', 'embedding_identity': identity,
                                             'owner': self.owner, 'slug': self.slug})
        self._ingest_embedder = embedder

    def prepare_historical_chunks(self, messages, *, source, chat_id, import_id=None):
        if self._ingest_embedder is None:
            raise ValueError('必须先验证本地索引模型身份')
        groups, group, previous = [], [], None
        for message in messages:
            if message.get('origin') != 'real_chat' or not message.get('message_id') or message.get('role') not in {'A', 'B'}:
                raise ValueError('向量材料缺少真实来源、身份或角色')
            current = epoch(message['timestamp'])
            day_changed = previous is not None and message['timestamp'][:10] != group[-1]['timestamp'][:10]
            if previous is not None and (current - previous > 1800 or current < previous or day_changed):
                groups.append(group)
                group = []
            group.append(message)
            previous = current
        if group:
            groups.append(group)
        chunks = []
        for group in groups:
            basic = Chunker().chunk_messages(group, source=source, chat_id=chat_id, chunk_turns=3, overlap_turns=1)
            for index, chunk in enumerate(basic):
                window = group[index * 2:index * 2 + 3]
                message_ids = [m['message_id'] for m in window]
                self.history.evidence(message_ids)  # 验证有原始来源，而不是只相信上传字段。
                for part, text in enumerate(self._ingest_embedder.split_document(chunk['text_for_embedding'])):
                    identity = f'{self.owner}:{self.slug}:{CHUNK_VERSION}:{part}:' + ':'.join(message_ids) + ':' + text
                    stable_id = hashlib.sha256(identity.encode()).hexdigest()
                    roles = Counter(m['role'] for m in window)
                    dominant = 'B' if roles['B'] > roles['A'] else 'A'
                    metadata = {**chunk['metadata'], 'origin': 'real_chat', 'source': 'chat',
                                'owner': self.owner, 'slug': self.slug, 'sender': dominant,
                                'message_ids': json.dumps(message_ids),
                                'source_lines': ','.join(str(m['source_line']) for m in window),
                                'source_line': window[0]['source_line'], 'source_id': window[0]['source_id'],
                                'start_epoch': epoch(window[0]['timestamp']), 'end_epoch': epoch(window[-1]['timestamp']),
                                'chunk_version': CHUNK_VERSION, 'part': part}
                    chunks.append({**chunk, 'id': stable_id, 'text_for_embedding': text, 'metadata': metadata})
        return chunks

    def ingest(self, chunks, embedder, batch_size=64, on_batch=None):
        self.validate_embedding_identity(embedder)
        existing = set(self.collection.get(include=[])['ids'])
        pending = [c for c in chunks if c['id'] not in existing]
        self.stats_last_ingest = {'requested_chunks': len(chunks), 'existing_chunks': len(existing),
                                  'new_chunks': len(pending), 'skipped_existing_chunks': len(chunks) - len(pending)}
        if pending:
            super().ingest(pending, embedder, batch_size=batch_size, on_batch=on_batch)
        self.stats_last_ingest['collection_count'] = self.count()
        current = json.loads(self.index_manifest.read_text())
        current.update(status='ready', vector_count=self.count(), last_import=self.stats_last_ingest)
        atomic_json(self.index_manifest, current)

    def search(self, query, embedder, top_k=10, where=None, where_document=None):
        self.validate_embedding_identity(embedder)
        filters = [{'origin': {'$eq': 'real_chat'}}]
        if where:
            filters.append(where)
        if self.cutoff is not None:
            filters.append({'end_epoch': {'$lte': int(self.cutoff)}})
        combined = filters[0] if len(filters) == 1 else {'$and': filters}
        results = super().search(query, embedder, top_k=min(max(top_k * 3, 10), 60),
                                 where=combined, where_document=where_document)
        cleaned, seen = [], set()
        for result in results:
            score = result.get('score', 0)
            if not math.isfinite(score) or score < self.relevance_threshold:
                continue
            metadata = result['metadata']
            try:
                ids = json.loads(metadata['message_ids'])
                evidence = self.history.evidence(ids)
            except (ValueError, KeyError, json.JSONDecodeError):
                continue
            if self.cutoff is not None and any(epoch(e['timestamp']) > self.cutoff for e in evidence):
                continue
            signature = tuple(ids)
            if signature in seen:
                continue
            seen.add(signature)
            result['evidence'] = evidence
            result['display_text'] = '\n'.join(f"[{e['timestamp']}] {e['raw_sender']}: {e['quote']}" for e in evidence)
            cleaned.append(result)
            if len(cleaned) >= top_k:
                break
        return cleaned

    def search_target_only(self, query, embedder, top_k=10):
        # 双方原话都是事实证据；目标B的表达习惯由证据speaker单独识别。
        return self.search(query, embedder, top_k=top_k)

    def add_session_summary(self, text, slug, embedder):
        raise ValueError('禁止将AI会话摘要写入真实历史 collection；请使用独立 SessionMemory')
