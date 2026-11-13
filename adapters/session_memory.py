"""持久 AI 交互记忆：独立 SQLite/Chroma，永远不回填真实聊天事实。"""
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import uuid

import chromadb


def validate_conversation_id(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,80}', value):
        raise ValueError('会话ID无效')
    return value


def validate_relationship_mode(value):
    if value not in ('evidence', 'fiction'):
        raise ValueError('关系模式只能是 evidence 或 fiction')
    return value


class SessionMemory:
    def __init__(self, mirror_dir, *, owner, slug, embedder):
        if not getattr(embedder, 'is_local', False):
            raise ValueError('AI记忆只使用本地Embedding')
        self.base = Path(mirror_dir) / 'ai_sessions'
        self.base.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.base.chmod(0o700)
        self.path = self.base / 'sessions.sqlite'
        self.owner, self.slug, self.embedder = owner, slug, embedder
        self.client = chromadb.PersistentClient(path=str(self.base / 'chroma_db'))
        name = 'ai_' + hashlib.sha256(f'{owner}:{slug}'.encode()).hexdigest()[:32]
        self.collection = self.client.get_or_create_collection(name, metadata={'hnsw:space': 'cosine', 'origin': 'ai_simulation'})
        with self.connect() as db:
            db.executescript('''CREATE TABLE IF NOT EXISTS conversations(
                 id TEXT PRIMARY KEY, era TEXT NOT NULL, relationship_mode TEXT NOT NULL DEFAULT 'evidence',
                 created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
                 CREATE TABLE IF NOT EXISTS turns(id TEXT PRIMARY KEY, conversation_id TEXT NOT NULL,
                 era TEXT NOT NULL, relationship_mode TEXT NOT NULL DEFAULT 'evidence', user_text TEXT NOT NULL,
                 ai_text TEXT NOT NULL, sources_json TEXT NOT NULL,
                 created_at TEXT NOT NULL, origin TEXT NOT NULL DEFAULT 'ai_simulation',
                 verified_real_history INTEGER NOT NULL DEFAULT 0, indexed INTEGER NOT NULL DEFAULT 0,
                 FOREIGN KEY(conversation_id) REFERENCES conversations(id));
                 CREATE INDEX IF NOT EXISTS ai_turn_conversation ON turns(conversation_id,created_at);''')
            conversation_columns = {row[1] for row in db.execute('PRAGMA table_info(conversations)')}
            turn_columns = {row[1] for row in db.execute('PRAGMA table_info(turns)')}
            if 'relationship_mode' not in conversation_columns:
                db.execute("ALTER TABLE conversations ADD COLUMN relationship_mode TEXT NOT NULL DEFAULT 'evidence'")
            if 'relationship_mode' not in turn_columns:
                db.execute("ALTER TABLE turns ADD COLUMN relationship_mode TEXT NOT NULL DEFAULT 'evidence'")
            db.execute('CREATE INDEX IF NOT EXISTS ai_turn_mode ON turns(conversation_id,relationship_mode)')
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

    def ensure_conversation(self, conversation_id=None, era=None, relationship_mode='evidence'):
        conversation_id = validate_conversation_id(conversation_id or str(uuid.uuid4()))
        relationship_mode = validate_relationship_mode(relationship_mode)
        era = era or 'latest'
        now = datetime.now(timezone.utc).isoformat()
        with self.connect() as db:
            previous = db.execute('SELECT era,relationship_mode FROM conversations WHERE id=?', (conversation_id,)).fetchone()
            if previous and previous['era'] != era:
                raise ValueError('同一会话不能混入不同历史时期，请新建会话')
            if previous and previous['relationship_mode'] != relationship_mode:
                raise ValueError('同一会话不能混用史实与虚构关系模式，请新建会话')
            db.execute('INSERT OR IGNORE INTO conversations(id,era,relationship_mode,created_at,updated_at) VALUES(?,?,?,?,?)',
                       (conversation_id, era, relationship_mode, now, now))
        return conversation_id

    def history(self, conversation_id, era=None, max_turns=20, max_bytes=6500, relationship_mode='evidence'):
        conversation_id = validate_conversation_id(conversation_id)
        relationship_mode = validate_relationship_mode(relationship_mode)
        with self.connect() as db:
            row = db.execute('SELECT era,relationship_mode FROM conversations WHERE id=?', (conversation_id,)).fetchone()
            if row is None:
                return []
            if row['era'] != (era or 'latest'):
                raise ValueError('会话时期不一致，请新建会话')
            if row['relationship_mode'] != relationship_mode:
                raise ValueError('会话关系模式不一致，请新建会话')
            rows = db.execute('SELECT user_text,ai_text FROM turns WHERE conversation_id=? ORDER BY rowid DESC LIMIT ?',
                              (conversation_id, max_turns)).fetchall()
        selected, used = [], 0
        for row in rows:
            size = len(row['user_text'].encode()) + len(row['ai_text'].encode())
            if used + size > max_bytes:
                break
            selected.append(row)
            used += size
        history = []
        for row in reversed(selected):
            history.extend([{'role': 'user', 'content': row['user_text']}, {'role': 'assistant', 'content': row['ai_text']}])
        return history

    def duplicate_turn(self, conversation_id, user_text, *, relationship_mode='evidence', within_seconds=30):
        """返回同会话中最近的同文本轮，避免客户端重试重复调用模型/重复记账。"""
        conversation_id = validate_conversation_id(conversation_id)
        relationship_mode = validate_relationship_mode(relationship_mode)
        if not isinstance(user_text, str) or not user_text or within_seconds < 0:
            return None
        with self.connect() as db:
            row = db.execute(
                'SELECT id,user_text,ai_text,sources_json,relationship_mode,created_at,indexed '
                'FROM turns WHERE conversation_id=? ORDER BY rowid DESC LIMIT 1',
                (conversation_id,),
            ).fetchone()
        if row is None or row['user_text'] != user_text or row['relationship_mode'] != relationship_mode:
            return None
        try:
            created = datetime.fromisoformat(row['created_at'])
        except (TypeError, ValueError):
            return None
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        age = (datetime.now(timezone.utc) - created).total_seconds()
        if age < 0 or age > within_seconds:
            return None
        try:
            sources = json.loads(row['sources_json'])
        except (TypeError, ValueError):
            sources = []
        return {'turn_id': row['id'], 'reply': row['ai_text'], 'sources': sources,
                'indexed': bool(row['indexed'])}

    def append_turn(self, conversation_id, user_text, reply, sources, era=None, *, ai_generated=False, relationship_mode='evidence'):
        conversation_id = self.ensure_conversation(conversation_id, era, relationship_mode)
        if ai_generated is not True:
            raise ValueError('拒绝存储未经可信调用链标记的模拟回复')
        if not isinstance(reply, str) or not reply.strip():
            raise ValueError('拒绝存储空回复')
        turn_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        with self.connect() as db:
            db.execute('INSERT INTO turns(id,conversation_id,era,relationship_mode,user_text,ai_text,sources_json,created_at) '
                       'VALUES(?,?,?,?,?,?,?,?)',
                       (turn_id, conversation_id, era or 'latest', relationship_mode, user_text, reply,
                        json.dumps(sources, ensure_ascii=False), now))
            db.execute('UPDATE conversations SET updated_at=? WHERE id=?', (now, conversation_id))
        # 先保存不可丢失的完整会话；向量失败不丢失正文，pending 可重建。
        try:
            self.index_pending(limit=8)
            with self.connect() as db:
                indexed = bool(db.execute('SELECT indexed FROM turns WHERE id=?', (turn_id,)).fetchone()[0])
        except Exception:
            indexed = False
        return {'turn_id': turn_id, 'conversation_id': conversation_id, 'indexed': indexed,
                'origin': 'ai_simulation', 'verified_real_history': False}

    def index_pending(self, limit=100):
        with self.connect() as db:
            rows = db.execute('SELECT * FROM turns WHERE indexed=0 ORDER BY rowid LIMIT ?', (limit,)).fetchall()
        for row in rows:
            mode = row['relationship_mode'] if 'relationship_mode' in row.keys() else 'evidence'
            label = '虚构亲密对话，不是真实聊天历史' if mode == 'fiction' else '独立对话记忆，不是真实聊天事实'
            document = (f'[{label}]\n用户本次声明：' + row['user_text'] + '\n角色回复：' + row['ai_text'])
            parts = self.embedder.split_document(document)
            ids = [hashlib.sha256(f"ai:{self.owner}:{self.slug}:{row['id']}:{i}".encode()).hexdigest() for i in range(len(parts))]
            metadata = [{'origin': 'ai_simulation', 'verified_real_history': False,
                         'turn_id': row['id'], 'conversation_id': row['conversation_id'], 'era': row['era'],
                         'relationship_mode': mode, 'created_at': row['created_at'], 'owner': self.owner} for _ in parts]
            self.collection.upsert(ids=ids, embeddings=self.embedder.embed(parts), documents=parts, metadatas=metadata)
            with self.connect() as db:
                db.execute('UPDATE turns SET indexed=1 WHERE id=?', (row['id'],))
        return len(rows)

    def search(self, query, *, era=None, top_k=4, exclude_recent_ids=None, relationship_mode='evidence'):
        relationship_mode = validate_relationship_mode(relationship_mode)
        if not self.collection.count():
            return []
        result = self.collection.query(query_embeddings=[self.embedder.embed_one(query)],
                  n_results=min(top_k * 8, self.collection.count()),
                  where={'$and': [{'origin': 'ai_simulation'}, {'era': era or 'latest'}]})
        memories, seen = [], set(exclude_recent_ids or [])
        for index, metadata in enumerate((result.get('metadatas') or [[]])[0]):
            distance = result['distances'][0][index]
            if metadata.get('relationship_mode', 'evidence') != relationship_mode:
                continue
            if distance > 0.48 or metadata['turn_id'] in seen:
                continue
            seen.add(metadata['turn_id'])
            memories.append({'content': result['documents'][0][index], 'origin': 'ai_simulation',
                             'created_at': metadata['created_at'], 'turn_id': metadata['turn_id'],
                             'verified_real_history': False, 'score': 1 - distance})
            if len(memories) >= top_k:
                break
        return memories

    def conversations(self):
        with self.connect() as db:
            return [dict(row) for row in db.execute('SELECT c.*,count(t.id) AS turns FROM conversations c LEFT JOIN turns t '
                     'ON t.conversation_id=c.id GROUP BY c.id ORDER BY c.updated_at DESC LIMIT 100')]

    def stats(self):
        with self.connect() as db:
            turns = db.execute('SELECT count(*) FROM turns').fetchone()[0]
            pending = db.execute('SELECT count(*) FROM turns WHERE indexed=0').fetchone()[0]
        return {'origin': 'ai_simulation', 'turns': turns, 'vectors': self.collection.count(),
                'pending_index_turns': pending, 'verified_real_history': False}
