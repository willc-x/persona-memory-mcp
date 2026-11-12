"""合成测试：JSON 角色溯源、本地 Embedding 预检和配置开关。

测试不访问网络、真实聊天、Keychain 或用户配置，仅使用 pytest 临时目录。
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from memory import ingest
from memory.chunker import Chunker
from parsers import chat_parser


PROVENANCE = {
    "message_id": "synthetic-message-a",
    "source_id": "synthetic-source",
    "source_file": "/tmp/synthetic-chat.json",
    "source_sha256": "a" * 64,
    "source_line": 1,
    "origin": "real_chat",
    "role": "A",
    "raw_sender": "合成用户",
    "header_tag": "text",
    "event_type": "message",
}


def _json_messages() -> list[dict]:
    return [
        {
            "sender": "合成用户",
            "content": "合成问题",
            "timestamp": "2030-01-01 08:00:00",
            "is_target": False,
            **PROVENANCE,
        },
        {
            "sender": "合成目标",
            "content": "合成回答",
            "timestamp": "2030-01-01 08:00:01",
            "is_target": True,
            **{**PROVENANCE, "message_id": "synthetic-message-b",
               "source_line": 2, "role": "B", "raw_sender": "合成目标"},
        },
    ]


def test_json_import_keeps_exact_role_and_provenance(tmp_path: Path):
    source = tmp_path / "synthetic-chat.json"
    source.write_text(json.dumps({"messages": _json_messages()}, ensure_ascii=False),
                      encoding="utf-8")

    messages = chat_parser.parse(str(source), target_name="合成目标")

    assert [row["role"] for row in messages] == ["A", "B"]
    assert [row["is_target"] for row in messages] == [False, True]
    for index, row in enumerate(messages, 1):
        for key in PROVENANCE:
            if key in {"message_id", "source_line", "role", "raw_sender"}:
                continue
            assert row[key] == PROVENANCE[key]
        assert row["source_line"] == index
        assert row["origin"] == "real_chat"


@pytest.mark.parametrize("change", [
    {"role": "A", "is_target": True},
    {"role": "B", "is_target": False},
    {"role": "C", "is_target": False},
    {"origin": "ai_summary"},
])
def test_json_role_and_origin_conflicts_fail_closed(tmp_path: Path, change):
    bad = {**_json_messages()[0], **change}
    source = tmp_path / "synthetic-chat.json"
    source.write_text(json.dumps([bad], ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError, match="角色与目标人物映射不一致"):
        chat_parser.parse(str(source), target_name="合成目标")


class _LocalEmbedder:
    is_local = True


def _install_import_stubs(monkeypatch):
    calls = []
    monkeypatch.setattr(ingest, "_archive_messages",
                        lambda *args, **kwargs: calls.append("archive"))

    def prepare(messages, **kwargs):
        calls.append(("prepare", kwargs.get("import_id")))
        return [{"content": messages[0]["content"]}]

    monkeypatch.setattr(ingest, "_mask_messages", lambda messages: messages)
    monkeypatch.setattr(ingest, "_validate_import_messages", lambda *args: None)
    monkeypatch.setattr(Chunker, "chunk_messages",
                        lambda *args, **kwargs: pytest.fail("本地历史库不应回退到普通分块"))
    return calls, prepare


def test_local_identity_is_checked_before_archive_on_success(
        tmp_path: Path, monkeypatch):
    calls, prepare = _install_import_stubs(monkeypatch)
    messages = _json_messages()
    monkeypatch.setattr(chat_parser, "parse", lambda *args, **kwargs: messages)

    class Vector:
        def validate_embedding_identity(self, embedder):
            calls.append("identity")

        def prepare_historical_chunks(self, *args, **kwargs):
            calls.append("chunks")
            return prepare(*args, **kwargs)

        def ingest(self, chunks, embedder):
            calls.append("vectors")

        def count(self):
            return 1

    result, count = ingest.ingest_chat_file(
        str(tmp_path / "unused.json"), "synthetic", "合成目标",
        Vector(), _LocalEmbedder(), owner=1, import_id="synthetic-import",
    )

    assert count == 1 and len(result) == 2
    assert calls == [
        "identity",
        "archive",
        "chunks",
        ("prepare", "synthetic-import"),
        "identity",
        "vectors",
    ]


def test_local_identity_failure_never_archives_or_ingests(
        tmp_path: Path, monkeypatch):
    calls, _ = _install_import_stubs(monkeypatch)
    monkeypatch.setattr(chat_parser, "parse", lambda *args, **kwargs: _json_messages())

    class Vector:
        def validate_embedding_identity(self, embedder):
            calls.append("identity")
            raise RuntimeError("synthetic identity mismatch")

        def prepare_historical_chunks(self, *args, **kwargs):
            pytest.fail("身份校验失败后不应切片")

        def ingest(self, chunks, embedder):
            pytest.fail("身份校验失败后不应写入向量")

    with pytest.raises(RuntimeError, match="identity mismatch"):
        ingest.ingest_chat_file(
            str(tmp_path / "unused.json"), "synthetic", "合成目标",
            Vector(), _LocalEmbedder(), owner=1,
        )
    assert calls == ["identity"]


def test_config_can_disable_dotenv_and_keychain_without_secrets():
    script = r"""
import dotenv
import core.keychain
import config

def forbidden(*args, **kwargs):
    raise AssertionError("外部配置入口被调用")

dotenv.load_dotenv = forbidden
core.keychain.get_key = forbidden
assert config.PROJECT_DIR.is_dir()
print("guarded")
"""
    env = {
        key: value for key, value in os.environ.items()
        if key not in {"LLM_API_KEY", "DEEPSEEK_API_KEY", "EMBEDDING_API_KEY"}
    }
    env.update({
        "EX_MEMORY_LOAD_DOTENV": "false",
        "EX_MEMORY_USE_KEYCHAIN": "false",
        "PYTHONDONTWRITEBYTECODE": "1",
    })
    repository = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repository,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "guarded"