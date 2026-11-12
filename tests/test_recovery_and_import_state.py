"""Synthetic tests for offline checkpoint recovery and history state changes.

No real chat, account, embedding model, provider, or network is used here.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from distillation.planning import assert_plan_source, file_digest, make_plan
from distillation.runner import NODE_SCHEMA
from scripts.import_history import (_finalize_import_state, _prepare_import_state)
from scripts.recover_failed_node import recover
from scripts.drive_full_run import (drive, failure_from_log,
                                    fresh_completed_event, latest_diagnostic)


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def _messages() -> list[dict]:
    return [
        {"source_line": 1, "ts": "2026-01-01T10:00:00", "sender": "A",
         "text": "我喜欢喝茶"},
        {"source_line": 2, "ts": "2026-01-01T10:01:00", "sender": "A",
         "text": "今天有点累"},
    ]


def _claim(line: int = 1, sender: str = "A"):
    quote = {"1": "我喜欢喝茶", "2": "今天有点累"}[str(line)]
    return {
        "id": "synthetic_claim", "category": "style",
        "statement": "合成用户用简短句子表达日常状态。",
        "evidence": [{"source_line": line, "sender": sender, "quote": quote}],
        "time_start": "2026-01-01", "time_end": "2026-01-01", "confidence": 0.8,
    }


@pytest.fixture()
def recovery_case(tmp_path: Path):
    data_root = tmp_path / "private-data"
    data_root.mkdir()
    source = data_root / "prepared" / "synthetic" / "messages.json"
    _write(source, {"messages": _messages()})
    plan = make_plan(_messages(), source_path=str(source), source_sha256=file_digest(source),
                     max_messages=10, context_tokens=16_000, reserved_tokens=4_000,
                     holdout_rate=0.0, target_name="合成目标", slug="synthetic-target")
    workdir = data_root / "results"
    workdir.mkdir()
    _write(workdir / "plan.json", plan)
    segment = plan["segments"][0]
    segment_id, kind = segment["id"], "persona"
    checkpoint = workdir / "segments" / f"{segment_id}.{kind}.json"
    _write(checkpoint, {
        "schema_version": NODE_SCHEMA,
        "identity": {"plan_hash": plan["plan_hash"], "segment_id": segment_id, "kind": kind},
        "model": "synthetic-model", "status": "failed",
        "attempts": [
            {"attempt": 1, "outcome": "invalid_response"},
            {"attempt": 2, "outcome": "invalid_response"},
        ],
    })
    response_text = json.dumps({"claims": [_claim()]}, ensure_ascii=False)
    digest = hashlib.sha256(response_text.encode()).hexdigest()[:16]
    diagnostic = workdir / "diagnostics" / f"segments.{segment_id}.{kind}.attempt_2.{digest}.json"
    _write(diagnostic, {"response_text": response_text})
    _write(data_root / "config" / "privacy-consent.json", {
        "authorizations": [{
            "id": "synthetic-full-grant", "plan_hash": plan["plan_hash"],
            "purposes": ["distill_full"],
        }]
    })
    _write(data_root / "runtime" / "api-usage.json", {
        "events": [{
            "id": 7, "grant_id": "synthetic-full-grant", "status": "completed",
            "prompt_tokens": 11, "completion_tokens": 7,
        }]
    })
    return SimpleNamespace(data_root=data_root, workdir=workdir, plan=plan,
                           segment=segment, kind=kind, checkpoint=checkpoint,
                           diagnostic=diagnostic, response_text=response_text)


def test_recover_uses_plan_source_and_never_calls_provider(recovery_case):
    case = recovery_case
    result = recover(case.workdir, case.segment["id"], case.kind, 2, 7,
                     "synthetic-full-grant", data_root=case.data_root)
    assert result["network_calls"] == 0
    assert result["provider_recall"] is False
    saved = json.loads(case.checkpoint.read_text(encoding="utf-8"))
    assert saved["status"] == "completed"
    assert saved["provider_recall"] is False
    assert saved["attempts"] == [
        {"attempt": 1, "outcome": "invalid_response"},
        {"attempt": 2, "outcome": "completed",
         "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}},
    ]
    assert len(saved["attempts"]) == 2  # the per-node provider attempt cap
    assert saved["migrated_from_diagnostics"] == case.diagnostic.name
    assert saved["response"]["claims"][0]["id"].startswith("claim_")


def test_recover_never_permits_a_third_attempt(recovery_case):
    case = recovery_case
    with pytest.raises(ValueError, match="1 或 2"):
        recover(case.workdir, case.segment["id"], case.kind, 3, 7,
                "synthetic-full-grant", data_root=case.data_root)
    saved = json.loads(case.checkpoint.read_text(encoding="utf-8"))
    saved["attempts"].append({"attempt": 3, "outcome": "invalid_response"})
    _write(case.checkpoint, saved)
    with pytest.raises(ValueError, match="两次上限"):
        recover(case.workdir, case.segment["id"], case.kind, 2, 7,
                "synthetic-full-grant", data_root=case.data_root)
    assert json.loads(case.checkpoint.read_text(encoding="utf-8"))["status"] == "failed"


def test_salvage_keeps_only_individually_verifiable_claims(recovery_case):
    case = recovery_case
    bad = {**_claim(), "id": "synthetic_bad",
           "evidence": [{"source_line": 2, "sender": "A", "quote": "不存在的合成原话"}]}
    response_text = json.dumps({"claims": [_claim(), bad]}, ensure_ascii=False)
    digest = hashlib.sha256(response_text.encode()).hexdigest()[:16]
    case.diagnostic.unlink()
    case.diagnostic = case.diagnostic.parent / (
        f"segments.{case.segment['id']}.{case.kind}.attempt_2.{digest}.json")
    _write(case.diagnostic, {"response_text": response_text})
    result = recover(case.workdir, case.segment["id"], case.kind, 2, 7,
                     "synthetic-full-grant", salvage=True, data_root=case.data_root)
    assert result["validation_mode"] == "per_claim_salvage"
    assert result["rejected_claims"] == 1
    saved = json.loads(case.checkpoint.read_text(encoding="utf-8"))
    assert saved["status"] == "completed"
    assert saved["rejected_claim_count"] == 1
    assert len(saved["response"]["claims"]) == 1
    assert saved["response"]["claims"][0]["id"].startswith("claim_")


def test_recover_rejects_source_bytes_that_differ_from_plan(recovery_case):
    case = recovery_case
    (case.data_root / "prepared" / "synthetic" / "messages.json").write_text(
        json.dumps({"messages": _messages() + [{
            "source_line": 3, "ts": "2026-01-01T10:02:00", "sender": "A",
            "text": "合成变更",
        }]}, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="计划 source"):
        recover(case.workdir, case.segment["id"], case.kind, 2, 7,
                "synthetic-full-grant", data_root=case.data_root)
    assert json.loads(case.checkpoint.read_text(encoding="utf-8"))["status"] == "failed"


def test_recover_rejects_workdir_and_source_outside_data_root(recovery_case):
    case = recovery_case
    outside = case.data_root.parent / "outside"
    outside.mkdir()
    with pytest.raises(ValueError, match="DATA_ROOT"):
        recover(outside, case.segment["id"], case.kind, 2, 7,
                "synthetic-full-grant", data_root=case.data_root)
    external_source = outside / "messages.json"
    _write(external_source, {"messages": _messages()})
    modified = json.loads((case.workdir / "plan.json").read_text(encoding="utf-8"))
    modified["source"]["path"] = str(external_source)
    # Recompute the plan hash only to reach the source-boundary check; the file
    # remains outside DATA_ROOT and must still be rejected.
    from distillation.planning import digest
    unsigned = {k: v for k, v in modified.items() if k != "plan_hash"}
    modified["plan_hash"] = digest(unsigned)
    _write(case.workdir / "plan.json", modified)
    consent = json.loads((case.data_root / "config" / "privacy-consent.json").read_text())
    consent["authorizations"][0]["plan_hash"] = modified["plan_hash"]
    _write(case.data_root / "config" / "privacy-consent.json", consent)
    with pytest.raises(ValueError, match="DATA_ROOT"):
        recover(case.workdir, case.segment["id"], case.kind, 2, 7,
                "synthetic-full-grant", data_root=case.data_root)


def test_drive_recovers_only_with_a_fresh_completed_event(tmp_path, monkeypatch):
    data_root = tmp_path / "root"
    workdir = data_root / "work"
    workdir.mkdir(parents=True)
    _write(workdir / "plan.json", {"plan_hash": "synthetic-plan"})
    _write(data_root / "runtime" / "api-usage.json", {
        "events": [{"id": 1, "grant_id": "grant", "status": "completed"}]
    })
    diagnostic = (workdir / "diagnostics" /
                  "segments.segment_000000.persona.attempt_1.0123456789abcdef.json")
    _write(diagnostic, {"response_text": "{}"})
    monkeypatch.setenv("CHAT_PYTHON", "/bin/sh")
    calls = {"runner": 0, "recover": []}

    def runner(command, *, stdout, stderr=None, **kwargs):
        calls["runner"] += 1
        if calls["runner"] == 1:
            _write(data_root / "runtime" / "api-usage.json", {
                "events": [
                    {"id": 1, "grant_id": "grant", "status": "completed"},
                    {"id": 2, "grant_id": "grant", "status": "completed"},
                ]
            })
            stdout.write(b"node failed after two attempts (invalid_response): "
                         b"segment_000000.persona.json\n")
            stdout.flush()
            return SimpleNamespace(returncode=1)
        return SimpleNamespace(returncode=0)

    def fake_recover(workdir_arg, segment_id, kind, attempt, event, grant, **kwargs):
        calls["recover"].append((segment_id, kind, attempt, event, grant))
        return {"node": "checkpoint", "claims": 1}

    result = drive(workdir, "grant", max_recoveries=2, data_root=data_root,
                   runner=runner, recover_fn=fake_recover)
    assert result["status"] == "complete"
    assert calls["runner"] == 2
    assert calls["recover"] == [("segment_000000", "persona", 1, 2, "grant")]


def test_drive_does_not_recover_provider_error_or_without_new_usage(tmp_path, monkeypatch):
    data_root = tmp_path / "root"
    workdir = data_root / "work"
    workdir.mkdir(parents=True)
    _write(workdir / "plan.json", {"plan_hash": "synthetic-plan"})
    _write(data_root / "runtime" / "api-usage.json", {
        "events": [{"id": 1, "grant_id": "grant", "status": "completed"}]
    })
    monkeypatch.setenv("CHAT_PYTHON", "/bin/sh")

    def provider_error(command, *, stdout, **kwargs):
        stdout.write(b"node failed after two attempts (provider_error): "
                     b"segment_000000.persona.json\n")
        stdout.flush()
        return SimpleNamespace(returncode=1)

    called = []
    result = drive(workdir, "grant", data_root=data_root, runner=provider_error,
                   recover_fn=lambda *args, **kwargs: called.append(args))
    assert result == {"status": "stopped", "reason": "provider_error"}
    assert not called


def test_failure_parser_uses_last_reported_node_and_diagnostic_attempt():
    text = ("node failed after two attempts (invalid_response): segment_000000.persona.json\n"
            "node failed after two attempts (invalid_response): segment_000001.memories.json\n")
    assert failure_from_log(text) == ("segment_000001", "memories", "invalid_response")
    assert fresh_completed_event({"events": [
        {"id": 1, "grant_id": "g", "status": "completed"},
        {"id": 2, "grant_id": "g", "status": "failed"},
        {"id": 3, "grant_id": "other", "status": "completed"},
    ]}, {1}, "g") is None


def _audit(source_sha="a" * 64) -> dict:
    return {"source_sha256": source_sha, "target": "合成目标", "role_mapping": {"A": "我"}}


def test_same_source_import_keeps_existing_persona_state(tmp_path):
    data_root = tmp_path / "root"
    persona = data_root / "mirror" / "persona"
    installation = {
        "persona_directory": str(persona), "real_source_sha256": "a" * 64,
        "persona_integrated": True, "stale_after_history_append": False,
    }
    meta = {"pipeline_state": "index_ready_persona_integrated",
            "personality_distilled": True, "stale_after_history_append": False}
    new_meta, new_installation = _prepare_import_state(
        meta, installation, _audit(), 1, "persona", mirror=data_root / "mirror", data_root=data_root)
    assert new_installation["persona_directory"] == str(persona)
    assert new_meta["pipeline_state"] == "index_ready_persona_integrated"
    assert new_meta["import_source_changed"] is False
    assert new_meta["stale_after_history_append"] is False
    final_meta, final_installation, status = _finalize_import_state(
        new_meta, new_installation, message_count=2, vector_count=1)
    assert status == "history_ready_persona_existing"
    assert final_installation["stale_after_history_append"] is False
    assert final_meta["pipeline_state"] == "index_ready_persona_integrated"


def test_new_source_preserves_persona_directory_but_marks_stale(tmp_path):
    data_root = tmp_path / "root"
    persona = data_root / "mirror" / "persona"
    installation = {
        "persona_directory": str(persona), "real_source_sha256": "a" * 64,
        "persona_integrated": True, "stale_after_history_append": False,
    }
    meta = {"pipeline_state": "index_ready_persona_integrated",
            "personality_distilled": True, "stale_after_history_append": False}
    new_meta, new_installation = _prepare_import_state(
        meta, installation, _audit("b" * 64), 1, "persona",
        mirror=data_root / "mirror", data_root=data_root)
    assert new_installation["persona_directory"] == str(persona)
    assert new_meta["pipeline_state"] == "index_building"
    assert new_meta["import_source_changed"] is True
    assert new_meta["stale_after_history_append"] is True
    assert new_meta["previous_source_sha256"] == "a" * 64
    final_meta, final_installation, status = _finalize_import_state(
        new_meta, new_installation, message_count=3, vector_count=2)
    assert status == "history_ready_persona_stale"
    assert final_meta["pipeline_state"] == "index_ready_persona_stale"
    assert final_installation["stale_after_history_append"] is True
    # Presence of a physical old persona is not proof that it covers new source.
    assert final_installation.get("persona_integrated") is True


def test_unsafe_previous_persona_path_falls_back_inside_data_root(tmp_path):
    data_root = tmp_path / "root"
    outside = tmp_path / "elsewhere"
    meta = {}
    installation = {"persona_directory": str(outside)}
    new_meta, new_installation = _prepare_import_state(
        meta, installation, _audit(), 1, "persona", mirror=data_root / "mirror",
        data_root=data_root)
    assert new_installation["persona_directory"] == str(data_root / "artifacts" / "persona")
    assert new_meta["persona_directory"] == str(data_root / "artifacts" / "persona")