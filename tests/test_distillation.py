"""Synthetic-only tests: never read prepared data, credentials or call APIs.

All sources and provider responses below are manufactured in TemporaryDirectory.
Run with: python -m unittest discover -s tests -p test_distillation.py -v
"""
from __future__ import annotations

import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

from adapters.evidence import canonicalize_claim_interval
from distillation.outputs import ARTIFACT_FILES, claim_union
from distillation.pipeline import main
from distillation.planning import (PlanningError, assert_plan_source, canonical_bytes,
                                   digest, file_digest, make_plan, validate_messages)
from distillation.prompts import extraction_api_messages, hard_context_upper
from distillation.runner import (NodeExecutor, PipelineError, _LazyClient,
                                 _request_views, run_pipeline, write_local_baseline)
from distillation.validation import (CLAIM_KEYS, EVIDENCE_KEYS, LOCAL_TIME_KEYS,
                                     ClaimValidationError, response_object,
                                     validate_cached_claim, validate_claim,
                                     validate_extraction, validate_merge_response,
                                     validate_stored_extraction)


def synthetic_messages(count=6, *, separate_sessions=False, start="2024-01-01T09:00:00"):
    base = datetime.fromisoformat(start)
    return [{"source_line": i + 1, "ts": (base + timedelta(minutes=i * (60 if separate_sessions else 2))).isoformat(),
             "sender": "B" if i % 2 == 0 else "A", "text": f"合成消息{i + 1}：今天一起喝咖啡。",
             "type": "text", "kind": "text", "raw_sender": "PRIVATE_METADATA_NOT_TO_SEND",
             "source_file": "synthetic_fixture", "source_hash": "fixture_hash"}
            for i in range(count)]


def claim_for(message, *, category="fact", statement="B发送了一条合成消息", confidence=0.9, claim_id="c1"):
    return {"id": claim_id, "category": category, "statement": statement,
            "evidence": [{"source_line": message["source_line"], "sender": message["sender"], "quote": message["text"]}],
            "time_start": message["ts"], "time_end": message["ts"], "confidence": confidence}


class ReportedPreferenceTests(unittest.TestCase):
    """明确的当时偏好自述可保存；关系与动机不能借词面放行。"""
    def test_direct_self_preference_with_b_quote_is_valid(self):
        message = {"source_line": 9, "sender": "B", "ts": "2024-01-02T03:04:05", "text": "我喜欢听慢节奏的 抒情歌"}
        claim = {"id": "b_music_pref", "category": "fact", "statement": "目标人物表示自己喜欢听慢节奏的抒情歌。",
                 "evidence": [{"source_line": 9, "sender": "B", "quote": "我喜欢听慢节奏的 抒情歌"}],
                 "time_start": message["ts"], "time_end": None, "confidence": 0.9}
        self.assertTrue(validate_claim(claim, {9: message})["id"].startswith("claim_"))

    def test_inferred_affection_relationship_and_motive_stay_rejected(self):
        message = {"source_line": 9, "sender": "B", "ts": "2024-01-02T03:04:05", "text": "我隔了很久才回"}
        base = {"id": "bad", "category": "fact", "evidence": [{"source_line": 9, "sender": "B", "quote": message["text"]}],
                "time_start": message["ts"], "time_end": None, "confidence": 0.9}
        for statement in ("目标人物喜欢用户", "目标人物在乎用户", "目标人物故意慢回，动机不明"):
            with self.subTest(statement=statement), self.assertRaises(ClaimValidationError):
                validate_claim({**base, "statement": statement}, {9: message})

    def test_wrong_speaker_cannot_use_other_persons_first_person_quote(self):
        message = {"source_line": 9, "sender": "A", "ts": "2024-01-02T03:04:05", "text": "我喜欢喝咖啡"}
        claim = {"id": "wrong", "category": "fact", "statement": "目标人物表示自己喜欢喝咖啡",
                 "evidence": [{"source_line": 9, "sender": "A", "quote": "我喜欢喝咖啡"}],
                 "time_start": message["ts"], "time_end": None, "confidence": 0.9}
        with self.assertRaises(ClaimValidationError):
            validate_claim(claim, {9: message})


class EvidenceIntervalTests(unittest.TestCase):
    class Verifier:
        def __init__(self, rows):
            self.rows = rows

        def records_at(self, source_line, *, sender=None):
            return [row for row in self.rows
                    if row["source_line"] == source_line and row["sender"] == sender]

    def test_exact_quotes_become_min_max_chat_interval(self):
        rows = [
            {"source_line": 1, "sender": "A", "content": "合成早间消息",
             "timestamp": "2024-01-03T09:00:00+08:00"},
            {"source_line": 2, "sender": "B", "content": "合成晚间消息",
             "timestamp": "2024-01-04T21:00:00+08:00"},
        ]
        claim = {
            "id": "synthetic", "category": "style", "statement": "合成表达",
            "evidence": [
                {"source_line": 1, "sender": "A", "quote": "早间"},
                {"source_line": 2, "sender": "B", "quote": "晚间"},
            ],
            "time_start": "2024-01-01T00:00:00+08:00",
            "time_end": "2024-01-05T00:00:00+08:00", "confidence": .8,
        }
        result = canonicalize_claim_interval(self.Verifier(rows), claim)
        self.assertEqual(result["time_start"], rows[0]["timestamp"])
        self.assertEqual(result["time_end"], rows[1]["timestamp"])
        self.assertEqual(result["time_semantics"],
                         "chat_evidence_interval_only_not_event_occurrence_date")
        self.assertEqual(result["time_provenance"],
                         "derived_from_exactly_cited_source_timestamps")
        self.assertEqual(result["model_time_interval_before_canonicalization"],
                         [claim["time_start"], claim["time_end"]])
        self.assertEqual(claim["time_start"], "2024-01-01T00:00:00+08:00")

    def test_ambiguous_or_missing_evidence_is_unchanged(self):
        rows = [
            {"source_line": 1, "sender": "A", "content": "重复的合成内容",
             "timestamp": "2024-01-03T09:00:00"},
            {"source_line": 1, "sender": "A", "content": "重复的合成内容",
             "timestamp": "2024-01-03T10:00:00"},
        ]
        claim = {
            "id": "synthetic", "category": "style", "statement": "合成表达",
            "evidence": [{"source_line": 1, "sender": "A", "quote": "合成内容"}],
            "time_start": "2024-01-01", "time_end": "2024-01-05", "confidence": .8,
        }
        self.assertIs(canonicalize_claim_interval(self.Verifier(rows), claim), claim)
        for evidence in ([], [{"source_line": 1, "sender": "C", "quote": "合成内容"}],
                         [{"source_line": 99, "sender": "A", "quote": "合成内容"}]):
            bad = {**claim, "evidence": evidence}
            self.assertIs(canonicalize_claim_interval(self.Verifier(rows), bad), bad)


class SyntheticProvider:
    """Pure in-memory fake with the exact .model/.chat.completions.create facade."""
    def __init__(self, *, fail=None, invalid_first=False, invalid_merge=False, large=False, excluded=()):
        self.model = "synthetic-model"
        self.calls = []
        self.seen = {}
        self.fail = fail
        self.invalid_first = invalid_first
        self.invalid_merge = invalid_merge
        self.large = large
        self.excluded = set(excluded)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.calls.append(copy.deepcopy(kwargs))
        if set(kwargs) != {"model", "messages", "response_format", "max_tokens", "temperature"}:
            raise AssertionError("unexpected API keyword/extra_body")
        if kwargs["model"] != self.model or kwargs["max_tokens"] != 4000:
            raise AssertionError("caller model/output allowance mismatch")
        if self.fail:
            raise RuntimeError(self.fail)
        payload = json.loads(kwargs["messages"][1]["content"])
        is_merge = "rows" not in payload
        if is_merge:
            if self.invalid_merge:
                value = {"summary": "未经允许的摘要", "claim_ids": ["invented_id"]}
            else:
                ids = [c["id"] for c in payload["claims"]]
                value = {"summary": "合成摘要，原claim由调用方保真保存。", "claim_ids": ids[:1]}
        else:
            rows = [dict(zip(payload["columns"], row)) for row in payload["rows"]]
            if any(m["source_line"] in self.excluded for m in rows):
                raise AssertionError("holdout leaked")
            if any(m.get("session_id") is None for m in rows):
                raise AssertionError("missing independent session boundary")
            if "PRIVATE_METADATA_NOT_TO_SEND" in kwargs["messages"][1]["content"]:
                raise AssertionError("raw sender metadata leaked")
            message = next((m for m in rows if m["sender"] == "B"), rows[0])
            kind = "persona" if "人格与表达抽取器" in kwargs["messages"][0]["content"] else "memories"
            category = "style" if kind == "persona" else "fact"
            key = (payload["segment_id"], kind)
            self.seen[key] = self.seen.get(key, 0) + 1
            if self.large:
                claims = [claim_for(message, category=category,
                                    statement=f"B的合成记录{kind}-{i}：" + "规" * 500,
                                    claim_id=f"raw_{i}") for i in range(20)]
            else:
                claims = [claim_for(message, category=category,
                                    statement="B在这条消息使用简短文字" if kind == "persona" else "B发送了一条合成消息")]
            if self.invalid_first and self.seen[key] == 1:
                claims[0]["evidence"][0]["quote"] = "不存在的合成原话"
            value = {"claims": claims}
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(value, ensure_ascii=False)))],
                               usage=SimpleNamespace(prompt_tokens=100, completion_tokens=30, total_tokens=130))


class FixtureCase(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="distillation-synthetic-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def fixture(self, messages=None, **kwargs):
        messages = synthetic_messages() if messages is None else messages
        source = self.root / "synthetic_messages.json"
        source.write_text(json.dumps(messages, ensure_ascii=False), encoding="utf-8")
        kwargs.setdefault("holdout_rate", 0)
        kwargs.setdefault("target_name", "测试目标")
        kwargs.setdefault("slug", "synthetic-target")
        plan = make_plan(messages, source_path=str(source), source_sha256=file_digest(source), **kwargs)
        out = self.root / "work"
        out.mkdir(exist_ok=True)
        (out / "plan.json").write_text(json.dumps(plan, ensure_ascii=False), encoding="utf-8")
        return plan, source, out

    def read_json(self, path):
        return json.loads(Path(path).read_text(encoding="utf-8"))


class PlanningTests(FixtureCase):
    def test_strict_public_schema_and_stable_order(self):
        messages = synthetic_messages(3)
        result = validate_messages(list(reversed(messages)))
        self.assertEqual([m["source_line"] for m in result], [1, 2, 3])
        for field, value in (("source_line", True), ("sender", "合成目标"), ("ts", "bad"), ("text", None)):
            bad = copy.deepcopy(messages)
            bad[0][field] = value
            with self.subTest(field=field), self.assertRaises(PlanningError):
                validate_messages(bad)
        with self.assertRaises(PlanningError):
            validate_messages([messages[0], messages[0]])

    def test_month_stratified_whole_session_holdout(self):
        messages = []
        line = 1
        for month in (1, 2):
            base = datetime(2024, month, 1)
            for session in range(50):
                for offset in (0, 2):
                    messages.append({"source_line": line, "ts": (base + timedelta(hours=session, minutes=offset)).isoformat(),
                                     "sender": "B", "text": f"合成月份{month}会话{session}消息{offset}"})
                    line += 1
        plan, source, _ = self.fixture(messages, holdout_rate=.02)
        again = make_plan(messages, source_path=str(source), source_sha256=file_digest(source),
                          holdout_rate=.02, target_name="测试目标", slug="synthetic-target")
        self.assertEqual(plan["plan_hash"], again["plan_hash"])
        self.assertEqual([d["selected"] for d in plan["holdout"]["monthly_selection"]], [1, 1])
        held = {l for s in plan["holdout"]["segments"] for l in s["source_lines"]}
        trained = {l for s in plan["segments"] for l in s["source_lines"]}
        self.assertEqual(len(held), 4)
        self.assertFalse(held & trained)
        self.assertEqual(held | trained, set(range(1, line)))
        self.assertTrue(all(s["count"] == 2 for s in plan["holdout"]["segments"]))
        self.assertTrue(plan["holdout"]["excluded_from_llm"])

    def test_packing_retains_distinct_gap_sessions(self):
        messages = synthetic_messages(10, separate_sessions=True)
        plan, _, _ = self.fixture(messages, pack_window="day")
        self.assertEqual(plan["corpus"]["conversations"], 10)
        self.assertEqual(len(plan["segments"]), 1)
        segment = plan["segments"][0]
        self.assertEqual(len(segment["sessions"]), 10)
        views = _request_views(segment, {m["source_line"]: m for m in messages})
        api = extraction_api_messages("persona", segment["id"], views)
        payload = json.loads(api[1]["content"])
        self.assertEqual(len(payload["rows"]), 10)
        self.assertEqual(len({row[-1] for row in payload["rows"]}), 10)
        self.assertNotIn("PRIVATE_METADATA_NOT_TO_SEND", api[1]["content"])
        self.assertIn("不可当作连续谈话", api[0]["content"])

    def test_week_window_not_arbitrary_conversation_calls(self):
        messages = []
        for day in range(10):
            for session in range(2):
                message = synthetic_messages(1, start=f"2024-01-{day+1:02d}T{session*2:02d}:00:00")[0]
                message["source_line"] = len(messages) + 1
                messages.append(message)
        plan, _, _ = self.fixture(messages, pack_window="week")
        self.assertEqual(len(plan["segments"]), 2)
        self.assertEqual(sum(s["count"] for s in plan["segments"]), 20)

    def test_oversized_session_splits_without_drop(self):
        plan, _, _ = self.fixture(synthetic_messages(5), max_messages=2)
        self.assertEqual([s["count"] for s in plan["segments"]], [2, 2, 1])
        self.assertEqual([l for s in plan["segments"] for l in s["source_lines"]], [1, 2, 3, 4, 5])
        self.assertEqual({sid for s in plan["segments"] for sid in s["session_ids"]}, {"session_000000"})

    def test_single_large_message_is_never_truncated(self):
        messages = synthetic_messages(2)
        messages[1]["text"] = "长" * 20_000
        with self.assertRaisesRegex(PlanningError, "without truncation"):
            self.fixture(messages)

    def test_exact_request_upper_is_shared_with_run(self):
        messages = synthetic_messages(40)
        plan, _, _ = self.fixture(messages)
        by_line = {m["source_line"]: m for m in messages}
        for segment in plan["segments"]:
            views = _request_views(segment, by_line)
            for kind in ("persona", "memories"):
                api = extraction_api_messages(kind, segment["id"], views)
                self.assertEqual(hard_context_upper(api, len(views)), segment["token_bounds"]["per_call"][kind]["hard_upper"])
                self.assertLessEqual(hard_context_upper(api, len(views)), 32000)

    def test_price_pending_and_retry_cost_upper(self):
        plan, _, _ = self.fixture(max_messages=2)
        self.assertEqual(plan["cost"]["status"], "pending_price_confirmation")
        self.assertIsNone(plan["cost"]["estimated_cost_rmb"])
        priced, _, _ = self.fixture(max_messages=2, input_price="2", output_price="8")
        self.assertEqual(priced["calls"]["extraction"], 2 * len(priced["segments"]))
        self.assertEqual(priced["calls"]["merge"], len(priced["segments"]) - 1)
        self.assertEqual(Decimal(priced["cost"]["retry_2x_cost_upper_rmb"]), 2 * Decimal(priced["cost"]["base_cost_upper_rmb"]))
        for price in ("NaN", "Infinity", "-1", "oops"):
            with self.subTest(price=price), self.assertRaises(PlanningError):
                self.fixture(input_price=price, output_price="8")
        for rate in (float("nan"), float("inf"), -0.1, 1.1):
            with self.subTest(rate=rate), self.assertRaises(PlanningError):
                self.fixture(holdout_rate=rate)

    def test_plan_and_source_hashes_fail_closed(self):
        plan, source, _ = self.fixture()
        assert_plan_source(plan, source)
        bad = copy.deepcopy(plan)
        bad["parameters"]["max_messages"] += 1
        with self.assertRaisesRegex(PlanningError, "plan_hash mismatch"):
            assert_plan_source(bad, source)
        source.write_text("[]", encoding="utf-8")
        with self.assertRaisesRegex(PlanningError, "SHA-256 differs"):
            assert_plan_source(plan, source)


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.message = synthetic_messages(1)[0]
        self.messages = {1: self.message}

    def test_exact_schema_and_content_addressed_id(self):
        raw = claim_for(self.message)
        claim = validate_claim(raw, self.messages)
        self.assertTrue(CLAIM_KEYS.issubset(claim))
        self.assertTrue(set(claim) - CLAIM_KEYS <= LOCAL_TIME_KEYS)
        self.assertEqual(set(claim["evidence"][0]), EVIDENCE_KEYS)
        self.assertTrue(claim["id"].startswith("claim_"))
        stored = validate_stored_extraction({"claims": [claim]}, self.messages, {1})
        self.assertEqual(stored, [claim])
        with self.assertRaises(ClaimValidationError):
            validate_extraction({"claims": [claim]}, self.messages, {1})
        another = copy.deepcopy(raw)
        another["id"] = "different_model_id"
        self.assertEqual(claim, validate_claim(another, self.messages))

    def test_quote_line_sender_and_extra_fields_reject(self):
        for field, value in (("quote", "幻觉引文"), ("sender", "A"), ("source_line", 2)):
            raw = claim_for(self.message)
            raw["evidence"][0][field] = value
            with self.subTest(field=field), self.assertRaises(ClaimValidationError):
                validate_claim(raw, self.messages)
        raw = claim_for(self.message)
        raw["evidence"][0]["source_file"] = "not_allowed"
        with self.assertRaises(ClaimValidationError):
            validate_claim(raw, self.messages)
        raw = claim_for(self.message)
        raw["extra"] = 1
        with self.assertRaises(ClaimValidationError):
            validate_claim(raw, self.messages)

    def test_cross_segment_and_duplicate_claim_reject_whole_response(self):
        raw = claim_for(self.message)
        with self.assertRaises(ClaimValidationError):
            validate_extraction({"claims": [raw]}, self.messages, set())
        with self.assertRaises(ClaimValidationError):
            validate_extraction({"claims": [raw, raw]}, self.messages, {1})
        with self.assertRaises(ClaimValidationError):
            validate_extraction({"claims": [], "persona": {}}, self.messages, {1})

    def test_inference_requires_label_and_time_offsets_compare_by_instant(self):
        self.message["ts"] = "2024-01-01T08:00:00+08:00"
        inference = claim_for(self.message, category="inference", statement="可能表达临时偏好")
        inference.update(time_start="2023-12-31T16:00:00-08:00", time_end="2024-01-01T09:00:00+08:00")
        self.assertEqual(validate_claim(inference, self.messages)["category"], "inference")
        inference["time_end"] = "2024-01-01T07:00:00+08:00"
        with self.assertRaises(ClaimValidationError):
            validate_claim(inference, self.messages)
        canonical = claim_for(self.message, category="inference", statement="可能表达临时偏好")
        canonical.update(time_start="2024-01-01T00:00:00+08:00",
                         time_end="2024-01-02T00:00:00+08:00")
        validated = validate_claim(canonical, self.messages)
        self.assertEqual(validated["time_start"], "2024-01-01T08:00:00+08:00")
        self.assertEqual(validated["time_end"], "2024-01-01T08:00:00+08:00")
        self.assertEqual(validated["model_time_interval_before_canonicalization"],
                         [canonical["time_start"], canonical["time_end"]])
        self.assertEqual(validated["time_semantics"],
                         "chat_evidence_interval_only_not_event_occurrence_date")
        definite = claim_for(self.message, category="inference", statement="一定是在表达永久偏好")
        with self.assertRaises(ClaimValidationError):
            validate_claim(definite, self.messages)

    def test_motives_uncertainty_and_unknown(self):
        for statement in ("B故意不回A", "B爱A", "B可能去了咖啡店"):
            with self.subTest(statement=statement), self.assertRaises(ClaimValidationError):
                validate_claim(claim_for(self.message, statement=statement), self.messages)
        unknown = claim_for(self.message, category="unknown", statement="无法确定这条消息的心理动机")
        unknown.update(evidence=[], time_start=None, time_end=None, confidence=0)
        self.assertEqual(validate_claim(unknown, self.messages)["category"], "unknown")
        inference = claim_for(self.message, category="inference", statement="可能是临时的表达方式")
        self.assertEqual(validate_claim(inference, self.messages)["category"], "inference")

    def test_emoji_and_reply_latency_cannot_prove_inner_state(self):
        for text in ("[微笑]", "[笑脸]", "🙂", "慢回不是故意"):
            self.message["text"] = text
            with self.subTest(text=text), self.assertRaises(ClaimValidationError):
                validate_claim(claim_for(self.message, category="inference", statement="可能是在暗示关系", confidence=.9), self.messages)
        self.message["text"] = "[微笑]"
        low = claim_for(self.message, category="inference", statement="可能是礼貌性的回应", confidence=.3)
        self.assertEqual(validate_claim(low, self.messages)["confidence"], .3)

    def test_response_and_merge_do_not_repair_or_invent(self):
        self.assertEqual(response_object('{"claims":[]}'), {"claims": []})
        self.assertEqual(response_object([{"type": "text", "text": '{"claims":[]}'}]), {"claims": []})
        for bad in ("```json\n{}\n```", "[]", "", None):
            with self.subTest(bad=bad), self.assertRaises(ClaimValidationError):
                response_object(bad)
        with self.assertRaises(ClaimValidationError):
            validate_merge_response({"summary": "test", "claim_ids": ["invented"]}, {"valid"})
        with self.assertRaises(ClaimValidationError):
            validate_merge_response({"summary": "无依据摘要", "claim_ids": []}, {"valid"})
        self.assertEqual(validate_merge_response({"summary": "", "claim_ids": []}, set()), {"summary": "", "claim_ids": []})
        valid = validate_merge_response({"summary": "合成", "claim_ids": ["valid"]}, {"valid", "other"})
        self.assertEqual(valid["claim_ids"], ["valid"])


class IdentityAndCacheSurfaceTests(FixtureCase):
    def test_plan_identity_is_public_generic_metadata(self):
        plan, _, _ = self.fixture(target_name="展示目标", slug="display-target",
                                  user_alias="展示用户")
        self.assertEqual(plan["target_name"], "展示目标")
        self.assertEqual(plan["slug"], "display-target")
        self.assertEqual(plan["user_alias"], "展示用户")
        self.assertEqual(plan["parameters"]["target_name"], "展示目标")
        self.assertEqual(plan["parameters"]["slug"], "display-target")
        with self.assertRaises(PlanningError):
            make_plan(synthetic_messages(2), source_path="synthetic.json",
                      target_name="Upper", slug="Not-Lower")

    def test_cached_claim_allows_only_three_local_time_fields(self):
        message = synthetic_messages(1)[0]
        base = claim_for(message)
        base["id"] = "claim_synthetic_cache"
        derived = {**base, "derived_time_start": None,
                   "derived_time_end": None, "derived_time_source": "evidence"}
        validated = validate_cached_claim(derived, {message["source_line"]: message})
        self.assertEqual(validated["derived_time_source"], "evidence")
        self.assertEqual(set(validated), CLAIM_KEYS | {
            "derived_time_start", "derived_time_end", "derived_time_source"})
        # 模型原始响应仍必须恰好七键；派生字段不能借此混入新模型输出。
        with self.assertRaises(ClaimValidationError):
            validate_claim(derived, {message["source_line"]: message})
        with self.assertRaises(ClaimValidationError):
            validate_cached_claim({**derived, "unexpected": 1},
                                  {message["source_line"]: message})


class RunnerTests(FixtureCase):
    def test_trial_full_resume_and_upstream_artifacts(self):
        plan, source, out = self.fixture(synthetic_messages(4), max_messages=2, input_price="2", output_price="8")
        fake = SyntheticProvider()
        trial = run_pipeline(plan, source, out, limit=1, client_factory=lambda: fake)
        self.assertEqual(trial["status"], "trial")
        self.assertEqual(len(fake.calls), 2)
        self.assertFalse(self.read_json(out / "meta.json")["distillation_complete"])
        resumed = SyntheticProvider()
        full = run_pipeline(plan, source, out, full=True, client_factory=lambda: resumed)
        self.assertEqual(full["status"], "complete")
        self.assertEqual(len(resumed.calls), 3)  # two uncached extraction + one merge
        self.assertEqual(self.read_json(out / "claims.json")["claim_count"], 4)
        for name in ARTIFACT_FILES:
            self.assertTrue((out / name).is_file(), name)
        self.assertEqual(self.read_json(out / "merged.json")["slug"], "synthetic-target")
        self.assertEqual(self.read_json(out / "meta.json")["name"], "测试目标")
        self.assertEqual(self.read_json(out / "events.json")["slug"], "synthetic-target")
        self.assertEqual(self.read_json(out / "timeline.json")["status"], "complete")
        self.assertTrue(self.read_json(out / "quality.json")["merge"]["claim_union_verified"])
        self.assertEqual(self.read_json(out / "quality.json")["holdout"]["evaluation_status"], "pending_not_run")
        # Cache-only rerun must not even initialize an authorized client.
        run_pipeline(plan, source, out, full=True,
                     client_factory=lambda: (_ for _ in ()).throw(AssertionError("must stay local")))

    def test_holdout_messages_never_sent(self):
        messages = synthetic_messages(50, separate_sessions=True)
        plan, source, out = self.fixture(messages, holdout_rate=.02)
        held = {l for s in plan["holdout"]["segments"] for l in s["source_lines"]}
        self.assertEqual(len(held), 1)
        fake = SyntheticProvider(excluded=held)
        run_pipeline(plan, source, out, full=True, client_factory=lambda: fake)
        sent = {row[0] for call in fake.calls for row in json.loads(call["messages"][1]["content"]).get("rows", [])}
        self.assertFalse(sent & held)
        self.assertEqual(sent, set(range(1, 51)) - held)

    def test_retry_count_is_permanent_and_cached(self):
        plan, source, out = self.fixture(synthetic_messages(1))
        fake = SyntheticProvider(invalid_first=True)
        run_pipeline(plan, source, out, full=True, client_factory=lambda: fake)
        self.assertEqual(len(fake.calls), 4)  # invalid+valid for each of two tasks
        checkpoint = self.read_json(out / "segments" / "segment_000000.persona.json")
        self.assertEqual([a["outcome"] for a in checkpoint["attempts"]], ["invalid_response", "completed"])
        self.assertEqual(len(checkpoint["attempts"]), 2)
        diagnostics = list((out / 'diagnostics').glob('segments.segment_000000.persona.attempt_1.*.json'))
        self.assertEqual(len(diagnostics), 1)
        diagnostic = self.read_json(diagnostics[0])
        self.assertIn('quote', diagnostic['validation_error'])
        self.assertIn('不存在的合成原话', diagnostic['response_text'])
        self.assertEqual(diagnostics[0].stat().st_mode & 0o777, 0o600)
        run_pipeline(plan, source, out, full=True,
                     client_factory=lambda: (_ for _ in ()).throw(AssertionError("cached")))

    def test_provider_failure_never_merges_or_leaks_exception_text(self):
        plan, source, out = self.fixture(synthetic_messages(4), max_messages=2)
        fake = SyntheticProvider(fail="SYNTHETIC_SECRET_IN_PROVIDER_EXCEPTION")
        with self.assertRaises(PipelineError) as raised:
            run_pipeline(plan, source, out, full=True, client_factory=lambda: fake)
        self.assertEqual(len(fake.calls), 2)
        self.assertNotIn("SYNTHETIC_SECRET", str(raised.exception))
        self.assertEqual(self.read_json(out / "meta.json")["status"], "incomplete")
        self.assertFalse(self.read_json(out / "run_state.json")["merge_performed"])
        for path in out.rglob("*.json"):
            self.assertNotIn("SYNTHETIC_SECRET", path.read_text(encoding="utf-8"))
        again = SyntheticProvider()
        with self.assertRaisesRegex(PipelineError, "exhausted two attempts"):
            run_pipeline(plan, source, out, full=True, client_factory=lambda: again)
        self.assertEqual(again.calls, [])

    def test_checkpoint_tamper_blocks_success(self):
        plan, source, out = self.fixture(synthetic_messages(1))
        fake = SyntheticProvider()
        run_pipeline(plan, source, out, full=True, client_factory=lambda: fake)
        path = out / "segments" / "segment_000000.persona.json"
        checkpoint = self.read_json(path)
        checkpoint["response"]["claims"][0]["statement"] += "tampered"
        path.write_text(json.dumps(checkpoint, ensure_ascii=False), encoding="utf-8")
        again = SyntheticProvider()
        with self.assertRaisesRegex(PipelineError, "checkpoint hash/identity mismatch"):
            run_pipeline(plan, source, out, full=True, client_factory=lambda: again)
        self.assertEqual(again.calls, [])
        self.assertEqual(self.read_json(out / "merged.json")["status"], "incomplete")

    def test_missing_done_file_is_rebuilt_from_validated_nodes_not_skipped(self):
        plan, source, out = self.fixture(synthetic_messages(1))
        run_pipeline(plan, source, out, full=True, client_factory=lambda: SyntheticProvider())
        done = out / "segments" / "segment_000000.done.json"
        self.assertEqual(done.resolve(), (out.resolve() / "segments" / "segment_000000.done.json"))
        done.unlink()  # verified synthetic temp target, never an actual work output
        run_pipeline(plan, source, out, full=True,
                     client_factory=lambda: (_ for _ in ()).throw(AssertionError("cache-only")))
        self.assertTrue(done.exists())
        self.assertEqual(self.read_json(done)["status"], "completed")

    def test_invalid_merge_keeps_claims_with_lossless_fallback(self):
        plan, source, out = self.fixture(synthetic_messages(4), max_messages=2)
        fake = SyntheticProvider(invalid_merge=True)
        result = run_pipeline(plan, source, out, full=True, client_factory=lambda: fake)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(len(fake.calls), 6)
        self.assertEqual(self.read_json(out / "claims.json")["claim_count"], 4)
        self.assertEqual(self.read_json(out / "merged.json")["summary_views"], [])
        quality = self.read_json(out / "quality.json")
        self.assertTrue(quality["distillation_complete"])
        self.assertGreater(quality["merge"]["no_fit_pair_fallbacks"], 0)
        self.assertTrue(quality["merge"]["claim_union_verified"])

    def test_overbudget_merge_deterministically_unions_without_truncation(self):
        plan, source, out = self.fixture(synthetic_messages(4), max_messages=2)
        fake = SyntheticProvider(large=True)
        run_pipeline(plan, source, out, full=True, client_factory=lambda: fake)
        self.assertEqual(len(fake.calls), 4)
        claims = self.read_json(out / "claims.json")["claims"]
        self.assertEqual(len(claims), 80)
        audit = self.read_json(out / "quality.json")["merge"]
        self.assertEqual(audit["truncated_claims"], 0)
        self.assertGreater(audit["no_fit_pair_fallbacks"], 0)
        self.assertEqual(audit["final_unmerged_nodes"], 2)
        self.assertTrue(audit["claim_union_verified"])

    def test_partial_valid_claim_survives_later_failure(self):
        plan, source, out = self.fixture(synthetic_messages(1))

        class SecondNodeFailure(SyntheticProvider):
            def create(self, **kwargs):
                if "记忆事件抽取器" in kwargs["messages"][0]["content"]:
                    raise RuntimeError("synthetic provider failure")
                return super().create(**kwargs)

        with self.assertRaises(PipelineError):
            run_pipeline(plan, source, out, full=True, client_factory=lambda: SecondNodeFailure())
        library = self.read_json(out / "claims.json")
        self.assertEqual(library["status"], "incomplete")
        self.assertEqual(library["claim_count"], 1)
        self.assertEqual(library["claims"][0]["category"], "style")
        self.assertFalse(self.read_json(out / "run_state.json")["merge_performed"])

    def test_attempt_is_consumed_before_provider_crash(self):
        plan, source, out = self.fixture(synthetic_messages(3), max_messages=2)

        class CrashProvider(SyntheticProvider):
            def create(self, **kwargs):
                raise KeyboardInterrupt("simulated local interruption")

        with self.assertRaises(KeyboardInterrupt):
            run_pipeline(plan, source, out, full=True, client_factory=lambda: CrashProvider())
        checkpoint = self.read_json(out / "segments" / "segment_000000.persona.json")
        self.assertEqual([a["outcome"] for a in checkpoint["attempts"]], ["started"])
        second = CrashProvider()
        with self.assertRaises(KeyboardInterrupt):
            run_pipeline(plan, source, out, full=True, client_factory=lambda: second)
        third = SyntheticProvider()
        # Both attempts were consumed even though neither returned a response.
        checkpoint = self.read_json(out / "segments" / "segment_000000.persona.json")
        self.assertEqual([a["outcome"] for a in checkpoint["attempts"]], ["started", "started"])
        with self.assertRaisesRegex(PipelineError, "exhausted two attempts"):
            run_pipeline(plan, source, out, full=True, client_factory=lambda: third)
        self.assertEqual(third.calls, [])

    def test_completed_checkpoint_requires_canonical_usage(self):
        plan, source, out = self.fixture(synthetic_messages(1))
        fake = SyntheticProvider()
        run_pipeline(plan, source, out, full=True, client_factory=lambda: fake)
        checkpoint_path = out / "segments" / "segment_000000.persona.json"
        checkpoint = self.read_json(checkpoint_path)
        checkpoint["attempts"][-1]["usage"] = None
        sealed = {key: value for key, value in checkpoint.items() if key != "checkpoint_sha256"}
        checkpoint["checkpoint_sha256"] = digest(sealed)
        checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")
        with self.assertRaisesRegex(PipelineError, "completed usage"):
            run_pipeline(plan, source, out, full=True,
                         client_factory=lambda: SyntheticProvider())

    def test_provider_error_cannot_reflect_private_request_text(self):
        plan, source, out = self.fixture(synthetic_messages(1))
        secret = "SYNTHETIC-PRIVATE-TEXT"
        data = json.loads(source.read_text())
        data[0]["text"] = secret
        source.write_text(json.dumps(data))

        class ReflectingProvider(SyntheticProvider):
            def create(self, **kwargs):
                self.calls.append(kwargs)
                raise RuntimeError(str(kwargs["messages"]))

        with self.assertRaises(PipelineError) as raised:
            run_pipeline(plan, source, out, full=True,
                         client_factory=lambda: ReflectingProvider())
        self.assertNotIn(secret, str(raised.exception))
        for path in (out / "segments").glob("*.json"):
            self.assertNotIn(secret, path.read_text())

    def test_overbudget_request_rejected_before_client_initialization(self):
        executor = NodeExecutor(self.root, "synthetic-plan", 4001,
                                _LazyClient(lambda: (_ for _ in ()).throw(AssertionError("must not initialize"))))
        with self.assertRaisesRegex(PipelineError, "hard context bound"):
            executor.call(group="segments", name="test", identity={},
                          api_messages=[{"role": "user", "content": "long synthetic" * 500}],
                          source_message_count=1, validator=lambda raw: raw)

    def test_untimed_or_inference_claim_never_enters_persona_core(self):
        plan, source, out = self.fixture(synthetic_messages(1))

        class UntimedProvider(SyntheticProvider):
            def create(self, **kwargs):
                result = super().create(**kwargs)
                data = json.loads(result.choices[0].message.content)
                if "claims" in data:
                    data["claims"][0].update(time_start=None, time_end=None,
                                             category="inference", statement="可能是临时表达")
                    result.choices[0].message.content = json.dumps(data, ensure_ascii=False)
                return result

        run_pipeline(plan, source, out, full=True, client_factory=lambda: UntimedProvider())
        self.assertEqual(self.read_json(out / "claims.json")["claim_count"], 1)  # same evidence/content is deduplicated
        self.assertEqual(self.read_json(out / "persona_core.json")["claim_ids"], [])
        self.assertEqual(self.read_json(out / "merged.json")["persona"]["core_traits"], [])

    def test_baseline_is_explicit_and_merged_schema_is_stable(self):
        plan, source, out = self.fixture()
        result = write_local_baseline(plan, source, out)
        self.assertEqual(result["status"], "local_baseline")
        self.assertEqual(self.read_json(out / "claims.json")["claims"], [])
        self.assertEqual(self.read_json(out / "meta.json")["coverage"], "local_baseline")
        self.assertIn("仅本地骨架", (out / "persona.md").read_text(encoding="utf-8"))
        self.assertEqual(self.read_json(out / "run_state.json")["api_calls"], 0)
        merged = self.read_json(out / "merged.json")
        self.assertEqual(merged["slug"], "synthetic-target")
        self.assertEqual(merged["template_version"], 3)

    def test_cli_plan_is_local_and_run_mode_is_required(self):
        plan, source, out = self.fixture()
        other = self.root / "cli"
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = main(["plan", "--messages", str(source), "--out", str(other), "--max-messages", "2"])
        self.assertEqual(code, 0, stderr.getvalue())
        report = json.loads(stdout.getvalue())
        self.assertEqual(report["status"], "planned_local_only")
        self.assertEqual(report["cost"]["status"], "pending_price_confirmation")
        self.assertFalse((other / "run_state.json").exists())
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main(["run", "--plan", str(out / "plan.json")])


if __name__ == "__main__":
    unittest.main()
