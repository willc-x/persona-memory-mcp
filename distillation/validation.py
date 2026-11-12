"""Strict claim/schema validation and local evidence verification.

Validation is deliberately all-or-nothing for an extraction response: dropping
only the bad claims would turn an invalid model response into an apparently
successful segment.
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any, Mapping

from .planning import digest

CATEGORIES = ("fact", "emotion_expression", "inference", "unknown", "style", "relationship")
CLAIM_KEYS = {"id", "category", "statement", "evidence", "time_start", "time_end", "confidence"}
# 本地时间规范化可以补三项 provenance 元数据；模型原始响应仍必须恰好为
# CLAIM_KEYS，入口在 validate_extraction 中显式拒绝扩展字段。
LOCAL_TIME_KEYS = frozenset({
    "time_semantics", "time_provenance", "model_time_interval_before_canonicalization",
})
# 已写盘并重新读取的 checkpoint 还允许三项派生的展示时间；它们不进入模型输出，
# 也不改变 canonical claim ID。
DERIVED_TIME_KEYS = frozenset({"derived_time_start", "derived_time_end", "derived_time_source"})
CACHED_CLAIM_KEYS = CLAIM_KEYS | DERIVED_TIME_KEYS
EVIDENCE_KEYS = {"source_line", "sender", "quote"}
ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
# A direct utterance can be stored as emotion_expression.  It cannot become an
# unconditional psychological fact merely because a model states it confidently.
INNER_STATE_RE = re.compile(
    r"爱|喜欢|讨厌|在乎|想念|思念|故意|有意|内心|心里|觉得|认为|害怕|焦虑|生气|"
    r"嫉妒|冷淡|冷漠|暗示|舍不得|期待|渴望|依赖|占有欲|愧疚|内疚|后悔|委屈|"
    r"不安|轻视|尊重|信任|怀疑|动情|心动|放下|释怀|想要|希望|动机|意图|潜意识")
UNCERTAINTY_RE = re.compile(r"可能|或许|大概|也许|似乎|推测|推断|不确定|疑似|likely|perhaps|probably", re.I)
EMOJI_ONLY_RE = re.compile(r"^\s*(?:\[[^\[\]\n]{1,30}\]|[\U0001F000-\U0001FAFF])+\s*$")
SLOW_REPLY_RE = re.compile(r"慢回|回复慢|隔了.{0,12}才回|长时间没回|未及时回复|已读不回")


class ClaimValidationError(ValueError):
    """A model response is invalid or its evidence is not locally exact."""


def _instant(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _iso_or_none(value: Any, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ClaimValidationError(f"{field} must be an ISO timestamp or null")
    try:
        datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError as exc:
        raise ClaimValidationError(f"{field} must be an ISO timestamp or null") from exc
    return value.strip()


def response_object(content: Any) -> dict[str, Any]:
    """Parse an ordinary OpenAI JSON response; no markdown repair heuristics."""
    if isinstance(content, list):
        # OpenAI-compatible clients may expose text parts instead of a string.
        pieces = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str):
                pieces.append(part["text"])
            else:
                raise ClaimValidationError("response content contains a non-text part")
        content = "".join(pieces)
    if not isinstance(content, str) or not content.strip():
        raise ClaimValidationError("model response content must be non-empty JSON text")
    try:
        value = json.loads(content)
    except (json.JSONDecodeError, RecursionError) as exc:
        raise ClaimValidationError(f"invalid model JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ClaimValidationError("model response root must be an object")
    return value


def validate_claim(raw: Any, messages: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    allowed_keys = CLAIM_KEYS | LOCAL_TIME_KEYS
    if not isinstance(raw, dict) or not CLAIM_KEYS.issubset(raw) or not set(raw).issubset(allowed_keys):
        raise ClaimValidationError("claim must contain exactly CLAIM_KEYS and optional local time provenance")
    claim_id = raw.get("id")
    if not isinstance(claim_id, str) or not ID_RE.fullmatch(claim_id):
        raise ClaimValidationError("claim id must be 1-128 ASCII letters/digits/_/-")
    category = raw.get("category")
    if category not in CATEGORIES:
        raise ClaimValidationError(f"invalid claim category: {category!r}")
    statement = raw.get("statement")
    if not isinstance(statement, str) or not statement.strip() or len(statement) > 4000:
        raise ClaimValidationError("claim statement must be non-empty and <=4000 characters")
    statement = statement.strip()
    confidence = raw.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise ClaimValidationError("claim confidence must be a number in [0,1]")
    if not 0 <= confidence <= 1 or confidence != confidence:
        raise ClaimValidationError("claim confidence must be a finite number in [0,1]")
    time_start = _iso_or_none(raw.get("time_start"), "time_start")
    time_end = _iso_or_none(raw.get("time_end"), "time_end")
    if time_start and time_end:
        try:
            if _instant(time_start) > _instant(time_end):
                raise ClaimValidationError("time_start is after time_end")
        except TypeError as exc:
            raise ClaimValidationError("time_start/time_end mix naive and aware timestamps") from exc
    evidence_raw = raw.get("evidence")
    if not isinstance(evidence_raw, list):
        raise ClaimValidationError("claim evidence must be an array")
    if category != "unknown" and not evidence_raw:
        raise ClaimValidationError("non-unknown claims require at least one evidence item")
    if len(evidence_raw) > 100:
        raise ClaimValidationError("too many evidence items on one claim")
    evidence = []
    evidence_texts = []
    evidence_times = []
    for item in evidence_raw:
        if not isinstance(item, dict) or set(item) != EVIDENCE_KEYS:
            raise ClaimValidationError("evidence must have exactly source_line, sender, quote")
        line, sender, quote = item.get("source_line"), item.get("sender"), item.get("quote")
        if isinstance(line, bool) or not isinstance(line, int) or line not in messages:
            raise ClaimValidationError(f"unknown or invalid source_line: {line!r}")
        if sender not in ("A", "B"):
            raise ClaimValidationError("evidence sender must be A or B")
        if not isinstance(quote, str) or not quote:
            raise ClaimValidationError("evidence quote must be a non-empty string")
        message = messages[line]
        if sender != message["sender"]:
            raise ClaimValidationError(f"sender mismatch at source_line {line}")
        # Exact Unicode substring.  No trimming/case folding/ellipsis removal.
        if quote not in message["text"]:
            raise ClaimValidationError(f"quote is not an exact substring at source_line {line}")
        evidence.append({"source_line": line, "sender": sender, "quote": quote})
        evidence_texts.append(message["text"])
        evidence_times.append(str(message["ts"]))
    # 模型给出的日期窗口只是候选元数据，不能让它把已经通过逐字回查的证据
    # 拒之门外。最终 claim 的时间区间严格取所引用原话的实际时间；若模型日期
    # 不同则保留在 model_time_interval_before_canonicalization，明确不等于
    # 事件真实发生日期（对话发生时间只证明“当时这样说”）。
    model_interval = [time_start, time_end]
    try:
        instants = [_instant(value) for value in evidence_times]
    except TypeError as exc:
        raise ClaimValidationError("claim/evidence timestamps mix naive and aware offsets") from exc
    if evidence_times:
        ordered = sorted(zip(instants, evidence_times))
        time_start, time_end = ordered[0][1], ordered[-1][1]
    quoted_uncertainty = any(UNCERTAINTY_RE.search(item["quote"]) for item in evidence_raw)
    reported_speech = re.search(r"(?:说|表示|自述|提到|讲述|称|回忆|转述)", statement)
    if category in ("fact", "style", "relationship") and UNCERTAINTY_RE.search(statement) and not (
            reported_speech and quoted_uncertainty):
        raise ClaimValidationError("uncertain statements must use inference or unknown")
    if category == "inference" and not UNCERTAINTY_RE.search(statement):
        raise ClaimValidationError("inference statements must explicitly say 可能/推断/etc")
    if category == "fact" and INNER_STATE_RE.search(statement):
        # 仅接受有第一人称原话的明确非关系偏好自述，仍只证明当时说过。
        # “喜欢你/爱她/故意不回”等关系、动机与持续心理不能借此放行。
        reported = re.search(
            r"(?:说过|表示|自述|提到|讲述|说|称|回忆).{0,30}"
            r"(?:喜欢|讨厌|爱|好听|觉得|感觉|认为|希望|舍不得|期待)", statement)
        quoted = any(
            sender == "B" and re.search(
                r"(?:我[^。！？!?\n]{0,12}?)?"
                r"(?:喜欢|讨厌|爱|好听|觉得|感觉|认为|希望|舍不得|期待|想要)"
                r"(?!你|他|她|对方)", quote)
            for sender, quote in ((item["sender"], item["quote"]) for item in evidence_raw)
        )
        relational = re.search(r"(?:喜欢|讨厌|爱)(?:你|他|她|对方)|(?:在乎|想念|故意|有意|内心|动机|意图)", statement)
        if not (reported and quoted and not relational):
            raise ClaimValidationError(
                "psychological states/motives cannot be facts; use emotion_expression for a direct utterance, inference, or unknown")
    weak_context = bool(evidence_texts) and (
        all(EMOJI_ONLY_RE.fullmatch(text) for text in evidence_texts)
        or any(SLOW_REPLY_RE.search(text) for text in evidence_texts))
    if weak_context and category in ("fact", "relationship") and (
            INNER_STATE_RE.search(statement) or SLOW_REPLY_RE.search(statement)):
        raise ClaimValidationError("emoji or reply latency cannot establish a psychological/relationship conclusion")
    if weak_context and category == "inference" and confidence > 0.65:
        raise ClaimValidationError("emoji/latency-based inference must have confidence <=0.65")
    # IDs are content addressed, making union/idempotency independent of a
    # model reusing 'claim_1' for unrelated statements in different segments.
    content = {"category": category, "statement": statement, "evidence": evidence,
               "time_start": time_start, "time_end": time_end, "confidence": confidence}
    canonical_id = "claim_" + digest(content)[:24]
    result = {"id": canonical_id, **content}
    if evidence_times:
        result["time_semantics"] = "chat_evidence_interval_only_not_event_occurrence_date"
        result["time_provenance"] = "derived_from_exactly_cited_source_timestamps"
        if model_interval != [time_start, time_end]:
            result["model_time_interval_before_canonicalization"] = model_interval
    return result


def validate_cached_claim(raw: Any, messages: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    """重新核验本地 checkpoint claim，仅额外允许三项派生展示时间。

    模型原始输出先在 ``validate_extraction`` 中要求恰好 CLAIM_KEYS。缓存中的
    派生字段仅用于展示/过滤，不替代 canonical ``time_start/time_end``，也不改变
    claim 的内容寻址 ID。
    """
    if not isinstance(raw, dict) or set(raw) != CACHED_CLAIM_KEYS:
        raise ClaimValidationError(
            "cached claim must have exactly CLAIM_KEYS plus three derived time fields")
    base = {key: raw[key] for key in CLAIM_KEYS}
    claim = validate_claim(base, messages)
    start = _iso_or_none(raw.get("derived_time_start"), "derived_time_start")
    end = _iso_or_none(raw.get("derived_time_end"), "derived_time_end")
    if start and end:
        try:
            if _instant(start) > _instant(end):
                raise ClaimValidationError("derived_time_start is after derived_time_end")
        except TypeError as exc:
            raise ClaimValidationError("derived time fields mix naive and aware timestamps") from exc
    source = raw.get("derived_time_source")
    if not isinstance(source, str) or not source.strip() or len(source) > 80:
        raise ClaimValidationError("derived_time_source must be a non-empty string <=80 characters")
    # cached surface is exactly CLAIM_KEYS + three derived display fields;
    # validate_claim may add canonical provenance, which is not part of that
    # checkpoint-facing schema and must not leak into it.
    base_claim = {key: claim[key] for key in CLAIM_KEYS}
    return {**base_claim, "derived_time_start": start, "derived_time_end": end,
            "derived_time_source": source.strip()}


def _validate_claim_list(raw: Any, messages: Mapping[int, Mapping[str, Any]],
                         allowed_lines: set[int], *, allow_local_time: bool) -> list[dict[str, Any]]:
    if not isinstance(raw, dict) or set(raw) != {"claims"} or not isinstance(raw.get("claims"), list):
        raise ClaimValidationError("extraction response must be exactly {'claims': [...]}")
    if len(raw["claims"]) > 500:
        raise ClaimValidationError("extraction exceeds 500 claims")
    result = []
    seen_ids = set()
    for candidate in raw["claims"]:
        # Model output is still schema-exact: time provenance is added locally
        # below and must never be accepted as a model-provided extension. Only
        # reloaded checkpoints may carry those three local provenance keys.
        expected = CLAIM_KEYS | (LOCAL_TIME_KEYS if allow_local_time else set())
        if not isinstance(candidate, dict) or not CLAIM_KEYS.issubset(candidate) or not set(candidate).issubset(expected):
            raise ClaimValidationError(
                "claim must contain CLAIM_KEYS" +
                (" and optional local time provenance" if allow_local_time else
                 " and no local time provenance"))
        # First validate against only this segment.  A quote from a holdout or
        # another segment is invalid even when it happens to match the corpus.
        local = {line: messages[line] for line in allowed_lines if line in messages}
        claim = validate_claim(candidate, local)
        if claim["id"] in seen_ids:
            raise ClaimValidationError("duplicate canonical claim id in one response")
        seen_ids.add(claim["id"])
        result.append(claim)
    return result


def validate_extraction(raw: Any, messages: Mapping[int, Mapping[str, Any]],
                        allowed_lines: set[int]) -> list[dict[str, Any]]:
    """Validate a fresh model response; local provenance keys are forbidden."""
    return _validate_claim_list(raw, messages, allowed_lines, allow_local_time=False)


def validate_stored_extraction(raw: Any, messages: Mapping[int, Mapping[str, Any]],
                               allowed_lines: set[int]) -> list[dict[str, Any]]:
    """Revalidate a completed node after reloading it from local checkpoints."""
    return _validate_claim_list(raw, messages, allowed_lines, allow_local_time=True)


def validate_merge_response(raw: Any, allowed_ids: set[str]) -> dict[str, Any]:
    if not isinstance(raw, dict) or set(raw) != {"summary", "claim_ids"}:
        raise ClaimValidationError("merge response must be exactly {'summary': str, 'claim_ids': [...]}")
    summary, ids = raw.get("summary"), raw.get("claim_ids")
    if not isinstance(summary, str) or len(summary) > 12_000:
        raise ClaimValidationError("merge summary must be a string <=12000 characters")
    if not isinstance(ids, list) or any(not isinstance(i, str) for i in ids):
        raise ClaimValidationError("merge claim_ids must be an array of strings")
    if len(ids) != len(set(ids)) or not set(ids).issubset(allowed_ids):
        raise ClaimValidationError("merge claim_ids contain unknown or duplicate ids")
    if summary.strip() and not ids:
        raise ClaimValidationError("non-empty merge summary must cite at least one existing claim id")
    return {"summary": summary.strip(), "claim_ids": ids}
