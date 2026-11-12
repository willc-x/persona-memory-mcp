"""Deterministic local segmentation and explicit holdout selection.

No functions in this module make network calls or read credentials.  The
bound is an intentionally conservative UTF-8-byte estimate, not a promise of
provider tokenization; actual usage is recorded separately by ``run``.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from datetime import datetime
from decimal import Decimal, InvalidOperation
from math import isfinite
from pathlib import Path
from typing import Any, Iterable, Sequence

from .prompts import (MAX_OUTPUT_TOKENS, TARGET_NAME, USER_ALIAS, empirical_input_lower,
                      extraction_api_messages, hard_context_upper)

SCHEMA_VERSION = "distillation.plan.v3"
DEFAULT_CONTEXT_TOKENS = 16_000
DEFAULT_RESERVED_TOKENS = MAX_OUTPUT_TOKENS
DEFAULT_GAP_MINUTES = 30
DEFAULT_MAX_MESSAGES = 500
DEFAULT_HOLDOUT_RATE = 0.02
# Prompt, response envelope and request bookkeeping must fit too.  This is
# deliberately a separate allowance, rather than hiding data truncation.
PROMPT_OVERHEAD_TOKENS = 1_200
MERGE_OVERHEAD_TOKENS = 1_200
HOLDOUT_EXCLUDED = "holdout_excluded"
TRAIN = "train"


class PlanningError(ValueError):
    """The source cannot be planned without losing or misattributing data."""


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_digest(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    # Naive dates are source-local by contract; aware dates never get silently
    # converted to another timezone.
    return parsed


def validate_messages(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, dict):
        data = data.get("messages")
    if not isinstance(data, list):
        raise PlanningError("messages input must be a list or {'messages': list}")
    if not data:
        return []
    result: list[dict[str, Any]] = []
    seen_lines: set[int] = set()
    for index, raw in enumerate(data):
        if not isinstance(raw, dict):
            raise PlanningError(f"message {index} is not an object")
        message = dict(raw)
        sender = message.get("sender")
        line = message.get("source_line")
        if sender not in ("A", "B"):
            raise PlanningError(f"message {index}: sender must be A or B")
        if not isinstance(line, int) or isinstance(line, bool) or line < 1:
            raise PlanningError(f"message {index}: source_line must be a positive integer")
        if line in seen_lines:
            raise PlanningError(f"duplicate source_line: {line}")
        seen_lines.add(line)
        if not isinstance(message.get("text"), str):
            raise PlanningError(f"source_line {line}: text must be a string")
        if parse_timestamp(message.get("ts")) is None:
            raise PlanningError(f"source_line {line}: invalid or missing ISO ts")
        message["_input_index"] = index
        result.append(message)
    awareness = {parse_timestamp(m["ts"]).utcoffset() is not None for m in result}  # type: ignore[union-attr]
    if len(awareness) > 1:
        raise PlanningError("timestamps must be consistently naive source-local or timezone-aware")
    # Datetime ordering handles offsets; source_line remains the immutable
    # evidence anchor. Equal timestamps retain the source ordering.
    return sorted(result, key=lambda m: (parse_timestamp(m["ts"]), m["_input_index"]))


def load_messages(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as stream:
        return validate_messages(json.load(stream))


def token_bounds(text: str) -> dict[str, int]:
    """Return a documented estimate range for a UTF-8 string.

    ``upper`` charges one token per UTF-8 byte plus one.  This is conservative
    for byte-capable tokenizers; it is not a provider-side tokenizer.  The
    lower heuristic assumes four Unicode characters per token.  Both are
    estimates and are never used to remove text from a segment.
    """
    chars = len(text)
    return {"lower": max(1, (chars + 3) // 4),
            "upper": max(1, len(text.encode("utf-8")) + 1)}


def sum_bounds(items: Iterable[str]) -> dict[str, int]:
    result = {"lower": 0, "upper": 0}
    for text in items:
        bound = token_bounds(text)
        result["lower"] += bound["lower"]
        result["upper"] += bound["upper"]
    return result


def conversations(messages: Sequence[dict[str, Any]], gap_minutes: int) -> list[list[dict[str, Any]]]:
    if gap_minutes <= 0:
        raise PlanningError("gap_minutes must be positive")
    chunks: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    previous: datetime | None = None
    for message in messages:
        stamp = parse_timestamp(message["ts"])
        assert stamp is not None
        if previous is not None and (stamp - previous).total_seconds() > gap_minutes * 60:
            chunks.append(current)
            current = []
        current.append(message)
        previous = stamp
    if current:
        chunks.append(current)
    return chunks


def _month(chunk: Sequence[dict[str, Any]]) -> str:
    return str(chunk[0]["ts"])[:7]


def select_holdouts(chunks: Sequence[list[dict[str, Any]]], rate: float,
                    seed: str) -> tuple[set[int], list[dict[str, Any]]]:
    """Select whole conversations independently within each calendar month.

    Hash ranking makes selection reproducible without relying on corpus order
    or Python's randomized hash.  ``selection_detail`` exposes every decision,
    so one can verify that held-out messages are absent from requests.
    """
    if not 0 <= rate <= 1:
        raise PlanningError("holdout_rate must be in [0, 1]")
    groups: dict[str, list[int]] = {}
    for index, chunk in enumerate(chunks):
        groups.setdefault(_month(chunk), []).append(index)
    held: set[int] = set()
    detail: list[dict[str, Any]] = []
    for month in sorted(groups):
        indexes = groups[month]
        # Half-up rather than banker's rounding, retaining at least one train
        # conversation whenever another conversation exists.
        count = min(len(indexes) - 1, int(len(indexes) * rate + 0.5)) if len(indexes) > 1 else 0
        count = max(0, count)
        ranked = sorted(indexes, key=lambda i: digest({"seed": seed, "conversation": i}))
        chosen = sorted(ranked[:count])
        held.update(chosen)
        detail.append({"month": month, "conversations": len(indexes),
                       "target_rate": rate, "selected": len(chosen),
                       "actual_rate": count / len(indexes),
                       "conversation_indexes": chosen})
    return held, detail


def _message_request_view(message: dict[str, Any]) -> dict[str, Any]:
    # Every evidence anchor and independent-session boundary is explicit.
    view = {"source_line": message["source_line"], "ts": message["ts"],
            "sender": message["sender"], "text": message["text"]}
    if "_session_id" in message:
        view["session_id"] = message["_session_id"]
    return view


def _bounds_for_messages(segment_id: str, messages: Sequence[dict[str, Any]]) -> dict[str, int]:
    views = [_message_request_view(m) for m in messages]
    persona = extraction_api_messages("persona", segment_id, views)
    memories = extraction_api_messages("memories", segment_id, views)
    persona_upper = hard_context_upper(persona, len(views))
    memories_upper = hard_context_upper(memories, len(views))
    return {
        "lower": empirical_input_lower(persona) + empirical_input_lower(memories),
        "input_upper": hard_context_upper(persona, len(views), 0)
                      + hard_context_upper(memories, len(views), 0),
        "output_upper": 2 * MAX_OUTPUT_TOKENS,
        "upper": persona_upper + memories_upper,
        "per_call": {
            "persona": {"hard_upper": persona_upper,
                        "input_upper": hard_context_upper(persona, len(views), 0),
                        "output_upper": MAX_OUTPUT_TOKENS},
            "memories": {"hard_upper": memories_upper,
                         "input_upper": hard_context_upper(memories, len(views), 0),
                         "output_upper": MAX_OUTPUT_TOKENS},
        },
        "method": "exact UTF-8 bytes of compact columns+rows request +12/source message+32, plus 4000 output/call; lower is an empirical 4-chars estimate",
    }


def _fits(segment_id: str, messages: Sequence[dict[str, Any]], context_tokens: int) -> bool:
    bounds = _bounds_for_messages(segment_id, messages)
    return max(bounds["per_call"][kind]["hard_upper"]
               for kind in ("persona", "memories")) <= context_tokens


def _window_key(chunk: Sequence[dict[str, Any]], pack_window: str) -> str:
    stamp = parse_timestamp(chunk[0]["ts"])
    assert stamp is not None
    if pack_window == "day":
        return stamp.date().isoformat()
    if pack_window == "week":
        iso = stamp.isocalendar()
        return f"{iso.year}-W{iso.week:02d}"
    return "all"


def _session_units(chunks: Sequence[list[dict[str, Any]]], max_messages: int,
                   context_tokens: int) -> list[list[dict[str, Any]]]:
    """Split only when one session itself exceeds count/hard-byte bounds."""
    units: list[list[dict[str, Any]]] = []
    for conversation_index, original in enumerate(chunks):
        session_id = f"session_{conversation_index:06d}"
        chunk = [{**message, "_session_id": session_id} for message in original]
        current: list[dict[str, Any]] = []
        part = 0
        for message in chunk:
            candidate = current + [message]
            if current and (len(candidate) > max_messages
                            or not _fits("segment_000000", candidate, context_tokens)):
                units.append(current)
                part += 1
                current = []
                candidate = [message]
            if not _fits("segment_000000", candidate, context_tokens):
                raise PlanningError(
                    f"source_line {message['source_line']} cannot fit context without "
                    "truncation; increase --context-tokens")
            current = candidate
        if current:
            units.append(current)
    return units


def _segment_record(segment_id: str, messages: Sequence[dict[str, Any]],
                    disposition: str) -> dict[str, Any]:
    sessions: list[dict[str, Any]] = []
    for message in messages:
        sid = message["_session_id"]
        if not sessions or sessions[-1]["id"] != sid:
            sessions.append({"id": sid,
                             "conversation_index": int(sid.rsplit("_", 1)[1]),
                             "start": message["ts"], "end": message["ts"],
                             "source_lines": [message["source_line"]]})
        else:
            sessions[-1]["end"] = message["ts"]
            sessions[-1]["source_lines"].append(message["source_line"])
    views = [_message_request_view(message) for message in messages]
    return {
        "id": segment_id, "disposition": disposition,
        "conversation_index": sessions[0]["conversation_index"],
        "month": str(messages[0]["ts"])[:7],
        "months": sorted({str(message["ts"])[:7] for message in messages}),
        "start": messages[0]["ts"], "end": messages[-1]["ts"],
        "count": len(messages), "session_ids": [s["id"] for s in sessions],
        "sessions": sessions, "source_lines": [m["source_line"] for m in messages],
        "message_sha256": digest(views),
        "token_bounds": _bounds_for_messages(segment_id, messages),
        "packing": "whole sessions except explicit oversize session parts; session_id boundaries retained",
    }


def split_conversations(chunks: Sequence[list[dict[str, Any]]], holdout_indexes: set[int],
                        max_messages: int, data_budget: int,
                        context_tokens: int | None = None,
                        pack_window: str = "week") -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if max_messages <= 0 or data_budget <= 0:
        raise PlanningError("max_messages and usable data budget must be positive")
    if pack_window not in ("day", "week", "all"):
        raise PlanningError("pack_window must be day, week, or all")
    context_tokens = context_tokens or data_budget + DEFAULT_RESERVED_TOKENS
    units = _session_units(chunks, max_messages, context_tokens)
    packs: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_key: str | None = None
    for unit in units:
        conversation_index = int(unit[0]["_session_id"].rsplit("_", 1)[1])
        if conversation_index in holdout_indexes:
            continue
        key = _window_key(unit, pack_window)
        candidate = current + unit
        if current and (key != current_key or len(candidate) > max_messages
                        or not _fits("segment_000000", candidate, context_tokens)):
            packs.append(current)
            current = []
            candidate = list(unit)
        if not current:
            current_key = key
        current = candidate
    if current:
        packs.append(current)
    segments = [_segment_record(f"segment_{index:06d}", pack, TRAIN)
                for index, pack in enumerate(packs)]
    # Holdouts remain audit-only whole-session records and are never packed
    # into a training request, even when they share the same day/week.
    holdouts: list[dict[str, Any]] = []
    for index, chunk in enumerate(chunks):
        if index not in holdout_indexes:
            continue
        session_id = f"session_{index:06d}"
        marked = [{**message, "_session_id": session_id} for message in chunk]
        holdouts.append(_segment_record(f"holdout_{index:06d}", marked, HOLDOUT_EXCLUDED))
    return segments, holdouts


def _merge_levels(segments: Sequence[dict[str, Any]], data_budget: int) -> list[dict[str, Any]]:
    """Estimate a balanced binary hierarchy, charging a conservative claim cap.

    No source quotes are used to bypass the bound.  If a future run produces
    larger validated views it creates more child batches instead of trimming.
    """
    if not segments:
        return []
    # Two extraction responses per segment; this cap covers a structured,
    # non-quote claim/summary view and is checked again before every merge.
    units = [{"ids": [s["id"]],
              "tokens": min(data_budget + 1, 2 * DEFAULT_RESERVED_TOKENS + 600)}
             for s in segments]
    levels = []
    level = 0
    while len(units) > 1:
        next_units = []
        batches = []
        for i in range(0, len(units), 2):
            group = units[i:i + 2]
            if len(group) == 1:
                next_units.append(group[0])
                continue
            cost = sum(item["tokens"] for item in group) + MERGE_OVERHEAD_TOKENS
            if cost > data_budget:
                # A singleton is never truncated: it moves to the next level.
                for item in group:
                    next_units.append(item)
                continue
            ids = [sid for item in group for sid in item["ids"]]
            batches.append({"level": level, "inputs": [item["ids"] for item in group],
                            "estimated_upper_tokens": cost})
            next_units.append({"ids": ids,
                               "tokens": min(data_budget + 1, DEFAULT_RESERVED_TOKENS + 600)})
        if len(next_units) == len(units):
            # Runtime uses validated claim views with explicit per-claim bounds;
            # planning must still give a finite estimate rather than fake zero.
            batches.append({"level": level, "deferred_bounded_batches": len(units),
                            "estimated_upper_tokens": len(units) * data_budget})
            break
        levels.append({"level": level, "calls": len(batches), "batches": batches})
        units = next_units
        level += 1
    return levels


def estimate_merge_calls(segment_count: int) -> int:
    """Full binary fan-in has exactly n-1 merge calls (n > 0)."""
    return max(0, segment_count - 1)


def _price(value: str | None, name: str) -> None:
    if value is None:
        return
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError):
        raise PlanningError(f"{name} must be a non-negative finite decimal") from None
    if not parsed.is_finite() or parsed < 0:
        raise PlanningError(f"{name} must be a non-negative finite decimal")


def make_plan(messages: Sequence[dict[str, Any]], *, source_path: str,
              source_sha256: str | None = None, max_messages: int = DEFAULT_MAX_MESSAGES,
              context_tokens: int = DEFAULT_CONTEXT_TOKENS,
              reserved_tokens: int = DEFAULT_RESERVED_TOKENS,
              gap_minutes: int = DEFAULT_GAP_MINUTES,
              holdout_rate: float = DEFAULT_HOLDOUT_RATE,
              input_price: str | None = None, output_price: str | None = None,
              pack_window: str = "week", target_name: str = TARGET_NAME,
              slug: str = "persona", user_alias: str = USER_ALIAS) -> dict[str, Any]:
    if isinstance(max_messages, bool) or not isinstance(max_messages, int) or max_messages <= 0:
        raise PlanningError("max_messages must be a positive integer")
    if (not isinstance(holdout_rate, (int, float)) or isinstance(holdout_rate, bool)
            or not isfinite(holdout_rate)):
        raise PlanningError("holdout_rate must be a finite number in [0,1]")
    if isinstance(context_tokens, bool) or not isinstance(context_tokens, int):
        raise PlanningError("context_tokens must be an integer")
    _price(input_price, "input_price")
    _price(output_price, "output_price")
    if pack_window not in ("day", "week", "all"):
        raise PlanningError("pack_window must be day, week or all")
    if (not isinstance(target_name, str) or not target_name.strip()
            or len(target_name) > 80 or any(ch in target_name for ch in "\r\n")):
        raise PlanningError("target_name must be one non-empty line (max 80 characters)")
    if not isinstance(slug, str) or not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", slug):
        raise PlanningError("slug must match [a-z0-9][a-z0-9_-]* and be at most 64 characters")
    if (not isinstance(user_alias, str) or not user_alias.strip()
            or len(user_alias) > 80 or any(ch in user_alias for ch in "\r\n")):
        raise PlanningError("user_alias must be one non-empty line (max 80 characters)")
    if reserved_tokens != DEFAULT_RESERVED_TOKENS:
        raise PlanningError(f"this version requires --reserved-tokens {DEFAULT_RESERVED_TOKENS} (prompt + output reserve)")
    if context_tokens <= reserved_tokens:
        raise PlanningError("context_tokens must exceed the 4000-token reserve")
    data_budget = context_tokens - reserved_tokens
    ordered = validate_messages(list(messages))
    if not ordered:
        raise PlanningError("cannot plan an empty corpus")
    chunks = conversations(ordered, gap_minutes)
    selection_seed = source_sha256 or digest([_message_request_view(m) for m in ordered])
    held, selection = select_holdouts(chunks, holdout_rate, selection_seed)
    segments, holdouts = split_conversations(chunks, held, max_messages, data_budget,
                                             context_tokens=context_tokens,
                                             pack_window=pack_window)
    if not segments:
        raise PlanningError("all conversations are held out; no distillation is possible")
    merge_levels = _merge_levels(segments, data_budget)
    merge_calls = estimate_merge_calls(len(segments))
    base_calls = 2 * len(segments)
    train_messages = [m for i, chunk in enumerate(chunks) if i not in held for m in chunk]
    held_messages = [m for i, chunk in enumerate(chunks) if i in held for m in chunk]
    request_bounds = sum_bounds(json.dumps(_message_request_view(m), ensure_ascii=False)
                                for m in train_messages)
    extraction_upper = sum(s["token_bounds"]["upper"] for s in segments)
    # Every merge request gets the full hard context allowance. Runtime may use
    # fewer calls or deterministic no-truncation fallback, but never exceeds it.
    merge_upper = merge_calls * context_tokens
    base_upper = extraction_upper + merge_upper
    retry_upper = 2 * base_upper
    estimate = {
        "currency": "CNY", "price_unit": "RMB per million tokens",
        "input_price_per_million": input_price, "output_price_per_million": output_price,
        "status": "estimated" if input_price is not None and output_price is not None else "pending_price_confirmation",
        "estimated_cost_rmb": None,
        "note": "Token bounds are heuristic UTF-8/character estimates, not provider usage; merge upper bounds are rechecked before requests.",
    }
    if estimate["status"] == "estimated":
        try:
            pin, pout = Decimal(str(input_price)), Decimal(str(output_price))
            if not pin.is_finite() or not pout.is_finite() or pin < 0 or pout < 0:
                raise InvalidOperation
            extraction_input_upper = sum(s["token_bounds"]["input_upper"] for s in segments)
            merge_input_upper = merge_calls * data_budget
            output_upper = (base_calls + merge_calls) * MAX_OUTPUT_TOKENS
            base_cost = ((Decimal(extraction_input_upper + merge_input_upper) * pin)
                         + Decimal(output_upper) * pout) / Decimal(1_000_000)
            retry_cost = base_cost * 2
            estimate.update({"estimated_cost_rmb": str(retry_cost),
                             "base_cost_upper_rmb": str(base_cost),
                             "retry_2x_cost_upper_rmb": str(retry_cost),
                             "estimated_cost_range_rmb": ["0", str(retry_cost)]})
        except InvalidOperation as exc:
            raise PlanningError("prices must be non-negative decimal numbers") from exc
    segment_stats = [{"id": s["id"], "month": s["month"], "start": s["start"],
                      "end": s["end"], "count": s["count"],
                      "token_bounds": s["token_bounds"]} for s in segments]
    plan: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "source": {"path": str(source_path), "sha256": selection_seed},
        # 展示身份同时放在顶层，便于产物与计划分离读取；本版本会把它纳入
        # plan_hash（计划自洽），不承诺与不含该元数据的旧计划 hash 兼容。
        "target_name": target_name.strip(), "slug": slug, "user_alias": user_alias.strip(),
        "parameters": {"max_messages": max_messages, "context_tokens": context_tokens,
                       "reserved_prompt_output_tokens": reserved_tokens,
                       "prompt_overhead_tokens": PROMPT_OVERHEAD_TOKENS,
                       "gap_minutes": gap_minutes, "holdout_rate": holdout_rate,
                       "pack_window": pack_window, "target_name": target_name.strip(),
                       "slug": slug, "user_alias": user_alias.strip(),
                       "packing_policy": f"adjacent whole training sessions per {pack_window} window; independent session_id boundaries; split only oversized sessions",
                       "concurrency": 1, "max_attempts_per_request": 2},
        "corpus": {"messages": len(ordered), "conversations": len(chunks),
                   "senders": dict(sorted(Counter(m["sender"] for m in ordered).items())),
                   "start": ordered[0]["ts"], "end": ordered[-1]["ts"],
                   "train_messages": len(train_messages), "holdout_messages": len(held_messages),
                   "holdout_rate": len(held_messages) / len(ordered)},
        "holdout": {"policy": "whole conversations, deterministic hash ranking, stratified by month",
                    "excluded_from_llm": True, "conversation_indexes": sorted(held),
                    "segments": holdouts, "monthly_selection": selection},
        "segments": segments, "segment_stats": segment_stats,
        "calls": {"persona_per_segment": 1, "memories_per_segment": 1,
                  "extraction": base_calls, "merge": merge_calls,
                  "merge_policy": "balanced binary hierarchy; no truncation; deterministic union fallback",
                  "total_planned": base_calls + merge_calls,
                  "merge_levels": merge_levels},
        "token_estimate": {"train_messages": request_bounds,
                           "extraction_upper_with_reserve": extraction_upper,
                           "merge_upper_with_reserve": merge_upper,
                           "base_total_upper": base_upper,
                           "retry_2x_total_upper": retry_upper,
                           "reserved_tokens_per_request": reserved_tokens},
        "cost": estimate,
    }
    # Hash excludes only the hash itself; no wall-clock timestamp or absolute
    # output directory can perturb checkpoints and reproducible holdouts.
    plan["plan_hash"] = digest(plan)
    return plan


def assert_plan_source(plan: dict[str, Any], messages_path: str | Path) -> None:
    if plan.get("schema_version") != SCHEMA_VERSION or not isinstance(plan.get("plan_hash"), str):
        raise PlanningError("unsupported or incomplete plan")
    unsigned = {key: value for key, value in plan.items() if key != "plan_hash"}
    if digest(unsigned) != plan["plan_hash"]:
        raise PlanningError("plan_hash mismatch; refusing modified plan")
    if file_digest(messages_path) != plan.get("source", {}).get("sha256"):
        raise PlanningError("source SHA-256 differs from plan; refusing checkpoint reuse")
