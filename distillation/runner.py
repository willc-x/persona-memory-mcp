"""Serial, at-most-two-attempt execution with hash-bound checkpoints."""
from __future__ import annotations

import json
import math
import re
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .outputs import claim_union, write_artifacts, write_json
from .planning import (DEFAULT_RESERVED_TOKENS, PlanningError,
                       assert_plan_source, digest, load_messages)
from .prompts import (MAX_OUTPUT_TOKENS, extraction_api_messages,
                      hard_context_upper, merge_api_messages)
from .validation import (ClaimValidationError, response_object,
                         validate_extraction, validate_merge_response,
                         validate_stored_extraction)

NODE_SCHEMA = "distillation.node.v2"
SAFE_NODE_RE = re.compile(r"^[A-Za-z0-9_.-]{1,180}$")


class PipelineError(RuntimeError):
    """Sanitized operational/validation failure (never str(provider_error))."""


class _LazyClient:
    def __init__(self, factory: Callable[[], Any]):
        self.factory = factory
        self.instance: Any = None

    def get(self) -> Any:
        if self.instance is None:
            try:
                candidate = self.factory()
                model = candidate.model
                create = candidate.chat.completions.create
            except Exception:
                raise PipelineError("authorized client initialization failed") from None
            if not isinstance(model, str) or not model or not callable(create):
                raise PipelineError("authorized client lacks .model or .chat.completions.create")
            self.instance = candidate
        return self.instance


def _usage(response: Any) -> dict[str, int | float] | None:
    try:
        usage = response.usage
    except Exception:
        return None
    if usage is None:
        return None
    result: dict[str, int | float] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        try:
            value = usage.get(key) if isinstance(usage, Mapping) else getattr(usage, key)
        except Exception:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            return None
        result[key] = value
    return result


def _content(response: Any) -> Any:
    try:
        return response.choices[0].message.content
    except (AttributeError, IndexError, KeyError, TypeError):
        raise ClaimValidationError("provider response has no choices[0].message.content") from None


class NodeExecutor:
    def __init__(self, root: Path, plan_hash: str, context_tokens: int,
                 client: _LazyClient):
        self.root = root
        self.plan_hash = plan_hash
        self.context_tokens = context_tokens
        self.client = client
        self.attempt_log: list[dict[str, Any]] = []

    def _path(self, group: str, name: str) -> Path:
        if not SAFE_NODE_RE.fullmatch(group) or not SAFE_NODE_RE.fullmatch(name):
            raise PipelineError("unsafe checkpoint node name")
        return self.root / group / (name + ".json")

    def _sealed(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {**payload, "checkpoint_sha256": digest(payload)}

    def _save(self, path: Path, payload: dict[str, Any]) -> None:
        write_json(path, self._sealed(payload))

    def _load(self, path: Path, identity: Mapping[str, Any]) -> dict[str, Any] | None:
        if not path.exists():
            return None
        try:
            with path.open("r", encoding="utf-8") as stream:
                value = json.load(stream)
        except (OSError, json.JSONDecodeError):
            raise PipelineError(f"invalid checkpoint encoding: {path.name}") from None
        if not isinstance(value, dict):
            raise PipelineError(f"invalid checkpoint: {path.name}")
        checksum = value.pop("checkpoint_sha256", None)
        if checksum != digest(value) or value.get("schema_version") != NODE_SCHEMA or value.get("identity") != identity:
            raise PipelineError(f"checkpoint hash/identity mismatch: {path.name}")
        if value.get("status") not in ("completed", "failed") or not isinstance(value.get("attempts"), list):
            raise PipelineError(f"invalid checkpoint status: {path.name}")
        if not isinstance(value.get("model"), str) or not value["model"]:
            raise PipelineError(f"invalid checkpoint model: {path.name}")
        if any(not isinstance(a, dict) or set(a) != {"attempt", "outcome", "usage"}
               or a["outcome"] not in ("started", "completed", "invalid_response", "provider_error")
               for a in value["attempts"]):
            raise PipelineError(f"invalid checkpoint attempt record: {path.name}")
        if [a.get("attempt") for a in value["attempts"]] != list(range(1, len(value["attempts"]) + 1)):
            raise PipelineError(f"nonconsecutive checkpoint attempts: {path.name}")
        for attempt in value["attempts"]:
            usage = attempt["usage"]
            if attempt["outcome"] == "completed":
                if not isinstance(usage, dict) or set(usage) != {"prompt_tokens", "completion_tokens", "total_tokens"}:
                    raise PipelineError(f"invalid completed usage: {path.name}")
                if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0
                       for v in usage.values()):
                    raise PipelineError(f"invalid completed usage: {path.name}")
            elif usage is not None:
                raise PipelineError(f"unexpected usage on unfinished attempt: {path.name}")
        if len(value["attempts"]) > 2:
            raise PipelineError(f"checkpoint exceeds permanent two-attempt limit: {path.name}")
        if value["status"] == "completed" and (
                not value["attempts"] or value["attempts"][-1]["outcome"] != "completed"):
            raise PipelineError(f"completed checkpoint lacks completed attempt: {path.name}")
        if value["status"] == "completed" and value.get("response") is None:
            raise PipelineError(f"completed checkpoint lacks response: {path.name}")
        if value["status"] == "failed" and value.get("response") is not None:
            raise PipelineError(f"failed checkpoint unexpectedly has response: {path.name}")
        return value

    def call(self, *, group: str, name: str, identity: Mapping[str, Any],
             api_messages: list[dict[str, str]], source_message_count: int,
             validator: Callable[[Any], Any],
             stored_validator: Callable[[Any], Any] | None = None) -> tuple[Any, dict[str, Any]]:
        identity = {"plan_hash": self.plan_hash, **identity}
        path = self._path(group, name)
        cached = self._load(path, identity)
        if cached and cached["status"] == "completed":
            try:
                response = (stored_validator or validator)(cached["response"])
            except ClaimValidationError:
                raise PipelineError(f"cached response failed validation: {path.name}") from None
            self.attempt_log.extend(cached["attempts"])
            return response, cached
        attempts = list(cached["attempts"]) if cached else []
        if len(attempts) >= 2:
            self.attempt_log.extend(attempts)
            raise PipelineError(f"node exhausted two attempts: {path.name}")
        upper = hard_context_upper(api_messages, source_message_count)
        if upper > self.context_tokens:
            raise PipelineError(f"request exceeds hard context bound: {path.name}")
        try:
            model = self.client.get().model
        except PipelineError:
            self.attempt_log.extend(attempts)
            raise
        payload: dict[str, Any] = {"schema_version": NODE_SCHEMA, "identity": identity,
                                   "model": model, "status": "failed", "attempts": attempts,
                                   "response": None}
        last_code = "not_run"
        raw_response = None
        while len(attempts) < 2:
            number = len(attempts) + 1
            # Record consumption BEFORE entering the provider.  A crash after a
            # successful provider response but before its local checkpoint thus
            # cannot buy unlimited retries on the next resume.
            started = {"attempt": number, "outcome": "started", "usage": None}
            attempts.append(started)
            self.attempt_log.append(started)
            payload.update({"status": "failed", "attempts": attempts, "response": None})
            self._save(path, payload)
            try:
                # The privacy facade owns provider-specific options (including
                # disabled thinking); callers cannot override its extra_body.
                raw_response = self.client.get().chat.completions.create(
                    model=model, messages=api_messages,
                    response_format={"type": "json_object"},
                    max_tokens=MAX_OUTPUT_TOKENS, temperature=0)
                parsed = response_object(_content(raw_response))
                validated = validator(parsed)
            except ClaimValidationError as exc:
                code = "invalid_response"
                # 旧checkpoint不丢响应，因此不能事后恢复诊断。新响应原文只留在
                # 本机0600文件，不进入日志；用于定位协议/证据拒绝原因而非自动放宽。
                import hashlib
                import os
                content = _content(raw_response) if raw_response is not None else None
                if isinstance(content, str):
                    diagnostic_dir = self.root / 'diagnostics'
                    diagnostic_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
                    diagnostic_dir.chmod(0o700)
                    digest_content = hashlib.sha256(content.encode('utf-8')).hexdigest()
                    diagnostic_path = diagnostic_dir / f'{group}.{name}.attempt_{number}.{digest_content[:16]}.json'
                    temporary = diagnostic_path.with_suffix('.tmp')
                    with temporary.open('w', encoding='utf-8') as stream:
                        stream.write(json.dumps({'response_text': content, 'validation_error': str(exc)}, ensure_ascii=False))
                        stream.flush()
                        os.fsync(stream.fileno())
                    temporary.chmod(0o600)
                    os.replace(temporary, diagnostic_path)
            except Exception:
                # Provider exceptions can contain request text. Do not persist
                # or print str(error); authorization errors are not retried here
                # beyond this request's strict total of two.
                code = "provider_error"
            else:
                started.update({"outcome": "completed", "usage": _usage(raw_response)})
                payload.update({"status": "completed", "attempts": attempts, "response": validated})
                self._save(path, payload)
                return validated, self._sealed(payload)
            started["outcome"] = code
            last_code = code
            payload.update({"status": "failed", "attempts": attempts, "response": None})
            self._save(path, payload)
        raise PipelineError(f"node failed after two attempts ({last_code}): {path.name}")


def _request_views(segment: Mapping[str, Any], by_line: Mapping[int, Mapping[str, Any]]) -> list[dict[str, Any]]:
    views = []
    session_by_line = {line: session["id"] for session in segment.get("sessions", [])
                       for line in session.get("source_lines", [])}
    for line in segment["source_lines"]:
        if line not in by_line or line not in session_by_line:
            raise PipelineError("segment references a missing source_line/session boundary")
        message = by_line[line]
        views.append({"source_line": line, "ts": message["ts"],
                      "sender": message["sender"], "text": message["text"],
                      "session_id": session_by_line[line]})
    if digest(views) != segment["message_sha256"]:
        raise PipelineError("segment message hash differs from plan")
    return views


def _estimated_cost(plan: Mapping[str, Any], segments: Sequence[Mapping[str, Any]]) -> float | None:
    cost = plan["cost"]
    if cost["status"] != "estimated":
        return None
    pin = Decimal(str(cost["input_price_per_million"]))
    pout = Decimal(str(cost["output_price_per_million"]))
    extraction_input = sum(s["token_bounds"]["input_upper"] for s in segments)
    merge_count = max(0, len(segments) - 1)
    merge_input = merge_count * (plan["parameters"]["context_tokens"] - MAX_OUTPUT_TOKENS)
    outputs = (2 * len(segments) + merge_count) * MAX_OUTPUT_TOKENS
    # Authorization covers both attempts for every possible node, not just the
    # happy path. Deterministic no-fit merge fallbacks only reduce this bound.
    value = 2 * (Decimal(extraction_input + merge_input) * pin + Decimal(outputs) * pout) / Decimal(1_000_000)
    return float(value)


def run_pipeline(plan: dict[str, Any], messages_path: str | Path, root: Path,
                 *, limit: int | None = None, full: bool = False,
                 client_factory: Callable[[], Any] | None = None) -> dict[str, Any]:
    if full == (limit is not None):
        raise PipelineError("choose exactly one of --limit N or --full")
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0):
        raise PipelineError("--limit must be a positive integer")
    try:
        assert_plan_source(plan, messages_path)
    except PlanningError as exc:
        raise PipelineError(str(exc)) from None
    segments = list(plan["segments"])
    selected = segments if full else segments[:limit]
    if not selected:
        raise PipelineError("no selected training segments")
    run_mode = "distill_full" if full else "distill_trial"
    root.mkdir(parents=True, exist_ok=True)
    messages = load_messages(messages_path)
    by_line = {message["source_line"]: message for message in messages}
    holdout_lines = {line for segment in plan["holdout"]["segments"] for line in segment["source_lines"]}
    for segment in selected:
        if holdout_lines.intersection(segment["source_lines"]):
            raise PipelineError("plan attempted to distill holdout evidence")
    if client_factory is None:
        estimated = _estimated_cost(plan, selected)
        def client_factory() -> Any:
            try:
                from adapters.privacy import get_allowed_client
                return get_allowed_client(purpose=run_mode, estimated_cost=estimated,
                                          plan_hash=plan["plan_hash"])
            except Exception:
                raise PipelineError("privacy authorization failed") from None
    lazy = _LazyClient(client_factory)
    executor = NodeExecutor(root, plan["plan_hash"], plan["parameters"]["context_tokens"], lazy)
    completed: dict[str, list[dict[str, Any]]] = {}
    incremental_claims: dict[str, dict[str, Any]] = {}
    checkpoint_models: set[str] = set()
    try:
        for segment in selected:
            views = _request_views(segment, by_line)
            allowed = set(segment["source_lines"])
            claims: list[dict[str, Any]] = []
            for kind in ("persona", "memories"):
                def validator(raw: Any, *, allowed=allowed) -> Any:
                    return {"claims": validate_extraction(raw, by_line, allowed)}
                def stored_validator(raw: Any, *, allowed=allowed) -> Any:
                    return {"claims": validate_stored_extraction(raw, by_line, allowed)}
                response, node_checkpoint = executor.call(
                    group="segments", name=f"{segment['id']}.{kind}",
                    identity={"segment_id": segment["id"], "kind": kind,
                              "message_sha256": segment["message_sha256"]},
                    api_messages=extraction_api_messages(kind, segment["id"], views),
                    source_message_count=len(views), validator=validator,
                    stored_validator=stored_validator)
                checkpoint_models.add(node_checkpoint["model"])
                for claim in response["claims"]:
                    incremental_claims[claim["id"]] = claim
                claims = claim_union(claims + response["claims"])
                # Every locally validated response enters the durable library,
                # including persona when the later memory node fails.
                write_artifacts(root, plan, status="incomplete", run_mode=run_mode,
                                claims=claim_union(list(incremental_claims.values())), messages=by_line,
                                selected_segment_ids=[segment["id"]],
                                attempts=executor.attempt_log)
            claims = claim_union(claims)
            done_identity = {"plan_hash": plan["plan_hash"], "segment_id": segment["id"],
                             "message_sha256": segment["message_sha256"]}
            done_path = root / "segments" / f"{segment['id']}.done.json"
            done_payload = {"schema_version": "distillation.segment_done.v1",
                            "identity": done_identity, "status": "completed", "claims": claims}
            sealed = {**done_payload, "checkpoint_sha256": digest(done_payload)}
            if done_path.exists():
                try:
                    existing = json.loads(done_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    raise PipelineError("invalid segment done checkpoint") from None
                if existing != sealed:
                    raise PipelineError(
                        f"segment done hash/content mismatch: {segment['id']}")
            else:
                write_json(done_path, sealed)
            completed[segment["id"]] = claims
            # Durable incremental claim library: a later failed node cannot
            # erase claims already validated and checkpointed.
            write_artifacts(root, plan, status="incomplete", run_mode=run_mode,
                            claims=claim_union(list(incremental_claims.values())),
                            messages=by_line, selected_segment_ids=list(completed),
                            attempts=executor.attempt_log)
        root_claims = claim_union([c for values in completed.values() for c in values])
        nodes = [{"claims": completed[s["id"]], "summary_views": [],
                  "segment_ids": [s["id"]]} for s in selected]
        merge_calls = 0
        fallbacks = 0
        level = 0
        while len(nodes) > 1:
            next_nodes = []
            for batch_index, offset in enumerate(range(0, len(nodes), 2)):
                pair = nodes[offset:offset + 2]
                if len(pair) == 1:
                    next_nodes.append(pair[0])
                    continue
                combined_claims = claim_union(pair[0]["claims"] + pair[1]["claims"])
                prior_views = pair[0]["summary_views"] + pair[1]["summary_views"]
                payload = {"claims": combined_claims, "prior_summaries": prior_views}
                api_messages = merge_api_messages(payload)
                if hard_context_upper(api_messages, 0) > plan["parameters"]["context_tokens"]:
                    # Carry BOTH children upward. Never truncate claims or
                    # replace an over-budget child with a lossy summary.
                    next_nodes.extend(pair)
                    fallbacks += 1
                    continue
                allowed_ids = {claim["id"] for claim in combined_claims}
                def merge_validator(raw: Any, *, allowed_ids=allowed_ids) -> Any:
                    return validate_merge_response(raw, allowed_ids)
                try:
                    response, _ = executor.call(
                        group="merges", name=f"level_{level:04d}_batch_{batch_index:05d}",
                        identity={"level": level, "batch": batch_index,
                                  "input_sha256": digest(payload)},
                        api_messages=api_messages, source_message_count=0,
                        validator=merge_validator)
                except PipelineError as exc:
                    # 摘要合并是可选的压缩层；任何已保存原文 claim 的合并失败
                    # 都保留两个完整子节点继续向上，不得丢失或改写 claim。
                    if "provider_error" not in str(exc) and "invalid_response" not in str(exc):
                        raise
                    next_nodes.extend(pair)
                    fallbacks += 1
                    continue
                merge_calls += 1
                next_nodes.append({"claims": combined_claims,
                                   "summary_views": prior_views + [response],
                                   "segment_ids": pair[0]["segment_ids"] + pair[1]["segment_ids"]})
            if len(next_nodes) == len(nodes):
                # No pair fits. A deterministic root is lossless; forcing another
                # identical level would loop forever without making progress.
                break
            nodes = next_nodes
            level += 1
        final_claims = claim_union([c for node in nodes for c in node["claims"]])
        if {c["id"] for c in final_claims} != {c["id"] for c in root_claims}:
            raise PipelineError("merge lost or invented claims")
        summary_views = [view for node in nodes for view in node["summary_views"]]
        audit = {"planned_calls_upper": max(0, len(selected) - 1),
                 "calls": merge_calls, "no_fit_pair_fallbacks": fallbacks,
                 "final_unmerged_nodes": len(nodes), "claim_union_verified": True,
                 "truncated_claims": 0, "levels_attempted": level + 1}
        status = "complete" if full and len(completed) == len(segments) else "trial"
        result = write_artifacts(root, plan, status=status, run_mode=run_mode,
            claims=final_claims, messages=by_line, summary_views=summary_views,
            merge_audit=audit, selected_segment_ids=list(completed),
            models=sorted(checkpoint_models | ({lazy.instance.model} if lazy.instance is not None else set())),
            attempts=executor.attempt_log)
        write_json(root / "run_state.json", {"schema_version": "distillation.run_state.v1",
            "status": status, "run_mode": run_mode, "plan_hash": plan["plan_hash"],
            "selected_segment_ids": list(completed), "merge": audit})
        return result
    except PipelineError:
        # Preserve all already validated claims but make the incomplete status
        # explicit. No catch-all converts missing/invalid nodes into success.
        write_artifacts(root, plan, status="incomplete", run_mode=run_mode,
            claims=claim_union(list(incremental_claims.values())),
            messages=by_line, selected_segment_ids=list(completed),
            attempts=executor.attempt_log,
            models=sorted(checkpoint_models | ({lazy.instance.model} if lazy.instance is not None else set())))
        write_json(root / "run_state.json", {"schema_version": "distillation.run_state.v1",
            "status": "incomplete", "run_mode": run_mode,
            "plan_hash": plan["plan_hash"], "selected_segment_ids": list(completed),
            "merge_performed": False})
        raise


def write_local_baseline(plan: dict[str, Any], messages_path: str | Path,
                         root: Path) -> dict[str, Any]:
    try:
        assert_plan_source(plan, messages_path)
    except PlanningError as exc:
        raise PipelineError(str(exc)) from None
    messages = load_messages(messages_path)
    by_line = {m["source_line"]: m for m in messages}
    result = write_artifacts(root, plan, status="local_baseline", run_mode="local_baseline",
        claims=[], messages=by_line, selected_segment_ids=[],
        merge_audit={"calls": 0, "fallbacks": 0, "claim_union_verified": True})
    write_json(root / "run_state.json", {"schema_version": "distillation.run_state.v1",
        "status": "local_baseline", "run_mode": "local_baseline",
        "plan_hash": plan["plan_hash"], "api_calls": 0,
        "note": "offline skeleton only; no distillation or quality evaluation performed"})
    return result
