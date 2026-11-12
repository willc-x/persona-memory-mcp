#!/usr/bin/env python3
"""从本地已保存的模型响应恢复一个精确节点，不重发聊天内容。

恢复只使用原计划 ``source`` 指向的本地消息文件，并要求该文件仍位于
``DATA_ROOT`` 内且 SHA-256 与计划一致。恢复不调用模型，只把已完成的 usage
账本事件与唯一诊断绑定到已有失败节点，保留每节点最多两次请求的上限。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from adapters.upstream import DATA_ROOT
from distillation.outputs import write_json
from distillation.planning import assert_plan_source, digest, load_messages
from distillation.runner import NODE_SCHEMA
from distillation.validation import (ClaimValidationError, response_object,
                                     validate_claim, validate_extraction)

MAX_ATTEMPTS = 2
SEGMENT_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}\Z")
KINDS = {"persona", "memories"}


def _inside(path: Path, data_root: Path, label: str, *, directory: bool = False) -> Path:
    """把输入限制到 DATA_ROOT 内，拒绝根目录、越界和错误类型。"""
    if not isinstance(path, Path):
        path = Path(path)
    if not path.is_absolute():
        path = data_root / path
    candidate = path.expanduser().resolve()
    if candidate == data_root or not candidate.is_relative_to(data_root):
        raise ValueError(f"{label} 必须位于 DATA_ROOT 内且不能是根目录")
    if directory:
        if not candidate.is_dir():
            raise ValueError(f"{label} 不是目录")
    elif not candidate.is_file():
        raise ValueError(f"{label} 不是普通文件")
    return candidate


def _complete_claim_prefix(raw: str) -> list[dict]:
    """只取截断响应中语法完整且已闭合的 claims；不补全、不改写任何claim。"""
    marker = '"claims"'
    start = raw.find(marker)
    if start < 0:
        return []
    index = raw.find("[", start + len(marker))
    if index < 0:
        return []
    decoder = json.JSONDecoder()
    claims: list[dict] = []
    index += 1
    while index < len(raw):
        while index < len(raw) and raw[index] in " \t\r\n":
            index += 1
        if index >= len(raw) or raw[index] in "]}":
            break
        try:
            value, end = decoder.raw_decode(raw, index)
        except (json.JSONDecodeError, RecursionError):
            break
        if not isinstance(value, dict):
            break
        claims.append(value)
        index = end
        while index < len(raw) and raw[index] in " \t\r\n":
            index += 1
        if index >= len(raw) or raw[index] != ",":
            break
        index += 1
    return claims


def _load_json_inside(path: Path, data_root: Path, label: str) -> tuple[Path, dict]:
    resolved = _inside(path, data_root, label)
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} 不是可读 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} 必须是 JSON 对象")
    return resolved, value


def recover(workdir: Path, segment_id: str, kind: str, attempt: int, usage_event: int,
            grant_id: str, *, consent_path: Path | None = None,
            usage_path: Path | None = None, salvage: bool = False,
            data_root: Path | None = None) -> dict:
    """重建一个失败节点；所有输入输出都受 DATA_ROOT 边界约束。"""
    data_root = Path(data_root or DATA_ROOT).expanduser().resolve()
    if data_root == Path("/") or not data_root.is_dir():
        raise ValueError("DATA_ROOT 不是有效目录")
    if not isinstance(segment_id, str) or not SEGMENT_ID_RE.fullmatch(segment_id):
        raise ValueError("segment_id 格式不合法")
    if kind not in KINDS:
        raise ValueError("kind 必须是 persona 或 memories")
    if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt not in (1, MAX_ATTEMPTS):
        raise ValueError("attempt 只能为 1 或 2")
    if isinstance(usage_event, bool) or not isinstance(usage_event, int) or usage_event < 1:
        raise ValueError("usage_event 必须是正整数")
    if not isinstance(grant_id, str) or not grant_id.strip():
        raise ValueError("grant_id 不能为空")

    workdir = _inside(workdir, data_root, "workdir", directory=True)
    _, plan = _load_json_inside(workdir / "plan.json", data_root, "plan.json")
    consent_path = consent_path or data_root / "config" / "privacy-consent.json"
    usage_path = usage_path or data_root / "runtime" / "api-usage.json"
    _, consent = _load_json_inside(consent_path, data_root, "consent")
    _, usage = _load_json_inside(usage_path, data_root, "usage")

    grants = [g for g in consent.get("authorizations", []) if isinstance(g, dict)
              and g.get("id") == grant_id]
    if len(grants) != 1:
        raise ValueError("恢复需要唯一的全量蒸馏授权")
    grant = grants[0]
    purposes = grant.get("purposes", [])
    if isinstance(purposes, str):
        purposes = [purposes]
    if grant.get("plan_hash") != plan.get("plan_hash") or "distill_full" not in purposes:
        raise ValueError("恢复只能使用绑定同一计划与全量用途的显式授权")

    events = [e for e in usage.get("events", []) if isinstance(e, dict)
              and e.get("id") == usage_event]
    if len(events) != 1:
        raise ValueError("usage 事件不存在或不唯一")
    event = events[0]
    if event.get("grant_id") != grant_id or event.get("status") != "completed":
        raise ValueError("usage 事件不属于该授权或不是已完成请求")

    segments = [s for s in plan.get("segments", []) if isinstance(s, dict)
                and s.get("id") == segment_id]
    if len(segments) != 1:
        raise ValueError("segment 不属于该计划")
    segment = segments[0]

    # 消息文件必须由计划 source 推导，不能使用脚本内置路径。
    source_path = plan.get("source", {}).get("path")
    if not isinstance(source_path, str) or not source_path.strip():
        raise ValueError("计划缺少 source.path")
    source = _inside(Path(source_path), data_root, "计划消息文件")
    try:
        assert_plan_source(plan, source)
        messages = {m["source_line"]: m for m in load_messages(source)}
    except Exception as exc:
        raise ValueError("计划 source 与本地消息文件不一致") from exc

    checkpoint = workdir / "segments" / f"{segment_id}.{kind}.json"
    if checkpoint.is_symlink():
        raise ValueError("节点 checkpoint 不能是符号链接")
    checkpoint = _inside(checkpoint, data_root, "节点 checkpoint")
    _, old = _load_json_inside(checkpoint, data_root, "节点 checkpoint")
    if old.get("status") == "completed" or not isinstance(old.get("identity"), dict):
        raise ValueError("节点已完成或 schema 不合规，拒绝覆盖")
    identity = old["identity"]
    if (identity.get("plan_hash") != plan.get("plan_hash")
            or identity.get("segment_id") != segment_id
            or identity.get("kind") != kind):
        raise ValueError("旧节点身份与计划不一致")

    candidates = []
    diagnostics = workdir / "diagnostics"
    if diagnostics.is_symlink() or not diagnostics.is_dir():
        raise ValueError("诊断目录不存在或不能是符号链接")
    for path in diagnostics.glob(f"segments.{segment_id}.{kind}.attempt_{attempt}.*.json"):
        try:
            candidates.append(_inside(path, data_root, "诊断文件"))
        except ValueError:
            continue
    candidates = sorted(set(candidates))
    if len(candidates) != 1:
        raise ValueError("指定的诊断 attempt 不存在或不唯一")
    saved = json.loads(candidates[0].read_text(encoding="utf-8"))
    raw = saved.get("response_text")
    if not isinstance(raw, str):
        raise ValueError("诊断缺少 response_text")
    if hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16] not in candidates[0].name:
        raise ValueError("诊断内容与文件名哈希不符")

    allowed = set(segment.get("source_lines", []))
    rejected = 0
    wrapper_salvaged = False
    if salvage:
        # 只逐条接收通过同一 canonical 校验的 claim；原始诊断保留用于审计。
        # 网关截断时只解析已闭合的独立对象，不补齐任何字段。
        try:
            parsed_claims = response_object(raw).get("claims", [])
        except ClaimValidationError:
            parsed_claims = _complete_claim_prefix(raw)
            if not parsed_claims:
                raise
            wrapper_salvaged = True
        claims = []
        seen = set()
        for candidate in parsed_claims:
            try:
                claim = validate_claim(
                    candidate,
                    {line: messages[line] for line in allowed if line in messages},
                )
            except ClaimValidationError:
                rejected += 1
                continue
            if claim["id"] in seen:
                rejected += 1
                continue
            seen.add(claim["id"])
            claims.append(claim)
        if not claims:
            raise ValueError("诊断响应没有任何逐条可核验的 claim")
    else:
        claims = validate_extraction(response_object(raw), messages, allowed)

    old_attempts = old.get("attempts")
    if not isinstance(old_attempts, list) or not old_attempts:
        raise ValueError("旧节点缺少 attempts")
    indexes = []
    for item in old_attempts:
        if not isinstance(item, dict) or isinstance(item.get("attempt"), bool):
            raise ValueError("旧节点 attempts schema 不合规")
        index = item.get("attempt")
        if not isinstance(index, int) or index not in (1, MAX_ATTEMPTS):
            raise ValueError("旧节点 attempt 超出两次上限")
        indexes.append(index)
    if len(indexes) != len(set(indexes)) or indexes != sorted(indexes):
        raise ValueError("旧节点 attempts 顺序或编号不合规")
    if indexes[-1] != attempt:
        raise ValueError("只能恢复最后一次失败 attempt")

    try:
        prompt_tokens = int(event["prompt_tokens"])
        completion_tokens = int(event["completion_tokens"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("usage 事件缺少有效 token 计数") from exc
    if prompt_tokens < 0 or completion_tokens < 0:
        raise ValueError("usage 事件 token 计数不能为负")
    token = {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
             "total_tokens": prompt_tokens + completion_tokens}
    attempts = []
    for item in old_attempts:
        if item["attempt"] == attempt:
            attempts.append({"attempt": attempt, "outcome": "completed", "usage": token})
        else:
            attempts.append(item)
    if len(attempts) > MAX_ATTEMPTS:
        raise ValueError("恢复后 attempt 数超过两次上限")

    optional = ("time_semantics", "time_provenance", "model_time_interval_before_canonicalization")
    response = {"claims": [
        {"id": c["id"], "category": c["category"], "statement": c["statement"],
         "evidence": c["evidence"], "time_start": c["time_start"],
         "time_end": c["time_end"], "confidence": c["confidence"],
         **{key: c[key] for key in optional if key in c}}
        for c in claims
    ]}
    value = {"schema_version": NODE_SCHEMA, "identity": identity, "model": old.get("model"),
             "status": "completed", "attempts": attempts, "response": response,
             "migrated_from_diagnostics": candidates[0].name, "provider_recall": False,
             "validation_mode": "per_claim_salvage" if salvage else "whole_response",
             "response_salvage": "complete_claim_prefix" if wrapper_salvaged else "none",
             "rejected_claim_count": rejected}
    value["checkpoint_sha256"] = digest(value)
    write_json(checkpoint, value)
    return {"node": str(checkpoint), "claims": len(claims), "network_calls": 0,
            "provider_recall": False, "validation_mode": value["validation_mode"],
            "rejected_claims": rejected, "usage_event": usage_event,
            "diagnostic": candidates[0].name}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--segment-id", required=True)
    parser.add_argument("--kind", choices=sorted(KINDS), required=True)
    parser.add_argument("--attempt", type=int, required=True)
    parser.add_argument("--usage-event", type=int, required=True)
    parser.add_argument("--grant", required=True)
    parser.add_argument("--consent", type=Path)
    parser.add_argument("--usage", type=Path)
    parser.add_argument("--salvage", action="store_true",
                        help="仅逐条接收可核验 claims；拒绝数写入节点审计")
    args = parser.parse_args()
    try:
        result = recover(args.workdir, args.segment_id, args.kind, args.attempt,
                         args.usage_event, args.grant, consent_path=args.consent,
                         usage_path=args.usage, salvage=args.salvage)
    except Exception as exc:
        print(json.dumps({"status": "error", "error": type(exc).__name__, "detail": str(exc)},
                         ensure_ascii=False), file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())