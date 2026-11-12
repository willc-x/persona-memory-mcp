"""Lossless claim/timeline views and self-describing final artifacts."""
from __future__ import annotations

import json
import os
import re
import shlex
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .planning import canonical_bytes, digest

ARTIFACT_FILES = ("persona.md", "memories.md", "user_profile.md", "SKILL.md",
                  "claims.json", "timeline.json", "events.json", "quality.json",
                  "meta.json", "merged.json", "persona_core.json", "persona_core.md")
STATUS_LABELS = {
    "complete": "complete：计划内全部训练段完成；holdout未参与蒸馏",
    "trial": "trial：仅指定训练段完成；不是全量蒸馏",
    "local_baseline": "local_baseline：仅本地骨架，未调用模型、未进行蒸馏",
    "incomplete": "incomplete：存在失败或缺段；未生成成功合并结果",
}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.parent.chmod(0o700)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        temporary.chmod(0o600)
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8"))
        stream.write(b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _identity(plan: Mapping[str, Any]) -> tuple[str, str]:
    """读取计划顶层的展示身份；兼容早期计划中的 parameters 和通用环境变量。"""
    parameters = plan.get("parameters")
    if not isinstance(parameters, Mapping):
        parameters = {}
    name = plan.get("target_name", parameters.get("target_name"))
    slug = plan.get("slug", parameters.get("slug"))
    if name is None:
        name = os.environ.get("CHAT_TARGET_NAME", "目标人物")
    if slug is None:
        slug = os.environ.get("CHAT_DISTILL_SLUG", "persona")
    if (not isinstance(name, str) or not name.strip() or len(name) > 80
            or any(ch in name for ch in "\r\n")):
        raise ValueError("plan target_name must be one non-empty line (max 80 characters)")
    if not isinstance(slug, str) or not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", slug):
        raise ValueError("plan slug must be lowercase [a-z0-9][a-z0-9_-]* and at most 64 characters")
    return name.strip(), slug


def claim_union(claims: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for claim in claims:
        previous = by_id.get(claim["id"])
        if previous is not None and canonical_bytes(previous) != canonical_bytes(claim):
            raise ValueError("same claim id has different content")
        by_id[claim["id"]] = claim
    return sorted(by_id.values(), key=lambda c: (
        min((e["source_line"] for e in c["evidence"]), default=2**63), c["id"]))


def claim_time(claim: Mapping[str, Any], messages: Mapping[int, Mapping[str, Any]], start: bool) -> str | None:
    explicit = claim.get("time_start" if start else "time_end")
    if explicit:
        return str(explicit)
    times = [str(messages[e["source_line"]]["ts"]) for e in claim.get("evidence", [])
             if e["source_line"] in messages]
    if not times:
        return None
    return min(times) if start else max(times)


def make_events(claims: Sequence[dict[str, Any]], messages: Mapping[int, Mapping[str, Any]],
                status: str, target_slug: str = "persona") -> dict[str, Any]:
    events = []
    for claim in claims:
        events.append({
            "id": "event_" + claim["id"].removeprefix("claim_"),
            "claim_ids": [claim["id"]], "category": claim["category"],
            "time_start": claim_time(claim, messages, True),
            "time_end": claim_time(claim, messages, False),
            "summary": claim["statement"], "confidence": claim["confidence"],
            "evidence": claim["evidence"],
            "epistemic_note": "推断，不是事实" if claim["category"] == "inference" else
                              "未知/证据不足，不是事实" if claim["category"] == "unknown" else "有精确原文锚点",
        })
    events.sort(key=lambda e: (e["time_start"] is None, e["time_start"] or "", e["id"]))
    return {"schema_version": "distillation.events.v1", "status": status,
            "slug": target_slug, "events": events}


def make_timeline(claims: Sequence[dict[str, Any]], messages: Mapping[int, Mapping[str, Any]],
                  status: str, target_slug: str = "persona") -> dict[str, Any]:
    groups: dict[str, list[tuple[dict[str, Any], str | None, str | None]]] = defaultdict(list)
    undated = []
    for claim in claims:
        start, end = claim_time(claim, messages, True), claim_time(claim, messages, False)
        if start is None:
            undated.append(claim["id"])
        else:
            groups[start[:7]].append((claim, start, end))
    eras = []
    for month in sorted(groups):
        items = groups[month]
        starts = [start for _, start, _ in items if start]
        ends = [end or start for _, start, end in items if end or start]
        eras.append({"id": "era_" + month.replace("-", "_"),
                     "start": min(starts), "end": max(ends), "label": month,
                     "claim_ids": sorted({claim["id"] for claim, _, _ in items})})
    return {"schema_version": "distillation.timeline.v1", "status": status,
            "slug": target_slug, "eras": eras, "undated_claim_ids": sorted(undated)}


def _claim_view(claim: Mapping[str, Any], inference: bool = False) -> dict[str, Any]:
    return {"statement": claim["statement"], "claim_id": claim["id"],
            "confidence": claim["confidence"], "evidence": claim["evidence"],
            "category": claim["category"],
            "epistemic_status": "inference" if inference else "evidence_checked"}


def make_merged(claims: Sequence[dict[str, Any]], events: Mapping[str, Any],
                timeline: Mapping[str, Any], summary_views: Sequence[dict[str, Any]],
                status: str, run_mode: str, plan_hash: str, target_name: str = "目标人物",
                target_slug: str = "persona") -> dict[str, Any]:
    categories: dict[str, list[dict[str, Any]]] = {key: [] for key in
        ("fact", "style", "emotion_expression", "relationship", "inference", "unknown")}
    for claim in claims:
        if claim["category"] in categories:
            categories[claim["category"]].append(claim)
    # Only direct fact/style claims become candidate core traits.  Even these
    # remain evidence-linked candidates, not unconditional permanent truths.
    core_claims = [c for c in categories["fact"] + categories["style"]
                   if c["time_start"] is not None and c["time_end"] is not None
                   and any(e["sender"] == "B" for e in c["evidence"])]
    nodes = []
    for event in events["events"]:
        claim = next(c for c in claims if c["id"] == event["claim_ids"][0])
        nodes.append({"id": event["id"], "event": event["summary"], "ts": event["time_start"],
                      "time_end": event["time_end"], "emotion": 0,
                      "importance": claim["confidence"],
                      "evidence": json.dumps(event["evidence"], ensure_ascii=False),
                      "claim_ids": event["claim_ids"], "category": claim["category"],
                      "evidence_level": "impression" if claim["category"] in ("inference", "unknown") else "verbatim"})
    persona_eras = [{"id": era["id"], "name": era["label"], "start": era["start"],
                     "end": era["end"], "claim_ids": era["claim_ids"],
                     "summary": "该时段内已核验claim的集合；不自动推断稳定人格。"}
                    for era in timeline["eras"]]
    core_ids = [claim["id"] for claim in core_claims
                if any(e["sender"] == "B" for e in claim["evidence"])]
    return {
        "schema_version": "suhui.merged.v3", "template_version": 3,
        "name": target_name, "slug": target_slug, "status": status, "run_mode": run_mode,
        "plan_hash": plan_hash, "generation": "local_baseline" if status == "local_baseline" else "api",
        "coverage": "local_baseline" if status == "local_baseline" else "complete_training_segments" if status == "complete" else "trial",
        "summary": STATUS_LABELS[status] + f"；{len(claims)}条证据核验claim。",
        "persona": {
            "core_traits": [{"trait": c["statement"], "evidence_level": "verbatim",
                             "claim_id": c["id"], "confidence": c["confidence"],
                             "candidate_only": True} for c in core_claims],
            "eras": persona_eras,
            "core": {"stable_traits": [], "claim_ids": core_ids, "candidate_only": True,
                     "note": "不声称无条件稳定真相；运行时应按证据与截止时间过滤。"},
            "evolution": [],
            "expression": {"claims": [_claim_view(c) for c in categories["style"]],
                           "catchphrases": [], "sentence_length": {}, "punctuation": [],
                           "emoji_pattern": {}, "classic_quotes": []},
            "emotion": {"expressions": [_claim_view(c) for c in categories["emotion_expression"]],
                        "triggers": [], "day_night": []},
            "emotion_decoder": [],
            "relationship": {"claims": [_claim_view(c) for c in categories["relationship"]]},
            "platform_style": [], "values": [], "decision_weights": [],
            "knowledge_boundary": [_claim_view(c) for c in categories["unknown"]],
            "language_fingerprint": {},
            "speculative": [{"inference": c["statement"], "claim_id": c["id"],
                             "evidence": c["evidence"], "confidence": c["confidence"],
                             "evidence_level": "impression"} for c in categories["inference"]],
            "first_mes": "", "alternate_greetings": {},
        },
        "memories": {
            "timeline": [{"stage": era["label"], "start": era["start"], "end": era["end"],
                          "temperature": "unknown", "claim_ids": era["claim_ids"],
                          "events": [e["summary"] for e in events["events"]
                                     if e["claim_ids"][0] in era["claim_ids"]]}
                         for era in timeline["eras"]],
            "nodes": nodes,
            "daily_patterns": [{"pattern": c["statement"], "evidence_level": "verbatim",
                                "claim_id": c["id"]} for c in categories["style"]],
            "unfinished": [],
        },
        "entity_clusters": [], "conflicts": [], "corrections": [],
        "user_profile": {"claim_ids": [c["id"] for c in claims
                                        if any(e["sender"] == "A" for e in c["evidence"])],
                          "evidence_level": "impression",
                          "note": "A相关claim候选，不自动上升为稳定用户画像。"},
        "summary_views": list(summary_views),
        # Deliberately no corpus: holdout and private raw metadata are not
        # copied into the runtime view merely to satisfy an optional strict flag.
    }


def _status_header(status: str, plan_hash: str) -> str:
    return f"> 状态：**{STATUS_LABELS[status]}**  \n> plan_hash：`{plan_hash}`\n\n"


def _markdown(claims: Sequence[dict[str, Any]], status: str, plan_hash: str,
              summary_views: Sequence[dict[str, Any]], section: str,
              target_name: str = "目标人物") -> str:
    lines = [f"# {target_name} · {section}", "", _status_header(status, plan_hash)]
    if summary_views:
        lines += ["## 层次摘要（非独立事实，须回溯claim）", ""]
        for view in summary_views:
            lines.append(f"- {view.get('summary', '')}（依据：{', '.join(view.get('claim_ids', [])) or '无'}）")
        lines.append("")
    labels = {"fact": "可核验事实", "style": "表达风格候选",
              "emotion_expression": "直接情绪表达", "relationship": "明确关系陈述",
              "inference": "隔离推断（不是事实）", "unknown": "未知/证据不足（不是事实）"}
    for category, label in labels.items():
        selected = [c for c in claims if c["category"] == category]
        if not selected:
            continue
        lines += [f"## {label}", ""]
        for claim in selected:
            anchors = ", ".join(f"{e['source_line']}:{e['sender']}" for e in claim["evidence"]) or "无证据"
            lines.append(f"- {claim['statement']} — `{claim['id']}`，置信度{claim['confidence']}，锚点 {anchors}")
        lines.append("")
    if not claims:
        lines += ["无已蒸馏claim。此文件不能作为全量人物画像。", ""]
    return "\n".join(lines)


def write_artifacts(root: Path, plan: Mapping[str, Any], *, status: str,
                    run_mode: str, claims: Sequence[dict[str, Any]],
                    messages: Mapping[int, Mapping[str, Any]],
                    summary_views: Sequence[dict[str, Any]] = (),
                    merge_audit: Mapping[str, Any] | None = None,
                    selected_segment_ids: Sequence[str] = (),
                    models: Sequence[str] = (), attempts: Sequence[Mapping[str, Any]] = ()) -> dict[str, Any]:
    if status not in STATUS_LABELS:
        raise ValueError("invalid artifact status")
    claims = claim_union(list(claims))
    target_name, target_slug = _identity(plan)
    events = make_events(claims, messages, status, target_slug)
    timeline = make_timeline(claims, messages, status, target_slug)
    merged = make_merged(claims, events, timeline, summary_views, status, run_mode,
                         str(plan["plan_hash"]), target_name, target_slug)
    core_ids = merged["persona"]["core"]["claim_ids"]
    core = {"schema_version": "distillation.persona_core.v1", "status": status,
            "slug": target_slug, "claim_ids": core_ids, "candidate_only": True,
            "note": "事实/风格证据候选，不是无条件稳定真相；inference/unknown不在core。"}
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    usage_known = False
    for attempt in attempts:
        item = attempt.get("usage")
        if isinstance(item, Mapping):
            usage_known = True
            for key in usage:
                value = item.get(key)
                if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0:
                    usage[key] += value
    quality = {
        "schema_version": "distillation.quality.v1", "status": status,
        "run_mode": run_mode, "plan_hash": plan["plan_hash"],
        "distillation_complete": status == "complete",
        "planned_training_segments": len(plan["segments"]),
        "selected_segments": len(selected_segment_ids),
        "claims": len(claims), "claims_by_category": dict(sorted(Counter(c["category"] for c in claims).items())),
        "evidence_validation": "exact quote substring + source_line + sender; whole response rejected on any invalid claim",
        "holdout": {"excluded_from_requests": True,
                    "conversations": len(plan["holdout"]["conversation_indexes"]),
                    "segments": len(plan["holdout"]["segments"]),
                    "evaluation_status": "pending_not_run"},
        "merge": dict(merge_audit or {"calls": 0, "fallbacks": 0,
                                      "policy": "deterministic union; no truncation"}),
        "request_attempts": len(attempts), "actual_usage": usage if usage_known else None,
        "actual_cost_rmb": None,
        "notes": ["usage/cost depend on provider response; estimates are in plan.json",
                  "holdout has not been sent or evaluated; no quality score is claimed"],
    }
    if usage_known and plan["cost"]["status"] == "estimated":
        # Decimal avoids binary-money rounding in reports.
        from decimal import Decimal
        pin = Decimal(str(plan["cost"]["input_price_per_million"]))
        pout = Decimal(str(plan["cost"]["output_price_per_million"]))
        quality["actual_cost_rmb"] = str((Decimal(usage["prompt_tokens"]) * pin
                                        + Decimal(usage["completion_tokens"]) * pout) / Decimal(1_000_000))
    meta = {
        "schema_version": "distillation.meta.v1", "name": target_name, "slug": target_slug,
        "status": status, "run_mode": run_mode, "local_baseline": status == "local_baseline",
        "distillation_complete": status == "complete", "coverage": merged["coverage"],
        "plan_hash": plan["plan_hash"], "source_sha256": plan["source"]["sha256"],
        "selected_segment_ids": list(selected_segment_ids),
        "completed_segments": len(selected_segment_ids) if status in ("complete", "trial") else 0,
        "planned_training_segments": len(plan["segments"]),
        "holdout_excluded": True, "models": sorted(set(models)),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "artifact_files": list(ARTIFACT_FILES),
        "external_builder_hint": ("合并结构可由外部工具消费：传入 merged.json、显示名 "
                                  + shlex.quote(target_name) + " 与 slug " + shlex.quote(target_slug)
                                  + "；外部工具不属于本仓库"),
        "corpus_note": "merged intentionally omits raw corpus/holdout; pass an explicitly authorized corpus to upstream build if needed",
    }
    write_json(root / "claims.json", {"schema_version": "distillation.claims.v1",
        "status": status, "run_mode": run_mode, "slug": target_slug,
        "plan_hash": plan["plan_hash"], "claim_count": len(claims), "claims": claims})
    write_json(root / "timeline.json", timeline)
    write_json(root / "events.json", events)
    write_json(root / "quality.json", quality)
    write_json(root / "meta.json", meta)
    write_json(root / "merged.json", merged)
    write_json(root / "persona_core.json", core)
    def private_write(name: str, content: str) -> None:
        path = root / name
        path.touch(mode=0o600, exist_ok=True)
        path.chmod(0o600)
        path.write_text(content, encoding='utf-8')

    private_write("persona.md", _markdown(claims, status, str(plan["plan_hash"]), summary_views, "人格与表达", target_name))
    private_write("memories.md", _markdown(claims, status, str(plan["plan_hash"]), (), "记忆节点", target_name))
    a_claims = [c for c in claims if any(e["sender"] == "A" for e in c["evidence"])]
    private_write("user_profile.md", _markdown(a_claims, status, str(plan["plan_hash"]), (), "A（我）相关候选", target_name))
    private_write("persona_core.md", _status_header(status, str(plan["plan_hash"]))
        + "# persona_core候选\n\n仅包含fact/style证据候选；不表示无条件稳定真相。\n\n"
        + ("\n".join(f"- {cid}" for cid in core_ids) or "无") + "\n")
    skill = f"""---
name: {target_slug}
description: "{target_name}证据约束记忆镜像 · {status}"
---
# {target_name}（{target_slug}）

{_status_header(status, str(plan['plan_hash']))}
- A=我，B={target_name}。
- 仅使用claims.json中当前status允许、且符合时间过滤的证据。
- inference必须说成推断；unknown不得补全；表情/慢回不得推断心理定论。
- persona_core只是fact/style候选，不是永久人格真相。
- holdout未参与蒸馏；不得声称已评估或覆盖holdout。
"""
    private_write("SKILL.md", skill)
    return {"status": status, "claims": len(claims), "files": list(ARTIFACT_FILES)}
