#!/usr/bin/env python3
"""把用户显式确认写成精确绑定的授权；绝不创建客户端或发起请求。"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from adapters.privacy import DATA_ROOT, PrivacyError, _validate_provider, provider_fingerprint
from distillation.outputs import write_json
from distillation.planning import PlanningError, assert_plan_source, file_digest


def _source_sha256() -> str | None:
    """与 adapters.privacy._source_sha 相同；导入历史是授权的语料边界。"""
    database = DATA_ROOT / "data" / "real_history.sqlite"
    if database.is_file():
        import sqlite3
        with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as connection:
            hashes = [row[0] for row in connection.execute("SELECT sha256 FROM sources ORDER BY sha256")]
        if hashes:
            return hashes[0] if len(hashes) == 1 else hashlib.sha256(json.dumps(hashes).encode()).hexdigest()
    manifest = DATA_ROOT / "reports" / "source_manifest.json"
    if manifest.is_file():
        return json.loads(manifest.read_text(encoding="utf-8")).get("sha256")
    return None


def _bounded_number(value: str, name: str) -> float:
    try:
        number = float(value)
    except ValueError:
        raise ValueError(f"{name} 必须是非负有限数") from None
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{name} 必须是非负有限数")
    return number


def create_grant(args: argparse.Namespace) -> dict:
    provider = json.loads((DATA_ROOT / "config" / "provider.json").read_text(encoding="utf-8"))
    base, model, _ = _validate_provider(provider)
    plan = None
    plan_hash = None
    source_sha = _source_sha256()
    if args.purpose.startswith("distill"):
        if args.plan is None:
            raise ValueError("蒸馏授权必须绑定 --plan")
        args.plan = args.plan.resolve(strict=True)
        plan = json.loads(args.plan.read_text(encoding="utf-8"))
        assert_plan_source(plan, plan["source"]["path"])
        plan_source = plan["source"]["sha256"]
        if source_sha is not None and plan_source != source_sha:
            raise ValueError("计划源哈希与已导入的真实历史不一致；请从同一次准备结果重新计划")
        source_sha = plan_source
        plan_hash = plan["plan_hash"]
        if args.purpose == "distill_trial" and args.max_calls > 2 * min(2, len(plan["segments"])) * 2:
            raise ValueError("试跑授权不能超过前两段各两次调用及重试上限")
    if not isinstance(source_sha, str) or len(source_sha) != 64:
        raise ValueError("先导入真实历史，或使用 --plan 指向可核验的完整计划；禁止猜测语料哈希")
    consent_path = DATA_ROOT / "config" / "privacy-consent.json"
    consent = json.loads(consent_path.read_text(encoding="utf-8")) if consent_path.is_file() else {
        "schema_version": 1, "authorizations": [],
    }
    if not isinstance(consent.get("authorizations"), list):
        raise ValueError("授权文件格式不正确；请先备份并人工检查")
    if any(item.get("id") == args.grant_id for item in consent["authorizations"]):
        raise ValueError("授权 ID 必须唯一；新用途、计划或范围使用新 ID")
    grant = {
        "id": args.grant_id,
        "purposes": [args.purpose],
        "provider_fingerprint": provider_fingerprint(provider),
        "service": provider["service_name"],
        "allowed_base_url": base,
        "allowed_model": model,
        "source_sha256": source_sha,
        "plan_hash": plan_hash,
        "max_calls": args.max_calls,
        "privacy_confirmed": True,
        "confirmed_by_user": True,
    }
    usage_only = args.confirm_route_usage or provider.get("billing_mode") == "usage_only"
    if usage_only:
        if not args.confirm_route_usage or args.max_cost_rmb is not None or args.confirm_full_cost:
            raise ValueError("usage_only 授权需 --confirm-route-usage，且不能同时填写固定费用确认")
        if args.purpose != "distill_full" and not args.allow_usage_trial_or_chat:
            raise ValueError("试用/聊天需要固定费用上限；usage_only 仅默认允许明确确认的全量蒸馏路由")
        grant.update(billing_mode="usage_only", route_usage_confirmed=True, max_cost_rmb=None)
    else:
        if args.max_cost_rmb is None:
            raise ValueError("请根据计划和实际单价提供 --max-cost-rmb；不要误用 usage_only")
        budget = _bounded_number(args.max_cost_rmb, "--max-cost-rmb")
        grant["max_cost_rmb"] = budget
        if args.purpose == "distill_full":
            if not args.confirm_full_cost:
                raise ValueError("全量蒸馏还需 --confirm-full-cost")
            if plan is None or plan.get("cost", {}).get("status") != "estimated":
                raise ValueError("全量蒸馏计划必须含实际输入/输出价格；未知价格不能按零费用授权")
            estimate = plan["cost"].get("estimated_cost_rmb")
            if estimate is None or _bounded_number(estimate, "计划费用") > budget:
                raise ValueError("授权预算低于计划保守费用上界；请重新核对计划与单价")
            grant["full_cost_confirmed"] = True
            grant["estimated_cost_rmb"] = estimate
    consent["authorizations"].append(grant)
    write_json(consent_path, consent)
    return {
        "grant_id": grant["id"], "purpose": args.purpose,
        "provider_fingerprint": grant["provider_fingerprint"],
        "source_sha256": source_sha, "plan_hash": plan_hash,
        "max_calls": args.max_calls, "max_cost_rmb": grant.get("max_cost_rmb"),
        "billing_mode": grant.get("billing_mode", "priced"),
        "network_requests": 0, "credentials_read": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="为指定服务、用途、语料和计划创建精确授权；不发送任何请求")
    parser.add_argument("--purpose", required=True, choices=["distill_trial", "distill_full", "chat"])
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--max-calls", required=True, type=int)
    parser.add_argument("--grant-id", required=True)
    parser.add_argument("--max-cost-rmb")
    parser.add_argument("--confirm", action="store_true", help="确认已同意该服务、用途和隐私风险")
    parser.add_argument("--confirm-full-cost", action="store_true", help="确认全量计划的费用上界")
    parser.add_argument("--confirm-route-usage", action="store_true", help="仅全量：明确接受按实际路由/usage 计费")
    parser.add_argument("--allow-usage-trial-or-chat", action="store_true", help="高级：明确允许试用/聊天仅受次数限制，不设费用上限")
    return parser


def main(argv=None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if not args.confirm or isinstance(args.max_calls, bool) or args.max_calls < 1:
        parser.error("必须显式 --confirm，且 max-calls 为正整数")
    if args.plan and not args.purpose.startswith("distill"):
        parser.error("--plan 只用于蒸馏授权")
    try:
        result = create_grant(args)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError, PlanningError, PrivacyError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
