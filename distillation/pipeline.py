"""CLI: local planning/baseline and explicitly authorized serial distillation."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Sequence

from .outputs import write_json
from .planning import (DEFAULT_CONTEXT_TOKENS, DEFAULT_GAP_MINUTES,
                       DEFAULT_HOLDOUT_RATE, DEFAULT_MAX_MESSAGES,
                       DEFAULT_RESERVED_TOKENS, PlanningError,
                       assert_plan_source, file_digest, load_messages, make_plan)
from .runner import PipelineError, run_pipeline, write_local_baseline


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m distillation.pipeline",
        description="证据精确核验、holdout隔离、hash断点的聊天蒸馏")
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan", help="仅本地分段与估价，不初始化API客户端")
    plan.add_argument("--messages", required=True, help="公开schema messages.json")
    plan.add_argument("--out", required=True, help="工作目录；plan.json写在这里")
    plan.add_argument("--max-messages", type=int, default=DEFAULT_MAX_MESSAGES)
    plan.add_argument("--context-tokens", type=int, default=DEFAULT_CONTEXT_TOKENS)
    plan.add_argument("--reserved-tokens", type=int, default=DEFAULT_RESERVED_TOKENS,
                      help="每次请求prompt+output保留；当前必须为4000")
    plan.add_argument("--gap-minutes", type=int, default=DEFAULT_GAP_MINUTES)
    plan.add_argument("--holdout-rate", type=float, default=DEFAULT_HOLDOUT_RATE)
    plan.add_argument("--pack-window", choices=("day", "week", "all"), default="week",
                      help="相邻训练会话packing窗口；30min session边界始终保留")
    plan.add_argument("--target-name", default=os.environ.get("CHAT_TARGET_NAME", "目标人物"),
                      help="目标人物显示名；仅进入产物，不进入源SHA")
    plan.add_argument("--slug", default=os.environ.get("CHAT_DISTILL_SLUG", "persona"),
                      help="产物中的通用小写slug，默认 persona")
    plan.add_argument("--user-alias", default=os.environ.get("CHAT_USER_ALIAS", "我"),
                      help="本人A的显示名，仅作展示元数据")
    plan.add_argument("--input-price", "--input-price-per-million", dest="input_price",
                      default=None, help="人民币/百万input tokens；不传则费用待确认")
    plan.add_argument("--output-price", "--output-price-per-million", dest="output_price",
                      default=None, help="人民币/百万output tokens；不传则费用待确认")
    run = commands.add_parser("run", help="串行执行，节点永久最多2次尝试")
    run.add_argument("--plan", required=True)
    mode = run.add_mutually_exclusive_group(required=True)
    mode.add_argument("--limit", type=int, default=None, help="trial：只选前N个训练段")
    mode.add_argument("--full", action="store_true", help="完成计划内全部训练段")
    baseline = commands.add_parser("baseline", help="只写明确标注的本地空骨架，不调用API")
    baseline.add_argument("--plan", required=True)
    return parser


def _load_plan(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise PlanningError("plan root must be an object")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "plan":
            messages_path = Path(args.messages)
            messages = load_messages(messages_path)
            plan = make_plan(messages, source_path=str(messages_path),
                source_sha256=file_digest(messages_path), max_messages=args.max_messages,
                context_tokens=args.context_tokens, reserved_tokens=args.reserved_tokens,
                gap_minutes=args.gap_minutes, holdout_rate=args.holdout_rate,
                input_price=args.input_price, output_price=args.output_price,
                pack_window=args.pack_window, target_name=args.target_name, slug=args.slug,
                user_alias=args.user_alias)
            out = Path(args.out)
            out.mkdir(parents=True, exist_ok=True)
            path = out / "plan.json"
            if path.exists() and any((out / group).is_dir() and any((out / group).iterdir())
                                     for group in ("segments", "merges")):
                raise PlanningError("refusing to overwrite plan in a checkpoint-bearing directory; use a new --out")
            write_json(path, plan)
            summary = {"status": "planned_local_only", "plan": str(path),
                       "plan_hash": plan["plan_hash"], "corpus": plan["corpus"],
                       "training_segments": len(plan["segments"]),
                       "holdout_segments": len(plan["holdout"]["segments"]),
                       "calls": plan["calls"], "cost": plan["cost"],
                       "token_estimate": plan["token_estimate"]}
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            return 0
        plan_path = Path(args.plan)
        plan = _load_plan(plan_path)
        assert_plan_source(plan, plan["source"]["path"])
        if args.command == "baseline":
            result = write_local_baseline(plan, plan["source"]["path"], plan_path.parent)
        else:
            result = run_pipeline(plan, plan["source"]["path"], plan_path.parent,
                                  limit=args.limit, full=args.full)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (PlanningError, PipelineError) as exc:
        # These are our sanitized messages, never raw provider exceptions.
        print(json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2
    except (OSError, json.JSONDecodeError):
        print(json.dumps({"status": "failed", "error": "local input/output or JSON error"},
                         ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
