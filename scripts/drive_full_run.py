#!/usr/bin/env python3
"""串行驱动全量蒸馏，只用本地诊断恢复 invalid_response。

不绕过每节点两次请求上限，不重发聊天内容，也不在 provider_error、授权失败或
缺少诊断时继续。每次恢复都必须绑定 usage 账本中本轮新完成的事件。
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from adapters.upstream import DATA_ROOT
from distillation.validation import ClaimValidationError
from scripts.recover_failed_node import recover

FAILURE_RE = re.compile(
    r"node failed after two attempts \((?P<reason>provider_error|invalid_response)\): "
    r"(?P<segment>[A-Za-z0-9][A-Za-z0-9_-]{0,127})\.(?P<kind>persona|memories)\.json"
)
DIAGNOSTIC_RE = re.compile(r"\.attempt_(?P<attempt>[12])\.[0-9a-f]{16}\.json\Z")


def _inside(path: Path, data_root: Path, label: str, *, directory: bool = False,
            must_exist: bool = True) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = data_root / candidate
    candidate = candidate.resolve()
    if candidate == data_root or not candidate.is_relative_to(data_root):
        raise ValueError(f"{label} 必须位于 DATA_ROOT 内且不能是根目录")
    if directory and not candidate.is_dir():
        raise ValueError(f"{label} 不是目录")
    if not directory and must_exist and not candidate.is_file():
        raise ValueError(f"{label} 不是普通文件")
    return candidate


def failure_from_log(text: str) -> tuple[str, str, str] | None:
    """返回最后一个可恢复的 (segment_id, kind, reason)，provider_error 不恢复。"""
    matches = list(FAILURE_RE.finditer(text))
    if not matches:
        return None
    last = matches[-1]
    return last.group("segment"), last.group("kind"), last.group("reason")


def fresh_completed_event(usage: dict, known_ids: set[int], grant_id: str) -> dict | None:
    events = []
    for event in usage.get("events", []):
        if not isinstance(event, dict):
            continue
        try:
            event_id = int(event.get("id"))
        except (TypeError, ValueError):
            continue
        if (event_id not in known_ids and event.get("grant_id") == grant_id
                and event.get("status") == "completed"):
            events.append(event)
    return sorted(events, key=lambda item: int(item["id"]))[-1] if events else None


def latest_diagnostic(workdir: Path, segment_id: str, kind: str) -> tuple[Path, int]:
    directory = workdir / "diagnostics"
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("诊断目录不存在或不能是符号链接")
    candidates = []
    for path in directory.glob(f"segments.{segment_id}.{kind}.attempt_*.json"):
        match = DIAGNOSTIC_RE.search(path.name)
        if match and not path.is_symlink() and path.is_file():
            candidates.append((path.stat().st_mtime_ns, path, int(match.group("attempt"))))
    if not candidates:
        raise ValueError("没有可恢复的本地诊断")
    _, path, attempt = max(candidates, key=lambda item: (item[0], item[1].name))
    return path, attempt


def _interpreter() -> Path:
    value = os.environ.get("CHAT_PYTHON")
    path = Path(value).expanduser() if value else ROOT / ".venv" / "bin" / "python"
    if not path.is_file():
        raise ValueError("找不到项目虚拟环境 Python；请设置 CHAT_PYTHON")
    return path.resolve()


def _run_once(workdir: Path, plan: dict, log_path: Path, offset: int,
              runner=subprocess.run) -> tuple[int, str]:
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    # 计划持久化了当时的 reserved/prompt+output 上限；子进程必须与该计划
    # 一致，不能写死某个输出上限，否则会被计划自洽校验拒绝。
    reserved = plan.get("parameters", {}).get("reserved_prompt_output_tokens")
    if isinstance(reserved, int) and not isinstance(reserved, bool) and reserved >= 4000:
        environment["DISTILL_MAX_OUTPUT_TOKENS"] = str(reserved)
    with log_path.open("ab") as stream:
        stream.write(f"round plan={plan.get('plan_hash', '')} action=run\n".encode("utf-8"))
        stream.flush()
        result = runner(
            [str(_interpreter()), "-m", "distillation.pipeline", "run",
             "--plan", str(workdir / "plan.json"), "--full"],
            cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT,
            timeout=8 * 60 * 60, check=False,
        )
    with log_path.open("r", encoding="utf-8", errors="replace") as stream:
        stream.seek(offset)
        output = stream.read()
    return int(result.returncode), output


def drive(plan_dir: Path, grant_id: str, max_recoveries: int = 100, *,
          data_root: Path | None = None, runner=subprocess.run,
          recover_fn=recover) -> dict:
    data_root = Path(data_root or DATA_ROOT).expanduser().resolve()
    if data_root == Path("/") or not data_root.is_dir():
        raise ValueError("DATA_ROOT 不是有效目录")
    if not isinstance(grant_id, str) or not grant_id.strip():
        raise ValueError("grant_id 不能为空")
    if isinstance(max_recoveries, bool) or not isinstance(max_recoveries, int) or not 0 <= max_recoveries <= 1000:
        raise ValueError("max_recoveries 必须在 0..1000")
    workdir = _inside(plan_dir, data_root, "plan-dir", directory=True)
    plan_path = _inside(workdir / "plan.json", data_root, "plan.json")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if not isinstance(plan, dict) or not isinstance(plan.get("plan_hash"), str):
        raise ValueError("计划缺少 plan_hash")
    log_path = data_root / "logs" / "drive-full.log"
    log_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if log_path.is_symlink():
        raise ValueError("driver 日志不能是符号链接")
    log_path = _inside(log_path, data_root, "driver log", must_exist=False)
    usage_path = data_root / "runtime" / "api-usage.json"

    for round_no in range(max_recoveries + 1):
        usage_path = _inside(usage_path, data_root, "usage")
        before = json.loads(usage_path.read_text(encoding="utf-8"))
        known_ids = {int(item["id"]) for item in before.get("events", [])
                     if isinstance(item, dict) and str(item.get("id", "")).isdigit()}
        offset = log_path.stat().st_size if log_path.exists() else 0
        code, output = _run_once(workdir, plan, log_path, offset, runner=runner)
        if code == 0:
            return {"status": "complete", "rounds": round_no, "recoveries": round_no}
        if round_no >= max_recoveries:
            break
        failure = failure_from_log(output)
        if failure is None:
            return {"status": "stopped", "reason": "not a recoverable invalid_response"}
        segment_id, kind, reason = failure
        if reason != "invalid_response":
            return {"status": "stopped", "reason": reason}
        usage = json.loads(usage_path.read_text(encoding="utf-8"))
        event = fresh_completed_event(usage, known_ids, grant_id)
        if event is None:
            return {"status": "stopped", "reason": "no fresh completed usage event"}
        try:
            _, attempt = latest_diagnostic(workdir, segment_id, kind)
        except ValueError as exc:
            return {"status": "stopped", "reason": str(exc)}
        try:
            receipt = recover_fn(
                workdir, segment_id, kind, attempt, int(event["id"]), grant_id,
                data_root=data_root,
            )
        except ClaimValidationError:
            try:
                receipt = recover_fn(
                    workdir, segment_id, kind, attempt, int(event["id"]), grant_id,
                    salvage=True, data_root=data_root,
                )
            except Exception as exc:
                return {"status": "stopped", "reason": type(exc).__name__, "detail": str(exc)}
        except Exception as exc:
            return {"status": "stopped", "reason": type(exc).__name__, "detail": str(exc)}
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(f"recovered={receipt['node']} claims={receipt['claims']} "
                         f"network_calls=0 provider_recall=false\n")
    return {"status": "stopped", "reason": "recovery cap reached"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-dir", type=Path, required=True)
    parser.add_argument("--grant", required=True)
    parser.add_argument("--max-recoveries", type=int, default=100)
    args = parser.parse_args()
    try:
        result = drive(args.plan_dir, args.grant, args.max_recoveries)
    except Exception as exc:
        print(json.dumps({"status": "error", "error": type(exc).__name__, "detail": str(exc)},
                         ensure_ascii=False), file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())