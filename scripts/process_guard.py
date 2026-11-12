#!/usr/bin/env python3
"""启停只操作 cwd 和 uvicorn 参数都匹配本项目的 PID。"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


def managed_process(pid: int, root: Path) -> bool:
    if pid <= 1:
        return False
    cwd = None
    # 新版 lsof 支持 -0（NUL 分隔，中文/空格/换行路径都安全）；macOS 自带
    # 4.91 可能把 -n0 判为非法参数，此时退回 -n 的换行字段格式。
    for flags in (("-n0",), ("-n",)):
        try:
            result = subprocess.run(["lsof", *flags, "-a", "-p", str(pid), "-d", "cwd", "-Fn"],
                                    capture_output=True, timeout=10, check=False)
        except (OSError, subprocess.TimeoutExpired):
            return False
        if getattr(result, "returncode", 0) != 0:
            continue
        raw = result.stdout
        fields = raw.split(b"\0") if b"\0" in raw else raw.splitlines()
        cwd = next((value[1:] for value in fields if value.startswith(b"n") and len(value) > 1), None)
        if cwd is not None:
            break
    if cwd is None:
        return False
    try:
        actual = Path(os.fsdecode(cwd)).resolve(strict=True)
        if actual != root.resolve(strict=True):
            return False
        command = subprocess.run(["ps", "-p", str(pid), "-o", "command="],
                                 capture_output=True, timeout=10, check=True).stdout.decode("utf-8", "strict")
    except (OSError, ValueError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False
    parts = command.split()
    return ("uvicorn" in parts and "adapters.service:app" in parts
            and "-m" in parts and parts.index("-m") + 1 < len(parts)
            and parts[parts.index("-m") + 1] == "uvicorn")


def wait_exit(pid: int, root: Path, timeout: float) -> bool:
    import time
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not managed_process(pid, root):
            return True
        time.sleep(0.2)
    return not managed_process(pid, root)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["check", "wait"])
    parser.add_argument("pid", type=int)
    parser.add_argument("root", type=Path)
    parser.add_argument("--timeout", type=float, default=15)
    args = parser.parse_args()
    matched = managed_process(args.pid, args.root) if args.action == "check" else wait_exit(args.pid, args.root, args.timeout)
    return 0 if matched else 1


if __name__ == "__main__":
    sys.exit(main())
