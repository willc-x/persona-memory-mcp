#!/bin/sh
# 只终止本项目 cwd + uvicorn 命令均匹配的 PID，绝不按端口/名称全局清理。
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
. "$ROOT/scripts/runtime_common.sh"
if [ ! -f "$PIDFILE" ]; then
  printf '%s\n' '没有本项目 uvicorn PID 文件；不按端口猜测，也不终止其他进程。'
  exit 0
fi
PID=$(sed -n '1p' "$PIDFILE")
case $PID in ''|*[!0-9]*) printf '%s\n' 'PID 文件无效；未发送信号。' >&2; rm -f "$PIDFILE" "$META"; exit 1;; esac
if ! "$PYTHON" "$ROOT/scripts/process_guard.py" check "$PID" "$ROOT"; then
  printf '%s\n' 'PID 已退出或 cwd/uvicorn 不匹配；未发送信号，仅移除旧状态文件。'
  rm -f "$PIDFILE" "$META"
  exit 0
fi
kill -TERM "$PID"
if "$PYTHON" "$ROOT/scripts/process_guard.py" wait "$PID" "$ROOT" --timeout 15; then
  rm -f "$PIDFILE" "$META"
  printf '已停止本项目 uvicorn（原 PID=%s）。\n' "$PID"
else
  printf '%s\n' '进程 15 秒内未退出；保留 PID 文件，未强制终止，请人工核对。' >&2
  exit 1
fi
