#!/bin/sh
# 启动一个 cwd 属于本项目的 uvicorn；不启动、不管理其他目录的服务。
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
. "$ROOT/scripts/runtime_common.sh"
if [ -f "$PIDFILE" ]; then
  OLD=$(sed -n '1p' "$PIDFILE")
  case $OLD in ''|*[!0-9]*) rm -f "$PIDFILE" "$META";; *)
    if "$PYTHON" "$ROOT/scripts/process_guard.py" check "$OLD" "$ROOT"; then
      printf '本项目服务已运行，PID=%s，http://%s:%s/chat\n' "$OLD" "$HOST" "$PORT"
      exit 0
    fi
    printf '%s\n' '旧 PID 文件不属于本项目匹配进程，仅移除状态文件，不发送信号。' >&2
    rm -f "$PIDFILE" "$META"
  ;; esac
fi
umask 077
mkdir -m 700 -p "$ROOT/logs"
nohup "$PYTHON" -m uvicorn adapters.service:app --host "$HOST" --port "$PORT" \
  >>"$ROOT/logs/uvicorn.log" 2>&1 </dev/null &
PID=$!
printf '%s\n' "$PID" >"$PIDFILE"
"$PYTHON" - "$META" "$PID" "$ROOT" "$HOST" "$PORT" <<'PY'
import json, pathlib, sys
path, pid, root, host, port = sys.argv[1:]
pathlib.Path(path).write_text(json.dumps({"pid": int(pid), "cwd": root, "command": "python -m uvicorn adapters.service:app", "host": host, "port": int(port)}, indent=2) + "\n")
PY
READY=0
N=0
while [ "$N" -lt 60 ]; do
  if ! kill -0 "$PID" 2>/dev/null; then break; fi
  if "$PYTHON" - "$HOST" "$PORT" 2>/dev/null <<'PY'
import sys, urllib.request
host, port = sys.argv[1:]
url = f"http://{'[::1]' if host == '::1' else host}:{port}/chat/health"
with urllib.request.urlopen(url, timeout=1) as response:
    assert response.status == 200
PY
  then READY=1; break; fi
  N=$((N + 1))
  sleep 1
done
if [ "$READY" -ne 1 ]; then
  printf '%s\n' '健康检查未通过；将仅回收刚启动且仍匹配本项目 cwd/uvicorn 的进程。' >&2
  if "$PYTHON" "$ROOT/scripts/process_guard.py" check "$PID" "$ROOT"; then
    kill -TERM "$PID"
    "$PYTHON" "$ROOT/scripts/process_guard.py" wait "$PID" "$ROOT" --timeout 15 || true
  fi
  rm -f "$PIDFILE" "$META"
  exit 1
fi
printf '启动完成：PID=%s，http://%s:%s/chat\n' "$PID" "$HOST" "$PORT"
