#!/bin/sh
# 仅 shell 参数和状态元数据；不把正文、密钥写进命令行或日志。
set -eu
# 本文件由项目根部脚本 source；调用者必须先按自身路径设置 ROOT，
# 不能使用 $0（source 时 $0 是调用脚本，会把项目根误算到父目录）。
: "${ROOT:?必须先设置项目根 ROOT}"
ROOT=$(CDPATH= cd -- "$ROOT" && pwd -P)
PYTHON=${CHAT_PYTHON:-"$ROOT/.venv/bin/python"}
if [ ! -x "$PYTHON" ]; then
  printf '%s\n' '未找到项目虚拟环境：请先创建 .venv 并安装 requirements.txt，或设置 CHAT_PYTHON。' >&2
  exit 1
fi
HOST=${CHAT_HOST:-127.0.0.1}
PORT=${CHAT_PORT:-8765}
case $HOST in 127.0.0.1|localhost|::1) ;; *) printf '%s\n' '默认只允许 loopback 监听。' >&2; exit 1;; esac
case $PORT in ''|*[!0-9]*) printf '%s\n' 'CHAT_PORT 必须是数字。' >&2; exit 1;; esac
[ "$PORT" -ge 1 ] && [ "$PORT" -le 65535 ] || { printf '%s\n' 'CHAT_PORT 超出范围。' >&2; exit 1; }
STATE="$ROOT/runtime"
PIDFILE="$STATE/uvicorn.pid"
META="$STATE/uvicorn.json"
[ -d "$STATE" ] || mkdir -m 700 -p "$STATE"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export EX_MEMORY_USE_KEYCHAIN=false EX_MEMORY_LOAD_DOTENV=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HUB_DISABLE_TELEMETRY=1 DO_NOT_TRACK=1
export ANONYMIZED_TELEMETRY=False CHROMA_TELEMETRY=False TOKENIZERS_PARALLELISM=false
export CHAT_MCP_OWNER=${CHAT_MCP_OWNER:-1}
export CHAT_MCP_SLUG=${CHAT_MCP_SLUG:-persona}
cd "$ROOT"
