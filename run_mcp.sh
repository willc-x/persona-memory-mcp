#!/bin/sh
# 前台 stdio MCP：stdout 只传 JSON-RPC；运行状态请看 stderr。
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
. "$ROOT/scripts/runtime_common.sh"
export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
exec "$PYTHON" -m adapters.mcp_server
