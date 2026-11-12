#!/bin/sh
# 创建项目私有虚拟环境并安装固定版本；不访问聊天内容，不使用 sudo。
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$ROOT"
umask 077
PYTHON=${PYTHON:-python3}

"$PYTHON" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else "需要 Python 3.10 或更高版本")'
if [ ! -x .venv/bin/python ]; then
  "$PYTHON" -m venv .venv
fi
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python setup.py --python .venv/bin/python

printf '%s\n' '依赖和私有目录已准备。下一步：阅读 README 的“准备数据”，再按需下载公开 embedding。'