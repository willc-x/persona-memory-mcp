#!/usr/bin/env python3
"""创建本项目隔离数据目录和不含秘密的示例配置；不安装依赖、不联网。"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# 与运行时一致：默认把运行数据放在项目根；高级部署可显式改用独立数据根。
DATA_ROOT = Path(os.environ.get("CHAT_DATA_DIR", str(ROOT))).expanduser().resolve()

PROVIDER_EXAMPLE = {
    "service_name": "YOUR_PROVIDER",
    "base_url": "https://YOUR_PROVIDER.example/v1",
    "model": "YOUR_MODEL",
    "context_tokens": 16000,
    "billing_mode": "priced",
    "prices": {
        "input_rmb_per_million": None,
        "output_rmb_per_million": None,
    },
    "request_timeout_seconds": 120,
    "api_key_env": "CHAT_LLM_API_KEY",
}


def private_json(path: Path, value: dict, *, overwrite: bool = False) -> bool:
    if path.exists() and not overwrite:
        return False
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.chmod(0o600)
    os.replace(temporary, path)
    return True


def ensure_layout(root: Path = ROOT) -> list[Path]:
    """Create only private runtime directories below the selected data root."""
    created = []
    for relative in ("data", "prepared", "runtime", "logs", "reports", "artifacts", "backups", "config"):
        path = root / relative
        path.mkdir(parents=True, exist_ok=True, mode=0o700)
        path.chmod(0o700)
        created.append(path)
    return created


def write_examples(root: Path = ROOT, *, overwrite: bool = False) -> None:
    private_json(root / "config" / "provider.example.json", PROVIDER_EXAMPLE,
                 overwrite=overwrite)
    env_example = root / "config" / "provider.env.example"
    if overwrite or not env_example.exists():
        env_example.write_text("# 仅本项目使用；不要粘贴到聊天、issue 或提交到 Git。\n"
                               "CHAT_LLM_API_KEY=YOUR_API_KEY\n", encoding="utf-8")
        env_example.chmod(0o600)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", help="只检查指定的 Python 可执行文件路径，不执行它")
    parser.add_argument("--overwrite-example", action="store_true", help="仅覆盖示例文件，不触碰 provider.json")
    args = parser.parse_args()
    if args.python and not Path(args.python).expanduser().is_file():
        parser.error("--python 必须是已存在的解释器；请自行创建 .venv 后重试")
    ensure_layout(DATA_ROOT)
    write_examples(DATA_ROOT, overwrite=args.overwrite_example)
    print(json.dumps({
        "status": "local_directories_ready",
        "data_root": str(DATA_ROOT),
        "next": "复制 provider.example.json 为 provider.json，按实际服务填写；不要提交。",
        "dependencies_installed": False,
        "network_requests": 0,
        "credentials_read": False,
        "distillation_completed": False,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
