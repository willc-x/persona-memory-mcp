#!/usr/bin/env python3
"""下载公开、固定 commit 的 BGE embedding；离线核验不需要任何凭据。

这是唯一主动联网的模型文件准备步骤，只访问 Hugging Face 的公开模型文件。
不发送聊天、查询、授权文件或 API key；运行检索时不调用本脚本。
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from adapters.upstream import DATA_ROOT
from distillation.outputs import write_json

MODEL = "BAAI/bge-small-zh-v1.5"
REVISION = "7999e1d3359715c523056ef9478215996d62a620"
# 固定 revision 的 model.safetensors SHA-256（96 MB 级；不是 MD5/伪哈希）。
WEIGHTS_SHA256 = "354763b9b1357bc9c44f62c6be2276321081ed2567773608c0d0785b61d5a026"
WEIGHTS_SIZE = 95_827_648
FILES = (
    "1_Pooling/config.json", "README.md", "config.json",
    "config_sentence_transformers.json", "model.safetensors", "modules.json",
    "sentence_bert_config.json", "special_tokens_map.json", "tokenizer.json",
    "tokenizer_config.json", "vocab.txt",
)
DESTINATION = DATA_ROOT / "data" / "models" / "bge-small-zh-v1.5"
MANIFEST = DATA_ROOT / "reports" / "embedding-model-manifest.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _check_response(response, filename: str) -> None:
    """先核对固定 commit/公开权重元数据，再接收文件；内容仍做完整 SHA。"""
    revision = response.headers.get("x-repo-commit")
    if revision and revision != REVISION:
        raise RuntimeError("公开模型响应的 commit 与固定 revision 不一致")
    if filename == "model.safetensors":
        size = response.headers.get("Content-Length")
        etag = (response.headers.get("x-linked-etag") or response.headers.get("ETag") or "").strip('"')
        if size is not None and int(size) != WEIGHTS_SIZE:
            raise RuntimeError("公开 embedding 权重长度与固定发布不符")
        if not re.fullmatch(r"[0-9a-f]{64}", etag) or etag != WEIGHTS_SHA256:
            raise RuntimeError("公开 embedding 权重的 SHA-256 响应头不匹配；拒绝接收")


def download_snapshot(destination: Path = DESTINATION, manifest_path: Path = MANIFEST,
                      *, opener=None) -> dict:
    """全新 staging 下载并原子发布；不覆盖已有模型。opener 仅供合成测试。"""
    destination, manifest_path = Path(destination), Path(manifest_path)
    if destination.is_symlink():
        raise ValueError("模型目标目录不能是符号链接")
    destination, manifest_path = destination.resolve(), manifest_path.resolve()
    if destination.exists():
        raise FileExistsError("模型目录已存在；请先 --verify，不静默覆盖或拼接旧权重")
    opener = opener or urlopen
    stage = Path(tempfile.mkdtemp(prefix=".bge-staging-", dir=destination.parent))
    receipts = {}
    try:
        for filename in FILES:
            if Path(filename).is_absolute() or ".." in Path(filename).parts:
                raise ValueError("不安全的模型文件名")
            url = f"https://huggingface.co/{MODEL}/resolve/{REVISION}/{filename}"
            request = Request(url, headers={
                "Accept-Encoding": "identity", "User-Agent": "local-bge-snapshot/1.0",
            })
            with opener(request, timeout=120) as response:
                _check_response(response, filename)
                target = stage / filename
                target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                with target.open("xb") as stream:
                    while block := response.read(1024 * 1024):
                        stream.write(block)
            digest = sha256_file(target)
            if filename == "model.safetensors" and (
                    digest != WEIGHTS_SHA256 or target.stat().st_size != WEIGHTS_SIZE):
                raise RuntimeError("权重内容 SHA-256 或长度校验失败；不发布该目录")
            receipts[filename] = {"sha256": digest, "bytes": target.stat().st_size}
        manifest = {
            "schema_version": 1, "model": MODEL, "revision": REVISION,
            "weights_file": "model.safetensors", "weights_sha256": WEIGHTS_SHA256,
            "weights_bytes": WEIGHTS_SIZE, "files": receipts,
            "source": "public-huggingface", "anonymous_download": True,
            "local_files_only": True, "trust_remote_code": False,
            "license": "MIT", "downloaded_at": datetime.now(timezone.utc).isoformat(),
        }
        write_json(stage / "snapshot-manifest.json", manifest)
        verify_snapshot(stage, require_manifest=True)
        os.replace(stage, destination)
        write_json(manifest_path, manifest)
        return {key: manifest[key] for key in ("model", "revision", "weights_sha256", "local_files_only")}
    except Exception:
        shutil.rmtree(stage)
        raise


def verify_snapshot(destination: Path = DESTINATION, *, require_manifest: bool = False) -> dict:
    """纯本地完整复核；不 import 模型、不访问 Hub、不读任何 token。"""
    destination = Path(destination)
    if destination.is_symlink():
        raise ValueError("本地 embedding 快照不能是符号链接")
    destination = destination.resolve()
    if not destination.is_dir():
        raise ValueError("本地 embedding 快照不存在")
    manifest_path = destination / "snapshot-manifest.json"
    if require_manifest or manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("model") != MODEL or manifest.get("revision") != REVISION:
            raise ValueError("本地模型身份不是固定的公开 BGE revision")
        if manifest.get("weights_sha256") != WEIGHTS_SHA256:
            raise ValueError("本地 manifest 权重哈希被修改")
        expected = manifest.get("files", {})
        if set(expected) != set(FILES):
            raise ValueError("本地 manifest 文件清单不完整")
        for filename, receipt in expected.items():
            file = destination / filename
            if not file.is_file() or file.is_symlink() or sha256_file(file) != receipt["sha256"]:
                raise ValueError(f"本地模型文件哈希校验失败: {filename}")
            if file.stat().st_size != receipt["bytes"]:
                raise ValueError(f"本地模型文件长度校验失败: {filename}")
    else:
        missing = [filename for filename in FILES if not (destination / filename).is_file()]
        if missing or sha256_file(destination / "model.safetensors") != WEIGHTS_SHA256:
            raise ValueError("本地快照缺文件或权重哈希不匹配；没有 manifest 时不发布伪造身份")
    return {"status": "verified_offline", "model": MODEL, "revision": REVISION,
            "weights_sha256": WEIGHTS_SHA256, "network_requests": 0}


def main() -> int:
    os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("DO_NOT_TRACK", "1")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", help="只离线验证现有文件，绝不联网")
    parser.add_argument("--yes-public-download", action="store_true", help="明确允许下载公开模型文件")
    parser.add_argument("--destination", type=Path, default=DESTINATION)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    args = parser.parse_args()
    try:
        if args.verify:
            result = verify_snapshot(args.destination)
        else:
            if not args.yes_public_download:
                parser.error("下载需 --yes-public-download；离线检查使用 --verify")
            args.destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            result = download_snapshot(args.destination, args.manifest)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"embedding 准备失败：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
