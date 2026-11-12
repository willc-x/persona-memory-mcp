"""列出并归档当前安装的本机私有数据。

备份路径以 `DATA_ROOT/runtime/installation.json` 的
`mirror_directory`/`persona_directory` 为准，而不是假设固定的
`DATA_ROOT/exes/<owner>/<slug>`。所有安装路径必须是 DATA_ROOT 内的
绝对路径；不存在的固定目录会被跳过，镜像目录缺失则失败关闭。
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adapters.upstream import DATA_ROOT

FIXED_TOP_LEVEL = ('config', 'data', 'reports', 'runtime', 'prepared', 'artifacts')


def _safe_directory(raw: object, field: str, data_root: Path) -> Path:
    """Return an absolute, existing directory contained by data_root."""
    if not isinstance(raw, str) or not raw:
        raise ValueError(f'{field} 必须是非空路径')
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        raise ValueError(f'{field} 必须是绝对路径')
    path = candidate.resolve()
    if path == data_root or data_root not in path.parents:
        raise ValueError(f'{field} 必须位于 DATA_ROOT 内，且不能等于 DATA_ROOT')
    if not path.is_dir():
        raise ValueError(f'{field} 指向的目录不存在')
    return path


def private_paths(installation: dict, data_root: Path = DATA_ROOT) -> tuple[Path, ...]:
    """Return unique existing data-root directories required for a restorable backup.

    The installation mirror can be outside the conventional `exes/` tree.
    Include it explicitly, and collapse a persona directory that is already
    covered by the mirror or one of the fixed top-level directories.
    """
    root = Path(data_root).expanduser().resolve()
    if not isinstance(installation, dict):
        raise ValueError('installation 必须是对象')
    candidates: list[Path] = []
    for name in FIXED_TOP_LEVEL:
        path = root / name
        if path.exists():
            candidates.append(_safe_directory(str(path), name, root))

    mirror = _safe_directory(installation.get('mirror_directory'), 'mirror_directory', root)
    candidates.append(mirror)

    raw_persona = installation.get('persona_directory')
    if raw_persona is not None:
        persona = _safe_directory(raw_persona, 'persona_directory', root)
        candidates.append(persona)

    # Prefer broader parents; this avoids archiving persona_directory twice
    # when it is mirror/persona or an initial DATA_ROOT/artifacts/<slug>.
    unique: list[Path] = []
    for path in sorted(candidates, key=lambda p: (len(p.parts), str(p))):
        if any(path == parent or parent in path.parents for parent in unique):
            continue
        unique.append(path)
    return tuple(sorted(unique, key=lambda p: p.relative_to(root).as_posix()))


def _backup_output(output: Path | None, data_root: Path) -> Path:
    expected_root = data_root / 'backups'
    if expected_root.is_symlink():
        raise ValueError('DATA_ROOT/backups 不得是符号链接')
    backup_root = expected_root.resolve()
    root = data_root.resolve()
    if backup_root == root or root not in backup_root.parents:
        raise ValueError('DATA_ROOT/backups 必须位于 DATA_ROOT 内')
    target = (backup_root / (
        'persona-' + datetime.now().strftime('%Y%m%d-%H%M%S') + '.tgz'
    )) if output is None else Path(output).expanduser()
    if not target.is_absolute():
        raise ValueError('备份输出必须是绝对路径')
    target = target.resolve()
    if target == backup_root or backup_root not in target.parents:
        raise ValueError('备份输出必须位于 DATA_ROOT/backups 内')
    if target.exists():
        raise FileExistsError(f'拒绝覆盖已有备份：{target}')
    if target.parent != backup_root:
        raise ValueError('备份输出必须直接位于 DATA_ROOT/backups 内')
    return target


def archive(installation: dict, data_root: Path = DATA_ROOT, output: Path | None = None) -> Path:
    """Atomically create a gzip tar of unique private paths; returns the archive path."""
    root = Path(data_root).expanduser().resolve()
    paths = private_paths(installation, root)
    target = _backup_output(output, root)
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, temporary = tempfile.mkstemp(prefix='.backup-', suffix='.part', dir=target.parent)
    os.close(fd)
    os.chmod(temporary, 0o600)
    try:
        with tarfile.open(temporary, 'w:gz') as bundle:
            for path in paths:
                bundle.add(path, arcname=path.relative_to(root).as_posix(), recursive=True)
        os.replace(temporary, target)
        os.chmod(target, 0o600)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return target


def load_installation(data_root: Path = DATA_ROOT) -> dict:
    path = Path(data_root) / 'runtime' / 'installation.json'
    if not path.is_file():
        raise ValueError(f'未找到安装记录：{path}')
    value = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError('installation.json 必须是对象')
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description='列出或归档 DATA_ROOT 内当前安装的私有数据。')
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--list', action='store_true', help='只列出将归档的路径')
    mode.add_argument('--backup', action='store_true', help='创建 0600 gzip 备份')
    parser.add_argument('--output', type=Path, help='备份文件路径，必须直接在 DATA_ROOT/backups 下')
    args = parser.parse_args()
    if args.output and not args.backup:
        parser.error('--output 只能与 --backup 一起使用')
    installation = load_installation()
    paths = private_paths(installation)
    if args.list:
        print(json.dumps([p.relative_to(Path(DATA_ROOT).resolve()).as_posix() for p in paths],
                         ensure_ascii=False, indent=2))
        return 0
    target = archive(installation, output=args.output)
    print(f'备份完成：{target}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())