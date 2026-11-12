"""合成安装记录下的私有数据备份边界测试；不接触真实 DATA_ROOT。"""
import json
import shutil
from pathlib import Path
import stat
import tarfile

import pytest

from scripts import manage_data


def _installation(tmp_path: Path, *, mirror: Path | None = None) -> tuple[Path, dict]:
    root = tmp_path / 'data-root'
    for name in manage_data.FIXED_TOP_LEVEL:
        (root / name).mkdir(parents=True, exist_ok=True)
    mirror = mirror or (root / 'ex-memory' / 'exes' / '7' / 'synthetic-persona')
    mirror.mkdir(parents=True, exist_ok=True)
    (mirror / 'chroma.sqlite3').write_bytes(b'synthetic-chroma')
    persona = mirror / 'persona'
    persona.mkdir(exist_ok=True)
    (persona / 'ai_sessions.sqlite').write_bytes(b'synthetic-ai-sessions')
    (root / 'runtime' / 'installation.json').write_text(
        json.dumps({'mirror_directory': str(mirror), 'persona_directory': str(persona)}),
        encoding='utf-8',
    )
    return root, {
        'mirror_directory': str(mirror),
        'persona_directory': str(persona),
    }


def test_custom_mirror_is_included_and_persona_is_deduped(tmp_path):
    root, installation = _installation(tmp_path)
    paths = manage_data.private_paths(installation, root)
    mirror = (root / 'ex-memory' / 'exes' / '7' / 'synthetic-persona').resolve()
    assert mirror in paths
    assert (root / 'data').resolve() in paths
    assert (mirror / 'persona').resolve() not in paths
    assert len(paths) == len(set(paths))


def test_persona_outside_mirror_is_added_once(tmp_path):
    root, installation = _installation(tmp_path)
    separate_persona = root / 'persona-data'
    separate_persona.mkdir()
    (separate_persona / 'persona.md').write_text('synthetic', encoding='utf-8')
    installation['persona_directory'] = str(separate_persona)
    paths = manage_data.private_paths(installation, root)
    assert separate_persona.resolve() in paths
    assert sum(path == separate_persona.resolve() for path in paths) == 1


def test_persona_inside_fixed_artifacts_is_not_archived_twice(tmp_path):
    root, installation = _installation(tmp_path)
    initial_persona = root / 'artifacts' / 'synthetic-persona'
    initial_persona.mkdir()
    installation['persona_directory'] = str(initial_persona)
    paths = manage_data.private_paths(installation, root)
    assert (root / 'artifacts').resolve() in paths
    assert initial_persona.resolve() not in paths


@pytest.mark.parametrize('mirror_value', ['relative/mirror', None, 7])
def test_mirror_must_be_absolute_nonempty_string(tmp_path, mirror_value):
    root, installation = _installation(tmp_path)
    installation['mirror_directory'] = mirror_value
    with pytest.raises(ValueError):
        manage_data.private_paths(installation, root)


def test_mirror_outside_data_root_is_rejected(tmp_path):
    root, installation = _installation(tmp_path)
    outside = tmp_path / 'outside'
    outside.mkdir()
    installation['mirror_directory'] = str(outside)
    with pytest.raises(ValueError, match='DATA_ROOT'):
        manage_data.private_paths(installation, root)


def test_mirror_symlink_to_outside_is_rejected(tmp_path):
    root, installation = _installation(tmp_path)
    outside = tmp_path / 'outside'
    outside.mkdir()
    linked = root / 'linked-mirror'
    linked.symlink_to(outside, target_is_directory=True)
    installation['mirror_directory'] = str(linked)
    with pytest.raises(ValueError, match='DATA_ROOT'):
        manage_data.private_paths(installation, root)


def test_missing_mirror_fails_closed(tmp_path):
    root, installation = _installation(tmp_path)
    installation['mirror_directory'] = str(root / 'does-not-exist')
    with pytest.raises(ValueError, match='不存在'):
        manage_data.private_paths(installation, root)


def test_archive_contains_mirror_and_has_no_duplicate_members(tmp_path):
    root, installation = _installation(tmp_path)
    target = root / 'backups' / 'synthetic.tgz'
    result = manage_data.archive(installation, root, target)
    assert result == target.resolve()
    assert stat.S_IMODE(result.stat().st_mode) == 0o600
    with tarfile.open(result, 'r:gz') as bundle:
        names = bundle.getnames()
    assert len(names) == len(set(names))
    assert any(name.endswith('ex-memory/exes/7/synthetic-persona/chroma.sqlite3')
               for name in names)
    assert any(name.endswith('synthetic-persona/persona/ai_sessions.sqlite') for name in names)
    assert not any(name.startswith('backups/') for name in names)


def test_archive_output_must_be_inside_backups(tmp_path):
    root, installation = _installation(tmp_path)
    with pytest.raises(ValueError, match='backups'):
        manage_data.archive(installation, root, tmp_path / 'outside.tgz')


def test_load_installation_uses_runtime_record(tmp_path):
    root, installation = _installation(tmp_path)
    assert manage_data.load_installation(root) == installation


def test_fixed_directory_symlink_to_outside_is_rejected(tmp_path):
    root, installation = _installation(tmp_path)
    outside = tmp_path / 'outside-data'
    outside.mkdir()
    shutil.rmtree(root / 'data')
    (root / 'data').symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match='DATA_ROOT'):
        manage_data.private_paths(installation, root)


def test_backups_symlink_to_outside_is_rejected(tmp_path):
    root, installation = _installation(tmp_path)
    outside = tmp_path / 'outside-backups'
    outside.mkdir()
    (root / 'backups').symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match='符号链接'):
        manage_data.archive(installation, root)