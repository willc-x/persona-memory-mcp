"""把证据约束的蒸馏产物接入实际 GroundedChatEngine，而不只是并排安装两个项目。"""
import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from adapters.upstream import DATA_ROOT
from adapters.evidence import CanonicalVerifier, canonicalize_claim_interval
from adapters.local_memory import RealHistory, atomic_json
from scripts.chat_memory_tool import inject

ARTIFACTS = ('persona.md', 'memories.md', 'user_profile.md', 'SKILL.md', 'claims.json',
             'timeline.json', 'events.json', 'quality.json', 'meta.json', 'merged.json', 'persona_core.json')


def run(args):
    source = Path(args.source).resolve()
    installation = json.loads((DATA_ROOT / 'runtime/installation.json').read_text(encoding='utf-8'))
    mirror = Path(installation['mirror_directory']).resolve()
    destination = (mirror / 'persona' / 'runtime').resolve()
    if destination != mirror / 'persona' / 'runtime':
        raise ValueError('人物运行目录超出镜像范围')
    required = [source / filename for filename in ARTIFACTS]
    if not all(path.is_file() for path in required):
        raise ValueError('蒸馏产物不完整，拒绝标记整合完成')
    claims_doc = json.loads((source / 'claims.json').read_text(encoding='utf-8'))
    claims = claims_doc['claims'] if isinstance(claims_doc, dict) else claims_doc
    status = claims_doc.get('status', 'schema_list') if isinstance(claims_doc, dict) else 'schema_list'
    history = RealHistory()
    verifier = CanonicalVerifier(history)
    accepted, rejected = [], []
    for claim in claims:
        result = verifier.verify(canonicalize_claim_interval(verifier, claim))
        (accepted if result else rejected).append(result or claim['id'])
    if rejected:
        raise ValueError(f'{len(rejected)}条claim未通过原话/角色/时间回查，拒绝集成')
    if status == 'complete':
        meta = json.loads((source / 'meta.json').read_text(encoding='utf-8'))
        if not meta.get('distillation_complete') or meta.get('planned_training_segments') != meta.get('completed_segments'):
            raise ValueError('全量状态与分段覆盖清单不符')
    # 使用工具包原有幂等标记注入，先自动备份ex-memory原生SKILL与meta。
    ex_skill = mirror / 'SKILL.md'
    backup_root = DATA_ROOT / 'backups' / ('persona-' + datetime.now().strftime('%Y%m%d-%H%M%S'))
    backup_root.mkdir(parents=True, mode=0o700)
    if ex_skill.exists():
        shutil.copy2(ex_skill, backup_root / 'SKILL.md')
    if (mirror / 'meta.json').exists():
        shutil.copy2(mirror / 'meta.json', backup_root / 'meta.json')
    inject(argparse.Namespace(ex_skill=str(ex_skill), suhui_dir=str(source), max_chars=16000, keep_stage=False))
    staging = mirror / 'persona' / '.runtime-staging'
    staging.mkdir(parents=True, exist_ok=True, mode=0o700)
    for path in required:
        shutil.copy2(path, staging / path.name)
    if isinstance(claims_doc, dict):
        claims_doc['claims'] = accepted
        claims_doc['canonical_scope'] = 'exact quote/sender/timestamp/source only; not endorsement of motive or psychological certainty'
        claims_doc['rejected_claims'] = rejected
    else:
        claims_doc = {'status': status, 'claims': accepted, 'canonical_scope': 'quote provenance only'}
    atomic_json(staging / 'claims.json', claims_doc)
    timeline = json.loads((source / 'timeline.json').read_text(encoding='utf-8'))
    valid_ids = {claim['id'] for claim in accepted}
    eras = timeline.get('eras', []) if isinstance(timeline, dict) else timeline
    for era in eras:
        era['claim_ids'] = [claim_id for claim_id in era.get('claim_ids', []) if claim_id in valid_ids]
    atomic_json(staging / 'timeline.json', timeline)
    (staging / 'relationship_timeline.json').write_text((staging / 'timeline.json').read_text(encoding='utf-8'), encoding='utf-8')
    if status == 'complete':
        atomic_json(staging / 'manifest.json', {'status': 'complete', 'integrated_at': datetime.now(timezone.utc).isoformat(),
            'source': str(source), 'canonical_verified_claims': len(accepted), 'rejected_claims': rejected,
            'claim_scope': 'quote provenance, not mental-state certainty', 'ai_sessions_separate': True})
    else:
        manifest = staging / 'manifest.json'
        if manifest.exists():
            manifest.unlink()
    previous = mirror / 'persona' / '.runtime-previous'
    if previous.exists():
        shutil.rmtree(previous)
    if destination.exists():
        destination.rename(previous)
    staging.rename(destination)
    if previous.exists():
        shutil.rmtree(previous)
    installation.update(persona_directory=str(mirror / 'persona'), persona_integrated=True,
                        persona_distilled=status == 'complete', persona_status=status,
                        canonical_verified_claims=len(accepted), last_integrated_at=datetime.now(timezone.utc).isoformat())
    atomic_json(DATA_ROOT / 'runtime/installation.json', installation)
    print(json.dumps({'status': status, 'canonical_verified_claims': len(accepted), 'rejected': len(rejected),
                      'destination': str(destination), 'backup': str(backup_root), 'dynamic_adapter': 'adapters.runtime_adapter.GroundedChatEngine',
                      'static_inject_also_backed_up_and_idempotent': True}, ensure_ascii=False, indent=2))
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True, help='完整的证据约束蒸馏产物目录')
    return run(parser.parse_args())


if __name__ == '__main__':
    raise SystemExit(main())
