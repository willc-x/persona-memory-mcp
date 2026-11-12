"""只在本机复核旧成功/失败响应，绝不联网、重置或覆盖旧checkpoint。"""
import argparse
import hashlib
import os
import json
import re
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from distillation.outputs import claim_union, write_artifacts, write_json
from distillation.planning import digest
from distillation.runner import NODE_SCHEMA
from distillation.validation import validate_extraction


def load_json(path):
    value = json.loads(path.read_text(encoding='utf-8'))
    return value


def valid_node(path):
    value = load_json(path)
    checksum = value.pop('checkpoint_sha256')
    if checksum != digest(value):
        raise ValueError(f'旧节点校验和错误：{path.name}')
    return value


def revalidate_trial(old_plan_path, new_plan_path, messages_path, *, limit=2,
                     consent_path=None, usage_path=None):
    if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
        raise ValueError('limit必须是正整数')
    root = Path(os.environ.get('CHAT_DATA_DIR', Path(__file__).resolve().parents[1])).expanduser().resolve()
    consent_path = Path(consent_path or root / 'config/privacy-consent.json')
    usage_path = Path(usage_path or root / 'runtime/api-usage.json')
    old_plan_path, new_plan_path = Path(old_plan_path), Path(new_plan_path)
    messages_path = Path(messages_path)
    old_root = old_plan_path.resolve().parent
    new_root = new_plan_path.resolve().parent
    if old_root == new_root:
        raise ValueError('禁止覆盖旧试跑目录')
    old_plan, new_plan = load_json(old_plan_path), load_json(new_plan_path)
    for plan in (old_plan, new_plan):
        unsigned = {k: v for k, v in plan.items() if k != 'plan_hash'}
        if digest(unsigned) != plan['plan_hash'] or plan['source'] != old_plan['source']:
            raise ValueError('计划完整性或来源不一致')
    source_hash = hashlib.sha256(messages_path.read_bytes()).hexdigest()
    if source_hash != new_plan['source']['sha256']:
        raise ValueError('真实消息输入与计划哈希不一致')
    if limit > len(new_plan['segments']) or limit > len(old_plan['segments']):
        raise ValueError('limit超过计划分段数')
    # 只对本次真实前缀的已完成节点做本地迁移。后续分段可以重新打包，
    # 但后续差异绝不能挪用前面的用量；全量续跑仍重新执行所有未迁移节点。
    for old_segment, new_segment in zip(old_plan['segments'][:limit], new_plan['segments'][:limit]):
        if (old_segment['source_lines'] != new_segment['source_lines']
                or old_segment['message_sha256'] != new_segment['message_sha256']):
            raise ValueError('本次前缀的源消息顺序或哈希不同，拒绝迁移')
    by_line = {m['source_line']: m for m in load_json(messages_path)}
    recovered, pending, migrated = [], [], []
    consent = load_json(consent_path)
    grants = [g for g in consent.get('authorizations', []) if g.get('plan_hash') == old_plan['plan_hash']]
    if len(grants) != 1 or grants[0].get('purposes') != ['distill_trial']:
        raise ValueError('旧计划必须唯一对应已有试跑授权，不能借用全量或聊天用量')
    grant_id = grants[0]['id']
    all_events = load_json(usage_path)['events']
    events = [e for e in all_events if e.get('grant_id') == grant_id]
    old_call_offset = 0
    original_usage = [dict(event) for event in all_events]
    for new_segment, old_segment in zip(new_plan['segments'][:limit], old_plan['segments'][:limit]):
        if new_segment['source_lines'] != old_segment['source_lines'] or new_segment['message_sha256'] != old_segment['message_sha256']:
            raise ValueError('源行或消息哈希不一致，拒绝迁移')
        for kind in ('persona', 'memories'):
            name = f"{new_segment['id']}.{kind}"
            old_path = old_root / 'segments' / f'{name}.json'
            if not old_path.is_file():
                pending.append(name)
                continue
            old = valid_node(old_path)
            if old['identity'] != {'plan_hash': old_plan['plan_hash'], 'segment_id': old_segment['id'],
                                   'kind': kind, 'message_sha256': old_segment['message_sha256']}:
                raise ValueError(f'旧节点身份错误：{name}')
            if old.get('schema_version') != NODE_SCHEMA or old['status'] not in ('completed', 'failed') or len(old.get('attempts', [])) > 2:
                raise ValueError(f'旧节点状态或尝试历史无效：{name}')
            old_attempts = old.get('attempts', [])
            if not old_attempts or any(set(item) != {'attempt', 'outcome', 'usage'} for item in old_attempts):
                raise ValueError(f'旧节点尝试记录格式错误：{name}')
            if [item.get('attempt') for item in old_attempts] != list(range(1, len(old_attempts) + 1)):
                raise ValueError(f'旧节点请求序号不连续：{name}')
            responses = []
            if old['status'] == 'completed':
                if len(old_attempts) != 1 or old_attempts[-1]['outcome'] != 'completed':
                    raise ValueError('只迁移单次实际完成的节点，拒绝重复调用历史')
                usage = old_attempts[-1]['usage']
                if not isinstance(usage, dict) or set(usage) != {'prompt_tokens', 'completion_tokens', 'total_tokens'}:
                    raise ValueError(f'旧节点缺少真实usage：{name}')
                event = events[old_call_offset] if old_call_offset < len(events) else None
                if not event or event.get('status') != 'completed' or event.get('model') != old['model']:
                    raise ValueError(f'旧完成节点找不到对应provider用量：{name}')
                actual = {'prompt_tokens': event['prompt_tokens'], 'completion_tokens': event['completion_tokens'],
                          'total_tokens': event['prompt_tokens'] + event['completion_tokens']}
                if usage != actual:
                    raise ValueError(f'旧完成节点用量不一致：{name}')
                responses.append((old['response'], old_attempts[-1], 'completed_checkpoint', old_path))
            else:
                # 只在旧文件保持failed、原响应原文有哈希文件且现在严格校验通过时恢复。
                diagnostics = sorted((old_root / 'diagnostics').glob(f'segments.{name}.attempt_*.json'))
                expected_attempts = {item['attempt'] for item in old_attempts if item['outcome'] == 'invalid_response'}
                diagnostic_attempts = {int(re.search(r'\.attempt_(\d+)\.', path.name).group(1)) for path in diagnostics}
                if diagnostic_attempts != expected_attempts or len(diagnostic_attempts) != len(diagnostics):
                    raise ValueError(f'失败诊断不能完整对应已发请求：{name}')
                for diagnostic in diagnostics:
                    match = re.search(r'\.attempt_(\d+)\.([0-9a-f]{16})\.json$', diagnostic.name)
                    if not match:
                        raise ValueError('诊断文件名不能绑定旧请求序号')
                    number, expected_digest = int(match[1]), match[2]
                    diag = load_json(diagnostic)
                    raw_text = diag['response_text']
                    if hashlib.sha256(raw_text.encode()).hexdigest()[:16] != expected_digest:
                        raise ValueError('诊断原文哈希不一致')
                    event = events[old_call_offset + number - 1] if old_call_offset + number <= len(events) else None
                    if (not event or event.get('status') != 'completed' or event.get('grant_id') != grant_id
                            or old_attempts[number - 1]['outcome'] != 'invalid_response'):
                        raise ValueError('诊断响应找不到对应的实际provider用量')
                    if event.get('model') != old['model']:
                        raise ValueError('诊断响应与provider模型不一致')
                    usage = {'prompt_tokens': event['prompt_tokens'], 'completion_tokens': event['completion_tokens'],
                             'total_tokens': event['prompt_tokens'] + event['completion_tokens']}
                    responses.append((json.loads(raw_text), {'attempt': number, 'outcome': 'completed', 'usage': usage},
                                      'failed_response_revalidated_locally', diagnostic))
            accepted = None
            for response, attempt, method, evidence_path in responses:
                try:
                    validated = {'claims': validate_extraction(response, by_line, set(new_segment['source_lines']))}
                except Exception:
                    continue
                accepted = (validated, attempt, method, evidence_path)
                break
            if accepted is None:
                pending.append(name)
                continue
            validated, attempt, method, evidence_path = accepted
            if attempt['outcome'] != 'completed' or not isinstance(attempt.get('usage'), dict):
                raise ValueError(f'恢复节点缺少真实usage：{name}')
            attempts = [{'attempt': i + 1, 'outcome': old_attempts[i]['outcome'], 'usage': old_attempts[i]['usage']}
                        for i in range(len(old_attempts))]
            attempts[attempt['attempt'] - 1] = {'attempt': attempt['attempt'], 'outcome': 'completed', 'usage': attempt['usage']}
            payload = {'schema_version': NODE_SCHEMA,
                       'identity': {'plan_hash': new_plan['plan_hash'], 'segment_id': new_segment['id'],
                                    'kind': kind, 'message_sha256': new_segment['message_sha256']},
                       'model': old['model'], 'status': 'completed', 'attempts': attempts,
                       'response': validated, 'migrated_from': str(old_path), 'migration_method': method,
                       'provider_recall': False}
            if (new_root / 'segments' / f'{name}.json').exists():
                existing = load_json(new_root / 'segments' / f'{name}.json')
                if existing != {**payload, 'checkpoint_sha256': digest(payload)}:
                    raise ValueError(f'新节点已有不同结果，拒绝覆盖：{name}')
            else:
                write_json(new_root / 'segments' / f'{name}.json', {**payload, 'checkpoint_sha256': digest(payload)})
            migrated.append({'node': name, 'method': method, 'claims': len(validated['claims']), 'new_api_calls': 0})
            recovered.extend(validated['claims'])
            old_call_offset += len(old_attempts)
        done_path = new_root / 'segments' / f"{new_segment['id']}.done.json"
        nodes = [new_root / 'segments' / f"{new_segment['id']}.{kind}.json" for kind in ('persona', 'memories')]
        if all(p.is_file() for p in nodes):
            claims = claim_union([c for p in nodes for c in load_json(p)['response']['claims']])
            done = {'schema_version': 'distillation.segment_done.v1',
                    'identity': {'plan_hash': new_plan['plan_hash'], 'segment_id': new_segment['id'],
                                 'message_sha256': new_segment['message_sha256']},
                    'status': 'completed', 'claims': claims}
            if done_path.exists() and load_json(done_path) != {**done, 'checkpoint_sha256': digest(done)}:
                raise ValueError('已有done文件不一致')
            write_json(done_path, {**done, 'checkpoint_sha256': digest(done)})
    after_usage = load_json(usage_path)['events']
    if after_usage != original_usage:
        raise RuntimeError('本地复核不得增加模型用量，已停止')
    result = {'status': 'local_revalidation_only', 'network_calls': 0, 'new_paid_calls': 0,
              'migrated_nodes': migrated, 'pending_nodes': pending,
              'validated_claims': len(claim_union(recovered)), 'old_files_modified': False,
              'scope': f'only first {limit} segments with identical ordered source message hashes; later repacking not migrated',
              'old_plan_hash': old_plan['plan_hash'], 'new_plan_hash': new_plan['plan_hash']}
    write_json(new_root / 'migration-report.json', result)
    write_artifacts(new_root, new_plan, status='incomplete', run_mode='distill_trial',
                    claims=claim_union(recovered), messages=by_line,
                    selected_segment_ids=[s['id'] for s in new_plan['segments'][:limit]])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old-plan', type=Path, required=True)
    parser.add_argument('--new-plan', type=Path, required=True)
    parser.add_argument('--messages', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=2)
    parser.add_argument('--consent', type=Path, help='仅测试或审计时覆盖默认授权文件')
    parser.add_argument('--usage', type=Path, help='仅测试或审计时覆盖默认用量文件')
    args = parser.parse_args()
    result = revalidate_trial(args.old_plan, args.new_plan, args.messages, limit=args.limit,
                              consent_path=args.consent, usage_path=args.usage)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
