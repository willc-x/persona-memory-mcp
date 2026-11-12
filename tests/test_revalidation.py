"""Offline migration tests. Only manufactured messages, responses and usage."""
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from distillation.planning import digest, file_digest, make_plan
from distillation.runner import NODE_SCHEMA
from scripts.revalidate_trial import revalidate_trial


class RevalidationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='revalidation-synthetic-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.messages = [
            {'source_line': 1, 'ts': '2024-01-01T09:00:00', 'sender': 'A', 'text': '喜欢什么歌'},
            {'source_line': 2, 'ts': '2024-01-01T09:01:00', 'sender': 'B', 'text': '我喜欢慢节奏的合成歌曲'},
            {'source_line': 3, 'ts': '2024-01-01T09:02:00', 'sender': 'B', 'text': '合成消息，今天喝咖啡。'},
            {'source_line': 4, 'ts': '2024-01-01T09:03:00', 'sender': 'A', 'text': '好的。'},
        ]
        self.source = self.root / 'messages.json'
        self.source.write_text(json.dumps(self.messages, ensure_ascii=False), encoding='utf-8')
        self.plans = []
        for label, context in [('old', 32000), ('new', 32100)]:
            plan = make_plan(self.messages, source_path=str(self.source),
                             source_sha256=file_digest(self.source), holdout_rate=0,
                             context_tokens=context, max_messages=2)
            work = self.root / label
            (work / 'plan.json').parent.mkdir()
            (work / 'plan.json').write_text(json.dumps(plan), encoding='utf-8')
            self.plans.append((work / 'plan.json', plan))
        self.events = []
        self.grant = {'id': 'synthetic-grant', 'plan_hash': self.plans[0][1]['plan_hash'],
                      'purposes': ['distill_trial']}
        self.consent = self.root / 'consent.json'
        self.consent.write_text(json.dumps({'authorizations': [self.grant]}), encoding='utf-8')
        self.usage = self.root / 'usage.json'

    def claim(self, message, statement):
        return {'id': 'raw-id', 'category': 'fact', 'statement': statement,
                'evidence': [{'source_line': message['source_line'], 'sender': message['sender'],
                              'quote': message['text']}],
                'time_start': message['ts'], 'time_end': message['ts'], 'confidence': 0.9}

    def node(self, root, segment, kind, response, attempts):
        payload = {'schema_version': NODE_SCHEMA,
                   'identity': {'plan_hash': self.plans[0][1]['plan_hash'], 'segment_id': segment['id'],
                                'kind': kind, 'message_sha256': segment['message_sha256']},
                   'model': 'synthetic-model', 'status': 'completed', 'attempts': attempts,
                   'response': response}
        path = root / 'segments' / f"{segment['id']}.{kind}.json"
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps({**payload, 'checkpoint_sha256': digest(payload)}, ensure_ascii=False),
                        encoding='utf-8')
        return path

    def event(self, number):
        result = {'id': len(self.events), 'grant_id': 'synthetic-grant', 'status': 'completed',
                  'model': 'synthetic-model', 'prompt_tokens': 100 + number,
                  'completion_tokens': 10 + number, 'actual_cost_rmb': 0}
        self.events.append(result)
        return {'prompt_tokens': result['prompt_tokens'], 'completion_tokens': result['completion_tokens'],
                'total_tokens': result['prompt_tokens'] + result['completion_tokens']}

    def diagnostic(self, root, segment, kind, attempt, value):
        raw = json.dumps(value, ensure_ascii=False)
        name = f"segments.{segment['id']}.{kind}.attempt_{attempt}.{hashlib.sha256(raw.encode()).hexdigest()[:16]}.json"
        path = root / 'diagnostics' / name
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps({'response_text': raw, 'validation_error': 'synthetic invalid'}, ensure_ascii=False),
                        encoding='utf-8')

    def prepare(self):
        old_root = self.plans[0][0].parent
        for index, segment in enumerate(self.plans[0][1]['segments']):
            for kind_index, kind in enumerate(('persona', 'memories')):
                segment_messages = [self.messages[line - 1] for line in segment['source_lines']]
                message = next((item for item in segment_messages if item['sender'] == 'B'), segment_messages[0])
                statement = ('目标人物表示自己喜欢慢节奏歌曲' if kind == 'persona' and '我喜欢' in message['text']
                             else '目标人物发送了这条合成消息')
                response = {'claims': [self.claim(message, statement)]}
                usage = self.event(index * 2 + kind_index)
                self.node(old_root, segment, kind, response, [{'attempt': 1, 'outcome': 'completed', 'usage': usage}])
        self.usage.write_text(json.dumps({'events': self.events}), encoding='utf-8')

    def run_recovery(self):
        return revalidate_trial(self.plans[0][0], self.plans[1][0], self.source, limit=2,
                                consent_path=self.consent, usage_path=self.usage)

    def test_valid_migration_is_offline_and_idempotent(self):
        self.prepare()
        result = self.run_recovery()
        self.assertEqual(len(result['migrated_nodes']), 4)
        self.assertEqual(result['new_paid_calls'], 0)
        self.assertFalse(result['pending_nodes'])
        for kind in ('persona', 'memories'):
            value = json.loads((self.plans[1][0].parent / 'segments' / f'segment_000000.{kind}.json').read_text())
            self.assertFalse(value['provider_recall'])
            self.assertEqual(value['migration_method'], 'completed_checkpoint')
        self.assertEqual(self.run_recovery(), result)

    def test_failed_diagnostic_gets_its_actual_usage_and_old_stays_failed(self):
        self.prepare()
        old_root = self.plans[0][0].parent
        segment = self.plans[0][1]['segments'][0]
        name = f"{segment['id']}.persona"
        path = old_root / 'segments' / f'{name}.json'
        node = json.loads(path.read_text())
        node.pop('checkpoint_sha256')
        response = node['response']
        node.update(status='failed', response=None,
                    attempts=[{'attempt': 1, 'outcome': 'invalid_response', 'usage': None}])
        path.write_text(json.dumps({**node, 'checkpoint_sha256': digest(node)}, ensure_ascii=False))
        self.diagnostic(old_root, segment, 'persona', 1, response)
        result = self.run_recovery()
        self.assertEqual(result['validated_claims'], 3)
        new = json.loads((self.plans[1][0].parent / 'segments' / f'{name}.json').read_text())
        self.assertEqual(new['attempts'][0]['usage'], {key: self.events[0][key]
                         for key in ('prompt_tokens', 'completion_tokens')} | {'total_tokens': 110})
        self.assertEqual(new['attempts'][0]['outcome'], 'completed')
        self.assertEqual(new['migration_method'], 'failed_response_revalidated_locally')
        self.assertEqual(json.loads(path.read_text())['status'], 'failed')

    def test_missing_usage_or_wrong_message_hash_fails_closed(self):
        self.prepare()
        events = copy.deepcopy(self.events)
        self.usage.write_text(json.dumps({'events': events[:-1]}))
        with self.assertRaisesRegex(ValueError, '用量'):
            self.run_recovery()
        self.usage.write_text(json.dumps({'events': events}))
        new_plan_path, plan = self.plans[1]
        changed = copy.deepcopy(plan)
        changed['segments'][0]['message_sha256'] = '0' * 64
        changed['plan_hash'] = digest({k: v for k, v in changed.items() if k != 'plan_hash'})
        new_plan_path.write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, '前缀'):
            self.run_recovery()

    def test_other_grant_usage_is_never_borrowed(self):
        self.prepare()
        self.consent.write_text(json.dumps({'authorizations': [{**self.grant, 'purposes': ['distill_full']}]}))
        with self.assertRaisesRegex(ValueError, '不能借用'):
            self.run_recovery()


if __name__ == '__main__':
    unittest.main()
