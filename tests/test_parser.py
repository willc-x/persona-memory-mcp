"""合成格式测试；不读取私密聊天记录。"""
import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from scripts.chat_memory_tool import parse_message_line, prepare


class ParserTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def run_prepare(self, text, allow=False, target='对象', self_name='我'):
        src = self.root / 'source.txt'
        src.write_bytes(text.encode('utf-8'))
        before = src.read_bytes()
        args = Namespace(input=str(src), out=str(self.root / 'prepared'),
                         encoding='auto', expect_lines=len(text.splitlines()),
                         target=target, self_name=self_name, target_alias=[],
                         allow_quarantine=allow)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            rc = prepare(args)
        self.assertEqual(src.read_bytes(), before)
        return rc, args, json.loads((self.root / 'prepared/audit.json').read_text())

    def test_preserve_text_whitespace_and_builtin_emoji(self):
        raw = '[2025-01-01 12:00:00] 我:  前后空格  '
        row = parse_message_line(raw)
        self.assertEqual(row['content'], ' 前后空格  ')
        rc, _, audit = self.run_prepare(raw + '\r\n[2025-01-01 12:00:01] 对象: [破涕为笑]')
        self.assertEqual(rc, 0)
        self.assertEqual(audit['ex_memory_index_messages'], 2)
        rows = json.loads((self.root / 'prepared/ex_memory.json').read_text())['messages']
        self.assertEqual(rows[0]['content'], ' 前后空格  ')
        self.assertEqual(rows[1]['content'], '[破涕为笑]')
        ledger = [json.loads(s) for s in (self.root / 'prepared/source_ledger.jsonl').read_text().splitlines()]
        self.assertEqual(''.join(r['raw_line'] + r['line_ending'] for r in ledger), raw + '\r\n[2025-01-01 12:00:01] 对象: [破涕为笑]')

    def test_headers_roles_system_body_nickname(self):
        text = '\n'.join([
            '[2025-01-01 12:00:00] 我: 合成正文',
            '[2025-01-01 12:00:01] 对象: 好呀',
            '[2025-01-01 12:00:02] [类型11000] 对象: ',
            '[2025-01-01 12:00:03] [网络电话] 我: [网络电话]',
            '[2025-01-01 12:00:04] [图片] 对象: [图片]',
            '[2025-01-01 12:00:05] [表情] 对象: [表情]',
            '[2025-01-01 12:00:06] [链接/应用] 我: [链接] 附加文字',
            '[2025-01-01 12:00:07] [系统消息] 对象: [系统消息] "陌生昵称" 撤回了一条消息',
        ])
        rc, _, audit = self.run_prepare(text)
        self.assertEqual(rc, 0)
        self.assertEqual(audit['role_mapping'], {'A': '我', 'B': '对象'})
        self.assertEqual(audit['senders'], {'我': 3, '对象': 4})
        self.assertEqual(audit['unknown_sender_lines'], 0)
        self.assertEqual(audit['system_lines_archived'], 1)
        self.assertEqual(audit['media_only_messages'], 3)
        self.assertEqual(audit['ex_memory_index_messages'], 3)
        rows = json.loads((self.root / 'prepared/suhui/messages.json').read_text())
        self.assertEqual(rows[2]['sender'], 'B')
        self.assertEqual(rows[2]['event_type'], '类型11000')
        self.assertEqual(rows[3]['sender'], 'A')
        self.assertFalse(any(r['text'].startswith('[系统消息]') for r in rows))
        self.assertEqual(rows[-1]['text'], '[链接] 附加文字')

    def test_orphan_is_not_attributed_and_blank_is_preserved(self):
        text = '[2025-01-01 12:00:00] 我: 合成问候\n[2025-01-01 12:00:01] 对象: 好\n\n[皱眉][皱眉]\n'
        rc, _, audit = self.run_prepare(text)
        self.assertEqual(rc, 2)
        self.assertFalse((self.root / 'prepared/ex_memory.json').exists())
        rc, _, audit = self.run_prepare(text, allow=True)
        self.assertEqual(rc, 0)
        self.assertEqual(audit['status'], 'ready_with_quarantine')
        self.assertEqual(audit['parse_error_lines'], 1)
        self.assertFalse(audit['strict_parse_passed'])
        self.assertTrue(audit['all_lines_accounted'])
        q = json.loads((self.root / 'prepared/quarantine.json').read_text())[0]
        self.assertIsNone(q['sender'])
        self.assertIsNone(q['timestamp'])
        self.assertEqual(q['source_line'], 4)
        self.assertFalse(q['attributable'])
        self.assertEqual(audit['blank_lines'], 1)

    def test_invalid_date_and_unknown_sender_fail_closed(self):
        for extra in ('[2025-02-30 12:00:02] 对象: 假日期', '[2025-01-01 12:00:02] 第三人: 未授权角色'):
            rc, _, audit = self.run_prepare('[2025-01-01 12:00:00] 我: 合成\n[2025-01-01 12:00:01] 对象: 好\n' + extra, allow=True)
            self.assertEqual(rc, 2)
            self.assertEqual(audit['status'], 'rejected')
            self.assertTrue(audit['all_lines_accounted'])

    def test_duplicate_preserved_and_prepare_idempotent(self):
        text = '[2025-01-01 12:00:00] 我: 合成\n[2025-01-01 12:00:01] 对象: 好\n[2025-01-01 12:00:01] 对象: 好'
        rc, args, audit = self.run_prepare(text)
        self.assertEqual(rc, 0)
        self.assertEqual(audit['exact_duplicate_extra_lines'], 1)
        p = self.root / 'prepared/ex_memory.json'
        original = p.read_bytes()
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(prepare(args), 0)
        self.assertEqual(p.read_bytes(), original)
        rows = json.loads(original)['messages']
        self.assertNotEqual(rows[1]['message_id'], rows[2]['message_id'])
        self.assertEqual(len(rows), 3)
        self.assertEqual(audit['source_sha256'], hashlib.sha256(Path(args.input).read_bytes()).hexdigest())

    def test_role_conflict_is_rejected(self):
        with self.assertRaises(ValueError):
            self.run_prepare('[2025-01-01 12:00:00] 我: 合成', target='我')


if __name__ == '__main__':
    unittest.main()
