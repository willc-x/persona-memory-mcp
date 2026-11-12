"""仅测试截断 JSON 中完整 claim 对象的本地提取，不含真实聊天。"""
import unittest

from scripts.recover_failed_node import _complete_claim_prefix


class CompleteClaimPrefixTests(unittest.TestCase):
    def test_returns_only_closed_claim_objects(self):
        raw = ('{"claims":[{"id":"a","category":"fact","statement":"合成一","evidence":[],'
               '"time_start":"2024-01-01T00:00:00","time_end":"2024-01-01T00:00:00","confidence":0.9},'
               '{"id":"b","category":"fact","statement":"合成二未闭合')
        values = _complete_claim_prefix(raw)
        self.assertEqual([item['id'] for item in values], ['a'])

    def test_does_not_accept_empty_or_non_claim_prefix(self):
        self.assertEqual(_complete_claim_prefix('not json'), [])
        self.assertEqual(_complete_claim_prefix('{"claims":[]}'), [])
        self.assertEqual(_complete_claim_prefix('{"claims":[1,2]}'), [])


if __name__ == '__main__':
    unittest.main()