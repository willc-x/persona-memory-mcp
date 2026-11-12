"""Synthetic tests for the imported-source containment boundary."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.import_history import _verified_source


class ImportSourceBoundaryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='import-boundary-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / 'chat.txt'
        self.text = b'[2024-01-01 09:00:00] A: synthetic message\n'
        self.source.write_bytes(self.text)
        self.audit = {
            'input_file': str(self.source),
            'source_sha256': hashlib.sha256(self.text).hexdigest(),
        }

    def test_accepts_stable_file_below_data_root(self):
        self.assertEqual(_verified_source(self.audit, self.root), self.source.resolve())

    def test_rejects_data_root_directory(self):
        audit = {'input_file': str(self.root), 'source_sha256': '0' * 64}
        with self.assertRaisesRegex(ValueError, '普通文件'):
            _verified_source(audit, self.root)

    def test_rejects_file_outside_data_root(self):
        outside = self.root.parent / f'{self.root.name}-outside.txt'
        outside.write_bytes(self.text)
        self.addCleanup(outside.unlink, missing_ok=True)
        audit = {'input_file': str(outside), 'source_sha256': self.audit['source_sha256']}
        with self.assertRaisesRegex(ValueError, '不在数据目录'):
            _verified_source(audit, self.root)

    def test_rejects_changed_source(self):
        self.source.write_text('changed', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, '发生变化'):
            _verified_source(self.audit, self.root)

    def test_rejects_malformed_audit_path_or_hash(self):
        for audit in ({}, {'input_file': '', 'source_sha256': 'x' * 64},
                      {'input_file': str(self.source), 'source_sha256': 'not-a-digest'}):
            with self.assertRaises(ValueError):
                _verified_source(audit, self.root)


if __name__ == '__main__':
    unittest.main()
