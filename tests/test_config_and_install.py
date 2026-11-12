"""Synthetic tests for local setup, precise grants, model verification and PID guard.

Never reads real chats, provider credentials, or the network.
"""
from __future__ import annotations

import contextlib
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adapters.privacy import provider_fingerprint
from distillation.outputs import write_json
from distillation.planning import file_digest, make_plan
from scripts import configure


PROVIDER = {
    "service_name": "Synthetic Gateway",
    "base_url": "https://provider.invalid/v1",
    "model": "synthetic-chat",
    "context_tokens": 16000,
    "request_timeout_seconds": 10,
    "billing_mode": "priced",
    "api_key_env": "CHAT_LLM_API_KEY",
    "prices": {"input_rmb_per_million": 1.25, "output_rmb_per_million": 3.5},
}


def synthetic_messages():
    return [{
        "source_line": 1, "ts": "2024-01-01T09:00:00",
        "sender": "B", "text": "这是一条完全合成的测试消息。",
        "type": "text", "kind": "text", "raw_sender": "SYNTHETIC",
        "source_file": "synthetic_fixture", "source_hash": "fixture_hash",
    }]


class PrivateDataRoot:
    def __init__(self, *, provider=None):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        (self.root / "config").mkdir()
        (self.root / "reports").mkdir()
        (self.root / "data").mkdir()
        if provider is not False:
            (self.root / "config" / "provider.json").write_text(
                json.dumps(provider or PROVIDER), encoding="utf-8")

    def close(self):
        self.temporary.cleanup()


class SetupTests(unittest.TestCase):
    def test_layout_and_examples_are_private_and_secret_free(self):
        setup = importlib.import_module("setup")
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            setup.ensure_layout(root)
            setup.write_examples(root)
            for name in ("data", "prepared", "runtime", "logs", "reports",
                         "artifacts", "backups", "config"):
                self.assertEqual((root / name).stat().st_mode & 0o777, 0o700)
            example = json.loads((root / "config" / "provider.example.json").read_text())
            self.assertIsNone(example["prices"]["input_rmb_per_million"])
            self.assertNotIn("YOUR_API_KEY", (root / "config" / "provider.example.json").read_text())
            self.assertFalse((root / "config" / "provider.json").exists())

    def test_checked_in_example_matches_setup_template(self):
        setup = importlib.import_module("setup")
        checked = json.loads((ROOT / "config" / "provider.example.json").read_text())
        self.assertEqual(checked, setup.PROVIDER_EXAMPLE)


class FingerprintTests(unittest.TestCase):
    def test_fingerprint_binds_each_routing_field_and_ignores_extras(self):
        base = provider_fingerprint(PROVIDER)
        self.assertNotEqual(base, provider_fingerprint({**PROVIDER, "model": "other"}))
        self.assertNotEqual(base, provider_fingerprint({**PROVIDER, "base_url": "https://other.invalid/v1"}))
        self.assertNotEqual(base, provider_fingerprint({**PROVIDER, "service_name": "Other"}))
        self.assertNotEqual(base, provider_fingerprint({**PROVIDER, "context_tokens": 32000}))
        self.assertNotEqual(base, provider_fingerprint({
            **PROVIDER, "prices": {"input_rmb_per_million": 2, "output_rmb_per_million": 3.5}}))
        self.assertEqual(base, provider_fingerprint({**PROVIDER, "client_note": "ignored"}))


class ConfigureTests(unittest.TestCase):
    def setUp(self):
        self.data = PrivateDataRoot()
        self.addCleanup(self.data.close)
        self.old_data_root = configure.DATA_ROOT
        configure.DATA_ROOT = self.data.root
        self.addCleanup(setattr, configure, "DATA_ROOT", self.old_data_root)

        self.source = self.data.root / "prepared" / "source.txt"
        self.source.parent.mkdir()
        self.source.write_text("synthetic only\n", encoding="utf-8")
        self.plan = make_plan(
            synthetic_messages(), source_path=str(self.source),
            source_sha256=file_digest(self.source), input_price="1.25", output_price="3.5",
            target_name="测试目标", slug="synthetic-target")
        self.plan_path = self.data.root / "reports" / "plan.json"
        write_json(self.plan_path, self.plan)
        write_json(self.data.root / "reports" / "source_manifest.json",
                   {"sha256": self.plan["source"]["sha256"]})

    def args(self, **updates):
        values = dict(
            purpose="distill_trial", plan=self.plan_path, max_calls=1,
            grant_id="grant-synthetic", max_cost_rmb="1000000", confirm=True,
            confirm_full_cost=False, confirm_route_usage=False,
            allow_usage_trial_or_chat=False)
        values.update(updates)
        return SimpleNamespace(**values)

    def test_trial_grant_binds_real_plan_hash_source_and_exact_fingerprint(self):
        result = configure.create_grant(self.args())
        self.assertEqual(result["provider_fingerprint"], provider_fingerprint(PROVIDER))
        self.assertEqual(result["source_sha256"], self.plan["source"]["sha256"])
        self.assertEqual(result["plan_hash"], self.plan["plan_hash"])
        self.assertFalse(result["credentials_read"])
        consent = json.loads((self.data.root / "config" / "privacy-consent.json").read_text())
        self.assertEqual(len(consent["authorizations"]), 1)
        self.assertNotIn("api_key", json.dumps(consent).lower())

    def test_full_fixed_cost_requires_explicit_confirmation_and_adequate_budget(self):
        estimate = float(self.plan["cost"]["estimated_cost_rmb"])
        with self.assertRaisesRegex(ValueError, "confirm-full-cost"):
            configure.create_grant(self.args(purpose="distill_full", confirm_full_cost=False))
        with self.assertRaisesRegex(ValueError, "低于计划"):
            configure.create_grant(self.args(purpose="distill_full", confirm_full_cost=True,
                                             max_cost_rmb=str(estimate / 2)))
        result = configure.create_grant(self.args(
            purpose="distill_full", confirm_full_cost=True, max_cost_rmb=str(estimate + 1)))
        self.assertTrue(result["max_cost_rmb"] >= estimate)

    def test_usage_only_full_requires_route_confirmation_and_no_fixed_budget(self):
        write_json(self.data.root / "config" / "provider.json",
                   {**PROVIDER, "billing_mode": "usage_only", "prices": None})
        with self.assertRaisesRegex(ValueError, "confirm-route-usage"):
            configure.create_grant(self.args(
                purpose="distill_full", max_cost_rmb=None, confirm_route_usage=False))
        result = configure.create_grant(self.args(
            purpose="distill_full", max_cost_rmb=None, confirm_route_usage=True))
        self.assertEqual(result["billing_mode"], "usage_only")
        self.assertIsNone(result["max_cost_rmb"])

    def test_usage_only_trial_is_denied_unless_advanced_flag(self):
        write_json(self.data.root / "config" / "provider.json",
                   {**PROVIDER, "billing_mode": "usage_only", "prices": None})
        result = configure.create_grant(self.args(
            max_cost_rmb=None, confirm_route_usage=True, allow_usage_trial_or_chat=True))
        self.assertEqual(result["billing_mode"], "usage_only")
        with self.assertRaisesRegex(ValueError, "固定费用"):
            configure.create_grant(self.args(
                grant_id="grant-second", max_cost_rmb=None,
                confirm_route_usage=True, allow_usage_trial_or_chat=False))

    def test_source_hash_mismatch_is_rejected(self):
        write_json(self.data.root / "reports" / "source_manifest.json",
                   {"sha256": "0" * 64})
        with self.assertRaisesRegex(ValueError, "哈希"):
            configure.create_grant(self.args())

    def test_chat_does_not_require_a_distillation_plan(self):
        result = configure.create_grant(self.args(purpose="chat", plan=None, max_calls=2))
        self.assertEqual(result["purpose"], "chat")
        self.assertIsNone(result["plan_hash"])


class DownloadVerificationTests(unittest.TestCase):
    def setUp(self):
        self.module = importlib.import_module("scripts.download_embedding")
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)

    def test_offline_snapshot_requires_exact_manifest_and_hashes(self):
        destination = self.root / "model"
        destination.mkdir()
        receipts = {}
        files = {}
        originals = {}
        for index, name in enumerate(self.module.FILES):
            data = f"synthetic-{index}\n".encode()
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            files[name] = data
            receipts[name] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
        weight = files["model.safetensors"]
        with mock.patch.object(self.module, "WEIGHTS_SHA256", hashlib.sha256(weight).hexdigest()), \
                mock.patch.object(self.module, "WEIGHTS_SIZE", len(weight)):
            write_json(destination / "snapshot-manifest.json", {
                "model": self.module.MODEL, "revision": self.module.REVISION,
                "weights_sha256": self.module.WEIGHTS_SHA256, "files": receipts})
            result = self.module.verify_snapshot(destination, require_manifest=True)
            self.assertEqual(result["network_requests"], 0)
            self.assertEqual(result["revision"], self.module.REVISION)
            (destination / "config.json").write_text("tampered", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "哈希"):
                self.module.verify_snapshot(destination, require_manifest=True)

    def test_symlink_destination_is_rejected_before_resolution_or_network(self):
        target = self.root / "real-model"
        target.mkdir()
        link = self.root / "linked-model"
        link.symlink_to(target, target_is_directory=True)

        def forbidden_opener(*args, **kwargs):
            raise AssertionError("network must not be reached for a symlink destination")

        with self.assertRaisesRegex(ValueError, "符号链接"):
            self.module.verify_snapshot(link)
        with self.assertRaisesRegex(ValueError, "符号链接"):
            self.module.download_snapshot(
                link, self.root / "never-written-manifest.json", opener=forbidden_opener)
        self.assertFalse((self.root / "never-written-manifest.json").exists())

    def test_download_uses_only_synthetic_opener_and_publishes_atomically(self):
        destination = self.root / "new-model"
        manifest_path = self.root / "reports" / "manifest.json"
        payload = b"synthetic weights\n"
        files = {name: (payload if name == "model.safetensors" else (name + "\n").encode())
                 for name in self.module.FILES}
        weight_sha = hashlib.sha256(files["model.safetensors"]).hexdigest()

        class Response:
            def __init__(self, name):
                body = files[name]
                headers = {"Content-Length": str(len(body)), "x-repo-commit": self_module.REVISION}
                if name == "model.safetensors":
                    headers["ETag"] = weight_sha
                self.headers = headers
                self.body = body
                self.offset = 0

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def read(self, size):
                block = self.body[self.offset:self.offset + size]
                self.offset += len(block)
                return block

        self_module = self.module
        requests = []

        def opener(request, timeout):
            requests.append(request.full_url)
            marker = f"/resolve/{self.module.REVISION}/"
            name = request.full_url.split(marker, 1)[1]
            return Response(name)

        with mock.patch.object(self.module, "WEIGHTS_SHA256", weight_sha), \
                mock.patch.object(self.module, "WEIGHTS_SIZE", len(payload)):
            result = self.module.download_snapshot(destination, manifest_path, opener=opener)
            self.assertFalse((destination / "snapshot-manifest.json").is_symlink())
            verified = self.module.verify_snapshot(destination, require_manifest=True)
            self.assertEqual(verified["weights_sha256"], weight_sha)
            self.assertEqual(result["revision"], self.module.REVISION)
            self.assertEqual(len(requests), len(self.module.FILES))
            self.assertTrue(all("7999e1d3359715c523056ef9478215996d62a620" in url for url in requests))


class ProcessGuardTests(unittest.TestCase):
    def test_only_matching_cwd_and_uvicorn_argv_is_managed(self):
        module = importlib.import_module("scripts.process_guard")
        with tempfile.TemporaryDirectory(prefix="合成 路径 ") as raw:
            root = Path(raw).resolve()
            lsof = SimpleNamespace(stdout=b"p123\0n" + os.fsencode(str(root)) + b"\0")
            ok = SimpleNamespace(stdout=b"python -m uvicorn adapters.service:app\n")
            bad_cwd = SimpleNamespace(stdout=b"p123\0n/tmp/elsewhere\0")
            bad_argv = SimpleNamespace(stdout=b"python -m http.server\n")
            with mock.patch.object(module.subprocess, "run", side_effect=[lsof, ok]):
                self.assertTrue(module.managed_process(123, root))
            with mock.patch.object(module.subprocess, "run", return_value=bad_cwd):
                self.assertFalse(module.managed_process(123, root))
            with mock.patch.object(module.subprocess, "run", side_effect=[lsof, bad_argv]):
                self.assertFalse(module.managed_process(123, root))

    def test_lsof_without_nul_support_falls_back_to_line_fields(self):
        module = importlib.import_module("scripts.process_guard")
        with tempfile.TemporaryDirectory(prefix="合成 路径 ") as raw:
            root = Path(raw).resolve()
            unsupported = SimpleNamespace(returncode=1, stdout=b"")
            line_fields = SimpleNamespace(
                returncode=0,
                stdout=b"p123\nfcwd\nn" + os.fsencode(str(root)) + b"\n",
            )
            ok = SimpleNamespace(returncode=0, stdout=b"python -m uvicorn adapters.service:app\n")
            with mock.patch.object(module.subprocess, "run", side_effect=[unsupported, line_fields, ok]):
                self.assertTrue(module.managed_process(123, root))


if __name__ == "__main__":
    unittest.main()