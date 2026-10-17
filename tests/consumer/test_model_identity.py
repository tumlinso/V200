#!/usr/bin/env python3
"""Tiny-file tests for the model identity cache; never inspect real models."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import model_identity as identity


class ModelIdentityTests(unittest.TestCase):
    def test_fresh_hash_writes_stat_signature_then_cache_hit_skips_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "tiny-model.gguf"
            cache = root / "runtime/model-identity-cache.json"
            payload = b"small model fixture\0bytes"
            model.write_bytes(payload)

            fresh = identity.identify_model(model, cache_path=cache)
            self.assertEqual(fresh["sha256"], hashlib.sha256(payload).hexdigest())
            self.assertEqual(fresh["provenance"], "fresh_hash")
            self.assertEqual(set(fresh["signature"]), identity._SIGNATURE_KEYS)
            self.assertEqual(fresh["signature"]["resolved_path"], str(model.resolve()))
            self.assertTrue(cache.exists())

            with mock.patch.object(identity, "_sha256_file",
                                   side_effect=AssertionError("cache hit rehashed model")):
                cached = identity.identify_model(model, cache_path=cache)
            self.assertEqual(cached["sha256"], fresh["sha256"])
            self.assertEqual(cached["provenance"], "cached_stat_match")
            self.assertEqual(cached["signature"], fresh["signature"])

    def test_changed_stat_identity_forces_fresh_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "tiny-model.gguf"
            cache = root / "cache.json"
            model.write_bytes(b"first")
            first = identity.identify_model(model, cache_path=cache)
            model.write_bytes(b"second")

            second = identity.identify_model(model, cache_path=cache)
            self.assertEqual(second["provenance"], "fresh_hash")
            self.assertNotEqual(second["sha256"], first["sha256"])

    def test_missing_or_malformed_cache_falls_back_to_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "tiny-model.gguf"
            cache = root / "cache.json"
            model.write_bytes(b"tiny")
            cache.write_text("{malformed", encoding="utf-8")

            result = identity.identify_model(model, cache_path=cache)
            self.assertEqual(result["provenance"], "fresh_hash")
            self.assertEqual(result["sha256"], hashlib.sha256(b"tiny").hexdigest())
            self.assertEqual(json.loads(cache.read_text(encoding="utf-8"))["schema"],
                             identity.CACHE_SCHEMA)

    def test_change_during_hash_is_an_error_and_not_cached(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "tiny-model.gguf"
            cache = root / "cache.json"
            model.write_bytes(b"before")
            original_hash = identity._sha256_file

            def hash_then_change(path: Path) -> str:
                result = original_hash(path)
                replacement = model.with_name("replacement.gguf")
                replacement.write_bytes(b"after!")
                replacement.replace(model)
                return result

            with mock.patch.object(identity, "_sha256_file", side_effect=hash_then_change):
                with self.assertRaises(identity.ModelChangedError):
                    identity.identify_model(model, cache_path=cache)
            self.assertFalse(cache.exists())

    def test_change_during_cache_hit_check_is_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "tiny-model.gguf"
            cache = root / "cache.json"
            model.write_bytes(b"stable")
            identity.identify_model(model, cache_path=cache)
            signature = identity._stat_identity(model)
            changed = {**signature, "ctime_ns": signature["ctime_ns"] + 1}
            with mock.patch.object(identity, "_stat_identity",
                                   side_effect=[signature, changed]):
                with self.assertRaises(identity.ModelChangedError):
                    identity.identify_model(model, cache_path=cache)


if __name__ == "__main__":
    unittest.main()
