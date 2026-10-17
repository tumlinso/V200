#!/usr/bin/env python3
"""Small CPU-only checks for V200 deployment profile and build path choices."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("v200_diagnose", ROOT / "tools/v200/diagnose.py")
assert SPEC and SPEC.loader
diagnose = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnose)


class DeploymentProfileTests(unittest.TestCase):
    def test_checked_in_profile_matches_fixture_and_pair_ordinals(self) -> None:
        profile, fixture, profile_sha, fixture_sha = diagnose.load_profile()
        self.assertEqual(len(profile_sha), 64)
        self.assertEqual(profile["fixture_sha256"], fixture_sha)
        self.assertEqual(profile["pairs"], fixture["topology"]["logical_pairs"])
        self.assertEqual([pair["pair_index"] for pair in profile["pairs"]], [0, 1])

    def test_build_path_is_multipair_and_never_the_native_build(self) -> None:
        selected = diagnose.select_build_dir(ROOT, {"build": {"directory": "build-multipair"}})
        self.assertEqual(selected, ROOT / "build-multipair")
        self.assertNotEqual(selected, ROOT / "build")
        with self.assertRaises(diagnose.DiagnosisError):
            diagnose.select_build_dir(ROOT, {"build": {"directory": "build"}})

    def test_rejects_fixture_hash_or_pair_mapping_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "docs").mkdir()
            (root / "benchmarks").mkdir()
            fixture = json.loads((ROOT / "benchmarks/fixtures/llama-server-multipair.json").read_text())
            profile = json.loads((ROOT / "docs/v200/multipair-profile.json").read_text())
            fixture_path = root / "benchmarks/fixture.json"
            profile_path = root / "docs/profile.json"
            fixture_path.write_text(json.dumps(fixture), encoding="utf-8")
            profile["fixture"] = "../benchmarks/fixture.json"
            profile["fixture_sha256"] = "0" * 64
            profile_path.write_text(json.dumps(profile), encoding="utf-8")
            with self.assertRaisesRegex(diagnose.DiagnosisError, "SHA-256"):
                diagnose.load_profile(profile_path, fixture_path)

            profile["fixture_sha256"] = __import__("hashlib").sha256(fixture_path.read_bytes()).hexdigest()
            profile["pairs"][0]["peer_uuid"] = "GPU-invalid"
            profile_path.write_text(json.dumps(profile), encoding="utf-8")
            with self.assertRaisesRegex(diagnose.DiagnosisError, "mapping"):
                diagnose.load_profile(profile_path, fixture_path)

    def test_build_script_pins_12_9_and_does_not_target_build(self) -> None:
        script = (ROOT / "packaging/build.sh").read_text(encoding="utf-8")
        self.assertIn("/cuda/12.9", script)
        self.assertIn('BUILD_DIR="$ROOT/build-multipair"', script)
        self.assertIn('"$ROOT/build"', script)
        self.assertIn("V200_ENABLE_GPU_TESTS=OFF", script)
        self.assertNotIn("nvidia-smi", script)
        self.assertNotIn("rm -rf", script)


if __name__ == "__main__":
    unittest.main()
