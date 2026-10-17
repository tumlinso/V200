import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "benchmarks/comparison"))
import run_pair_comparison as comparison


class ManifestTests(unittest.TestCase):
    def _write_test_manifest(self, root: Path) -> tuple[Path, Path]:
        artifacts = []
        first_artifact = None
        for index, role in enumerate(sorted(comparison.REQUIRED_ROLES)):
            path = root / f"artifact-{index}"
            path.write_bytes(f"pinned {role}".encode())
            first_artifact = first_artifact or path
            artifacts.append({"role": role, "path": str(path),
                              "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        prior = root / "prior-result.json"
        prior.write_text("{}\n", encoding="utf-8")
        manifest = {
            "schema": "v200-comparison-binary-manifest/v1",
            "qualified_v200_source_commit": "2bd3020146cd4378a6bb60a614b361d030a3bc34",
            "consumer_harness_source_commit": "df12371c23801291a08c631527e36f22dd21d66c",
            "prior_graph_controller_evidence_id": "797dc352-ff64-40cf-ae18-3dcfc29874f2",
            "prior_graph_result": {
                "path": str(prior),
                "sha256": hashlib.sha256(prior.read_bytes()).hexdigest(),
            },
            "artifacts": artifacts,
        }
        manifest_path = root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        assert first_artifact is not None
        return manifest_path, first_artifact

    def test_manifest_accepts_pinned_files_and_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path, _ = self._write_test_manifest(root)
            result = comparison.verify_manifest(manifest_path, root)
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(result["artifact_count"], len(comparison.REQUIRED_ROLES))

    def test_manifest_fails_closed_when_a_qualified_file_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path, artifact = self._write_test_manifest(root)
            artifact.write_bytes(b"changed after qualification")
            with self.assertRaises(comparison.ComparisonError):
                comparison.verify_manifest(manifest_path, root)


class RequestTests(unittest.TestCase):
    def test_generation_request_is_fixed_and_disables_prompt_cache(self):
        body = comparison.generation_request()
        self.assertEqual(body["n_predict"], 64)
        self.assertTrue(body["ignore_eos"])
        self.assertFalse(body["cache_prompt"])
        self.assertFalse(body["stream"])
        self.assertEqual(body["temperature"], 0.0)

    def test_sample_requires_fixed_token_count_and_server_timings(self):
        response = {
            "content": "peer memory can be read safely through ordered peer access.",
            "tokens_predicted": 64,
            "tokens_evaluated": 28,
            "timings": {
                "prompt_n": 28, "prompt_ms": 17.0,
                "predicted_n": 64, "predicted_ms": 4200.0,
                "prompt_per_second": 1647.0, "predicted_per_second": 15.2,
            },
        }
        sample = comparison.validate_sample(200, response)
        self.assertEqual(sample["tokens_predicted"], 64)
        self.assertEqual(sample["timings"]["predicted_ms"], 4200.0)
        self.assertTrue(comparison.check_exact_output(response["content"], response["content"]))
        self.assertFalse(comparison.check_exact_output(response["content"], "different"))

    def test_sample_rejects_short_output_and_missing_split_timings(self):
        response = {"content": "short", "tokens_predicted": 3,
                    "tokens_evaluated": 28, "timings": {}}
        with self.assertRaises(comparison.ComparisonError):
            comparison.validate_sample(200, response)

    def test_summary_preserves_sample_spread(self):
        summary = comparison._describe([10.0, 20.0, 30.0])
        self.assertEqual(summary["median"], 20.0)
        self.assertEqual(summary["min"], 10.0)
        self.assertEqual(summary["max"], 30.0)
        self.assertGreater(summary["population_sd"], 0.0)


if __name__ == "__main__":
    unittest.main()
