import signal
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "benchmarks/comparison"))
import run_extended_comparison as extended
import run_pair_comparison as pair


class ExtendedCaseTests(unittest.TestCase):
    def test_pair_and_extended_cases_are_explicit_and_bounded(self):
        self.assertEqual([case["id"] for case in extended.CASES],
                         ["native_pair", "v200_extended"])
        self.assertEqual(extended.CASES[0]["control"], "native_pair")
        self.assertFalse(extended.CASES[0]["pair_compute"])
        self.assertTrue(extended.CASES[1]["pair_compute"])
        self.assertTrue(extended.CASES[1]["extended_expert_compute"])
        self.assertEqual((extended.WARMUP_REQUESTS, extended.MEASURED_REQUESTS), (1, 3))
        self.assertEqual((extended.SERVICE_SECONDS, extended.MAX_BUNDLE_SECONDS), (15, 90))
        self.assertEqual(pair.generation_request()["n_predict"], 64)

    def test_helper_manifest_selects_only_current_pair_runtime_libraries(self):
        provenance = {"artifacts": [
            {"role": "llama_server", "path": "/server", "sha256": "a"},
            *[{"role": role, "path": f"/build-multipair/{role}.so", "sha256": "b"}
              for role in extended.DSO_PATHS],
        ]}
        manifest = extended._helper_manifest(provenance)
        self.assertEqual([row["role"] for row in manifest["artifacts"]],
                         list(extended.DSO_PATHS))
        self.assertTrue(all("build-multipair" in row["path"]
                            for row in manifest["artifacts"]))

    def test_server_exit_requires_clean_requested_shutdown(self):
        case = {"consumer": {
            "server_process": {"exit_code": -signal.SIGTERM},
            "cleanup": {"signaled_owned_group": True, "sigkill_escalation": False,
                        "no_live_group_members": True},
        }}
        self.assertEqual(extended._server_exit_evidence(case)["status"], "PASS")
        case["consumer"]["cleanup"]["sigkill_escalation"] = True
        self.assertEqual(extended._server_exit_evidence(case)["status"], "FAIL")
        case["consumer"]["server_process"]["exit_code"] = 7
        case["consumer"]["cleanup"]["sigkill_escalation"] = False
        self.assertEqual(extended._server_exit_evidence(case)["status"], "FAIL")

    def test_extended_operator_coverage_requires_real_q4_and_q6_peer_calls(self):
        case = {"consumer": {"v200_profile_telemetry": {
            "status": "PASS",
            "operator_records": [
                {"event": "q4k_mmvq_process_exit", "expert_extended_peer_owner_calls": 18,
                 "expert_extended_plain_verified_outputs": 16384},
                {"event": "q6k_mmvq_expert_process_exit", "q6_expert_peer_owner_calls": 18,
                 "q6_expert_plain_peer_owner_calls": 18,
                 "q6_expert_plain_verified_outputs": 16384},
            ],
        }}}
        result = extended._extended_operator_evidence(case)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["q4_low_column_peer_calls"], 18)
        self.assertEqual(result["q6_plain_verified_outputs"], 16384)
        case["consumer"]["v200_profile_telemetry"]["operator_records"].pop()
        self.assertEqual(extended._extended_operator_evidence(case)["status"],
                         "MISSING_EXTENDED_FAMILY")

    def test_all_request_responses_are_checked_against_native_reference(self):
        cases = [
            {"id": "native_pair", "requests": [
                {"kind": "warmup", "response": {"content": "same"}},
                {"kind": "measured", "ordinal": 1, "response": {"content": "same"}},
            ]},
            {"id": "v200_extended", "requests": [
                {"kind": "warmup", "response": {"content": "same"}},
                {"kind": "measured", "ordinal": 1, "response": {"content": "different"}},
            ]},
        ]
        reference, divergences = extended.compare_case_outputs(cases)
        self.assertEqual(reference, "same")
        self.assertEqual(len(divergences), 1)
        self.assertEqual(cases[0]["cross_case_response_equivalence"], "MATCH")
        self.assertEqual(cases[1]["cross_case_response_equivalence"], "DIVERGED")
        self.assertFalse(cases[1]["requests"][1]["exact_response_match_to_native_pair"])


if __name__ == "__main__":
    unittest.main()
