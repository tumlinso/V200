#!/usr/bin/env python3
"""CPU-only contracts for the bounded four-owner launcher and evidence parser."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/multi_pair"))
sys.path.insert(0, str(ROOT / "tests/consumer"))
sys.path.insert(0, str(ROOT / "scripts/v200"))
import run_multi_consumer as multi  # noqa: E402
import run_consumer as consumer  # noqa: E402
import serve_multi as launcher  # noqa: E402


FIXTURE = ROOT / "benchmarks/fixtures/llama-server-multipair.json"
PROFILE = ROOT / "docs/v200/multipair-profile.json"
NATIVE_DEVICE_LOG = ROOT / "tests/multi_pair/fixtures/native-four-device-info.log"


class MultiPairContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture, cls.profile = multi.load_inputs(FIXTURE, PROFILE)

    def test_static_full_resident_preflight_fits_all_four_owners(self) -> None:
        result = multi.static_preflight(self.fixture)
        self.assertEqual(result["status"], "PASS_STATIC_ESTIMATE")
        self.assertGreater(result["resident_tensor_bytes"], 16 * 1024**3)
        self.assertEqual(result["v200_pair_owner_budget_bytes"], [4096 * multi.MiB, 12288 * multi.MiB])
        self.assertLessEqual(max(result["v200_pair_estimated_owner_bytes"]), 12288 * multi.MiB)
        self.assertLessEqual(result["native_four_owner_estimated_bytes_each"], 12288 * multi.MiB)

    def test_fixture_pins_unchanged_layer_split_and_pair_receipts(self) -> None:
        controls = self.fixture["controls"]
        self.assertEqual(controls["native_four"]["arguments"][-1], "1,1,1,1")
        self.assertEqual(controls["v200_two"]["arguments"][-1], "1,1")
        self.assertEqual(controls["v200_two"]["required_b01_evidence_ids"], multi.PAIR_RECEIPTS)
        self.assertEqual(self.fixture["topology"]["physical_uuid_order"], multi.UUIDS)

    def test_fixture_rejects_non_loopback_service_binding(self) -> None:
        changed = copy.deepcopy(self.fixture)
        changed["server"]["bind_host"] = "0.0.0.0"
        with self.assertRaisesRegex(multi.MultiPairError, "loopback only"):
            multi.validate_inputs(changed, self.profile)

    def test_facade_environment_uses_both_receipts_budgets_and_preloads(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = [Path(temporary) / name for name in
                     ("libv200_frontdoor.so", "libv200_pair_init.so", "libv200_operators.so")]
            for path in paths:
                path.touch()
            source = {
                "CUDA_VISIBLE_DEVICES": "0,1,2,3",
                "V200_DEVICE_UUIDS": "stale", "V200_B01_EVIDENCE_ID": "stale",
                "V200_PAIR_COMPUTE": "stale", "V200_OWNER0_BUDGET_MIB": "stale",
                "LD_PRELOAD": "/tmp/libunrelated.so:/tmp/libv200_old.so",
                "GGML_CUDA_DISABLE_GRAPHS": "0", "LLAMA_ARG_N_GPU_LAYERS": "4",
            }
            profile_path = Path(temporary) / "ledger.jsonl"
            env = multi.clean_environment("v200_two", paths, profile_path, source)
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], ",".join(multi.UUIDS))
        self.assertEqual(env["V200_DEVICE_UUIDS"], ",".join(multi.UUIDS))
        self.assertEqual(env["V200_B01_EVIDENCE_IDS"], ",".join(multi.PAIR_RECEIPTS))
        self.assertNotIn("V200_B01_EVIDENCE_ID", env)
        self.assertEqual(env["V200_OWNER0_BUDGET_MIB"], "4096")
        self.assertEqual(env["V200_OWNER1_BUDGET_MIB"], "12288")
        self.assertEqual(env["V200_PAIR_COMPUTE"], "1")
        self.assertEqual(env["GGML_CUDA_DISABLE_GRAPHS"], "1")
        self.assertNotIn("LLAMA_ARG_N_GPU_LAYERS", env)
        self.assertTrue(env["LD_PRELOAD"].endswith("/tmp/libunrelated.so"))

    def test_native_environment_scrubs_v200_but_preserves_controller_mask(self) -> None:
        env = multi.clean_environment("native_four", [], None, {
            "CUDA_VISIBLE_DEVICES": "0,1,2,3",
            "V200_MODE": "pair", "V200_PAIR_COMPUTE": "1",
            "V200_B01_EVIDENCE_IDS": ",".join(multi.PAIR_RECEIPTS),
            "LD_PRELOAD": "/tmp/libv200_frontdoor.so:/tmp/libother.so",
        })
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], ",".join(multi.UUIDS))
        self.assertFalse(any(key.startswith("V200_") for key in env))
        self.assertEqual(env["LD_PRELOAD"], "/tmp/libother.so")

    def test_live_selection_accepts_numeric_mask_and_normalizes_child_order(self) -> None:
        environment = {
            "CUDA_VISIBLE_DEVICES": "0,1,2,3",
            "TODO_GPU_LEASE_RECEIPT": "/tmp/fake-lease-receipt.json",
        }
        # The controller's numeric order is the host's physical order. The
        # V200 child needs the same exact set reordered into consecutive pairs.
        numeric_order_uuids = [multi.UUIDS[0], multi.UUIDS[2], multi.UUIDS[1], multi.UUIDS[3]]
        bootstrap = SimpleNamespace(live_lease=mock.Mock(return_value={
            "visible_uuids": numeric_order_uuids,
        }))
        with (mock.patch.dict(os.environ, environment, clear=True),
              mock.patch.object(multi.bounded_runner, "_load_bootstrap", return_value=bootstrap),
              mock.patch.object(consumer, "_verify_nvidia_smi_uuid_selection",
                                return_value=("/usr/bin/nvidia-smi", list(multi.UUIDS))) as identity_query):
            nvidia_smi, normalized = multi.validate_live_selection(environment)
        bootstrap.live_lease.assert_called_once_with(ROOT, multi.UUIDS)
        identity_query.assert_called_once_with(multi.UUIDS)
        self.assertEqual(nvidia_smi, "/usr/bin/nvidia-smi")
        self.assertEqual(normalized, numeric_order_uuids)
        child = multi.clean_environment("native_four", [], None, environment)
        self.assertEqual(child["CUDA_VISIBLE_DEVICES"], ",".join(multi.UUIDS))

    def test_headroom_preflight_checks_each_physical_owner(self) -> None:
        memory = {uuid: {"free_bytes": 15 * multi.GiB, "total_bytes": 16 * multi.GiB}
                  for uuid in multi.UUIDS}
        result = multi.validate_headroom(self.fixture, memory, "v200_two")
        self.assertEqual(result["status"], "PASS")
        short = copy.deepcopy(memory)
        short[multi.UUIDS[3]]["free_bytes"] = 5 * multi.GiB
        with self.assertRaisesRegex(multi.MultiPairError, "insufficient live four-owner headroom"):
            multi.validate_headroom(self.fixture, short, "v200_two")

    def test_profile_parser_requires_both_pairs_payload_copy_directions_and_clean_exit(self) -> None:
        payload = int(self.fixture["model"]["tensor_layout"]["total_tensor_bytes"])
        budget = [4096 * multi.MiB, 12288 * multi.MiB]
        records = []
        for pair_index, pair in enumerate(self.fixture["topology"]["logical_pairs"]):
            peak = [budget[0], payload // 2 - budget[0] + 128 * multi.MiB]
            base = {
                "schema": multi.LEDGER_SCHEMA,
                "event": "snapshot", "mode": "pair",
                "pair_index": pair_index,
                "anchor_uuid": pair["anchor_uuid"], "peer_uuid": pair["peer_uuid"],
                "qualification_id": pair["b01_evidence_id"],
                "owner_budget_bytes": budget,
                "owner_used_bytes": peak,
                "live_allocation_count": 4, "live_reservation_count": 1,
                "cleanup_pending_count": 0,
            }
            records.append(base)
            records.append({**base, "event": "process_exit_final",
                            "owner_used_bytes": [0, 0], "live_allocation_count": 0,
                            "live_reservation_count": 0, "cleanup_pending_count": 0})
        records.extend([
            {"schema": multi.COPY_SCHEMA, "event": "cross_pair_copy", "dst_pair": 1,
             "src_pair": 0, "bytes": 65536, "status": 0, "route": "host_stage",
             "blocking": True, "elapsed_ms": 0.5},
            {"schema": multi.COPY_SCHEMA, "event": "cross_pair_copy", "dst_pair": 0,
             "src_pair": 1, "bytes": 65536, "status": 0, "route": "host_stage",
             "blocking": True, "elapsed_ms": 0.6},
        ])
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "profile.jsonl"
            path.write_text("".join(json.dumps(item) + "\n" for item in records), encoding="utf-8")
            report = multi.read_multi_profile_log(path, self.fixture)
            self.assertEqual(report["status"], "PASS", report["errors"])
            self.assertEqual(set(report["pair_results"]), {"0", "1"})
            self.assertGreaterEqual(report["aggregate_pair_peak_bytes"], payload)

            no_reverse = records[:-1]
            path.write_text("".join(json.dumps(item) + "\n" for item in no_reverse), encoding="utf-8")
            failed = multi.read_multi_profile_log(path, self.fixture)
            self.assertEqual(failed["status"], "FAIL")
            self.assertIn("missing one or both directions", failed["errors"][0])

    def test_profile_parser_rejects_wrong_pair_receipt_and_nonzero_copy(self) -> None:
        records = []
        for pair in self.fixture["topology"]["logical_pairs"]:
            records.append({
                "schema": multi.LEDGER_SCHEMA, "event": "process_exit_final", "mode": "pair",
                "anchor_uuid": pair["anchor_uuid"], "peer_uuid": pair["peer_uuid"],
                "qualification_id": "wrong", "owner_budget_bytes":[4096*multi.MiB,12288*multi.MiB],
                "owner_used_bytes":[0,0], "live_allocation_count":0,
                "live_reservation_count":0, "cleanup_pending_count":0,
            })
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "profile.jsonl"
            path.write_text("".join(json.dumps(item) + "\n" for item in records), encoding="utf-8")
            result = multi.read_multi_profile_log(path, self.fixture, require_resident=False,
                                                  require_copy_directions=False)
        self.assertEqual(result["status"], "INVALID")

    def _run_bundle_with_mocks(self, temporary: str, probe_result: dict,
                               control_result: dict | None = None,
                               control_error: Exception | None = None) -> tuple[dict, mock.Mock]:
        root = Path(temporary)
        preload = root / "libv200_frontdoor.so"
        probe = root / "v200_multi_pair_copy_probe"
        preload.touch()
        probe.touch()
        free = {uuid: {"free_bytes": 15 * multi.GiB, "total_bytes": 16 * multi.GiB}
                for uuid in multi.UUIDS}
        control = mock.Mock(return_value=control_result or {
            "control": "native_four", "application_status": "PASS", "cleanup": {"no_live_group_members": True},
        })
        if control_error is not None:
            control.side_effect = control_error
        with (mock.patch.object(multi, "validate_live_selection", return_value=("nvidia-smi", multi.UUIDS)),
              mock.patch.object(consumer, "verify_identities", return_value={"status": "PASS"}),
              mock.patch.object(consumer, "_nvidia_smi_memory", return_value=free),
              mock.patch.object(multi, "_run_copy_probe", return_value=probe_result),
              mock.patch.object(multi, "_run_control", control)):
            report = multi.run_bundle(FIXTURE, PROFILE, root / "output", [preload], probe,
                                      {"CUDA_VISIBLE_DEVICES": ",".join(multi.UUIDS)})
        return report, control

    def test_failed_copy_probe_stops_before_both_model_controls(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            report, run_control = self._run_bundle_with_mocks(
                temporary, {"status": "FAIL", "error": "copy route failed"})
        self.assertFalse(run_control.called)
        controls = {item["control"]: item for item in report["controls"]}
        self.assertEqual(controls["native_four"]["application_status"], "NOT_RUN")
        self.assertEqual(controls["v200_two"]["application_status"], "NOT_RUN")
        self.assertEqual(report["status"], "FAIL")

    def test_failed_native_control_stops_before_v200_model_load(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            report, run_control = self._run_bundle_with_mocks(
                temporary, {"status": "PASS"},
                {"control": "native_four", "application_status": "FAIL",
                 "failure": "cleanup uncertain", "cleanup": {"no_live_group_members": False}})
        run_control.assert_called_once()
        self.assertEqual(run_control.call_args.args[0], "native_four")
        controls = {item["control"]: item for item in report["controls"]}
        self.assertEqual(controls["native_four"]["application_status"], "FAIL")
        self.assertEqual(controls["v200_two"]["application_status"], "NOT_RUN")
        self.assertEqual(report["status"], "FAIL")

    def test_uncertain_control_cleanup_exception_is_recorded_and_stops_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            report, run_control = self._run_bundle_with_mocks(
                temporary, {"status": "PASS"}, control_error=RuntimeError("cleanup uncertain"))
        run_control.assert_called_once()
        controls = {item["control"]: item for item in report["controls"]}
        self.assertEqual(controls["native_four"]["execution_status"],
                         "RUNNER_ERROR_CLEANUP_UNKNOWN")
        self.assertEqual(controls["native_four"]["cleanup"]["status"], "UNKNOWN")
        self.assertEqual(controls["v200_two"]["application_status"], "NOT_RUN")
        self.assertEqual(report["status"], "FAIL")

    def test_server_exit_validation_accepts_only_normal_or_requested_term(self) -> None:
        self.assertEqual(multi._validate_server_exit_code(0, {})["status"], "PASS")
        requested_term = {"signaled_owned_group": True, "sigkill_escalation": False}
        self.assertEqual(multi._validate_server_exit_code(-15, requested_term)["status"], "PASS")
        self.assertEqual(multi._validate_server_exit_code(-15, {})["status"], "FAIL")
        for exit_code in (-11, -6, -9, None):
            with self.subTest(exit_code=exit_code):
                self.assertEqual(multi._validate_server_exit_code(exit_code, requested_term)["status"],
                                 "FAIL")

    def test_device_info_parser_uses_exact_cuda_ordinals_from_real_native_log(self) -> None:
        # This compact log is copied from the retained B06 native-four run's
        # common_param device_info header for the pinned llama.cpp binary.
        log_text = NATIVE_DEVICE_LOG.read_text(encoding="utf-8")
        evidence = multi._device_count_evidence(log_text, 4)
        self.assertEqual(evidence["status"], "PASS")
        self.assertEqual(evidence["observed_counts"], [4])
        self.assertEqual(evidence["observed_ordinals"], [0, 1, 2, 3])

        missing = log_text.replace("CUDA2", "GPU2")
        self.assertEqual(multi._device_count_evidence(missing, 4)["status"], "MISMATCH")
        extra = log_text + "0.01.000.000 I cmn  common_param:   - CUDA4 : extra device\n"
        self.assertEqual(multi._device_count_evidence(extra, 4)["status"], "MISMATCH")
        reordered = log_text.replace("CUDA0", "CUDA_TMP").replace("CUDA1", "CUDA0")
        reordered = reordered.replace("CUDA_TMP", "CUDA1")
        self.assertEqual(multi._device_count_evidence(reordered, 4)["status"], "MISMATCH")

    def test_copy_probe_requires_each_logical_vmm_round_trip_marker(self) -> None:
        stdout = "".join(f"COPY_OK route={index}\n" for index in range(6))
        stdout += "VMM_OK device=0 bytes=65536\nVMM_OK device=1 bytes=65536\n"
        evidence = multi._copy_probe_marker_evidence(stdout)
        self.assertEqual(evidence["status"], "PASS")
        self.assertEqual(evidence["vmm_completed_devices"], [0, 1])
        self.assertEqual(multi._copy_probe_marker_evidence(
            stdout.replace("VMM_OK device=1 bytes=65536\n", ""))["status"], "FAIL")
        self.assertEqual(multi._copy_probe_marker_evidence(
            stdout.replace("device=1", "device=2"))["status"], "FAIL")

    def test_multi_launcher_default_remains_the_b06_correctness_bundle(self) -> None:
        args = launcher._parse_args([])
        self.assertEqual(args.mode, "correctness")
        plan = launcher._make_plan("test-default")
        self.assertEqual(plan["mode"], "correctness")
        self.assertEqual(plan["controller_spec"]["timeout"], 315)
        self.assertIn("--copy-probe", plan["bounded_child_argv"])
        self.assertNotIn("--service", plan["bounded_child_argv"])
        self.assertIsNone(plan["endpoint"])

    def test_multi_launcher_service_is_finite_loopback_and_two_logical_device_mode(self) -> None:
        plan = launcher._make_plan("test-service", mode="serve",
                                   duration_seconds=600, port=9123)
        argv = plan["bounded_child_argv"]
        self.assertEqual(plan["endpoint"], "http://127.0.0.1:9123")
        self.assertIn("/evidence/B06/serving/runs/test-service/consumer",
                      plan["output_directory"])
        self.assertEqual(plan["bounded_seconds"], 600 + launcher.SERVE_ENVELOPE_RESERVE_SECONDS)
        self.assertEqual(plan["controller_spec"]["resources"]["gpu_uuids"], multi.UUIDS)
        self.assertEqual(plan["controller_spec"]["timeout"],
                         plan["bounded_seconds"] + launcher.SERVE_CONTROLLER_MARGIN_SECONDS)
        self.assertIn("--service", argv)
        self.assertIn("--duration-seconds", argv)
        self.assertIn("600", argv)
        self.assertIn("--port", argv)
        self.assertIn("9123", argv)
        self.assertNotIn("--copy-probe", argv)
        self.assertIn("CUDA0,CUDA1", plan["logical_device_arguments"])
        self.assertEqual(plan["logical_device_arguments"][-1], "1,1")
        self.assertEqual(plan["controller_spec"]["toolchain"]["root"], launcher.CUDA_TOOLKIT_ROOT)

    def test_multi_launcher_service_defaults_and_rejects_out_of_range_values(self) -> None:
        plan = launcher._make_plan("test-service-defaults", mode="serve")
        self.assertEqual(plan["duration_seconds"], launcher.SERVE_DEFAULT_SECONDS)
        self.assertEqual(plan["port"], launcher.DEFAULT_SERVER_PORT)
        for duration, port in ((0, 8080), (launcher.SERVE_MAX_SECONDS + 1, 8080),
                               (10, 0), (10, 65536)):
            with self.subTest(duration=duration, port=port):
                with self.assertRaises(launcher.LaunchError):
                    launcher._make_plan("invalid-service", mode="serve",
                                        duration_seconds=duration, port=port)
        with self.assertRaisesRegex(launcher.LaunchError, "apply only"):
            launcher._make_plan("invalid-mode-args", duration_seconds=10)

    def test_service_runner_refuses_invalid_duration_before_live_preflight(self) -> None:
        with self.assertRaisesRegex(multi.MultiPairError, "service duration"):
            multi.run_service(FIXTURE, PROFILE, Path("/tmp/unwritten-v200-service"),
                              [], 8080, 3001, environ={})
        args = multi._parse_args([
            "--fixture", str(FIXTURE), "--profile", str(PROFILE),
            "--output-dir", "/tmp/service-test", "--preload-library", "/tmp/frontdoor.so",
            "--preload-library", "/tmp/pair-init.so", "--preload-library", "/tmp/operators.so",
            "--service", "--duration-seconds", "10", "--port", "8090", "--execute",
        ])
        self.assertTrue(args.service)
        self.assertEqual(args.duration_seconds, 10)
        self.assertEqual(args.port, 8090)

    def test_pinned_llama_args_include_full_gpu_embedding_trace(self) -> None:
        port = 9123
        workload = self.fixture["workloads"]["two_v200_resident"]
        native = consumer._argv_for("native_four", workload, self.fixture,
                                    self.profile["binary"]["path"], port,
                                    workload_name="resident_capacity")
        facade = consumer._argv_for("v200_two", workload, self.fixture,
                                    self.profile["binary"]["path"], port,
                                    workload_name="resident_capacity")
        self.assertIn("CUDA0,CUDA1,CUDA2,CUDA3", native)
        self.assertIn("1,1,1,1", native)
        self.assertIn("CUDA0,CUDA1", facade)
        self.assertIn("1,1", facade)
        self.assertIn(r"^token_embd\.weight$=CUDA0", facade)
        self.assertIn("999", facade)


if __name__ == "__main__":
    unittest.main(verbosity=2)
