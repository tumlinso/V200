#!/usr/bin/env python3
"""Offline fixture/oracle/process tests; no llama binary or GPU is launched."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "scripts/v200"))
import run_consumer as consumer  # noqa: E402
import bounded_runner  # noqa: E402
import serve as serve_launcher  # noqa: E402


class FixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture, cls.profile = consumer.load_documents()

    def test_resident_service_reaches_the_finite_runner(self) -> None:
        with mock.patch.object(consumer, "validate_live_environment", return_value=["GPU-a", "GPU-b"]), \
             mock.patch.object(consumer, "verify_identities", return_value={}), \
             mock.patch.object(consumer, "_run_one", return_value={"application_status": "PASS"}) as run:
            consumer.run_matrix(
                "resident_capacity", ["facade_owner"], Path("/tmp/unused-v200-service"),
                [Path("/tmp/libv200_frontdoor.so"), Path("/tmp/libv200_pair_init.so")],
                monitor=False, facade_mode="pair", server_port=18080,
                serve_duration_seconds=5, cuda_graphs_enabled=True)
        self.assertEqual(run.call_args.args[0], "resident_capacity")
        self.assertEqual(run.call_args.kwargs["serve_duration_seconds"], 5)
        self.assertTrue(run.call_args.kwargs["cuda_graphs_enabled"])

    def test_fixture_pins_consumer_model_and_all_runtime_objects(self) -> None:
        self.assertEqual(self.fixture["model"]["sha256"],
                         "fadc3e5f8d42bf7e894a785b05082e47daee4df26680389817e2093056f088ad")
        self.assertEqual(self.profile["source"]["commit"],
                         "f280b26983ad0fdb705a0d9ebf0503e76f2899b0")
        self.assertEqual(len(self.profile["runtime_closure"]), 8)
        self.assertTrue(all(consumer._is_sha(obj["sha256"])
                            for obj in self.profile["runtime_closure"]))
        self.assertEqual(self.profile["binary"]["sha256"],
                         "b42facfb93d85e1a581862c872615834223dc479871701c0b797aeee1fa11e9d")

    def test_smoke_is_explicit_partial_offload_and_under_one_owner_cap(self) -> None:
        smoke = self.fixture["workloads"]["smoke"]
        self.assertEqual(smoke["n_gpu_layers"], 28)
        self.assertEqual(smoke["ctx_size"], 2048)
        self.assertIn("partial CPU offload", smoke["purpose"])
        estimate = consumer.memory_preflight("smoke", "native_single", self.fixture)
        self.assertTrue(estimate["fits_static_preflight"])
        self.assertLess(estimate["estimated_bytes_per_owner"], estimate["application_limit_bytes_per_owner"])
        self.assertEqual(estimate["effective_gpu_layers"], 28)

    def test_full_resident_is_preflighted_per_owner(self) -> None:
        pair = consumer.memory_preflight("resident_capacity", "native_pair", self.fixture)
        facade = consumer.memory_preflight("resident_capacity", "facade_owner", self.fixture)
        single = consumer.memory_preflight("resident_capacity", "native_single", self.fixture)
        self.assertTrue(pair["fits_static_preflight"])
        self.assertTrue(facade["fits_static_preflight"])
        self.assertFalse(single["fits_static_preflight"])
        self.assertEqual(pair["owner_count_assumed"], 2)
        self.assertEqual(facade["placement_policy"], "preferred_owner_with_peer_spill")
        self.assertEqual(facade["estimated_bytes_by_owner"][0],
                         facade["application_limit_bytes_per_owner"])
        self.assertGreater(facade["estimated_bytes_by_owner"][1], 0)
        self.assertGreater(facade["gpu_layer_tensor_bytes_total"],
                           16 * 1024 * 1024 * 1024)

    def test_command_is_finite_fixed_and_does_not_duplicate_flags(self) -> None:
        argv = consumer._argv_for("native_single", self.fixture["workloads"]["smoke"],
                                  self.fixture, "/fake/llama-server", 12345)
        self.assertEqual(argv.count("--host"), 1)
        self.assertEqual(argv.count("--port"), 1)
        self.assertEqual(argv.count("--ctx-size"), 1)
        self.assertEqual(argv.count("--n-gpu-layers"), 1)
        self.assertIn("off", argv[argv.index("--fit") + 1])
        self.assertEqual(argv[argv.index("--threads") + 1], "4")
        self.assertEqual(argv[argv.index("--threads-batch") + 1], "4")
        self.assertEqual(argv[argv.index("--device") + 1], "CUDA0")

    def test_resident_capacity_keeps_input_embedding_on_single_visible_device(self) -> None:
        argv = consumer._argv_for(
            "facade_owner", self.fixture["workloads"]["resident_capacity"],
            self.fixture, "/fake/llama-server", 12345, "resident_capacity")
        override_index = argv.index("--override-tensor")
        self.assertEqual(argv[override_index + 1], r"^token_embd\.weight$=CUDA0")
        self.assertEqual(argv[argv.index("--verbosity") + 1], "5")

    def test_controller_identity_validation_and_facade_modes(self) -> None:
        env = {"CUDA_VISIBLE_DEVICES": "GPU-a,GPU-b",
               "V200_DEVICE_UUIDS": "GPU-aaaaaaaa,GPU-bbbbbbbb"}
        self.assertEqual(consumer.validate_live_environment("native_single", self.fixture, env),
                         ["GPU-aaaaaaaa"])
        self.assertEqual(consumer.validate_live_environment("native_pair", self.fixture, env),
                         ["GPU-aaaaaaaa", "GPU-bbbbbbbb"])
        self.assertEqual(consumer.validate_live_environment("facade_owner", self.fixture, env),
                         ["GPU-aaaaaaaa", "GPU-bbbbbbbb"])
        self.assertEqual(consumer.validate_live_environment("facade_owner", self.fixture, env, "anchor"),
                         ["GPU-aaaaaaaa"])
        with self.assertRaises(consumer.ConsumerError):
            consumer.validate_live_environment("facade_owner", self.fixture, env, "unexpected")

    def test_nvidia_smi_identity_probe_fails_closed_on_missing_uuid(self) -> None:
        completed = subprocess.CompletedProcess([], 0, "GPU-aaaaaaaa\nGPU-bbbbbbbb\n", "")
        with mock.patch.object(consumer.shutil, "which", return_value="/fake/nvidia-smi"), \
             mock.patch.object(consumer.subprocess, "run", return_value=completed):
            path, observed = consumer._verify_nvidia_smi_uuid_selection(
                ["GPU-aaaaaaaa", "GPU-bbbbbbbb"])
            self.assertEqual(path, "/fake/nvidia-smi")
            self.assertEqual(len(observed), 2)
        with mock.patch.object(consumer.shutil, "which", return_value="/fake/nvidia-smi"), \
             mock.patch.object(consumer.subprocess, "run", return_value=completed):
            with self.assertRaisesRegex(consumer.ConsumerError, "not visible"):
                consumer._verify_nvidia_smi_uuid_selection(["GPU-cccccccc"])

    def test_live_headroom_uses_owner_first_spill_and_refuses_short_pair(self) -> None:
        uuids = ["GPU-aaaaaaaa", "GPU-bbbbbbbb"]
        preflight = consumer.memory_preflight(
            "resident_capacity", "facade_owner", self.fixture, "pair")
        enough = {uuid: {"total_bytes": 16 * 1024**3,
                         "free_bytes": 15 * 1024**3,
                         "total_mib": 16 * 1024,
                         "free_mib": 15 * 1024} for uuid in uuids}
        result = consumer._check_physical_headroom(
            "resident_capacity", "facade_owner", self.fixture, preflight,
            uuids, enough, "pair")
        self.assertTrue(result["payload_exceeds_each_physical_gpu_capacity"])
        self.assertEqual(result["status"], "PASS")
        short = dict(enough)
        short[uuids[1]] = {**enough[uuids[1]], "free_bytes": 5 * 1024**3}
        with self.assertRaisesRegex(consumer.ConsumerError, "insufficient live pair headroom"):
            consumer._check_physical_headroom(
                "resident_capacity", "facade_owner", self.fixture, preflight,
                uuids, short, "pair")

    def test_child_environment_clears_hidden_fit_and_keeps_controller_selection(self) -> None:
        env = {"PATH": "/bin", "CUDA_VISIBLE_DEVICES": "GPU-a,GPU-b",
               "V200_DEVICE_UUIDS": "GPU-aaaaaaaa,GPU-bbbbbbbb",
               "V200_EXTENDED_EXPERT_COMPUTE": "1",
               "LLAMA_ARG_N_GPU_LAYERS": "999", "GGML_CUDA_FORCE_CUBLAS": "1",
               "GGML_CUDA_DISABLE_GRAPHS": "0", "V200_B01_EVIDENCE_ID": "untrusted-parent-value",
               "LD_PRELOAD": "/tmp/libv200_frontdoor.so /tmp/libunrelated.so"}
        native = consumer._clean_environment("native_single", self.fixture, None, env,
                                             selected_uuids=["GPU-aaaaaaaa"])
        self.assertNotIn("LLAMA_ARG_N_GPU_LAYERS", native)
        self.assertNotIn("GGML_CUDA_FORCE_CUBLAS", native)
        self.assertNotIn("V200_MODE", native)
        self.assertNotIn("V200_DEVICE_UUIDS", native)
        self.assertNotIn("V200_B01_EVIDENCE_ID", native)
        self.assertNotIn("V200_EXTENDED_EXPERT_COMPUTE", native)
        self.assertEqual(native["CUDA_VISIBLE_DEVICES"], "GPU-a,GPU-b")
        self.assertEqual(native["GGML_CUDA_DISABLE_GRAPHS"], "1")
        self.assertEqual(native["LD_PRELOAD"], "/tmp/libunrelated.so")
        with tempfile.TemporaryDirectory() as directory:
            preload = Path(directory) / "libv200_frontdoor.so"
            initializer = Path(directory) / "libv200_pair_init.so"
            preload.touch()
            initializer.touch()
            facade = consumer._clean_environment("facade_owner", self.fixture,
                                                 [preload, initializer], env,
                                                 facade_mode="pair",
                                                 selected_uuids=["GPU-aaaaaaaa", "GPU-bbbbbbbb"])
            self.assertEqual(facade["V200_MODE"], "pair")
            self.assertEqual(facade["V200_DEVICE_UUIDS"], "GPU-aaaaaaaa,GPU-bbbbbbbb")
            self.assertEqual(facade["GGML_CUDA_DISABLE_GRAPHS"], "1")
            self.assertEqual(facade["V200_B01_EVIDENCE_ID"],
                             self.fixture["controls"]["facade_owner"]["required_b01_evidence_id"])
            self.assertNotIn("V200_EXTENDED_EXPERT_COMPUTE", facade)
            self.assertEqual(facade["LD_PRELOAD"],
                             f"{preload}:{initializer}:/tmp/libunrelated.so")

    def test_pair_compute_flags_are_facade_only_and_require_operator_library(self) -> None:
        env = {"PATH": "/bin", "CUDA_VISIBLE_DEVICES": "GPU-a,GPU-b",
               "V200_DEVICE_UUIDS": "GPU-aaaaaaaa,GPU-bbbbbbbb",
               "V200_PAIR_COMPUTE": "1", "V200_OPERATOR_VERIFY": "1",
               "V200_EXTENDED_EXPERT_COMPUTE": "1",
               "LD_PRELOAD": "/tmp/libv200_operators.so"}
        native = consumer._clean_environment("native_pair", self.fixture, None, env,
                                             selected_uuids=["GPU-aaaaaaaa", "GPU-bbbbbbbb"])
        self.assertNotIn("V200_PAIR_COMPUTE", native)
        self.assertNotIn("V200_OPERATOR_VERIFY", native)
        self.assertNotIn("V200_EXTENDED_EXPERT_COMPUTE", native)
        with tempfile.TemporaryDirectory() as directory:
            preload = Path(directory) / "libv200_frontdoor.so"
            initializer = Path(directory) / "libv200_pair_init.so"
            operators = Path(directory) / "libv200_operators.so"
            for path in (preload, initializer, operators):
                path.touch()
            facade = consumer._clean_environment(
                "facade_owner", self.fixture, [preload, initializer, operators], env,
                facade_mode="pair", selected_uuids=["GPU-aaaaaaaa", "GPU-bbbbbbbb"])
            self.assertEqual(facade["V200_PAIR_COMPUTE"], "1")
            self.assertEqual(facade["V200_OPERATOR_VERIFY"], "1")
            self.assertEqual(facade["V200_EXTENDED_EXPERT_COMPUTE"], "1")
            self.assertEqual(facade["LD_PRELOAD"], f"{preload}:{initializer}:{operators}")
            with self.assertRaisesRegex(consumer.ConsumerError, "operator library preload"):
                consumer._clean_environment(
                    "facade_owner", self.fixture, [preload, initializer], env,
                    facade_mode="pair", selected_uuids=["GPU-aaaaaaaa", "GPU-bbbbbbbb"])

    def test_local_hbm_preference_is_scrubbed_and_injected_only_on_extended_pair_facade(self) -> None:
        env = {"PATH": "/bin", "CUDA_VISIBLE_DEVICES": "GPU-a,GPU-b",
               "V200_DEVICE_UUIDS": "GPU-aaaaaaaa,GPU-bbbbbbbb",
               "V200_PAIR_COMPUTE": "1", "V200_EXTENDED_EXPERT_COMPUTE": "1",
               "V200_PREFER_LOCAL_HBM": "1",
               "LD_PRELOAD": "/tmp/libv200_operators.so"}
        native = consumer._clean_environment(
            "native_pair", self.fixture, None, env,
            selected_uuids=["GPU-aaaaaaaa", "GPU-bbbbbbbb"])
        self.assertNotIn("V200_PREFER_LOCAL_HBM", native)
        with tempfile.TemporaryDirectory() as directory:
            preloads = [Path(directory) / name for name in
                        ("libv200_frontdoor.so", "libv200_pair_init.so", "libv200_operators.so")]
            for path in preloads:
                path.touch()
            facade = consumer._clean_environment(
                "facade_owner", self.fixture, preloads, env,
                facade_mode="pair", selected_uuids=["GPU-aaaaaaaa", "GPU-bbbbbbbb"])
            self.assertEqual(facade["V200_PREFER_LOCAL_HBM"], "1")

            no_extended = dict(env)
            no_extended.pop("V200_EXTENDED_EXPERT_COMPUTE")
            with self.assertRaisesRegex(consumer.ConsumerError, "requires pair mode"):
                consumer._clean_environment(
                    "facade_owner", self.fixture, preloads, no_extended,
                    facade_mode="pair", selected_uuids=["GPU-aaaaaaaa", "GPU-bbbbbbbb"])

    def test_cuda_graph_opt_in_is_matched_but_frontdoor_flag_is_facade_only(self) -> None:
        env = {"PATH": "/bin", "CUDA_VISIBLE_DEVICES": "GPU-a,GPU-b",
               "V200_GRAPH_CAPTURE": "stale", "GGML_CUDA_DISABLE_GRAPHS": "1"}
        native = consumer._clean_environment(
            "native_pair", self.fixture, None, env,
            selected_uuids=["GPU-aaaaaaaa", "GPU-bbbbbbbb"],
            cuda_graphs_enabled=True)
        self.assertNotIn("GGML_CUDA_DISABLE_GRAPHS", native)
        self.assertNotIn("V200_GRAPH_CAPTURE", native)
        with tempfile.TemporaryDirectory() as directory:
            preloads = [Path(directory) / name for name in
                        ("libv200_frontdoor.so", "libv200_pair_init.so")]
            for path in preloads:
                path.touch()
            facade = consumer._clean_environment(
                "facade_owner", self.fixture, preloads, env,
                facade_mode="pair", selected_uuids=["GPU-aaaaaaaa", "GPU-bbbbbbbb"],
                cuda_graphs_enabled=True)
            self.assertNotIn("GGML_CUDA_DISABLE_GRAPHS", facade)
            self.assertEqual(facade["V200_GRAPH_CAPTURE"], "1")

    def test_graph_records_remain_inside_the_strict_memory_ledger_contract(self) -> None:
        budget = 12288 * 1024 * 1024
        record = {"schema": "v200-frontdoor-ledger/v1", "event": "graph_capture_rewrite",
                  "mode": "pair", "anchor_uuid": "GPU-aaaaaaaa",
                  "peer_uuid": "GPU-bbbbbbbb", "qualification_id": "q-1",
                  "owner_budget_bytes": [budget, budget], "owner_used_bytes": [100, 200],
                  "live_allocation_count": 1, "live_reservation_count": 0,
                  "cleanup_pending_count": 0, "graph_handle": "0x1",
                  "graph_matched_nodes": 2, "graph_rewritten_nodes": 1,
                  "graph_status": 0}
        cleanup = dict(record, event="cuda_free", owner_used_bytes=[0, 0],
                       live_allocation_count=0, graph_matched_nodes=0,
                       graph_rewritten_nodes=0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.jsonl"
            path.write_text(json.dumps(record) + "\n" + json.dumps(cleanup) + "\n")
            result = consumer._read_profile_log(path, ["GPU-aaaaaaaa", "GPU-bbbbbbbb"], budget)
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(result["graph_records"], [record])

    def test_cli_preserves_single_preload_and_accepts_initializer_as_second(self) -> None:
        args = consumer._parse_args([
            "--workload", "smoke", "--control", "facade_owner",
            "--preload-library", "/tmp/libv200_frontdoor.so",
            "--preload-library", "/tmp/libv200_pair_init.so",
            "--output-dir", "/tmp/v200-test", "--execute"])
        self.assertEqual([path.name for path in args.preloads],
                         ["libv200_frontdoor.so", "libv200_pair_init.so"])

    def test_cli_exposes_opt_in_cuda_graph_mode(self) -> None:
        args = consumer._parse_args([
            "--workload", "smoke", "--control", "facade_owner",
            "--output-dir", "/tmp/v200-test", "--cuda-graphs"])
        self.assertTrue(args.cuda_graphs)

    def test_pair_profile_log_is_hash_recorded_and_budget_checked(self) -> None:
        budget = 12288 * 1024 * 1024
        record = {"schema": "v200-frontdoor-ledger/v1", "event": "snapshot",
                  "mode": "pair", "anchor_uuid": "GPU-aaaaaaaa",
                  "peer_uuid": "GPU-bbbbbbbb", "qualification_id": "q-1",
                  "owner_budget_bytes": [budget // 2, budget], "owner_used_bytes": [100, 200],
                  "live_allocation_count": 1, "live_reservation_count": 0,
                  "cleanup_pending_count": 0}
        cleanup = dict(record, event="cuda_free", owner_used_bytes=[0, 0],
                       live_allocation_count=0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.jsonl"
            path.write_text(json.dumps(record) + "\n" + json.dumps(cleanup) + "\n")
            result = consumer._read_profile_log(path,
                                                ["GPU-aaaaaaaa", "GPU-bbbbbbbb"], budget,
                                                minimum_resident_bytes=250)
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(result["records"], [record, cleanup])
            self.assertEqual(result["peak_owner_used_bytes"], [100, 200])
            self.assertEqual(result["peak_total_owner_used_bytes"], 300)
            self.assertTrue(result["cleanup_accounting_zero"])
            self.assertTrue(consumer._is_sha(result["sha256"]))
            record["owner_used_bytes"] = [budget + 1, 200]
            path.write_text(json.dumps(record) + "\n")
            over = consumer._read_profile_log(path,
                                              ["GPU-aaaaaaaa", "GPU-bbbbbbbb"], budget)
            self.assertEqual(over["status"], "OVER_BUDGET")
            self.assertEqual(over["records"][0]["owner_used_bytes"][0], budget + 1)
            record["owner_used_bytes"] = [100, 200]
            record["qualification_id"] = "wrong-evidence"
            path.write_text(json.dumps(record) + "\n")
            mismatch = consumer._read_profile_log(
                path, ["GPU-aaaaaaaa", "GPU-bbbbbbbb"], budget, "expected-evidence")
            self.assertEqual(mismatch["status"], "INVALID")

    def test_profile_ledger_requires_real_pair_payload_and_final_zero_accounting(self) -> None:
        budget = 12 * 1024 * 1024 * 1024
        base = {"schema": "v200-frontdoor-ledger/v1", "event": "snapshot",
                "mode": "pair", "anchor_uuid": "GPU-aaaaaaaa",
                "peer_uuid": "GPU-bbbbbbbb", "qualification_id": "q-1",
                "owner_budget_bytes": [budget, budget],
                "owner_used_bytes": [1024, 0], "live_allocation_count": 1,
                "live_reservation_count": 0, "cleanup_pending_count": 0}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.jsonl"
            path.write_text(json.dumps(base) + "\n")
            insufficient = consumer._read_profile_log(
                path, ["GPU-aaaaaaaa", "GPU-bbbbbbbb"], budget,
                minimum_resident_bytes=16 * 1024)
            self.assertEqual(insufficient["status"], "LIVE_RESOURCES_AT_EXIT")
            base["owner_used_bytes"] = [8 * 1024, 0]
            peak = dict(base, event="peak", live_allocation_count=1)
            cleanup = dict(base, event="cuda_free", owner_used_bytes=[0, 0],
                           live_allocation_count=0)
            path.write_text(json.dumps(peak) + "\n" + json.dumps(cleanup) + "\n")
            insufficient = consumer._read_profile_log(
                path, ["GPU-aaaaaaaa", "GPU-bbbbbbbb"], budget,
                minimum_resident_bytes=16 * 1024)
            self.assertEqual(insufficient["status"], "INSUFFICIENT_RESIDENT_PAYLOAD")

    def test_profile_log_ignores_only_well_formed_operator_records_for_memory_accounting(self) -> None:
        budget = 12 * 1024 * 1024 * 1024
        frontdoor = {"schema": "v200-frontdoor-ledger/v1", "event": "snapshot",
                     "mode": "pair", "anchor_uuid": "GPU-aaaaaaaa",
                     "peer_uuid": "GPU-bbbbbbbb", "qualification_id": "q-1",
                     "owner_budget_bytes": [budget, budget], "owner_used_bytes": [100, 200],
                     "live_allocation_count": 1, "live_reservation_count": 0,
                     "cleanup_pending_count": 0}
        operator = {"schema": "v200-operator-ledger/v1", "event": "q4k_mmvq_route",
                    "operation": "q4k_mmvq", "route": "peer-owner",
                    "matched_calls": 1, "anchor_local_calls": 0, "peer_owner_calls": 1,
                    "split_calls": 0, "anchor_rows": 0, "peer_rows": 16,
                    "verified_rows": 0, "max_abs_error": 0.0, "status": 0}
        cleanup = dict(frontdoor, event="cuda_free", owner_used_bytes=[0, 0],
                       live_allocation_count=0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.jsonl"
            path.write_text("\n".join(json.dumps(value)
                                      for value in (frontdoor, operator, cleanup)) + "\n")
            result = consumer._read_profile_log(path, ["GPU-aaaaaaaa", "GPU-bbbbbbbb"], budget)
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(result["records"], [frontdoor, cleanup])
            self.assertEqual(result["operator_records"], [operator])
            self.assertEqual(result["peak_total_owner_used_bytes"], 300)
            operator.pop("route")
            path.write_text(json.dumps(frontdoor) + "\n" + json.dumps(operator) + "\n")
            malformed = consumer._read_profile_log(path, ["GPU-aaaaaaaa", "GPU-bbbbbbbb"], budget)
            self.assertEqual(malformed["status"], "INVALID")


class LauncherPlanTests(unittest.TestCase):
    def setUp(self) -> None:
        self.uuids, self.pair_identity = serve_launcher._load_selected_pair()

    def test_default_service_plan_is_bounded_pair_facade_and_controller_only(self) -> None:
        imported = bounded_runner._load_bootstrap()
        self.assertEqual(Path(imported.__file__).resolve(),
                         serve_launcher.BOOTSTRAP_RUNNER.resolve())
        self.assertTrue(callable(imported.live_lease))
        plan = serve_launcher._make_plan("serve", self.uuids, "serve-test",
                                         duration_seconds=1800, port=8123)
        self.assertEqual(plan["bundle"], "V200-SERVE")
        self.assertEqual(plan["selected_gpu_uuids"], self.uuids)
        self.assertEqual(plan["duration_seconds"], 1800)
        self.assertEqual(plan["endpoint"], "http://127.0.0.1:8123")
        self.assertEqual(plan["controller_spec"]["toolchain"], {
            "root": "/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/cuda/12.9",
            "require_sanitizer": False,
        })
        self.assertEqual(plan["bounded_seconds"], 2080)
        self.assertLessEqual(plan["controller_timeout_seconds"], 3600)
        self.assertEqual(plan["controller_spec"]["resources"]["gpus"], 2)
        self.assertEqual(plan["controller_spec"]["resources"]["gpu_uuids"], self.uuids)
        self.assertEqual(plan["controller_spec"]["resources"]["cpu_threads"], 4)
        self.assertEqual(plan["controller_spec"]["binary_paths"], [
            "build/src/frontdoor/libv200_frontdoor.so",
            "build/src/integration/libv200_pair_init.so",
        ])
        self.assertEqual(plan["build_directory"], "build")
        self.assertIn("-u", plan["bounded_child_argv"])
        self.assertNotIn("V200_EXTENDED_EXPERT_COMPUTE=1", plan["bounded_child_argv"])
        pinned_llama = consumer.load_documents()[1]["binary"]["path"]
        self.assertNotIn(pinned_llama, plan["controller_spec"]["binary_paths"])
        self.assertEqual(plan["controller_spec"]["argv"][1],
                         str(serve_launcher.BOUNDED_ADAPTER))
        self.assertIn("V200_DEVICE_UUIDS=" + ",".join(self.uuids),
                      plan["bounded_child_argv"])
        self.assertEqual(plan["static_memory_preflight"][0]["control"], "facade_owner")
        self.assertTrue(plan["static_memory_preflight"][0]["fits_static_preflight"])
        self.assertFalse(plan["cuda_graphs_enabled"])
        default_plan = serve_launcher._make_plan("serve", self.uuids, "default-serve")
        self.assertEqual(default_plan["duration_seconds"], serve_launcher.SERVE_DEFAULT_SECONDS)
        self.assertEqual(default_plan["endpoint"], "http://127.0.0.1:8080")

    def test_default_cli_selects_finite_serve_but_keeps_explicit_execution_gate(self) -> None:
        args = serve_launcher._parse_args([])
        self.assertEqual(args.mode, "serve")
        self.assertEqual(args.workload, "smoke")
        self.assertFalse(args.execute)
        self.assertEqual(args.duration_seconds, None)
        self.assertEqual(args.port, None)

    def test_matched_preview_uses_graph_off_native_and_facade_through_b03_bound(self) -> None:
        plan = serve_launcher._make_plan("preview", self.uuids, "preview-test")
        self.assertEqual(plan["bundle"], "B03")
        self.assertEqual(plan["controls"], ["native_single", "facade_owner"])
        self.assertEqual(plan["bounded_seconds"], 300)
        self.assertEqual(plan["controller_timeout_seconds"], 315)
        self.assertEqual(plan["controller_spec"]["toolchain"], {
            "root": "/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/cuda/12.9",
            "require_sanitizer": False,
        })
        self.assertEqual(plan["controller_spec"]["argv"][1],
                         str(serve_launcher.BOOTSTRAP_RUNNER))
        self.assertEqual([item["fits_static_preflight"]
                          for item in plan["static_memory_preflight"]], [True, True])

    def test_capacity_preview_selects_full_resident_b04_workload(self) -> None:
        plan = serve_launcher._make_plan(
            "preview", self.uuids, "capacity-test", workload_name="resident_capacity")
        self.assertEqual(plan["workload"], "resident_capacity")
        self.assertEqual(plan["bundle"], "B04")
        self.assertEqual(plan["controls"], ["native_pair", "facade_owner"])
        self.assertEqual(plan["bounded_seconds"], 300)
        self.assertEqual(plan["controller_timeout_seconds"], 315)
        self.assertEqual(plan["controller_spec"]["argv"][1],
                         str(serve_launcher.BOUNDED_ADAPTER))
        self.assertIn("resident_capacity", plan["bounded_child_argv"])
        self.assertTrue(all(item["fits_static_preflight"]
                            for item in plan["static_memory_preflight"]))

    def test_full_resident_workload_runs_as_bounded_foreground_service(self) -> None:
        plan = serve_launcher._make_plan(
            "serve", self.uuids, "resident-serve", duration_seconds=120, port=8124,
            workload_name="resident_capacity")
        self.assertEqual(plan["controls"], ["facade_owner"])
        self.assertEqual(plan["bundle"], "V200-SERVE")
        self.assertEqual(plan["workload"], "resident_capacity")
        self.assertEqual(plan["endpoint"], "http://127.0.0.1:8124")
        self.assertEqual(plan["duration_seconds"], 120)
        self.assertTrue(plan["static_memory_preflight"][0]["fits_static_preflight"])
        self.assertIn("resident_capacity", plan["bounded_child_argv"])

    def test_pair_compute_is_explicit_and_preloads_operators_only_when_enabled(self) -> None:
        default = serve_launcher._make_plan("preview", self.uuids, "native-default")
        self.assertFalse(default["pair_compute"])
        self.assertNotIn(str(serve_launcher.PAIR_OPERATORS_DSO), default["preload_libraries"])
        self.assertNotIn("V200_PAIR_COMPUTE=1", default["bounded_child_argv"])
        enabled = serve_launcher._make_plan(
            "preview", self.uuids, "pair-compute", workload_name="resident_capacity",
            pair_compute=True, verify_pair_compute=True)
        self.assertTrue(enabled["pair_compute"])
        self.assertTrue(enabled["operator_verification"])
        self.assertEqual(enabled["preload_libraries"][-1], str(serve_launcher.PAIR_OPERATORS_DSO))
        self.assertIn("build/src/operators/libv200_operators.so",
                      enabled["controller_spec"]["binary_paths"])
        self.assertIn(str(serve_launcher.PAIR_OPERATORS_DSO), enabled["required_files"])
        self.assertIn("V200_PAIR_COMPUTE=1", enabled["bounded_child_argv"])
        self.assertIn("V200_OPERATOR_VERIFY=1", enabled["bounded_child_argv"])
        self.assertIn("-u", enabled["bounded_child_argv"])
        with self.assertRaisesRegex(serve_launcher.LaunchError, "requires --pair-compute"):
            serve_launcher._make_plan("preview", self.uuids, "invalid-verify",
                                      verify_pair_compute=True)

    def test_cli_exposes_pair_compute_opt_in(self) -> None:
        args = serve_launcher._parse_args(["--pair-compute", "--workload", "resident_capacity"])
        self.assertTrue(args.pair_compute)
        self.assertEqual(args.workload, "resident_capacity")

    def test_extended_expert_opt_in_selects_multipair_build_only_when_requested(self) -> None:
        with self.assertRaisesRegex(serve_launcher.LaunchError, "requires --pair-compute"):
            serve_launcher._make_plan("serve", self.uuids, "extended-without-pair",
                                      extended_expert_compute=True)

        plan = serve_launcher._make_plan(
            "serve", self.uuids, "extended-expert", duration_seconds=120,
            workload_name="resident_capacity", pair_compute=True,
            verify_pair_compute=True, extended_expert_compute=True)
        self.assertEqual(plan["build_directory"], "build-multipair")
        self.assertTrue(plan["extended_expert_compute"])
        self.assertEqual(plan["preload_libraries"], [
            str(serve_launcher.REPO_ROOT / "build-multipair/src/frontdoor/libv200_frontdoor.so"),
            str(serve_launcher.REPO_ROOT / "build-multipair/src/integration/libv200_pair_init.so"),
            str(serve_launcher.REPO_ROOT / "build-multipair/src/operators/libv200_operators.so"),
        ])
        self.assertEqual(plan["controller_spec"]["binary_paths"], [
            "build-multipair/src/frontdoor/libv200_frontdoor.so",
            "build-multipair/src/integration/libv200_pair_init.so",
            "build-multipair/src/operators/libv200_operators.so",
        ])
        self.assertTrue(all(path.startswith(str(serve_launcher.REPO_ROOT / "build-multipair"))
                            for path in plan["required_files"]
                            if "libv200_" in path))
        self.assertIn("V200_PAIR_COMPUTE=1", plan["bounded_child_argv"])
        self.assertIn("V200_OPERATOR_VERIFY=1", plan["bounded_child_argv"])
        self.assertIn("V200_EXTENDED_EXPERT_COMPUTE=1", plan["bounded_child_argv"])
        printed = io.StringIO()
        with redirect_stdout(printed):
            serve_launcher._print_plan(plan, [], execute=False)
        self.assertIn("Build outputs: build-multipair", printed.getvalue())

        args = serve_launcher._parse_args(["--pair-compute", "--extended-expert-compute"])
        self.assertTrue(args.pair_compute)
        self.assertTrue(args.extended_expert_compute)

    def test_local_hbm_preference_requires_extended_compute_and_selects_perf_dsos(self) -> None:
        with self.assertRaisesRegex(serve_launcher.LaunchError, "requires --pair-compute"):
            serve_launcher._make_plan("serve", self.uuids, "local-no-pair",
                                      prefer_local_hbm=True)
        with self.assertRaisesRegex(serve_launcher.LaunchError, "requires --extended-expert-compute"):
            serve_launcher._make_plan("serve", self.uuids, "local-no-extended",
                                      pair_compute=True, prefer_local_hbm=True)

        plan = serve_launcher._make_plan(
            "serve", self.uuids, "local-hbm", pair_compute=True,
            extended_expert_compute=True, prefer_local_hbm=True)
        self.assertTrue(plan["prefer_local_hbm"])
        self.assertEqual(plan["build_directory"], "build-perf")
        self.assertEqual(plan["preload_libraries"], [
            str(serve_launcher.REPO_ROOT / "build-perf/src/frontdoor/libv200_frontdoor.so"),
            str(serve_launcher.REPO_ROOT / "build-perf/src/integration/libv200_pair_init.so"),
            str(serve_launcher.REPO_ROOT / "build-perf/src/operators/libv200_operators.so"),
        ])
        self.assertEqual(plan["controller_spec"]["binary_paths"], [
            "build-perf/src/frontdoor/libv200_frontdoor.so",
            "build-perf/src/integration/libv200_pair_init.so",
            "build-perf/src/operators/libv200_operators.so",
        ])
        self.assertIn("src/memory/manager.cpp", plan["controller_spec"]["paths"])
        self.assertTrue(all(path.startswith(str(serve_launcher.REPO_ROOT / "build-perf"))
                            for path in plan["required_files"] if "libv200_" in path))
        self.assertIn("-u", plan["bounded_child_argv"])
        self.assertIn("V200_PREFER_LOCAL_HBM=1", plan["bounded_child_argv"])
        printed = io.StringIO()
        with redirect_stdout(printed):
            serve_launcher._print_plan(plan, [], execute=False)
        self.assertIn("Build outputs: build-perf", printed.getvalue())
        self.assertIn("Local HBM preference: enabled", printed.getvalue())

        args = serve_launcher._parse_args([
            "--pair-compute", "--extended-expert-compute", "--prefer-local-hbm"])
        self.assertTrue(args.prefer_local_hbm)

    def test_prefer_local_hbm_cli_rejects_missing_prerequisites_before_profile(self) -> None:
        with mock.patch.object(serve_launcher, "_load_selected_pair") as load_pair, \
             redirect_stderr(io.StringIO()):
            self.assertEqual(serve_launcher.main(["--prefer-local-hbm"]), 2)
        load_pair.assert_not_called()

    def test_extended_expert_cli_requires_pair_compute_before_loading_hardware_profile(self) -> None:
        with mock.patch.object(serve_launcher, "_load_selected_pair") as load_pair, \
             redirect_stderr(io.StringIO()):
            self.assertEqual(serve_launcher.main(["--extended-expert-compute"]), 2)
        load_pair.assert_not_called()

    def test_cuda_graph_opt_in_is_explicit_and_keeps_default_graph_off(self) -> None:
        default = serve_launcher._make_plan("preview", self.uuids, "graph-default")
        self.assertFalse(default["cuda_graphs_enabled"])
        self.assertNotIn("--cuda-graphs", default["bounded_child_argv"])
        enabled = serve_launcher._make_plan(
            "preview", self.uuids, "graph-enabled", workload_name="resident_capacity",
            pair_compute=True, cuda_graphs_enabled=True)
        self.assertTrue(enabled["cuda_graphs_enabled"])
        self.assertIn("--cuda-graphs", enabled["bounded_child_argv"])
        self.assertIn("V200_PAIR_COMPUTE=1", enabled["bounded_child_argv"])
        self.assertIn("src/operators/graph_rewrite.cpp", enabled["controller_spec"]["paths"])
        self.assertIn("opt-in enabled", enabled["graph_policy"])
        args = serve_launcher._parse_args(["--cuda-graphs", "--pair-compute"])
        self.assertTrue(args.cuda_graphs)
        self.assertTrue(args.pair_compute)

    def test_service_duration_and_pair_overrides_fail_closed(self) -> None:
        with self.assertRaisesRegex(serve_launcher.LaunchError, "between 1 and 3000"):
            serve_launcher._make_plan("serve", self.uuids, "too-long",
                                      duration_seconds=3601, port=8123)
        with self.assertRaisesRegex(serve_launcher.LaunchError, "exact ordered UUID pair"):
            serve_launcher._parse_gpu_option("GPU-not-the-pair,GPU-other", self.uuids)


class OracleTests(unittest.TestCase):
    def setUp(self) -> None:
        fixture, _ = consumer.load_documents()
        self.oracle = fixture["workloads"]["smoke"]["oracle"]

    def test_exact_normalized_output_passes(self) -> None:
        result = consumer.validate_completion_response(
            {"content": " 2\n", "tokens_predicted": 1}, self.oracle)
        self.assertTrue(result["exact_match"])
        self.assertEqual(result["normalized_content"], "2")

    def test_nonempty_wrong_answer_does_not_pass(self) -> None:
        with self.assertRaisesRegex(consumer.ConsumerError, "exact output mismatch"):
            consumer.validate_completion_response(
                {"content": "The answer is 2", "tokens_predicted": 5}, self.oracle)

    def test_malformed_content_or_token_count_does_not_pass(self) -> None:
        for response in ({"content": "2"}, {"content": 2, "tokens_predicted": 1},
                         {"content": "2", "tokens_predicted": True},
                         {"content": "2", "tokens_predicted": 9}):
            with self.subTest(response=response), self.assertRaises(consumer.ConsumerError):
                consumer.validate_completion_response(response, self.oracle)

    def test_resident_capacity_requires_all_llama_layers_in_gpu_log(self) -> None:
        fixture, _ = consumer.load_documents()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "llama.log"
            path.write_text("llama_model_load: offloaded 49/49 layers to GPU\n")
            full = consumer._layer_offload_evidence(path, "resident_capacity", fixture)
            self.assertEqual(full["status"], "PASS")
            self.assertTrue(full["full_gpu_offload"])
            path.write_text("llama_model_load: offloaded 28/49 layers to GPU\n")
            partial = consumer._layer_offload_evidence(path, "resident_capacity", fixture)
            self.assertEqual(partial["status"], "PARTIAL_OFFLOAD")

    def test_input_embedding_override_must_be_observed_in_cuda_buffer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "llama.log"
            path.write_text("tensor token_embd.weight (256 MiB Q4_K) buffer type overridden to CUDA0\n")
            gpu = consumer._input_embedding_evidence(path, "resident_capacity")
            self.assertEqual(gpu["status"], "PASS")
            path.write_text("tensor token_embd.weight (256 MiB Q4_K) buffer type overridden to CPU\n")
            cpu = consumer._input_embedding_evidence(path, "resident_capacity")
            self.assertEqual(cpu["status"], "WRONG_BUFFER_TYPE")

    def test_completion_request_has_total_wall_clock_deadline(self) -> None:
        def slow_request(*_args: object, **_kwargs: object) -> tuple[int, object]:
            time.sleep(1.0)
            return 200, {"content": "2", "tokens_predicted": 1}

        with mock.patch.object(consumer, "_http_json", side_effect=slow_request):
            with self.assertRaises(consumer.RequestWallTimeout):
                consumer._completion_request("http://127.0.0.1/unused", {}, 0.05)


class ProcessLifetimeTests(unittest.TestCase):
    def test_bounded_adapter_allows_nested_foreground_cleanup_before_escalation(self) -> None:
        source = """
import os, signal, subprocess, sys, time
nested = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], start_new_session=True)
print(nested.pid, flush=True)
def stop(_signum, _frame):
    try:
        os.killpg(nested.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    nested.wait(timeout=5)
    raise SystemExit(143)
signal.signal(signal.SIGTERM, stop)
while True:
    time.sleep(0.1)
"""
        with tempfile.TemporaryDirectory() as directory:
            result = bounded_runner.run_child(
                [sys.executable, "-u", "-c", source], 0.25, Path(directory), dict(os.environ))
            stdout = (Path(directory) / "stdout.txt").read_text()
            nested_pid = int(stdout.strip())
            self.assertTrue(result["timed_out"])
            self.assertFalse(result["interrupted"])
            self.assertFalse(result["termination_uncertain"])
            self.assertFalse(result["orphaned_children"])
            self.assertFalse(self._process_is_live(nested_pid))
            self.assertTrue(consumer._is_sha(result["stdout_sha256"]))
            self.assertTrue(consumer._is_sha(result["stderr_sha256"]))

    def test_fake_http_server_and_child_are_cleaned_as_owned_process_group(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            child_pid_path = root / "child.pid"
            log_path = root / "fake-server.log"
            port = consumer._free_loopback_port()
            env = dict(os.environ)
            env["V200_FAKE_CHILD_PID_FILE"] = str(child_pid_path)
            with log_path.open("wb") as log:
                proc = subprocess.Popen([sys.executable, "-u", str(Path(__file__).with_name("fake_server.py")),
                                         str(port)], env=env, stdout=log, stderr=subprocess.STDOUT,
                                        start_new_session=True)
                try:
                    elapsed = consumer.wait_for_readiness(
                        proc, f"http://127.0.0.1:{port}", timeout_s=4.0)
                    self.assertGreaterEqual(elapsed, 0.0)
                    status, response = consumer._http_json(
                        "POST", f"http://127.0.0.1:{port}/completion",
                        {"prompt": "cpu test"}, timeout_s=2.0)
                    self.assertEqual(status, 200)
                    oracle = {"response_field": "content",
                              "normalization": "strip surrounding Unicode whitespace only",
                              "expected_exact": "2", "tokens_predicted_min": 1,
                              "tokens_predicted_max": 8}
                    self.assertTrue(consumer.validate_completion_response(response, oracle)["exact_match"])
                    self.assertTrue(child_pid_path.is_file())
                    child_pid = int(child_pid_path.read_text())
                    cleanup = consumer.cleanup_owned_group(proc, shutdown_timeout_s=3.0)
                    self.assertTrue(cleanup["no_live_group_members"], cleanup)
                    self.assertEqual(cleanup["owned_process_group"], proc.pid)
                    self.assertEqual(cleanup["owned_session"], proc.pid)
                    self.assertFalse(self._process_is_live(child_pid))
                finally:
                    if proc.poll() is None:
                        consumer.cleanup_owned_group(proc, shutdown_timeout_s=2.0)

    def test_readiness_fails_when_process_exits_before_ready(self) -> None:
        proc = subprocess.Popen(["/bin/true"], start_new_session=True)
        with self.assertRaisesRegex(consumer.ConsumerError, "exited before readiness"):
            consumer.wait_for_readiness(proc, "http://127.0.0.1:1", timeout_s=1.0)
        proc.wait(timeout=1)
        cleanup = consumer.cleanup_owned_group(proc, shutdown_timeout_s=0.5)
        self.assertTrue(cleanup["no_live_group_members"])

    def test_finite_runner_writes_exact_result_without_gpu_or_monitor(self) -> None:
        fixture, profile = consumer.load_documents()
        fake = Path(__file__).with_name("fake_server.py")

        def fake_popen(_argv: list[str], **kwargs: object) -> subprocess.Popen[bytes]:
            args = _argv
            port = args[args.index("--port") + 1]
            return subprocess.Popen([sys.executable, "-u", str(fake), port], **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            result = consumer._run_one(
                "smoke", "native_single", fixture, profile, {"test_identity": "fake"},
                ["GPU-aaaaaaaa"], Path(directory), preloads=None, monitor=False,
                environ=dict(os.environ), popen_factory=fake_popen)
            self.assertEqual(result["application_status"], "PASS")
            self.assertEqual(result["gpu_qualification"], "NOT_CLAIMED")
            self.assertTrue(result["partial_offload_preview"])
            self.assertEqual(result["preview_environment"]["GGML_CUDA_DISABLE_GRAPHS"], "1")
            self.assertEqual(result["response"]["normalized_content"], "2")
            self.assertEqual(result["memory_guard"]["telemetry_status"], "not-started")
            self.assertTrue(result["cleanup"]["no_live_group_members"])
            saved = json.loads(Path(result["result_path"]).read_text())
            self.assertEqual(saved["application_status"], "PASS")
            self.assertEqual(saved["allocation_evidence"]["native_limit_enforcement"],
                             "descriptor-based preflight only; no native allocator ledger")

    def test_correct_response_does_not_hide_failed_server_teardown(self) -> None:
        fixture, profile = consumer.load_documents()
        fake = Path(__file__).with_name("fake_server.py")

        def fake_popen(argv: list[str], **kwargs: object) -> subprocess.Popen[bytes]:
            port = argv[argv.index("--port") + 1]
            return subprocess.Popen([sys.executable, "-u", str(fake), port], **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            result = consumer._run_one(
                "smoke", "native_single", fixture, profile, {"test_identity": "fake"},
                ["GPU-aaaaaaaa"], Path(directory), preloads=None, monitor=False,
                environ={**os.environ, "V200_FAKE_EXIT_ON_TERM": "1"},
                popen_factory=fake_popen)
            self.assertEqual(result["response"]["normalized_content"], "2")
            self.assertTrue(result["cleanup"]["no_live_group_members"])
            self.assertEqual(result["server_process"]["exit_code"], 139)
            self.assertEqual(result["application_status"], "FAIL")
            self.assertIn("exited unsuccessfully", result["failure"])
            saved = json.loads(Path(result["result_path"]).read_text())
            self.assertEqual(saved["application_status"], "FAIL")

    def test_finite_serve_mode_keeps_verified_loopback_server_alive_then_cleans_it(self) -> None:
        fixture, profile = consumer.load_documents()
        fake = Path(__file__).with_name("fake_server.py")

        def fake_popen(_argv: list[str], **kwargs: object) -> subprocess.Popen[bytes]:
            return subprocess.Popen([sys.executable, "-u", str(fake), "19080"], **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            result = consumer._run_one(
                "smoke", "native_single", fixture, profile, {"test_identity": "fake"},
                ["GPU-aaaaaaaa"], Path(directory), preloads=None, monitor=False,
                environ=dict(os.environ), popen_factory=fake_popen, server_port=19080,
                serve_duration_seconds=1)
            self.assertEqual(result["application_status"], "PASS")
            self.assertEqual(result["service"]["endpoint"], "http://127.0.0.1:19080")
            self.assertEqual(result["service"]["duration_requested_seconds"], 1)
            self.assertGreaterEqual(result["service"]["duration_observed_seconds"], 1)
            self.assertEqual(result["service"]["status"], "PASS")
            self.assertTrue(result["cleanup"]["no_live_group_members"])
            self.assertFalse(result["cleanup"]["sigkill_escalation"],
                             "SIGTERM must reach the child after the parent masks it for Popen assignment")

    def test_process_memory_guard_stops_only_its_fake_owned_group(self) -> None:
        fake = Path(__file__).with_name("fake_server.py")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            smi = root / "fake-nvidia-smi"
            smi.write_text("#!/bin/sh\nprintf '0, GPU-aaaaaaaa, 20000\\n'\n")
            smi.chmod(0o755)
            env = dict(os.environ)
            with (root / "server.log").open("wb") as log:
                proc = subprocess.Popen([sys.executable, "-u", str(fake), "0"], env=env,
                                        stdout=log, stderr=subprocess.STDOUT,
                                        start_new_session=True)
                smi.write_text(f"#!/bin/sh\nprintf '{proc.pid}, GPU-aaaaaaaa, 20000\\n'\n")
                smi.chmod(0o755)
                # Replace the child PID used by the synthetic telemetry row.
                guard = consumer.ProcessMemoryGuard(proc, ["GPU-aaaaaaaa"], 14336,
                                                    0.05, str(smi))
                try:
                    guard.start()
                    deadline = time.monotonic() + 3.0
                    while time.monotonic() < deadline and guard.over_limit is None and guard.error is None:
                        time.sleep(0.025)
                    self.assertIsNotNone(guard.over_limit)
                    self.assertEqual(guard.over_limit["gpu_uuid"], "GPU-aaaaaaaa")
                    self.assertEqual(guard.over_limit["used_memory_mib"], 20000)
                    proc.wait(timeout=3)
                finally:
                    guard.finish()
                    cleanup = consumer.cleanup_owned_group(proc, shutdown_timeout_s=1.0)
                self.assertTrue(cleanup["no_live_group_members"])
                self.assertEqual(cleanup["owned_process_group"], proc.pid)

    @staticmethod
    def _process_is_live(pid: int) -> bool:
        try:
            stat = Path(f"/proc/{pid}/stat").read_text()
            state = stat[stat.rfind(")") + 2:].split()[0]
            return state != "Z"
        except FileNotFoundError:
            return False


if __name__ == "__main__":
    unittest.main(verbosity=2)
