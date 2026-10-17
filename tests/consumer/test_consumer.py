#!/usr/bin/env python3
"""Offline fixture/oracle/process tests; no llama binary or GPU is launched."""

from __future__ import annotations

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

    def test_child_environment_clears_hidden_fit_and_keeps_controller_selection(self) -> None:
        env = {"PATH": "/bin", "CUDA_VISIBLE_DEVICES": "GPU-a,GPU-b",
               "V200_DEVICE_UUIDS": "GPU-aaaaaaaa,GPU-bbbbbbbb",
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
            self.assertEqual(facade["LD_PRELOAD"],
                             f"{preload}:{initializer}:/tmp/libunrelated.so")

    def test_cli_preserves_single_preload_and_accepts_initializer_as_second(self) -> None:
        args = consumer._parse_args([
            "--workload", "smoke", "--control", "facade_owner",
            "--preload-library", "/tmp/libv200_frontdoor.so",
            "--preload-library", "/tmp/libv200_pair_init.so",
            "--output-dir", "/tmp/v200-test", "--execute"])
        self.assertEqual([path.name for path in args.preloads],
                         ["libv200_frontdoor.so", "libv200_pair_init.so"])

    def test_pair_profile_log_is_hash_recorded_and_budget_checked(self) -> None:
        budget = 12288 * 1024 * 1024
        record = {"schema": "v200-frontdoor-ledger/v1", "event": "snapshot",
                  "mode": "pair", "anchor_uuid": "GPU-aaaaaaaa",
                  "peer_uuid": "GPU-bbbbbbbb", "qualification_id": "q-1",
                  "owner_budget_bytes": [budget, budget], "owner_used_bytes": [100, 200],
                  "live_allocation_count": 1, "live_reservation_count": 0,
                  "cleanup_pending_count": 0}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.jsonl"
            path.write_text(json.dumps(record) + "\n")
            result = consumer._read_profile_log(path,
                                                ["GPU-aaaaaaaa", "GPU-bbbbbbbb"], budget)
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(result["records"], [record])
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
        pinned_llama = consumer.load_documents()[1]["binary"]["path"]
        self.assertNotIn(pinned_llama, plan["controller_spec"]["binary_paths"])
        self.assertEqual(plan["controller_spec"]["argv"][1],
                         str(serve_launcher.BOUNDED_ADAPTER))
        self.assertIn("V200_DEVICE_UUIDS=" + ",".join(self.uuids),
                      plan["bounded_child_argv"])
        self.assertEqual(plan["static_memory_preflight"][0]["control"], "facade_owner")
        self.assertTrue(plan["static_memory_preflight"][0]["fits_static_preflight"])
        default_plan = serve_launcher._make_plan("serve", self.uuids, "default-serve")
        self.assertEqual(default_plan["duration_seconds"], serve_launcher.SERVE_DEFAULT_SECONDS)
        self.assertEqual(default_plan["endpoint"], "http://127.0.0.1:8080")

    def test_default_cli_selects_finite_serve_but_keeps_explicit_execution_gate(self) -> None:
        args = serve_launcher._parse_args([])
        self.assertEqual(args.mode, "serve")
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
