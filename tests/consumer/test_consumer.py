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
import run_consumer as consumer  # noqa: E402


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

    def test_child_environment_clears_hidden_fit_and_keeps_controller_selection(self) -> None:
        env = {"PATH": "/bin", "CUDA_VISIBLE_DEVICES": "GPU-a,GPU-b",
               "V200_DEVICE_UUIDS": "GPU-aaaaaaaa,GPU-bbbbbbbb",
               "LLAMA_ARG_N_GPU_LAYERS": "999", "GGML_CUDA_FORCE_CUBLAS": "1",
               "LD_PRELOAD": "/tmp/libv200_frontdoor.so /tmp/libunrelated.so"}
        native = consumer._clean_environment("native_single", self.fixture, None, env,
                                             selected_uuids=["GPU-aaaaaaaa"])
        self.assertNotIn("LLAMA_ARG_N_GPU_LAYERS", native)
        self.assertNotIn("GGML_CUDA_FORCE_CUBLAS", native)
        self.assertNotIn("V200_MODE", native)
        self.assertNotIn("V200_DEVICE_UUIDS", native)
        self.assertEqual(native["CUDA_VISIBLE_DEVICES"], "GPU-a,GPU-b")
        self.assertEqual(native["LD_PRELOAD"], "/tmp/libunrelated.so")
        with tempfile.TemporaryDirectory() as directory:
            preload = Path(directory) / "libv200_frontdoor.so"
            preload.touch()
            facade = consumer._clean_environment("facade_owner", self.fixture, [preload], env,
                                                 facade_mode="anchor",
                                                 selected_uuids=["GPU-aaaaaaaa"])
            self.assertEqual(facade["V200_MODE"], "anchor")
            self.assertEqual(facade["V200_DEVICE_UUIDS"], "GPU-aaaaaaaa")
            self.assertEqual(facade["LD_PRELOAD"], f"{preload}:/tmp/libunrelated.so")

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
            self.assertEqual(result["response"]["normalized_content"], "2")
            self.assertEqual(result["memory_guard"]["telemetry_status"], "not-started")
            self.assertTrue(result["cleanup"]["no_live_group_members"])
            saved = json.loads(Path(result["result_path"]).read_text())
            self.assertEqual(saved["application_status"], "PASS")
            self.assertEqual(saved["allocation_evidence"]["native_limit_enforcement"],
                             "descriptor-based preflight only; no native allocator ledger")

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
