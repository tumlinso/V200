#!/usr/bin/env python3
"""Artifact-only acceptance gate for the first V200 consumer evidence.

This program reads retained JSON, text, and binary hashes only.  It does not
import CUDA bindings, contact a controller, enumerate devices, or launch a
consumer.  Its acceptance scope is deliberately limited to the named B01,
B02, and B03 evidence already captured under this repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any


CANONICAL_ROOT = Path("/home/tumlinson/V200")
PAIR = [
    "GPU-21131915-1488-23af-38dd-1743ae1f5cc8",
    "GPU-d745da9c-7649-8334-41f1-483afa6f3206",
]
B01_ID = "8426eab5-278d-4687-ab7d-f5ec937e46ec"
RUNS = {
    "B01": ("evidence/foundation/B01", B01_ID),
    "B02 frontdoor": (
        "evidence/B02/frontdoor",
        "2c6cad1b-d526-4c1f-837b-a1a66b8d13ba",
    ),
    "B02 pair-memory-r2": (
        "evidence/B02/pair-memory-r2",
        "63a510fd-21c4-4960-84ca-572232fea914",
    ),
    "B02 execution-r2": (
        "evidence/B02/execution-r2",
        "f9e1d6eb-599e-41f0-ab7e-54ce164b9c05",
    ),
    "B03 native": ("evidence/B03/native", "f2b89fd2-0a43-4dc4-82e8-56fade091835"),
    "B03 facade": ("evidence/B03/facade", "aa82e8d5-f6a4-4f62-b5d6-4e56acc53c96"),
}
CONSUMERS = {
    "native": "evidence/B03/native/consumer/smoke-native_single-1792239954218618376.json",
    "facade": "evidence/B03/facade/consumer/smoke-facade_owner-1792240609253615134.json",
}


class Gate:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.checks: list[str] = []
        self.failures: list[str] = []

    def require(self, ok: bool, name: str, detail: str | None = None) -> None:
        if ok:
            self.checks.append(name)
        else:
            self.failures.append(name + (f": {detail}" if detail else ""))

    def path(self, recorded: str | Path) -> Path:
        path = Path(recorded)
        try:
            rel = path.resolve().relative_to(CANONICAL_ROOT)
        except (ValueError, OSError):
            return path
        return self.root / rel

    def json(self, path: str | Path) -> dict[str, Any]:
        value = json.loads(self.path(path).read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"expected JSON object: {path}")
        return value

    def sha256(self, path: str | Path) -> str:
        digest = hashlib.sha256()
        with self.path(path).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def check_hash(self, path: str | Path, expected: str, label: str) -> None:
        try:
            actual = self.sha256(path)
            self.require(actual == expected, label, f"expected {expected}, got {actual}")
        except (OSError, ValueError) as exc:
            self.require(False, label, str(exc))


def _parse_env(argv: list[str]) -> tuple[dict[str, str], list[str]]:
    if not argv or Path(argv[0]).name != "env":
        return {}, argv
    env: dict[str, str] = {}
    index = 1
    while index < len(argv) and "=" in argv[index] and not argv[index].startswith("/"):
        key, value = argv[index].split("=", 1)
        if not key or key.startswith("-"):
            break
        env[key] = value
        index += 1
    return env, argv[index:]


def verify_run(g: Gate, label: str, relbase: str, expected_id: str,
               controller_stem: str | None = None,
               expected_bundle: str | None = None) -> dict[str, Any]:
    base = Path(relbase)
    stem = Path(controller_stem) if controller_stem else base
    result = g.json(stem.with_name(stem.name + ".controller-result.json"))
    wrapped = g.json(base / "execution.json")
    evidence_doc = g.json(stem.with_name(stem.name + ".controller-evidence.json"))
    release = g.json(stem.with_name(stem.name + ".host-release.json"))
    evidence = evidence_doc["evidence"]

    g.require(result.get("ok") is True and result.get("status") == "succeeded"
              and result.get("returncode") == 0, f"{label}: controller result succeeded")
    g.require(result.get("evidence_id") == expected_id == evidence.get("id"),
              f"{label}: canonical evidence id")
    g.require(evidence.get("status") == "succeeded"
              and evidence.get("kind") == "foreground-baseline"
              and evidence.get("severity") == 0 and evidence.get("contaminated") == 0,
              f"{label}: controller evidence clean")
    bundle = expected_bundle or ("B01" if label == "B01" else "B02" if label.startswith("B02") else "B03")
    g.require(wrapped.get("bundle") == bundle
              and wrapped.get("status") == "CHILD_COMPLETED"
              and wrapped.get("returncode") == 0
              and not any(wrapped.get(key) for key in
                          ("timed_out", "interrupted", "orphaned_children", "termination_uncertain")),
              f"{label}: bounded wrapper completed cleanly")

    for key, local_name in (("stdout_sha256", "stdout.txt"), ("stderr_sha256", "stderr.txt")):
        g.check_hash(base / local_name, wrapped.get(key, ""), f"{label}: wrapper {local_name} hash")

    artifacts = {item.get("kind"): item for item in evidence.get("artifacts", [])}
    for kind, result_path_key in (("stdout", "stdout_path"), ("stderr", "stderr_path")):
        artifact = artifacts.get(kind, {})
        raw_path = artifact.get("path", "")
        expected_hash = artifact.get("content_hash", "")
        g.require(artifact.get("complete") == 1 and bool(raw_path),
                  f"{label}: controller {kind} artifact complete")
        g.require(result.get(result_path_key) == raw_path,
                  f"{label}: controller result {kind} path linked")
        g.check_hash(raw_path, expected_hash, f"{label}: controller raw {kind} hash")
        # The controller foreground stream is the bounded-runner wrapper's
        # stdout, while execution.json records the child stream saved beside
        # the run.  Hash each independently; they are different artifacts.

    receipt_path = wrapped.get("lease", {}).get("receipt", "")
    g.require(bool(receipt_path) and result.get("lease_receipt") == receipt_path,
              f"{label}: controller lease path linked")
    try:
        receipt = g.json(receipt_path)
    except (OSError, ValueError, KeyError) as exc:
        receipt = {}
        g.require(False, f"{label}: archived active lease readable", str(exc))
    uuids = wrapped.get("lease", {}).get("visible_uuids", [])
    expected_resources = [f"accelerator:{uuid}" for uuid in uuids]
    owner_id = receipt.get("owner_id")
    pid = wrapped.get("lease", {}).get("owner_pid")
    g.require(receipt.get("format") == "CUDA-FOREGROUND-LEASE/1"
              and receipt.get("state") == "active"
              and receipt.get("project_root") == str(CANONICAL_ROOT)
              and receipt.get("resource_ids") == expected_resources
              and uuids == PAIR,
              f"{label}: archived lease project and ordered UUIDs")
    g.require(owner_id == release.get("id")
              and receipt.get("pid") == pid == release.get("pid")
              and release.get("project_root") == str(CANONICAL_ROOT)
              and release.get("owner_kind") == "foreground"
              and release.get("state") == "released"
              and release.get("resources") == [],
              f"{label}: owner release linked and empty")

    snapshot = evidence.get("snapshot", {})
    provenance_source = evidence.get("summary", {}).get("provenance", {}).get("source", {})
    g.require(snapshot == provenance_source
              and snapshot.get("project_root") == str(CANONICAL_ROOT)
              and snapshot.get("safe") is True
              and snapshot.get("fingerprint") == evidence.get("source_fingerprint"),
              f"{label}: retained source snapshot identity coherent")
    toolchain = evidence.get("summary", {}).get("toolchain", {})
    g.require(toolchain.get("version") == "12.9"
              and toolchain.get("root", "").endswith("/cuda/12.9")
              and toolchain.get("environment", {}).get("CUDAToolkit_ROOT", "").endswith("/cuda/12.9")
              and toolchain.get("nvcc", "").endswith("/cuda/12.9/bin/nvcc"),
              f"{label}: retained CUDA 12.9 toolchain identity")
    source_root = snapshot.get("source_root", "")
    source_path = g.path(source_root)
    g.require(bool(source_root) and source_path.is_dir()
              and source_path.name == "source"
              and source_path.parent.name == snapshot.get("fingerprint"),
              f"{label}: immutable source snapshot exists")
    for item in snapshot.get("untracked", []):
        relpath, digest = item.get("path", ""), item.get("sha256", "")
        g.check_hash(source_path / relpath, digest, f"{label}: archived untracked source {relpath}")
    return {"evidence_id": expected_id, "visible_uuids": uuids}


def _run_stdout(g: Gate, relbase: str) -> str:
    return g.path(Path(relbase) / "stdout.txt").read_text(encoding="utf-8")


def _controller_summary(g: Gate, relbase: str) -> dict[str, Any]:
    return g.json(Path(relbase).with_name(Path(relbase).name + ".controller-evidence.json"))["evidence"]["summary"]


def verify_b01(g: Gate) -> None:
    profile = g.json("evidence/foundation/pair-profile.json")
    g.require(profile.get("status") == "GPU_PASS"
              and profile.get("evidence_id") == B01_ID
              and profile.get("physical_uuids") == PAIR
              and profile.get("host_reservation") == "released"
              and profile.get("fixture", {}).get("status") == "GPU_PASS"
              and profile.get("fixture", {}).get("qualification_only") is True
              and profile.get("consumer_compatibility") == "NOT_RUN"
              and profile.get("performance") == "NOT_MEASURED",
              "B01: accepted pair reference and limited claim")
    source_hashes = profile.get("source_sha256", {})
    for relpath, expected in source_hashes.items():
        g.check_hash(relpath, expected, f"B01: source hash {relpath}")


def verify_b02(g: Gate) -> None:
    front = g.json("evidence/B02/frontdoor/execution.json")
    env, command = _parse_env(front.get("argv", []))
    front_uuids = env.get("V200_DEVICE_UUIDS", "").split(",")
    g.require(env.get("V200_MODE") == "pair" and front_uuids == PAIR
              and env.get("V200_B01_EVIDENCE_ID") == B01_ID
              and env.get("V200_OWNER0_BUDGET_MIB") == "64"
              and env.get("V200_OWNER1_BUDGET_MIB") == "64",
              "B02 frontdoor: pair mode, B01 anchor, and owner budgets")
    g.require("v200_frontdoor_closure" in " ".join(command)
              and "--execute" in command
              and front.get("allocation_cap_mib_per_gpu") == 1024,
              "B02 frontdoor: bounded closure command")
    g.require("controller-backed frontdoor CUDA closure checks passed" in
              _run_stdout(g, "evidence/B02/frontdoor"),
              "B02 frontdoor: closure output")
    front_profile = Path(env.get("V200_PROFILE_LOG", ""))
    try:
        rows = [json.loads(line) for line in g.path(front_profile).read_text().splitlines() if line]
    except (OSError, ValueError) as exc:
        rows = []
        g.require(False, "B02 frontdoor: readable profile ledger", str(exc))
    g.require(bool(rows) and rows[0].get("event") == "profile_initialized"
              and any(any(value > 0 for value in row.get("owner_used_bytes", []))
                      or row.get("live_allocation_count", 0) > 0 for row in rows)
              and rows[-1].get("event") == "abi_closure_complete"
              and rows[-1].get("owner_used_bytes") == [0, 0]
              and rows[-1].get("live_allocation_count") == 0
              and rows[-1].get("live_reservation_count") == 0
              and rows[-1].get("cleanup_pending_count") == 0
              and all(row.get("mode") == "pair" and row.get("qualification_id") == B01_ID
                      and row.get("anchor_uuid") == PAIR[0] and row.get("peer_uuid") == PAIR[1]
                      for row in rows),
              "B02 frontdoor: positive allocation and zero closure ledger")

    mem = g.json("evidence/B02/pair-memory-r2/execution.json")
    env, command = _parse_env(mem.get("argv", []))
    g.require(env.get("V200_MODE") == "pair"
              and env.get("V200_DEVICE_UUIDS", "").split(",") == PAIR
              and env.get("V200_B01_EVIDENCE_ID") == B01_ID
              and env.get("V200_OWNER0_BUDGET_MIB") == "64"
              and env.get("V200_OWNER1_BUDGET_MIB") == "64"
              and mem.get("allocation_cap_mib_per_gpu") == 1024
              and "v200_pair_memory_gpu_smoke" in " ".join(command)
              and "--execute" in command,
              "B02 pair memory: exact bounded pair command")
    try:
        output = json.loads(_run_stdout(g, "evidence/B02/pair-memory-r2"))
    except (OSError, ValueError) as exc:
        output = {}
        g.require(False, "B02 pair memory: parseable captured output", str(exc))
    g.require(output.get("status") == "GPU_PASS"
              and output.get("qualification_only") is True
              and output.get("bundle") == "B02"
              and output.get("ordered_uuids") == PAIR
              and output.get("requested_bytes") == output.get("mapped_bytes") == 100663296
              and output.get("owner_bytes") == [33554432, 67108864]
              and output.get("full_range_anchor_write_read") is True
              and output.get("ledger_zero_after_free") is True
              and output.get("arbitrary_kernel_transparency") is False,
              "B02 pair memory: bounded captured result fields")

    execution = g.json("evidence/B02/execution-r2/execution.json")
    command = execution.get("argv", [])
    g.require(execution.get("allocation_cap_mib_per_gpu") == 1024
              and command and Path(command[0]).name == "v200_execution_gpu_smoke"
              and command[1:] == ["--execute", "--expect-a", PAIR[0], "--expect-b", PAIR[1]],
              "B02 execution-r2: exact pair command within 1024 MiB cap")
    controller_summary = _controller_summary(g, "evidence/B02/execution-r2")
    binary = next((item for item in controller_summary.get("binaries", [])
                   if Path(item.get("path", "")).name == "v200_execution_gpu_smoke"), {})
    g.require(bool(binary) and binary.get("sha256") == "708f4c3e93d240c9a44b948b920de8b08c6e50120175893aedb816d6bc9a7e4c",
              "B02 execution-r2: recorded smoke binary identity")
    g.check_hash(binary.get("path", ""), binary.get("sha256", ""),
                 "B02 execution-r2: retained smoke binary hash")
    g.require(g.path("evidence/B02/execution-r2/stderr.txt").read_bytes() == b"",
              "B02 execution-r2: empty child stderr excludes late-cleanup failure")
    try:
        dataflow = json.loads(_run_stdout(g, "evidence/B02/execution-r2"))
    except (OSError, ValueError) as exc:
        dataflow = {}
        g.require(False, "B02 execution-r2: parseable captured output", str(exc))
    g.require(dataflow.get("status") == "GPU_PASS"
              and dataflow.get("qualification_only") is True
              and dataflow.get("bundle") == "B02"
              and dataflow.get("ordered_uuids") == PAIR
              and dataflow.get("owner_bytes") == 4096
              and all(dataflow.get(key) is True for key in (
                  "owner_a_copy_ok", "owner_b_copy_ok", "native_prefix",
                  "peer_fanin_wait", "peer_dependency_dataflow_ok", "context_restored"))
              and dataflow.get("arbitrary_kernel_transparency") is False,
              "B02 execution-r2: owner copies, native prefix, peer dataflow and context restoration")


def _consumer(g: Gate, kind: str) -> dict[str, Any]:
    return g.json(CONSUMERS[kind])


def verify_b03(g: Gate) -> None:
    native = _consumer(g, "native")
    facade = _consumer(g, "facade")
    for kind, item, control in (("native", native, "native_single"),
                                ("facade", facade, "facade_owner")):
        g.require(item.get("schema") == "v200-consumer-result/v1"
                  and item.get("application_status") == "PASS"
                  and item.get("failure") is None
                  and item.get("workload") == "smoke"
                  and item.get("control") == control
                  and item.get("gpu_qualification") == "NOT_CLAIMED",
                  f"B03 {kind}: application output passed without GPU qualification claim")
        oracle = item.get("expected_oracle", {})
        response = item.get("response", {})
        g.require(oracle.get("expected_exact") == "2"
                  and response.get("expected_exact") == "2"
                  and response.get("normalized_content") == "2"
                  and response.get("exact_match") is True
                  and oracle.get("tokens_predicted_min") <= response.get("tokens_predicted", -1)
                  <= oracle.get("tokens_predicted_max"),
                  f"B03 {kind}: exact consumer oracle")
        cleanup = item.get("cleanup", {})
        server = item.get("server_process", {})
        g.require(server.get("exit_code") == 0
                  and cleanup.get("no_live_group_members") is True
                  and cleanup.get("remaining_members") == []
                  and cleanup.get("zombie_members_only") is False
                  and cleanup.get("sigkill_escalation") is False,
                  f"B03 {kind}: owned server cleanup")
        g.require(item.get("preview_environment", {}).get("GGML_CUDA_DISABLE_GRAPHS") == "1",
                  f"B03 {kind}: graphs disabled in preview")

    native_ids = native.get("identities", {})
    facade_ids = facade.get("identities", {})
    compare_keys = ("binary", "model", "fixture_sha256", "profile_sha256", "runtime_closure", "source")
    g.require(all(native_ids.get(key) == facade_ids.get(key) for key in compare_keys),
              "B03: matched oracle model binary and source identities")
    def normalized_argv(item: dict[str, Any]) -> list[str]:
        argv = list(item.get("argv", []))
        try:
            port_index = argv.index("--port")
            del argv[port_index:port_index + 2]
        except ValueError:
            pass
        return argv

    g.require(native.get("expected_oracle") == facade.get("expected_oracle")
              and native.get("request") == facade.get("request")
              and native.get("fixture_id") == facade.get("fixture_id")
              and normalized_argv(native) == normalized_argv(facade)
              and native.get("preview_environment", {}).get("GGML_CUDA_DISABLE_GRAPHS")
              == facade.get("preview_environment", {}).get("GGML_CUDA_DISABLE_GRAPHS"),
              "B03: matched consumer request and model arguments")
    # Fixture/profile identity files are small inputs and cheap to verify.  The
    # model file is intentionally not rehashed here; it was checked separately.
    g.check_hash("benchmarks/fixtures/llama-server-v200.json",
                 native_ids.get("fixture_sha256", ""), "B03: fixture identity hash")
    g.check_hash("docs/v200/consumer-profile.json",
                 native_ids.get("profile_sha256", ""), "B03: consumer profile identity hash")

    g.require(native.get("selected_gpu_uuids") == [PAIR[0]]
              and native.get("v200_mode") is None
              and facade.get("selected_gpu_uuids") == PAIR
              and facade.get("v200_mode") == "pair"
              and facade.get("v200_b01_evidence_id") == B01_ID,
              "B03: native single-device and facade pair ownership")

    allocation = facade.get("allocation_evidence", {})
    expected_budgets = [{"budget_bytes": 12 * 1024**3, "gpu_uuid": uuid} for uuid in PAIR]
    telemetry = facade.get("v200_profile_telemetry", {})
    rows = telemetry.get("records", [])
    g.require(allocation.get("v200_memory_api_receipt") == "PASS"
              and allocation.get("fixture_expected_v200_owner_budgets") == expected_budgets
              and allocation.get("facade_limit_enforcer") == "V200 per-owner allocation ledger"
              and facade.get("preflight", {}).get("owner_count_assumed") == 2,
              "B03 facade: pair owner budgets and managed API receipt")
    profile_path = telemetry.get("path")
    raw_rows: list[dict[str, Any]] = []
    try:
        raw_rows = [json.loads(line) for line in g.path(profile_path).read_text(encoding="utf-8").splitlines()
                    if line.strip()]
        g.require(raw_rows == rows, "B03 facade: embedded and retained profile rows agree")
    except (OSError, TypeError, ValueError) as exc:
        g.require(False, "B03 facade: retained managed-allocation ledger readable", str(exc))
    positive = any(
        any(value > 0 for value in row.get("owner_used_bytes", []))
        or row.get("live_allocation_count", 0) > 0
        or row.get("live_reservation_count", 0) > 0
        for row in raw_rows
    )
    final = raw_rows[-1] if raw_rows else {}
    g.require(bool(raw_rows)
              and positive
              and raw_rows[0].get("event") == "profile_initialized"
              and final.get("owner_used_bytes") == [0, 0]
              and final.get("live_allocation_count") == 0
              and final.get("live_reservation_count") == 0
              and final.get("cleanup_pending_count") == 0
              and all(row.get("schema") == "v200-frontdoor-ledger/v1"
                      and row.get("mode") == "pair"
                      and row.get("anchor_uuid") == PAIR[0]
                      and row.get("peer_uuid") == PAIR[1]
                      and row.get("qualification_id") == B01_ID
                      and row.get("owner_budget_bytes") == [12 * 1024**3, 12 * 1024**3]
                      for row in raw_rows),
              "B03 facade: positive managed allocations and zero final ledger")

    facade_run = g.json("evidence/B03/facade/execution.json")
    env, _ = _parse_env(facade_run.get("argv", []))
    g.require(env.get("V200_DEVICE_UUIDS", "").split(",") == PAIR,
              "B03 facade: controller used the same ordered pair")
    summary_bins = _controller_summary(g, "evidence/B03/facade")
    summary_by_path = {item.get("path"): item.get("sha256") for item in summary_bins.get("binaries", [])}
    preload_ids = facade_ids.get("preloads", [])
    g.require([item.get("ordinal") for item in preload_ids] == [0, 1]
              and [Path(item.get("path", "")).name for item in preload_ids]
              == ["libv200_frontdoor.so", "libv200_pair_init.so"],
              "B03 facade: recorded frontdoor and pair-init preloads")
    for item in preload_ids:
        path, expected = item.get("path", ""), item.get("sha256", "")
        g.require(summary_by_path.get(path) == expected,
                  f"B03 facade: consumer and controller DSO hash agree ({Path(path).name})")
        g.check_hash(path, expected, f"B03 facade: retained DSO hash ({Path(path).name})")


def _flag_value(argv: list[str], flag: str) -> str | None:
    try:
        return argv[argv.index(flag) + 1]
    except (ValueError, IndexError):
        return None


def verify_service(g: Gate) -> dict[str, Any]:
    stem = "evidence/V200-SERVE/first-service"
    result = g.json(stem + ".controller-result.json")
    evidence = g.json(stem + ".controller-evidence.json")["evidence"]
    provenance_argv = evidence.get("summary", {}).get("provenance", {}).get("argv", [])
    recorded_run = _flag_value(provenance_argv, "--evidence-dir")
    g.require(bool(recorded_run), "V200-SERVE: run directory derived from retained controller argv")
    if not recorded_run:
        return {"evidence_id": result.get("evidence_id"), "status": "unverifiable"}
    runbase = Path(recorded_run)
    run = verify_run(g, "V200-SERVE first-service", str(runbase),
                     "547cfd3a-4eba-426a-9e1f-9b3448f7e1b8",
                     controller_stem=stem, expected_bundle="V200-SERVE")
    execution = g.json(runbase / "execution.json")
    env, command = _parse_env(execution.get("argv", []))
    output_dir = _flag_value(command, "--output-dir")
    g.require(execution.get("allocation_cap_mib_per_gpu") == 12288
              and "--serve-duration-seconds" in command
              and _flag_value(command, "--serve-duration-seconds") == "20"
              and _flag_value(command, "--server-port") == "8123"
              and env.get("V200_DEVICE_UUIDS", "").split(",") == PAIR
              and env.get("V200_MODE") in (None, "pair")
              and "--facade-mode" in command
              and _flag_value(command, "--facade-mode") == "pair",
              "V200-SERVE: exact bounded 20-second pair service invocation")
    g.require(bool(output_dir), "V200-SERVE: output directory comes from retained argv")
    if not output_dir:
        return run
    output_path = g.path(output_dir)
    consumer_candidates = sorted(output_path.glob("smoke-facade_owner-*.json"))
    g.require(len(consumer_candidates) == 1,
              "V200-SERVE: unique consumer record in recorded output directory")
    if len(consumer_candidates) != 1:
        return run
    consumer = g.json(consumer_candidates[0])
    client = g.json("evidence/V200-SERVE/first-service-client.json")
    g.require(consumer.get("application_status") == "PASS"
              and consumer.get("control") == "facade_owner"
              and consumer.get("v200_mode") == "pair"
              and consumer.get("selected_gpu_uuids") == PAIR
              and consumer.get("service", {}).get("status") == "PASS"
              and consumer.get("service", {}).get("duration_requested_seconds") == 20
              and consumer.get("service", {}).get("duration_observed_seconds", 0) >= 20
              and consumer.get("response", {}).get("normalized_content") == "2"
              and consumer.get("response", {}).get("exact_match") is True,
              "V200-SERVE: retained service-run completion")
    g.require(client.get("status") == "PASS"
              and client.get("health", {}).get("status") == "ok"
              and client.get("endpoint") == "http://127.0.0.1:8123"
              and client.get("request", {}).get("messages", [{}])[0].get("content")
              == "What is 1 + 1? Answer with exactly one digit."
              and client.get("response", {}).get("choices", [{}])[0].get("message", {}).get("content") == "2"
              and client.get("normalized_answer") == "2",
              "V200-SERVE: independent health and chat client response")
    ledger = consumer.get("v200_profile_telemetry", {})
    try:
        rows = [json.loads(line) for line in g.path(ledger.get("path", "")).read_text(encoding="utf-8").splitlines()
                if line.strip()]
        g.require(rows == ledger.get("records"), "V200-SERVE: embedded and retained ledger rows agree")
    except (OSError, TypeError, ValueError) as exc:
        rows = []
        g.require(False, "V200-SERVE: retained managed-allocation ledger readable", str(exc))
    final = rows[-1] if rows else {}
    g.require(any(any(x > 0 for x in row.get("owner_used_bytes", []))
                      or row.get("live_allocation_count", 0) > 0 for row in rows)
              and final.get("owner_used_bytes") == [0, 0]
              and final.get("live_allocation_count") == 0
              and final.get("live_reservation_count") == 0
              and final.get("cleanup_pending_count") == 0,
              "V200-SERVE: positive managed allocation and zero final ledger")
    summary_bins = _controller_summary(g, "evidence/V200-SERVE/first-service")
    summary_by_path = {item.get("path"): item.get("sha256") for item in summary_bins.get("binaries", [])}
    for item in consumer.get("identities", {}).get("preloads", []):
        path, expected = item.get("path", ""), item.get("sha256", "")
        g.require(summary_by_path.get(path) == expected,
                  f"V200-SERVE: consumer and controller DSO hash agree ({Path(path).name})")
        g.check_hash(path, expected, f"V200-SERVE: retained DSO hash ({Path(path).name})")
    return {"evidence_id": run.get("evidence_id"), "service_status": "PASS",
            "client_status": client.get("status"), "observed_seconds": consumer.get("service", {}).get("duration_observed_seconds")}


def verify(root: Path, include_service: bool = False) -> dict[str, Any]:
    g = Gate(root)
    run_summaries: dict[str, Any] = {}
    for label, (base, evidence_id) in RUNS.items():
        run_summaries[label] = verify_run(g, label, base, evidence_id)
    verify_b01(g)
    verify_b02(g)
    verify_b03(g)
    service_summary = verify_service(g) if include_service else None
    report = {
        "schema": "v200-first-consumer-acceptance/v1",
        "status": "accepted" if not g.failures else "rejected",
        "scope": "retained B01 pair reference + B02 frontdoor, owner-completion, and pair-memory fixtures + B03 matched native/facade correctness preview",
        "checks_passed": len(g.checks),
        "failures": g.failures,
        "runs": run_summaries,
        "claims": [
            "B01 pair probe passed its recorded qualification-only fixture.",
            "B02 controller-backed pair frontdoor closure and bounded pair-memory smoke passed their recorded fixtures.",
            "B02 execution-r2 recorded owner-copy, native-prefix, peer-dependency, and context-restoration checks passed.",
            "B03 native-single and V200-facade pair controls returned exact oracle output 2 with matched consumer identities and clean owned-server cleanup.",
        ],
        "limits": [
            "Artifact verification only; it does not rerun CUDA, the controller, or the consumer.",
            "B01/B02 GPU_PASS is limited to the named fixtures; arbitrary kernel transparency and consumer compatibility are not established by those operations.",
            "B03 is a correctness and ownership preview, not performance or GPU qualification; CUDA graphs are disabled.",
            "Memory limits are fixture/application ledger limits, not OS hard quotas; model bytes are not rehashed by this gate.",
            "Historical current-source or current-build equality is not inferred; retained immutable snapshot identity is checked.",
        ],
    }
    if include_service:
        report["service"] = service_summary
    return report


def _copy_recorded(root: Path, temp_root: Path, path: str | Path) -> None:
    try:
        rel = Path(path).resolve().relative_to(CANONICAL_ROOT)
    except (ValueError, OSError):
        return
    source = root / rel
    target = temp_root / rel
    if source.is_dir():
        if not target.exists():
            shutil.copytree(source, target, dirs_exist_ok=True)
    elif source.is_file():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def tamper_test(root: Path, include_service: bool = False) -> dict[str, Any]:
    """Copy the retained corpus, mutate a controller raw artifact, require rejection."""
    root = root.resolve()
    with tempfile.TemporaryDirectory(prefix="v200-first-consumer-") as temp:
        temp_root = Path(temp)
        for rel in (
            "evidence", "benchmarks/fixtures",
            "docs/v200/consumer-profile.json", "build/src/frontdoor/libv200_frontdoor.so",
            "build/src/integration/libv200_pair_init.so",
            "build-execution-hw-smoke/v200_execution_gpu_smoke",
        ):
            source = root / rel
            target = temp_root / rel
            if source.is_dir():
                shutil.copytree(source, target, dirs_exist_ok=True)
            elif source.is_file():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
        b01_profile = json.loads((root / "evidence/foundation/pair-profile.json").read_text())
        for source_path in b01_profile.get("source_sha256", {}):
            _copy_recorded(root, temp_root, root / source_path)
        # Snapshot roots and controller raw streams are absolute paths in the
        # immutable JSON records; copy only paths those records reference.
        for base, _ in RUNS.values():
            base_path = root / base
            evidence_path = base_path.with_name(base_path.name + ".controller-evidence.json")
            evidence = json.loads(evidence_path.read_text())["evidence"]
            for artifact in evidence.get("artifacts", []):
                _copy_recorded(root, temp_root, artifact.get("path", ""))
            source_root = evidence.get("snapshot", {}).get("source_root", "")
            _copy_recorded(root, temp_root, source_root)
            result_path = base_path.with_name(base_path.name + ".controller-result.json")
            result = json.loads(result_path.read_text())
            _copy_recorded(root, temp_root, result.get("lease_receipt", ""))
        if include_service:
            service_stem = root / "evidence/V200-SERVE/first-service"
            service_evidence = json.loads(
                service_stem.with_name("first-service.controller-evidence.json").read_text()
            )["evidence"]
            for artifact in service_evidence.get("artifacts", []):
                _copy_recorded(root, temp_root, artifact.get("path", ""))
            _copy_recorded(root, temp_root, service_evidence.get("snapshot", {}).get("source_root", ""))
            service_result = json.loads(
                service_stem.with_name("first-service.controller-result.json").read_text()
            )
            _copy_recorded(root, temp_root, service_result.get("lease_receipt", ""))
        # Facade ledger is pointed to by an absolute evidence path.
        facade_json = json.loads((root / CONSUMERS["facade"]).read_text())
        _copy_recorded(root, temp_root, facade_json["v200_profile_telemetry"]["path"])
        pristine = verify(temp_root, include_service=include_service)
        if pristine["status"] != "accepted":
            return {"status": "failed", "reason": "copied fixture did not validate", "failures": pristine["failures"]}
        raw = json.loads((root / "evidence/B02/frontdoor.controller-evidence.json").read_text())
        raw_path = raw["evidence"]["artifacts"][0]["path"]
        copied_raw = Gate(temp_root).path(raw_path)
        with copied_raw.open("ab") as stream:
            stream.write(b"tampered\n")
        mutated = verify(temp_root, include_service=include_service)
        detected = mutated["status"] == "rejected" and any("raw stdout hash" in failure for failure in mutated["failures"])
        return {"status": "passed" if detected else "failed", "tamper_detected": detected,
                "mutation": "copied B02 frontdoor raw stdout appended in temporary fixture",
                "failure_count": len(mutated["failures"])}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--tamper-test", action="store_true",
                        help="also copy evidence to a temporary directory and prove a raw-hash mutation is rejected")
    parser.add_argument("--service", action="store_true",
                        help="also verify the retained first-service and independent client artifacts")
    args = parser.parse_args()
    try:
        report = verify(args.repo_root, include_service=args.service)
        if args.tamper_test:
            report["tamper_test"] = tamper_test(args.repo_root, include_service=args.service)
            if report["tamper_test"]["status"] != "passed":
                report["status"] = "rejected"
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["status"] == "accepted" else 1
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"schema": "v200-first-consumer-acceptance/v1",
                          "status": "rejected", "error": f"{type(exc).__name__}: {exc}"},
                         indent=2, sort_keys=True))
        return 1


if __name__ == "__main__":
    sys.exit(main())
