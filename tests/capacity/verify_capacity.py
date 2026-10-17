#!/usr/bin/env python3
"""Verify one retained V200-07 B04 run without initializing CUDA or hashing the model."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/consumer"))
import run_consumer as consumer  # noqa: E402


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _verify_case(path: Path, control: str, fixture: dict[str, Any],
                 fixture_sha: str, pair: list[str]) -> dict[str, Any]:
    case = _json(path)
    _require(case.get("workload") == "resident_capacity", f"wrong workload in {path}")
    _require(case.get("control") == control, f"wrong control in {path}")
    _require(case.get("application_status") == "PASS", f"consumer case failed: {path}")
    _require(case.get("selected_gpu_uuids") == pair, f"selected GPU order differs in {path}")
    _require(case.get("identities", {}).get("fixture_sha256") == fixture_sha,
             f"fixture identity changed in {path}")
    _require(case.get("identities", {}).get("model", {}).get("sha256") ==
             fixture["model"]["sha256"], f"model identity differs in {path}")
    tensor_bytes = int(fixture["model"]["tensor_layout"]["total_tensor_bytes"])
    physical = case.get("physical_gpu_memory_before_launch", {})
    capacities = [int(physical[uuid]["total_bytes"]) for uuid in pair]
    _require(tensor_bytes > max(capacities),
             f"model tensor payload does not exceed one physical GPU in {path}")
    _require(case.get("response", {}).get("exact_match") is True,
             f"exact completion oracle failed in {path}")
    consumer.validate_completion_response(
        case["response"], fixture["workloads"]["resident_capacity"]["oracle"])
    _require(case.get("cleanup", {}).get("no_live_group_members") is True,
             f"server process group was not cleaned up in {path}")
    _require(case.get("server_process", {}).get("exit_code") == 0,
             f"server exited unsuccessfully in {path}")
    _require(case.get("memory_guard", {}).get("telemetry_status") == "PASS" and
             case["memory_guard"].get("error") is None and
             case["memory_guard"].get("over_limit") is None,
             f"physical memory guard failed in {path}")
    peaks = case["memory_guard"].get("peak_owned_used_memory_bytes_by_uuid", {})
    _require(all(int(peaks.get(uuid, 0)) > 0 for uuid in pair),
             f"NVML did not observe the process on both devices in {path}")
    _require(case.get("layer_offload_evidence", {}).get("status") == "PASS" and
             case["layer_offload_evidence"].get("offloaded_layers") ==
             case["layer_offload_evidence"].get("required_layers"),
             f"all layers were not offloaded in {path}")
    _require(case.get("input_embedding_evidence", {}).get("status") == "PASS" and
             case["input_embedding_evidence"].get("buffer_type") == "CUDA0",
             f"input embedding was not placed on CUDA0 in {path}")

    log_path = Path(case["log_path"])
    log = log_path.read_text(encoding="utf-8", errors="replace")
    model_buffer_matches = re.findall(
        r"\b(CUDA\d+)\s+model buffer size\s*=\s*([0-9]+(?:\.[0-9]+)?)\s+MiB",
        log)
    cpu_buffer_matches = re.findall(
        r"\b(CPU(?:_Mapped)?)\s+model buffer size\s*=\s*([0-9]+(?:\.[0-9]+)?)\s+MiB",
        log, flags=re.IGNORECASE)
    gpu_buffer_bytes = sum(float(size) * 1024 * 1024
                           for _, size in model_buffer_matches)
    rounding_allowance = len(model_buffer_matches) * 0.005 * 1024 * 1024
    _require(model_buffer_matches and
             gpu_buffer_bytes + rounding_allowance >= tensor_bytes,
             f"CUDA model-buffer logs do not account for all {tensor_bytes} tensor bytes")
    _require(not any(float(size) > 0 for _, size in cpu_buffer_matches),
             f"model tensors remain in CPU model buffers in {path}")

    if control == "facade_owner":
        telemetry = case.get("v200_profile_telemetry", {})
        ledger_path = Path(telemetry.get("path", ""))
        raw = ledger_path.read_bytes()
        raw_sha = hashlib.sha256(raw).hexdigest()
        _require(raw_sha == telemetry.get("sha256"), "owner-ledger raw hash differs from case JSON")
        profile = consumer._read_profile_log(
            ledger_path, pair,
            int(fixture["memory_guard"]["per_physical_gpu_application_limit_mib"]) * 1024 * 1024,
            str(fixture["controls"]["facade_owner"]["required_b01_evidence_id"]),
            minimum_resident_bytes=tensor_bytes)
        _require(profile["status"] == "PASS", f"owner-ledger verification failed: {profile['status']}")
        _require(profile["cleanup_accounting_zero"] is True,
                 "owner-ledger cleanup did not return to zero")
        _require(min(profile["peak_owner_used_bytes"]) > 0,
                 "owner ledger did not record backing on both physical GPUs")
        _require(case.get("physical_headroom_preflight", {}).get("status") == "PASS",
                 "live per-owner headroom preflight did not pass")
        return {"control": control, "tensor_bytes": tensor_bytes,
                "cuda_model_buffer_bytes_logged": int(gpu_buffer_bytes),
                "peak_owner_used_bytes": profile["peak_owner_used_bytes"],
                "peak_pair_backing_bytes": profile["peak_total_owner_used_bytes"],
                "owner_ledger_sha256": raw_sha,
                "cleanup_accounting_zero": profile["cleanup_accounting_zero"]}
    return {"control": control, "tensor_bytes": tensor_bytes,
            "cuda_model_buffer_bytes_logged": int(gpu_buffer_bytes)}


def verify(run_dir: Path, controller_result_path: Path | None = None,
           controller_evidence_path: Path | None = None,
           host_release_path: Path | None = None) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    controller_result_path = controller_result_path or run_dir.with_name(
        run_dir.name + ".controller-result.json")
    controller_evidence_path = controller_evidence_path or run_dir.with_name(
        run_dir.name + ".controller-evidence.json")
    host_release_path = host_release_path or run_dir.with_name(
        run_dir.name + ".host-release.json")
    fixture, _ = consumer.load_documents()
    fixture_sha = consumer.sha256_file(consumer.FIXTURE_PATH)
    evidence = _json(controller_evidence_path).get("evidence", {})
    controller = _json(controller_result_path)
    execution = _json(run_dir / "execution.json")
    release = _json(host_release_path)
    _require(evidence.get("id") == controller.get("evidence_id"),
             "controller evidence and result IDs differ")
    _require(evidence.get("status") == "succeeded" and evidence.get("valid") in (1, True)
             and evidence.get("contaminated") in (0, False),
             "controller evidence is invalid or contaminated")
    summary = evidence.get("summary", {})
    _require(summary.get("valid") is True and
             summary.get("resource_samples", {}).get("quiescence", {}).get("uncontaminated") is True,
             "controller evidence lacks valid uncontaminated admission")
    _require(controller.get("ok") is True and controller.get("status") == "succeeded"
             and controller.get("returncode") == 0,
             "controller did not report a successful bounded run")
    _require(execution.get("bundle") == "B04" and execution.get("status") == "CHILD_COMPLETED"
             and execution.get("returncode") == 0,
             "B04 bounded execution did not complete successfully")
    for field in ("timed_out", "interrupted", "orphaned_children", "termination_uncertain"):
        _require(execution.get(field) is False, f"unclean controller execution: {field}")
    _require(release.get("state") == "released" and release.get("resources") == []
             and release.get("project_root") == str(ROOT),
             "host release does not show that all resources were returned")
    lease_path = Path(controller["lease_receipt"])
    lease = _json(lease_path)
    _require(lease.get("format") == "CUDA-FOREGROUND-LEASE/1"
             and lease.get("project_root") == str(ROOT), "invalid CUDA controller lease receipt")
    _require(lease.get("owner_id") == release.get("id") and
             lease.get("pid") == release.get("pid") == execution.get("lease", {}).get("owner_pid"),
             "lease and release identities differ")
    visible = execution.get("lease", {}).get("visible_uuids", [])
    lease_uuids = [item.removeprefix("accelerator:")
                   for item in lease.get("resource_ids", [])]
    _require(len(visible) == 2 and sorted(visible) == sorted(lease_uuids),
             "controller lease does not cover exactly the selected pair")
    expected_pair = visible
    _require(execution.get("lease", {}).get("receipt") == controller.get("lease_receipt"),
             "controller execution and result use different lease receipts")
    stdout = (run_dir / "stdout.txt").read_bytes()
    stderr = (run_dir / "stderr.txt").read_bytes()
    _require(hashlib.sha256(stdout).hexdigest() == execution.get("stdout_sha256") and
             hashlib.sha256(stderr).hexdigest() == execution.get("stderr_sha256"),
             "controller stdout/stderr hashes differ from the execution record")
    for artifact in evidence.get("artifacts", []):
        if artifact.get("kind") in ("stdout", "stderr"):
            raw = Path(artifact["path"]).read_bytes()
            _require(hashlib.sha256(raw).hexdigest() == artifact.get("content_hash"),
                     "controller output artifact hash differs")

    results = list((run_dir / "consumer").glob("resident_capacity-*.json"))
    cases = {str(_json(path).get("control")): path for path in results}
    _require(set(cases) == {"native_pair", "facade_owner"},
             "expected exactly one native-pair and one facade-owner consumer case")
    verified = [_verify_case(cases[name], name, fixture, fixture_sha, expected_pair)
                for name in ("native_pair", "facade_owner")]
    return {"status": "PASS_V200_07", "controller_evidence_id": evidence["id"],
            "physical_uuids": expected_pair, "cases": verified,
            "host_resources_released": True,
            "performance": "NOT_MEASURED", "dual_gpu_compute": "NOT_DEMONSTRATED"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path,
                        default=ROOT / "evidence/B04/capacity-r1")
    parser.add_argument("--controller-result", type=Path)
    parser.add_argument("--controller-evidence", type=Path)
    parser.add_argument("--host-release", type=Path)
    args = parser.parse_args(argv)
    try:
        report = verify(args.run_dir, args.controller_result,
                        args.controller_evidence, args.host_release)
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "FAIL", "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
