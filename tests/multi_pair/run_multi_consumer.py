#!/usr/bin/env python3
"""Run a finite native-four-GPU vs two-logical-V200 llama correctness bundle.

This runner is invoked only inside the admitted CUDA controller. It runs a
small bidirectional copy check, then one unchanged resident llama-server per
control, and records process-scoped GPU memory, ledger, and cleanup evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
CONSUMER_DIR = ROOT / "tests/consumer"
if str(CONSUMER_DIR) not in sys.path:
    sys.path.insert(0, str(CONSUMER_DIR))
V200_SCRIPT_DIR = ROOT / "scripts/v200"
if str(V200_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(V200_SCRIPT_DIR))
import run_consumer as consumer  # noqa: E402
import bounded_runner  # noqa: E402

UUIDS = [
    "GPU-21131915-1488-23af-38dd-1743ae1f5cc8",
    "GPU-d745da9c-7649-8334-41f1-483afa6f3206",
    "GPU-cf22c41f-5b58-77b1-3535-8fadd1ca6505",
    "GPU-6c1cac7f-a360-0aef-ba98-2828bfd1db1a",
]
PAIR_RECEIPTS = [
    "8426eab5-278d-4687-ab7d-f5ec937e46ec",
    "1c97d504-468e-434c-81ea-112a4bb83e74",
]
LEDGER_SCHEMA = "v200-frontdoor-ledger/v1"
COPY_SCHEMA = "v200-multi-pair-copy/v1"
OPERATOR_SCHEMA = "v200-operator-ledger/v1"
MiB = 1024 * 1024
GiB = 1024 * MiB


class MultiPairError(RuntimeError):
    """The admitted multi-pair run cannot satisfy its pinned contract."""


def _validate_server_exit_code(exit_code: int | None,
                               cleanup: dict[str, Any]) -> dict[str, Any]:
    """Accept normal completion or the runner's deliberate SIGTERM shutdown."""
    if exit_code == 0:
        return {"status": "PASS", "reason": "normal_exit"}
    requested_sigterm = (
        exit_code == -signal.SIGTERM
        and cleanup.get("signaled_owned_group") is True
        and cleanup.get("sigkill_escalation") is False
    )
    if requested_sigterm:
        return {"status": "PASS", "reason": "requested_sigterm"}
    return {
        "status": "FAIL",
        "reason": "unexpected_child_exit",
        "exit_code": exit_code,
        "requested_shutdown": "SIGTERM",
        "cleanup_signaled_group": cleanup.get("signaled_owned_group") is True,
        "cleanup_escalated_to_sigkill": cleanup.get("sigkill_escalation") is True,
    }


def _device_count_evidence(log_text: str, expected_count: int) -> dict[str, Any]:
    """Validate the pinned llama.cpp device_info enumeration exactly."""
    ordinals = [int(value) for value in re.findall(
        r"common_param:\s+-\s*CUDA(\d+)\s*:", log_text, re.IGNORECASE)]
    expected_ordinals = list(range(expected_count))
    return {
        "status": "PASS" if ordinals == expected_ordinals else "MISMATCH",
        "observed_counts": [len(ordinals)] if ordinals else [],
        "observed_ordinals": ordinals,
        "expected_visible_devices": expected_count,
        "expected_ordinals": expected_ordinals,
    }


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_inputs(fixture_path: Path, profile_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        fixture_raw = fixture_path.read_bytes()
        fixture = json.loads(fixture_raw.decode("utf-8"))
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise MultiPairError(f"cannot load pinned multipair inputs: {exc}") from exc
    if not isinstance(fixture, dict) or not isinstance(profile, dict):
        raise MultiPairError("fixture and profile must be JSON objects")
    validate_inputs(fixture, profile, fixture_raw)
    return fixture, profile


def validate_inputs(fixture: dict[str, Any], profile: dict[str, Any],
                    fixture_raw: bytes | None = None) -> None:
    if fixture.get("schema") != "v200-llama-server-multi-pair-fixture/v1":
        raise MultiPairError("unsupported multipair fixture schema")
    if profile.get("schema") != "v200-multi-pair-profile/v1":
        raise MultiPairError("unsupported multipair profile schema")
    if fixture_raw is not None and profile.get("fixture_sha256") != _sha256_bytes(fixture_raw):
        raise MultiPairError("profile fixture SHA-256 does not match the exact fixture bytes")
    topology = fixture.get("topology", {})
    if topology.get("physical_uuid_order") != UUIDS:
        raise MultiPairError("fixture UUID order must remain physical indices 0,2,1,3")
    pairs = topology.get("logical_pairs")
    if (not isinstance(pairs, list) or len(pairs) != 2
            or [item.get("b01_evidence_id") for item in pairs] != PAIR_RECEIPTS
            or [(item.get("anchor_uuid"), item.get("peer_uuid")) for item in pairs]
            != [(UUIDS[0], UUIDS[1]), (UUIDS[2], UUIDS[3])]):
        raise MultiPairError("fixture must pin both qualified consecutive UUID pairs and receipts")
    if profile.get("pairs") != pairs:
        raise MultiPairError("profile pair order/receipt contract differs from fixture")
    controls = fixture.get("controls", {})
    if controls.get("native_four", {}).get("arguments") != [
            "--device", "CUDA0,CUDA1,CUDA2,CUDA3", "--split-mode", "layer",
            "--tensor-split", "1,1,1,1"]:
        raise MultiPairError("native control must use the matched four-device layer split")
    if controls.get("v200_two", {}).get("arguments") != [
            "--device", "CUDA0,CUDA1", "--split-mode", "layer", "--tensor-split", "1,1"]:
        raise MultiPairError("V200 control must use the ordinary two-device layer split")
    if controls.get("v200_two", {}).get("required_b01_evidence_ids") != PAIR_RECEIPTS:
        raise MultiPairError("V200 control is missing the pair-ordered B01 receipts")
    workload = fixture.get("workloads", {}).get("two_v200_resident", {})
    if int(workload.get("n_gpu_layers", 0)) < 999 or int(workload.get("ctx_size", 0)) != 8192:
        raise MultiPairError("multipair workload must retain the pinned full-resident context/layer request")
    if workload.get("controls") != ["native_four", "v200_two"]:
        raise MultiPairError("workload controls must be the native and V200 pair runs")
    model = fixture.get("model", {})
    if not isinstance(model.get("tensor_layout", {}).get("total_tensor_bytes"), int):
        raise MultiPairError("fixture is missing its measured GGUF tensor payload size")
    budgets = fixture.get("memory_guard", {}).get("owner_budgets_mib_per_v200_pair")
    if budgets != [4096, 12288]:
        raise MultiPairError("qualification must preserve the authorized 4 GiB/12 GiB pair owner caps")
    if int(fixture.get("memory_guard", {}).get("per_physical_gpu_application_limit_mib", 0)) != 12288:
        raise MultiPairError("the per-physical-owner application cap must remain 12 GiB")
    if fixture.get("server", {}).get("environment", {}).get("GGML_CUDA_DISABLE_GRAPHS") != "1":
        raise MultiPairError("this bounded first qualification is explicitly graph-off")
    server = fixture.get("server", {})
    if server.get("bind_host") != "127.0.0.1":
        raise MultiPairError("the finite service must bind to loopback only")
    fixed_arguments = server.get("fixed_arguments", [])
    if not any(fixed_arguments[index:index + 2] == ["--host", "127.0.0.1"]
               for index in range(max(0, len(fixed_arguments) - 1))):
        raise MultiPairError("llama-server arguments must bind to 127.0.0.1")


def static_preflight(fixture: dict[str, Any]) -> dict[str, Any]:
    """Check the pinned model/cache against the four-owner qualification caps."""
    workload = fixture["workloads"]["two_v200_resident"]
    model = fixture["model"]
    layout = model["tensor_layout"]
    layers = int(model["block_count"])
    ctx = int(workload["ctx_size"])
    kv = (ctx * layers * int(model["kv_heads"])
          * (int(model["key_length"]) + int(model["value_length"])) * 2)
    tensor = int(layout["total_tensor_bytes"])
    non_block = int(layout["non_block_tensor_bytes"])
    reserve = int(fixture["memory_guard"]["workspace_reserve_bytes_per_gpu"])
    cap = 12288 * MiB

    native_owner = (tensor + non_block + kv + 3) // 4 + reserve
    if native_owner > cap:
        raise MultiPairError(f"native four-way descriptor preflight exceeds 12 GiB: {native_owner}")

    pair_payload = (tensor + non_block + kv + 1) // 2 + 2 * reserve
    pair_budgets = [int(value) * MiB for value in
                    fixture["memory_guard"]["owner_budgets_mib_per_v200_pair"]]
    anchor = min(pair_payload, pair_budgets[0])
    peer = max(0, pair_payload - anchor)
    if anchor > pair_budgets[0] or peer > pair_budgets[1]:
        raise MultiPairError("V200 pair owner budget preflight cannot hold its half of the resident workload")
    return {
        "status": "PASS_STATIC_ESTIMATE",
        "resident_tensor_bytes": tensor,
        "kv_cache_bytes": kv,
        "native_four_owner_estimated_bytes_each": native_owner,
        "v200_pair_estimated_bytes_each": pair_payload,
        "v200_pair_owner_budget_bytes": pair_budgets,
        "v200_pair_estimated_owner_bytes": [anchor, peer],
        "basis": "GGUF tensor descriptors, f16 KV formula, equal logical tensor split, and a per-owner workspace reserve; live NVML headroom is checked before each process.",
    }


def validate_live_selection(environ: dict[str, str]) -> tuple[str, list[str]]:
    process_mask = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if environ.get("CUDA_VISIBLE_DEVICES", "") != process_mask:
        raise MultiPairError("runner environment differs from the controller's live CUDA mask")
    try:
        bootstrap = bounded_runner._load_bootstrap()
        lease = bootstrap.live_lease(ROOT, UUIDS)
    except Exception as exc:
        raise MultiPairError(f"controller lease validation failed: {exc}") from exc
    visible = lease.get("visible_uuids") if isinstance(lease, dict) else None
    if (not isinstance(visible, list) or len(visible) != len(UUIDS)
            or sorted(visible) != sorted(UUIDS)):
        raise MultiPairError("controller lease validator returned a different physical GPU set")
    nvidia_smi, _observed_inventory = consumer._verify_nvidia_smi_uuid_selection(UUIDS)
    return nvidia_smi, visible


def validate_headroom(fixture: dict[str, Any], memory: dict[str, dict[str, int]],
                      control: str) -> dict[str, Any]:
    reserve = int(fixture["memory_guard"]["workspace_reserve_bytes_per_gpu"])
    cap = 12288 * MiB
    available = {uuid: max(0, min(cap, int(memory[uuid]["free_bytes"]) - reserve))
                 for uuid in UUIDS}
    if control == "v200_two":
        budgets = fixture["memory_guard"]["owner_budgets_mib_per_v200_pair"]
        required = [int(budgets[index % 2]) * MiB for index in range(4)]
    else:
        required = [static_preflight(fixture)["native_four_owner_estimated_bytes_each"]] * 4
    failures = [{"gpu_uuid": uuid, "required_bytes": need, "available_bytes": available[uuid]}
                for uuid, need in zip(UUIDS, required) if need > available[uuid]]
    if failures:
        raise MultiPairError(f"insufficient live four-owner headroom before {control}: {failures}")
    return {"physical_reserve_bytes_per_owner": reserve,
            "available_application_bytes_by_uuid": available,
            "required_application_bytes_by_uuid": dict(zip(UUIDS, required)),
            "status": "PASS"}


def clean_environment(control: str, preloads: list[Path], profile_log: Path | None,
                      source_env: dict[str, str]) -> dict[str, str]:
    child = {key: value for key, value in source_env.items()
             if not key.startswith("V200_")
             and not key.startswith("LLAMA_ARG_")
             and not key.startswith("GGML_CUDA_")}
    # Controller leases may expose physical devices as numeric indices. The
    # workload requires a specific pair-consecutive order, so normalize only
    # the child mask after the exact admitted set has been validated.
    child["CUDA_VISIBLE_DEVICES"] = ",".join(UUIDS)
    inherited = [part for part in child.pop("LD_PRELOAD", "").split(":")
                 if "libv200" not in Path(part).name.lower()]
    child["GGML_CUDA_DISABLE_GRAPHS"] = "1"
    if control == "v200_two":
        child.update({
            "V200_MODE": "pair",
            "V200_DEVICE_UUIDS": ",".join(UUIDS),
            "V200_B01_EVIDENCE_IDS": ",".join(PAIR_RECEIPTS),
            "V200_OWNER0_BUDGET_MIB": "4096",
            "V200_OWNER1_BUDGET_MIB": "12288",
            "V200_PAIR_COMPUTE": "1",
        })
        if profile_log is not None:
            child["V200_PROFILE_LOG"] = str(profile_log.resolve())
        child["LD_PRELOAD"] = ":".join([*(str(path) for path in preloads), *inherited])
    elif inherited:
        child["LD_PRELOAD"] = ":".join(inherited)
    else:
        child.pop("LD_PRELOAD", None)
    return child


def read_multi_profile_log(path: Path, fixture: dict[str, Any],
                           require_resident: bool = True,
                           require_copy_directions: bool = True) -> dict[str, Any]:
    if not path.is_file():
        return {"status": "MISSING", "path": str(path), "records": [], "copy_records": []}
    raw = path.read_bytes()
    pairs = fixture["topology"]["logical_pairs"]
    expected = {(pair["anchor_uuid"], pair["peer_uuid"]): pair for pair in pairs}
    ledger_by_pair: dict[int, list[dict[str, Any]]] = {0: [], 1: []}
    copy_records: list[dict[str, Any]] = []
    operator_records: list[dict[str, Any]] = []
    errors: list[str] = []
    budgets_expected = [4096 * MiB, 12288 * MiB]
    try:
        for line_number, line in enumerate(raw.splitlines(), start=1):
            if not line.strip():
                continue
            value = json.loads(line.decode("utf-8"))
            if not isinstance(value, dict):
                raise ValueError(f"line {line_number} is not an object")
            schema = value.get("schema")
            if schema == LEDGER_SCHEMA:
                required = ("event", "mode", "anchor_uuid", "peer_uuid", "qualification_id",
                            "owner_budget_bytes", "owner_used_bytes", "live_allocation_count",
                            "live_reservation_count", "cleanup_pending_count")
                missing = [field for field in required if field not in value]
                if missing:
                    raise ValueError(f"line {line_number} is missing ledger fields {missing}")
                key = (value["anchor_uuid"], value["peer_uuid"])
                pair = expected.get(key)
                if pair is None:
                    raise ValueError(f"line {line_number} has unknown physical pair {key}")
                if value["mode"] != "pair" or value["qualification_id"] != pair["b01_evidence_id"]:
                    raise ValueError(f"line {line_number} has wrong mode or pair receipt")
                index = int(pair["pair_index"])
                budgets = value["owner_budget_bytes"]
                usage = value["owner_used_bytes"]
                if (not isinstance(budgets, list) or len(budgets) != 2
                        or any(isinstance(item, bool) or not isinstance(item, int)
                               or item <= 0 or item > bound
                               for item, bound in zip(budgets, budgets_expected))):
                    raise ValueError(f"line {line_number} has invalid pair owner budgets")
                if budgets != budgets_expected:
                    raise ValueError(f"line {line_number} pair budgets do not match the qualification envelope")
                if (not isinstance(usage, list) or len(usage) != 2
                        or any(isinstance(item, bool) or not isinstance(item, int)
                               or item < 0 for item in usage)):
                    raise ValueError(f"line {line_number} has invalid pair owner usage")
                if any(used > budget for used, budget in zip(usage, budgets)):
                    raise ValueError(f"line {line_number} pair usage exceeds its budget")
                for field in ("live_allocation_count", "live_reservation_count", "cleanup_pending_count"):
                    if isinstance(value[field], bool) or not isinstance(value[field], int) or value[field] < 0:
                        raise ValueError(f"line {line_number} has invalid {field}")
                ledger_by_pair[index].append(value)
            elif schema == COPY_SCHEMA:
                fields = ("event", "dst_pair", "src_pair", "bytes", "status", "route", "blocking", "elapsed_ms")
                missing = [field for field in fields if field not in value]
                if missing:
                    raise ValueError(f"line {line_number} is missing copy fields {missing}")
                if (value["event"] != "cross_pair_copy"
                        or value["dst_pair"] not in (0, 1) or value["src_pair"] not in (0, 1)
                        or value["dst_pair"] == value["src_pair"]
                        or isinstance(value["bytes"], bool) or not isinstance(value["bytes"], int)
                        or value["bytes"] <= 0
                        or isinstance(value["status"], bool) or not isinstance(value["status"], int)
                        or value["route"] != "host_stage" or value["blocking"] is not True
                        or isinstance(value["elapsed_ms"], bool)
                        or not isinstance(value["elapsed_ms"], (int, float))
                        or value["elapsed_ms"] < 0):
                    raise ValueError(f"line {line_number} has malformed cross-pair copy telemetry")
                if value["status"] != 0:
                    errors.append(f"line {line_number} cross-pair copy returned status {value['status']}")
                copy_records.append(value)
            elif schema == OPERATOR_SCHEMA:
                operator_status = value.get("status")
                if isinstance(operator_status, bool) or not isinstance(operator_status, int):
                    raise ValueError(f"line {line_number} operator status is malformed")
                if operator_status != 0:
                    errors.append(f"line {line_number} pair operator returned status {operator_status}")
                operator_records.append(value)
            else:
                raise ValueError(f"line {line_number} has unknown telemetry schema {schema!r}")

        pair_results: dict[str, Any] = {}
        total_peak = 0
        owner_peaks = {uuid: 0 for uuid in UUIDS}
        payload_share = (int(fixture["model"]["tensor_layout"]["all_block_tensor_bytes"]) + 1) // 2
        for index, pair in enumerate(pairs):
            records = ledger_by_pair[index]
            if not records:
                raise ValueError(f"profile log has no ledger records for pair {index}")
            peak_usage = [max(record["owner_used_bytes"][owner] for record in records)
                          for owner in range(2)]
            for owner, uuid in enumerate((pair["anchor_uuid"], pair["peer_uuid"])):
                owner_peaks[uuid] = peak_usage[owner]
            pair_peak = sum(peak_usage)
            total_peak += pair_peak
            last = records[-1]
            cleanup = (last["owner_used_bytes"] == [0, 0]
                       and last["live_allocation_count"] == 0
                       and last["live_reservation_count"] == 0
                       and last["cleanup_pending_count"] == 0)
            if not cleanup:
                errors.append(f"pair {index} ledger was not clean at process exit")
            if require_resident and (min(peak_usage) <= 0 or pair_peak < payload_share):
                errors.append(f"pair {index} did not hold its expected resident model share")
            pair_results[str(index)] = {
                "anchor_uuid": pair["anchor_uuid"], "peer_uuid": pair["peer_uuid"],
                "qualification_id": pair["b01_evidence_id"],
                "peak_owner_used_bytes": peak_usage,
                "peak_total_owner_used_bytes": pair_peak,
                "minimum_pair_block_payload_bytes": payload_share,
                "cleanup_accounting_zero": cleanup,
                "record_count": len(records),
            }
        directions = {(record["src_pair"], record["dst_pair"]) for record in copy_records
                      if record["status"] == 0}
        required_directions = {(0, 1), (1, 0)}
        if require_copy_directions and not required_directions.issubset(directions):
            errors.append("successful cross-pair copy telemetry is missing one or both directions")
        minimum_total = (int(fixture["model"]["tensor_layout"]["total_tensor_bytes"])
                         if require_resident else 0)
        status = "PASS" if not errors and total_peak >= minimum_total else "FAIL"
        if total_peak < minimum_total:
            errors.append("aggregate pair owner ledger peak is below the GGUF tensor payload")
            status = "FAIL"
        return {
            "status": status, "path": str(path), "sha256": hashlib.sha256(raw).hexdigest(),
            "pair_results": pair_results, "peak_owner_used_bytes_by_uuid": owner_peaks,
            "aggregate_pair_peak_bytes": total_peak, "required_tensor_payload_bytes": int(
                fixture["model"]["tensor_layout"]["total_tensor_bytes"]),
            "copy_records": copy_records, "operator_records": operator_records,
            "errors": errors,
        }
    except (UnicodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        return {"status": "INVALID", "path": str(path),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "pair_records": {str(key): value for key, value in ledger_by_pair.items()},
                "copy_records": copy_records, "operator_records": operator_records,
                "errors": [f"{type(exc).__name__}: {exc}"]}


def _run_copy_probe(probe: Path, output_dir: Path, fixture: dict[str, Any],
                    preloads: list[Path], environ: dict[str, str]) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    profile_path = output_dir / "v200-copy-probe.profile.jsonl"
    profile_path.unlink(missing_ok=True)
    child_env = clean_environment("v200_two", preloads, profile_path, environ)
    log_path = output_dir / "v200-copy-probe.log"
    command = [str(probe)]
    started = time.monotonic()
    try:
        completed = subprocess.run(command, cwd=str(ROOT), env=child_env,
                                   capture_output=True, text=True, timeout=45,
                                   start_new_session=True)
        log_path.write_text(completed.stdout + completed.stderr, encoding="utf-8")
        profile = read_multi_profile_log(profile_path, fixture,
                                         require_resident=False,
                                         require_copy_directions=True)
        marker_evidence = _copy_probe_marker_evidence(completed.stdout)
        passed = (completed.returncode == 0
                  and marker_evidence["status"] == "PASS"
                  and profile["status"] == "PASS")
        return {"status": "PASS" if passed else "FAIL", "argv": command,
                "returncode": completed.returncode, "stdout": completed.stdout,
                "stderr": completed.stderr, "elapsed_seconds": time.monotonic() - started,
                "marker_evidence": marker_evidence,
                "profile_telemetry": profile, "log_path": str(log_path)}
    except (OSError, subprocess.SubprocessError) as exc:
        return {"status": "FAIL", "argv": command,
                "error": f"{type(exc).__name__}: {exc}",
                "elapsed_seconds": time.monotonic() - started,
                "log_path": str(log_path)}


def _copy_probe_marker_evidence(stdout: str) -> dict[str, Any]:
    copy_markers = stdout.count("COPY_OK ")
    vmm_devices = [int(value) for value in re.findall(
        r"^VMM_OK device=(\d+) bytes=\d+$", stdout, re.MULTILINE)]
    passed = copy_markers == 6 and vmm_devices == [0, 1]
    return {"status": "PASS" if passed else "FAIL",
            "copy_marker_count": copy_markers,
            "vmm_completed_devices": vmm_devices,
            "required_vmm_devices": [0, 1]}


def _run_control(control: str, fixture: dict[str, Any], profile: dict[str, Any],
                 output_dir: Path, preloads: list[Path], nvidia_smi: str,
                 environ: dict[str, str], *, server_port: int | None = None,
                 serve_duration_seconds: int = 0) -> dict[str, Any]:
    workload = fixture["workloads"]["two_v200_resident"]
    profile_path = output_dir / f"{control}.profile.jsonl" if control == "v200_two" else None
    if profile_path is not None:
        profile_path.unlink(missing_ok=True)
    port = consumer._free_loopback_port() if server_port is None else server_port
    argv = consumer._argv_for(control, workload, fixture,
                              profile["binary"]["path"], port,
                              workload_name="resident_capacity")
    child_env = clean_environment(control, preloads, profile_path, environ)
    log_path = output_dir / f"{control}.log"
    result_path = output_dir / f"{control}.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    monotonic_start = time.monotonic()
    result: dict[str, Any] = {
        "schema": "v200-multi-pair-consumer-result/v1", "control": control,
        "backend": fixture["controls"][control]["backend"],
        "fixture_id": fixture["fixture_id"], "selected_gpu_uuids": UUIDS,
        "cuda_visible_devices": child_env.get("CUDA_VISIBLE_DEVICES"),
        "v200_logical_b01_evidence_ids": PAIR_RECEIPTS if control == "v200_two" else [],
        "argv": argv, "request": workload["request"], "expected_oracle": workload["oracle"],
        "started_unix_time": started, "readiness_seconds": None,
        "request_seconds": None, "response": None, "device_count_evidence": {"status": "PENDING"},
        "layer_offload_evidence": {"status": "PENDING"},
        "input_embedding_evidence": {"status": "PENDING"},
        "memory_guard": {"status": "PENDING", "samples": []},
        "profile_telemetry": {"status": "NOT_APPLICABLE"},
        "service": ({"status": "PENDING", "endpoint": f"http://127.0.0.1:{port}",
                     "duration_seconds": serve_duration_seconds}
                    if serve_duration_seconds else None),
        "application_status": "FAIL", "failure": None, "cleanup": None,
        "log_path": str(log_path), "result_path": str(result_path),
    }
    process: subprocess.Popen[Any] | None = None
    guard: consumer.ProcessMemoryGuard | None = None
    try:
        with log_path.open("wb") as log:
            old_mask = None
            if hasattr(signal, "pthread_sigmask"):
                old_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM})
            try:
                options: dict[str, Any] = {}
                if old_mask is not None:
                    options["preexec_fn"] = lambda: signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGTERM})
                process = subprocess.Popen(argv, cwd=profile["source"]["repository"],
                                           env=child_env, stdout=log,
                                           stderr=subprocess.STDOUT,
                                           start_new_session=True, **options)
            finally:
                if old_mask is not None:
                    signal.pthread_sigmask(signal.SIG_SETMASK, old_mask)
        result["server_process"] = {"pid": process.pid, "process_group": process.pid,
                                    "session": process.pid}
        guard = consumer.ProcessMemoryGuard(
            process, UUIDS,
            int(fixture["memory_guard"]["physical_process_used_memory_guard_mib"]),
            float(fixture["memory_guard"]["nvidia_smi_poll_seconds"]), nvidia_smi)
        guard.start()
        result["memory_guard"]["status"] = "running_fail_closed"
        result["readiness_seconds"] = consumer.wait_for_readiness(
            process, f"http://127.0.0.1:{port}",
            float(fixture["server"]["timeouts_seconds"]["startup"]))
        if guard.error or guard.over_limit:
            raise MultiPairError(f"NVML process guard stopped {control}: {guard.error or guard.over_limit}")
        request_start = time.monotonic()
        status, response = consumer._completion_request(
            f"http://127.0.0.1:{port}{workload['request']['path']}",
            workload["request"]["body"],
            float(fixture["server"]["timeouts_seconds"]["request"]))
        result["request_seconds"] = time.monotonic() - request_start
        if status != 200:
            raise MultiPairError(f"{control} completion returned HTTP {status}: {response!r}")
        result["response"] = consumer.validate_completion_response(response, workload["oracle"])
        if guard.error or guard.over_limit:
            raise MultiPairError(f"NVML process guard stopped {control}: {guard.error or guard.over_limit}")
        result["application_status"] = "PASS"
        if serve_duration_seconds:
            service_started = time.monotonic()
            endpoint = f"http://127.0.0.1:{port}"
            print(json.dumps({"event": "V200_SERVER_READY", "endpoint": endpoint,
                              "duration_seconds": serve_duration_seconds}, sort_keys=True),
                  flush=True)
            while True:
                elapsed = time.monotonic() - service_started
                if elapsed >= serve_duration_seconds:
                    break
                if process.poll() is not None:
                    raise MultiPairError(
                        f"llama-server exited during finite service with code {process.returncode}")
                if guard.error or guard.over_limit:
                    raise MultiPairError(
                        f"NVML process guard stopped {control}: {guard.error or guard.over_limit}")
                time.sleep(min(0.25, serve_duration_seconds - elapsed))
            result["service"].update({"status": "PASS",
                                      "observed_seconds": time.monotonic() - service_started})
    except BaseException as exc:
        result["failure"] = f"{type(exc).__name__}: {exc}"
        if serve_duration_seconds:
            result["application_status"] = "FAIL"
        if result.get("service") is not None and result["service"].get("status") == "PENDING":
            result["service"].update({"status": "INTERRUPTED" if isinstance(
                exc, (KeyboardInterrupt, SystemExit)) else "FAIL",
                "observed_seconds": (time.monotonic() - service_started
                                     if "service_started" in locals() else 0.0)})
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
    finally:
        prior_term = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        try:
            if guard is not None:
                guard.finish()
                result["memory_guard"].update({
                    "samples": guard.samples,
                    "status": "FAIL" if guard.error or guard.over_limit else "PASS",
                    "error": guard.error, "over_limit": guard.over_limit,
                })
                peaks = {uuid: 0 for uuid in UUIDS}
                for sample in guard.samples:
                    for uuid, used in sample["owned_used_memory_bytes_by_uuid"].items():
                        peaks[uuid] = max(peaks[uuid], int(used))
                result["memory_guard"]["peak_owned_used_memory_bytes_by_uuid"] = peaks
                if guard.error or guard.over_limit or any(value <= 0 for value in peaks.values()):
                    result["application_status"] = "FAIL"
                    result["failure"] = result["failure"] or "one or more physical owners had no sampled process memory"
            if process is not None:
                cleanup = consumer.cleanup_owned_group(
                    process, float(fixture["server"]["timeouts_seconds"]["shutdown"]))
                result["server_process"]["exit_code"] = process.returncode
                result["cleanup"] = cleanup
                exit_validation = _validate_server_exit_code(process.returncode, cleanup)
                result["server_process"]["exit_validation"] = exit_validation
                if exit_validation["status"] != "PASS":
                    result["application_status"] = "FAIL"
                    result["failure"] = result["failure"] or (
                        f"llama-server exited unexpectedly after cleanup: {exit_validation}")
                if not cleanup["no_live_group_members"]:
                    result["application_status"] = "FAIL"
                    result["failure"] = result["failure"] or "owned llama-server process group did not clean up"
            result["layer_offload_evidence"] = consumer._layer_offload_evidence(
                log_path, "resident_capacity", fixture)
            result["input_embedding_evidence"] = consumer._input_embedding_evidence(
                log_path, "resident_capacity")
            try:
                log_text = log_path.read_text(encoding="utf-8", errors="replace")
                expected_count = int(fixture["controls"][control]["visible_device_count"])
                result["device_count_evidence"] = _device_count_evidence(
                    log_text, expected_count)
            except OSError as exc:
                result["device_count_evidence"] = {"status": "MISSING", "error": str(exc)}
            for key in ("layer_offload_evidence", "input_embedding_evidence", "device_count_evidence"):
                if result[key]["status"] != "PASS":
                    result["application_status"] = "FAIL"
                    result["failure"] = result["failure"] or f"{key} failed: {result[key]}"
            if control == "v200_two":
                telemetry = read_multi_profile_log(
                    profile_path, fixture, require_resident=True,
                    require_copy_directions=False) if profile_path else {
                    "status": "MISSING", "errors": ["V200 profile log was not assigned"]}
                result["profile_telemetry"] = telemetry
                if telemetry["status"] != "PASS":
                    result["application_status"] = "FAIL"
                    result["failure"] = result["failure"] or f"V200 owner ledger failed: {telemetry.get('errors')}"
                if guard is not None:
                    peaks = result["memory_guard"].get("peak_owned_used_memory_bytes_by_uuid", {})
                    if any(int(peaks.get(uuid, 0)) <= 0 for uuid in UUIDS):
                        result["application_status"] = "FAIL"
                        result["failure"] = result["failure"] or "all four physical owners must show application residency"
            if result.get("service") is not None and result["application_status"] != "PASS":
                if result["service"].get("status") == "PASS":
                    result["service"]["status"] = "FAIL"
            result["elapsed_seconds"] = time.monotonic() - monotonic_start
            result["finished_unix_time"] = time.time()
            consumer._write_json_atomic(result_path, result)
        finally:
            signal.signal(signal.SIGTERM, prior_term)
    return result


def run_bundle(fixture_path: Path, profile_path: Path, output_dir: Path,
               preloads: list[Path], probe: Path,
               environ: dict[str, str] | None = None) -> dict[str, Any]:
    fixture, profile = load_inputs(fixture_path, profile_path)
    env = dict(os.environ if environ is None else environ)
    nvidia_smi, visible = validate_live_selection(env)
    output_dir.mkdir(parents=True, exist_ok=True)
    preflight = static_preflight(fixture)
    try:
        identities = consumer.verify_identities(profile, fixture, fixture_path,
                                                profile_path, preloads)
    except consumer.ConsumerError as exc:
        raise MultiPairError(f"pinned llama/model identity check failed: {exc}") from exc
    runtime_hashes = [{"path": str(path.resolve()), "sha256": consumer.sha256_file(path)}
                      for path in [*preloads, probe]]
    report: dict[str, Any] = {
        "schema": "v200-multi-pair-bundle-result/v1",
        "fixture_id": fixture["fixture_id"],
        "selected_gpu_uuids": UUIDS,
        "nvidia_smi_visible_gpu_uuids": visible,
        "identities": identities,
        "v200_runtime_artifacts": runtime_hashes,
        "static_preflight": preflight,
        "controls": [],
        "copy_probe": None,
        "status": "FAIL",
        "qualification": "CANDIDATE_NOT_QUALIFIED",
        "performance_claim": "NONE",
    }
    # Exercise the narrow cross-pair API before loading the 18.5 GB model. A
    # failed copy route, process cleanup, or owner ledger is a hard stop: later
    # workloads would add GPU activity without making this bundle more useful.
    report["copy_probe"] = _run_copy_probe(probe, output_dir, fixture, preloads, env)
    stop_reason = None
    if report["copy_probe"]["status"] != "PASS":
        stop_reason = "cross-pair copy probe failed; model controls were not started"
    else:
        for control in ("native_four", "v200_two"):
            try:
                before = consumer._nvidia_smi_memory(nvidia_smi, UUIDS)
                headroom = validate_headroom(fixture, before, control)
            except (MultiPairError, consumer.ConsumerError, OSError) as exc:
                result = {
                    "schema": "v200-multi-pair-consumer-result/v1",
                    "control": control,
                    "backend": fixture["controls"][control]["backend"],
                    "fixture_id": fixture["fixture_id"],
                    "selected_gpu_uuids": UUIDS,
                    "application_status": "FAIL",
                    "failure": f"{type(exc).__name__}: {exc}",
                    "execution_status": "NOT_STARTED_HEADROOM_PREFLIGHT",
                    "cleanup": {"status": "NOT_REQUIRED"},
                }
            else:
                try:
                    result = _run_control(control, fixture, profile, output_dir,
                                          preloads, nvidia_smi, env)
                    result["physical_memory_before_launch"] = before
                    result["live_headroom_preflight"] = headroom
                except Exception as exc:
                    # Cleanup exceptions mean process/resource state is
                    # uncertain. Preserve that failure and never start the
                    # next model control.
                    result = {
                        "schema": "v200-multi-pair-consumer-result/v1",
                        "control": control,
                        "backend": fixture["controls"][control]["backend"],
                        "fixture_id": fixture["fixture_id"],
                        "selected_gpu_uuids": UUIDS,
                        "application_status": "FAIL",
                        "failure": f"runner raised {type(exc).__name__}: {exc}",
                        "execution_status": "RUNNER_ERROR_CLEANUP_UNKNOWN",
                        "cleanup": {"status": "UNKNOWN"},
                    }
            report["controls"].append(result)
            if result.get("application_status") != "PASS":
                stop_reason = (f"{control} failed or did not clean up; remaining model controls "
                               "were not started")
                break

    completed_controls = {item["control"] for item in report["controls"]}
    for control in ("native_four", "v200_two"):
        if control not in completed_controls:
            report["controls"].append({
                "schema": "v200-multi-pair-consumer-result/v1",
                "control": control,
                "backend": fixture["controls"][control]["backend"],
                "fixture_id": fixture["fixture_id"],
                "selected_gpu_uuids": UUIDS,
                "application_status": "NOT_RUN",
                "failure": stop_reason or "prior required case did not pass",
                "execution_status": "NOT_STARTED_AFTER_PRIOR_FAILURE",
                "cleanup": {"status": "NOT_REQUIRED"},
            })

    reports = report["controls"]
    controls_by_name = {item["control"]: item for item in reports}
    v200 = controls_by_name["v200_two"]
    probe_ok = report["copy_probe"]["status"] == "PASS"
    controls_ok = all(controls_by_name[name]["application_status"] == "PASS"
                      for name in ("native_four", "v200_two"))
    report["status"] = "PASS" if controls_ok and probe_ok else "FAIL"
    if report["status"] == "PASS":
        report["qualification"] = "PASS_TWO_LOGICAL_V200_RESIDENT_CORRECTNESS"
    report["v200_summary"] = {
        "visible_cuda_devices": v200.get("device_count_evidence", {}).get("observed_counts"),
        "all_model_layers_offloaded": v200.get("layer_offload_evidence", {}).get("full_gpu_offload"),
        "input_embedding_on_cuda0": v200.get("input_embedding_evidence", {}).get("status") == "PASS",
        "four_owner_peak_bytes": v200.get("profile_telemetry", {}).get("peak_owner_used_bytes_by_uuid"),
        "aggregate_pair_peak_bytes": v200.get("profile_telemetry", {}).get("aggregate_pair_peak_bytes"),
        "cross_pair_copy_directions": sorted({
            f"{record['src_pair']}->{record['dst_pair']}"
            for record in report["copy_probe"].get("profile_telemetry", {}).get("copy_records", [])
            if record.get("status") == 0}),
        "cleanup_accounting_zero": all(
            pair.get("cleanup_accounting_zero", False)
            for pair in v200.get("profile_telemetry", {}).get("pair_results", {}).values()),
    }
    consumer._write_json_atomic(output_dir / "bundle.json", report)
    return report


def run_service(fixture_path: Path, profile_path: Path, output_dir: Path,
                preloads: list[Path], port: int, duration_seconds: int,
                environ: dict[str, str] | None = None) -> dict[str, Any]:
    """Run one finite, foreground two-logical-device V200 HTTP service."""
    if isinstance(port, bool) or not 1 <= port <= 65535:
        raise MultiPairError("service port must be between 1 and 65535")
    if isinstance(duration_seconds, bool) or not 1 <= duration_seconds <= 3000:
        raise MultiPairError("service duration must be between 1 and 3000 seconds")
    fixture, profile = load_inputs(fixture_path, profile_path)
    env = dict(os.environ if environ is None else environ)
    nvidia_smi, visible = validate_live_selection(env)
    output_dir.mkdir(parents=True, exist_ok=True)
    preflight = static_preflight(fixture)
    try:
        identities = consumer.verify_identities(profile, fixture, fixture_path,
                                                profile_path, preloads)
    except consumer.ConsumerError as exc:
        raise MultiPairError(f"pinned llama/model identity check failed: {exc}") from exc
    runtime_hashes = [{"path": str(path.resolve()), "sha256": consumer.sha256_file(path)}
                      for path in preloads]
    service: dict[str, Any] = {
        "schema": "v200-multi-pair-service-result/v1",
        "fixture_id": fixture["fixture_id"],
        "selected_gpu_uuids": UUIDS,
        "nvidia_smi_visible_gpu_uuids": visible,
        "identities": identities,
        "v200_runtime_artifacts": runtime_hashes,
        "static_preflight": preflight,
        "endpoint": f"http://127.0.0.1:{port}",
        "duration_seconds": duration_seconds,
        "control": None,
        "status": "FAIL",
        "performance_claim": "NONE",
    }
    try:
        before = consumer._nvidia_smi_memory(nvidia_smi, UUIDS)
        headroom = validate_headroom(fixture, before, "v200_two")
    except (MultiPairError, consumer.ConsumerError, OSError) as exc:
        control_result = {
            "schema": "v200-multi-pair-consumer-result/v1",
            "control": "v200_two",
            "backend": fixture["controls"]["v200_two"]["backend"],
            "fixture_id": fixture["fixture_id"],
            "selected_gpu_uuids": UUIDS,
            "application_status": "FAIL",
            "failure": f"{type(exc).__name__}: {exc}",
            "execution_status": "NOT_STARTED_HEADROOM_PREFLIGHT",
            "cleanup": {"status": "NOT_REQUIRED"},
        }
    else:
        try:
            control_result = _run_control(
                "v200_two", fixture, profile, output_dir, preloads, nvidia_smi, env,
                server_port=port, serve_duration_seconds=duration_seconds)
            control_result["physical_memory_before_launch"] = before
            control_result["live_headroom_preflight"] = headroom
        except Exception as exc:
            control_result = {
                "schema": "v200-multi-pair-consumer-result/v1",
                "control": "v200_two",
                "backend": fixture["controls"]["v200_two"]["backend"],
                "fixture_id": fixture["fixture_id"],
                "selected_gpu_uuids": UUIDS,
                "application_status": "FAIL",
                "failure": f"runner raised {type(exc).__name__}: {exc}",
                "execution_status": "RUNNER_ERROR_CLEANUP_UNKNOWN",
                "cleanup": {"status": "UNKNOWN"},
            }
    service["control"] = control_result
    service["status"] = ("PASS" if control_result.get("application_status") == "PASS"
                         and control_result.get("service", {}).get("status") == "PASS"
                         else "FAIL")
    consumer._write_json_atomic(output_dir / "service.json", service)
    return service


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preload-library", type=Path, action="append", required=True)
    parser.add_argument("--copy-probe", type=Path)
    parser.add_argument("--service", action="store_true",
                        help="serve the v200_two resident workload for a finite foreground duration")
    parser.add_argument("--duration-seconds", type=int,
                        help="finite service lifetime after the exact startup self-check (1..3000)")
    parser.add_argument("--port", type=int,
                        help="loopback HTTP port for --service (1..65535)")
    parser.add_argument("--execute", action="store_true",
                        help="required to launch the finite consumers inside controller admission")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if not args.execute:
        print("Refusing live multi-pair work without --execute inside CUDA-controller admission.",
              file=sys.stderr)
        return 2
    if args.service:
        if args.copy_probe is not None:
            print("--copy-probe is not used by --service.", file=sys.stderr)
            return 2
        if args.duration_seconds is None or args.port is None:
            print("--service requires --duration-seconds and --port.", file=sys.stderr)
            return 2
        prior_sigterm = signal.getsignal(signal.SIGTERM)

        def _handle_service_sigterm(_signum: int, _frame: Any) -> None:
            raise MultiPairError("foreground V200 service received SIGTERM; cleaning up its owned server")

        signal.signal(signal.SIGTERM, _handle_service_sigterm)
        try:
            report = run_service(args.fixture, args.profile, args.output_dir,
                                 [path.resolve() for path in args.preload_library],
                                 args.port, args.duration_seconds)
        except (MultiPairError, consumer.ConsumerError, OSError, KeyError, TypeError, ValueError) as exc:
            print(f"multi-pair service refused: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 2
        finally:
            signal.signal(signal.SIGTERM, prior_sigterm)
        print(json.dumps({"status": report["status"], "endpoint": report["endpoint"],
                          "duration_seconds": report["duration_seconds"],
                          "result": str(args.output_dir / "service.json")}, sort_keys=True))
        return 0 if report["status"] == "PASS" else 1
    if args.duration_seconds is not None or args.port is not None:
        print("--duration-seconds and --port require --service.", file=sys.stderr)
        return 2
    if args.copy_probe is None:
        print("the correctness bundle requires --copy-probe.", file=sys.stderr)
        return 2
    try:
        report = run_bundle(args.fixture, args.profile, args.output_dir,
                            [path.resolve() for path in args.preload_library],
                            args.copy_probe.resolve())
    except (MultiPairError, consumer.ConsumerError, OSError, KeyError, TypeError, ValueError) as exc:
        print(f"multi-pair run refused: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"status": report["status"], "qualification": report["qualification"],
                      "bundle": str(args.output_dir / "bundle.json"),
                      "v200_summary": report.get("v200_summary")}, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
