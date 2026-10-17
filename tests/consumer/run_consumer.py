#!/usr/bin/env python3
"""Run a finite, evidence-recording llama-server consumer check.

This runner is deliberately not registered as a GPU test. A live invocation
must be launched inside the project's CUDA controller admission. CPU unit
tests exercise the HTTP/oracle and owned-process lifetime against a fake local
server and never inspect or allocate a GPU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

import model_identity


REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = REPO_ROOT / "benchmarks/fixtures/llama-server-v200.json"
PROFILE_PATH = REPO_ROOT / "docs/v200/consumer-profile.json"
PROFILE_SCHEMA = "v200-consumer-profile/v1"
FIXTURE_SCHEMA = "v200-llama-server-fixture/v1"
STARTUP_TIMEOUT_S = 600.0
REQUEST_TIMEOUT_S = 120.0
SHUTDOWN_TIMEOUT_S = 15.0
MAX_SERVE_DURATION_S = 3000
DEFAULT_SERVE_DURATION_S = 1800
PROC_POLL_S = 0.10
UUID_RE = re.compile(r"^GPU-[A-Za-z0-9-]+$")


class ConsumerError(RuntimeError):
    """A fail-closed consumer run or fixture error."""


class RequestWallTimeout(TimeoutError):
    """Completion exceeded its total wall-clock allowance."""


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ConsumerError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ConsumerError(f"expected an object in {path}")
    return value


def load_documents(fixture_path: Path = FIXTURE_PATH,
                   profile_path: Path = PROFILE_PATH) -> tuple[dict[str, Any], dict[str, Any]]:
    fixture = _read_json(fixture_path)
    profile = _read_json(profile_path)
    validate_documents(fixture, profile)
    return fixture, profile


def validate_documents(fixture: dict[str, Any], profile: dict[str, Any]) -> None:
    if fixture.get("schema") != FIXTURE_SCHEMA:
        raise ConsumerError("unsupported fixture schema")
    if profile.get("schema") != PROFILE_SCHEMA:
        raise ConsumerError("unsupported consumer profile schema")
    model = fixture.get("model", {})
    binary = profile.get("binary", {})
    if (not _is_sha(model.get("sha256"))
            or not _is_sha(model.get("tokenizer", {}).get("sha256"))
            or model.get("tokenizer", {}).get("sha256") != model.get("sha256")
            or not _is_sha(binary.get("sha256"))):
        raise ConsumerError("model and executable must have pinned SHA-256 identities")
    if int(model.get("block_count", 0)) <= 0:
        raise ConsumerError("fixture needs a positive model block count")
    controls = fixture.get("controls", {})
    workloads = fixture.get("workloads", {})
    if not {"native_single", "native_pair", "facade_owner"}.issubset(controls):
        raise ConsumerError("fixture must define native single, native pair and facade controls")
    if not {"smoke", "resident_capacity"}.issubset(workloads):
        raise ConsumerError("fixture must define smoke and resident-capacity workloads")
    guard = fixture.get("memory_guard", {})
    if int(guard.get("per_physical_gpu_application_limit_mib", 0)) != 12288:
        raise ConsumerError("application allocation limit must be the authorized 12,288 MiB")
    if int(guard.get("physical_process_used_memory_guard_mib", 0)) <= 12288:
        raise ConsumerError("physical process-memory guard must include explicit context overhead")
    preview_environment = fixture.get("server", {}).get("environment", {})
    if preview_environment.get("GGML_CUDA_DISABLE_GRAPHS") != "1":
        raise ConsumerError("matched preview must explicitly disable CUDA graphs")
    if not fixture.get("server", {}).get("preview_policy"):
        raise ConsumerError("fixture must disclose the graph-off preview limitation")
    evidence_id = fixture.get("controls", {}).get("facade_owner", {}).get("required_b01_evidence_id")
    if (not isinstance(evidence_id, str)
            or re.fullmatch(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", evidence_id) is None):
        raise ConsumerError("pair facade must pin its required B01 setup evidence id")
    for workload_name, workload in workloads.items():
        oracle = workload.get("oracle", {})
        request = workload.get("request", {}).get("body", {})
        if not isinstance(request, dict) or request.get("stream") is not False:
            raise ConsumerError(f"{workload_name} must use a finite non-streaming request")
        if not isinstance(oracle.get("expected_exact"), str) or not oracle["expected_exact"]:
            raise ConsumerError(f"{workload_name} has no independent exact output oracle")
        if int(oracle.get("tokens_predicted_min", 0)) < 1:
            raise ConsumerError(f"{workload_name} has an invalid token-count lower bound")
        if int(oracle.get("tokens_predicted_max", 0)) < int(oracle["tokens_predicted_min"]):
            raise ConsumerError(f"{workload_name} has an invalid token-count upper bound")


def _is_sha(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def effective_gpu_layers(workload: dict[str, Any], fixture: dict[str, Any]) -> int:
    requested = int(workload["n_gpu_layers"])
    block_count = int(fixture["model"]["block_count"])
    return min(requested, block_count)


def memory_preflight(workload_name: str, control_name: str,
                     fixture: dict[str, Any], facade_mode: str | None = None) -> dict[str, Any]:
    """Estimate CUDA allocations using the selected control's actual placement."""
    try:
        workload = fixture["workloads"][workload_name]
        control = fixture["controls"][control_name]
        model = fixture["model"]
        layout = model["tensor_layout"]
        guard = fixture["memory_guard"]
        layers = effective_gpu_layers(workload, fixture)
        if layers == 0:
            layer_tensor_bytes = 0
        elif layers == int(model["block_count"]):
            layer_tensor_bytes = int(layout["all_block_tensor_bytes"])
        else:
            key = f"last_{layers}_block_tensor_bytes"
            if key not in layout:
                raise ConsumerError(f"missing descriptor byte count {key}")
            layer_tensor_bytes = int(layout[key])
        backend = control["backend"]
        if control_name == "native_pair":
            owner_count = 2
        elif (control_name == "facade_owner"
              and (facade_mode or control.get("v200_mode")) == "pair"):
            owner_count = 2
        else:
            owner_count = 1
        owners = 2 if owner_count == 2 else 1
        ctx = int(workload["ctx_size"])
        kv_bytes = (ctx * layers * int(model["kv_heads"])
                    * (int(model["key_length"]) + int(model["value_length"])) * 2)
        non_block = int(layout["non_block_tensor_bytes"])
        workspace = int(guard["workspace_reserve_bytes_per_gpu"])
        limit_bytes = int(guard["per_physical_gpu_application_limit_mib"]) * 1024 * 1024
        if control_name == "facade_owner" and owner_count == 2:
            # Every ordinary single-device allocation targets the visible anchor.
            # The VMM planner consumes that owner's budget first, then backs the
            # remainder from the peer. Model the same placement instead of
            # pretending the allocator stripes every llama buffer evenly.
            pair_payload = layer_tensor_bytes + non_block + kv_bytes
            estimate_total = pair_payload + workspace * owners
            owner_estimates = [min(estimate_total, limit_bytes),
                               max(0, estimate_total - limit_bytes)]
            placement = "preferred_owner_with_peer_spill"
        elif owner_count == 2:
            layer_first = (layer_tensor_bytes + 1) // 2
            layer_second = layer_tensor_bytes // 2
            kv_first = (kv_bytes + 1) // 2
            kv_second = kv_bytes // 2
            owner_estimates = [layer_first + non_block + kv_first + workspace,
                               layer_second + kv_second + workspace]
            estimate_total = sum(owner_estimates)
            placement = "native_layer_split_with_non_block_on_anchor"
        else:
            estimate_total = layer_tensor_bytes + non_block + kv_bytes + workspace
            owner_estimates = [estimate_total]
            placement = "single_owner"
        estimate = max(owner_estimates)
        result = {
            "workload": workload_name,
            "control": control_name,
            "effective_gpu_layers": layers,
            "requested_gpu_layers": int(workload["n_gpu_layers"]),
            "owner_count_assumed": owner_count,
            "gpu_layer_tensor_bytes_total": layer_tensor_bytes,
            "non_block_tensor_bytes_total_assumed": non_block,
            "kv_bytes_total": kv_bytes,
            "workspace_reserve_bytes_per_owner": workspace,
            "placement_policy": placement,
            "estimated_allocation_bytes_total": estimate_total,
            "estimated_bytes_by_owner": owner_estimates,
            "estimated_bytes_per_owner": estimate,
            "estimated_mib_per_owner": round(estimate / (1024 * 1024), 3),
            "application_limit_bytes_per_owner": limit_bytes,
            "fits_static_preflight": all(value <= limit_bytes for value in owner_estimates),
            "basis": "GGUF tensor descriptors + explicit f16 KV formula + workspace reserve, placed according to native split or V200 preferred-owner allocation with peer spill",
        }
        if backend not in ("native", "v200-preload"):
            raise ConsumerError(f"unknown backend {backend!r}")
        return result
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, ConsumerError):
            raise
        raise ConsumerError(f"invalid memory preflight inputs: {exc}") from exc


def validate_completion_response(response: Any, oracle: dict[str, Any]) -> dict[str, Any]:
    """Validate structure and exact normalized output, not mere non-emptiness."""
    if not isinstance(response, dict):
        raise ConsumerError("completion response must be a JSON object")
    content = response.get(str(oracle.get("response_field", "content")))
    if not isinstance(content, str):
        raise ConsumerError("completion response content must be a string")
    tokens = response.get("tokens_predicted")
    if isinstance(tokens, bool) or not isinstance(tokens, int):
        raise ConsumerError("completion response tokens_predicted must be an integer")
    minimum = int(oracle["tokens_predicted_min"])
    maximum = int(oracle["tokens_predicted_max"])
    if not minimum <= tokens <= maximum:
        raise ConsumerError(f"tokens_predicted {tokens} is outside [{minimum}, {maximum}]")
    if oracle.get("normalization") != "strip surrounding Unicode whitespace only":
        raise ConsumerError("unknown output normalization")
    normalized = content.strip()
    expected = oracle["expected_exact"]
    if normalized != expected:
        raise ConsumerError(f"exact output mismatch: expected {expected!r}, got {normalized!r}")
    return {"content": content, "normalized_content": normalized,
            "tokens_predicted": tokens, "expected_exact": expected,
            "exact_match": True}


def validate_live_environment(control_name: str, fixture: dict[str, Any],
                              environ: dict[str, str],
                              facade_mode: str | None = None) -> list[str]:
    cvd = environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not cvd:
        raise ConsumerError("CUDA_VISIBLE_DEVICES is unset; run under the CUDA controller")
    raw_uuids = environ.get("V200_DEVICE_UUIDS", "")
    uuids = [part.strip() for part in raw_uuids.split(",") if part.strip()]
    if not uuids or any(UUID_RE.fullmatch(value) is None for value in uuids):
        raise ConsumerError("V200_DEVICE_UUIDS must contain controller-admitted GPU UUIDs")
    mode = facade_mode or fixture["controls"].get("facade_owner", {}).get("v200_mode")
    if control_name == "facade_owner" and mode not in ("anchor", "pair"):
        raise ConsumerError(f"unsupported facade mode {mode!r}")
    expected_count = 2 if control_name == "native_pair" or (
        control_name == "facade_owner" and mode == "pair") else 1
    if len(uuids) < expected_count:
        raise ConsumerError(f"{control_name} needs {expected_count} selected UUID(s), got {len(uuids)}")
    return uuids[:expected_count]


def _clean_environment(control_name: str, fixture: dict[str, Any],
                       preloads: list[Path] | None,
                       environ: dict[str, str] | None = None,
                       facade_mode: str | None = None,
                       selected_uuids: list[str] | None = None,
                       profile_log_path: Path | None = None,
                       cuda_graphs_enabled: bool = False) -> dict[str, str]:
    source = dict(os.environ if environ is None else environ)
    child = {key: value for key, value in source.items()
             if not key.startswith("LLAMA_ARG_") and not key.startswith("GGML_CUDA_")}
    pair_compute_requested = child.pop("V200_PAIR_COMPUTE", None) == "1"
    operator_verify_requested = child.pop("V200_OPERATOR_VERIFY", None) == "1"
    child.pop("V200_GRAPH_CAPTURE", None)
    inherited = [part for part in child.pop("LD_PRELOAD", "").split()
                 if "libv200" not in Path(part).name.lower()]
    child.pop("V200_MODE", None)
    child.pop("V200_DEVICE_UUIDS", None)
    child.pop("V200_PROFILE_LOG", None)
    child.pop("V200_B01_EVIDENCE_ID", None)
    for key, value in fixture["server"].get("environment", {}).items():
        child[str(key)] = str(value)
    if cuda_graphs_enabled:
        # The native and facade controls must exercise the same llama graph
        # path. Only the pair facade receives V200's explicit capture opt-in.
        child.pop("GGML_CUDA_DISABLE_GRAPHS", None)
    if control_name == "facade_owner":
        if not preloads:
            raise ConsumerError("facade_owner requires at least one --preload-library")
        control = fixture["controls"][control_name]
        mode = str(facade_mode or control.get("v200_mode", ""))
        selected = ",".join(selected_uuids or [])
        if not mode or not selected:
            raise ConsumerError("facade fixture must define a mode and inherit controller UUIDs")
        if mode == "pair" and len(preloads) < 2:
            raise ConsumerError("facade pair mode requires frontdoor and root pair-initializer libraries")
        if pair_compute_requested and mode == "pair":
            if not any(Path(path).name == "libv200_operators.so" for path in preloads):
                raise ConsumerError("pair compute requires the pair operator library preload")
            child["V200_PAIR_COMPUTE"] = "1"
            if operator_verify_requested:
                child["V200_OPERATOR_VERIFY"] = "1"
        child["V200_MODE"] = mode
        child["V200_DEVICE_UUIDS"] = selected
        child["LD_PRELOAD"] = ":".join([*(str(path) for path in preloads), *inherited])
        if mode == "pair":
            child["V200_B01_EVIDENCE_ID"] = str(control["required_b01_evidence_id"])
            if cuda_graphs_enabled:
                child["V200_GRAPH_CAPTURE"] = "1"
        if profile_log_path is not None:
            child["V200_PROFILE_LOG"] = str(profile_log_path.resolve())
    elif inherited:
        child["LD_PRELOAD"] = " ".join(inherited)
    return child


def _git_source_identity(profile: dict[str, Any]) -> dict[str, Any]:
    source = profile["source"]
    root = Path(source["repository"])
    if not root.is_dir():
        raise ConsumerError(f"pinned source repository is missing: {root}")
    head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                          check=True, capture_output=True, text=True, timeout=10).stdout.strip()
    status = subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
                            check=True, capture_output=True, text=True, timeout=10).stdout
    if head != source["commit"]:
        raise ConsumerError(f"llama.cpp HEAD {head} differs from pinned {source['commit']}")
    if status.strip():
        raise ConsumerError("llama.cpp tracked worktree is dirty")
    return {"repository": str(root), "commit": head, "tracked_worktree_clean": True}


def verify_identities(profile: dict[str, Any], fixture: dict[str, Any],
                      fixture_path: Path = FIXTURE_PATH,
                      profile_path: Path = PROFILE_PATH,
                      preloads: list[Path] | None = None) -> dict[str, Any]:
    binary = profile["binary"]
    try:
        model_identity_result = model_identity.identify_model(fixture["model"]["path"])
    except model_identity.ModelIdentityError as exc:
        raise ConsumerError(f"cannot establish stable model identity: {exc}") from exc
    expected_model_sha = fixture["model"]["sha256"]
    if model_identity_result["sha256"] != expected_model_sha:
        raise ConsumerError(
            f"hash mismatch for {fixture['model']['path']}: expected {expected_model_sha}, "
            f"got {model_identity_result['sha256']}")
    identities: dict[str, Any] = {
        "profile_sha256": sha256_file(profile_path),
        "fixture_sha256": sha256_file(fixture_path),
        "binary": _verify_hash_entry(binary),
        "runtime_closure": [_verify_hash_entry(item) for item in profile["runtime_closure"]],
        "model": {"path": fixture["model"]["path"],
                  "sha256": model_identity_result["sha256"],
                  "identity_provenance": model_identity_result["provenance"],
                  "file_signature": model_identity_result["signature"]},
        "source": _git_source_identity(profile),
    }
    if preloads:
        identities["preloads"] = []
        for index, preload in enumerate(preloads):
            if not preload.is_file():
                raise ConsumerError(f"preload library not found: {preload}")
            identities["preloads"].append({"ordinal": index, "path": str(preload.resolve()),
                                            "sha256": sha256_file(preload)})
    return identities


def _verify_hash_entry(entry: dict[str, Any]) -> dict[str, str]:
    return {"path": entry["path"], "sha256": _verify_hash(entry["path"], entry["sha256"])}


def _verify_hash(path_text: str, expected: str) -> str:
    path = Path(path_text)
    if not path.is_file():
        raise ConsumerError(f"pinned file is missing: {path}")
    observed = sha256_file(path)
    if observed != expected:
        raise ConsumerError(f"hash mismatch for {path}: expected {expected}, got {observed}")
    return observed


def _argv_for(control_name: str, workload: dict[str, Any], fixture: dict[str, Any],
              binary_path: str, port: int, workload_name: str = "smoke") -> list[str]:
    control = fixture["controls"][control_name]
    argv = [binary_path]
    replacements = {"{model}": fixture["model"]["path"],
                    "{port}": str(port),
                    "{ctx_size}": str(int(workload["ctx_size"])),
                    "{n_gpu_layers}": str(int(workload["n_gpu_layers"]))}
    for value in fixture["server"]["fixed_arguments"]:
        argv.append(replacements.get(value, value))
    argv.extend(str(arg) for arg in control["arguments"])
    if workload_name == "resident_capacity":
        # llama.cpp normally keeps its input embedding on CPU even when every
        # supported layer is offloaded. Keep the full-resident workload's GGUF
        # tensor payload on CUDA0, and retain trace logs proving that placement.
        argv.extend(["--override-tensor", r"^token_embd\.weight$=CUDA0",
                     "--verbosity", "5"])
    return argv


def _free_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _proc_group_identity(pid: int) -> tuple[int, int] | None:
    try:
        data = Path(f"/proc/{pid}/stat").read_text()
        tail = data[data.rfind(")") + 2:].split()
        return int(tail[2]), int(tail[3])  # pgrp, session
    except (OSError, ValueError, IndexError):
        return None


def _owned_members(pgid: int, session_id: int) -> list[dict[str, Any]]:
    result = []
    proc = Path("/proc")
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        identity = _proc_group_identity(pid)
        if identity != (pgid, session_id):
            continue
        try:
            tail = (entry / "stat").read_text()
            state = tail[tail.rfind(")") + 2:].split()[0]
            result.append({"pid": pid, "state": state})
        except (OSError, IndexError):
            continue
    return result


def cleanup_owned_group(proc: subprocess.Popen[Any], shutdown_timeout_s: float = SHUTDOWN_TIMEOUT_S,
                        sleeper: Callable[[float], None] = time.sleep) -> dict[str, Any]:
    pgid = proc.pid
    session = proc.pid
    attempted = False
    escalated = False
    try:
        os.killpg(pgid, signal.SIGTERM)
        attempted = True
    except ProcessLookupError:
        pass
    deadline = time.monotonic() + shutdown_timeout_s
    while time.monotonic() < deadline:
        try:
            proc.poll()
        except OSError:
            pass
        live = [item for item in _owned_members(pgid, session) if item["state"] != "Z"]
        if not live:
            break
        sleeper(PROC_POLL_S)
    else:
        try:
            os.killpg(pgid, signal.SIGKILL)
            escalated = True
        except ProcessLookupError:
            pass
    try:
        proc.wait(timeout=2)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(pgid, signal.SIGKILL)
            escalated = True
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            pass
    remaining = _owned_members(pgid, session)
    live_remaining = [item for item in remaining if item["state"] != "Z"]
    return {"signaled_owned_group": attempted or escalated,
            "sigkill_escalation": escalated,
            "owned_process_group": pgid,
            "owned_session": session,
            "remaining_members": remaining,
            "no_live_group_members": not live_remaining,
            "zombie_members_only": bool(remaining) and not live_remaining}


def _proc_ids_for_group(pgid: int, session: int) -> set[int]:
    return {entry["pid"] for entry in _owned_members(pgid, session)
            if entry["state"] != "Z"}


def _nvidia_smi_sample(nvidia_smi: str, pgid: int, session: int,
                       selected_uuids: list[str], timeout_s: float = 5.0) -> dict[str, Any]:
    command = [nvidia_smi, "--query-compute-apps=pid,gpu_uuid,used_memory",
               "--format=csv,noheader,nounits"]
    completed = subprocess.run(command, check=True, capture_output=True,
                               text=True, timeout=timeout_s)
    owned = _proc_ids_for_group(pgid, session)
    per_uuid: dict[str, int] = {uuid: 0 for uuid in selected_uuids}
    rows = []
    for raw in completed.stdout.splitlines():
        if not raw.strip():
            continue
        parts = [part.strip() for part in raw.split(",")]
        if len(parts) != 3:
            raise ConsumerError(f"malformed nvidia-smi compute-app row: {raw!r}")
        pid_text, gpu_uuid, memory_text = parts
        try:
            pid, memory = int(pid_text), int(memory_text)
        except ValueError as exc:
            raise ConsumerError(f"unparseable nvidia-smi row: {raw!r}") from exc
        if gpu_uuid not in per_uuid:
            continue
        is_owned = pid in owned
        rows.append({"pid": pid, "gpu_uuid": gpu_uuid, "used_memory_mib": memory,
                     "owned_process_group": is_owned})
        if is_owned:
            per_uuid[gpu_uuid] += memory
    return {"sampled_unix_time": time.time(), "owned_pids": sorted(owned),
            "rows": rows, "owned_used_memory_mib_by_uuid": per_uuid,
            "owned_used_memory_bytes_by_uuid": {
                uuid: memory_mib * 1024 * 1024 for uuid, memory_mib in per_uuid.items()}}


def _verify_nvidia_smi_uuid_selection(selected_uuids: list[str]) -> tuple[str, list[str]]:
    executable = shutil.which("nvidia-smi")
    if not executable:
        raise ConsumerError("nvidia-smi is unavailable; refusing unmonitored live execution")
    command = [executable, "--query-gpu=uuid", "--format=csv,noheader"]
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError) as exc:
        raise ConsumerError(f"cannot verify controller-selected GPU UUIDs: {exc}") from exc
    observed = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    missing = [uuid for uuid in selected_uuids if uuid not in observed]
    if missing:
        raise ConsumerError(f"selected GPU UUIDs are not visible through nvidia-smi: {missing}")
    return executable, observed


def _nvidia_smi_memory(nvidia_smi: str, selected_uuids: list[str]) -> dict[str, dict[str, int]]:
    command = [nvidia_smi, "--query-gpu=uuid,memory.total,memory.free",
               "--format=csv,noheader,nounits"]
    try:
        completed = subprocess.run(command, check=True, capture_output=True,
                                   text=True, timeout=5)
    except (OSError, subprocess.SubprocessError) as exc:
        raise ConsumerError(f"cannot query selected physical GPU memory: {exc}") from exc
    observed: dict[str, dict[str, int]] = {}
    for raw in completed.stdout.splitlines():
        if not raw.strip():
            continue
        parts = [part.strip() for part in raw.split(",")]
        if len(parts) != 3:
            raise ConsumerError(f"malformed nvidia-smi GPU-memory row: {raw!r}")
        gpu_uuid, total_text, free_text = parts
        try:
            total_mib, free_mib = int(total_text), int(free_text)
        except ValueError as exc:
            raise ConsumerError(f"unparseable nvidia-smi GPU-memory row: {raw!r}") from exc
        if total_mib <= 0 or free_mib < 0 or free_mib > total_mib:
            raise ConsumerError(f"invalid nvidia-smi physical memory values: {raw!r}")
        observed[gpu_uuid] = {
            "total_mib": total_mib,
            "free_mib": free_mib,
            "total_bytes": total_mib * 1024 * 1024,
            "free_bytes": free_mib * 1024 * 1024,
        }
    missing = [gpu_uuid for gpu_uuid in selected_uuids if gpu_uuid not in observed]
    if missing:
        raise ConsumerError(f"selected physical GPU memory is unavailable for UUIDs: {missing}")
    return {gpu_uuid: observed[gpu_uuid] for gpu_uuid in selected_uuids}


def _check_physical_headroom(workload_name: str, control_name: str,
                             fixture: dict[str, Any], preflight: dict[str, Any],
                             selected_uuids: list[str],
                             memory_by_uuid: dict[str, dict[str, int]],
                             facade_mode: str | None) -> dict[str, Any]:
    """Refuse a full-resident run when live per-owner headroom cannot fit it."""
    reserve = 1024 * 1024 * 1024
    cap = int(fixture["memory_guard"]["per_physical_gpu_application_limit_mib"]) * 1024 * 1024
    qualified = [max(0, min(cap, memory_by_uuid[item]["free_bytes"] - reserve))
                 for item in selected_uuids]
    required = list(preflight["estimated_bytes_by_owner"])
    if control_name == "facade_owner" and (facade_mode or "pair") == "pair":
        # Each V200 allocation goes to the visible anchor first. If that owner
        # has less live headroom than the static cap, the spill boundary moves
        # with it; the manager independently enforces the same startup budgets.
        total_required = int(preflight["estimated_allocation_bytes_total"])
        required = [min(total_required, qualified[0]),
                    max(0, total_required - min(total_required, qualified[0]))]
    if len(required) != len(selected_uuids):
        raise ConsumerError("preflight owner count differs from controller-selected physical GPUs")
    failures = [
        {"gpu_uuid": gpu_uuid, "required_bytes": needed,
         "qualified_app_bytes": budget}
        for gpu_uuid, needed, budget in zip(selected_uuids, required, qualified)
        if needed > budget
    ]
    if failures:
        raise ConsumerError(f"insufficient live pair headroom for the unchanged workload: {failures}")
    capacity_payload_bytes = int(
        fixture["model"]["tensor_layout"]["total_tensor_bytes"])
    physical_capacity_max = max(memory_by_uuid[item]["total_bytes"]
                                for item in selected_uuids)
    exceeds_physical_capacity = capacity_payload_bytes > physical_capacity_max
    if workload_name == "resident_capacity" and not exceeds_physical_capacity:
        raise ConsumerError(
            "resident-capacity payload does not exceed the largest selected physical GPU capacity")
    return {
        "status": "PASS",
        "physical_reserve_bytes_per_owner": reserve,
        "qualified_application_bytes_by_uuid": dict(zip(selected_uuids, qualified)),
        "required_application_bytes_by_uuid": dict(zip(selected_uuids, required)),
        "resident_payload_bytes": capacity_payload_bytes,
        "maximum_selected_physical_capacity_bytes": physical_capacity_max,
        "payload_exceeds_each_physical_gpu_capacity": (
            workload_name == "resident_capacity" and exceeds_physical_capacity),
    }


class ProcessMemoryGuard:
    def __init__(self, proc: subprocess.Popen[Any], selected_uuids: list[str],
                 limit_mib: int, poll_seconds: float,
                 nvidia_smi: str | None = None):
        self.proc = proc
        self.selected_uuids = selected_uuids
        self.limit_mib = limit_mib
        self.poll_seconds = poll_seconds
        self.nvidia_smi = nvidia_smi or shutil.which("nvidia-smi")
        self.samples: list[dict[str, Any]] = []
        self.error: str | None = None
        self.over_limit: dict[str, Any] | None = None
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._loop, name="v200-process-memory-guard", daemon=True)

    def start(self) -> None:
        if not self.nvidia_smi:
            raise ConsumerError("nvidia-smi is unavailable; refusing unguarded live launch")
        self.thread.start()

    def _loop(self) -> None:
        while not self.stop.is_set() and self.proc.poll() is None:
            try:
                sample = _nvidia_smi_sample(self.nvidia_smi, self.proc.pid, self.proc.pid,
                                            self.selected_uuids)
                self.samples.append(sample)
                for uuid, used in sample["owned_used_memory_mib_by_uuid"].items():
                    if used > self.limit_mib:
                        self.over_limit = {"gpu_uuid": uuid, "used_memory_mib": used,
                                           "limit_mib": self.limit_mib}
                        os.killpg(self.proc.pid, signal.SIGTERM)
                        self.stop.set()
                        return
            except Exception as exc:  # fail closed on telemetry gaps
                self.error = f"{type(exc).__name__}: {exc}"
                try:
                    os.killpg(self.proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                self.stop.set()
                return
            self.stop.wait(self.poll_seconds)

    def finish(self) -> None:
        self.stop.set()
        if self.thread.is_alive():
            self.thread.join(timeout=6)


def _http_json(method: str, url: str, body: dict[str, Any] | None = None,
               timeout_s: float = 5.0) -> tuple[int, Any]:
    data = None if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json"} if data is not None else {}
    request = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            payload = response.read()
            return response.status, json.loads(payload.decode("utf-8")) if payload else None
    except urllib.error.HTTPError as exc:
        try:
            payload = exc.read()
            parsed = json.loads(payload.decode("utf-8")) if payload else None
        except (UnicodeError, json.JSONDecodeError):
            parsed = None
        return exc.code, parsed


def _completion_request(url: str, body: dict[str, Any], timeout_s: float) -> tuple[int, Any]:
    """Apply a total wall-clock deadline in addition to urllib's socket timeout."""
    if threading.current_thread() is not threading.main_thread() or not hasattr(signal, "setitimer"):
        raise ConsumerError("finite completion wall timer requires the main thread on POSIX")
    prior_handler = signal.getsignal(signal.SIGALRM)
    prior_timer = signal.getitimer(signal.ITIMER_REAL)

    def raise_timeout(_signum: int, _frame: Any) -> None:
        raise RequestWallTimeout(f"completion exceeded {timeout_s:g}s wall deadline")

    signal.signal(signal.SIGALRM, raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, timeout_s)
    try:
        return _http_json("POST", url, body, timeout_s)
    finally:
        signal.setitimer(signal.ITIMER_REAL, *prior_timer)
        signal.signal(signal.SIGALRM, prior_handler)


def wait_for_readiness(proc: subprocess.Popen[Any], base_url: str,
                       timeout_s: float = STARTUP_TIMEOUT_S,
                       sleeper: Callable[[float], None] = time.sleep) -> float:
    start = time.monotonic()
    deadline = start + timeout_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise ConsumerError(f"server exited before readiness with status {proc.returncode}")
        try:
            status, response = _http_json("GET", base_url + "/health", timeout_s=2.0)
            if status == 200 and isinstance(response, dict):
                if response.get("status") in ("ok", "ready") or "status" not in response:
                    return time.monotonic() - start
        except (OSError, TimeoutError, json.JSONDecodeError):
            pass
        sleeper(0.25)
    raise ConsumerError(f"server did not become ready within {timeout_s:g}s")


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as target:
            json.dump(value, target, indent=2, sort_keys=True, ensure_ascii=False)
            target.write("\n")
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _read_profile_log(path: Path, selected_uuids: list[str],
                      max_owner_budget_bytes: int,
                      expected_qualification_id: str | None = None,
                      minimum_resident_bytes: int = 0) -> dict[str, Any]:
    if not path.exists():
        return {"status": "MISSING", "path": str(path), "records": []}
    try:
        raw = path.read_bytes()
        records = []
        operator_records = []
        accepted_budgets: list[int] | None = None
        peak_used = [0, 0]
        peak_total = 0
        peak_event = None
        for line_number, line in enumerate(raw.splitlines(), start=1):
            if not line.strip():
                continue
            value = json.loads(line.decode("utf-8"))
            if not isinstance(value, dict):
                raise ValueError(f"line {line_number} is not a JSON object")
            if value.get("schema") == "v200-operator-ledger/v1":
                operator_required = (
                    "event", "operation", "route", "matched_calls",
                    "anchor_local_calls", "peer_owner_calls", "split_calls",
                    "anchor_rows", "peer_rows", "verified_rows",
                    "max_abs_error", "status",
                )
                missing = [field for field in operator_required if field not in value]
                if missing:
                    raise ValueError(f"line {line_number} operator record missing fields {missing}")
                if any(not isinstance(value[field], str)
                       for field in ("event", "operation", "route")):
                    raise ValueError(f"line {line_number} operator event fields are malformed")
                if any(isinstance(value[field], bool) or not isinstance(value[field], int)
                       for field in (*operator_required[3:10], "status")):
                    raise ValueError(f"line {line_number} operator counters are malformed")
                if (isinstance(value["max_abs_error"], bool)
                        or not isinstance(value["max_abs_error"], (int, float))):
                    raise ValueError(f"line {line_number} operator error is malformed")
                operator_records.append(value)
                continue
            if value.get("schema") != "v200-frontdoor-ledger/v1":
                raise ValueError(f"line {line_number} has unexpected ledger schema")
            records.append(value)
            required = ("schema", "event", "mode", "anchor_uuid", "peer_uuid",
                        "qualification_id", "owner_budget_bytes", "owner_used_bytes",
                        "live_allocation_count", "live_reservation_count", "cleanup_pending_count")
            missing = [field for field in required if field not in value]
            if missing:
                raise ValueError(f"line {line_number} missing fields {missing}")
            if value["mode"] != "pair":
                raise ValueError(f"line {line_number} has unexpected ledger mode")
            if value["anchor_uuid"] != selected_uuids[0] or value["peer_uuid"] != selected_uuids[1]:
                raise ValueError(f"line {line_number} UUID ordering differs from controller selection")
            if (expected_qualification_id is not None
                    and value["qualification_id"] != expected_qualification_id):
                raise ValueError(f"line {line_number} qualification id differs from the fixture-pinned B01 evidence")
            budgets = value["owner_budget_bytes"]
            usage = value["owner_used_bytes"]
            if (not isinstance(budgets, list) or len(budgets) != 2
                    or any(isinstance(item, bool) or not isinstance(item, int)
                           or item <= 0 or item > max_owner_budget_bytes for item in budgets)):
                raise ValueError(f"line {line_number} owner budgets are malformed")
            if accepted_budgets is None:
                accepted_budgets = list(budgets)
            elif budgets != accepted_budgets:
                raise ValueError(f"line {line_number} owner budgets changed during the process")
            if (not isinstance(usage, list) or len(usage) != 2
                    or any(isinstance(item, bool) or not isinstance(item, int) for item in usage)):
                raise ValueError(f"line {line_number} owner usage is malformed")
            if any(item < 0 for item in usage):
                raise ValueError(f"line {line_number} owner usage is negative")
            if any(item > budget for item, budget in zip(usage, budgets)):
                raise OverflowError(f"line {line_number} owner usage exceeded configured budget")
            total_used = sum(usage)
            if total_used > peak_total:
                peak_total = total_used
                peak_used = list(usage)
                peak_event = value["event"]
            for field in ("live_allocation_count", "live_reservation_count", "cleanup_pending_count"):
                if isinstance(value[field], bool) or not isinstance(value[field], int) or value[field] < 0:
                    raise ValueError(f"line {line_number} {field} is malformed")
        cleanup_ok = bool(records) and all(records[-1][field] == 0 for field in (
            "live_allocation_count", "live_reservation_count", "cleanup_pending_count")) \
            and records[-1]["owner_used_bytes"] == [0, 0]
        if not records:
            status = "EMPTY"
        elif not cleanup_ok:
            status = "LIVE_RESOURCES_AT_EXIT"
        elif minimum_resident_bytes > 0 and (
                peak_total < minimum_resident_bytes or min(peak_used) <= 0):
            status = "INSUFFICIENT_RESIDENT_PAYLOAD"
        else:
            status = "PASS"
        return {"status": status, "path": str(path),
                "sha256": hashlib.sha256(raw).hexdigest(), "records": records,
                "operator_records": operator_records,
                "graph_records": [record for record in records
                                  if str(record.get("event", "")).startswith("graph_")],
                "owner_budget_bytes": accepted_budgets,
                "peak_owner_used_bytes": peak_used,
                "peak_total_owner_used_bytes": peak_total,
                "peak_event": peak_event,
                "cleanup_accounting_zero": cleanup_ok,
                "minimum_resident_bytes_required": minimum_resident_bytes}
    except OverflowError as exc:
        return {"status": "OVER_BUDGET", "path": str(path),
                "sha256": hashlib.sha256(raw).hexdigest() if "raw" in locals() else None,
                "records": records if "records" in locals() else [],
                "operator_records": operator_records if "operator_records" in locals() else [],
                "graph_records": [record for record in records
                                  if str(record.get("event", "")).startswith("graph_")]
                                if "records" in locals() else [],
                "error": str(exc)}
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        return {"status": "INVALID", "path": str(path),
                "sha256": hashlib.sha256(raw).hexdigest() if "raw" in locals() else None,
                "records": records if "records" in locals() else [],
                "operator_records": operator_records if "operator_records" in locals() else [],
                "graph_records": [record for record in records
                                  if str(record.get("event", "")).startswith("graph_")]
                                if "records" in locals() else [],
                "error": f"{type(exc).__name__}: {exc}"}


def _layer_offload_evidence(path: Path, workload_name: str,
                            fixture: dict[str, Any]) -> dict[str, Any]:
    try:
        log = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return {"status": "MISSING", "path": str(path), "error": str(exc)}
    matches = re.findall(r"offloaded\s+(\d+)\s*/\s*(\d+)\s+layers\s+to\s+GPU",
                         log, flags=re.IGNORECASE)
    if not matches:
        return {"status": "MISSING", "path": str(path), "matched_lines": []}
    offloaded, available = (int(item) for item in matches[-1])
    model_layers = int(fixture["model"]["block_count"])
    expected = model_layers + 1 if workload_name == "resident_capacity" else None
    full = offloaded == available and (expected is None or available >= expected)
    return {
        "status": "PASS" if full or expected is None else "PARTIAL_OFFLOAD",
        "path": str(path),
        "offloaded_layers": offloaded,
        "available_layers": available,
        "full_gpu_offload": full,
        "required_layers": expected,
    }


def _input_embedding_evidence(path: Path, workload_name: str) -> dict[str, Any]:
    if workload_name != "resident_capacity":
        return {"status": "NOT_REQUIRED"}
    try:
        log = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return {"status": "MISSING", "path": str(path), "error": str(exc)}
    matches = re.findall(
        r"tensor\s+(token_embd\.weight)\s+\([^\n]*?\)\s+buffer type overridden to\s+(\S+)",
        log, flags=re.IGNORECASE)
    if not matches:
        return {"status": "MISSING", "path": str(path),
                "expected_buffer_type": "CUDA0"}
    tensor_name, buffer_type = matches[-1]
    return {"status": "PASS" if buffer_type == "CUDA0" else "WRONG_BUFFER_TYPE",
            "path": str(path), "tensor": tensor_name,
            "buffer_type": buffer_type, "required_buffer_type": "CUDA0"}


def _run_one(workload_name: str, control_name: str, fixture: dict[str, Any],
             profile: dict[str, Any], identities: dict[str, Any],
             selected_uuids: list[str], output_dir: Path,
             preloads: list[Path] | None, monitor: bool = True,
             facade_mode: str | None = None,
             environ: dict[str, str] | None = None,
             popen_factory: Callable[..., Any] = subprocess.Popen,
             nvidia_smi_path: str | None = None,
             visible_gpu_uuids: list[str] | None = None,
             server_port: int | None = None,
             serve_duration_seconds: int = 0,
             physical_memory_baseline: dict[str, dict[str, int]] | None = None,
             cuda_graphs_enabled: bool = False) -> dict[str, Any]:
    workload = fixture["workloads"][workload_name]
    preflight = memory_preflight(workload_name, control_name, fixture, facade_mode)
    if not preflight["fits_static_preflight"]:
        raise ConsumerError(
            f"static preflight estimates {preflight['estimated_mib_per_owner']} MiB/owner, "
            f"over {preflight['application_limit_bytes_per_owner'] / (1024 * 1024):.0f} MiB"
        )
    physical_headroom = None
    if physical_memory_baseline is not None:
        physical_headroom = _check_physical_headroom(
            workload_name, control_name, fixture, preflight, selected_uuids,
            physical_memory_baseline, facade_mode)
    elif monitor and workload_name == "resident_capacity":
        raise ConsumerError("resident-capacity workload requires live selected-GPU headroom")
    if isinstance(serve_duration_seconds, bool) or not 0 <= serve_duration_seconds <= MAX_SERVE_DURATION_S:
        raise ConsumerError(f"serve duration must be between 0 and {MAX_SERVE_DURATION_S} seconds")
    port = _free_loopback_port() if server_port is None else int(server_port)
    if not 1 <= port <= 65535:
        raise ConsumerError("server port must be between 1 and 65535")
    argv = _argv_for(control_name, workload, fixture,
                     profile["binary"]["path"], port, workload_name)
    started = time.time()
    monotonic_start = time.monotonic()
    run_id = f"{workload_name}-{control_name}-{time.time_ns()}"
    log_path = output_dir / f"{run_id}.log"
    result_path = output_dir / f"{run_id}.json"
    resolved_facade_mode = facade_mode or fixture["controls"]["facade_owner"].get("v200_mode")
    profile_log_path = (output_dir / f"{run_id}.v200-profile.jsonl"
                        if control_name == "facade_owner" and resolved_facade_mode == "pair" else None)
    child_env = _clean_environment(control_name, fixture, preloads, environ,
                                   facade_mode, selected_uuids, profile_log_path,
                                   cuda_graphs_enabled)
    result: dict[str, Any] = {
        "schema": "v200-consumer-result/v1",
        "run_id": run_id,
        "workload": workload_name,
        "workload_purpose": workload["purpose"],
        "control": control_name,
        "backend": fixture["controls"][control_name]["backend"],
        "partial_offload_preview": workload_name == "smoke",
        "capacity_claim": ("NONE: partial CPU-offload correctness preview"
                           if workload_name == "smoke"
                           else "PENDING: requires full layer-offload and live pair-residency evidence"),
        "fixture_id": fixture["fixture_id"],
        "identities": identities,
        "selected_gpu_uuids": selected_uuids,
        "cuda_visible_devices": child_env.get("CUDA_VISIBLE_DEVICES"),
        "v200_mode": child_env.get("V200_MODE"),
        "v200_b01_evidence_id": child_env.get("V200_B01_EVIDENCE_ID"),
        "preview_environment": {
            "GGML_CUDA_DISABLE_GRAPHS": child_env.get("GGML_CUDA_DISABLE_GRAPHS"),
            "V200_GRAPH_CAPTURE": child_env.get("V200_GRAPH_CAPTURE"),
            "cuda_graphs_enabled": cuda_graphs_enabled,
            "policy": fixture["server"]["preview_policy"],
        },
        "preflight": preflight,
        "physical_gpu_memory_before_launch": physical_memory_baseline or {},
        "physical_headroom_preflight": physical_headroom,
        "allocation_evidence": {
            "application_limit_bytes_per_owner": preflight["application_limit_bytes_per_owner"],
            "facade_limit_enforcer": "V200 per-owner allocation ledger" if control_name == "facade_owner" else None,
            "native_limit_enforcement": "descriptor-based preflight only; no native allocator ledger",
            "v200_memory_api_receipt": "not exported by this consumer process launch contract",
            "fixture_v200_owner_budget_caps": ([
                {"gpu_uuid": uuid,
                 "maximum_budget_bytes": int(fixture["memory_guard"]["per_physical_gpu_application_limit_mib"]) * 1024 * 1024,
                 "startup_policy": "min(cap, native free bytes minus 1 GiB reserve)"}
                for uuid in selected_uuids] if control_name == "facade_owner" else []),
        },
        "argv": argv,
        "request": workload["request"],
        "expected_oracle": workload["oracle"],
        "started_unix_time": started,
        "started_monotonic": monotonic_start,
        "server_process": None,
        "memory_guard": {"application_allocation_limit_mib": int(
            fixture["memory_guard"]["per_physical_gpu_application_limit_mib"]),
            "physical_process_used_memory_guard_mib": int(
                fixture["memory_guard"]["physical_process_used_memory_guard_mib"]),
            "nvidia_smi_path": nvidia_smi_path,
            "nvidia_smi_visible_gpu_uuids": visible_gpu_uuids or [],
            "samples": [], "telemetry_status": "not-started"},
        "v200_profile_telemetry": {"status": ("PENDING" if profile_log_path is not None else
                                                "NOT_AVAILABLE_ANCHOR_NO_LEDGER" if control_name == "facade_owner" else "NOT_APPLICABLE"),
                                    "path": str(profile_log_path) if profile_log_path else None},
        "readiness_seconds": None,
        "request_seconds": None,
        "response": None,
        "service": ({"endpoint": f"http://127.0.0.1:{port}",
                     "duration_requested_seconds": serve_duration_seconds,
                     "duration_observed_seconds": None,
                     "status": "PENDING"} if serve_duration_seconds else None),
        "application_status": "FAIL",
        "gpu_qualification": "NOT_CLAIMED",
        "layer_offload_evidence": {"status": "PENDING"},
        "failure": None,
        "cleanup": None,
        "log_path": str(log_path),
        "result_path": str(result_path),
    }
    process: subprocess.Popen[Any] | None = None
    memory_guard: ProcessMemoryGuard | None = None
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        with log_path.open("wb") as log:
            old_mask = None
            if hasattr(signal, "pthread_sigmask"):
                old_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM})
            try:
                popen_options: dict[str, Any] = {}
                if old_mask is not None:
                    # Popen children inherit the calling thread's signal mask
                    # across exec. Unblock SIGTERM in the child before exec so
                    # its owned-process cleanup remains effective while the
                    # parent closes the Popen assignment race.
                    popen_options["preexec_fn"] = lambda: signal.pthread_sigmask(
                        signal.SIG_UNBLOCK, {signal.SIGTERM})
                process = popen_factory(argv, cwd=str(profile["source"]["repository"]), env=child_env,
                                        stdout=log, stderr=subprocess.STDOUT,
                                        start_new_session=True, **popen_options)
                result["server_process"] = {"pid": process.pid, "process_group": process.pid,
                                            "session": process.pid}
            finally:
                if old_mask is not None:
                    signal.pthread_sigmask(signal.SIG_SETMASK, old_mask)
            if monitor:
                memory_guard = ProcessMemoryGuard(
                    process, selected_uuids,
                    int(fixture["memory_guard"]["physical_process_used_memory_guard_mib"]),
                    float(fixture["memory_guard"]["nvidia_smi_poll_seconds"]), nvidia_smi_path)
                memory_guard.start()
                result["memory_guard"]["telemetry_status"] = "running-fail-closed"
            result["readiness_seconds"] = wait_for_readiness(
                process, f"http://127.0.0.1:{port}",
                float(fixture["server"]["timeouts_seconds"]["startup"]))
            if memory_guard and (memory_guard.error or memory_guard.over_limit):
                raise ConsumerError(f"memory guard stopped server: {memory_guard.error or memory_guard.over_limit}")
            request_start = time.monotonic()
            if str(workload["request"]["method"]) != "POST":
                raise ConsumerError("fixture completion method must be POST")
            status, response = _completion_request(
                f"http://127.0.0.1:{port}{workload['request']['path']}",
                workload["request"]["body"],
                float(fixture["server"]["timeouts_seconds"]["request"]))
            result["request_seconds"] = time.monotonic() - request_start
            if status != 200:
                raise ConsumerError(f"completion returned HTTP {status}: {response!r}")
            result["response"] = validate_completion_response(response, workload["oracle"])
            if memory_guard and (memory_guard.error or memory_guard.over_limit):
                raise ConsumerError(f"memory guard stopped server: {memory_guard.error or memory_guard.over_limit}")
            if serve_duration_seconds:
                print(f"V200_SERVER_READY http://127.0.0.1:{port}", flush=True)
                service_start = time.monotonic()
                service_deadline = service_start + serve_duration_seconds
                while time.monotonic() < service_deadline:
                    if process.poll() is not None:
                        raise ConsumerError(
                            f"server exited during foreground service with status {process.returncode}")
                    if memory_guard and (memory_guard.error or memory_guard.over_limit):
                        raise ConsumerError(
                            f"memory guard stopped server: {memory_guard.error or memory_guard.over_limit}")
                    time.sleep(min(0.25, service_deadline - time.monotonic()))
                result["service"]["duration_observed_seconds"] = time.monotonic() - service_start
                result["service"]["status"] = "PASS"
            result["application_status"] = "PASS"
    except BaseException as exc:
        result["failure"] = f"{type(exc).__name__}: {exc}"
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
    finally:
        prior_term = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        try:
            if memory_guard is not None:
                memory_guard.finish()
                result["memory_guard"]["samples"] = memory_guard.samples
                result["memory_guard"]["telemetry_status"] = (
                    "FAIL" if memory_guard.error or memory_guard.over_limit else "PASS")
                result["memory_guard"]["error"] = memory_guard.error
                result["memory_guard"]["over_limit"] = memory_guard.over_limit
                memory_peaks = {uuid: 0 for uuid in selected_uuids}
                for sample in memory_guard.samples:
                    for uuid, used in sample["owned_used_memory_bytes_by_uuid"].items():
                        memory_peaks[uuid] = max(memory_peaks.get(uuid, 0), int(used))
                result["memory_guard"]["peak_owned_used_memory_bytes_by_uuid"] = memory_peaks
                if memory_guard.error or memory_guard.over_limit:
                    result["application_status"] = "FAIL"
                    result["failure"] = result["failure"] or str(memory_guard.error or memory_guard.over_limit)
            if process is not None:
                cleanup = cleanup_owned_group(
                    process, float(fixture["server"]["timeouts_seconds"]["shutdown"]))
                result["server_process"]["exit_code"] = process.returncode
                result["cleanup"] = cleanup
                if not cleanup["no_live_group_members"]:
                    result["application_status"] = "FAIL"
                    result["failure"] = result["failure"] or "owned process group did not terminate cleanly"
            offload = _layer_offload_evidence(log_path, workload_name, fixture)
            result["layer_offload_evidence"] = offload
            if workload_name == "resident_capacity" and offload["status"] != "PASS":
                result["application_status"] = "FAIL"
                result["failure"] = result["failure"] or (
                    f"resident-capacity run did not offload every model layer: {offload}")
            embedding = _input_embedding_evidence(log_path, workload_name)
            result["input_embedding_evidence"] = embedding
            if workload_name == "resident_capacity" and embedding["status"] != "PASS":
                result["application_status"] = "FAIL"
                result["failure"] = result["failure"] or (
                    f"resident-capacity input embedding was not confirmed in CUDA0 memory: {embedding}")
            if (workload_name == "resident_capacity" and memory_guard is not None
                    and any(value <= 0 for value in result["memory_guard"].get(
                        "peak_owned_used_memory_bytes_by_uuid", {}).values())):
                result["application_status"] = "FAIL"
                result["failure"] = result["failure"] or (
                    "resident-capacity process was not observed resident on both selected GPUs")
            if profile_log_path is not None:
                budget = int(fixture["memory_guard"]["per_physical_gpu_application_limit_mib"]) * 1024 * 1024
                minimum_resident = (int(fixture["model"]["tensor_layout"]["total_tensor_bytes"])
                                    if workload_name == "resident_capacity" else 0)
                profile_info = _read_profile_log(
                    profile_log_path, selected_uuids, budget,
                    str(fixture["controls"]["facade_owner"]["required_b01_evidence_id"]),
                    minimum_resident_bytes=minimum_resident)
                result["v200_profile_telemetry"] = profile_info
                graph_records = profile_info.get("graph_records", [])
                graph_capture_records = [record for record in graph_records
                                         if record.get("event") == "graph_capture_rewrite"
                                         and record.get("graph_status") == 0]
                graph_replay_records = [record for record in graph_records
                                        if record.get("event") == "graph_replay"
                                        and record.get("graph_status") == 0]
                result["graph_capture_evidence"] = {
                    "enabled": cuda_graphs_enabled,
                    "status": ("PASS_CAPTURE_AND_REPLAY"
                               if graph_capture_records and graph_replay_records
                               else "NOT_REQUESTED" if not cuda_graphs_enabled
                               else "MISSING_CAPTURE_OR_REPLAY"),
                    "successful_capture_count": len(graph_capture_records),
                    "successful_replay_samples": len(graph_replay_records),
                    "max_matched_nodes": max(
                        (int(record.get("graph_matched_nodes", 0))
                         for record in graph_capture_records), default=0),
                    "max_rewritten_nodes": max(
                        (int(record.get("graph_rewritten_nodes", 0))
                         for record in graph_capture_records), default=0),
                }
                if cuda_graphs_enabled and result["graph_capture_evidence"]["status"] != "PASS_CAPTURE_AND_REPLAY":
                    result["application_status"] = "FAIL"
                    result["failure"] = result["failure"] or (
                        "explicit CUDA graph mode did not record both capture and replay")
                result["allocation_evidence"]["v200_memory_api_receipt"] = profile_info["status"]
                if workload_name == "resident_capacity":
                    result["allocation_evidence"]["minimum_real_gpu_payload_bytes"] = minimum_resident
                    result["allocation_evidence"]["measured_peak_pair_backing_bytes"] = profile_info.get(
                        "peak_total_owner_used_bytes")
                if profile_info["status"] != "PASS":
                    result["application_status"] = "FAIL"
                    result["failure"] = result["failure"] or (
                        f"pair facade owner-ledger receipt is {profile_info['status']}")
            if workload_name == "resident_capacity":
                if control_name == "facade_owner" and result["application_status"] == "PASS":
                    result["gpu_qualification"] = "PASS_RESIDENT_PAIR_CAPACITY"
                    result["capacity_claim"] = (
                        "PASS: one-device llama-server loaded the full GGUF tensor payload onto the V200 pair; "
                        "the payload exceeds either physical V100 and the owner ledger recorded both GPUs")
                elif control_name == "native_pair" and result["application_status"] == "PASS":
                    result["gpu_qualification"] = "NATIVE_PAIR_BASELINE"
                    result["capacity_claim"] = "native two-device split baseline; not a V200 capacity claim"
            result["elapsed_seconds"] = time.monotonic() - monotonic_start
            result["finished_unix_time"] = time.time()
            _write_json_atomic(result_path, result)
        finally:
            signal.signal(signal.SIGTERM, prior_term)
    return result


def run_matrix(workload_name: str, controls: list[str], output_dir: Path,
               preloads: list[Path] | None, fixture_path: Path = FIXTURE_PATH,
               profile_path: Path = PROFILE_PATH,
               environ: dict[str, str] | None = None,
               monitor: bool = True,
               facade_mode: str | None = None,
               server_port: int | None = None,
               serve_duration_seconds: int = 0,
               cuda_graphs_enabled: bool = False) -> list[dict[str, Any]]:
    fixture, profile = load_documents(fixture_path, profile_path)
    if workload_name not in fixture["workloads"]:
        raise ConsumerError(f"unknown workload {workload_name!r}")
    for control in controls:
        if control not in fixture["workloads"][workload_name]["controls"]:
            raise ConsumerError(f"control {control!r} is not defined for {workload_name}")
        if control not in fixture["controls"]:
            raise ConsumerError(f"unknown control {control!r}")
        preflight = memory_preflight(workload_name, control, fixture, facade_mode)
        if not preflight["fits_static_preflight"]:
            raise ConsumerError(f"{control} fails static per-owner memory preflight")
    if len(set(controls)) != len(controls):
        raise ConsumerError("matched matrix controls must be unique")
    if serve_duration_seconds:
        if workload_name != "smoke" or controls != ["facade_owner"]:
            raise ConsumerError("finite service mode requires only the smoke facade control")
        if server_port is None:
            raise ConsumerError("finite service mode requires a launcher-selected loopback port")
        if not 1 <= serve_duration_seconds <= MAX_SERVE_DURATION_S:
            raise ConsumerError(f"serve duration must be in [1, {MAX_SERVE_DURATION_S}] seconds")
    env = dict(os.environ if environ is None else environ)
    uuid_map: dict[str, list[str]] = {}
    for control in controls:
        uuid_map[control] = validate_live_environment(control, fixture, env, facade_mode)
    if len({uuids[0] for uuids in uuid_map.values()}) > 1:
        raise ConsumerError("all controls in a matched run must share the first selected physical UUID")
    nvidia_smi_path: str | None = None
    visible_gpu_uuids: list[str] = []
    if monitor:
        all_selected = list(dict.fromkeys(uuid for selected in uuid_map.values() for uuid in selected))
        nvidia_smi_path, visible_gpu_uuids = _verify_nvidia_smi_uuid_selection(all_selected)
    if "facade_owner" in controls and not preloads:
        raise ConsumerError("matched facade control requires --preload-library")
    preloads = [path.resolve() for path in (preloads or [])]
    effective_facade_mode = facade_mode or fixture["controls"]["facade_owner"].get("v200_mode")
    if "facade_owner" in controls and effective_facade_mode == "pair" and len(preloads) < 2:
        raise ConsumerError("facade pair mode requires both frontdoor and root pair-initializer libraries")
    identities = verify_identities(profile, fixture, fixture_path, profile_path,
                                    preloads if "facade_owner" in controls else None)
    results = []
    for control in controls:
        physical_memory = (_nvidia_smi_memory(nvidia_smi_path, uuid_map[control])
                           if monitor and nvidia_smi_path is not None else None)
        result = _run_one(workload_name, control, fixture, profile, identities,
                          uuid_map[control], output_dir, preloads, monitor,
                          facade_mode, env, nvidia_smi_path=nvidia_smi_path,
                          visible_gpu_uuids=visible_gpu_uuids,
                          server_port=server_port,
                          serve_duration_seconds=serve_duration_seconds,
                          physical_memory_baseline=physical_memory,
                          cuda_graphs_enabled=cuda_graphs_enabled)
        results.append(result)
        if result["application_status"] != "PASS":
            break
    return results


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", choices=("smoke", "resident_capacity"), required=True)
    parser.add_argument("--control", action="append", dest="controls",
                        choices=("native_single", "native_pair", "facade_owner"), required=True,
                        help="repeat to run a matched sequential matrix")
    parser.add_argument("--preload-library", type=Path, action="append", dest="preloads",
                        help="repeat in order for frontdoor and root pair-initializer; one library remains supported for anchor mode")
    parser.add_argument("--facade-mode", choices=("anchor", "pair"), default=None,
                        help="override the fixture's facade mode; default is pair for integrated proof")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--server-port", type=int,
                        help="launcher-selected loopback port, used only by finite serve mode")
    parser.add_argument("--serve-duration-seconds", type=int, default=0,
                        help=f"keep the smoke facade available after its exact self-check (max {MAX_SERVE_DURATION_S}s)")
    parser.add_argument("--cuda-graphs", action="store_true",
                        help="opt in to llama CUDA graph capture/replay for matched controls")
    parser.add_argument("--execute", action="store_true",
                        help="required explicit confirmation for the live app process")
    parser.add_argument("--cpu-fake-no-memory-guard", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if not args.execute:
        print("Refusing to launch llama-server without --execute inside CUDA-controller admission.",
              file=sys.stderr)
        return 2
    if args.cpu_fake_no_memory_guard:
        print("The CPU-only guard bypass is available only to in-process unit tests.", file=sys.stderr)
        return 2
    try:
        if args.serve_duration_seconds < 0 or args.serve_duration_seconds > MAX_SERVE_DURATION_S:
            raise ConsumerError(f"serve duration must be in [0, {MAX_SERVE_DURATION_S}] seconds")
        prior_sigterm = signal.getsignal(signal.SIGTERM)

        def _handle_sigterm(_signum: int, _frame: Any) -> None:
            raise ConsumerError("foreground runner received SIGTERM; cleaning up the owned server")

        signal.signal(signal.SIGTERM, _handle_sigterm)
        results = run_matrix(args.workload, args.controls, args.output_dir,
                             args.preloads, facade_mode=args.facade_mode,
                             server_port=args.server_port,
                             serve_duration_seconds=args.serve_duration_seconds,
                             cuda_graphs_enabled=args.cuda_graphs)
    except ConsumerError as exc:
        print(f"consumer run refused: {exc}", file=sys.stderr)
        return 2
    finally:
        if "prior_sigterm" in locals():
            signal.signal(signal.SIGTERM, prior_sigterm)
    for result in results:
        print(json.dumps({"workload": result["workload"], "control": result["control"],
                          "application_status": result["application_status"],
                          "result_path": result["result_path"], "log_path": result["log_path"],
                          "gpu_qualification": result["gpu_qualification"]}, sort_keys=True))
    return 0 if results and all(result["application_status"] == "PASS" for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
