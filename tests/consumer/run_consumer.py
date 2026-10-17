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


REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = REPO_ROOT / "benchmarks/fixtures/llama-server-v200.json"
PROFILE_PATH = REPO_ROOT / "docs/v200/consumer-profile.json"
PROFILE_SCHEMA = "v200-consumer-profile/v1"
FIXTURE_SCHEMA = "v200-llama-server-fixture/v1"
STARTUP_TIMEOUT_S = 600.0
REQUEST_TIMEOUT_S = 120.0
SHUTDOWN_TIMEOUT_S = 15.0
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
    """Conservative descriptor-based estimate; not a substitute for the cap."""
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
        # Assume all non-block tensors may land on one owner; this intentionally
        # overestimates the balanced pair case rather than relying on placement.
        non_block = int(layout["non_block_tensor_bytes"])
        per_owner_layers = (layer_tensor_bytes + owners - 1) // owners
        per_owner_kv = (kv_bytes + owners - 1) // owners
        workspace = int(guard["workspace_reserve_bytes_per_gpu"])
        estimate = per_owner_layers + non_block + per_owner_kv + workspace
        limit_bytes = int(guard["per_physical_gpu_application_limit_mib"]) * 1024 * 1024
        result = {
            "workload": workload_name,
            "control": control_name,
            "effective_gpu_layers": layers,
            "requested_gpu_layers": int(workload["n_gpu_layers"]),
            "owner_count_assumed": owner_count,
            "gpu_layer_tensor_bytes_total": layer_tensor_bytes,
            "non_block_tensor_bytes_assumed_per_owner": non_block,
            "kv_bytes_total": kv_bytes,
            "kv_bytes_assumed_per_owner": per_owner_kv,
            "workspace_reserve_bytes_per_owner": workspace,
            "estimated_bytes_per_owner": estimate,
            "estimated_mib_per_owner": round(estimate / (1024 * 1024), 3),
            "application_limit_bytes_per_owner": limit_bytes,
            "fits_static_preflight": estimate <= limit_bytes,
            "basis": "GGUF tensor descriptors + explicit f16 KV formula + 1 GiB workspace reserve; conservative non-block assignment",
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
                       profile_log_path: Path | None = None) -> dict[str, str]:
    source = dict(os.environ if environ is None else environ)
    child = {key: value for key, value in source.items()
             if not key.startswith("LLAMA_ARG_") and not key.startswith("GGML_CUDA_")}
    inherited = [part for part in child.pop("LD_PRELOAD", "").split()
                 if "libv200" not in Path(part).name.lower()]
    child.pop("V200_MODE", None)
    child.pop("V200_DEVICE_UUIDS", None)
    child.pop("V200_PROFILE_LOG", None)
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
        child["V200_MODE"] = mode
        child["V200_DEVICE_UUIDS"] = selected
        child["LD_PRELOAD"] = ":".join([*(str(path) for path in preloads), *inherited])
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
    identities: dict[str, Any] = {
        "profile_sha256": sha256_file(profile_path),
        "fixture_sha256": sha256_file(fixture_path),
        "binary": _verify_hash_entry(binary),
        "runtime_closure": [_verify_hash_entry(item) for item in profile["runtime_closure"]],
        "model": {"path": fixture["model"]["path"],
                  "sha256": _verify_hash(fixture["model"]["path"], fixture["model"]["sha256"])},
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
              binary_path: str, port: int) -> list[str]:
    control = fixture["controls"][control_name]
    argv = [binary_path]
    replacements = {"{model}": fixture["model"]["path"],
                    "{port}": str(port),
                    "{ctx_size}": str(int(workload["ctx_size"])),
                    "{n_gpu_layers}": str(int(workload["n_gpu_layers"]))}
    for value in fixture["server"]["fixed_arguments"]:
        argv.append(replacements.get(value, value))
    argv.extend(str(arg) for arg in control["arguments"])
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
                      expected_budget_bytes: int) -> dict[str, Any]:
    if not path.exists():
        return {"status": "MISSING", "path": str(path), "records": []}
    try:
        raw = path.read_bytes()
        records = []
        for line_number, line in enumerate(raw.splitlines(), start=1):
            if not line.strip():
                continue
            value = json.loads(line.decode("utf-8"))
            if not isinstance(value, dict):
                raise ValueError(f"line {line_number} is not a JSON object")
            records.append(value)
            required = ("schema", "event", "mode", "anchor_uuid", "peer_uuid",
                        "qualification_id", "owner_budget_bytes", "owner_used_bytes",
                        "live_allocation_count", "live_reservation_count", "cleanup_pending_count")
            missing = [field for field in required if field not in value]
            if missing:
                raise ValueError(f"line {line_number} missing fields {missing}")
            if value["schema"] != "v200-frontdoor-ledger/v1" or value["mode"] != "pair":
                raise ValueError(f"line {line_number} has unexpected ledger schema or mode")
            if value["anchor_uuid"] != selected_uuids[0] or value["peer_uuid"] != selected_uuids[1]:
                raise ValueError(f"line {line_number} UUID ordering differs from controller selection")
            budgets = value["owner_budget_bytes"]
            usage = value["owner_used_bytes"]
            if (not isinstance(budgets, list) or len(budgets) != 2
                    or any(isinstance(item, bool) or not isinstance(item, int) for item in budgets)):
                raise ValueError(f"line {line_number} owner budgets are malformed")
            if budgets != [expected_budget_bytes, expected_budget_bytes]:
                raise ValueError(f"line {line_number} owner budgets differ from fixture cap")
            if (not isinstance(usage, list) or len(usage) != 2
                    or any(isinstance(item, bool) or not isinstance(item, int) for item in usage)):
                raise ValueError(f"line {line_number} owner usage is malformed")
            if any(item < 0 for item in usage):
                raise ValueError(f"line {line_number} owner usage is negative")
            if any(item > budget for item, budget in zip(usage, budgets)):
                raise OverflowError(f"line {line_number} owner usage exceeded configured budget")
            for field in ("live_allocation_count", "live_reservation_count", "cleanup_pending_count"):
                if isinstance(value[field], bool) or not isinstance(value[field], int) or value[field] < 0:
                    raise ValueError(f"line {line_number} {field} is malformed")
        return {"status": "PASS" if records else "EMPTY", "path": str(path),
                "sha256": hashlib.sha256(raw).hexdigest(), "records": records}
    except OverflowError as exc:
        return {"status": "OVER_BUDGET", "path": str(path),
                "sha256": hashlib.sha256(raw).hexdigest() if "raw" in locals() else None,
                "records": records if "records" in locals() else [], "error": str(exc)}
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        return {"status": "INVALID", "path": str(path), "records": [],
                "error": f"{type(exc).__name__}: {exc}"}


def _run_one(workload_name: str, control_name: str, fixture: dict[str, Any],
             profile: dict[str, Any], identities: dict[str, Any],
             selected_uuids: list[str], output_dir: Path,
             preloads: list[Path] | None, monitor: bool = True,
             facade_mode: str | None = None,
             environ: dict[str, str] | None = None,
             popen_factory: Callable[..., Any] = subprocess.Popen,
             nvidia_smi_path: str | None = None,
             visible_gpu_uuids: list[str] | None = None) -> dict[str, Any]:
    workload = fixture["workloads"][workload_name]
    preflight = memory_preflight(workload_name, control_name, fixture, facade_mode)
    if not preflight["fits_static_preflight"]:
        raise ConsumerError(
            f"static preflight estimates {preflight['estimated_mib_per_owner']} MiB/owner, "
            f"over {preflight['application_limit_bytes_per_owner'] / (1024 * 1024):.0f} MiB"
        )
    port = _free_loopback_port()
    argv = _argv_for(control_name, workload, fixture, profile["binary"]["path"], port)
    started = time.time()
    monotonic_start = time.monotonic()
    run_id = f"{workload_name}-{control_name}-{time.time_ns()}"
    log_path = output_dir / f"{run_id}.log"
    result_path = output_dir / f"{run_id}.json"
    resolved_facade_mode = facade_mode or fixture["controls"]["facade_owner"].get("v200_mode")
    profile_log_path = (output_dir / f"{run_id}.v200-profile.jsonl"
                        if control_name == "facade_owner" and resolved_facade_mode == "pair" else None)
    child_env = _clean_environment(control_name, fixture, preloads, environ,
                                   facade_mode, selected_uuids, profile_log_path)
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
                           else "candidate-only-until-controller-and-allocator-receipts"),
        "fixture_id": fixture["fixture_id"],
        "identities": identities,
        "selected_gpu_uuids": selected_uuids,
        "cuda_visible_devices": child_env.get("CUDA_VISIBLE_DEVICES"),
        "v200_mode": child_env.get("V200_MODE"),
        "preflight": preflight,
        "allocation_evidence": {
            "application_limit_bytes_per_owner": preflight["application_limit_bytes_per_owner"],
            "facade_limit_enforcer": "V200 per-owner allocation ledger" if control_name == "facade_owner" else None,
            "native_limit_enforcement": "descriptor-based preflight only; no native allocator ledger",
            "v200_memory_api_receipt": "not exported by this consumer process launch contract",
            "fixture_expected_v200_owner_budgets": ([
                {"gpu_uuid": uuid,
                 "budget_bytes": int(fixture["memory_guard"]["per_physical_gpu_application_limit_mib"]) * 1024 * 1024}
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
        "application_status": "FAIL",
        "gpu_qualification": "NOT_CLAIMED",
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
            process = popen_factory(argv, cwd=str(profile["source"]["repository"]), env=child_env,
                                    stdout=log, stderr=subprocess.STDOUT,
                                    start_new_session=True)
            result["server_process"] = {"pid": process.pid, "process_group": process.pid,
                                        "session": process.pid}
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
            result["application_status"] = "PASS"
    except BaseException as exc:
        result["failure"] = f"{type(exc).__name__}: {exc}"
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
    finally:
        if memory_guard is not None:
            memory_guard.finish()
            result["memory_guard"]["samples"] = memory_guard.samples
            result["memory_guard"]["telemetry_status"] = (
                "FAIL" if memory_guard.error or memory_guard.over_limit else "PASS")
            result["memory_guard"]["error"] = memory_guard.error
            result["memory_guard"]["over_limit"] = memory_guard.over_limit
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
        if profile_log_path is not None:
            budget = int(fixture["memory_guard"]["per_physical_gpu_application_limit_mib"]) * 1024 * 1024
            profile_info = _read_profile_log(profile_log_path, selected_uuids, budget)
            result["v200_profile_telemetry"] = profile_info
            result["allocation_evidence"]["v200_memory_api_receipt"] = profile_info["status"]
            if profile_info["status"] != "PASS":
                result["application_status"] = "FAIL"
                result["failure"] = result["failure"] or (
                    f"pair facade owner-ledger receipt is {profile_info['status']}")
        result["elapsed_seconds"] = time.monotonic() - monotonic_start
        result["finished_unix_time"] = time.time()
        _write_json_atomic(result_path, result)
    return result


def run_matrix(workload_name: str, controls: list[str], output_dir: Path,
               preloads: list[Path] | None, fixture_path: Path = FIXTURE_PATH,
               profile_path: Path = PROFILE_PATH,
               environ: dict[str, str] | None = None,
               monitor: bool = True,
               facade_mode: str | None = None) -> list[dict[str, Any]]:
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
        result = _run_one(workload_name, control, fixture, profile, identities,
                          uuid_map[control], output_dir, preloads, monitor,
                          facade_mode, env, nvidia_smi_path=nvidia_smi_path,
                          visible_gpu_uuids=visible_gpu_uuids)
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
        results = run_matrix(args.workload, args.controls, args.output_dir,
                             args.preloads, facade_mode=args.facade_mode)
    except ConsumerError as exc:
        print(f"consumer run refused: {exc}", file=sys.stderr)
        return 2
    for result in results:
        print(json.dumps({"workload": result["workload"], "control": result["control"],
                          "application_status": result["application_status"],
                          "result_path": result["result_path"], "log_path": result["log_path"],
                          "gpu_qualification": result["gpu_qualification"]}, sort_keys=True))
    return 0 if results and all(result["application_status"] == "PASS" for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
