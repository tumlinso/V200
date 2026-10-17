#!/usr/bin/env python3
"""Plan or run a finite llama-server session under the admitted CUDA controller.

Default mode is a matched graph-off native-vs-facade correctness preview. Use
``--mode serve`` for a finite foreground loopback server (30 minutes by
default). Both modes are dry-run unless ``--execute`` is supplied.
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
import uuid
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = REPO_ROOT / "benchmarks/fixtures/llama-server-v200.json"
PROFILE_PATH = REPO_ROOT / "docs/v200/consumer-profile.json"
PAIR_PROFILE_PATH = REPO_ROOT / "evidence/foundation/pair-profile.json"
CONSUMER_RUNNER = REPO_ROOT / "tests/consumer/run_consumer.py"
CONSUMER_TESTS_DIR = CONSUMER_RUNNER.parent
if str(CONSUMER_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(CONSUMER_TESTS_DIR))
import run_consumer as consumer  # noqa: E402

BOOTSTRAP_RUNNER = REPO_ROOT / "planning/v200-bootstrap/tools/run_bounded.py"
BOUNDED_ADAPTER = Path(__file__).with_name("bounded_runner.py")
CONTROLLER_SHIM = REPO_ROOT / "build-support/run_controller.py"
CUDA_CONTROLLER = Path("/home/tumlinson/.agents/skills/cuda/scripts/cuda_controller.py")
PROJECT_CONTROL_PY = Path("/home/tumlinson/project-control/scripts/pc-dev")
FRONTDOOR_DSO = REPO_ROOT / "build/src/frontdoor/libv200_frontdoor.so"
PAIR_INIT_DSO = REPO_ROOT / "build/src/integration/libv200_pair_init.so"

APP_LIMIT_MIB = 12288
SERVE_DEFAULT_SECONDS = 1800
SERVE_MAX_SECONDS = 3000
SERVE_ENVELOPE_RESERVE_SECONDS = 280
SERVE_CONTROLLER_MARGIN_SECONDS = 20
PREVIEW_BOUNDED_SECONDS = 300
PREVIEW_CONTROLLER_MARGIN_SECONDS = 15
DEFAULT_SERVER_PORT = 8080
UUID_RE = re.compile(r"^GPU-[A-Za-z0-9-]+$")


class LaunchError(RuntimeError):
    """Invalid launch plan or unavailable runtime prerequisite."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise LaunchError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise LaunchError(f"expected a JSON object in {path}")
    return value


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _load_selected_pair() -> tuple[list[str], dict[str, Any]]:
    fixture = _read_json(FIXTURE_PATH)
    profile_bytes = PAIR_PROFILE_PATH.read_bytes()
    pair = json.loads(profile_bytes.decode("utf-8"))
    if not isinstance(pair, dict):
        raise LaunchError("pair profile must be a JSON object")
    expected_evidence = fixture["controls"]["facade_owner"]["required_b01_evidence_id"]
    if pair.get("status") != "GPU_PASS" or pair.get("evidence_id") != expected_evidence:
        raise LaunchError("pair profile is not the fixture-pinned B01 GPU_PASS receipt")
    uuids = pair.get("physical_uuids")
    if (not isinstance(uuids, list) or len(uuids) != 2
            or any(not isinstance(item, str) or UUID_RE.fullmatch(item) is None for item in uuids)
            or len(set(uuids)) != 2):
        raise LaunchError("B01 pair profile must list two distinct physical GPU UUIDs")
    return [str(item) for item in uuids], {
        "path": str(PAIR_PROFILE_PATH),
        "sha256": _sha256_bytes(profile_bytes),
        "evidence_id": expected_evidence,
    }


def _parse_gpu_option(value: str | None, authorized_pair: list[str]) -> list[str]:
    if not value:
        return list(authorized_pair)
    selected = [part.strip() for part in value.split(",") if part.strip()]
    if selected != authorized_pair:
        raise LaunchError("--gpus must match the exact ordered UUID pair in the accepted B01 profile")
    return selected


def _server_args(mode: str, duration_seconds: int | None,
                 port: int | None) -> tuple[list[str], str, int]:
    if mode == "serve":
        duration = SERVE_DEFAULT_SECONDS if duration_seconds is None else duration_seconds
        if isinstance(duration, bool) or not 1 <= duration <= SERVE_MAX_SECONDS:
            raise LaunchError(f"serve duration must be between 1 and {SERVE_MAX_SECONDS} seconds")
        selected_port = DEFAULT_SERVER_PORT if port is None else port
        if isinstance(selected_port, bool) or not 1 <= selected_port <= 65535:
            raise LaunchError("loopback port must be between 1 and 65535")
        return ["facade_owner"], "V200-SERVE", duration
    if mode == "preview":
        if duration_seconds is not None:
            raise LaunchError("--duration-seconds applies only to --mode serve")
        if port is not None:
            raise LaunchError("--port applies only to --mode serve")
        return ["native_single", "facade_owner"], "B03", 0
    raise LaunchError(f"unsupported mode {mode!r}")


def _make_plan(mode: str, uuids: list[str], run_id: str,
               duration_seconds: int | None = None,
               port: int | None = None,
               python_executable: str | None = None) -> dict[str, Any]:
    fixture = _read_json(FIXTURE_PATH)
    selected_port = DEFAULT_SERVER_PORT if mode == "serve" and port is None else port
    controls, bundle, duration = _server_args(mode, duration_seconds, selected_port)
    if len(uuids) != 2 or len(set(uuids)) != 2 or any(UUID_RE.fullmatch(item) is None for item in uuids):
        raise LaunchError("launch requires the exact two UUIDs from the B01 pair profile")
    python = python_executable or sys.executable
    fixture, _profile = consumer.load_documents(FIXTURE_PATH, PROFILE_PATH)
    profile = _read_json(PROFILE_PATH)
    workload = fixture["workloads"]["smoke"]
    preflights = [consumer.memory_preflight("smoke", control, fixture, "pair")
                  for control in controls]
    refused = [item for item in preflights if not item["fits_static_preflight"]]
    if refused:
        raise LaunchError("static per-owner smoke memory preflight refuses this launch")
    if mode == "serve":
        evidence_dir = REPO_ROOT / "evidence/V200-SERVE/runs" / run_id
        spec_path = REPO_ROOT / "evidence/V200-SERVE/launch-specs" / f"{run_id}.controller.json"
        bounded_seconds = duration + SERVE_ENVELOPE_RESERVE_SECONDS
        controller_timeout = bounded_seconds + SERVE_CONTROLLER_MARGIN_SECONDS
        wrapper = BOUNDED_ADAPTER
    else:
        evidence_dir = REPO_ROOT / "evidence/B03/preview-runs" / run_id
        spec_path = REPO_ROOT / "evidence/B03/preview-launch-specs" / f"{run_id}.controller.json"
        bounded_seconds = PREVIEW_BOUNDED_SECONDS
        controller_timeout = PREVIEW_BOUNDED_SECONDS + PREVIEW_CONTROLLER_MARGIN_SECONDS
        wrapper = BOOTSTRAP_RUNNER

    output_dir = evidence_dir / "consumer"
    inner = [python, str(CONSUMER_RUNNER), "--workload", "smoke"]
    for control in controls:
        inner.extend(["--control", control])
    for library in (FRONTDOOR_DSO, PAIR_INIT_DSO):
        inner.extend(["--preload-library", str(library)])
    inner.extend(["--facade-mode", "pair", "--output-dir", str(output_dir), "--execute"])
    if duration:
        inner.extend(["--serve-duration-seconds", str(duration), "--server-port", str(selected_port)])

    bounded_argv = [
        python, str(wrapper),
        "--project-root", str(REPO_ROOT),
        "--seconds", str(bounded_seconds),
        "--allocation-cap-mib", str(APP_LIMIT_MIB),
        "--evidence-dir", str(evidence_dir),
        "--bundle", bundle,
    ]
    for gpu_uuid in uuids:
        bounded_argv.extend(["--expected-uuid", gpu_uuid])
    bounded_argv.extend(["--", "/usr/bin/env", f"V200_DEVICE_UUIDS={','.join(uuids)}", *inner])

    paths = [
        "scripts/v200/serve.py",
        "scripts/v200/bounded_runner.py",
        "tests/consumer/run_consumer.py",
        "benchmarks/fixtures/llama-server-v200.json",
        "docs/v200/source-profile.json",
        "docs/v200/consumer-profile.json",
        "evidence/foundation/pair-profile.json",
        "planning/v200-bootstrap/tools/run_bounded.py",
    ]
    spec: dict[str, Any] = {
        "schema_version": 1,
        "project_root": str(REPO_ROOT),
        "command_cwd": str(REPO_ROOT),
        "recipe": "baseline",
        "resources": {
            "gpu_uuids": list(uuids),
            "gpus": 2,
            "cpu_threads": 4,
            "ram_bytes": 34359738368,
            "isolate_nvlink_domain": True,
            "isolate_pcie_root": False,
        },
        "argv": bounded_argv,
        "timeout": controller_timeout,
        "binary_paths": [
            str(FRONTDOOR_DSO.relative_to(REPO_ROOT)),
            str(PAIR_INIT_DSO.relative_to(REPO_ROOT)),
        ],
        "paths": paths,
    }
    return {
        "mode": mode,
        "bundle": bundle,
        "controls": controls,
        "run_id": run_id,
        "selected_gpu_uuids": list(uuids),
        "duration_seconds": duration if duration else None,
        "endpoint": f"http://127.0.0.1:{selected_port}" if duration else None,
        "graph_policy": fixture["server"].get("preview_policy"),
        "model": {"name": fixture["model"]["name"], "path": fixture["model"]["path"],
                  "sha256": fixture["model"]["sha256"]},
        "consumer_binary": {"path": profile["binary"]["path"],
                            "sha256": profile["binary"]["sha256"]},
        "smoke_config": {"context": workload["ctx_size"],
                          "gpu_layers": workload["n_gpu_layers"],
                          "request": workload["request"],
                          "oracle": workload["oracle"]},
        "preload_libraries": [str(FRONTDOOR_DSO), str(PAIR_INIT_DSO)],
        "static_memory_preflight": preflights,
        "app_allocation_cap_mib_per_gpu": APP_LIMIT_MIB,
        "bounded_seconds": bounded_seconds,
        "controller_timeout_seconds": controller_timeout,
        "controller_argv": [
            str(PROJECT_CONTROL_PY), "python", str(CONTROLLER_SHIM),
            str(CUDA_CONTROLLER), "run", "--spec", str(spec_path), "--json",
        ],
        "controller_spec_path": str(spec_path),
        "controller_spec": spec,
        "bounded_child_argv": bounded_argv,
        "required_files": [
            str(PROJECT_CONTROL_PY), str(CONTROLLER_SHIM), str(CUDA_CONTROLLER),
            str(wrapper), str(CONSUMER_RUNNER), str(FIXTURE_PATH), str(PROFILE_PATH),
            str(PAIR_PROFILE_PATH), str(fixture["model"]["path"]),
            str(REPO_ROOT / "docs/v200/source-profile.json"),
            str(FRONTDOOR_DSO), str(PAIR_INIT_DSO),
        ],
    }


def _missing_runtime_files(plan: dict[str, Any]) -> list[str]:
    missing = []
    for path in plan["required_files"]:
        if not Path(path).is_file():
            missing.append(path)
    return missing


def _write_spec(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as target:
            json.dump(value, target, indent=2, sort_keys=True)
            target.write("\n")
            target.flush()
            os.fsync(target.fileno())
    except FileExistsError as exc:
        raise LaunchError(f"refusing to overwrite controller spec {path}") from exc


def _proc_group_session(pid: int) -> tuple[int, int] | None:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        tail = stat[stat.rfind(")") + 2:].split()
        return int(tail[2]), int(tail[3])
    except (OSError, ValueError, IndexError):
        return None


def _bounded_child_in_session(session_id: int) -> int | None:
    wrappers = {str(BOOTSTRAP_RUNNER.resolve()), str(BOUNDED_ADAPTER.resolve())}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if _proc_group_session(pid) != (session_id, session_id):
            continue
        try:
            argv = [part.decode(errors="replace") for part in (entry / "cmdline").read_bytes().split(b"\0") if part]
        except OSError:
            continue
        if any(item in wrappers for item in argv):
            return pid
    return None


def _stop_foreground_child(controller: subprocess.Popen[Any]) -> int:
    """Stop only the controller's bounded runner so its lease finalizer survives."""
    signaled: set[int] = set()
    admission_cancelled = False
    deadline = time.monotonic() + 45.0
    while time.monotonic() < deadline:
        wrapper_pid = _bounded_child_in_session(controller.pid)
        if wrapper_pid is not None and wrapper_pid not in signaled:
            try:
                os.kill(wrapper_pid, signal.SIGTERM)
                signaled.add(wrapper_pid)
            except ProcessLookupError:
                pass
        elif (wrapper_pid is None and controller.poll() is None and not signaled
              and not admission_cancelled):
            # Cancellation during admission, before the bounded child exists.
            controller.send_signal(signal.SIGINT)
            admission_cancelled = True
        returncode = controller.poll()
        if returncode is not None and wrapper_pid is None:
            return returncode
        time.sleep(0.1)
    # Preserve unrelated processes. The wrapper remains the only cancellation
    # target; its own adapter escalates against the consumer's owned process group.
    wrapper_pid = _bounded_child_in_session(controller.pid)
    if wrapper_pid is not None:
        try:
            os.kill(wrapper_pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    if controller.poll() is not None:
        return int(controller.returncode)
    controller.send_signal(signal.SIGINT)
    try:
        return controller.wait(timeout=10)
    except subprocess.TimeoutExpired:
        return 130


def _print_plan(plan: dict[str, Any], missing: list[str], execute: bool) -> None:
    print(f"V200 llama-server mode: {plan['mode']} ({plan['bundle']})")
    print(f"Selected ordered physical UUIDs: {','.join(plan['selected_gpu_uuids'])}")
    print(f"Graph policy: {plan['graph_policy']}")
    print(f"Pinned model: {json.dumps(plan['model'], sort_keys=True)}")
    print(f"Pinned consumer: {json.dumps(plan['consumer_binary'], sort_keys=True)}")
    print(f"Smoke configuration: {json.dumps(plan['smoke_config'], sort_keys=True)}")
    print(f"Pair facade preloads: {json.dumps(plan['preload_libraries'])}")
    print(f"Static smoke memory preflight: {json.dumps(plan['static_memory_preflight'], sort_keys=True)}")
    print(f"Per-GPU application allocation budget: {plan['app_allocation_cap_mib_per_gpu']} MiB")
    if "pair_profile_identity" in plan:
        print("B01 pair receipt: " + json.dumps(plan["pair_profile_identity"], sort_keys=True))
    if plan["endpoint"]:
        print(f"Loopback endpoint (announced before controller startup): {plan['endpoint']}")
        print(f"Service duration after the exact startup self-check: {plan['duration_seconds']} seconds")
        print("The command stays in the foreground; query /health until ready and stop it with Ctrl-C.")
    else:
        print("Preview runs native_single then facade_owner sequentially with the same smoke workload.")
    print(f"Controller hard timeout: {plan['controller_timeout_seconds']} seconds")
    print(f"Controller spec: {plan['controller_spec_path']}")
    print("Controller spec preview: " + json.dumps(plan["controller_spec"], sort_keys=True))
    print("Controller command: " + json.dumps(plan["controller_argv"]))
    if missing:
        print("Missing runtime files:")
        for path in missing:
            print(f"  - {path}")
    if not execute:
        print("Dry run only. Add --execute after the V200 build outputs are present.")


def _run_controller(plan: dict[str, Any]) -> int:
    spec_path = Path(plan["controller_spec_path"])
    _write_spec(spec_path, plan["controller_spec"])
    command = plan["controller_argv"]
    print("Admitting through the configured CUDA controller; no direct llama-server launch follows.", flush=True)
    print("Executing: " + json.dumps(command), flush=True)
    previous = signal.getsignal(signal.SIGTERM)
    process: subprocess.Popen[Any] | None = None

    def _interrupt(_signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _interrupt)
    try:
        process = subprocess.Popen(command, cwd=str(REPO_ROOT), start_new_session=True)
        return process.wait()
    except KeyboardInterrupt:
        if process is None:
            return 130
        return _stop_foreground_child(process)
    finally:
        signal.signal(signal.SIGTERM, previous)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("serve", "preview"), default="serve",
                        help="serve is the finite foreground facade server; preview runs matched one-shot controls")
    parser.add_argument("--gpus", help="optional exact comma-separated B01 UUID pair; defaults to pair-profile.json")
    parser.add_argument("--duration-seconds", type=int, default=None,
                        help=f"serve lifetime after correctness check (default {SERVE_DEFAULT_SECONDS}, max {SERVE_MAX_SECONDS})")
    parser.add_argument("--port", type=int, default=None,
                        help=f"loopback server port in serve mode (default {DEFAULT_SERVER_PORT})")
    parser.add_argument("--execute", action="store_true",
                        help="explicitly enter the CUDA controller; without this flag, print a dry-run plan")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        authorized_pair, pair_identity = _load_selected_pair()
        selected = _parse_gpu_option(args.gpus, authorized_pair)
        run_id = f"{args.mode}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{uuid.uuid4().hex[:12]}"
        plan = _make_plan(args.mode, selected, run_id,
                          duration_seconds=args.duration_seconds, port=args.port)
        plan["pair_profile_identity"] = pair_identity
    except (LaunchError, OSError, KeyError, TypeError, ValueError) as exc:
        print(f"V200 serve refused: {exc}", file=sys.stderr)
        return 2

    missing = _missing_runtime_files(plan)
    _print_plan(plan, missing, args.execute)
    if not args.execute:
        return 0
    if missing:
        print("Execution refused until every pinned runtime file exists.", file=sys.stderr)
        return 2
    if plan["endpoint"]:
        print("Start a separate client with: curl " + plan["endpoint"] + "/health", flush=True)
    try:
        return _run_controller(plan)
    except OSError as exc:
        print(f"cannot start CUDA controller command: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
