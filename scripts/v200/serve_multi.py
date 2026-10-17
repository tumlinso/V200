#!/usr/bin/env python3
"""Run the multipair correctness bundle or a finite two-device V200 service.

The default is the B06 correctness bundle. ``--mode serve`` runs one
full-resident two-logical-device llama-server in the foreground for a finite
duration. Without ``--execute`` this prints a dry-run plan. Execution builds
the CUDA 12.9 candidate and CPU gates first, then asks the CUDA controller to
admit the workload.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = REPO_ROOT / "benchmarks/fixtures/llama-server-multipair.json"
PROFILE_PATH = REPO_ROOT / "docs/v200/multipair-profile.json"
RUNNER = REPO_ROOT / "tests/multi_pair/run_multi_consumer.py"
BOUNDED_ADAPTER = REPO_ROOT / "scripts/v200/bounded_runner.py"
COPY_PROBE = REPO_ROOT / "build-multipair/tests/multi_pair/v200_multi_pair_copy_probe"
BUILD_DIR = REPO_ROOT / "build-multipair"
FRONTDOOR_DSO = BUILD_DIR / "src/frontdoor/libv200_frontdoor.so"
PAIR_INIT_DSO = BUILD_DIR / "src/integration/libv200_pair_init.so"
OPERATORS_DSO = BUILD_DIR / "src/operators/libv200_operators.so"
CUDA_TOOLKIT_ROOT = "/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/cuda/12.9"
PROJECT_CONTROL_PY = Path("/home/tumlinson/project-control/scripts/pc-dev")
CONTROLLER_SHIM = REPO_ROOT / "build-support/run_controller.py"
CUDA_CONTROLLER = Path("/home/tumlinson/.agents/skills/cuda/scripts/cuda_controller.py")
BOUNDED_SECONDS = 300
CONTROLLER_MARGIN_SECONDS = 15
SERVE_DEFAULT_SECONDS = 1800
SERVE_MAX_SECONDS = 3000
SERVE_ENVELOPE_RESERVE_SECONDS = 280
SERVE_CONTROLLER_MARGIN_SECONDS = 20
DEFAULT_SERVER_PORT = 8080
OWNER_CAP_MIB = 12288

if str(REPO_ROOT / "tests/multi_pair") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "tests/multi_pair"))
import run_multi_consumer as runner  # noqa: E402
import serve as single_pair_serve  # noqa: E402


class LaunchError(RuntimeError):
    """Invalid plan, build failure, or unavailable controller prerequisite."""


def _make_plan(run_id: str, python: str | None = None, *, mode: str = "correctness",
               duration_seconds: int | None = None,
               port: int | None = None) -> dict[str, Any]:
    if mode not in ("correctness", "serve"):
        raise LaunchError(f"unsupported multi-pair mode {mode!r}")
    if mode == "serve":
        duration_seconds = SERVE_DEFAULT_SECONDS if duration_seconds is None else duration_seconds
        port = DEFAULT_SERVER_PORT if port is None else port
        if isinstance(duration_seconds, bool) or not 1 <= duration_seconds <= SERVE_MAX_SECONDS:
            raise LaunchError(f"service duration must be between 1 and {SERVE_MAX_SECONDS} seconds")
        if isinstance(port, bool) or not 1 <= port <= 65535:
            raise LaunchError("loopback port must be between 1 and 65535")
        bounded_seconds = duration_seconds + SERVE_ENVELOPE_RESERVE_SECONDS
        controller_margin = SERVE_CONTROLLER_MARGIN_SECONDS
        bundle_name = "V200-MULTI-SERVE"
        evidence_dir = REPO_ROOT / "evidence/B06/serving/runs" / run_id
        spec_path = REPO_ROOT / "evidence/B06/serving/launch-specs" / f"{run_id}.controller.json"
    else:
        if duration_seconds is not None or port is not None:
            raise LaunchError("--duration-seconds and --port apply only to --mode serve")
        bounded_seconds = BOUNDED_SECONDS
        controller_margin = CONTROLLER_MARGIN_SECONDS
        bundle_name = "B06"
        evidence_dir = REPO_ROOT / "evidence/B06/multipair-runs" / run_id
        spec_path = REPO_ROOT / "evidence/B06/multipair-launch-specs" / f"{run_id}.controller.json"
    try:
        fixture, profile = runner.load_inputs(FIXTURE_PATH, PROFILE_PATH)
        static = runner.static_preflight(fixture)
    except (runner.MultiPairError, OSError, KeyError, TypeError, ValueError) as exc:
        raise LaunchError(f"multi-pair fixture/profile refused: {exc}") from exc
    uuids = list(fixture["topology"]["physical_uuid_order"])
    evidence_ids = [pair["b01_evidence_id"] for pair in fixture["topology"]["logical_pairs"]]
    python = python or sys.executable
    consumer_dir = evidence_dir / "consumer"
    preload = [FRONTDOOR_DSO, PAIR_INIT_DSO, OPERATORS_DSO]
    inner = [
        python, str(RUNNER),
        "--fixture", str(FIXTURE_PATH),
        "--profile", str(PROFILE_PATH),
        "--output-dir", str(consumer_dir),
    ]
    if mode == "correctness":
        inner.extend(["--copy-probe", str(COPY_PROBE)])
    for library in preload:
        inner.extend(["--preload-library", str(library)])
    if mode == "serve":
        inner.extend(["--service", "--duration-seconds", str(duration_seconds),
                      "--port", str(port)])
    inner.append("--execute")
    child_env = [
        "-u", "V200_MODE",
        "-u", "V200_DEVICE_UUIDS",
        "-u", "V200_B01_EVIDENCE_ID",
        "-u", "V200_B01_EVIDENCE_IDS",
        "-u", "V200_OWNER0_BUDGET_MIB",
        "-u", "V200_OWNER1_BUDGET_MIB",
        "-u", "V200_PAIR_COMPUTE",
        "-u", "V200_PROFILE_LOG",
        f"V200_DEVICE_UUIDS={','.join(uuids)}",
        f"V200_B01_EVIDENCE_IDS={','.join(evidence_ids)}",
        "V200_OWNER0_BUDGET_MIB=4096",
        "V200_OWNER1_BUDGET_MIB=12288",
    ]
    bounded_argv = [
        python, str(BOUNDED_ADAPTER),
        "--project-root", str(REPO_ROOT),
        "--seconds", str(bounded_seconds),
        "--allocation-cap-mib", str(OWNER_CAP_MIB),
        "--evidence-dir", str(evidence_dir),
        "--bundle", bundle_name,
    ]
    for value in uuids:
        bounded_argv.extend(["--expected-uuid", value])
    bounded_argv.extend(["--", "/usr/bin/env", *child_env, *inner])

    paths = [
        "scripts/v200/serve_multi.py",
        "scripts/v200/bounded_runner.py",
        "tests/multi_pair/run_multi_consumer.py",
        "tests/multi_pair/copy_probe.cpp",
        "tests/multi_pair/copy_contract.cpp",
        "tests/multi_pair/test_multi_pair.py",
        "docs/v200/two-device.md",
        "tests/multi_pair/CMakeLists.txt",
        "benchmarks/fixtures/llama-server-multipair.json",
        "docs/v200/multipair-profile.json",
        "docs/v200/consumer-profile.json",
        "src/frontdoor/cuda_wrappers.cpp",
        "src/frontdoor/runtime_state.cpp",
        "src/integration/runtime_init.cpp",
        "src/memory/manager.cpp",
        "src/multi_pair/copy.cpp",
        "src/operators/launch.cpp",
        "src/operators/pair_runtime.cpp",
        "include/v200/multi_pair/copy.hpp",
        "include/v200/core/runtime_init.hpp",
        "include/v200/memory/api.h",
        "planning/v200-bootstrap/tools/run_bounded.py",
        "CMakeLists.txt",
    ]
    binary_paths = [
        str(FRONTDOOR_DSO.relative_to(REPO_ROOT)),
        str(PAIR_INIT_DSO.relative_to(REPO_ROOT)),
        str(OPERATORS_DSO.relative_to(REPO_ROOT)),
    ]
    if mode == "correctness":
        binary_paths.append(str(COPY_PROBE.relative_to(REPO_ROOT)))
    spec = {
        "schema_version": 1,
        "project_root": str(REPO_ROOT),
        "command_cwd": str(REPO_ROOT),
        "recipe": "baseline",
        "toolchain": {"root": CUDA_TOOLKIT_ROOT, "require_sanitizer": False},
        "resources": {
            "gpu_uuids": uuids,
            "gpus": 4,
            "cpu_threads": 4,
            "ram_bytes": 34359738368,
            "isolate_nvlink_domain": True,
            "isolate_pcie_root": False,
        },
        "argv": bounded_argv,
        "timeout": bounded_seconds + controller_margin,
        "binary_paths": binary_paths,
        "paths": paths,
    }
    controller_argv = [
        str(PROJECT_CONTROL_PY), "python", str(CONTROLLER_SHIM),
        str(CUDA_CONTROLLER), "run", "--spec", str(spec_path), "--json",
    ]
    required_files = [
        PROJECT_CONTROL_PY, CONTROLLER_SHIM, CUDA_CONTROLLER,
        BOUNDED_ADAPTER, RUNNER, FIXTURE_PATH, PROFILE_PATH,
        FRONTDOOR_DSO, PAIR_INIT_DSO, OPERATORS_DSO,
        Path(profile["binary"]["path"]), Path(fixture["model"]["path"]),
        Path(profile["source"]["repository"]),
    ]
    if mode == "correctness":
        required_files.append(COPY_PROBE)
    return {
        "run_id": run_id,
        "mode": mode,
        "selected_gpu_uuids": uuids,
        "pair_evidence_ids": evidence_ids,
        "workload": "two_v200_resident",
        "logical_device_arguments": fixture["controls"]["v200_two"]["arguments"],
        "native_control_arguments": fixture["controls"]["native_four"]["arguments"],
        "model": {"path": fixture["model"]["path"], "sha256": fixture["model"]["sha256"],
                  "tensor_bytes": fixture["model"]["tensor_layout"]["total_tensor_bytes"]},
        "static_preflight": static,
        "owner_budgets_mib_per_pair": [4096, 12288],
        "owner_allocation_cap_mib": OWNER_CAP_MIB,
        "copy_probe": str(COPY_PROBE) if mode == "correctness" else None,
        "build_directory": str(BUILD_DIR),
        "cuda_toolkit": CUDA_TOOLKIT_ROOT,
        "bounded_seconds": bounded_seconds,
        "controller_timeout_seconds": bounded_seconds + controller_margin,
        "duration_seconds": duration_seconds,
        "port": port,
        "endpoint": f"http://127.0.0.1:{port}" if mode == "serve" else None,
        "output_directory": str(consumer_dir),
        "controller_spec_path": str(spec_path),
        "controller_spec": spec,
        "controller_argv": controller_argv,
        "bounded_child_argv": bounded_argv,
        "required_files": [str(path) for path in required_files],
    }


def _build_candidate() -> None:
    configure = [
        "cmake", "-S", str(REPO_ROOT), "-B", str(BUILD_DIR),
        f"-DCUDAToolkit_ROOT={CUDA_TOOLKIT_ROOT}",
        f"-DV200_CUDA_TOOLKIT_ROOT={CUDA_TOOLKIT_ROOT}",
        "-DCMAKE_CXX_COMPILER=/usr/bin/g++",
        "-DBUILD_TESTING=ON",
        "-DV200_BUILD_BOOTSTRAP_EXAMPLES=OFF",
    ]
    build = ["cmake", "--build", str(BUILD_DIR), "--parallel", "2", "--target",
             "v200_frontdoor", "v200_pair_init", "v200_operators",
             "v200_multi_pair_copy_probe", "v200_multi_pair_copy_contract"]
    ctest = ["ctest", "--test-dir", str(BUILD_DIR), "--output-on-failure",
             "-R", "^v200_multi_pair_(copy_contract|host)$"]
    for command in (configure, build, ctest):
        print("Build step: " + json.dumps(command), flush=True)
        try:
            subprocess.run(command, cwd=REPO_ROOT, check=True)
        except (OSError, subprocess.CalledProcessError) as exc:
            raise LaunchError(f"pre-admission CUDA 12.9 build/CPU gate failed: {exc}") from exc


def _missing_files(plan: dict[str, Any]) -> list[str]:
    return [path for path in plan["required_files"] if not Path(path).exists()]


def _write_plan_line(plan: dict[str, Any], missing: list[str], execute: bool) -> None:
    if plan["mode"] == "serve":
        print("V200 two-device finite foreground llama-server service")
    else:
        print("V200 multi-pair finite workload: native four-GPU control, then two logical V200 devices")
    print("Ordered physical UUIDs: " + ",".join(plan["selected_gpu_uuids"]))
    print("Pair qualification receipts: " + ",".join(plan["pair_evidence_ids"]))
    device_arguments = {"v200_two": plan["logical_device_arguments"]}
    if plan["mode"] == "correctness":
        device_arguments["native_four"] = plan["native_control_arguments"]
    print("llama device arguments: " + json.dumps(device_arguments, sort_keys=True))
    print("Pinned model: " + json.dumps(plan["model"], sort_keys=True))
    print("Owner budgets per V200 pair: [4096, 12288] MiB; maximum owner cap: 12288 MiB")
    print("Static descriptor preflight: " + json.dumps(plan["static_preflight"], sort_keys=True))
    print("Build directory/toolkit: " + plan["build_directory"] + " / " + plan["cuda_toolkit"])
    if plan["mode"] == "serve":
        print(f"Loopback endpoint: {plan['endpoint']}")
        print(f"Finite service duration after startup correctness check: {plan['duration_seconds']} seconds")
        print("The command stays in the foreground and cleans up the model/server when it ends.")
    print(f"Finite controller limit: {plan['bounded_seconds']} seconds, 4 GPUs, 4 CPU threads")
    print("Controller spec: " + plan["controller_spec_path"])
    print("Controller spec preview: " + json.dumps(plan["controller_spec"], sort_keys=True))
    if missing:
        print("Missing runtime files:")
        for path in missing:
            print("  - " + path)
    if not execute:
        print("Dry run only. --execute builds CPU/runtime targets first, then asks the CUDA controller for admission.")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("correctness", "serve"), default="correctness",
                        help="correctness runs the B06 native/V200 bundle; serve runs one finite foreground V200 HTTP service")
    parser.add_argument("--gpus", help="optional exact ordered four UUID list from the B01 pair receipts")
    parser.add_argument("--duration-seconds", type=int, default=None,
                        help=f"service lifetime after its correctness check (default {SERVE_DEFAULT_SECONDS}, max {SERVE_MAX_SECONDS})")
    parser.add_argument("--port", type=int, default=None,
                        help=f"loopback HTTP port for serve mode (default {DEFAULT_SERVER_PORT})")
    parser.add_argument("--execute", action="store_true",
                        help="build the candidate, then enter the finite CUDA-controller admission")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    run_id = f"multipair-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{uuid.uuid4().hex[:10]}"
    try:
        plan = _make_plan(run_id, mode=args.mode, duration_seconds=args.duration_seconds,
                          port=args.port)
        if args.gpus:
            requested = [item.strip() for item in args.gpus.split(",") if item.strip()]
            if requested != plan["selected_gpu_uuids"]:
                raise LaunchError("--gpus must exactly match the ordered four UUIDs pinned by the fixture")
    except (LaunchError, OSError, KeyError, TypeError, ValueError) as exc:
        print(f"V200 multipair launch refused: {exc}", file=sys.stderr)
        return 2
    if args.execute:
        try:
            _build_candidate()
        except LaunchError as exc:
            print(f"V200 multipair build refused: {exc}", file=sys.stderr)
            return 2
    missing = _missing_files(plan)
    _write_plan_line(plan, missing, args.execute)
    if not args.execute:
        return 0
    if missing:
        print("Execution refused until the build and every pinned runtime file exist.", file=sys.stderr)
        return 2
    if plan["endpoint"]:
        print("Start a client with: curl " + plan["endpoint"] + "/health", flush=True)
    try:
        return single_pair_serve._run_controller(plan)
    except OSError as exc:
        print(f"cannot start CUDA controller command: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
