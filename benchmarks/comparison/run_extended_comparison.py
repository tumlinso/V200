#!/usr/bin/env python3
"""Compare native-pair inference with the opt-in extended V200 expert path.

This is a bounded, single-pair consumer comparison. It reuses the resident
consumer server lifecycle and request/timing helpers from
``run_pair_comparison.py``. The V200 DSO hashes are captured from the current
``build-multipair`` outputs at preflight and checked again before each case;
this runner does not inherit B05's older binary qualification.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterator


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "benchmarks/comparison"))
import run_pair_comparison as pair  # noqa: E402


FIXTURE_PATH = REPO_ROOT / "benchmarks/fixtures/llama-server-v200.json"
PROFILE_PATH = REPO_ROOT / "docs/v200/consumer-profile.json"
MULTIPAIR_PROFILE_PATH = REPO_ROOT / "docs/v200/multipair-profile.json"
BUILD_DIR = REPO_ROOT / "build-multipair"
ALLOWED_BUILD_DIRS = (BUILD_DIR, REPO_ROOT / "build-perf")
HARNESS_PATH = REPO_ROOT / "tests/consumer/run_consumer.py"
HELPER_PATH = REPO_ROOT / "benchmarks/comparison/run_pair_comparison.py"

DSO_PATHS = {
    "v200_frontdoor": BUILD_DIR / "src/frontdoor/libv200_frontdoor.so",
    "v200_pair_initializer": BUILD_DIR / "src/integration/libv200_pair_init.so",
    "v200_operators": BUILD_DIR / "src/operators/libv200_operators.so",
}
BUILD_CACHE_PATH = BUILD_DIR / "CMakeCache.txt"

SCHEMA = "v200-extended-comparison/v1"
WORKLOAD = "resident_capacity"
WARMUP_REQUESTS = 1
MEASURED_REQUESTS = 3
SERVICE_SECONDS = 15
MAX_BUNDLE_SECONDS = 90
EXTENDED_ENV = "V200_EXTENDED_EXPERT_COMPUTE"
PREFER_LOCAL_HBM_ENV = "V200_PREFER_LOCAL_HBM"

CASES = (
    {
        "id": "native_pair",
        "control": "native_pair",
        "pair_compute": False,
        "extended_expert_compute": False,
        "description": "Native two-V100 layer split, CUDA graphs enabled",
    },
    {
        "id": "v200_extended",
        "control": "facade_owner",
        "pair_compute": True,
        "extended_expert_compute": True,
        "description": (
            "One logical V200 backed by the selected pair, CUDA graphs and "
            "extended Q4/Q6 expert execution enabled"
        ),
    },
)


class ExtendedComparisonError(RuntimeError):
    """The current artifacts or bounded comparison cannot meet its contract."""


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ExtendedComparisonError(f"expected JSON object in {path}")
    return value


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(["git", "-C", str(root), *args], check=True,
                            capture_output=True, text=True, timeout=10)
    return result.stdout.strip()


def _sha256(path: Path) -> str:
    return pair.sha256_file(path)


def _artifact(role: str, path: Path, expected_sha256: str | None = None) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise ExtendedComparisonError(f"required {role} artifact is missing: {path}")
    observed = _sha256(path)
    if expected_sha256 is not None and observed != expected_sha256:
        raise ExtendedComparisonError(
            f"{role} hash mismatch: expected {expected_sha256}, observed {observed} ({path})"
        )
    return {
        "role": role,
        "path": str(path),
        "sha256": observed,
        "identity": "profile-pinned" if expected_sha256 is not None else "captured-current-build",
        "status": "PASS",
    }


def _selected_build_dir(value: Path | str = BUILD_DIR) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    resolved = path.resolve()
    allowed = tuple(item.resolve() for item in ALLOWED_BUILD_DIRS)
    if resolved not in allowed:
        raise ExtendedComparisonError(
            "V200 build directory must be the frozen build-multipair or candidate build-perf"
        )
    return resolved


def _selected_dso_paths(build_dir: Path) -> dict[str, Path]:
    return {
        "v200_frontdoor": build_dir / "src/frontdoor/libv200_frontdoor.so",
        "v200_pair_initializer": build_dir / "src/integration/libv200_pair_init.so",
        "v200_operators": build_dir / "src/operators/libv200_operators.so",
    }


def capture_current_inputs(v200_build_dir: Path | str = BUILD_DIR) -> dict[str, Any]:
    """Hash pinned consumer artifacts and capture, without qualifying, current DSOs."""
    consumer = pair.consumer
    fixture, profile = consumer.load_documents(FIXTURE_PATH, PROFILE_PATH)
    build_dir = _selected_build_dir(v200_build_dir)
    multipair_profile = _read_object(MULTIPAIR_PROFILE_PATH)
    if multipair_profile.get("schema") != "v200-multi-pair-profile/v1":
        raise ExtendedComparisonError("unsupported multipair build profile")
    profile_paths = [Path(item) for item in multipair_profile.get("runtime_libraries", [])]
    profile_relative: list[Path] = []
    try:
        profile_relative = [path.relative_to(BUILD_DIR.relative_to(REPO_ROOT))
                            for path in profile_paths]
    except ValueError as exc:
        raise ExtendedComparisonError(
            "multipair profile runtime library paths no longer match the source profile"
        ) from exc
    expected_relative = [path.relative_to(BUILD_DIR) for path in DSO_PATHS.values()]
    if profile_relative != expected_relative:
        raise ExtendedComparisonError(
            f"multipair profile DSO order/paths differ from this runner: {profile_paths}"
        )
    dso_paths = _selected_dso_paths(build_dir)

    artifacts: list[dict[str, Any]] = []
    binary = profile["binary"]
    artifacts.append(_artifact("llama_server", Path(binary["path"]), binary["sha256"]))
    closure = profile.get("runtime_closure")
    if not isinstance(closure, list) or not closure:
        raise ExtendedComparisonError("consumer profile has no pinned llama runtime closure")
    for item in closure:
        if not isinstance(item, dict):
            raise ExtendedComparisonError("consumer runtime-closure entry is malformed")
        artifacts.append(_artifact(f"llama_{item['role']}", Path(item["path"]), item["sha256"]))

    artifacts.extend([
        _artifact("consumer_harness", HARNESS_PATH),
        _artifact("pair_comparison_helper", HELPER_PATH),
        _artifact("extended_comparison_runner", Path(__file__)),
        _artifact("consumer_fixture", FIXTURE_PATH),
        _artifact("consumer_profile", PROFILE_PATH),
        _artifact("multipair_build_profile", MULTIPAIR_PROFILE_PATH),
    ])
    for role, path in dso_paths.items():
        artifacts.append(_artifact(role, path))
    build_cache = _artifact("selected_v200_cmake_cache", build_dir / "CMakeCache.txt")
    artifacts.append(build_cache)

    model = fixture["model"]
    model_path = Path(model["path"])
    if not model_path.is_file():
        raise ExtendedComparisonError(f"pinned model is missing: {model_path}")
    model_stat = model_path.stat()
    if int(model_stat.st_size) != int(model["size_bytes"]):
        raise ExtendedComparisonError(
            f"model size changed: expected {model['size_bytes']}, observed {model_stat.st_size}"
        )

    source_head = _git(REPO_ROOT, "rev-parse", "HEAD")
    tracked_status = _git(REPO_ROOT, "status", "--porcelain", "--untracked-files=no")
    llama_source = profile["source"]
    llama_head = _git(Path(llama_source["repository"]), "rev-parse", "HEAD")
    llama_status = _git(Path(llama_source["repository"]), "status", "--porcelain",
                        "--untracked-files=no")
    clean = not tracked_status and llama_head == llama_source["commit"] and not llama_status
    return {
        "status": "PASS_CAPTURE_ONLY",
        "qualification": "CURRENT_ARTIFACT_CAPTURE_ONLY; no B05 qualification inherited",
        "live_run_eligible": clean,
        "v200_source": {
            "repository": str(REPO_ROOT),
            "commit": source_head,
            "tracked_worktree_clean": not bool(tracked_status),
            "tracked_status": tracked_status,
        },
        "llama_source": {
            "repository": llama_source["repository"],
            "profile_commit": llama_source["commit"],
            "observed_commit": llama_head,
            "tracked_worktree_clean": not bool(llama_status),
            "tracked_status": llama_status,
            "status": "PASS" if llama_head == llama_source["commit"] and not llama_status else "MISMATCH",
        },
        "consumer": {
            "fixture_id": fixture["fixture_id"],
            "fixture_sha256": _sha256(FIXTURE_PATH),
            "profile_sha256": _sha256(PROFILE_PATH),
            "model": {
                "path": str(model_path.resolve()),
                "sha256_expected": model["sha256"],
                "size_bytes": model_stat.st_size,
            "file_signature": {
                "device": model_stat.st_dev,
                "inode": model_stat.st_ino,
                "mtime_ns": model_stat.st_mtime_ns,
                "ctime_ns": model_stat.st_ctime_ns,
            },
                "sha256_verification": "consumer verifies full model identity before server launch",
            },
            "workload": WORKLOAD,
        },
        "build": {
            "directory": str(build_dir),
            "variant": build_dir.name,
            "toolkit_root": multipair_profile.get("build", {}).get("cuda_toolkit_root"),
            "cmake_cache_sha256": build_cache["sha256"],
        },
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
    }


def _assert_inputs_unchanged(initial: dict[str, Any],
                             v200_build_dir: Path | str = BUILD_DIR) -> None:
    current = capture_current_inputs(v200_build_dir)
    prior_hashes = [(row["role"], row["sha256"]) for row in initial["artifacts"]]
    current_hashes = [(row["role"], row["sha256"]) for row in current["artifacts"]]
    if current_hashes != prior_hashes:
        raise ExtendedComparisonError("current input artifact changed during the comparison")
    if current["v200_source"]["commit"] != initial["v200_source"]["commit"]:
        raise ExtendedComparisonError("V200 source commit changed during the comparison")
    if current["consumer"]["model"] != initial["consumer"]["model"]:
        raise ExtendedComparisonError("pinned model identity changed during the comparison")
    if not current["live_run_eligible"]:
        raise ExtendedComparisonError("tracked source changed or llama source identity is not clean")


def _helper_manifest(provenance: dict[str, Any]) -> dict[str, Any]:
    return {"artifacts": [row for row in provenance["artifacts"]
                          if row["role"] in DSO_PATHS]}


def _measured_reference(case: dict[str, Any]) -> str:
    for request in case.get("requests", []):
        if request.get("kind") == "measured":
            content = request.get("response", {}).get("content")
            if isinstance(content, str):
                return content
            break
    raise ExtendedComparisonError(f"{case.get('id')} produced no measured reference response")


def _server_exit_evidence(case: dict[str, Any]) -> dict[str, Any]:
    consumer_result = case.get("consumer")
    if not isinstance(consumer_result, dict):
        return {"status": "FAIL", "reason": "consumer result is missing"}
    process = consumer_result.get("server_process") or {}
    cleanup = consumer_result.get("cleanup") or {}
    exit_code = process.get("exit_code")
    normal_exit = exit_code == 0
    requested_sigterm = (
        exit_code == -signal.SIGTERM
        and cleanup.get("signaled_owned_group") is True
        and cleanup.get("sigkill_escalation") is False
    )
    pass_status = ((normal_exit or requested_sigterm)
                   and cleanup.get("no_live_group_members") is True)
    return {
        "status": "PASS" if pass_status else "FAIL",
        "exit_code": exit_code,
        "reason": "normal_exit" if normal_exit else "requested_sigterm" if requested_sigterm else "unexpected_exit",
        "cleanup_no_live_group_members": cleanup.get("no_live_group_members"),
        "cleanup_sigkill_escalation": cleanup.get("sigkill_escalation"),
    }


def _extended_operator_evidence(case: dict[str, Any]) -> dict[str, Any]:
    consumer_result = case.get("consumer")
    telemetry = (consumer_result or {}).get("v200_profile_telemetry", {})
    records = telemetry.get("operator_records", [])
    q4 = [row for row in records if row.get("event") == "q4k_mmvq_process_exit"]
    q6 = [row for row in records if row.get("event") == "q6k_mmvq_expert_process_exit"]
    q4_summary = q4[-1] if q4 else {}
    q6_summary = q6[-1] if q6 else {}
    q4_calls = int(q4_summary.get("expert_extended_peer_owner_calls", 0))
    q6_calls = int(q6_summary.get("q6_expert_peer_owner_calls", 0))
    return {
        "status": "PASS" if q4_calls > 0 and q6_calls > 0 else "MISSING_EXTENDED_FAMILY",
        "q4_low_column_peer_calls": q4_calls,
        "q4_low_column_plain_verified_outputs": int(
            q4_summary.get("expert_extended_plain_verified_outputs", 0)),
        "q6_peer_calls": q6_calls,
        "q6_plain_peer_calls": int(q6_summary.get("q6_expert_plain_peer_owner_calls", 0)),
        "q6_plain_verified_outputs": int(q6_summary.get("q6_expert_plain_verified_outputs", 0)),
        "q4_process_exit_record_present": bool(q4_summary),
        "q6_process_exit_record_present": bool(q6_summary),
        "operator_profile_status": telemetry.get("status"),
    }


def _case_contract(case: dict[str, Any]) -> dict[str, Any]:
    consumer_result = case.get("consumer")
    if not isinstance(consumer_result, dict):
        return {"status": "FAIL", "failure": "consumer result is missing"}
    failures: list[str] = []
    exit_evidence = _server_exit_evidence(case)
    if exit_evidence["status"] != "PASS":
        failures.append("server did not exit cleanly after requested bounded service")
    if consumer_result.get("application_status") != "PASS":
        failures.append("consumer startup/oracle/cleanup status did not pass")
    if consumer_result.get("layer_offload_evidence", {}).get("status") != "PASS":
        failures.append("full GPU layer residency was not confirmed")
    if consumer_result.get("input_embedding_evidence", {}).get("status") != "PASS":
        failures.append("CUDA input embedding residency was not confirmed")

    ledger = consumer_result.get("v200_profile_telemetry", {})
    ledger_evidence: dict[str, Any] = {"status": "NOT_APPLICABLE"}
    if case["control"] == "facade_owner":
        ledger_evidence = {
            "status": ledger.get("status"),
            "cleanup_accounting_zero": ledger.get("cleanup_accounting_zero"),
            "peak_total_owner_used_bytes": ledger.get("peak_total_owner_used_bytes"),
            "owner_budget_bytes": ledger.get("owner_budget_bytes"),
        }
        if ledger.get("status") != "PASS" or ledger.get("cleanup_accounting_zero") is not True:
            failures.append("V200 owner ledger did not pass zero-accounting cleanup")
        graph = consumer_result.get("graph_capture_evidence", {})
        if graph.get("status") != "PASS_CAPTURE_AND_REPLAY":
            failures.append("V200 CUDA graph capture/replay did not pass")

    extended = _extended_operator_evidence(case) if case["id"] == "v200_extended" else None
    if extended is not None and extended["status"] != "PASS":
        failures.append("Q4 low-column and Q6 peer expert families were not both observed")
    return {
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "server_exit": exit_evidence,
        "owner_ledger": ledger_evidence,
        "extended_operator_coverage": extended,
        "cuda_graphs_enabled": consumer_result.get("preview_environment", {}).get(
            "cuda_graphs_enabled"),
        "extended_environment_requested": case.get("extended_expert_compute_enabled", False),
    }


def compare_case_outputs(cases: list[dict[str, Any]]) -> tuple[str | None, list[dict[str, Any]]]:
    if not cases:
        return None, [{"status": "NO_NATIVE_REFERENCE_OUTPUT"}]
    try:
        reference = _measured_reference(cases[0])
    except ExtendedComparisonError:
        return None, [{"status": "NO_NATIVE_REFERENCE_OUTPUT"}]
    reference_hash = hashlib.sha256(reference.encode("utf-8")).hexdigest()
    divergences: list[dict[str, Any]] = []
    for record in cases:
        requests = record.get("requests", [])
        matches: list[bool] = []
        for request in requests:
            content = request.get("response", {}).get("content")
            match = isinstance(content, str) and content == reference
            request["exact_response_match_to_native_pair"] = match
            matches.append(match)
            if not match:
                observed = content if isinstance(content, str) else ""
                divergences.append({
                    "case": record.get("id"),
                    "kind": request.get("kind"),
                    "ordinal": request.get("ordinal"),
                    "expected_sha256": reference_hash,
                    "observed_sha256": hashlib.sha256(observed.encode("utf-8")).hexdigest(),
                })
        record["cross_case_response_equivalence"] = (
            "MATCH" if matches and all(matches) else "DIVERGED")
        if record["cross_case_response_equivalence"] != "MATCH":
            record["status"] = "FAIL"
    return reference, divergences


@contextlib.contextmanager
def _extended_environment(enabled: bool) -> Iterator[None]:
    previous = os.environ.get(EXTENDED_ENV)
    if enabled:
        os.environ[EXTENDED_ENV] = "1"
    else:
        os.environ.pop(EXTENDED_ENV, None)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(EXTENDED_ENV, None)
        else:
            os.environ[EXTENDED_ENV] = previous


@contextlib.contextmanager
def _local_hbm_environment(preference: bool | None) -> Iterator[None]:
    previous = os.environ.get(PREFER_LOCAL_HBM_ENV)
    if preference is not None:
        os.environ[PREFER_LOCAL_HBM_ENV] = "1" if preference else "0"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(PREFER_LOCAL_HBM_ENV, None)
        else:
            os.environ[PREFER_LOCAL_HBM_ENV] = previous


def run_comparison(output_dir: Path, *, v200_build_dir: Path | str = BUILD_DIR,
                   case_definitions: tuple[dict[str, Any], ...] = CASES,
                   repetitions: int = MEASURED_REQUESTS,
                   service_seconds: int = SERVICE_SECONDS,
                   max_bundle_seconds: int = MAX_BUNDLE_SECONDS,
                   prefer_local_hbm: bool | None = None) -> dict[str, Any]:
    provenance = capture_current_inputs(v200_build_dir)
    if not provenance["live_run_eligible"]:
        raise ExtendedComparisonError(
            "live comparison requires a committed, clean tracked V200 checkout and the pinned clean llama source"
        )
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise ExtendedComparisonError(f"output directory must be empty: {output_dir}")

    started = time.monotonic()
    deadline = started + max_bundle_seconds
    cases: list[dict[str, Any]] = []
    failure: str | None = None
    reference: str | None = None
    helper_manifest = _helper_manifest(provenance)
    for definition in case_definitions:
        _assert_inputs_unchanged(provenance, v200_build_dir)
        if deadline - time.monotonic() < service_seconds + 10:
            failure = "insufficient remaining bundle budget to start the next server"
            break
        case_deadline = deadline
        with _extended_environment(bool(definition["extended_expert_compute"])):
            with _local_hbm_environment(prefer_local_hbm):
                record = pair._serve_case(
                    dict(definition), helper_manifest, output_dir,
                    repetitions, service_seconds, case_deadline, reference,
                )
        record["extended_expert_compute_enabled"] = bool(definition["extended_expert_compute"])
        if definition["id"] == "native_pair" and record.get("requests"):
            try:
                reference = _measured_reference(record)
            except ExtendedComparisonError as exc:
                failure = str(exc)
        record["contract"] = _case_contract(record)
        if record["contract"]["status"] != "PASS":
            record["status"] = "FAIL"
            record["failure"] = record.get("failure") or "; ".join(record["contract"]["failures"])
        cases.append(record)
        if record.get("status") != "PASS":
            failure = failure or record.get("failure") or f"{definition['id']} failed"
            break
        if time.monotonic() > deadline:
            failure = f"comparison exceeded its {max_bundle_seconds}-second bundle ceiling"
            break

    paired_comparison = {case["id"] for case in case_definitions} >= {
        "native_pair", "v200_extended"}
    if paired_comparison:
        reference, divergences = compare_case_outputs(cases)
    else:
        reference, divergences = None, []
        for record in cases:
            responses = [request.get("response", {}).get("content")
                         for request in record.get("requests", [])]
            stable = bool(responses) and all(
                isinstance(content, str) and content == responses[0]
                for content in responses)
            record["within_case_response_stability"] = "MATCH" if stable else "DIVERGED"
            if not stable:
                failure = failure or f"{record['id']} response changed between warmup and measurement"
    if divergences:
        failure = failure or "one or more raw response outputs differ from the native-pair reference"
    if paired_comparison and reference is None:
        failure = failure or "native-pair case produced no output reference"

    _assert_inputs_unchanged(provenance, v200_build_dir)
    elapsed = time.monotonic() - started
    status = "PASS" if failure is None and len(cases) == len(case_definitions) else "FAIL"
    result = {
        "schema": SCHEMA,
        "status": status,
        "failure": failure,
        "source_provenance": provenance,
        "workload": WORKLOAD,
        "gpu_scope": "one controller-admitted NVLink V100 pair",
        "cases": cases,
        "response_equivalence": ("MATCH" if not divergences else "DIVERGED")
            if paired_comparison else "NOT_CHECKED_CROSS_CASE",
        "response_divergences": divergences,
        "measurement_contract": {
            "warmup_requests_per_server": WARMUP_REQUESTS,
            "measured_requests_per_server": repetitions,
            "fixed_generated_tokens": pair.GENERATION_TOKENS,
            "server_service_seconds_per_case": service_seconds,
            "bundle_ceiling_seconds": max_bundle_seconds,
            "graphs": "enabled in all selected cases",
            "extended_v200_environment": f"{EXTENDED_ENV}=1 only for V200_extended",
            "v200_prefer_local_hbm": ("inherited" if prefer_local_hbm is None else
                                      "1" if prefer_local_hbm else "0"),
            "pair_compute": "enabled only for V200_extended",
            "output_equivalence": ("all warmup and measured response content is compared exactly to the native-pair measured response"
                                    if paired_comparison else "cross-case output equivalence not checked; within-case warmup/measured response stability is checked"),
            "server_exit": "normal exit or deliberate SIGTERM with no SIGKILL escalation and no surviving process-group members",
            "v200_cleanup": "pair ledger must close with zero allocations, reservations, cleanup-pending entries and owner usage",
            "interpretation": f"{repetitions} measured request(s) for one resident model and prompt; descriptive only",
        },
        "request": pair.generation_request(),
        "elapsed_seconds": elapsed,
        "raw_timing_fields": ["wall_milliseconds", "server.prompt_ms", "server.predicted_ms",
                              "server.prompt_per_second", "server.predicted_per_second"],
    }
    result_path = output_dir / "comparison.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
                           encoding="utf-8")
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-binaries", action="store_true",
                        help="CPU-only current artifact hash and source identity capture")
    parser.add_argument("--run", action="store_true",
                        help="run the bounded comparison inside CUDA controller admission")
    parser.add_argument("--execute", action="store_true",
                        help="required acknowledgement for live GPU work")
    parser.add_argument("--output-dir", type=Path,
                        help="new or empty directory for comparison evidence")
    parser.add_argument("--v200-build-dir", type=Path, default=BUILD_DIR,
                        help="V200 build artifact root (build-multipair or build-perf; no rebuild)")
    parser.add_argument("--prefer-local-hbm", choices=("0", "1"),
                        help="set V200_PREFER_LOCAL_HBM for each selected server; omit to inherit")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.verify_binaries:
            current = capture_current_inputs(args.v200_build_dir)
            print(json.dumps(current, indent=2, sort_keys=True))
            return 0 if current["live_run_eligible"] else 2
        if not args.run or not args.execute:
            raise ExtendedComparisonError(
                "live mode requires both --run and --execute; --verify-binaries is CPU-only"
            )
        if args.output_dir is None:
            raise ExtendedComparisonError("live mode requires --output-dir")
        result = run_comparison(
            args.output_dir, v200_build_dir=args.v200_build_dir,
            prefer_local_hbm=(None if args.prefer_local_hbm is None
                              else args.prefer_local_hbm == "1"))
        print(json.dumps({"status": result["status"], "failure": result["failure"],
                          "result": str(args.output_dir.resolve() / "comparison.json"),
                          "elapsed_seconds": result["elapsed_seconds"]}, sort_keys=True))
        return 0 if result["status"] == "PASS" else 1
    except KeyboardInterrupt:
        print("comparison interrupted; active worker received bounded cleanup signal", file=sys.stderr)
        return 130
    except (ExtendedComparisonError, pair.ComparisonError, pair.consumer.ConsumerError,
            OSError, ValueError, KeyError, TypeError, json.JSONDecodeError,
            subprocess.SubprocessError) as exc:
        print(f"extended comparison refused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
