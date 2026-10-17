#!/usr/bin/env python3
"""Run a bounded, matched resident-llama comparison under CUDA controller admission.

The server lifecycle and resident-capacity safety checks come from the frozen
tests/consumer/run_consumer.py harness. Each case starts exactly one server,
lets that harness validate the exact startup oracle and memory ownership, then
serves one warmup and three measured fixed-length requests before clean exit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import queue
import re
import signal
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = Path(__file__).with_name("qualified-binaries.json")
HARNESS_PATH = REPO_ROOT / "tests/consumer"
sys.path.insert(0, str(HARNESS_PATH))
import run_consumer as consumer  # noqa: E402


SCHEMA = "v200-pair-comparison/v1"
WORKLOAD = "resident_capacity"
MAX_BUNDLE_SECONDS = 270
DEFAULT_BUNDLE_SECONDS = 260
DEFAULT_SERVICE_SECONDS = 65
MAX_SERVICE_SECONDS = 80
DEFAULT_REPETITIONS = 3
GENERATION_TOKENS = 64
REQUIRED_ROLES = {
    "llama_server", "llama_server_impl", "llama_common", "llama_multimodal",
    "llama", "ggml_loader", "ggml_cpu", "ggml_cuda", "ggml_base",
    "v200_frontdoor", "v200_pair_initializer", "v200_operators",
    "consumer_harness", "consumer_fixture", "consumer_profile",
}
READY_RE = re.compile(r"^V200_SERVER_READY http://127\.0\.0\.1:(\d+)\s*$")

BENCHMARK_PROMPT = (
    "You are a concise systems engineer. Explain how a GPU can safely read peer GPU memory over "
    "NVLink during inference. Give one coherent technical explanation in plain prose."
)

CASES = (
    {"id": "native_pair", "control": "native_pair", "pair_compute": False,
     "description": "Native two-V100 layer split, CUDA graphs enabled"},
    {"id": "v200_anchor", "control": "facade_owner", "pair_compute": False,
     "description": "One logical V200 backed by the pair, graph capture enabled, pair compute disabled"},
    {"id": "v200_pair_compute", "control": "facade_owner", "pair_compute": True,
     "description": "One logical V200 backed by the pair, graph capture and qualified pair compute enabled"},
)


class ComparisonError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ComparisonError(f"expected a JSON object in {path}")
    return value


def _manifest_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def verify_manifest(manifest_path: Path = MANIFEST_PATH,
                    root: Path = REPO_ROOT) -> dict[str, Any]:
    manifest = read_json(manifest_path)
    if manifest.get("schema") != "v200-comparison-binary-manifest/v1":
        raise ComparisonError("unsupported binary-manifest schema")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ComparisonError("binary manifest has no artifacts")
    verified = []
    failures = []
    roles: set[str] = set()
    for entry in artifacts:
        if not isinstance(entry, dict):
            raise ComparisonError("binary-manifest entries must be objects")
        role, path_text, expected = entry.get("role"), entry.get("path"), entry.get("sha256")
        if not isinstance(role, str) or not isinstance(path_text, str):
            raise ComparisonError("binary-manifest role/path is malformed")
        if role in roles:
            raise ComparisonError(f"binary manifest repeats artifact role {role!r}")
        roles.add(role)
        if not isinstance(expected, str) or re.fullmatch(r"[0-9a-f]{64}", expected) is None:
            raise ComparisonError(f"{role}: expected SHA-256 must contain 64 lowercase hex digits")
        path = _manifest_path(root, path_text)
        if not path.is_file():
            failures.append({"role": role, "path": str(path), "status": "MISSING"})
            continue
        observed = sha256_file(path)
        row = {"role": role, "path": str(path), "expected_sha256": expected,
               "observed_sha256": observed,
               "status": "PASS" if observed == expected else "HASH_MISMATCH"}
        verified.append(row)
        if observed != expected:
            failures.append(row)

    missing_roles = sorted(REQUIRED_ROLES - roles)
    if missing_roles:
        failures.append({"status": "MISSING_REQUIRED_ROLES", "roles": missing_roles})

    prior = manifest.get("prior_graph_result", {})
    prior_path_text, prior_hash = prior.get("path"), prior.get("sha256")
    if not isinstance(prior_path_text, str) or not isinstance(prior_hash, str):
        raise ComparisonError("prior graph-run provenance is malformed")
    prior_path = _manifest_path(root, prior_path_text)
    if not prior_path.is_file() or sha256_file(prior_path) != prior_hash:
        failures.append({"role": "prior_graph_result", "path": str(prior_path),
                         "status": "MISSING_OR_HASH_MISMATCH"})

    required_provenance = {
        "qualified_v200_source_commit": "2bd3020146cd4378a6bb60a614b361d030a3bc34",
        "consumer_harness_source_commit": "df12371c23801291a08c631527e36f22dd21d66c",
        "prior_graph_controller_evidence_id": "797dc352-ff64-40cf-ae18-3dcfc29874f2",
    }
    for field, expected in required_provenance.items():
        if manifest.get(field) != expected:
            failures.append({"role": "provenance", "field": field,
                             "expected": expected, "observed": manifest.get(field),
                             "status": "MISMATCH"})
    if failures:
        raise ComparisonError("qualified binaries/provenance failed preflight: "
                              + json.dumps(failures, sort_keys=True))
    return {"status": "PASS", "manifest": str(manifest_path.resolve()),
            "qualified_v200_source_commit": manifest["qualified_v200_source_commit"],
            "consumer_harness_source_commit": manifest["consumer_harness_source_commit"],
            "prior_graph_controller_evidence_id": manifest["prior_graph_controller_evidence_id"],
            "artifact_count": len(verified), "artifacts": verified,
            "prior_graph_result": {"path": str(prior_path), "sha256": prior_hash}}


def generation_request() -> dict[str, Any]:
    """Fixed deterministic 64-token decode request, independent of prompt caching."""
    return {
        "prompt": BENCHMARK_PROMPT,
        "temperature": 0.0,
        "top_k": 1,
        "top_p": 1.0,
        "seed": 42,
        "n_predict": GENERATION_TOKENS,
        "ignore_eos": True,
        "repeat_penalty": 1.0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "stream": False,
        "cache_prompt": False,
    }


def validate_sample(status: int, response: Any) -> dict[str, Any]:
    if status != 200 or not isinstance(response, dict):
        raise ComparisonError(f"completion returned HTTP {status}: {response!r}")
    content = response.get("content")
    predicted = response.get("tokens_predicted")
    evaluated = response.get("tokens_evaluated")
    timings = response.get("timings")
    if not isinstance(content, str):
        raise ComparisonError("64-token response has no string content")
    if isinstance(predicted, bool) or predicted != GENERATION_TOKENS:
        raise ComparisonError(f"expected exactly {GENERATION_TOKENS} generated tokens, got {predicted!r}")
    if isinstance(evaluated, bool) or not isinstance(evaluated, int) or evaluated < 1:
        raise ComparisonError(f"invalid server prompt-token count {evaluated!r}")
    if not isinstance(timings, dict):
        raise ComparisonError("server response omitted prefill/decode timings")
    required = ("prompt_n", "prompt_ms", "predicted_n", "predicted_ms",
                "prompt_per_second", "predicted_per_second")
    missing = [field for field in required if field not in timings]
    if missing:
        raise ComparisonError(f"server timings omitted fields {missing}")
    if int(timings["predicted_n"]) != GENERATION_TOKENS:
        raise ComparisonError(f"server timing reports {timings['predicted_n']} decode tokens")
    if int(timings["prompt_n"]) != evaluated:
        raise ComparisonError("server response prompt token count differs from timing record")
    for field in ("prompt_ms", "predicted_ms", "prompt_per_second", "predicted_per_second"):
        value = timings[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ComparisonError(f"server timing field {field} is not finite numeric data")
    return {"content": content, "tokens_evaluated": evaluated,
            "tokens_predicted": predicted, "timings": timings}


def check_exact_output(content: str, expected: str | None) -> bool:
    return expected is None or content == expected


def _worker_main(args: argparse.Namespace) -> int:
    """Use the existing consumer's full ownership, preflight, and cleanup path."""
    sys.path.insert(0, str(HARNESS_PATH))
    fixture, profile = consumer.load_documents()
    control = args.worker_control
    preloads = [Path(item).resolve() for item in args.worker_preload]
    facade_mode = "pair" if control == "facade_owner" else None
    selected = consumer.validate_live_environment(control, fixture, dict(os.environ), facade_mode)
    nvidia_smi, visible = consumer._verify_nvidia_smi_uuid_selection(selected)
    identity = consumer.verify_identities(profile, fixture,
                                          consumer.FIXTURE_PATH, consumer.PROFILE_PATH,
                                          preloads if control == "facade_owner" else None)
    memory = consumer._nvidia_smi_memory(nvidia_smi, selected)

    def terminate_as_interrupt(_signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt("comparison worker interrupted; consumer cleanup is running")

    signal.signal(signal.SIGTERM, terminate_as_interrupt)
    signal.signal(signal.SIGINT, terminate_as_interrupt)
    result = consumer._run_one(
        WORKLOAD, control, fixture, profile, identity, selected,
        Path(args.worker_output_dir), preloads, monitor=True,
        facade_mode=facade_mode, environ=dict(os.environ),
        nvidia_smi_path=nvidia_smi, visible_gpu_uuids=visible,
        server_port=args.worker_port,
        serve_duration_seconds=args.worker_serve_seconds,
        physical_memory_baseline=memory,
        cuda_graphs_enabled=True)
    print(json.dumps({"worker_status": result["application_status"],
                      "result_path": result["result_path"],
                      "failure": result["failure"]}, sort_keys=True), flush=True)
    return 0 if result["application_status"] == "PASS" else 1


def _case_preloads(case: dict[str, Any], manifest: dict[str, Any]) -> list[str]:
    if case["control"] != "facade_owner":
        return []
    by_role = {item["role"]: item["path"] for item in manifest["artifacts"]}
    # Graph rewriting resides in the qualified operator DSO. Both V200 cases
    # load the same DSOs; only the compute case sets V200_PAIR_COMPUTE=1.
    roles = ("v200_frontdoor", "v200_pair_initializer", "v200_operators")
    return [str(_manifest_path(REPO_ROOT, by_role[role])) for role in roles]


def _reader(stream: Any, messages: queue.Queue[str], log_path: Path) -> None:
    try:
        with log_path.open("w", encoding="utf-8") as log:
            for line in stream:
                log.write(line)
                log.flush()
                messages.put(line.rstrip("\n"))
    finally:
        messages.put("")


def _wait_ready(proc: subprocess.Popen[str], messages: queue.Queue[str],
                timeout_s: float, log_path: Path) -> int:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            line = messages.get(timeout=min(0.5, deadline - time.monotonic()))
        except queue.Empty:
            if proc.poll() is not None:
                raise ComparisonError(f"consumer worker exited early with status {proc.returncode}; see {log_path}")
            continue
        match = READY_RE.fullmatch(line)
        if match:
            return int(match.group(1))
        if line == "" and proc.poll() is not None:
            break
    raise ComparisonError(f"consumer worker did not announce a ready endpoint; see {log_path}")


def _serve_case(case: dict[str, Any], manifest: dict[str, Any],
                output_dir: Path, repetitions: int, serve_seconds: int,
                case_deadline: float, expected_text: str | None) -> dict[str, Any]:
    case_dir = output_dir / case["id"]
    consumer_dir = case_dir / "consumer"
    consumer_dir.mkdir(parents=True, exist_ok=True)
    port = consumer._free_loopback_port()
    preloads = _case_preloads(case, manifest)
    command = [sys.executable, str(Path(__file__).resolve()), "--worker",
               "--worker-control", case["control"],
               "--worker-output-dir", str(consumer_dir),
               "--worker-port", str(port),
               "--worker-serve-seconds", str(serve_seconds)]
    for preload in preloads:
        command.extend(["--worker-preload", preload])
    environment = dict(os.environ)
    environment.pop("V200_OPERATOR_VERIFY", None)
    environment.pop("V200_PAIR_COMPUTE", None)
    if case["pair_compute"]:
        environment["V200_PAIR_COMPUTE"] = "1"
    worker_log = case_dir / "worker.log"
    messages: queue.Queue[str] = queue.Queue()
    proc = subprocess.Popen(command, cwd=REPO_ROOT, env=environment,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    assert proc.stdout is not None
    reader = threading.Thread(target=_reader,
                              args=(proc.stdout, messages, worker_log), daemon=True)
    reader.start()
    raw_requests: list[dict[str, Any]] = []
    failure: str | None = None
    endpoint: str | None = None
    case_started_unix = time.time()
    try:
        ready_timeout = min(consumer.STARTUP_TIMEOUT_S + 5.0,
                            max(1.0, case_deadline - time.monotonic() - serve_seconds))
        active_port = _wait_ready(proc, messages, ready_timeout, worker_log)
        if active_port != port:
            raise ComparisonError(f"worker announced port {active_port}, expected {port}")
        endpoint = f"http://127.0.0.1:{active_port}/completion"
        body = generation_request()
        expected = expected_text
        fixture, _ = consumer.load_documents()
        request_timeout_s = float(fixture["server"]["timeouts_seconds"]["request"])
        rounds = [("warmup", None)] + [("measured", index + 1)
                                         for index in range(repetitions)]
        for kind, ordinal in rounds:
            started = time.perf_counter()
            request_remaining = case_deadline - time.monotonic()
            if request_remaining <= 0:
                raise ComparisonError("comparison reached its configured whole-bundle time limit")
            status, response = consumer._completion_request(
                endpoint, body, min(request_timeout_s, request_remaining))
            wall_seconds = time.perf_counter() - started
            validated = validate_sample(status, response)
            if expected is None:
                expected = validated["content"]
            exact_match = check_exact_output(validated["content"], expected)
            record = {"kind": kind, "ordinal": ordinal,
                      "wall_seconds": wall_seconds, "wall_milliseconds": wall_seconds * 1000.0,
                      "status_code": status, "exact_response_match": exact_match,
                      "response": response,
                      "server": {"tokens_evaluated": validated["tokens_evaluated"],
                                 "tokens_predicted": validated["tokens_predicted"],
                                 "prompt_ms": validated["timings"]["prompt_ms"],
                                 "predicted_ms": validated["timings"]["predicted_ms"],
                                 "prompt_per_second": validated["timings"]["prompt_per_second"],
                                 "predicted_per_second": validated["timings"]["predicted_per_second"]}}
            raw_requests.append(record)
        if time.monotonic() > case_deadline:
            raise ComparisonError("comparison reached its configured whole-bundle time limit")
    except KeyboardInterrupt as exc:
        failure = f"{type(exc).__name__}: {exc}"
        if proc.poll() is None:
            proc.send_signal(signal.SIGINT)
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            remaining = max(0.1, case_deadline - time.monotonic())
            proc.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            # SIGINT reaches the worker's handler; _run_one then executes its
            # existing owned-server, memory-guard, and ledger cleanup path.
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                failure = failure or "worker did not finish bounded cleanup after SIGINT"
        reader.join(timeout=2)

    result_files = list(consumer_dir.glob("resident_capacity-*.json"))
    consumer_result = None
    if len(result_files) == 1:
        try:
            consumer_result = read_json(result_files[0])
        except (OSError, json.JSONDecodeError, ComparisonError) as exc:
            failure = failure or f"cannot read consumer result: {exc}"
    else:
        failure = failure or f"expected one consumer result, found {len(result_files)}"
    if proc.returncode != 0:
        failure = failure or f"consumer worker returned {proc.returncode}; see {worker_log}"
    if not isinstance(consumer_result, dict) or consumer_result.get("application_status") != "PASS":
        failure = failure or (consumer_result or {}).get("failure") or "consumer startup oracle/cleanup failed"
    if consumer_result:
        if consumer_result.get("response", {}).get("normalized_content") != "2":
            failure = failure or "startup exact-answer oracle did not return 2"
        if consumer_result.get("preview_environment", {}).get("cuda_graphs_enabled") is not True:
            failure = failure or "consumer result says CUDA graphs were not enabled"
        if case["control"] == "facade_owner":
            graph = consumer_result.get("graph_capture_evidence", {})
            if graph.get("status") != "PASS_CAPTURE_AND_REPLAY":
                failure = failure or f"V200 graph capture/replay was not qualified: {graph}"
            if consumer_result.get("v200_profile_telemetry", {}).get("status") != "PASS":
                failure = failure or "V200 owner ledger did not close cleanly"
        if case["pair_compute"]:
            operators = consumer_result.get("v200_profile_telemetry", {}).get("operator_records", [])
            peer_calls = max((int(item.get("peer_owner_calls", 0))
                              + int(item.get("expert_peer_owner_calls", 0))
                              for item in operators), default=0)
            if peer_calls < 1:
                failure = failure or "pair-compute case recorded no real peer-owner operation"
        if case["control"] == "facade_owner" and consumer_result.get("layer_offload_evidence", {}).get("status") != "PASS":
            failure = failure or "full GPU layer residency was not confirmed"
        if case["control"] == "facade_owner" and consumer_result.get("input_embedding_evidence", {}).get("status") != "PASS":
            failure = failure or "GPU input-embedding residency was not confirmed"
    if len(raw_requests) != repetitions + 1:
        failure = failure or f"captured {len(raw_requests)} successful requests; expected {repetitions + 1}"
    sample_records = [record for record in raw_requests if record["kind"] == "measured"]
    wall = [float(record["wall_milliseconds"]) for record in sample_records]
    server_prefill = [float(record["server"]["prompt_ms"]) for record in sample_records]
    server_decode = [float(record["server"]["predicted_ms"]) for record in sample_records]
    stats = {
        "measured_samples": len(sample_records),
        "wall_ms": _describe(wall),
        "server_prefill_ms": _describe(server_prefill),
        "server_decode_ms": _describe(server_decode),
        "server_decode_tokens_per_second": _describe(
            [float(record["server"]["predicted_per_second"]) for record in sample_records]),
        "server_prompt_tokens_per_second": _describe(
            [float(record["server"]["prompt_per_second"]) for record in sample_records]),
    }
    return {
        "id": case["id"], "control": case["control"],
        "description": case["description"],
        "pair_compute_enabled": case["pair_compute"],
        "cuda_graphs_enabled": True,
        "status": "PASS" if failure is None else "FAIL",
        "failure": failure,
        "worker_log": str(worker_log),
        "consumer_result": str(result_files[0]) if len(result_files) == 1 else None,
        "consumer": consumer_result,
        "request_body": generation_request(),
        "exact_output_oracle": {"content": expected_text,
                                "sha256": hashlib.sha256((expected_text or "").encode()).hexdigest()
                                if expected_text is not None else None},
        "response_equivalence": "PENDING_REFERENCE",
        "requests": raw_requests,
        "summary": stats,
        "wall_started_unix": case_started_unix,
    }


def _describe(values: list[float]) -> dict[str, Any] | None:
    if not values:
        return None
    return {"n": len(values), "min": min(values), "median": statistics.median(values),
            "max": max(values), "mean": statistics.fmean(values),
            "population_sd": statistics.pstdev(values) if len(values) > 1 else 0.0}


def run_comparison(args: argparse.Namespace) -> dict[str, Any]:
    manifest_info = verify_manifest(Path(args.manifest), REPO_ROOT)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise ComparisonError(f"output directory must be empty to preserve artifacts: {output_dir}")
    started = time.monotonic()
    deadline = started + args.max_total_seconds
    cases: list[dict[str, Any]] = []
    oracle_text: str | None = None
    failure: str | None = None
    for case in CASES:
        if deadline - time.monotonic() < args.serve_seconds + 10:
            failure = "insufficient remaining bundle budget to start the next server"
            break
        # Verify the three qualified preload artifacts again immediately before
        # any V200 process starts; their build/ files are immutable inputs here.
        if case["control"] == "facade_owner":
            verify_manifest(Path(args.manifest), REPO_ROOT)
        record = _serve_case(case, read_json(Path(args.manifest)), output_dir,
                             args.repetitions, args.serve_seconds,
                             deadline, oracle_text)
        cases.append(record)
        if oracle_text is None and record["requests"]:
            # The first request is a warmup; use the first measured native-pair
            # response as the deterministic output oracle for all subsequent runs.
            measured = next((item for item in record["requests"]
                             if item["kind"] == "measured"), None)
            if measured is not None:
                oracle_text = measured["response"].get("content")
        if record["status"] == "FAIL":
            failure = record["failure"]
            break

    divergence_rows = []
    if oracle_text is not None:
        oracle_hash = hashlib.sha256(oracle_text.encode("utf-8")).hexdigest()
        for record in cases:
            for request in record["requests"]:
                content = request["response"].get("content", "")
                observed_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
                request["exact_response_match"] = content == oracle_text
                if content != oracle_text:
                    divergence_rows.append({
                        "case": record["id"], "kind": request["kind"],
                        "ordinal": request["ordinal"],
                        "expected_sha256": oracle_hash,
                        "observed_sha256": observed_hash,
                    })
            record["response_equivalence"] = (
                "MATCH" if all(item["exact_response_match"] for item in record["requests"])
                else "DIVERGED")
            record["exact_output_oracle"] = {
                "source_case": CASES[0]["id"], "content": oracle_text,
                "sha256": oracle_hash}
    else:
        divergence_rows.append({"status": "NO_NATIVE_REFERENCE_OUTPUT"})
    completed = time.monotonic()
    status = "PASS" if failure is None and len(cases) == len(CASES) else "FAIL"
    response_equivalence = ("MATCH" if oracle_text is not None and not divergence_rows
                            else "DIVERGED" if oracle_text is not None
                            else "NO_REFERENCE")
    result = {
        "schema": SCHEMA, "status": status,
        "failure": failure,
        "source_provenance": manifest_info,
        "workload": WORKLOAD,
        "gpu_scope": "one controller-admitted NVLink V100 pair; unchanged 18.55 GB resident model",
        "measurement_contract": {
            "warmup_requests_per_server": 1,
            "measured_requests_per_server": args.repetitions,
            "fixed_generated_tokens": GENERATION_TOKENS,
            "output_equivalence": "raw UTF-8 outputs are compared against the first native-pair measured response; any divergence is recorded and withholds comparative performance interpretation",
            "graphs": "enabled for all three cases",
            "pair_compute": "disabled in V200 anchor baseline; enabled only in V200 pair-compute case",
            "capacity": "each process must preserve full layer and embedding GPU residency, pair owner ledger cleanup, and physical-memory guards",
            "interpretation": "bounded single-model comparison; three measured requests characterize this prompt and run, not broad application throughput",
        },
        "request": generation_request(),
        "cases": cases,
        "response_equivalence": response_equivalence,
        "response_divergences": divergence_rows,
        "performance_interpretation": (
            "WITHHELD_OUTPUT_DIVERGENCE_REQUIRES_REVIEW" if response_equivalence != "MATCH"
            else "DESCRIPTIVE_ONLY_THREE_SAMPLES_ONE_PROMPT"),
        "elapsed_seconds": completed - started,
        "configured_max_bundle_seconds": args.max_total_seconds,
        "case_service_seconds": args.serve_seconds,
        "raw_timing_fields": ["wall_milliseconds", "server.prompt_ms", "server.predicted_ms",
                              "server.prompt_per_second", "server.predicted_per_second"],
    }
    result_path = output_dir / "comparison.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
                           encoding="utf-8")
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--verify-binaries", action="store_true",
                        help="hash all qualified binaries and frozen consumer inputs; CPU-only")
    parser.add_argument("--run", action="store_true",
                        help="run the finite consumer comparison under CUDA controller admission")
    parser.add_argument("--execute", action="store_true",
                        help="required acknowledgement for live GPU execution")
    parser.add_argument("--output-dir", type=Path,
                        help="new or empty directory for per-case and aggregate result JSON")
    parser.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)
    parser.add_argument("--serve-seconds", type=int, default=DEFAULT_SERVICE_SECONDS)
    parser.add_argument("--max-total-seconds", type=int, default=DEFAULT_BUNDLE_SECONDS)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-control", choices=("native_pair", "facade_owner"), help=argparse.SUPPRESS)
    parser.add_argument("--worker-output-dir", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-port", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--worker-serve-seconds", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--worker-preload", action="append", default=[], help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.worker:
            if not args.worker_control or not args.worker_output_dir or not args.worker_port:
                raise ComparisonError("internal worker arguments are incomplete")
            return _worker_main(args)
        if args.verify_binaries:
            print(json.dumps(verify_manifest(args.manifest, REPO_ROOT), indent=2, sort_keys=True))
            return 0
        if not args.run or not args.execute:
            raise ComparisonError("live mode requires both --run and --execute; use --verify-binaries for CPU preflight")
        if args.output_dir is None:
            raise ComparisonError("live mode requires --output-dir")
        if args.repetitions < 3 or args.repetitions > 5:
            raise ComparisonError("choose 3 to 5 measured repetitions")
        if args.serve_seconds < 30 or args.serve_seconds > MAX_SERVICE_SECONDS:
            raise ComparisonError(f"--serve-seconds must be in [30, {MAX_SERVICE_SECONDS}]")
        if args.max_total_seconds < 90 or args.max_total_seconds > MAX_BUNDLE_SECONDS:
            raise ComparisonError(f"--max-total-seconds must be in [90, {MAX_BUNDLE_SECONDS}]")
        signal.signal(signal.SIGTERM, lambda _signum, _frame: (_ for _ in ()).throw(KeyboardInterrupt()))
        result = run_comparison(args)
        print(json.dumps({"status": result["status"], "failure": result["failure"],
                          "result": str(args.output_dir.resolve() / "comparison.json"),
                          "elapsed_seconds": result["elapsed_seconds"]}, sort_keys=True))
        return 0 if result["status"] == "PASS" else 1
    except KeyboardInterrupt:
        print("comparison interrupted; active worker received bounded cleanup signal", file=sys.stderr)
        return 130
    except (ComparisonError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"comparison refused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
