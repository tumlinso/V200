#!/usr/bin/env python3
"""Profile one warmed 64-token resident llama-server decode case under admission.

Nsight Systems should wrap this command. It marks only the first measured HTTP
request with NVTX; post-hoc nsys stats can then filter CUDA activity to the
request window while the existing consumer worker retains model, GPU, ledger,
and cleanup guards.
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import json
import sys
from pathlib import Path
from typing import Any, Callable


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "benchmarks/comparison"))
import run_extended_comparison as extended  # noqa: E402
import run_pair_comparison as pair  # noqa: E402


TRACE_SCHEMA = "v200-decode-trace-case/v1"
WARMUP_REQUESTS = 1
MEASURED_REQUESTS = 1
SERVICE_SECONDS = 15
MAX_BUNDLE_SECONDS = 45


class TraceCaseError(RuntimeError):
    pass


class NvtxRange:
    """Small default-domain NVTX push/pop wrapper used only by this trace run."""

    def __init__(self) -> None:
        library = ctypes.util.find_library("nvToolsExt") or "libnvToolsExt.so.1"
        try:
            self._lib = ctypes.CDLL(library)
        except OSError as exc:
            raise TraceCaseError(f"cannot load NVTX library {library}: {exc}") from exc
        self._lib.nvtxRangePushA.argtypes = [ctypes.c_char_p]
        self._lib.nvtxRangePushA.restype = ctypes.c_int
        self._lib.nvtxRangePop.argtypes = []
        self._lib.nvtxRangePop.restype = ctypes.c_int

    def push(self, name: str) -> None:
        if self._lib.nvtxRangePushA(name.encode("utf-8")) < 0:
            raise TraceCaseError(f"NVTX refused to open range {name!r}")

    def pop(self) -> None:
        if self._lib.nvtxRangePop() < 0:
            raise TraceCaseError("NVTX refused to close decode range")


def mark_first_measured_request(
    original: Callable[..., Any], marker: NvtxRange, range_name: str,
) -> tuple[Callable[..., Any], Callable[[], int], Callable[[], int]]:
    """Return a client wrapper that brackets call 2: one warmup, then sample 1."""
    calls = 0
    marked = 0

    def invoke(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls, marked
        is_first_measured = calls == WARMUP_REQUESTS
        calls += 1
        if is_first_measured:
            marker.push(range_name)
            marked += 1
        try:
            return original(*args, **kwargs)
        finally:
            if is_first_measured:
                marker.pop()

    return invoke, lambda: calls, lambda: marked


def selected_case(case_id: str) -> dict[str, Any]:
    for case in extended.CASES:
        if case["id"] == case_id:
            return dict(case)
    raise TraceCaseError(f"unsupported comparison case {case_id!r}")


def run_case(args: argparse.Namespace) -> dict[str, Any]:
    case = selected_case(args.case)
    marker = NvtxRange()
    range_name = "native_decode" if args.case == "native_pair" else "v200_decode"
    original = pair.consumer._completion_request
    wrapped, request_count, marked_count = mark_first_measured_request(
        original, marker, range_name)
    pair.consumer._completion_request = wrapped
    try:
        result = extended.run_comparison(
            args.output_dir,
            v200_build_dir=args.v200_build_dir,
            case_definitions=(case,),
            repetitions=MEASURED_REQUESTS,
            service_seconds=SERVICE_SECONDS,
            max_bundle_seconds=MAX_BUNDLE_SECONDS,
            prefer_local_hbm=(None if args.prefer_local_hbm is None
                              else args.prefer_local_hbm == "1"),
        )
    finally:
        pair.consumer._completion_request = original

    result["trace_case"] = {
        "schema": TRACE_SCHEMA,
        "case": args.case,
        "case_build_directory": str(extended._selected_build_dir(args.v200_build_dir)),
        "prefer_local_hbm": args.prefer_local_hbm,
        "warmup_requests": WARMUP_REQUESTS,
        "measured_requests": MEASURED_REQUESTS,
        "nvtx_range": range_name,
        "nvtx_domain": "default",
        "nvtx_boundary": "client-side blocking request from before HTTP send through full response receipt",
        "request_calls_observed": request_count(),
        "nvtx_ranges_observed": marked_count(),
        "nsys_filter_hint": f"--filter-nvtx {range_name}",
        "cross_case_equivalence": "not measured by this single-case trace; B07 paired comparison remains the equivalence evidence",
        "profiling_is_not_a_throughput_sample": True,
    }
    if request_count() != WARMUP_REQUESTS + MEASURED_REQUESTS or marked_count() != 1:
        result["status"] = "FAIL"
        result["failure"] = result["failure"] or (
            "trace request sequence did not contain exactly one warmup and one marked measurement"
        )
    wrapper_artifact = extended._artifact("decode_trace_runner", Path(__file__))
    result["source_provenance"]["artifacts"].append(wrapper_artifact)
    result["source_provenance"]["artifact_count"] = len(result["source_provenance"]["artifacts"])
    result["trace_case"]["runner_sha256"] = wrapper_artifact["sha256"]
    result_path = args.output_dir.resolve() / "comparison.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
                           encoding="utf-8")
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true",
                        help="run one controller-admitted 64-token profile case")
    parser.add_argument("--execute", action="store_true",
                        help="required acknowledgement for live GPU work")
    parser.add_argument("--case", choices=("native_pair", "v200_extended"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--v200-build-dir", type=Path, default=extended.BUILD_DIR,
                        help="V200 build artifact root; no build is performed")
    parser.add_argument("--prefer-local-hbm", choices=("0", "1"),
                        help="set V200_PREFER_LOCAL_HBM for this case")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.run or not args.execute:
        print("profile case requires both --run and --execute", file=sys.stderr)
        return 2
    try:
        result = run_case(args)
    except (TraceCaseError, extended.ExtendedComparisonError, pair.ComparisonError,
            pair.consumer.ConsumerError, OSError, ValueError, KeyError, TypeError) as exc:
        print(f"decode trace case refused: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"status": result["status"], "failure": result["failure"],
                      "case": args.case,
                      "result": str((args.output_dir / "comparison.json").resolve()),
                      "trace_case": result["trace_case"]}, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
