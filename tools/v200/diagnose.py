#!/usr/bin/env python3
"""Read-only host diagnosis of the pinned V200 two-device deployment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
PROFILE = ROOT / "docs/v200/multipair-profile.json"
FIXTURE = ROOT / "benchmarks/fixtures/llama-server-multipair.json"
CUDA_ROOT = Path("/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/cuda/12.9")


class DiagnosisError(RuntimeError):
    pass


def _json(path: Path) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DiagnosisError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise DiagnosisError(f"expected a JSON object in {path}")
    return value, raw


def load_profile(profile_path: Path = PROFILE,
                 fixture_path: Path = FIXTURE) -> tuple[dict[str, Any], dict[str, Any], str, str]:
    profile, profile_raw = _json(profile_path)
    fixture, fixture_raw = _json(fixture_path)
    fixture_sha = hashlib.sha256(fixture_raw).hexdigest()
    if profile.get("schema") != "v200-multi-pair-profile/v1":
        raise DiagnosisError("unsupported multipair profile schema")
    if fixture.get("schema") != "v200-llama-server-multi-pair-fixture/v1":
        raise DiagnosisError("unsupported multipair fixture schema")
    if profile.get("fixture_sha256") != fixture_sha:
        raise DiagnosisError("fixture SHA-256 differs from the pinned profile")
    if (profile_path.parent / str(profile.get("fixture", ""))).resolve() != fixture_path.resolve():
        raise DiagnosisError("profile fixture path differs from the supplied fixture")
    pairs = fixture.get("topology", {}).get("logical_pairs")
    if len(profile.get("pairs", [])) != 2 or profile.get("pairs") != pairs:
        raise DiagnosisError("profile and fixture logical-pair mappings differ")
    return profile, fixture, hashlib.sha256(profile_raw).hexdigest(), fixture_sha


def select_build_dir(root: Path, profile: dict[str, Any]) -> Path:
    if profile.get("build", {}).get("directory") != "build-multipair":
        raise DiagnosisError("build output must stay in build-multipair/")
    root = root.resolve()
    target = root / "build-multipair"
    if target.is_symlink() or target.resolve(strict=False) == (root / "build").resolve(strict=False):
        raise DiagnosisError("unsafe multipair build directory")
    return target


def _run(args: list[str], cwd: Path | None = None) -> str | None:
    try:
        result = subprocess.run(args, cwd=cwd, text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, timeout=20, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _sha(path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
    except OSError:
        return None


def diagnose() -> dict[str, Any]:
    profile, fixture, profile_sha, fixture_sha = load_profile()
    build_dir = select_build_dir(ROOT, profile)
    head = _run(["git", "rev-parse", "HEAD"], ROOT)
    dirty = _run(["git", "status", "--porcelain", "--untracked-files=normal"], ROOT)
    source = Path(profile["source"]["repository"])
    source_head = _run(["git", "-C", str(source), "rev-parse", "HEAD"])
    source_dirty = _run(["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"])

    cache: dict[str, str] = {}
    try:
        for line in (build_dir / "CMakeCache.txt").read_text().splitlines():
            if "=" in line and not line.startswith(("#", "//")):
                key, value = line.split("=", 1)
                cache[key.split(":", 1)[0]] = value
    except OSError:
        pass
    cache_check = {key: {"actual": cache.get(key), "matches": cache.get(key) == expected}
                   for key, expected in {
                       "CMAKE_HOME_DIRECTORY": str(ROOT),
                       "CMAKE_CXX_COMPILER": "/usr/bin/g++",
                       "CUDAToolkit_ROOT": str(CUDA_ROOT),
                       "V200_CUDA_TOOLKIT_ROOT": str(CUDA_ROOT),
                       "BUILD_TESTING": "ON",
                       "V200_ENABLE_GPU_TESTS": "OFF",
                   }.items()}
    nvcc = _run([str(CUDA_ROOT / "bin/nvcc"), "--version"])
    nvcc_match = re.search(r"release\s+(\d+\.\d+)", nvcc or "")

    qualification = profile.get("qualification", {})
    bundle_path = ROOT / str(qualification.get("bundle", ""))
    try:
        bundle = json.loads(bundle_path.read_text())
    except (OSError, json.JSONDecodeError):
        bundle = {}
    accepted_dsos = {str(Path(x["path"]).resolve()): x.get("sha256")
                     for x in bundle.get("v200_runtime_artifacts", []) if isinstance(x, dict)}
    dsos = []
    for relative in profile.get("runtime_libraries", []):
        path = ROOT / relative
        actual = _sha(path)
        accepted = accepted_dsos.get(str(path.resolve()))
        dsos.append({"path": str(path), "sha256": actual,
                     "accepted_b06_sha256": accepted,
                     "matches_b06": bool(actual and accepted and actual == accepted)})

    binary = profile["binary"]
    binary_path = Path(binary["path"])
    binary_sha = _sha(binary_path)
    closure = []
    for item in profile.get("runtime_closure", []):
        path = Path(item["path"])
        actual = _sha(path)
        closure.append({"role": item.get("role"), "path": str(path),
                        "sha256": actual, "matches_profile": actual == item.get("sha256")})
    ldd = _run(["ldd", str(binary_path)]) if binary_path.is_file() else None
    cuda_links = [line.strip() for line in (ldd or "").splitlines()
                  if re.search(r"lib(cudart|cuda|cublas|nccl)", line)]
    unresolved = [line.strip() for line in (ldd or "").splitlines() if "not found" in line]

    model = Path(fixture["model"]["path"])
    try:
        model_bytes = model.stat().st_size
    except OSError:
        model_bytes = None
    pairs = [{"visible_ordinal": p["pair_index"], "logical_device_id": p["logical_device_id"],
              "anchor_uuid": p["anchor_uuid"], "peer_uuid": p["peer_uuid"],
              "b01_evidence_id": p["b01_evidence_id"],
              "owner_budgets_mib": fixture["memory_guard"]["owner_budgets_mib_per_v200_pair"]}
             for p in profile["pairs"]]

    return {
        "schema": "v200-deployment-diagnosis/v1",
        "gpu_initialized": False,
        "repository": {"path": str(ROOT), "head": head,
                       "dirty": bool(dirty), "status": (dirty or "").splitlines()[:20]},
        "profile": {"path": str(PROFILE), "sha256": profile_sha,
                    "fixture_sha256": fixture_sha},
        "accepted_b06": {"status": qualification.get("status"),
                         "bundle_status": bundle.get("status"),
                         "controller_evidence_id": qualification.get("controller_evidence_id"),
                         "bundle": str(bundle_path),
                         "finite_service": qualification.get("finite_service")},
        "logical_devices": pairs,
        "build": {"directory": str(build_dir), "separate_from_build": True,
                  "cuda_toolkit_root": str(CUDA_ROOT), "profile_cuda_version": profile["build"].get("cuda_toolkit_version"),
                  "nvcc_release": nvcc_match.group(1) if nvcc_match else None,
                  "cmake_version": (_run(["cmake", "--version"]) or "").splitlines()[:1],
                  "cache": cache_check},
        "consumer": {"source_repository": str(source), "source_head": source_head,
                     "source_matches_profile": source_head == profile["source"].get("commit"),
                     "source_tracked_clean": source_dirty == "",
                     "binary": str(binary_path), "binary_sha256": binary_sha,
                     "binary_matches_profile": binary_sha == binary.get("sha256"),
                     "model": str(model), "model_bytes": model_bytes,
                     "model_size_matches_fixture": model_bytes == fixture["model"].get("size_bytes"),
                     "runtime_closure": closure,
                     "closure_matches_profile": bool(closure) and all(x["matches_profile"] for x in closure),
                     "resolved_cuda_libraries": cuda_links, "ldd_unresolved": unresolved},
        "v200_dsos": dsos,
        "v200_dsos_match_accepted_b06": bool(dsos) and all(x["matches_b06"] for x in dsos),
        "performance": {"evidence": "evidence/B07/extended-r1/comparison/comparison.json",
                        "native_four_gpu_decode_median_tokens_per_second": 124.951,
                        "single_v200_extended_decode_median_tokens_per_second": 64.095,
                        "note": "Three-request descriptive sample; current V200 path is slower than the native control."},
        "limitations": [
            "The logical devices are process-local; system CUDA/NVML/PCI enumeration is unchanged.",
            "The pinned llama path is qualified; arbitrary CUDA kernels are not transparently distributed.",
            "The two-device profile keeps graphs disabled and stages cross-pair copies through host memory.",
            "This fixture uses 4 GiB plus 12 GiB per pair; no general 32 GiB capacity is promised.",
            "Performance evidence is one model/prompt with three measured requests, not sustained throughput.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit full paths and hashes as JSON")
    args = parser.parse_args(argv)
    try:
        report = diagnose()
    except (DiagnosisError, KeyError, TypeError, ValueError) as exc:
        print(f"V200 diagnosis failed: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print("V200 host diagnosis (no CUDA device initialization)")
        print(f"HEAD {report['repository']['head']} (dirty={report['repository']['dirty']})")
        print(f"Profile {report['profile']['sha256']}")
        print(f"Accepted B06 correctness: {report['accepted_b06']['status']}; 10-second service smoke: {report['accepted_b06']['finite_service'].get('status')}")
        print(f"CUDA toolkit {report['build']['profile_cuda_version']} (nvcc {report['build']['nvcc_release']}); build {report['build']['directory']}")
        for pair in report["logical_devices"]:
            print(f"CUDA{pair['visible_ordinal']}: {pair['anchor_uuid']} + {pair['peer_uuid']}")
        print(f"Pinned llama closure matches profile: {report['consumer']['closure_matches_profile']}; current V200 DSOs match hashes recorded by B06: {report['v200_dsos_match_accepted_b06']}")
        print(f"Decode medians: native 124.951, V200 extended 64.095 tokens/s (descriptive B07 sample)")
        print("See docs/v200/deployment.md for launch, fallback, and limits.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
