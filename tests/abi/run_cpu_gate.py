#!/usr/bin/env python3
"""Reproducible CPU-only build and ABI closure gate for the frontdoor."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import json


ROOT = Path(__file__).resolve().parents[2]
BUILD = ROOT / "build-frontdoor"
DEFAULT_CUDA_ROOT = Path(
    "/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/cuda/12.9/targets/x86_64-linux"
)


def run(argv: list[str]) -> None:
    print("+", " ".join(argv), flush=True)
    subprocess.run(argv, cwd=ROOT, check=True)


def main() -> int:
    cuda_root = os.environ.get("V200_CUDA_TOOLKIT_ROOT", str(DEFAULT_CUDA_ROOT))
    run([
        "cmake", "-S", "src/frontdoor", "-B", str(BUILD),
        f"-DV200_CUDA_TOOLKIT_ROOT={cuda_root}", "-DBUILD_TESTING=ON",
    ])
    run(["cmake", "--build", str(BUILD), "--parallel", "4"])
    run(["ctest", "--test-dir", str(BUILD), "--output-on-failure"])
    with (ROOT / "docs/v200/abi-profile.json").open(encoding="utf-8") as stream:
        json.load(stream)
    print("ABI profile JSON is valid", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
