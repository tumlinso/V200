#!/usr/bin/env python3
"""Configure, build, and run the injected-dispatch memory tests without a GPU."""

from pathlib import Path
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[2]
BUILD = Path(tempfile.gettempdir()) / "v200-memory-required-gate"


def run(*argv: str) -> None:
    subprocess.run(argv, cwd=ROOT, check=True)


def main() -> None:
    run(
        "cmake",
        "-S",
        "src/memory",
        "-B",
        str(BUILD),
        "-DV200_MEMORY_BUILD_TESTS=ON",
        "-DCMAKE_BUILD_TYPE=Debug",
    )
    run("cmake", "--build", str(BUILD), "--parallel", "2")
    run("ctest", "--test-dir", str(BUILD), "--output-on-failure")


if __name__ == "__main__":
    main()
