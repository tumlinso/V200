#!/usr/bin/env python3
"""Validate the portable shape of the checked-in V200 consumer profile."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


SHA256 = re.compile(r"^[0-9a-f]{64}$")
GIT_COMMIT = re.compile(r"^[0-9a-f]{40}$")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate(path: Path) -> None:
    profile = json.loads(path.read_text(encoding="utf-8"))
    require(profile.get("schema") == "v200-source-profile/v1", "unsupported profile schema")
    require(profile.get("consumer", {}).get("name") == "llama-server", "consumer is not pinned")

    source = profile["consumer"]["source"]
    require(GIT_COMMIT.fullmatch(source.get("commit", "")) is not None,
            "source commit must be a full Git object id")
    require(source.get("worktree") == "clean", "source worktree must be recorded clean")

    binary = profile["consumer"]["binary"]
    require(binary.get("format") == "ELF64-x86_64", "unexpected binary format")
    require(SHA256.fullmatch(binary.get("sha256", "")) is not None,
            "consumer binary hash must be SHA-256")
    artifacts = profile["consumer"].get("cuda_artifacts", [])
    require(any(item.get("role") == "runtime_backend" for item in artifacts),
            "CUDA backend artifact must be pinned separately from the executable")
    require(all(SHA256.fullmatch(item.get("sha256", "")) for item in artifacts),
            "every CUDA artifact must have a SHA-256")

    model = profile.get("model_fixture", {})
    require(model.get("status") == "not_selected", "model fixture status must be explicit")
    require(model.get("path") is None and model.get("sha256") is None,
            "an unselected model must not have guessed identity fields")
    require(profile.get("qualification", {}).get("status") == "NOT_RUN",
            "source inventory must not claim consumer qualification")


def main(argv: list[str]) -> int:
    path = Path(argv[1]) if len(argv) > 1 else Path(__file__).parents[2] / "docs/v200/source-profile.json"
    try:
        validate(path)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"source profile check failed: {exc}", file=sys.stderr)
        return 1
    print(f"source profile schema OK: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
