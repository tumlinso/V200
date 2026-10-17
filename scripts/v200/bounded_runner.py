#!/usr/bin/env python3
"""Use the preserved bootstrap lease validator with a graceful child stop.

The upstream project bootstrap is loaded from its exact repository path and
continues to own argument parsing, active-lease validation, ancestry/device
checks, bundle environment, result schema and status policy. Only its
``run_child`` implementation is replaced so a finite llama runner can clean
its separately-owned server process group before the final SIGKILL escalation.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import ModuleType
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
BOOTSTRAP_RUNNER = REPO_ROOT / "planning/v200-bootstrap/tools/run_bounded.py"
TERM_GRACE_SECONDS = 15.0
KILL_REAP_SECONDS = 2.0


def _load_bootstrap() -> ModuleType:
    if not BOOTSTRAP_RUNNER.is_file():
        raise RuntimeError(f"preserved bootstrap runner is missing: {BOOTSTRAP_RUNNER}")
    spec = importlib.util.spec_from_file_location("v200_preserved_run_bounded", BOOTSTRAP_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import preserved bootstrap runner from {BOOTSTRAP_RUNNER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _signal_group(pgid: int, sig: int) -> bool:
    try:
        os.killpg(pgid, sig)
        return True
    except ProcessLookupError:
        return False


def run_child(cmd: list[str], seconds: float, out: Path,
              env: dict[str, str] | None = None) -> dict[str, Any]:
    """Preserve the bootstrap result shape, allowing 15s for child cleanup."""
    out.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    timed_out = False
    uncertain = False
    interrupted = False
    with (out / "stdout.txt").open("wb") as stdout, (out / "stderr.txt").open("wb") as stderr:
        proc = subprocess.Popen(cmd, stdout=stdout, stderr=stderr,
                                start_new_session=True, env=env)
        try:
            proc.wait(timeout=seconds)
        except (subprocess.TimeoutExpired, KeyboardInterrupt) as exc:
            timed_out = isinstance(exc, subprocess.TimeoutExpired)
            interrupted = not timed_out
            _signal_group(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=TERM_GRACE_SECONDS)
            except subprocess.TimeoutExpired:
                _signal_group(proc.pid, signal.SIGKILL)
                try:
                    proc.wait(timeout=KILL_REAP_SECONDS)
                except subprocess.TimeoutExpired:
                    uncertain = True

    # Keep the original wrapper's owned-group orphan audit and status fields.
    orphaned = False
    try:
        os.killpg(proc.pid, 0)
        orphaned = True
    except ProcessLookupError:
        pass
    if orphaned:
        _signal_group(proc.pid, signal.SIGTERM)
        deadline = time.monotonic() + TERM_GRACE_SECONDS
        while time.monotonic() < deadline:
            try:
                os.killpg(proc.pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.02)
        else:
            _signal_group(proc.pid, signal.SIGKILL)
            uncertain = True
    return {
        "returncode": proc.returncode,
        "timed_out": timed_out,
        "interrupted": interrupted,
        "orphaned_children": orphaned,
        "termination_uncertain": uncertain,
        "wall_seconds": time.monotonic() - start,
        "stdout_sha256": hashlib.sha256((out / "stdout.txt").read_bytes()).hexdigest(),
        "stderr_sha256": hashlib.sha256((out / "stderr.txt").read_bytes()).hexdigest(),
    }


def main() -> int:
    try:
        bootstrap = _load_bootstrap()
    except Exception as exc:
        print(f"bounded launcher unavailable: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    # `main` retains the original SIGTERM handler and invokes this adapter via
    # its ordinary run_child lookup after validating the controller receipt.
    bootstrap.run_child = run_child
    return int(bootstrap.main())


if __name__ == "__main__":
    raise SystemExit(main())
