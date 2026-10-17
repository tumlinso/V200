#!/usr/bin/env python3
"""Verify retained B01 artifacts without importing CUDA or executing devices."""

import hashlib
import json
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify(root):
    evidence = root / "evidence/foundation"
    controller = json.loads((evidence / "B01.controller-evidence.json").read_text())["evidence"]
    result = json.loads((evidence / "B01.controller-result.json").read_text())
    execution = json.loads((evidence / "B01/execution.json").read_text())
    fixture = json.loads((evidence / "B01/stdout.txt").read_text())
    released = json.loads((evidence / "B01.host-release.json").read_text())
    binding = json.loads((root / "scripts/qualification/B01.binding.json").read_text())
    lease = json.loads(Path(result["lease_receipt"]).read_text())
    summary = controller["summary"]
    require(result["ok"] is True and summary["valid"] is True, "controller did not accept run")
    require(result["evidence_id"] == controller["id"], "controller evidence identity differs")
    require(not controller["contaminated"], "contaminated controller evidence")
    require(summary["resource_samples"]["quiescence"]["uncontaminated"], "idle proof missing")
    require(not summary["resource_samples"]["after"].get("foreign_processes"), "foreign GPU activity")
    require(execution["returncode"] == 0, "fixture failed")
    for field in ("timed_out", "interrupted", "orphaned_children", "termination_uncertain"):
        require(execution[field] is False, "unclean execution: " + field)
    for stream in ("stdout", "stderr"):
        data = (evidence / f"B01/{stream}.txt").read_bytes()
        require(hashlib.sha256(data).hexdigest() == execution[f"{stream}_sha256"], "changed fixture output")
    require(fixture["status"] == "GPU_PASS" and fixture["errors"] == 0, "pair correctness failed")
    require(fixture["checked_u32_per_device"] > 0, "missing both-device verification")
    require(fixture["qualification_only"] is True, "unexpected fixture claim")
    require(released["state"] == "released" and released["resources"] == [], "host reservation not released")
    require(str(root) == released["project_root"], "release belongs to another project")
    require(execution["lease"]["receipt"] == result["lease_receipt"], "lease receipt path differs")
    require(lease["format"] == "CUDA-FOREGROUND-LEASE/1", "not a canonical receipt")
    require(lease["owner_id"] == released["id"], "released owner differs from admitted owner")
    require(lease["pid"] == released["pid"] == execution["lease"]["owner_pid"], "controller PID differs")
    require(lease["project_root"] == str(root), "receipt belongs to another project")
    require(sorted(lease["resource_ids"]) == sorted("accelerator:" + u for u in binding["gpu_uuids"]), "receipt resources differ")
    require(sorted(execution["lease"]["visible_uuids"]) == sorted(binding["gpu_uuids"]), "pair differs")
    for artifact in controller["artifacts"]:
        require(hashlib.sha256(Path(artifact["path"]).read_bytes()).hexdigest() == artifact["content_hash"], "controller artifact changed")
    for binary in summary["binaries"]:
        require(hashlib.sha256(Path(binary["path"]).read_bytes()).hexdigest() == binary["sha256"], "binary changed")
    sources = {}
    snapshot = Path(controller["snapshot"]["source_root"])
    for name in binding["source_paths"]:
        data = (root / name).read_bytes()
        require(data == (snapshot / name).read_bytes(), "source differs from controller snapshot")
        sources[name] = hashlib.sha256(data).hexdigest()
    return {
        "status": "GPU_PASS",
        "evidence_id": controller["id"],
        "physical_uuids": binding["gpu_uuids"],
        "toolkit_version": summary["toolchain"]["version"],
        "fixture": fixture,
        "source_sha256": sources,
        "binaries": summary["binaries"],
        "host_reservation": "released",
        "scope": "explicit disjoint writes, joined completion, both-device full-view verification",
        "arbitrary_kernel_transparency": False,
        "consumer_compatibility": "NOT_RUN",
        "capacity_beyond_one_gpu": "NOT_RUN",
        "performance": "NOT_MEASURED",
    }


if __name__ == "__main__":
    print(json.dumps(verify(Path(__file__).resolve().parents[2]), indent=2))
