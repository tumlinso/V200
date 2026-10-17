#!/usr/bin/env python3
"""Check retained pair-compute consumer evidence without starting CUDA."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/capacity"))
import verify_capacity as capacity


def verify(run, require_experts=False, require_graphs=False,
           require_extended_experts=False):
    read = capacity._json
    require = capacity._require
    controller = read(run.with_name(run.name + ".controller-result.json"))
    evidence = read(run.with_name(run.name + ".controller-evidence.json"))["evidence"]
    execution = read(run / "execution.json")
    release = read(run.with_name(run.name + ".host-release.json"))
    lease = read(Path(controller["lease_receipt"]))
    require(controller.get("ok") is True and controller.get("returncode") == 0,
            "controller run failed")
    require(evidence["id"] == controller["evidence_id"] and
            evidence["status"] == "succeeded" and evidence["valid"] == 1 and
            evidence["contaminated"] == 0 and evidence["summary"]["valid"] is True,
            "invalid controller evidence")
    require(execution["bundle"] == "B03" and execution["status"] == "CHILD_COMPLETED" and
            execution["returncode"] == 0, "bounded child did not complete")
    require(not any(execution[key] for key in
                    ("timed_out", "interrupted", "orphaned_children", "termination_uncertain")),
            "uncertain process cleanup")
    require(release["state"] == "released" and release["resources"] == [] and
            release["id"] == lease["owner_id"] and
            release["pid"] == lease["pid"] == execution["lease"]["owner_pid"],
            "host resources were not released by the admitted owner")
    pair = execution["lease"]["visible_uuids"]
    require(sorted(pair) == sorted(item.removeprefix("accelerator:")
                                   for item in lease["resource_ids"]), "wrong admitted pair")
    for name in ("stdout", "stderr"):
        require(hashlib.sha256((run / (name + ".txt")).read_bytes()).hexdigest() ==
                execution[name + "_sha256"], "bounded output hash mismatch")
    cases = list((run / "consumer").glob("resident_capacity-facade_owner-*.json"))
    require(len(cases) == 1, "expected one full-resident facade case")
    fixture, _ = capacity.consumer.load_documents()
    result = capacity._verify_case(cases[0], "facade_owner", fixture,
                                   capacity.consumer.sha256_file(capacity.consumer.FIXTURE_PATH), pair)
    case = read(cases[0])
    records = [json.loads(line) for line in
               Path(case["v200_profile_telemetry"]["path"]).read_text().splitlines() if line.strip()]
    operations = [r for r in records if r["schema"] == "v200-operator-ledger/v1"]
    require(operations and all(r["status"] == 0 for r in operations),
            "operator launch or verification failed")
    summaries = [r for r in operations if r["event"] == "q4k_mmvq_process_exit"]
    require(len(summaries) == 1, "missing clean operator exit summary")
    totals = summaries[0]
    require(totals["anchor_rows"] > 0 and totals["peer_rows"] > 0 and
            totals["anchor_local_calls"] > 0 and
            totals["peer_owner_calls"] + totals["split_calls"] > 0,
            "both physical compute domains did not perform useful consumer work")
    verified = [r for r in operations if r["event"] == "q4k_mmvq_verification"]
    require(verified and all(r["verified_rows"] > 0 and r["max_abs_error"] == 0
                            for r in verified), "no completed exact native-output comparison")
    for route, count in (("peer-owner", totals["peer_owner_calls"]), ("split", totals["split_calls"])):
        require(not count or any(r["route"] == route for r in verified),
                "executed route lacks a completed native-output comparison: " + route)
    expert_verified = [r for r in operations if r["event"] == "q4k_mmvq_expert_verification"]
    if require_experts:
        require(totals.get("expert_peer_owner_calls", 0) > 0 and
                totals.get("expert_peer_rows", 0) > 0 and
                totals.get("expert_verification_failures") == 0,
                "no successful real-consumer expert execution on the peer")
        require(expert_verified and all(r["status"] == 0 and r["max_abs_error"] == 0 and
                                        r.get("expert_verified_outputs", 0) > 0
                                        for r in expert_verified),
                "expert route lacks a completed exact native-output comparison")
    q6_summaries = [r for r in operations
                    if r["event"] == "q6k_mmvq_expert_process_exit"]
    q6_verifications = [r for r in operations
                        if r["event"] == "q6k_mmvq_expert_verification"]
    if require_extended_experts:
        require(totals.get("expert_extended_peer_owner_calls", 0) > 0 and
                totals.get("expert_extended_plain_verified_calls", 0) > 0 and
                totals.get("expert_extended_plain_verified_outputs", 0) > 0 and
                totals.get("expert_verification_failures") == 0,
                "no bitwise-verified 768-2047-column plain Q4_K expert route")
        require(any(r["route"] == "peer-bank-plain" and
                    r.get("expert_extended_plain_verified_calls", 0) > 0 and
                    r["status"] == 0 and r["max_abs_error"] == 0
                    for r in expert_verified),
                "extended Q4_K route lacks its own exact native-output comparison")
        require(len(q6_summaries) == 1, "missing Q6_K expert exit summary")
        q6 = q6_summaries[0]
        require(q6.get("q6_expert_peer_owner_calls", 0) > 0 and
                q6.get("q6_expert_plain_peer_owner_calls", 0) > 0 and
                q6.get("q6_expert_plain_verified_calls", 0) > 0 and
                q6.get("q6_expert_plain_verified_outputs", 0) > 0 and
                q6.get("q6_expert_verification_failures") == 0,
                "no bitwise-verified plain Q6_K expert execution on the peer")
        require(any(r.get("q6_expert_plain_verified_calls", 0) > 0 and
                    r["status"] == 0 and r["max_abs_error"] == 0
                    for r in q6_verifications),
                "Q6_K route lacks its own exact native-output comparison")
        # If a fused expert family actually ran on the peer, it must have its
        # independent first-family exact comparison as well.
        require(q6.get("q6_expert_fused_peer_owner_calls", 0) == 0 or
                q6.get("q6_expert_fused_verified_calls", 0) > 0,
                "observed fused Q6_K route lacks an exact comparison")
        require(totals.get("expert_fused_peer_owner_calls", 0) == 0 or
                totals.get("expert_fused_verified_calls", 0) > 0,
                "observed fused Q4_K route lacks an exact comparison")
    graph_records = [r for r in records if r.get("event", "").startswith("graph_")]
    graph_replays = [r for r in graph_records if r["event"] == "graph_replay" and
                     r.get("graph_rewritten_nodes", 0) > 0]
    if require_graphs:
        require(case.get("graph_capture_evidence", {}).get("status") == "PASS_CAPTURE_AND_REPLAY",
                "consumer did not exercise graph capture and replay")
        require(graph_records and all(r.get("graph_status") == 0 for r in graph_records),
                "graph capture, rewrite, instantiate or replay failed")
        require(graph_replays, "no replay of a graph with peer-context kernel nodes")
        for replay in graph_replays:
            require(any(r["event"] in ("graph_instantiate", "graph_exec_update") and
                        r["graph_handle"] == replay["graph_handle"] and
                        r["graph_rewritten_nodes"] == replay["graph_rewritten_nodes"]
                        for r in graph_records), "peer replay lacks executable provenance")
    return {"status": "PASS_EXTENDED_EXPERTS" if require_extended_experts else "PASS_V200_10" if require_graphs else "PASS_V200_09" if require_experts else "PASS_V200_08", "controller_evidence_id": evidence["id"],
            "physical_uuids": pair, "capacity": result, "operator": totals,
            "verified_routes": [r["route"] for r in verified + expert_verified],
            "peer_graph_replay_samples": len(graph_replays),
            "maximum_peer_nodes_per_graph": max((r["graph_rewritten_nodes"] for r in graph_replays), default=0),
            "host_resources_released": True, "speedup": "NOT_ESTABLISHED"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "evidence/B03/operator-r2")
    parser.add_argument("--require-experts", action="store_true")
    parser.add_argument("--require-extended-experts", action="store_true")
    parser.add_argument("--require-graphs", action="store_true")
    args = parser.parse_args()
    try:
        print(json.dumps(verify(args.run_dir.resolve(), args.require_experts,
                                args.require_graphs, args.require_extended_experts),
                          indent=2))
    except (KeyError, OSError, TypeError, ValueError) as error:
        print(json.dumps({"status": "FAIL", "error": str(error)}), file=sys.stderr)
        sys.exit(1)
