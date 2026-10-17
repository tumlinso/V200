#!/usr/bin/env python3
"""Check retained preview evidence and current library identity; never run CUDA."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def read(relative):
    return json.loads((ROOT / relative).read_text())


def released(result, owner, expected_id):
    r, h = read(result), read(owner)
    assert r['ok'] and r['returncode'] == 0 and r['evidence_id'] == expected_id
    assert h['state'] == 'released' and h['resources'] == []


def clean(c):
    assert c['application_status'] == 'PASS' and c['response']['exact_match']
    assert c['server_process']['exit_code'] == 0
    assert c['cleanup']['no_live_group_members']
    assert c['layer_offload_evidence']['status'] == 'PASS'
    assert c['input_embedding_evidence']['status'] == 'PASS'


def verify():
    b = read('evidence/B06/multipair-runs/multipair-20261017T150959Z-75b105fcd3/consumer/bundle.json')
    assert b['status'] == 'PASS' and b['copy_probe']['status'] == 'PASS'
    assert b['v200_summary']['visible_cuda_devices'] == [2]
    assert b['v200_summary']['cleanup_accounting_zero']
    assert len(b['v200_summary']['four_owner_peak_bytes']) == 4
    assert min(b['v200_summary']['four_owner_peak_bytes'].values()) > 0
    assert {c['control'] for c in b['controls']} == {'native_four', 'v200_two'}
    for c in b['controls']:
        clean(c)
        assert c['device_count_evidence']['status'] == 'PASS'
    for item in b['v200_runtime_artifacts']:
        if item['path'].endswith('.so'):
            assert hashlib.sha256(Path(item['path']).read_bytes()).hexdigest() == item['sha256']
    released('evidence/B06/qualification-r5.controller-result.json',
             'evidence/B06/qualification-r5.host-release.json',
             'bdf88682-63b9-4788-8943-b411c03ee1f1')
    s = read('evidence/B06/serving/runs/multipair-20261017T151911Z-e871dbfdf1/consumer/service.json')
    assert s['status'] == 'PASS' and s['control']['service']['status'] == 'PASS'
    clean(s['control'])
    assert all(p['cleanup_accounting_zero'] for p in s['control']['profile_telemetry']['pair_results'].values())
    released('evidence/B06/serving-smoke-r1.controller-result.json',
             'evidence/B06/serving-smoke-r1.host-release.json',
             'd34961d6-ac1d-4998-81e8-24195639c9f6')
    p = read('evidence/B07/extended-r1/comparison/comparison.json')
    assert p['status'] == 'PASS' and p['response_equivalence'] == 'MATCH'
    assert len(p['cases']) == 2
    for c in p['cases']:
        clean(c['consumer'])
        assert c['contract']['status'] == 'PASS' and c['summary']['measured_samples'] == 3
        assert all(r['exact_response_match_to_native_pair'] for r in c['requests'])
    c = p['cases'][1]['consumer']
    payload = b['static_preflight']['resident_tensor_bytes']
    assert payload > max(x['total_bytes'] for x in c['physical_gpu_memory_before_launch'].values())
    assert min(c['v200_profile_telemetry']['peak_owner_used_bytes']) > 0
    assert c['v200_profile_telemetry']['peak_total_owner_used_bytes'] >= payload
    assert c['v200_profile_telemetry']['cleanup_accounting_zero']
    assert c['graph_capture_evidence']['status'] == 'PASS_CAPTURE_AND_REPLAY'
    coverage = p['cases'][1]['contract']['extended_operator_coverage']
    assert coverage['q4_low_column_peer_calls'] > 0 and coverage['q6_peer_calls'] > 0
    released('evidence/B07/extended-r1.controller-result.json',
             'evidence/B07/extended-r1.host-release.json',
             'd5c28e95-2772-474c-86c8-318bc533fc9a')
    subprocess.run([sys.executable, 'tests/operators/verify_consumer.py',
                    '--run-dir', 'evidence/B03/extended-r2', '--require-experts',
                    '--require-graphs', '--require-extended-experts'], cwd=ROOT, check=True,
                   stdout=subprocess.DEVNULL)
    print('PASS restricted preview: real residency, peer compute, graph replay, two logical devices, service and retained releases')
    print('Full V200 acceptance remains open: performance parity and broad CUDA transparency are not established.')


if __name__ == '__main__':
    verify()
