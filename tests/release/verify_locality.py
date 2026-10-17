#!/usr/bin/env python3
"""Validate saved locality evidence and current DSOs without running CUDA."""
import hashlib
from pathlib import Path
from verify_preview import read, clean, released

q = read('evidence/B08/locality-qualification.json')
a = read('evidence/B08/locality-r1/comparison/comparison.json')
b = read('evidence/B08/cache-only-r1/comparison/comparison.json')
assert a['status'] == b['status'] == 'PASS'
assert a['response_equivalence'] == 'MATCH'
reference = a['cases'][0]['requests'][1]['response']['content']
for case in a['cases'] + b['cases']:
    clean(case['consumer'])
    assert case['contract']['status'] == 'PASS'
    assert case['summary']['measured_samples'] == 3
    assert all(r['response']['content'] == reference for r in case['requests'])
for stem, key in [('locality-r1', 'locality_vs_native'), ('cache-only-r1', 'cache_only_ablation')]:
    released(f'evidence/B08/{stem}.controller-result.json',
             f'evidence/B08/{stem}.host-release.json', q['controller_evidence_ids'][key])
c = a['cases'][1]['consumer']
assert c['v200_profile_telemetry']['cleanup_accounting_zero']
assert min(c['v200_profile_telemetry']['peak_owner_used_bytes']) > 0
assert c['graph_capture_evidence']['status'] == 'PASS_CAPTURE_AND_REPLAY'
for artifact in q['source_inputs']:
    if artifact['role'] in {'v200_frontdoor', 'v200_pair_initializer', 'v200_operators'}:
        assert hashlib.sha256(Path(artifact['path']).read_bytes()).hexdigest() == artifact['sha256']
for artifact in read('evidence/B08/frozen-preview-identities.json'):
    assert hashlib.sha256(Path(artifact['path']).read_bytes()).hexdigest() == artifact['sha256']
print('PASS saved locality/ablation correctness, current candidate identity, cleanup/release and unchanged frozen preview')
