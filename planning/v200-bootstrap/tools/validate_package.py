#!/usr/bin/env python3
"""CPU-only package consistency checks, not the canonical Todo validator."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import sys


def check(root: Path, *, require_manifest: bool=True) -> dict:
    root=root.resolve(strict=True);errors=[];checks=0
    def demand(ok: bool, message: str):
        nonlocal checks
        checks+=1
        if not ok:errors.append(message)
    def load(p: str):return json.loads((root/p).read_text())
    descriptor=load('preledger.json');tasks=load('proposed_todos.json')['tasks'];bundles=load('bundles/catalog.json')['bundles']
    demand(descriptor.get('format')=='project-control-preledger' and descriptor.get('schema_version')==1,'preledger version')
    ids=[t['id'] for t in tasks];demand(len(ids)==len(set(ids)),'duplicate tasks')
    by_id={t['id']:t for t in tasks};edges={(d,t['id']) for t in tasks for d in t.get('builds_on',[])}
    demand(all(a in by_id and b in by_id and a!=b for a,b in edges),'unknown/self dependency')
    pending={i:set(by_id[i].get('builds_on',[])) for i in ids};ordered=[]
    while pending:
        ready=sorted(i for i,d in pending.items() if not d)
        if not ready:break
        ordered+=ready
        for i in ready:pending.pop(i)
        for d in pending.values():d.difference_update(ready)
    demand(not pending,'task graph cycle')
    with (root/'dependency_edges.csv').open(newline='') as f:
        actual={(x['producer_task'],x['consumer_task']) for x in csv.DictReader(f)}
    demand(actual==edges,'derived dependency index differs from tasks')
    bundle_ids={b['id'] for b in bundles}
    for t in tasks:
        demand(t.get('repository')=='V200',t['id']+' repository label')
        for key in ['purpose','implementation_mechanism','validation','completion_condition','write_scope']:
            demand(bool(t.get(key)),t['id']+' missing '+key)
        demand((root/'tasks'/f"{t['id']}.md").is_file(),t['id']+' sheet missing')
        demand(t.get('negative_result_acceptable') is False,t['id']+' missing product outcome cannot pass on a negative experiment')
        for p in t['write_scope']:
            demand(not Path(p).is_absolute() and '..' not in Path(p).parts,t['id']+' unsafe scope')
        gpu=t['validation'].get('gpu')
        if isinstance(gpu,list):demand(all(x in bundle_ids for x in gpu),t['id']+' unknown GPU bundle')
    for b in bundles:
        demand(b['status']=='NOT_RUN' and b['automatic_execution_enabled'] is False,b['id']+' invalid authoring execution state')
        demand(0<b['max_wall_seconds']<=600,b['id']+' excessive default duration')
        demand(b['max_gpu_count'] in (2,4),b['id']+' invalid GPU count')
        demand((root/'bundles'/f"{b['id']}.md").is_file(),b['id']+' missing runbook')
    for f in root.rglob('*.json'):
        try:json.loads(f.read_text());demand(True,str(f))
        except (ValueError,OSError):demand(False,'invalid JSON '+str(f.relative_to(root)))
    # Validate ordinary relative markdown links. External URLs and anchor-only
    # links are not fetched; this is offline consistency, not web verification.
    for f in root.rglob('*.md'):
        for target in re.findall(r'\[[^\]]*\]\(([^\s)]+)\)',f.read_text()):
            target=target.split('#')[0]
            if not target or '://' in target or target.startswith('mailto:'):continue
            demand((f.parent/target).resolve().exists(),f'{f.relative_to(root)} broken link {target}')
    native_path=root/'v200.todo-plan.json'
    if native_path.exists():
        native=json.loads(native_path.read_text());nt=native['tasks']
        demand(native['schema_version']==2,'native plan schema')
        demand({t['id'] for t in nt}==set(ids),'native task IDs differ')
        ne={(d['task_id'],t['id']) for t in nt for d in t.get('depends_on',[]) if d['type']=='task'}
        demand(ne==edges,'native edges differ')
        demand(all(t.get('notes','').startswith('preledger_provenance=') for t in nt),'compiler provenance missing')
    manifest=root/'MANIFEST.sha256'
    if require_manifest:
        demand(manifest.is_file(),'manifest missing')
        if manifest.is_file():
            for line in manifest.read_text().splitlines():
                h,p=line.split('  ',1);q=(root/p).resolve()
                demand(q.is_relative_to(root) and q.is_file(),'unsafe/missing manifest member '+p)
                if q.is_file():demand(hashlib.sha256(q.read_bytes()).hexdigest()==h,'hash mismatch '+p)
    return {'format':'V200-PACKAGE-CHECK/1','status':'CPU_PASS' if not errors else 'FAILED','checks':checks,
            'tasks':len(tasks),'task_edges':len(edges),'bundles':len(bundles),'topological_order':ordered,'errors':errors,
            'not_qualified':['live Todo authority/application','CUDA compilation','GPU execution','consumer compatibility/performance']}

def main()->int:
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('package',type=Path)
    p.add_argument('--no-manifest',action='store_true');p.add_argument('--output',type=Path);a=p.parse_args()
    try:r=check(a.package,require_manifest=not a.no_manifest)
    except (OSError,ValueError,KeyError,TypeError) as e:print(e,file=sys.stderr);return 2
    text=json.dumps(r,indent=2)+'\n'
    if a.output:a.output.write_text(text)
    print(text,end='');return 0 if not r['errors'] else 1
if __name__=='__main__':raise SystemExit(main())
