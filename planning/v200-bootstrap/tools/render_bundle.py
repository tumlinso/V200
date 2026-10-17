#!/usr/bin/env python3
"""Render an inert foreground-controller spec; never acquire or execute GPUs."""
from __future__ import annotations
import argparse
import json
import re
import sys
from pathlib import Path

UUID = re.compile(r"GPU-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\Z")

def positive(value: object, label: str, ceiling: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not 0 < value <= ceiling:
        raise ValueError(f"{label} must be an integer from 1 through {ceiling}")
    return value

def relative(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or Path(value).is_absolute() or '..' in Path(value).parts:
        raise ValueError(f"{label} must be a safe repository-relative path")
    if any(x in value for x in ['REPLACE', '<', '>', '\0']): raise ValueError(f"{label} contains a placeholder")
    return value

def argv(value: object, label: str) -> list[str]:
    if not isinstance(value, list) or not value or not all(isinstance(x,str) and x and '\0' not in x for x in value):
        raise ValueError(f"{label} must be a nonempty structured argv")
    if any(any(p in x for p in ['REPLACE','<','>']) for x in value): raise ValueError(f"unbound {label}")
    return value

def render(catalog: dict, bundle_id: str, b: dict) -> dict:
    matches=[x for x in catalog['bundles'] if x['id']==bundle_id]
    if len(matches)!=1: raise ValueError('unknown or duplicate bundle')
    c=matches[0]
    if b.get('approved') is not True: raise ValueError('actual execution approval must be bound; template is inert')
    if c['id']=='B07' and not b.get('decision_receipt'): raise ValueError('B07 requires a selected/approved decision receipt')
    root=Path(b['project_root']).resolve(strict=True)
    if not root.is_dir() or 'REPLACE' in str(root): raise ValueError('invalid project root')
    pkg=relative(b['package_relative'],'package_relative')
    runner=(root/pkg/'tools/run_bounded.py').resolve(strict=True)
    if not runner.is_relative_to(root): raise ValueError('runner escaped project')
    toolkit=Path(b['toolkit_root']).resolve(strict=True)
    if not toolkit.is_dir() or 'REPLACE' in str(toolkit): raise ValueError('invalid toolkit path')
    uuids=b.get('gpu_uuids')
    if not isinstance(uuids,list) or len(uuids)!=c['max_gpu_count'] or len(set(uuids))!=len(uuids) or any(not isinstance(u,str) or not UUID.fullmatch(u) for u in uuids):
        raise ValueError('exact, distinct physical GPU UUIDs required')
    seconds=positive(b.get('wall_seconds',c['max_wall_seconds']),'wall_seconds',c['max_wall_seconds'])
    cpu=positive(b.get('cpu_threads',2),'cpu_threads',c['max_cpu_threads'])
    cap=positive(b.get('allocation_cap_mib_per_gpu',c['application_allocation_cap_mib_per_gpu']),'allocation cap',c['application_allocation_cap_mib_per_gpu'])
    if b.get('fixture_enforces_allocation_cap') is not True: raise ValueError('fixture must enforce its declared allocation cap')
    cmd=argv(b['argv'],'argv')
    evidence=relative(b['evidence_relative'],'evidence_relative')
    source_paths=[relative(x,'source path') for x in b.get('source_paths',[])]
    binaries=[relative(x,'binary path') for x in b.get('binary_paths',[])]
    if not source_paths or not binaries: raise ValueError('source_paths and binary_paths are required')
    for p in source_paths:
        actual=(root/p).resolve(strict=True)
        if not actual.is_relative_to(root) or not actual.is_file(): raise ValueError('source outside project or not a file')
    command=['python3',str(runner),'--project-root',str(root),'--seconds',str(seconds),
             '--allocation-cap-mib',str(cap),'--evidence-dir',str(root/evidence),'--bundle',bundle_id]
    for u in uuids: command += ['--expected-uuid',u]
    command += ['--',*cmd]
    spec={'schema_version':1,'project_root':str(root),'command_cwd':str(root),'recipe':'baseline',
          'toolchain':{'root':str(toolkit),'require_sanitizer':False},
          'resources':{'gpu_uuids':uuids,'gpus':len(uuids),'cpu_threads':cpu,
                       'ram_bytes':c['host_ram_cap_mib']*1024*1024,'isolate_nvlink_domain':True,'isolate_pcie_root':False},
          'argv':command,'timeout':seconds+15,'build_timeout':c['max_build_seconds'],
          'binary_paths':binaries,'paths':source_paths}
    if b.get('build_argv'): spec['benchmark']={'build_argv':argv(b['build_argv'],'build_argv')}
    # No metric key: one wrapper invocation owns its finite cases/repetitions.
    # No background arm, private priority override, or duplicate resource manager.
    return spec

def main() -> int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--catalog',type=Path,required=True);p.add_argument('--bundle',required=True)
    p.add_argument('--binding',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    try:
        spec=render(json.loads(a.catalog.read_text()),a.bundle,json.loads(a.binding.read_text()))
        a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(spec,indent=2)+'\n')
        print(json.dumps({'status':'RENDERED_NOT_EXECUTED','output':str(a.output),'bundle':a.bundle}))
        return 0
    except (OSError,KeyError,ValueError,TypeError) as e: print(f'bundle binding: {e}',file=sys.stderr);return 2
if __name__=='__main__':raise SystemExit(main())
