#!/usr/bin/env python3
"""Finite child execution under an existing CUDA foreground lease.

This validates a receipt and bounds an owned child; it does not grant a lease,
assert a hard VRAM quota, schedule GPU work, or reset/kill foreign processes.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time


def validate_receipt(receipt: dict, root: Path, expected: list[str]) -> int:
    if receipt.get('format')!='CUDA-FOREGROUND-LEASE/1' or receipt.get('state')!='active':
        raise ValueError('not an active canonical foreground receipt')
    if receipt.get('project_root')!=str(root.resolve()): raise ValueError('receipt project mismatch')
    ids=receipt.get('resource_ids')
    wanted=[f'accelerator:{u}' for u in expected]
    if not isinstance(ids,list) or sorted(ids)!=sorted(wanted) or len(set(ids))!=len(ids):
        raise ValueError('receipt GPU set mismatch')
    owner=receipt.get('owner_id')
    if not isinstance(owner,str) or not re.fullmatch(r'foreground:[0-9a-fA-F-]{36}',owner): raise ValueError('invalid lease owner')
    pid=receipt.get('pid')
    if not isinstance(pid,int) or isinstance(pid,bool) or pid<=1: raise ValueError('invalid owner PID')
    return pid

def live_lease(root: Path, expected: list[str]) -> dict:
    raw=os.environ.get('TODO_GPU_LEASE_RECEIPT')
    if not raw: raise ValueError('TODO_GPU_LEASE_RECEIPT absent; use canonical controller')
    path=Path(raw).resolve(strict=True); receipt=json.loads(path.read_text())
    pid=validate_receipt(receipt,root,expected)
    os.kill(pid,0)
    if b'cuda_controller.py' not in Path(f'/proc/{pid}/cmdline').read_bytes(): raise ValueError('owner is not the controller')
    cur=os.getpid(); seen=set()
    for _ in range(64):
        if cur==pid:break
        if cur<=1 or cur in seen:raise ValueError('not a descendant of the lease owner')
        seen.add(cur)
        m=re.search(r'^PPid:\s*(\d+)\s*$',Path(f'/proc/{cur}/status').read_text(),re.MULTILINE)
        if not m:raise ValueError('cannot verify controller ancestry')
        cur=int(m.group(1))
    else:raise ValueError('controller ancestry depth exceeded')
    visible=[x.strip() for x in os.environ.get('CUDA_VISIBLE_DEVICES','').split(',') if x.strip()]
    if len(visible)!=len(expected):raise ValueError('visible count differs from lease')
    if any(not u.startswith('GPU-') for u in visible):
        q=subprocess.run(['nvidia-smi','--query-gpu=uuid,index','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5,check=True)
        by_index={row[1].strip():row[0].strip() for row in csv.reader(q.stdout.splitlines()) if len(row)==2}
        visible=[u if u.startswith('GPU-') else by_index.get(u,'') for u in visible]
    if sorted(visible)!=sorted(expected):raise ValueError('visible physical UUID set mismatch')
    return {'receipt':str(path),'owner_pid':pid,'visible_uuids':visible}

def run_child(cmd: list[str], seconds: float, out: Path, env: dict|None=None) -> dict:
    out.mkdir(parents=True,exist_ok=True)
    start=time.monotonic(); timed_out=False; uncertain=False; interrupted=False
    with (out/'stdout.txt').open('wb') as stdout,(out/'stderr.txt').open('wb') as stderr:
        proc=subprocess.Popen(cmd,stdout=stdout,stderr=stderr,start_new_session=True,env=env)
        try:proc.wait(timeout=seconds)
        except (subprocess.TimeoutExpired,KeyboardInterrupt) as e:
            timed_out=isinstance(e,subprocess.TimeoutExpired);interrupted=not timed_out
            try:os.killpg(proc.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            try:proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                try:os.killpg(proc.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                try:proc.wait(timeout=2)
                except subprocess.TimeoutExpired:uncertain=True
    # A successful fixture must not leave its own server/worker children alive.
    orphaned = False
    try:
        os.killpg(proc.pid, 0)
        orphaned = True
    except ProcessLookupError:
        pass
    if orphaned:
        try: os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError: pass
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            try: os.killpg(proc.pid, 0)
            except ProcessLookupError: break
            time.sleep(0.02)
        else:
            try: os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError: pass
            uncertain = True  # descendants are not directly reapable here
    return {'returncode':proc.returncode,'timed_out':timed_out,'interrupted':interrupted,'orphaned_children':orphaned,
            'termination_uncertain':uncertain,'wall_seconds':time.monotonic()-start,
            'stdout_sha256':hashlib.sha256((out/'stdout.txt').read_bytes()).hexdigest(),
            'stderr_sha256':hashlib.sha256((out/'stderr.txt').read_bytes()).hexdigest()}

def main() -> int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root',type=Path,required=True);p.add_argument('--seconds',type=int,required=True)
    p.add_argument('--allocation-cap-mib',type=int,required=True);p.add_argument('--evidence-dir',type=Path,required=True)
    p.add_argument('--bundle',required=True);p.add_argument('--expected-uuid',action='append',required=True)
    p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
    try:
        if not 0<a.seconds<=3600 or a.allocation_cap_mib<=0:raise ValueError('invalid resource bound')
        cmd=a.command[1:] if a.command and a.command[0]=='--' else a.command
        if not cmd:raise ValueError('missing command')
        root=a.project_root.resolve(strict=True);out=a.evidence_dir.resolve()
        if not out.is_relative_to(root):raise ValueError('evidence must remain inside project')
        if out.exists():raise ValueError('evidence directory exists; use a new run ID')
        lease=live_lease(root,a.expected_uuid)
        env=os.environ.copy();env['V200_BUNDLE_VRAM_CAP_MIB_PER_GPU']=str(a.allocation_cap_mib)
        env['V200_BUNDLE_ID']=a.bundle
        def interrupted(_signum, _frame):
            raise KeyboardInterrupt
        previous = signal.signal(signal.SIGTERM, interrupted)
        try: result=run_child(cmd,a.seconds,out,env)
        finally: signal.signal(signal.SIGTERM, previous)
        result.update(bundle=a.bundle,argv=cmd,lease=lease,allocation_cap_mib_per_gpu=a.allocation_cap_mib,
                      memory_cap_enforcement='fixture responsibility; not an OS hard quota',
                      gpu_correctness='read case-level fixture output; child exit alone is not GPU acceptance')
        result['status']='CHILD_COMPLETED' if result['returncode']==0 and not result['timed_out'] and not result['interrupted'] and not result['orphaned_children'] else 'FAILED'
        (out/'execution.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result))
        if result['termination_uncertain']:return 125
        if result['timed_out']:return 124
        if result['interrupted']:return 130
        return 0 if result['status']=='CHILD_COMPLETED' else 1
    except (OSError,ValueError,subprocess.SubprocessError,KeyError) as e:
        print(f'bundle refused: {e}',file=sys.stderr);return 2
if __name__=='__main__':raise SystemExit(main())
