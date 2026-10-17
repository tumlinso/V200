#!/usr/bin/env python3
"""Read a binary's ELF surface without executing it or loading CUDA."""
from __future__ import annotations
import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

def inspect(path: Path) -> dict:
    path=path.resolve(strict=True)
    data=path.read_bytes()
    if data[:4]!=b'\x7fELF':raise ValueError('not an ELF binary')
    readelf=shutil.which('readelf')
    if not readelf:raise ValueError('readelf unavailable')
    outputs={}
    for key,flag in [('dynamic','-d'),('symbols','-Ws'),('notes','-n')]:
        q=subprocess.run([readelf,flag,'--',str(path)],capture_output=True,text=True,timeout=10,check=True)
        outputs[key]=q.stdout
    symbols=[]
    for line in outputs['symbols'].splitlines():
        if ' UND ' in line:
            name=line.split()[-1].split('@')[0]
            if name.startswith(('cu','cuda','cublas','nccl')) or name in ('dlopen','dlsym'):symbols.append(name)
    markers=[s for s in ['cuGetProcAddress','cudaGetDriverEntryPoint','dlopen','dlsym'] if s.encode() in data]
    return {'format':'V200-ELF-AUDIT/1','status':'CPU_INSPECTED','path':str(path),'sha256':hashlib.sha256(data).hexdigest(),
            'needed':re.findall(r'\(NEEDED\).*?\[(.*?)\]',outputs['dynamic']),
            'visible_undefined_symbols':sorted(set(symbols)),'string_markers':markers,
            'readelf_output_sha256':{k:hashlib.sha256(v.encode()).hexdigest() for k,v in outputs.items()},
            'limits':['Static/internal calls and dynamically resolved private tables may be absent from this surface.',
                      'Presence of a string is not evidence that the path executes.',
                      'No target code, GPU query or shared library was executed by this inspector.']}

def main()->int:
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('binary',type=Path);p.add_argument('--output',type=Path)
    a=p.parse_args()
    try:
        text=json.dumps(inspect(a.binary),indent=2)+'\n'
        if a.output:a.output.write_text(text)
        else:print(text,end='')
        return 0
    except (OSError,ValueError,subprocess.SubprocessError) as e:print(e,file=sys.stderr);return 2
if __name__=='__main__':raise SystemExit(main())
