#!/usr/bin/env python3
"""Use the installed canonical Project Control compiler; no plan mutation."""
from __future__ import annotations
import argparse
import hashlib
import inspect
import json
import sys
from pathlib import Path

def main()->int:
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--package',type=Path,required=True)
    p.add_argument('--repository-label',default='V200');p.add_argument('--output',type=Path,required=True)
    p.add_argument('--compare',type=Path);a=p.parse_args()
    try:
        from project_control import preledger
    except ImportError:
        print('Use the canonical Project Control source environment or its plan compile CLI; no bundled compiler fork.',file=sys.stderr);return 2
    try:
        c=preledger.compile_preledger(a.package,target_repository=a.repository_label)
        text=c.canonical_plan_json()
        if a.compare and text!=a.compare.read_text():raise ValueError('canonical output changed; inspect compiler/source/package version before import')
        a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(text)
        print(json.dumps({'status':'COMPILED_NOT_APPLIED','tasks':c.selected_task_count,'edges':c.internal_dependency_count,
                          'warnings':c.warnings,'plan_sha256':c.plan_digest,'package_digest':c.package_digest,
                          'compiler_sha256':hashlib.sha256(Path(inspect.getfile(preledger)).read_bytes()).hexdigest()}))
        return 0
    except (OSError,ValueError) as e:print(e,file=sys.stderr);return 2
if __name__=='__main__':raise SystemExit(main())
