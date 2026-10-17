"""Small CPU-only tests of package helpers. No CUDA import or GPU discovery."""
from __future__ import annotations
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
def module(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/'tools'/f'{name}.py')
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
render=module('render_bundle');bounded=module('run_bounded');elf=module('inspect_elf');package=module('validate_package')
UUIDS=['GPU-00000000-0000-0000-0000-000000000001','GPU-00000000-0000-0000-0000-000000000002']

class Tools(unittest.TestCase):
    def binding(self,root):
        p=root/'planning/v200-bootstrap/tools';p.mkdir(parents=True);(p/'run_bounded.py').write_text('# dummy binding target; never executed')
        (root/'toolkit').mkdir();(root/'source.cpp').write_text('// fixture')
        return {'approved':True,'project_root':str(root),'package_relative':'planning/v200-bootstrap','toolkit_root':str(root/'toolkit'),
                'gpu_uuids':UUIDS.copy(),'argv':['build/probe','--execute'],'binary_paths':['build/probe'],
                'source_paths':['source.cpp'],'evidence_relative':'evidence/run-1','fixture_enforces_allocation_cap':True,'build_argv':['cmake','--build','build']}
    def test_render_is_inert_and_uses_native_uuid_key(self):
        with tempfile.TemporaryDirectory() as t:
            b=self.binding(Path(t));c=json.loads((ROOT/'bundles/catalog.json').read_text());r=render.render(c,'B01',b)
            self.assertEqual(r['resources']['gpu_uuids'],UUIDS);self.assertEqual(r['resources']['gpus'],2)
            self.assertIn('build_argv',r['benchmark']);self.assertNotIn('build_argv',r)
            self.assertNotIn('metric',r);self.assertFalse((Path(t)/'evidence').exists())
    def test_unapproved_binding_refused(self):
        b=json.loads((ROOT/'bundles/binding.example.json').read_text())
        with self.assertRaises(ValueError):render.render(json.loads((ROOT/'bundles/catalog.json').read_text()),'B01',b)
    def test_bad_uuid_and_extra_gpu_refused(self):
        with tempfile.TemporaryDirectory() as t:
            b=self.binding(Path(t));c=json.loads((ROOT/'bundles/catalog.json').read_text());b['gpu_uuids']=['0','2']
            with self.assertRaises(ValueError):render.render(c,'B01',b)
            b['gpu_uuids']=UUIDS+[UUIDS[0]]
            with self.assertRaises(ValueError):render.render(c,'B01',b)
    def test_caps_cannot_expand_silently(self):
        with tempfile.TemporaryDirectory() as t:
            b=self.binding(Path(t));c=json.loads((ROOT/'bundles/catalog.json').read_text());b['wall_seconds']=91
            with self.assertRaises(ValueError):render.render(c,'B01',b)
            b['wall_seconds']=90;b['allocation_cap_mib_per_gpu']=513
            with self.assertRaises(ValueError):render.render(c,'B01',b)
    def test_no_vram_enforcement_claim(self):
        with tempfile.TemporaryDirectory() as t:
            b=self.binding(Path(t));b['fixture_enforces_allocation_cap']=False
            with self.assertRaises(ValueError):render.render(json.loads((ROOT/'bundles/catalog.json').read_text()),'B01',b)
    def test_unsafe_paths_and_argv(self):
        for value in ['../escape','/absolute','REPLACE-me']:
            with self.assertRaises(ValueError):render.relative(value,'path')
        for value in ['echo shell',[],['x','\0']]:
            with self.assertRaises(ValueError):render.argv(value,'argv')
    def test_lease_shape_and_wrong_owner(self):
        root=Path('/tmp').resolve();r={'format':'CUDA-FOREGROUND-LEASE/1','state':'active','project_root':str(root),
             'owner_id':'foreground:00000000-0000-0000-0000-000000000000','pid':123,
             'resource_ids':['accelerator:'+u for u in UUIDS]}
        self.assertEqual(bounded.validate_receipt(r,root,UUIDS),123)
        for key,value in [('state','released'),('pid',True),('project_root','/other'),('resource_ids',[]),('owner_id','fake')]:
            bad=dict(r);bad[key]=value
            with self.assertRaises(ValueError):bounded.validate_receipt(bad,root,UUIDS)
    def test_missing_live_lease_fails_closed(self):
        old=os.environ.pop('TODO_GPU_LEASE_RECEIPT',None)
        try:
            with self.assertRaises(ValueError):bounded.live_lease(Path('/tmp'),UUIDS)
        finally:
            if old is not None:os.environ['TODO_GPU_LEASE_RECEIPT']=old
    def test_bounded_cpu_child(self):
        with tempfile.TemporaryDirectory() as t:
            r=bounded.run_child([sys.executable,'-c','print("cpu-only")'],2,Path(t)/'ok')
            self.assertEqual(r['returncode'],0);self.assertFalse(r['timed_out']);self.assertFalse(r['orphaned_children'])
    def test_bounded_timeout_cpu_child(self):
        with tempfile.TemporaryDirectory() as t:
            r=bounded.run_child([sys.executable,'-c','import time; time.sleep(5)'],0.05,Path(t)/'timeout')
            self.assertTrue(r['timed_out']);self.assertFalse(r['termination_uncertain'])
    def test_elf_not_executed(self):
        target=Path('/bin/true')
        if not target.exists():self.skipTest('Linux fixture unavailable')
        r=elf.inspect(target);self.assertEqual(r['status'],'CPU_INSPECTED');self.assertIn('sha256',r)
    def test_non_elf_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'bad';p.write_text('not executable')
            with self.assertRaises(ValueError):elf.inspect(p)
    def test_package_structure(self):
        r=package.check(ROOT,require_manifest=False);self.assertEqual(r['errors'],[])
    def test_canonical_adapter_does_not_silently_fork(self):
        # This environment has no installed Project Control; if one is installed,
        # the adapter must be tested with its own qualified fixture instead.
        if importlib.util.find_spec('project_control') is not None:self.skipTest('canonical environment installed')
        with tempfile.TemporaryDirectory() as t:
            q=subprocess.run([sys.executable,str(ROOT/'tools/compile_plan.py'),'--package',str(ROOT),'--output',str(Path(t)/'p.json')],capture_output=True,text=True)
            self.assertEqual(q.returncode,2);self.assertFalse((Path(t)/'p.json').exists())
if __name__=='__main__':unittest.main()
