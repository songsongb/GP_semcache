"""Stress-size CPU storage/reader contracts. No real-data or GPU claims."""
import contextlib,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from semcache.edgelora.communication import canonical,digest
from semcache.edgelora.r5_i5_evidence import seal
from semcache.edgelora.r5_i5r2_tensors import Evidence
from semcache.edgelora.r5_i5r4_plan_io import write_plan,read_plan,checked_json,LIMIT

class PlanIO(unittest.TestCase):
    def read(self,root):
        e=Evidence(root,max_input_bytes=128*2**20);r=e.verify();return read_plan(e),r
    def test_actual_prepare_dispatch_through_audit_inspect(self):
        from test_r5_i5r4 import PrepareBudgetTests
        fixture=PrepareBudgetTests()
        with contextlib.ExitStack() as stack,tempfile.TemporaryDirectory() as d:
            api=fixture.setup_inputs(stack);root=Path(d)
            api.reviewed.return_value={'commit':'CPU_FIXTURE','CPU_FIXTURE_ONLY':True}
            api.transport.return_value={'maximum_header_bytes':12345}
            api.baseline.return_value.update(saving_ms=100.,admission_ms=20.)
            prov=api.inputs.return_value[2]['qasper'];prov['manifest']={'source_path':'/CPU_FIXTURE/source'}
            # Stress the real constructor with long repeated, genuine-shaped
            # fixture contexts while retaining each independent context identity.
            for sample in api.inputs.return_value[3]['qasper']:sample['context']+=' context stress '*8000
            stack.enter_context(patch.object(api,'Evidence')).return_value.verify.return_value={'CPU_FIXTURE_ONLY':True}
            stack.enter_context(patch.object(api,'sha',return_value='CPU_FIXTURE_SHA'))
            args=fixture.args();args['rotations']=1
            r=api.prepare(output_dir=root/'plan',**args)
            self.assertEqual(r['status'],'SEALED_CPU_PREPARED_GPU_NOT_RUN')
            p,_=self.read(root/'plan');self.assertEqual(len(p['jobs']),18)
            self.assertGreater(len(canonical(p)),LIMIT)
            # Restore real Evidence for the audit, keeping external-input
            # reconstruction isolated behind the fixture loader.
            with patch.object(api,'load',side_effect=lambda path,*a:self.read(Path(path))),patch.object(api,'invariant_tests',return_value={'run':1,'passed':1,'failed':0,'skipped':0}),patch.object(api,'Evidence',Evidence):
                api.cpu_validate(root/'plan',root/'cpu')
                self.assertEqual(api.inspect(root/'plan',root/'cpu','CPU_FIXTURE',True)['plan_id'],p['plan_id'])
    def test_stress_prepare_storage_cpu_validate_inspect(self):
        from test_r5_i5r4 import CPUReportTests
        from semcache.edgelora import r5_i5r4 as api
        p=CPUReportTests().plan()
        p['jobs']=[j for j in p['jobs'] if j['block']['repetition']==0]
        self.assertEqual(len(p['jobs']),18)
        for j in p['jobs']:
            for t in j['block']['tasks']:t['prepared']['sample']['context']='real-size CPU stress context '*4500
        self.assertGreater(len(canonical(p)),LIMIT)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);plan=root/'plan';plan.mkdir();write_plan(plan,p);seal(plan)
            self.assertTrue(all(x.stat().st_size<=LIMIT for x in plan.rglob('*.json')))
            self.assertLess(sum(x.stat().st_size for x in plan.rglob('*.json')),len(canonical(p))/3)
            self.assertEqual(self.read(plan)[0],p)
            # Only external real-data/source validation and invariant subprocesses
            # are fixture substitutes; actual sealed reader/simulator/audit run.
            with patch.object(api,'load',side_effect=lambda experiment,*a:self.read(Path(experiment))),patch.object(api,'invariant_tests',return_value={'run':1,'passed':1,'failed':0,'skipped':0}):
                api.cpu_validate(plan,root/'cpu')
                r=api.inspect(plan,root/'cpu','CPU_FIXTURE',require_cpu_validation=True)
                self.assertEqual(r['status'],'PREAPPROVED_SEALED_CAMPAIGN_GPU_NOT_RUN')
            other=root/'other';other.mkdir();write_plan(other,p);seal(other)
            self.assertEqual({x.relative_to(plan).as_posix():x.read_bytes() for x in plan.rglob('*') if x.is_file()},
                             {x.relative_to(other).as_posix():x.read_bytes() for x in other.rglob('*') if x.is_file()})
    def test_oversized_legacy_identifies_name_and_size(self):
        from semcache.edgelora.r5_i5r4 import load
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'i4_plan.json').write_bytes(b' '* (LIMIT+1));seal(root)
            with self.assertRaisesRegex(ValueError,'i4_plan.json: measured=16777217 bytes, limit=16777216'):load(root)
    def test_corrupt_missing_and_resealed_wrong_shard(self):
        for mode in ('corrupt','missing','resealed'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as d:
                root=Path(d);write_plan(root,{'context':'abc'*5000});seal(root)
                shard=next((root/'plan_shards').glob('*.json'))
                if mode=='missing':shard.unlink()
                else:shard.write_text('"changed"')
                if mode=='resealed':seal(root)
                with self.assertRaises(ValueError):self.read(root)
    def test_large_string_chunking_and_exact_identity(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p={'context':'αβγ'*200000,'token_ids':[0,1,2],'original_request_id':'unchanged'}
            write_plan(root,p);seal(root);self.assertEqual(self.read(root)[0],p)
    def test_reserved_keys_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):write_plan(Path(d),{'$i4_shard':'untrusted'})

if __name__=='__main__':unittest.main()
