"""CPU contract tests and explicit Torch-dependent controlled-executor checks."""
from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace as NS

from semcache.diagnostics.r2a_overhead_audit import (boundaries,source_contract,ImmutableFixture,audit_trace,compare_execution)
from semcache.diagnostics.r1b3_break_even import MissingEvidence
ROOT=Path(__file__).resolve().parents[1]


def load_runner():
    spec=importlib.util.spec_from_file_location('r2a5_test_runner',ROOT/'scripts/84_audit_r2a_overhead.py')
    m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


class StaticTests(unittest.TestCase):
    def test_boundaries_and_source_contract(self):
        table=boundaries(ROOT); rows={r['component']:r for r in table}
        self.assertIn('INCLUDES snapshot_validation',rows['warm_mapping_lookup']['nesting'])
        self.assertIn('validate=False',rows['normal_warm_mapping']['exact_measured_boundary'])
        self.assertEqual(rows['attention_skip_claim']['status'],'CONTRADICTED')
        self.assertTrue(all(r['measured_difference_ms'] is None for r in table))
        self.assertEqual(source_contract(ROOT)['executor_descriptor_validation_calls'],1)
    def test_static_artifacts_trace_serialization_collision(self):
        m=load_runner()
        with tempfile.TemporaryDirectory() as tmp:
            trace=Path(tmp)/'trace.json'; trace.write_text(json.dumps({'traceEvents':[]}))
            root=Path(tmp)/'audit'
            self.assertEqual(m.main(['--output-dir',str(root),'--trace',str(trace)]),0)
            manifest=json.loads((root/'manifest.json').read_text())
            self.assertEqual(manifest['execution_status'],'COMPLETE')
            self.assertEqual(manifest['fixture_status'],'UNVERIFIED')
            self.assertFalse(manifest['results_are_new_gpu_measurements'])
            self.assertEqual(json.loads((root/'trace_attribution_audit.json').read_text())[0]['audit']['status'],'UNVERIFIED')
            with self.assertRaises(FileExistsError): m.main(['--output-dir',str(root)])
    def test_optin_and_failure(self):
        m=load_runner()
        with self.assertRaises(SystemExit): m.parse_args(['--phase','paired','--output-dir','unused'])
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'failure'
            with patch.object(m,'paired',side_effect=MissingEvidence('mutation detected')):
                self.assertEqual(m.main(['--phase','paired','--allow-validation-bypass','--output-dir',str(root)]),1)
            manifest=json.loads((root/'manifest.json').read_text())
            self.assertEqual(manifest['fixture_status'],'UNVERIFIED')
            self.assertEqual(manifest['execution_status'],'MISSING_EVIDENCE')
            self.assertEqual(json.loads((root/'decision_summary.json').read_text())['controlled_conclusion'],'INVALID_OR_INCOMPLETE')
    def test_sign_pairing_and_no_profiler_sum(self):
        m=load_runner(); rows=[]
        for i in range(4):
            for mode,cost in zip(m.MODES,(20,10)):
                rows.append(dict(length=32,phase='measured',repeat=i,mode=mode,wall_ms=cost,order_seed=4))
        report=m.summary_rows(rows)[0]
        self.assertEqual(report['paired']['mean'],10)
        self.assertEqual(report['status'],'SUPPORTED')
        with self.assertRaises(MissingEvidence): m.summary_rows(rows[:-1])
    def test_closest_attention_is_not_parent_inclusive(self):
        events=[dict(ph='X',cat='user_annotation',name='r2a::attention_inclusive',ts=0,dur=100,pid=1,tid=1),
                dict(ph='X',cat='user_annotation',name='r2a::q_base',ts=1,dur=20,pid=1,tid=1),
                dict(ph='X',cat='cpu_op',name='aten::addmm',ts=2,dur=5,pid=1,tid=1,args={'External id':7}),
                dict(ph='X',cat='cpu_op',name='aten::bmm',ts=30,dur=5,pid=1,tid=1,args={'External id':8}),
                dict(ph='X',cat='kernel',name='opaque',ts=120,dur=10,args={'External id':7}),
                dict(ph='X',cat='kernel',name='opaque2',ts=130,dur=6,args={'External id':8})]
        report=audit_trace({'traceEvents':events}); t=report['totals']
        self.assertAlmostEqual(t['attention_parent_inclusive_kernel_ms'],.016)
        self.assertAlmostEqual(t['attention_closest_tag_kernel_ms'],.006)
        self.assertAlmostEqual(t['linked_attention_math_candidate_ms'],.006)
        self.assertEqual(report['activities'][0]['component'],'q_base')
    def test_conflicting_links_and_no_trace_not_zero_evidence(self):
        events=[]
        for tid,tag in [(1,'q_base'),(2,'attention_inclusive')]:
            events.extend([dict(ph='X',cat='user_annotation',name='r2a::'+tag,ts=0,dur=10,pid=1,tid=tid),
                           dict(ph='X',cat='cpu_op',name='aten::mm',ts=1,dur=2,pid=1,tid=tid,args={'External id':9})])
        events.append(dict(ph='X',cat='kernel',name='gemm_not_proof',ts=30,dur=5,args={'External id':9}))
        report=audit_trace({'traceEvents':events})
        self.assertEqual(report['ambiguous_kernel_count'],1)
        self.assertEqual(audit_trace({})['status'],'UNVERIFIED')

    def test_paired_runner_complete_and_postcheck_gate(self):
        m=load_runner()
        class Runtime:
            def __init__(self,a): self.provenance={'fixture':'FAKE_CONTROL_FLOW_ONLY'}; self.removed=False; self.namespace='ns'; self.weights='weights'
            def prepare(self,n):
                self.ids=list(range(n)); self.snapshot='s'; self.costs={'source_cost':999}; self.preflight={}; self.backing_check=[]; self.meta={'resolved_model_revision':'rev'}
            def invoke(self,mode,capture=False):
                e=dict(request_ms=10 if self.removed else 20,reused_tokens=30,projection_audit={'skip':90},selection_hash='hits')
                return 0,e
            def gpu_state(self): return {'status':'NOT A GPU TEST'}
        class Lease:
            fail_close=False
            def __init__(self,r): self.r=r; self.checks=0
            def check(self): self.checks+=1
            @contextmanager
            def bypass(self):
                self.check(); self.r.removed=True
                try: yield
                finally: self.r.removed=False; self.check()
            def close(self):
                if self.fail_close: raise MissingEvidence('persistent data mutation')
                return {'status':'VERIFIED_CONTROLLED_FIXTURE'}
        with tempfile.TemporaryDirectory() as tmp:
            args=['--phase','paired','--allow-validation-bypass','--device','cpu','--cache-device','cpu','--measured-runs','3']
            with patch.object(m,'CostRuntime',Runtime),patch.object(m,'ImmutableFixture',Lease),patch.object(m,'compare_execution',return_value={'passed':True}),patch.object(m,'logit_metrics',return_value={'correctness_status':'STRICT_FAIL'}):
                root=Path(tmp)/'ok'
                self.assertEqual(m.main(args+['--output-dir',str(root)]),0)
                manifest=json.loads((root/'manifest.json').read_text())
                self.assertEqual(manifest['execution_status'],'COMPLETE')
                self.assertEqual(manifest['correctness_status'],'STRICT_FAIL')
                report=json.loads((root/'controlled_difference.json').read_text())['32']
                self.assertEqual(report['paired']['mean'],10)
                Lease.fail_close=True; root=Path(tmp)/'mutated'
                self.assertEqual(m.main(args+['--output-dir',str(root)]),1)
                self.assertEqual(json.loads((root/'manifest.json').read_text())['fixture_status'],'UNVERIFIED')
                self.assertEqual(json.loads((root/'decision_summary.json').read_text())['controlled_conclusion'],'INVALID_OR_INCOMPLETE')



class LeaseControlTests(unittest.TestCase):
    def fake(self):
        descriptor=NS(validate=lambda hits,book=None:None)
        rt=NS(plan=NS(descriptor=descriptor),hits=(object(),))
        class Lease(ImmutableFixture):
            def fingerprint(self): return ('fake immutable',rt.plan is plan)
            def backing_hashes(self): return ('fake content',)
            def model_hashes(self): return ('fake weights',)
        plan=rt.plan
        return rt,Lease(rt)
    def test_one_call_and_restore_on_exception(self):
        r,lease=self.fake(); original=r.plan.descriptor.validate
        with lease.bypass(): self.assertIsNone(r.plan.descriptor.validate(r.hits))
        self.assertIs(r.plan.descriptor.validate,original)
        with self.assertRaisesRegex(RuntimeError,'native failure'):
            with lease.bypass():
                r.plan.descriptor.validate(r.hits); raise RuntimeError('native failure')
        self.assertIs(r.plan.descriptor.validate,original)
        self.assertEqual(lease.close()['bypassed_calls'],2)
    def test_missing_extra_wrong_calls_refused(self):
        for calls in (0,2):
            r,lease=self.fake()
            with self.assertRaises(MissingEvidence):
                with lease.bypass():
                    for _ in range(calls): r.plan.descriptor.validate(r.hits)
        r,lease=self.fake()
        with self.assertRaises(MissingEvidence):
            with lease.bypass(): r.plan.descriptor.validate((object(),))
    def test_mutation_prevents_bypass_and_postcheck(self):
        r,lease=self.fake(); original=r.plan.descriptor.validate; r.plan=NS(descriptor=r.plan.descriptor)
        with self.assertRaises(MissingEvidence):
            with lease.bypass(): self.fail('must not execute')
        self.assertIs(r.plan.descriptor.validate,original)
        r,lease=self.fake()
        with self.assertRaises(MissingEvidence):
            with lease.bypass():
                r.plan.descriptor.validate(r.hits); r.plan=NS(descriptor=r.plan.descriptor)


try:
    import torch
except ImportError:
    torch=None


@unittest.skipIf(torch is None,'Torch absent: actual immutable-fixture/executor tests NOT EXECUTED')
class TensorTests(unittest.TestCase):
    def fixture(self,device='cpu'):
        from semcache.diagnostics.r2a_runtime import CostRuntime
        from semcache.diagnostics.r1b3_occurrences import OccurrenceCatalog
        from semcache.diagnostics.r1b3_fast_selection import FastOccurrenceIndex
        from semcache.edgelora.batched_mapping_optimized import SourceBackingDescriptors,PreparedMappingCache
        class Projection(torch.nn.Linear):
            def get_base_layer(self): return self
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.projections=torch.nn.ModuleDict({r:Projection(4,4).to(device) for r in 'qkv'}); self.active_adapters=['user_a']
            def set_adapter(self,name): self.active_adapters=[name]
            def forward(self,input_ids,**kwargs):
                x=input_ids.float().unsqueeze(-1).expand(-1,-1,4)
                return NS(logits=sum(m(x) for m in self.projections.values()))
        model=Model().eval(); ids=list(range(8)); x=torch.tensor([ids],device=device).float().unsqueeze(-1).expand(-1,-1,4)
        with torch.inference_mode(): qkv={0:tuple(m(x) for m in model.projections.values())}
        layout={0:((4,torch.float32),)*3}; catalog=OccurrenceCatalog('ns','weights',device,100000)
        catalog.add_source('source',ids,0,'user_a',qkv,layout,prefill_complete=True)
        index=FastOccurrenceIndex(catalog,layout); descriptor=SourceBackingDescriptors(index)
        hits=tuple(index.select(ids,0,'user_a','ns','weights',layout,'max')[0])
        plan,_=PreparedMappingCache(descriptor).get(hits,ids,'user_a','ns','weights',device,torch.float32,mode='contiguous')
        r=CostRuntime.__new__(CostRuntime); r.a=NS(device=device,dtype='float32'); r.model=model
        r.adapter=NS(layers=[object()],projection_modules=lambda i:dict(model.projections.items()))
        r.ids=ids; r.oracle_ids=tuple(ids); r.text='tokens'; r.length=len(ids); r.namespace='ns'; r.weights='weights'; r.storage=device
        r.tokenizer=lambda text:dict(input_ids=list(ids)); r.catalog=catalog; r.index=index; r.hits=hits; r.plan=plan; r.descriptor=descriptor
        r.snapshot=r.oracle_snapshot='snapshot'; return r
    def exercise(self,device):
        import semcache.models.lora_decomposition as decomposition
        r=self.fixture(device); lease=ImmutableFixture(r)
        with patch.object(decomposition,'validate_projection',lambda *a:None):
            a,ra=r.invoke('ORACLE_PRESELECTED',capture=True)
            with lease.bypass(): b,rb=r.invoke('ORACLE_PRESELECTED',capture=True)
        self.assertTrue(compare_execution(a,b,ra,rb)['passed'])
        self.assertEqual(lease.close()['bypassed_calls'],1)
        self.assertEqual(ra['projection_audit']['skipped_projection_rows_all_layers_roles'],18)
    def test_cpu_executor(self): self.exercise('cpu')
    def test_cuda_executor(self):
        if not torch.cuda.is_available(): self.skipTest('CUDA unavailable')
        self.exercise('cuda:0')
    def test_data_mutation_and_mapping_rejection(self):
        r=self.fixture(); lease=ImmutableFixture(r)
        r.plan.index_tensors[('fresh',0)].data.add_(1)
        with self.assertRaises(MissingEvidence): lease.close()
        r=self.fixture(); lease=ImmutableFixture(r); r.plan.groups[0]['source']=(1,2,3,4,5,6)
        with self.assertRaises(MissingEvidence):
            with lease.bypass(): self.fail('changed mapping cannot run')
        r=self.fixture(); lease=ImmutableFixture(r)
        with torch.no_grad(): next(r.model.parameters()).add_(1)
        with self.assertRaises(MissingEvidence): lease.check()

if __name__=='__main__': unittest.main()
