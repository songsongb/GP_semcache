import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from semcache.diagnostics.r2a_analysis import MODES,schedule,summaries,decisions
from semcache.diagnostics.r2a_runtime import SEMANTICS,CostRuntime
from semcache.diagnostics.r1b3_break_even import MissingEvidence

ROOT=Path(__file__).resolve().parents[1]

def runner():
    spec=importlib.util.spec_from_file_location('r2a_runner_test',ROOT/'scripts/83_profile_r2a_costs.py')
    m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

class Contracts(unittest.TestCase):
    def samples(self):
        return [dict(length=128,repeat=i,phase='measured',mode=mode,wall_ms=value+i,profiler_enabled=False)
            for i in range(4) for mode,value in zip(MODES,(75,110,140,100))]
    def test_sign_and_pairing(self):
        summary,bridge=summaries(self.samples()); self.assertEqual(len(summary),4)
        self.assertEqual([r['mean_ms'] for r in bridge],[65,35,40,25])
        self.assertEqual(decisions(bridge)[0]['hypotheses']['H-E']['status'],'SUPPORTED')
        self.assertEqual(decisions(bridge)[0]['hypotheses']['H-A']['status'],'UNRESOLVED')
    def test_incomplete_duplicate_profile_rejected(self):
        for rows in [self.samples()[:-1],self.samples()+[self.samples()[0]]]:
            with self.assertRaises(ValueError): summaries(rows)
        rows=self.samples(); rows[0]['profiler_enabled']=True
        with self.assertRaises(ValueError): summaries(rows)
    def test_balanced_seed(self):
        orders=schedule(MODES,4,123)
        for i in range(4): self.assertEqual(set(o[i] for o in orders),set(MODES))
        self.assertEqual(orders,schedule(MODES,4,123))
    def test_oracle_contract(self):
        o=SEMANTICS['ORACLE_PRESELECTED']
        self.assertIn('executor snapshot validation',o['executes'])
        self.assertIn('attention impact',o['bypasses'])
        self.assertIn('physical QKV reuse',SEMANTICS['LOOKUP_NO_REUSE']['bypasses'])
        self.assertIn('native-only Legacy wrapper',SEMANTICS['LOOKUP_NO_REUSE']['executes'])
    def test_failure_and_collision(self):
        m=runner()
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'run'
            with patch.object(m,'run',side_effect=MissingEvidence('missing projection audit')):
                self.assertEqual(m.main(['--output-dir',str(out)]),1)
            manifest=json.loads((out/'manifest.json').read_text())
            self.assertEqual(manifest['execution_status'],'MISSING_EVIDENCE')
            self.assertEqual(manifest['correctness_status'],'UNVERIFIED')
            self.assertTrue((out/'failure.json').exists())
            for file in m.CSV_FILES: self.assertTrue((out/(file+'.csv')).exists())
            with self.assertRaises(FileExistsError): m.main(['--output-dir',str(out)])
    def test_cli(self):
        m=runner(); a=m.parse_args(['--phase','trace','--prompt-lengths','128,256','--output-dir','unused'])
        self.assertEqual(a.measured_runs,10); self.assertEqual(a.warmup_runs,3)
        with self.assertRaises(SystemExit): m.parse_args(['--phase','minimal','--measured-runs','2','--output-dir','unused'])

    def test_complete_strict_fail_is_not_execution_failure(self):
        from types import SimpleNamespace as NS
        m=runner()
        class FakeRuntime:
            def __init__(self,a):
                self.provenance={'gpu':{'gpu_uuid':None}}; self.a=a
            def prepare(self,length):
                self.ids=list(range(length)); self.snapshot='source'; self.meta={'resolved_model_revision':'revision'}
                self.weights='w'; self.costs={'incremental_build_ms':999}; self.preflight={}; self.catalog=NS(resident_bytes=42)
                self.backing_check=[]
            def gpu_state(self): return {'gpu_uuid':None,'reason':'FAKE TEST, NOT GPU EXECUTION'}
            def invoke(self,mode):
                reuse=mode in ('CONTIGUOUS_WARM','ORACLE_PRESELECTED')
                return int(reuse),dict(request_ms=5+int(reuse),reused_tokens=30 if reuse else 0,
                    projection_audit={'skip':90 if reuse else 0},selection_hash='same',mapping_cache_hit=True)
            def validate_modes(self): return {'fixture':'FAKE TEST'}
        def quality(a,b): return {'correctness_status':'STRICT_PASS' if a==b else 'STRICT_FAIL'}
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict('sys.modules',{'torch':NS(equal=lambda a,b:a==b)}),patch.object(m,'CostRuntime',FakeRuntime),patch.object(m,'logit_metrics',quality):
                self.assertEqual(m.main(['--phase','minimal','--output-dir',str(Path(tmp)/'run')]),0)
            manifest=json.loads((Path(tmp)/'run/manifest.json').read_text())
            self.assertEqual(manifest['execution_status'],'COMPLETE')
            self.assertEqual(manifest['correctness_status'],'STRICT_FAIL')
            import csv
            with (Path(tmp)/'run/minimal_paired_samples.csv').open() as f: rows=list(csv.DictReader(f))
            self.assertEqual(len(rows),52)
            self.assertTrue(all(float(r['wall_ms'])<999 for r in rows))
            self.assertTrue(all(r['profiler_enabled']=='False' for r in rows))


try:
    import torch
except ImportError: torch=None

@unittest.skipIf(torch is None,'Torch unavailable: executor tensor/model tests NOT EXECUTED')
class TensorContracts(unittest.TestCase):
    def test_oracle_actual_executor_and_rows(self):
        from types import SimpleNamespace as NS
        from semcache.diagnostics.r1b3_occurrences import OccurrenceCatalog
        from semcache.diagnostics.r1b3_fast_selection import FastOccurrenceIndex
        from semcache.edgelora.batched_mapping_optimized import SourceBackingDescriptors,PreparedMappingCache
        from semcache.diagnostics.r1b3_break_even import MissingEvidence
        import semcache.models.lora_decomposition as decomposition
        import semcache.edgelora.mixed_projection as legacy_module
        class Projection(torch.nn.Linear):
            def get_base_layer(self): return self
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.projections=torch.nn.ModuleDict({r:Projection(4,4) for r in 'qkv'})
            def set_adapter(self,name): pass
            def forward(self,input_ids,**kwargs):
                x=input_ids.to(torch.float32).unsqueeze(-1).expand(-1,-1,4)
                return NS(logits=sum(m(x) for m in self.projections.values()))
        torch.manual_seed(7); model=Model().eval(); ids=list(range(8)); x=torch.tensor([ids]).float().unsqueeze(-1).expand(-1,-1,4)
        with torch.inference_mode(): captured={0:tuple(m(x).detach() for m in model.projections.values())}
        layout={0:((4,torch.float32),)*3}; catalog=OccurrenceCatalog('m','w','cpu',100000)
        catalog.add_source('source',ids,0,'user_a',captured,layout,prefill_complete=True)
        index=FastOccurrenceIndex(catalog,layout); descriptor=SourceBackingDescriptors(index)
        for coverage in (0,.5,'max'):
            rt=CostRuntime.__new__(CostRuntime); rt.a=NS(device='cpu'); rt.model=model
            rt.adapter=NS(layers=[object()],projection_modules=lambda i:dict(model.projections.items()))
            rt.tokenizer=lambda text:dict(input_ids=list(ids)); rt.text='fixture'; rt.ids=ids; rt.length=len(ids)
            rt.catalog=catalog; rt.snapshot=rt.oracle_snapshot='s'; rt.oracle_ids=tuple(ids)
            rt.hits=tuple(index.select(ids,0,'user_a','m','w',layout,coverage)[0])
            rt.plan,_=PreparedMappingCache(descriptor).get(rt.hits,ids,'user_a','m','w','cpu',torch.float32,mode='contiguous')
            with patch.object(decomposition,'validate_projection',lambda *a,**k:None):
                result,e=rt.invoke('ORACLE_PRESELECTED',capture=True)
            with patch.object(legacy_module,'validate_projection',lambda *a,**k:None),torch.inference_mode():
                with legacy_module.mixed_projection_path(rt.adapter,'user_a',rt.hits,len(ids)) as legacy_audit:
                    expected=model(input_ids=torch.tensor([ids])).logits
            self.assertTrue(torch.equal(expected,result))
            self.assertEqual(sum(r['reused_projection_rows'] for r in legacy_audit.records.values()),rt.plan.reused*3)
            self.assertEqual(e['projection_audit']['skipped_projection_rows_all_layers_roles'],rt.plan.reused*3)
            self.assertEqual(e['projection_audit']['native_projection_rows_all_layers_roles'],(len(ids)-rt.plan.reused)*3)
            self.assertTrue(all('forward' not in m.__dict__ for m in model.projections.values()))
            rt.oracle_ids=(99,)
            with self.assertRaises(MissingEvidence): rt.invoke('ORACLE_PRESELECTED')

if __name__=='__main__': unittest.main()
