"""C1 CPU evidence/routing tests; actual Torch/PEFT/CUDA paths skip explicitly."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS
import sys
import tempfile
import unittest
from unittest.mock import patch
import semcache.models.lora_decomposition as decomposition
from semcache.diagnostics.r1b3_gemm_shape import (prefix_layers,row_cost,classify_h1,NativeShapeIntervention,
    capture_call,shape_controls,assembly_controls,ReplayAudit)
from semcache.diagnostics.r1b3_break_even import MissingEvidence
ROOT=Path(__file__).resolve().parents[1]


def runner():
    spec=importlib.util.spec_from_file_location('c1_runner',ROOT/'scripts/82_verify_r1b3_b3_gemm_shape.py')
    m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


class EvidenceTests(unittest.TestCase):
    def fixture(self):
        rows=[]
        for label in ('FULL_NATIVE_REPEAT','SUBSET_NATIVE','FULL_THEN_SELECT','FULL_SHAPE_ASSEMBLY','FP32_FULL_VS_SUBSET'):
            for r in 'qkv':
                for c in ('base','lora_delta','total'):
                    different=label=='SUBSET_NATIVE' and c!='lora_delta'
                    rows.append(dict(control=label,role=r,component=c,exact_equal=not different,max_abs_error=.01 if different else 0.,
                        subset_values_equal_full_selection=True))
        controls=dict(layer0_input_exact=True,full_repeat_exact=True,source_target_qkv_exact=True,
            cache_backing_exact=True,subset_assembly_matches_ordinary=True,ordinary_strict_fail=True,shape_weight_identity_unchanged=True)
        replay=dict(full_shape_layers=list(range(4)),actual_skipped_projection_rows=0,logits=dict(exact_equal=True),
            mapping_and_source_unchanged=True,weight_identity_unchanged=True)
        return rows,controls,replay
    def test_prefixes_and_actual_rows_include_full_recomputation(self):
        self.assertEqual(prefix_layers('1,2,4,8,all',32),[1,2,4,8,32])
        self.assertEqual(prefix_layers('0,1,all,all',2),[0,1,2])
        with self.assertRaises(ValueError): prefix_layers('32',12)
        self.assertEqual(row_cost(128,2,32,[])['actual_skipped_projection_rows'],12096)
        self.assertEqual(row_cost(256,1,32,[])['actual_skipped_projection_rows'],24480)
        self.assertEqual(row_cost(128,2,32,[0])['actual_skipped_projection_rows'],11718)
        all_rows=row_cost(256,1,32,range(32))
        self.assertEqual(all_rows['actual_skipped_projection_rows'],0)
        self.assertEqual(all_rows['actual_native_projection_rows'],256*32*3)
        self.assertEqual(all_rows['reused_data_tokens'],255)
    def test_supported_causal_and_kernel_claims_are_distinct(self):
        rows,controls,replay=self.fixture()
        supported=classify_h1(rows,controls,[],4,True)
        self.assertTrue(supported['H1_A_supported']); self.assertFalse(supported['causal_contribution_isolated'])
        causal=classify_h1(rows,controls,[replay],4,True)
        self.assertTrue(causal['causal_contribution_isolated']); self.assertFalse(causal['kernel_mechanism_identified'])
        self.assertTrue(all(r['initial_base_only_evidence'] for r in causal['H1_D_base_lora_attribution']))
        self.assertFalse(classify_h1(rows,controls,[replay],4,False)['causal_contribution_isolated'])
    def test_missing_controls_or_changed_mapping_prevent_causal_claim(self):
        rows,controls,replay=self.fixture()
        for field in ('full_repeat_exact','cache_backing_exact','layer0_input_exact','source_target_qkv_exact','shape_weight_identity_unchanged'):
            altered=dict(controls); altered[field]=False
            self.assertFalse(classify_h1(rows,altered,[replay],4,True)['causal_contribution_isolated'])
        for field in ('mapping_and_source_unchanged','weight_identity_unchanged'):
            altered=dict(replay); altered[field]=False
            self.assertFalse(classify_h1(rows,controls,[altered],4,True)['causal_contribution_isolated'])
        incomplete=[r for r in rows if not (r['control']=='FULL_THEN_SELECT' and r['role']=='v')]
        self.assertFalse(classify_h1(incomplete,controls,[replay],4,True)['H1_A_supported'])
        altered=dict(replay,logits=dict(exact_equal=False))
        self.assertFalse(classify_h1(rows,controls,[altered],4,True)['causal_contribution_isolated'])
    def test_cli_failure_preserves_required_schema_and_no_overwrite(self):
        import json,io
        from contextlib import redirect_stderr
        m=runner()
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)/'run'; command=['--output-dir',str(out),'--replay-prefixes','1,2,4,8,all']
            self.assertEqual(m.parse_args(command).replay_prefixes,'1,2,4,8,all')
            with patch.object(m,'Runtime',side_effect=ModuleNotFoundError('Torch unavailable')),redirect_stderr(io.StringIO()):
                self.assertEqual(m.main(command),1)
            data=json.loads((out/'manifest.json').read_text())
            self.assertEqual(data['execution_status'],'FAILED'); self.assertEqual(data['correctness_status'],'UNVERIFIED')
            for name in ('shape_control_comparison.csv','base_lora_total_comparison.csv','counterfactual_layer_replay.csv','first_divergence_control.csv','logit_control_comparison.csv','h1_verification_summary.json'):
                self.assertTrue((out/name).exists())
            with self.assertRaises(FileExistsError): m.main(command)


class FakeTensor:
    def __init__(self,values): self.values=tuple(values); self.shape=(1,len(values),1); self.dtype='float16'; self.device='cpu'; self._version=0
    def data_ptr(self): return id(self)
    def stride(self): return (len(self.values),1,1)
    def storage_offset(self): return 0
    def index_select(self,dim,index):
        if dim!=1: raise ValueError('Only row selection in metadata fixture')
        return FakeTensor([self.values[i] for i in index])
class FakeProjection:
    def __init__(self): self.weight=FakeTensor([2]); self.rows=[]; self.fail=False
    def parameters(self): return iter([self.weight])
    def forward(self,x):
        self.rows.append(x.shape[1])
        if self.fail: raise RuntimeError('test native failure')
        return FakeTensor([v*2 for v in x.values])
class FakeTorch:
    long='long'
    @staticmethod
    def tensor(values,**kwargs): return tuple(values)
    @staticmethod
    def equal(a,b): return a.values==b.values


class RoutingTests(unittest.TestCase):
    def setUp(self):
        self.modules={i:{r:FakeProjection() for r in 'qkv'} for i in range(2)}
        self.adapter=NS(layers=[0,1],projection_modules=lambda i:self.modules[i])
    def test_exact_native_callable_routing_and_row_ledger(self):
        original=FakeProjection.forward; whole=FakeTensor(range(8)); fresh=[6,7]
        with patch.dict(sys.modules,{'torch':FakeTorch}),patch.object(decomposition,'validate_projection',lambda *args:None):
            with NativeShapeIntervention(self.adapter,[0]) as control:
                control.fresh=fresh
                for i,roles in self.modules.items():
                    for role,module in roles.items():
                        control.whole[(i,role)]=whole
                        output=module.forward(whole.index_select(1,fresh))
                        self.assertEqual(output.values,(12,14))
                        self.assertEqual(module.rows,[8 if i==0 else 2])
                        self.assertNotIn((i,role),control.whole)
                ledger=control.accounting(8,2)
                self.assertEqual(ledger['actual_native_projection_rows'],30)
                self.assertEqual(ledger['actual_skipped_projection_rows'],18)
        self.assertIs(FakeProjection.forward,original)
        self.assertFalse(any('forward' in m.__dict__ for roles in self.modules.values() for m in roles.values()))
    def test_bad_input_missing_ledger_and_failure_restore_class(self):
        original=FakeProjection.forward
        with patch.dict(sys.modules,{'torch':FakeTorch}),patch.object(decomposition,'validate_projection',lambda *args:None):
            with self.assertRaisesRegex(MissingEvidence,'Fresh input differs'):
                with NativeShapeIntervention(self.adapter,[0]) as control:
                    control.fresh=[6,7]; control.whole[(0,'q')]=FakeTensor(range(8))
                    self.modules[0]['q'].forward(FakeTensor([99,99]))
            self.assertIs(FakeProjection.forward,original)
            with NativeShapeIntervention(self.adapter,[0]) as control:
                with self.assertRaisesRegex(MissingEvidence,'Missing native'): control.accounting(8,2)
            self.modules[0]['q'].fail=True
            with self.assertRaisesRegex(RuntimeError,'native failure'):
                with NativeShapeIntervention(self.adapter,[0]) as control:
                    control.fresh=[6,7]; control.whole[(0,'q')]=FakeTensor(range(8))
                    self.modules[0]['q'].forward(FakeTensor([6,7]))
            self.assertIs(FakeProjection.forward,original)
    def test_parameter_mutation_is_detected_after_patch_restoration(self):
        original=FakeProjection.forward
        with patch.dict(sys.modules,{'torch':FakeTorch}),patch.object(decomposition,'validate_projection',lambda *args:None):
            with self.assertRaisesRegex(MissingEvidence,'weights changed'):
                with NativeShapeIntervention(self.adapter,[0]): self.modules[0]['q'].weight._version+=1
        self.assertIs(FakeProjection.forward,original)


TORCH=importlib.util.find_spec('torch') is not None
@unittest.skipUnless(TORCH,'NOT EXECUTED: Torch absent; actual tensor/assembly/model controls unavailable')
class TensorTests(unittest.TestCase):
    def fixture(self,device):
        import torch
        from semcache.diagnostics.r1b3_occurrences import OccurrenceCatalog
        from semcache.diagnostics.r1b3_fast_selection import FastOccurrenceIndex
        from semcache.edgelora.batched_mapping_optimized import SourceBackingDescriptors,PreparedMappingCache
        from semcache.diagnostics.r1b3_b3_runtime import b21
        import semcache.edgelora.mixed_projection as mixed
        class Projection(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.in_features=4; self.out_features=4; self.base=torch.nn.Linear(4,4)
                self.lora_A=torch.nn.ModuleDict({'user_a':torch.nn.Linear(4,2,bias=False)})
                self.lora_B=torch.nn.ModuleDict({'user_a':torch.nn.Linear(2,4,bias=False)})
                self.scaling={'user_a':1.}
            def get_base_layer(self): return self.base
            def forward(self,x):
                result=self.base(x)
                result+=self.lora_B['user_a'](self.lora_A['user_a'](x))
                return result
        torch.manual_seed(8); n=8; ids=list(range(n)); hidden=torch.randn(1,n,4,device=device)
        modules={r:Projection().to(device).eval() for r in 'qkv'}
        adapter=NS(model=NS(),layers=[object()],projection_modules=lambda _:modules)
        layout={0:((4,torch.float32),)*3}
        with torch.inference_mode(): qkv={0:tuple(modules[r](hidden) for r in 'qkv')}
        catalog=OccurrenceCatalog('model','weights',device,1000000)
        catalog.add_source('source',ids,0,'user_a',qkv,layout,prefill_complete=True)
        index=FastOccurrenceIndex(catalog,layout); descriptor=SourceBackingDescriptors(index)
        rt=NS(adapter=adapter,index=index,b=b21(),caches={'contiguous':PreparedMappingCache(descriptor)},
            namespace='model',weights='weights',a=NS(device=device,dtype='float32'),ids=ids,snapshot='source',original=mixed.mixed_projection_path)
        hits,mask,_=index.select(ids,0,'user_a','model','weights',layout,'max')
        return rt,modules,hidden,hits,[i for i,x in enumerate(mask) if not x]
    def run_device(self,device):
        import torch
        with patch.object(decomposition,'validate_projection',lambda *args:None):
            rt,modules,hidden,hits,fresh=self.fixture(device)
            inputs={r:hidden for r in 'qkv'}
            rows,values=shape_controls(rt.adapter,inputs,fresh)
            self.assertTrue(all(r['exact_equal'] for r in rows if r['control']=='FULL_THEN_SELECT'))
            for r in rows:
                if r['control']=='FULL_THEN_SELECT':
                    self.assertEqual(r['native_call']['native_output']['shape'][1],len(rt.ids))
                    self.assertEqual(r['native_call']['returned_output']['shape'][1],len(fresh))
                    self.assertEqual(r['native_call']['selection_device'],device)
            # Base capture must precede PEFT's in-place delta addition.
            with torch.inference_mode(): base=modules['q'].base(hidden).cpu()
            self.assertTrue(torch.equal(values['q']['base'],base))
            assembled,reports,_=assembly_controls(rt,inputs,fresh,values,hits)
            self.assertTrue(all(r['exact_equal'] for r in assembled if r['control']=='FULL_SHAPE_ASSEMBLY'))
            self.assertEqual(reports['FULL_SHAPE_ASSEMBLY']['ledger']['actual_skipped_projection_rows'],0)
            self.assertEqual(reports['SUBSET_ASSEMBLY']['ledger']['actual_skipped_projection_rows'],18)
            self.assertTrue(all(not m._forward_pre_hooks and not m._forward_hooks and 'forward' not in m.__dict__ for m in modules.values()))
    def test_actual_components_and_existing_assembly_cpu(self): self.run_device('cpu')
    def test_actual_components_and_existing_assembly_cuda(self):
        import torch
        if not torch.cuda.is_available(): self.skipTest('NOT EXECUTED: CUDA unavailable')
        self.run_device('cuda:0')

    def test_two_layer_replay_alignment_and_all_layer_logits(self):
        import torch
        from semcache.diagnostics.r1b3_fp16_divergence import StreamingAudit
        from semcache.diagnostics.r1b3_occurrences import OccurrenceCatalog
        from semcache.diagnostics.r1b3_fast_selection import FastOccurrenceIndex
        from semcache.edgelora.batched_mapping_optimized import SourceBackingDescriptors,PreparedMappingCache
        with patch.object(decomposition,'validate_projection',lambda *args:None):
            rt,modules,_,_,_=self.fixture('cpu')
            projection=type(modules['q'])
            class Attention(torch.nn.Module):
                def __init__(self,projections):
                    super().__init__()
                    for r,m in projections.items(): setattr(self,r+'_proj',m)
                def forward(self,x): return (self.q_proj(x)+self.k_proj(x)+self.v_proj(x),None)
            class Layer(torch.nn.Module):
                def __init__(self,projections): super().__init__(); self.self_attn=Attention(projections)
                def forward(self,hidden_states): return (hidden_states+self.self_attn(hidden_states)[0],)
            decoder=torch.nn.Module(); decoder.embed_tokens=torch.nn.Embedding(16,4); decoder.embed_positions=torch.nn.Embedding(16,4)
            decoder.layers=torch.nn.ModuleList([Layer(modules),Layer({r:projection().eval() for r in 'qkv'})]).eval()
            head=torch.nn.Linear(4,7).eval()
            adapter=NS(model=NS(model=NS(decoder=decoder)),layers=decoder.layers,
                projection_modules=lambda i:{r:getattr(decoder.layers[i].self_attn,r+'_proj') for r in 'qkv'})
            def forward():
                hidden=decoder.embed_tokens(torch.tensor([rt.ids]))+decoder.embed_positions(torch.arange(8)[None])
                for layer in decoder.layers: hidden=layer(hidden)[0]
                return head(hidden)
            source=StreamingAudit(adapter,'user_a',[0,1],'SOURCE',retain=True)
            reference=StreamingAudit(adapter,'user_a',[0,1],'FULL',retain=True)
            with torch.inference_mode():
                with source.hooks(): forward()
                with reference.hooks(): full=forward()
            catalog=OccurrenceCatalog('model','weights','cpu',1000000); layout={i:((4,torch.float32),)*3 for i in range(2)}
            captured={i:tuple(source.snapshots[(i,'total_qkv',r)] for r in 'qkv') for i in range(2)}
            catalog.add_source('source',rt.ids,0,'user_a',captured,layout,prefill_complete=True)
            index=FastOccurrenceIndex(catalog,layout); cache=PreparedMappingCache(SourceBackingDescriptors(index))
            hits,mask,_=index.select(rt.ids,0,'user_a','model','weights',layout,'max'); fresh=[i for i,x in enumerate(mask) if not x]
            original=projection.forward
            for prefix in (0,1,2):
                flow=rt.b.RequestMapping('contiguous',index,cache,'model','weights','cpu',torch.float32,{})
                flow.select(hits,rt.ids,'user_a')
                audit=ReplayAudit(adapter,reference.snapshots,[0,1],f'PREFIX_{prefix}')
                with NativeShapeIntervention(adapter,range(prefix)) as intervention,torch.inference_mode():
                    with flow.context(rt.original,adapter,'user_a',hits,8):
                        with intervention.observer(audit)(fresh): actual=forward()
                    cost=intervention.accounting(8,len(fresh))
                flow.report()
                for row in audit.rows:
                    if row['stage'] in ('base_projection','lora_delta'):
                        self.assertEqual(row['actual_output']['shape'][1],8 if row['layer']<prefix else len(fresh))
                if prefix==2:
                    self.assertEqual(cost['actual_skipped_projection_rows'],0)
                    self.assertTrue(torch.equal(full,actual))
                self.assertIs(projection.forward,original)
                self.assertTrue(all(not m._forward_hooks and not m._forward_pre_hooks for m in decoder.modules()))

if __name__=='__main__': unittest.main()
