"""Geometry/skip/runner tests without Torch; actual tiny OPT/PEFT is offline and gated."""
from types import SimpleNamespace as NS
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from semcache.diagnostics.r2b1_prefix_audit import geometry,validate_forward_rows,compare_suffix
from semcache.diagnostics.r2b1_prefix_cache import PrefixError
from semcache.diagnostics.r1b3_break_even import schedule
from semcache.diagnostics.r1b3_fp16_divergence import scalar_metrics
from test_r2b1_prefix_cache import identity
ROOT=Path(__file__).resolve().parents[1]


def runner():
    spec=importlib.util.spec_from_file_location('prefix_runner_test',ROOT/'scripts/85_run_r2b1_exact_prefix.py')
    m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


class FakeLogits:
    ndim=3
    def __init__(self,positions): self.positions=list(positions); self.shape=(1,len(self.positions),3)
    def __getitem__(self,index): return FakeLogits(self.positions[index[1]])


class ContractTests(unittest.TestCase):
    def test_geometry_and_physical_missing_evidence(self):
        for n,p in ((32,16),(32,31),(32,0),(128,64),(128,127),(256,192)):
            g=geometry(n,p,512); self.assertEqual(g['suffix_length'],n-p)
            rows=[dict(layer=l,role=role,batch=1,tokens=n-p) for l in range(2) for role in ('layer','q','k','v','fc1','fc2','attention')]
            self.assertEqual(validate_forward_rows(rows,2,n-p)['status'],'VERIFIED')
            with self.assertRaises(PrefixError): validate_forward_rows(rows[:-1],2,n-p)
            rows[0]['tokens']=n+1
            with self.assertRaises(PrefixError): validate_forward_rows(rows,2,n-p)
        for n,p in ((32,32),(0,0),(32,-1)):
            with self.assertRaises(PrefixError): geometry(n,p)
    def test_alignment_and_strict_fail_not_argmax(self):
        with patch('semcache.diagnostics.r2b1_prefix_audit.logit_metrics',side_effect=lambda a,b:dict(ref=a.positions,actual=b.positions)):
            r=compare_suffix(FakeLogits(range(8)),FakeLogits(range(3,8)),3)
            self.assertEqual(r['suffix']['ref'],list(range(3,8)))
            self.assertEqual(r['last_position']['ref'],[7])
        self.assertFalse(scalar_metrics([1.,2.,3.],[1.02,2.02,3.02])['allclose'])
    def test_pair_sign_and_source_not_added(self):
        m=runner(); samples=[]
        for i in range(3):
            for mode,cost in zip(m.MODES,(20,10)):
                samples.append(dict(case='c',phase='measured',pair_index=i,mode=mode,wall_ms=cost,total_length=32,prefix_length=16,suffix_length=16))
        s=m.summarize(samples,{'c':{'prefix_prefill_ms':999}})[0]
        self.assertEqual(s['paired_difference']['mean'],10); self.assertEqual(s['warm']['mean'],10)
        self.assertFalse(s['speedup_claim_qualified'])
        with self.assertRaises(PrefixError): m.summarize(samples[:-1],{'c':{}})
        orders=schedule(m.MODES,4,99)
        self.assertEqual(orders,schedule(m.MODES,4,99)); self.assertEqual([x[0] for x in orders].count(m.MODES[0]),2)
    def test_cli_artifacts_failure_and_collision(self):
        m=runner()
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'fail'
            with patch.object(m,'run',side_effect=PrefixError('UNSUPPORTED_CACHE_API','no compatible converter')):
                self.assertEqual(m.main(['--output-dir',str(out)]),1)
            manifest=json.loads((out/'manifest.json').read_text())
            self.assertEqual(manifest['execution_status'],'UNSUPPORTED'); self.assertEqual(manifest['classification'],'EXACT_PREFIX_BASELINE')
            self.assertEqual(manifest['correctness_status'],'UNVERIFIED'); self.assertEqual(manifest['implementation_status'],'IMPLEMENTED_LOCAL_ONLY')
            for name in m.ARTIFACTS: self.assertTrue((out/(name+'.json')).exists())
            with self.assertRaises(FileExistsError): m.main(['--output-dir',str(out)])
        self.assertEqual(m.parse_args(['--cases','32:16,32:31,32:0','--output-dir','unused']).case_values,[(32,16),(32,31),(32,0)])
        with self.assertRaises(SystemExit): m.parse_args(['--prefix-length','32','--output-dir','unused'])
    def test_complete_diagnostic_retains_strict_fail(self):
        m=runner()
        class Runtime:
            def __init__(self,a): self.provenance={'fixture':'FAKE_CONTROL_FLOW_ONLY'}; self.meta={'resolved_model_revision':'rev'}; self.adapter=NS(layers=[1]); self.snapshot=None
            def prepare(self,n,p):
                self.ids=list(range(n)); self.text='x'; self.geometry=geometry(n,p); self.source_forwards=0; self.preflight={}; self.source_report={'status':'NO_PREFIX_CONTROL'}
            def identity(self): return identity()
            def forward(self,path,audit=False):
                return dict(logits=0,audit={'fixture':'FAKE'},output_kv_length=32,request_input_after_lengths=[],prepared_input_audit=None,output_prefix_check=None,
                    cache_api='fake',wall_ms=10,cache_prepare_host_ms=None,cache_prepare_scope='fake',request_clone_bytes=0,query_tokens=32,past_tokens=0,cache_hit=False,memory={})
            def check(self): return {'status':'NO_PREFIX_CONTROL'}
            def negative_tests(self): return []
            def gpu_state(self): return {'status':'NOT_EXECUTED'}
            def verify_weight_hashes(self): return {'status':'FAKE'}
        metric=dict(allclose=False,correctness_status='STRICT_FAIL')
        with tempfile.TemporaryDirectory() as tmp,patch.dict('sys.modules',{'torch':NS(equal=lambda a,b:a==b)}),patch.object(m,'PrefixRuntime',Runtime),patch.object(m,'logit_metrics',return_value=metric),patch.object(m,'compare_suffix',return_value={'suffix':metric,'last_position':metric}):
            out=Path(tmp)/'run'; self.assertEqual(m.main(['--prefix-length','0','--output-dir',str(out)]),0)
            manifest=json.loads((out/'manifest.json').read_text())
            self.assertEqual(manifest['execution_status'],'COMPLETE'); self.assertEqual(manifest['correctness_status'],'STRICT_FAIL')
            self.assertFalse(json.loads((out/'decision_summary.json').read_text())['speedup_claim_qualified'])


DEPS=all(importlib.util.find_spec(name) is not None for name in ('torch','transformers','peft'))
@unittest.skipUnless(DEPS,'Torch/Transformers/PEFT unavailable: real tiny OPT integration NOT EXECUTED')
class OfflineOPTTests(unittest.TestCase):
    def runtime(self,device='cpu'):
        import torch
        from transformers import OPTConfig,OPTForCausalLM
        from semcache.models.lora_fixtures import create_controlled_users
        from semcache.diagnostics.r2b1_prefix_runtime import PrefixRuntime
        cfg=OPTConfig(vocab_size=64,hidden_size=16,word_embed_proj_dim=16,ffn_dim=32,num_hidden_layers=2,num_attention_heads=2,
            max_position_embeddings=64,dropout=0.,attention_dropout=0.,layerdrop=0.,bos_token_id=2,pad_token_id=1)
        cfg._attn_implementation='eager'; torch.manual_seed(7)
        base=OPTForCausalLM(cfg).to(device).eval(); model,fixture=create_controlled_users(base); model.set_adapter('user_a')
        class Tokenizer:
            all_special_ids=[1,2]
            def __call__(self,text): return {'input_ids':[2]+[3+sum(map(ord,w))%60 for w in text.split()]}
        a=NS(adapter='user_a',cache_device=device,device=device,dtype='float32',seed=42)
        metadata=dict(model='SYNTHETIC_TINY_OPT_NO_HUB',resolved_model_revision='synthetic',tokenizer='test',resolved_tokenizer_revision='synthetic')
        r=PrefixRuntime.__new__(PrefixRuntime); r.initialize(model,Tokenizer(),metadata,fixture,a); return r
    def exercise(self,device):
        import torch
        r=self.runtime(device)
        for p in (16,31,0):
            r.prepare(32,p)
            full=r.forward('FULL_USE_CACHE',audit=True)
            source_count=r.source_forwards
            for i in range(3):
                r.check(); h=r.forward('EXACT_PREFIX_WARM',audit=True); r.check()
                q=compare_suffix(full['logits'],h['logits'],p)
                self.assertTrue(q['suffix']['allclose']); self.assertEqual(h['query_tokens'],32-p)
                self.assertEqual(h['output_kv_length'],32); self.assertEqual(r.source_forwards,source_count)
                self.assertEqual(h['audit']['root_forward_calls'],1)
            if p:
                codes={row['code'] for row in r.negative_tests() if row.get('status')=='REJECTED'}
                self.assertIn('REJECT_ADAPTER_MISMATCH',codes); self.assertIn('REJECT_PREFIX_TOKEN_MISMATCH',codes)
            r.verify_weight_hashes()
    def test_cpu_random_opt_peft(self): self.exercise('cpu')
    def test_cuda_random_opt_peft(self):
        import torch
        if not torch.cuda.is_available(): self.skipTest('CUDA unavailable')
        self.exercise('cuda:0')

if __name__=='__main__': unittest.main()
