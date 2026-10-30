"""CPU controls for finite gates, exact workloads and per-step failure evidence."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from semcache.edgelora.communication import ProtocolError
from semcache.edgelora.r3a_artifacts import load_matrices,read
from semcache.edgelora.r3a_evidence import jsonl
from semcache.edgelora.r3a_tiny import prepare_tiny
from semcache.edgelora.r4a_diagnostics import logit_counts
from semcache.edgelora.r4a_evidence import make_suite,validate_controlled_suite
from semcache.edgelora.r4a_fixture import TinyBackend
from semcache.edgelora.r4a_generation import ForwardDriver,generate,generation_policy,past_summary
from semcache.edgelora.r4a_session import context


class ByteTokenizer:
    """Explicit deterministic tokenizer control, not evidence for the OPT tokenizer."""
    bos_token_id=2;eos_token_id=2;pad_token_id=1
    def __init__(self):self.calls=[]
    def __call__(self,text,**kwargs):
        self.calls.append(kwargs)
        ids=[c+10 for c in text.encode()]
        if kwargs['truncation']:ids=ids[:kwargs['max_length']]
        return dict(input_ids=ids,attention_mask=[1]*len(ids))


class ControlledWorkload(unittest.TestCase):
    def test_all_matrix_prompts_have_exact_real_budgets_and_deterministic_distinct_sources(self):
        tokenizer=ByteTokenizer();suite=make_suite(tokenizer,scope='matrix');again=make_suite(ByteTokenizer(),scope='matrix')
        self.assertEqual(suite['padding'],'none');self.assertFalse(suite['add_special_tokens'])
        self.assertEqual(len(suite['prompts']),8)
        for a,b in zip(suite['prompts'],again['prompts']):
            n=a['length'];self.assertIn(n,(32,128));self.assertEqual(len(a['token_ids']),n)
            self.assertEqual(a['attention_mask'],[1]*n);self.assertEqual(a['nonpadding_tokens'],n);self.assertEqual(a['padding_tokens'],0)
            self.assertGreater(a['source_token_count'],128);self.assertEqual(a['source_text'],b['source_text'])
            self.assertEqual(a['token_ids'],b['token_ids']);self.assertEqual(a['prompt_sha256'],b['prompt_sha256'])
        self.assertEqual(len({p['source_text'] for p in suite['prompts']}),4)
        self.assertEqual(len({p['prompt_sha256'] for p in suite['prompts']}),8)
        self.assertTrue(all(c['padding'] is False and c['add_special_tokens'] is False for c in tokenizer.calls))

    def test_short_source_padding_special_ids_and_collapsed_prompts_fail_closed(self):
        class Short(ByteTokenizer):
            def __call__(self,text,**kwargs):return dict(input_ids=[9]*17,attention_mask=[1]*17)
        class Padded(ByteTokenizer):
            def __call__(self,text,**kwargs):
                out=super().__call__(text,**kwargs)
                if kwargs['truncation']:out['attention_mask'][0]=0
                return out
        class Special(ByteTokenizer):
            def __call__(self,text,**kwargs):
                out=super().__call__(text,**kwargs);out['input_ids'][0]=1;return out
        class Collapsed(ByteTokenizer):
            def __call__(self,text,**kwargs):
                n=kwargs.get('max_length',256);return dict(input_ids=[9]*n,attention_mask=[1]*n)
        for tokenizer in (Short(),Padded(),Special(),Collapsed()):
            with self.subTest(tokenizer=type(tokenizer).__name__),self.assertRaises(ProtocolError):make_suite(tokenizer)
        suite=make_suite(ByteTokenizer());suite['prompts'][0]['attention_mask'][0]=0
        with self.assertRaises(ProtocolError):validate_controlled_suite(suite)


def record(mask,n,step=0):
    old=len(mask)-n
    return dict(forward_id=f'UNIT.s{step:02d}',context=context('UNIT',step,[9]*n,mask,old,3),
        cache_positions=list(range(old,len(mask))),input_past=None if not old else dict(sequence_length=old),
        past=dict(sequence_length=len(mask)),position_observation=dict(source='UNIT_CONTROL'),model_logits_dtype='float32')


class NonfiniteDiagnostics(unittest.TestCase):
    def test_padded_valid_and_final_nonfinites_are_distinct_and_always_fatal(self):
        # Case 0 has a finite final vector: padding-only NaN must still abort.
        mask=[0]*15+[1]*17  # Reported text_0 L32 geometry; logits are synthetic controls.
        cases=[([float('nan')]+[0.]*95,'padded_position_logits','nan'),
            ([0.]*45+[float('inf')]+[0.]*50,'valid_position_logits','positive_inf'),
            ([0.]*95+[float('-inf')],'last_position_logits','negative_inf')]
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'diagnostics.jsonl'
            driver=ForwardDriver({},'cpu',tiny=SimpleNamespace(heads=1,vocab=3),native=True,diagnostics_path=path)
            for values,group,kind in cases:
                out=SimpleNamespace(logits=values,shape=[1,32,3]);original=list(values)
                with self.subTest(group=group),self.assertRaisesRegex(ProtocolError,'nonfinite actual model logits.*sample_id'):
                    driver.last_logits(out,sample_id='text_0_L32',record=record(mask,32),mask=mask)
                self.assertEqual([repr(v) for v in values],[repr(v) for v in original])
                diagnostic=jsonl(path)[-1]
                self.assertEqual(diagnostic[group][kind],1);self.assertEqual(diagnostic['all_logits']['nonfinite'],1)
                self.assertEqual(diagnostic['padding_tokens'],15);self.assertEqual(diagnostic['nonpadding_tokens'],17)
                self.assertEqual(diagnostic['generation_step'],0);self.assertEqual(diagnostic['sample_id'],'text_0_L32')
                self.assertEqual(diagnostic['total_input_length'],32)
                if group=='padded_position_logits':
                    self.assertEqual(diagnostic['last_position_logits']['nonfinite'],0)
                    self.assertEqual(diagnostic['valid_position_logits']['nonfinite'],0)
            self.assertEqual(len(jsonl(path)),3)
            json.loads(path.read_text().splitlines()[-1],parse_constant=lambda s:self.fail('nonfinite JSON literal: '+s))

    def test_last_vector_selection_for_finite_padded_and_unpadded_prompts_and_decode(self):
        driver=ForwardDriver({},'cpu',tiny=SimpleNamespace(heads=1,vocab=3),native=True)
        for mask in ([1]*4,[0,0,1,1]):
            rec=record(mask,4);out=SimpleNamespace(logits=list(range(12)),shape=[1,4,3])
            values,blob=driver.last_logits(out,sample_id='unit',record=rec,mask=mask)
            self.assertEqual(values,[9,10,11]);self.assertEqual(len(blob),12)
        mask=[0,0,1,1,1];rec=record(mask,1,1);out=SimpleNamespace(logits=[1.,2.,3.],shape=[1,1,3])
        driver.last_logits(out,sample_id='unit',record=rec,mask=mask)
        d=rec['logit_diagnostic'];self.assertEqual(d['padding_tokens'],2);self.assertEqual(d['query_padding_tokens'],0)
        self.assertEqual(d['positions'],[2]);self.assertEqual(d['cache_positions'],[4])
        self.assertEqual(d['padded_position_logits']['elements'],0);self.assertEqual(d['valid_position_logits']['elements'],3)
        with self.assertRaises(ProtocolError):logit_counts([1.]*6,[1,2,3],[1,0],torch_backend=False)

    def test_prefill_success_then_decode_failure_retains_both_steps_and_stops_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=prepare_tiny(Path(tmp)/'base');pm=read(base/'r3a_prepare_manifest.json');am=read(base/'r3a_adapter_manifest.json')
            class DecodeFailure(TinyBackend):
                def forward(self,tokens,mask,past=None,*args,**kwargs):
                    out=super().forward(tokens,mask,past,*args,**kwargs)
                    if past is not None:out.logits[-1]=float('inf')
                    return out
                def decode_text(self,tokens):raise AssertionError('failed generation must never finalize text')
            tiny=DecodeFailure(read(base/'r3a_tiny_base.json'),load_matrices(base,am,torch_backend=False))
            driver=ForwardDriver(pm['identity'],'cpu',tiny=tiny,native=True,diagnostics_path=Path(tmp)/'steps.jsonl')
            prompt=dict(sample_id='decode_failure',token_ids=[1]*32,attention_mask=[1]*32,prompt_sha256='UNIT')
            policy=generation_policy(dict(bos_token_id=0,eos_token_id=6,pad_token_id=0),'fixed',3)
            emitted=[]
            with self.assertRaisesRegex(ProtocolError,'nonfinite actual model logits'):generate(driver,prompt,policy,'UNIT',on_token=emitted.append)
            diagnostics=jsonl(driver.diagnostics_path);self.assertEqual(len(diagnostics),2)
            self.assertEqual([d['generation_step'] for d in diagnostics],[0,1])
            self.assertEqual([d['all_logits']['nonfinite'] for d in diagnostics],[0,1])
            self.assertEqual(diagnostics[1]['total_input_length'],33);self.assertEqual(diagnostics[1]['nonpadding_tokens'],33)
            self.assertEqual(diagnostics[1]['input_length'],1);self.assertEqual(diagnostics[1]['positions'],[32])
            self.assertEqual(diagnostics[1]['input_past']['sequence_length'],32)
            self.assertEqual(diagnostics[1]['output_past']['sequence_length'],33)
            self.assertEqual(len(emitted),1)  # No selection/callback occurs for the nonfinite decode.

    def test_cache_reported_length_must_match_actual_tensors(self):
        class Cache:
            def get_seq_length(self):return 99
        with self.assertRaisesRegex(ProtocolError,'cache-reported sequence length'):past_summary(Cache(),32,32,1,4,'float32','cpu')


@unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('transformers'),
    'optional actual HF OPT CPU test requires Torch/Transformers; no download')
class ActualOPTCPU(unittest.TestCase):
    def test_actual_opt_padded_unpadded_positions_kv_and_multi_step_logits(self):
        import torch
        from transformers import OPTConfig,OPTForCausalLM
        torch.manual_seed(42)
        config=OPTConfig(vocab_size=67,hidden_size=16,word_embed_proj_dim=16,ffn_dim=32,num_hidden_layers=2,
            num_attention_heads=2,max_position_embeddings=160,dropout=0.,attention_dropout=0.,layerdrop=0.,
            pad_token_id=1,bos_token_id=2,eos_token_id=2)
        config._attn_implementation='eager';model=OPTForCausalLM(config).eval()
        identity=dict(layers=2,hidden_size=16,hidden_dtype='float32',precision='CPU_UNIT_OPT_RANDOM_WEIGHTS')
        driver=ForwardDriver(identity,'cpu',model=model,native=True)
        real=[3+i%60 for i in range(32)]
        for pads in (0,3):
            ids=[1]*pads+real;mask=[0]*pads+[1]*32;past=None;selected=[]
            for step in range(3):
                inputs=ids if step==0 else [selected[-1]]
                out,rec,_=driver.forward(f'CPU_{pads}',step,inputs,mask,past,3)
                values,_=driver.last_logits(out,sample_id=f'pads_{pads}',record=rec,mask=mask)
                with torch.inference_mode():
                    complete=model(input_ids=torch.tensor([ids+selected]),attention_mask=torch.tensor([mask]),use_cache=False).logits
                torch.testing.assert_close(out.logits[0,-1],complete[0,-1],rtol=1e-5,atol=1e-6)
                self.assertEqual(rec['position_observation']['positions'],list(range(32)) if step==0 and not pads else
                    [-1]*pads+list(range(32)) if step==0 else [31+step])
                self.assertTrue(rec['position_observation']['lookup_output_equal'])
                self.assertEqual(rec['past']['sequence_length'],len(mask))
                selected.append(max(range(len(values)),key=values.__getitem__));past=out.past_key_values;mask.append(1)
            if pads==0:unpadded=selected
            else:self.assertEqual(selected,unpadded)


if __name__=='__main__':unittest.main()
