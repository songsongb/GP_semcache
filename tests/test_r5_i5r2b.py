"""CPU synthetic arithmetic, bounded capture and immutable preparation tests."""
from copy import deepcopy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock,patch

from semcache.edgelora.r3a_artifacts import pack,ROOT
from semcache.edgelora.r3a_tiny import prepare_tiny
from semcache.edgelora.r3b_fixture import ArrayTensor
from semcache.edgelora.r4c_prepare import TinyTokenizer
from semcache.edgelora.r5_i5_artifacts import prepare as adapters
from semcache.edgelora.r5_multi_artifacts import validate,tokenizer_identity
from semcache.edgelora.r5_i5r1_plan import prepare as old_prepare
from semcache.edgelora.r5_i5r2b_plan import prepare,validate_plan,VERSION
from semcache.edgelora.r5_i5r2b_capture import Capture,operation_mode,installed_observer_check,model_hooks
from semcache.edgelora.r5_i5r2b_analysis import classify
from semcache.edgelora.r5_i5r2_tensors import metrics,Evidence
from test_r5_i5 import dataset


def config(**kw):
    return dict(version=VERSION,source_request_id='S',target_request_id='F',capture_bytes=32*2**20,max_files=128,TEST_FIXTURE_ONLY=True,**kw)


def equal_boundaries():
    names=['layer_input','norm_input','norm_output','attention_output']+[f'{n}_{r}' for r in 'qkv' for n in
        ('projection_input','base','delta_on_ES','CPU_input','CPU_A','CPU_B_input','CPU_B','CPU_scaled','sum_before_cast','total')]
    return [dict(boundary=name,metrics={'bitwise_equal':True}) for name in names]


class Classification(unittest.TestCase):
    def changed(self,*names):
        rows=equal_boundaries()
        for r in rows:
            if r['boundary'] in names:r['metrics']['bitwise_equal']=False
        return classify(rows)
    def test_case_A_input_difference_prevents_blame_on_linear(self):
        r=self.changed('projection_input_q','base_q')
        self.assertIn('CASE_A_INPUT_HIDDEN_DIFFERS',r['cases']);self.assertNotIn('CASE_B_GPU_BASE_CANDIDATE',r['cases'])
    def test_case_B_matching_input_different_base(self):
        r=self.changed('base_k');self.assertIn('CASE_B_GPU_BASE_CANDIDATE',r['cases'])
        self.assertEqual(r['FIRST_CAUSALLY_LOCALIZED_BOUNDARY'][0]['boundary'],'NATIVE_GPU_LINEAR')
        self.assertFalse(r['ROOT_CAUSE_CONFIRMED'])
    def test_case_C_matching_base_different_lora(self):
        r=self.changed('CPU_A_q','CPU_B_q','CPU_scaled_q','delta_on_ES_q')
        self.assertIn('CASE_C_CPU_OR_TRANSFER_CANDIDATE',r['cases'])
        self.assertTrue(any(x['boundary']=='CPU_A_q' for x in r['FIRST_CAUSALLY_LOCALIZED_BOUNDARY']))
    def test_case_D_addition_and_cast_separated(self):
        r=self.changed('sum_before_cast_v','total_v');self.assertIn('CASE_D_ADDITION_OR_CAST_CANDIDATE',r['cases'])
        self.assertTrue(any(x['boundary']=='sum_before_cast_v' for x in r['FIRST_CAUSALLY_LOCALIZED_BOUNDARY']))
        r=self.changed('total_v');self.assertTrue(any(x['boundary']=='total_v' for x in r['FIRST_CAUSALLY_LOCALIZED_BOUNDARY']))
    def test_case_E_attention_is_candidate_not_root_cause(self):
        r=self.changed('attention_output');self.assertIn('CASE_E_ATTENTION_CANDIDATE',r['cases']);self.assertEqual(r['ROOT_CAUSE'],'UNRESOLVED')
    def test_case_F_incomplete_is_not_success(self):
        r=classify([],missing=['sum']);self.assertIn('CASE_F_INCOMPLETE_CAPTURE',r['cases'])
    def test_earliest_difference_is_upstream_of_total(self):
        r=self.changed('CPU_input_q','CPU_A_q','total_q')
        self.assertEqual(r['FIRST_OBSERVED_DIFFERENCE']['boundary'],'CPU_input_q')
    def test_normalized_input_already_different_never_localizes_CPU_kernel(self):
        r=self.changed('CPU_input_q','CPU_A_q','CPU_B_input_q','CPU_B_q')
        self.assertFalse(any(x['boundary']=='CPU_A_q' for x in r['FIRST_CAUSALLY_LOCALIZED_BOUNDARY'] if isinstance(x,dict)))
    def test_identical_norm_input_and_different_output_localizes_boundary_only(self):
        r=self.changed('norm_output','projection_input_q')
        self.assertTrue(any(x['boundary']=='LAYER_0_PRE_ATTENTION_NORMALIZATION' for x in r['FIRST_CAUSALLY_LOCALIZED_BOUNDARY']))
        self.assertFalse(r['ROOT_CAUSE_CONFIRMED'])
    def test_actual_injected_tensor_values_and_frozen_thresholds(self):
        baseline=pack([1.]*256,'float32');perturbed=pack([1.02]+[1.]*255,'float32')
        for boundary,expected in (('projection_input_q','CASE_A_INPUT_HIDDEN_DIFFERS'),('base_q','CASE_B_GPU_BASE_CANDIDATE'),
                ('delta_on_ES_q','CASE_C_CPU_OR_TRANSFER_CANDIDATE'),('total_q','CASE_D_ADDITION_OR_CAST_CANDIDATE')):
            rows=equal_boundaries()
            for row in rows:row['metrics']=metrics(baseline,perturbed if row['boundary']==boundary else baseline,'float32')
            result=classify(rows);self.assertIn(expected,result['cases'])
            self.assertFalse(result['FIRST_OBSERVED_DIFFERENCE']['metrics']['activation_tolerance_pass'])
            self.assertEqual(result,classify(list(reversed(rows))))
    def test_transfer_boundary_precedes_downstream_CPU_and_total(self):
        rows=equal_boundaries()+[dict(boundary='wire_recv_HIDDEN',metrics={'bitwise_equal':False})]
        for row in rows:
            if row['boundary'] in ('CPU_input_q','total_q'):row['metrics']['bitwise_equal']=False
        self.assertEqual(classify(rows)['FIRST_OBSERVED_DIFFERENCE']['boundary'],'wire_recv_HIDDEN')


class Captures(unittest.TestCase):
    def store(self,path,**kw):
        writer=Mock();writer.bytes.side_effect=lambda p,b,**args:(p.parent.mkdir(parents=True,exist_ok=True),p.write_bytes(b))
        return Capture(path,writer,dict(config(),**kw),'SYNTHETIC'),writer
    def test_prefix_only_and_no_suffix_inclusion(self):
        with tempfile.TemporaryDirectory() as d:
            s,w=self.store(Path(d));s.select('F',0,256)
            result=s.save('input',ArrayTensor(list(range(256*4)),[1,256,4]))
            self.assertEqual(result['shape'],[1,64,4]);self.assertEqual(result['semantic_positions'],[0,64])
            self.assertEqual((Path(d)/result['file']).stat().st_size,64*4*4)
    def test_HIT_suffix_is_explicitly_omitted(self):
        with tempfile.TemporaryDirectory() as d:
            s,w=self.store(Path(d));s.select('F',64,192)
            self.assertIsNone(s.save('base',ArrayTensor([1.]*(192*4),[1,192,4])))
            self.assertIn('NO_PREFIX_QUERY_ROWS',s.entries['F/base']['reason']);w.bytes.assert_not_called()
    def test_byte_budget_fails_before_write(self):
        with tempfile.TemporaryDirectory() as d:
            s,w=self.store(Path(d),capture_bytes=4);s.select('S',0,64)
            with self.assertRaisesRegex(ValueError,'BUDGET'):s.save('base',ArrayTensor([1.]*256,[1,64,4]))
            w.bytes.assert_not_called()
    def test_file_budget_and_duplicate_boundary(self):
        with tempfile.TemporaryDirectory() as d:
            s,w=self.store(Path(d),max_files=1);s.select('S',0,64);t=ArrayTensor([1.]*256,[1,64,4]);s.save('base',t)
            with self.assertRaisesRegex(ValueError,'duplicate'):s.save('base',t)
            with self.assertRaisesRegex(ValueError,'BUDGET'):s.save('delta',t)
    def test_representation_shape_dtype_validation(self):
        with tempfile.TemporaryDirectory() as d:
            s,w=self.store(Path(d));s.select('S',0,64)
            with self.assertRaises(ValueError):s.blob('invalid',b'x',[1,64,4],'float16','cpu',1)
    def test_recorded_tensor_sha_and_corruption_are_verified(self):
        with tempfile.TemporaryDirectory() as d:
            s,w=self.store(Path(d));s.select('S',0,64);item=s.save('base',ArrayTensor([1.]*256,[1,64,4]))
            ev=Evidence(d);self.assertEqual(ev.tensor('',item),pack([1.]*256,'float32'))
            (Path(d)/item['file']).write_bytes(pack([2.]*256,'float32'))
            with self.assertRaisesRegex(ValueError,'corrupt'):ev.tensor('',item)
    def test_validated_layout_metadata_merges_without_duplicate_keywords(self):
        with tempfile.TemporaryDirectory() as d:
            s,w=self.store(Path(d));s.select('S',0,64)
            layout=dict(observed_stride=[1024,8,2],original_contiguous=False,storage_offset=4)
            item=s.blob('input',pack([1.]*256,'float32'),[1,64,4],'float32','cpu',1,observed=layout,actual_operation='linear')
            self.assertEqual({k:item[k] for k in layout},layout);self.assertEqual(item['actual_operation'],'linear')
            self.assertEqual(item['shape'],[1,64,4]);self.assertEqual(item['semantic_positions'],[0,64])
    def test_observed_metadata_cannot_override_tensor_identity(self):
        for key in ('shape','observed_shape','dtype','sha256','file','device','status','token_axis','semantic_positions','scale'):
            with self.subTest(key=key),tempfile.TemporaryDirectory() as d:
                s,w=self.store(Path(d));s.select('S',0,64)
                with self.assertRaisesRegex(ValueError,'observed capture metadata'):
                    s.blob('input',pack([1.]*256,'float32'),[1,64,4],'float32','cpu',1,observed={key:None})
                w.bytes.assert_not_called();self.assertEqual(s.files,0)
    def test_extra_metadata_cannot_override_identity_or_layout(self):
        for key in ('shape','dtype','device','sha256','file','status','token_axis','semantic_positions','observed_shape',
                'observed_stride','original_contiguous','storage_offset','observed','unknown_annotation','data'):
            with self.subTest(key=key),tempfile.TemporaryDirectory() as d:
                s,w=self.store(Path(d));s.select('S',0,64)
                with self.assertRaisesRegex(ValueError,'extra capture metadata'):
                    s.save('input',ArrayTensor([1.]*256,[1,64,4]),**{key:None})
                w.bytes.assert_not_called();self.assertEqual(s.entries,{})
    def test_invalid_layout_values_fail_before_writing(self):
        for layout in ([],{'observed_stride':[1,2]}, {'observed_stride':[1,-2,3]}, {'observed_stride':[True,2,3]},
                {'original_contiguous':1},{'storage_offset':-1},{'storage_offset':True}):
            with self.subTest(layout=layout),tempfile.TemporaryDirectory() as d:
                s,w=self.store(Path(d));s.select('S',0,64)
                with self.assertRaisesRegex(ValueError,'invalid observed'):
                    s.blob('input',pack([1.]*256,'float32'),[1,64,4],'float32','cpu',1,observed=layout)
                w.bytes.assert_not_called()
    def test_every_existing_extra_annotation_is_disjoint_and_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            s,w=self.store(Path(d));s.select('S',0,64)
            extras=dict(actual_operation='add',allow_negative_infinity=True,full_payload_sha256='a'*64,
                full_payload_shape=[1,64,4],full_payload_bytes=1024,scale=1.)
            item=s.save('input',ArrayTensor([1.]*256,[1,64,4]),**extras)
            self.assertEqual({k:item[k] for k in extras},extras)
    def test_OFF_never_imports_or_enters_torch_mode(self):
        with patch.dict('sys.modules',{'torch.overrides':None}):
            spy=Mock()
            with operation_mode(spy,enabled=False):value=3+4
            self.assertEqual(value,7);spy.assert_not_called()
    @unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('numpy'),'installed Torch observer CPU API NOT_RUN; dependencies absent')
    def test_actual_installed_Torch_mode_preserves_single_original_calls(self):
        self.assertEqual(installed_observer_check()['status'],'CPU_API_PASS_NOT_GPU_FIDELITY')


@unittest.skipUnless(all(importlib.util.find_spec(n) for n in ('torch','numpy')),
    'real CPU Torch capture NOT_RUN; torch/numpy absent')
class RealTensorCaptures(unittest.TestCase):
    store=Captures.store
    def check_tensor(self,noncontiguous):
        import torch
        from semcache.edgelora.r3b_cache import raw
        for dtype in (torch.float16,torch.float32):
            with self.subTest(dtype=dtype),tempfile.TemporaryDirectory() as d:
                base=torch.arange(1*257*8,dtype=dtype,device='cpu').reshape(1,257,8)
                tensor=base[:,1::2,::2] if noncontiguous else base[:,:256].clone()
                before=tensor.clone();stride=list(tensor.stride());offset=tensor.storage_offset()
                s,w=self.store(Path(d));s.select('F',0,tensor.shape[1]);item=s.save('input',tensor)
                self.assertEqual(item['observed_stride'],stride)
                self.assertEqual(item['original_contiguous'],tensor.is_contiguous())
                self.assertEqual(item['original_contiguous'],not noncontiguous)
                self.assertEqual(item['storage_offset'],offset)
                self.assertEqual(item['observed_shape'],list(tensor.shape));self.assertEqual(item['shape'],[1,64,tensor.shape[2]])
                self.assertEqual(item['semantic_positions'],[0,64]);self.assertEqual(item['device'],'cpu')
                self.assertEqual(Evidence(d).tensor('',item),raw(tensor[:,:64]))
                self.assertTrue(torch.equal(tensor,before));self.assertEqual(list(tensor.stride()),stride)
                if noncontiguous:self.assertGreater(offset,0)
    def test_save_real_contiguous_CPU_tensor(self):self.check_tensor(False)
    def test_save_real_noncontiguous_CPU_tensor_with_storage_offset(self):self.check_tensor(True)


@unittest.skipUnless(all(importlib.util.find_spec(n) for n in ('torch','numpy','transformers')),
    'actual OPT model-hook capture CPU NOT_RUN; torch/numpy/transformers absent')
class RealModelHookCapture(unittest.TestCase):
    store=Captures.store
    def test_actual_OPT_hook_path_preserves_forward_and_records_layout(self):
        import torch
        from transformers import OPTConfig,OPTForCausalLM
        cfg=OPTConfig(vocab_size=17,hidden_size=16,word_embed_proj_dim=16,ffn_dim=32,num_hidden_layers=1,
            num_attention_heads=4,max_position_embeddings=512,dropout=0.,attention_dropout=0.)
        cfg._attn_implementation='eager';model=OPTForCausalLM(cfg).eval()
        driver=SimpleNamespace(fixture=False,model=model,adapter=SimpleNamespace(layers=model.model.decoder.layers))
        ids=torch.tensor([[1+i%15 for i in range(256)]],device='cpu')
        with tempfile.TemporaryDirectory() as d,torch.inference_mode():
            s,w=self.store(Path(d));s.select('F',0,256)
            args=dict(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=True,output_attentions=False)
            baseline=model(**args).logits.clone()
            with model_hooks(driver,s):observed=model(**args).logits
            self.assertTrue(torch.equal(observed,baseline))
            expected=('token_embedding','position_embedding','layer_input','norm_input','norm_output','actual_attention_mask','attention_output')
            for name in expected:
                item=s.entries['F/'+name];self.assertEqual(item['status'],'CAPTURED');self.assertEqual(item['semantic_positions'],[0,64])
                self.assertIsInstance(item['observed_stride'],list);self.assertIs(type(item['original_contiguous']),bool)
                self.assertIs(type(item['storage_offset']),int);Evidence(d).tensor('',item)
            self.assertTrue(all(not module._forward_hooks and not module._forward_pre_hooks for module in model.modules()))


def fixture_inputs(root):
    base=prepare_tiny(root/'base');multi=adapters(base,root/'adapters','expanded')
    m,pm,_,_=validate(multi);data=dataset(root,'qasper',1);old=root/'historical_plan'
    old_prepare({'qasper':data},TinyTokenizer(),tokenizer_identity(pm),2048,dict(bos_token_id=0,eos_token_id=6,pad_token_id=0),old,multi_prepared=multi,fixture=True)
    return multi,old


class Planning(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory();cls.root=Path(cls.tmp.name);cls.multi,cls.old=fixture_inputs(cls.root)
    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()
    def test_fresh_plan_exact_prefix_no_32_layer_tensor_dump(self):
        before=(self.old/'i5_output_manifest.json').read_bytes()
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'plan';r=prepare(self.old,self.multi,p,fixture=True)
            s,t=r['blocks'][0]['tasks'];self.assertEqual(s['token_ids'],t['token_ids'][:64]);self.assertEqual(len(t['token_ids']),256)
            self.assertTrue(all(not x['strict'] for x in (s,t)));self.assertEqual(t['policy']['max_new_tokens'],2)
            self.assertEqual(validate_plan(p)['experiment_id'],r['experiment_id'])
        self.assertEqual(before,(self.old/'i5_output_manifest.json').read_bytes())
    def test_commit_changes_fail_closed(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'plan';r=prepare(self.old,self.multi,p,fixture=True)
            with patch('semcache.edgelora.r5_i5r2b_plan.source_identity',return_value=dict(r['i5r2b']['source'],commit='0'*40)):
                with self.assertRaisesRegex(ValueError,'source changed'):validate_plan(p)
    def test_historical_output_namespace_is_protected(self):
        with self.assertRaises(ValueError):prepare(self.old,self.multi,ROOT/'results/r5/i5r/diagnosis/PROHIBITED',fixture=True)
    def test_corrupt_plan_manifest_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'plan';prepare(self.old,self.multi,p,fixture=True);(p/'i5r2b_protocol.json').write_text('{}')
            with self.assertRaises(ValueError):validate_plan(p)


if __name__=='__main__':unittest.main()
