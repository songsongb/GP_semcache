"""D1 offline contracts; optional tiny-model controls do not download weights."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from semcache.diagnostics.r1b1_d1 import provenance_audit,guarded_selections,spans
from semcache.metrics.r1 import digest
from semcache.semantic.hit_selection import CacheHit
from semcache.semantic.subsequence import Subsequence


def fixture_rows():
    """Synthetic artifact, NOT the user's measured GPU data."""
    ids=list(range(64))
    ids[54:63]=ids[3:12]
    records=[]
    for start in range(0,63,3):
        ss=start-51 if start>=54 else start
        records.append(dict(source_start=ss,source_end=ss+3,destination_start=start,destination_end=start+3,
            token_ids=ids[start:start+3],source_adapter='user_a',destination_adapter='user_a'))
    row=dict(input_token_ids=ids,token_ids_sha256=digest(ids),adapter='user_a',repeat=0,
        impact_execution='token_precompute',storage='RAW_CPU',instrumentation='minimal',
        phase='measured',condition='same_user_exact',requested_prompt_tokens=64,
        reused_tokens=63,reuse_provenance=records,mode='RAW_PHYSICAL_REUSE',
        correctness=dict(atol=.001,rtol=.001,max_abs_logit_diff=13.4453125,allclose=False))
    strict=copy.deepcopy(row)
    strict.update(mode='POSITION_ALIGNED_DIAGNOSTIC',reused_tokens=54,
        reuse_provenance=records[:18],correctness=dict(atol=.001,rtol=.001,max_abs_logit_diff=.046875,allclose=False))
    return row,strict


class OfflineTests(unittest.TestCase):
    def test_expected_three_moves_and_context_on_synthetic_artifact(self):
        natural,strict=fixture_rows()
        audit=provenance_audit(natural,strict)
        self.assertTrue(audit['expected_64_case_verified'])
        self.assertEqual(audit['mismatched_tokens'],9)
        self.assertTrue(audit['selected_guard_equals_strict'])
        self.assertFalse(any(r['prefix_token_ids_equal'] for r in audit['context_audit']))

    def test_wrong_moves_and_overlap_rejected(self):
        natural,strict=fixture_rows()
        natural['reuse_provenance'][-1]['source_start']=8
        with self.assertRaises(ValueError):
            provenance_audit(natural,strict)
        natural,strict=fixture_rows()
        natural['reuse_provenance'][-1]=natural['reuse_provenance'][0]
        with self.assertRaisesRegex(ValueError,'coverage'):
            provenance_audit(natural,strict)

    def test_position_guard_is_not_general_strict_authorization(self):
        ids=(1,2,3,4)
        entry=SimpleNamespace(key=(0,(1,2,3)),positions=(0,3),token_ids=(1,2,3),
            qkv_metadata=dict(source_query_token_ids=ids,source_user='other',source_adapter='other'))
        hit=CacheHit(Subsequence((1,2,3),0,3),entry)
        result=guarded_selections([hit],4,ids,'user_a')
        self.assertEqual(len(result['POSITION_GUARD']),1)
        self.assertEqual(result['STRICT'],[])
        self.assertEqual(entry.qkv_metadata['source_user'],'other')

    def test_offline_cli_outputs_no_tensor_arrays_and_does_not_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            raw=root/'raw.jsonl'
            raw.write_text(''.join(json.dumps(r)+'\n' for r in fixture_rows()))
            (root/'manifest.json').write_text('{}')
            out=root/'audit'
            cmd=[sys.executable,str(ROOT/'scripts/71_diagnose_r1b1_correctness.py'),'--raw',str(raw),
                '--lengths','64','--audit-only','--output-dir',str(out)]
            process=subprocess.run(cmd,capture_output=True,text=True)
            self.assertEqual(process.returncode,0,process.stderr)
            state=json.loads((out/'manifest.json').read_text())
            self.assertEqual(state['status'],'ARTIFACT_AUDIT_ONLY')
            self.assertFalse(state['gpu_executed'])
            before=(out/'manifest.json').read_bytes()
            again=subprocess.run(cmd,capture_output=True,text=True)
            self.assertNotEqual(again.returncode,0)
            self.assertEqual(before,(out/'manifest.json').read_bytes())
            self.assertLess(sum(p.stat().st_size for p in out.iterdir()),200000)


HAS_MODELS=all(importlib.util.find_spec(n) is not None for n in ('torch','transformers','peft'))


@unittest.skipUnless(HAS_MODELS,'NOT EXECUTED: torch/transformers/peft unavailable')
class TensorControls(unittest.TestCase):
    def setUp(self):
        from test_r1_raw_semcache import TinyModelTests
        TinyModelTests.setUp(self)

    def test_error_reports_original_token_and_threshold_without_relaxation(self):
        import torch
        from semcache.diagnostics.r1b1_d1 import error
        a=torch.ones(1,2,3)
        b=a.clone(); b[0,1,2]+=.1
        result=error(a,b,1e-3,1e-3,[3,7])
        self.assertEqual(result['first_tolerance_failure']['token_index'],7)
        self.assertFalse(result['allclose'])

    def check_controls(self,device):
        import torch
        from semcache.cache.cache_entry import CacheEntry
        from semcache.models.model_adapter import OPTModelAdapter
        from semcache.diagnostics.r1b1_d1 import forward,layer_errors,shape_probes,cache_errors
        self.model.to(device)
        self.adapter=OPTModelAdapter(self.model)
        ids=[2,3,4,5,6,7,8,9]
        full=forward(self.model,self.adapter,ids,'user_a')
        hits=[]
        for start in (0,3):
            tensors={i:tuple(full['captured'][(i,r)][:,start:start+3] for r in 'qkv') for i in range(2)}
            entry=CacheEntry.from_tensors(0,tuple(ids[start:start+3]),(start,start+3),tensors)
            entry.qkv_metadata=dict(component_scope='total_qkv',source_user='user_a',source_adapter='user_a',source_query_token_ids=ids)
            hits.append(CacheHit(Subsequence(tuple(ids[start:start+3]),start,start+3),entry))
        for control in ('mixed','injection','full_assembly'):
            output=forward(self.model,self.adapter,ids,'user_a',hits,control)
            if control!='mixed' or next(self.model.parameters()).dtype==torch.float32:
                torch.testing.assert_close(full['logits'],output['logits'],atol=1e-6,rtol=1e-5)
            rows,firsts=layer_errors(full['captured'],output['captured'],1e-6,1e-5)
            self.assertEqual(len(firsts),2)
            self.assertEqual(len(rows),16)
            cached=cache_errors(hits,full['captured'],full['captured'],1e-6,1e-5,output['captured'])
            self.assertFalse(any(r['nonexact_windows'] for r in cached))
        probes=shape_probes(self.adapter,full['captured'],hits,len(ids),'user_a',1e-6,1e-5)
        self.assertEqual(len(probes),24)
        self.assertEqual({r['subset_rows'] for r in probes},{2})
        for control in ('mixed','injection','full_assembly'):
            with patch.object(self.model,'forward',side_effect=RuntimeError('intentional')):
                with self.assertRaisesRegex(RuntimeError,'intentional'):
                    forward(self.model,self.adapter,ids,'user_a',hits,control)
        for i in range(2):
            for module in self.adapter.projection_modules(i).values():
                self.assertFalse(module._forward_hooks or module._forward_pre_hooks)
                self.assertNotIn('forward',module.__dict__)

    def test_tiny_fp32_controls_cpu(self):
        self.check_controls('cpu')

    def test_tiny_controls_cuda(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest('NOT EXECUTED: CUDA unavailable')
        self.model.half()
        self.check_controls('cuda:0')


if __name__=='__main__':
    unittest.main()
