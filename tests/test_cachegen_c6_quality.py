"""CPU tests runnable with stdlib unittest; tensor tests skip without PyTorch."""
import builtins
import copy
import importlib.util
import io
import inspect
import json
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from semcache.experiments.cachegen import c6_quality as c6
from semcache.experiments.cachegen.c6_runtime import forbid_storage_fitting, Storage


def rows(dataset='snips'):
    return [dict(dataset=dataset, source_id=str(i), query_text=f'query {i}',
        token_ids=[2,50+i,90,10,11,12,100+i], cluster_id=4,
        model_id=c6.MODEL_ID, model_revision=c6.MODEL_REVISION,
        tokenizer_id=f'{c6.MODEL_ID}@{c6.MODEL_REVISION}', semantic_assignment_source=c6.assignment_source(dataset),
        semantic_execution_provenance='MEASURED', reference_text=f'label{i%7}',
        domain_or_intent=f'label{i%7}', conversation_id=f'dialogue{i//3}') for i in range(30)]


class C6Tests(unittest.TestCase):
    def test_selection_deterministic_diverse_quality_blind(self):
        data=rows()
        a=c6.select(data,'snips',14)
        self.assertEqual(a,c6.select(data,'snips',14))
        self.assertEqual(len({e['intent'] for e in a[:7]}),7)
        changed=copy.deepcopy(data)
        for r in changed:
            r.update(generated_text='arbitrary', bleu=999, codec_size=0)
        self.assertEqual(a,c6.select(changed,'snips',14))
        self.assertEqual({e['source_user'] for e in a}|{e['target_user'] for e in a},{'user_a','user_b'})
        for e in a:
            self.assertNotEqual(e['source_id'],e['target_id'])
            self.assertGreaterEqual(e['source_start'],3)
            self.assertGreaterEqual(e['target_start'],3)
            self.assertEqual(e['token_ids'],[10,11,12])
            self.assertEqual(e['selected_hit_count'],1)
        b=c6.select(rows('multiwoz'),'multiwoz',10)
        self.assertEqual(len({e['conversation_id'] for e in b}),10)

    def test_guards(self):
        for change in ('prefix','tokens','cluster','same_text','special'):
            with self.subTest(change=change):
                data=rows()[:2]
                if change=='prefix':
                    for r in data: r['token_ids']=[10,11,12,99+int(r['source_id'])]
                if change=='tokens': data[1]['token_ids'][4]=888
                if change=='cluster': data[1]['cluster_id']=9
                if change=='same_text': data[1]['query_text']=data[0]['query_text']
                if change=='special':
                    for r in data: r['special_token_ids']=[11]
                with self.assertRaisesRegex(ValueError,'eligible'): c6.select(data,'snips',1)

    def test_reference_and_measured_required(self):
        data=rows(); data[1]['reference_text']=None
        with self.assertRaisesRegex(ValueError,'reference'): c6.select(data,'snips',1)
        data=rows(); data[0]['semantic_execution_provenance']='TEST_STUB'
        with self.assertRaisesRegex(ValueError,'measured'): c6.select(data,'snips',1)

    def test_observed_hit_must_match_frozen_plan(self):
        from types import SimpleNamespace
        from semcache.cache.cache_entry import CacheEntry
        from semcache.semantic.subsequence import Subsequence
        episode=c6.select(rows(),'snips',1)[0]
        entry=CacheEntry(episode['cluster'],episode['token_ids'],(3,6),72,
            qkv_metadata=dict(source_user=episode['source_user'],source_id=episode['source_id']))
        hit=SimpleNamespace(entry=entry,window=Subsequence(tuple(episode['token_ids']),3,6))
        self.assertEqual(c6.validate_hit(episode,hit),c6.digest(episode))
        entry.qkv_metadata['source_user']='different_user'
        with self.assertRaisesRegex(ValueError,'Observed hit'):
            c6.validate_hit(episode,hit)

    def test_mode_output_schema_and_csv_escaping(self):
        import csv
        episode=c6.select(rows(),'snips',1)[0]
        results=[c6.case_result(episode,m,[4,5],'a,"quoted"\nline',{}) for m in c6.MODES]
        full=results[0]
        self.assertFalse(full['cache_hit'])
        self.assertEqual(full['selected_hit_count'],0)
        self.assertEqual(full['executed_hit_count'],0)
        self.assertIsNone(full['logical_event_hash'])
        self.assertEqual(len({r['logical_event_hash'] for r in results[1:]}),1)
        self.assertTrue(all(r['reference_sha256']==episode['reference_sha256'] for r in results))
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'per_case.csv'
            c6.write_csv(path,results)
            with path.open(newline='') as stream:
                restored=list(csv.DictReader(stream))
            self.assertEqual(restored[0]['generated_text'],'a,"quoted"\nline')
            self.assertEqual(json.loads(restored[0]['generated_token_ids']),[4,5])

    def test_generation(self):
        self.assertEqual(c6.normalized_label(' PlayMusic \n'),'playmusic')
        m=c6.generation_metrics([1,2,3],[1,4])
        self.assertEqual(m['first_divergent_position'],1)
        self.assertEqual(m['common_prefix_length'],1)
        self.assertAlmostEqual(m['normalized_edit_distance'],2/3)
        self.assertEqual(m['position_agreement'],.5)
        self.assertIsNone(c6.generation_metrics([1],[1])['first_divergent_position'])
        self.assertEqual(c6.generation_metrics([1],[1,2])['first_divergent_position'],1)
        self.assertTrue(c6.generation_metrics([],[])['exact_generation'])

    @unittest.skipUnless(importlib.util.find_spec('torch'),'PyTorch unavailable')
    def test_logits(self):
        import torch
        a=torch.tensor([[1.,2.,3.,4.,5.,6.],[6.,5.,4.,3.,2.,1.]])
        m=c6.logit_metrics(a,a)
        self.assertEqual(m['mean_kl'],0)
        self.assertEqual(m['top1_agreement'],1)
        self.assertEqual(m['top5_overlap'],1)
        self.assertEqual(m['raw_top1_rank'],1)
        m=c6.logit_metrics(a,-a)
        self.assertGreater(m['mean_kl'],0)
        self.assertEqual(m['top1_agreement'],0)
        self.assertEqual(m['raw_top1_rank'],6)

    def test_corpus_once_per_dataset_mode(self):
        with patch.object(c6,'compute_bleu',return_value=dict(value=12,signature='test')) as scorer:
            cases=[dict(dataset='snips',mode='RAW_SEMCACHE',generated_text=x,reference_text=x) for x in ['A','B']]
            summary=c6.summarize(cases,[])
            self.assertEqual(scorer.call_count,1)
            self.assertEqual(len(scorer.call_args.args[0]),2)
            self.assertEqual(scorer.call_args.kwargs,c6.BLEU)
            self.assertEqual(summary[0]['intent_accuracy'],1)
            self.assertFalse(summary[0]['paper_bleu_claimed'])

    def test_factors_profile_and_fitting(self):
        self.assertEqual(c6.MODES,dict(FULL_RECOMPUTE=(False,False,False),RAW_SEMCACHE=(True,False,False),
            STORAGE_KV_COMP=(True,False,True),TRANSPORT_QKV_COMP=(True,True,False),FULL_PIPELINE=(True,True,True)))
        with tempfile.TemporaryDirectory() as td:
            profile=Path(td)/'bad.bin'; profile.write_bytes(b'bad')
            with self.assertRaisesRegex(ValueError,'SHA256'): c6.verify_profile(profile)
        for name in ('calculate_cdf','cdf_from_counts','fit_profiles','fit'):
            namespace={}
            exec('def '+name+'():\n raise AssertionError("must not execute")',namespace)
            with self.assertRaisesRegex(RuntimeError,'forbidden'):
                with forbid_storage_fitting(): namespace[name]()

    @unittest.skipUnless(importlib.util.find_spec('torch'),'PyTorch unavailable')
    def test_storage_preserves_q(self):
        import torch
        from types import SimpleNamespace
        from semcache.cache.cache_entry import CacheEntry
        if 'compressed_kv' not in inspect.signature(CacheEntry).parameters:
            self.skipTest('C2 CacheEntry API unavailable in this checkout')
        storage=Storage.__new__(Storage)
        tensors={0:tuple(torch.ones(1,3,4,dtype=torch.float16)*i for i in (1,2,3))}
        class Codec:
            def make_entry(self,factory,*args):
                return factory(0,(10,11,12),(3,6),72,q_tensors={0:tensors[0][0].clone()},
                    compressed_kv=SimpleNamespace(profile_sha256=c6.PROFILE_SHA,bitstream=b'kv',local_metadata_bytes=4))
            def decode_entry(self,entry):
                return SimpleNamespace(resident=entry,tensors={0:(entry.q_tensors[0],tensors[0][1],tensors[0][2])})
        storage.codec=Codec()
        resident, accounting=storage.encode(dict(cluster=0,token_ids=[10,11,12],source_start=3,source_user='user_a',source_id='source'),tensors)
        view=storage.codec.decode_entry(resident)
        storage.validate_decoded(resident,view)
        output=view.tensors
        self.assertIsNone(resident.tensors)
        self.assertTrue(hasattr(resident,'compressed_kv'))
        self.assertTrue(torch.equal(output[0][0],tensors[0][0]))
        self.assertEqual(accounting['resident_q_bytes'],24)

    def test_dry_run_no_model_codec(self):
        original=builtins.__import__
        def guarded(name,*a,**kw):
            if name.startswith(('torch','transformers','peft')) or 'cachegen_codec' in name or 'physical_storage' in name:
                raise AssertionError(f'forbidden import {name}')
            return original(name,*a,**kw)
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); paths=[]
            for dataset in ('snips','multiwoz'):
                p=root/f'{dataset}.jsonl'
                p.write_text(''.join(json.dumps(r)+'\n' for r in rows(dataset)))
                paths.extend(['--'+dataset,str(p)])
            out=root/'out'
            with patch('builtins.__import__',guarded), redirect_stdout(io.StringIO()) as output:
                c6.main(paths+['--dry-run','--per-dataset','2','--device','cpu','--output-dir',str(out)])
            selection=json.loads((out/'selection.json').read_text())
            manifest=json.loads((out/'manifest.json').read_text())
            self.assertEqual(len(selection['episodes']),4)
            self.assertEqual(set(selection['modes']),set(c6.MODES))
            self.assertEqual(manifest['status'],'DRY_RUN')
            self.assertIsNone(manifest['resolved_revision'])
            self.assertFalse(manifest['trained_adapter'])
            self.assertFalse(manifest['paper_bleu_claimed'])
            self.assertEqual(manifest['selection_hash'],c6.file_hash(out/'selection.json'))
            self.assertIn('selected_counts',output.getvalue())

class MixedTransportBoundaryTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('torch'),'PyTorch unavailable')
    def test_callback_only_receives_fresh_rows_and_restores_modules(self):
        import torch
        from types import SimpleNamespace
        from semcache.edgelora import mixed_projection as mixed
        from semcache.cache.cache_entry import CacheEntry
        from semcache.semantic.hit_selection import CacheHit
        from semcache.semantic.subsequence import Subsequence
        modules={name:torch.nn.Linear(4,4,bias=False).eval() for name in 'qkv'}
        adapter=SimpleNamespace(layers=[None],projection_modules=lambda layer:modules)
        hidden=torch.arange(24,dtype=torch.float32).reshape(1,6,4)
        entry=CacheEntry.from_tensors(0,(10,11,12),(3,6),{0:tuple(torch.full((1,3,4),99.) for _ in range(3))})
        entry.qkv_metadata['component_scope']='total_qkv'
        hit=CacheHit(Subsequence((10,11,12),3,6),entry)
        calls=[]
        def transmit(layer,role,module,fresh,native):
            calls.append(fresh.clone())
            return native(fresh)+1
        with patch.object(mixed,'validate_projection'):
            with mixed.mixed_projection_path(adapter,'user_a',[hit],6,fresh_projection=transmit) as audit:
                outputs=[module(hidden) for module in modules.values()]
        self.assertEqual(len(calls),3)
        self.assertTrue(all(torch.equal(x,hidden[:,:3]) for x in calls))
        self.assertTrue(all(torch.equal(x[:,3:],entry.tensors[0][0]) for x in outputs))
        self.assertTrue(all('forward' not in module.__dict__ for module in modules.values()))
        self.assertTrue(all(r['reconstructed_projection_rows']==3 and r['native_call_count']==0 for r in audit.records.values()))


if __name__ == '__main__':
    unittest.main()
