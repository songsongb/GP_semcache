"""No-model B2 provenance, orchestration, paired-task and accounting tests."""
import builtins
import copy
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from semcache.experiments.cachegen import c6b2_snips as b2
from semcache.experiments.cachegen import c6b_snips as b1
from semcache.experiments.cachegen.c6_quality import select
from test_cachegen_c6b_snips import workload


def artifacts(root):
    rows=workload()
    selection=dict(episodes=select(rows,'snips',32))
    data=root/'workload.jsonl'; data.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    selected=root/'selection.json'; b1.write_json(selected,selection)
    adapter=root/'adapters'; adapter.mkdir()
    capability=root/'capability'; capability.mkdir()
    plan=b1.build_plan(rows,selection)
    b1.write_json(adapter/'train_selection.json',plan)
    b1.write_json(capability/'train_selection.json',plan)
    for user in b1.USERS:
        (adapter/user).mkdir()
        config=dict(**b1.ADAPTER_CONFIG,peft_type='LORA',base_model_name_or_path=b1.MODEL_ID,revision=b1.MODEL_REVISION)
        b1.write_json(adapter/user/'adapter_config.json',config)
        (adapter/user/'adapter_model.safetensors').write_bytes(b'synthetic hash fixture; not real tensors')
    training=dict(**b1.SCOPE,status='COMPLETE',trained_adapter=True,adapter_source='task_finetuned_snips',
        base_model=b1.MODEL_ID,model_revision=b1.MODEL_REVISION,tokenizer_revision=b1.MODEL_REVISION,
        resolved_model_revision=b1.MODEL_REVISION,resolved_tokenizer_revision=b1.MODEL_REVISION,
        adapter_config=b1.ADAPTER_CONFIG,base_qkv_unchanged=True,label_serialization=b1.LABEL_SERIALIZATION,
        train_row_ids=plan['train_row_ids'],train_ids_sha256=plan['train_ids_sha256'],counts=plan['counts'],
        holdout_ids=plan['holdout_ids'],holdout_ids_sha256=plan['holdout_ids_sha256'],
        workload_sha256=b1.file_hash(data),selection_sha256=b1.file_hash(selected),
        train_selection_sha256=b1.file_hash(adapter/'train_selection.json'),
        adapter_hashes={u:b1.file_hashes(adapter/u) for u in b1.USERS})
    b1.write_json(adapter/'training_manifest.json',training)
    cases=[]
    for i,target in enumerate(plan['targets']):
        scores={l:-3.0 for l in b1.LABELS}
        # Deliberately imperfect capability: no hard-coded accuracy eligibility gate.
        predicted=target['reference_text'] if i%3 else b1.LABELS[0]
        scores[predicted]=-1.0
        cases.append(dict(**target,mode='FULL_RECOMPUTE',**b1.classify(scores,target['reference_text']),
            generated_text=' '+predicted,greedy_exact_label_match=predicted==target['reference_text']))
    b1.write_csv(capability/'capability_per_case.csv',cases)
    summary=b1.aggregate(cases); summary.update(trained_adapter=True,adapter_source='task_finetuned_snips')
    b1.write_json(capability/'capability_summary.json',summary)
    cap=dict(training,training_manifest_sha256=b1.file_hash(adapter/'training_manifest.json'),
        output_hashes={n:b1.file_hash(capability/n) for n in ('train_selection.json','capability_summary.json','capability_per_case.csv')})
    b1.write_json(capability/'capability_manifest.json',cap)
    profile=root/'profile.bin'; profile.write_bytes(b'synthetic profile hash fixture')
    args=SimpleNamespace(snips=data,selection=selected,adapter_root=adapter,capability_root=capability,
        profile_path=profile,max_episodes=32)
    return args,rows,selection,plan


class SyntheticBackend:
    """Tests production orchestration, without pretending to run model/codec math."""
    def __init__(self):
        self.counts={k:0 for k in b2.COUNTERS}; self.builds=[]; self.candidates=[]; self.contexts=[]
    def prepare(self,episode,mode):
        reuse,transport,storage=b2.MODES[mode]
        self.builds.append((episode['episode_id'],mode))
        self.counts['source_forward_count']+=int(reuse)
        self.counts['storage_decode_count']+=int(storage)
        if transport:
            for key in ('transport_encode_calls','transport_decode_calls','runtime_transport_cdf_fit_count'):
                self.counts[key]+=96
        q=b2.RAW_ENTRY_BYTES//3
        measured=dict(resident_q_bytes=q,resident_payload_bytes=q+1000 if storage else b2.RAW_ENTRY_BYTES,
            resident_kv_frame_bytes=1000,resident_local_metadata_bytes=100)
        return SimpleNamespace(mode=mode,identity=b1.digest(episode) if reuse else None,accounting=measured)
    def event_identity(self,context,episode): return context.identity
    def transport_calls(self,context,forwards):
        if b2.MODES[context.mode][1]:
            for key in ('transport_encode_calls','transport_decode_calls','runtime_transport_cdf_fit_count'):
                self.counts[key]+=96*forwards
    def score(self,context,episode,label):
        self.transport_calls(context,1)
        self.counts['target_candidate_forward_count']+=1
        self.candidates.append(label); self.contexts.append(id(context))
        return -1.0 if label==episode['reference_text'] else -3.0
    def greedy(self,context,episode):
        self.counts['target_greedy_forward_count']+=2
        self.transport_calls(context,2)
        return [20,2],' '+episode['reference_text']


class B2Tests(unittest.TestCase):
    def test_frozen_32_plan_provenance_and_prefix_subset(self):
        with tempfile.TemporaryDirectory() as td:
            args,rows,selection,plan=artifacts(Path(td))
            frozen=copy.deepcopy(selection)
            with patch.object(b2,'verify_profile') as verify:
                actual,episodes,provenance=b2.prepare(args)
            verify.assert_called_once_with(args.profile_path)
            self.assertEqual(episodes,frozen['episodes'])
            self.assertEqual(actual,rows)
            self.assertFalse(provenance['subset'])
            self.assertLess(provenance['capability']['measured_summary']['accuracy'],1.0)
            args.max_episodes=4
            with patch.object(b2,'verify_profile'):
                _,subset,partial=b2.prepare(args)
            self.assertEqual(subset,frozen['episodes'][:4])
            self.assertTrue(partial['subset'])
            self.assertEqual(partial['holdout_ids'],plan['holdout_ids'])
            self.assertEqual(b2.read_json(args.selection),frozen)

    def test_artifact_and_event_mismatches_rejected(self):
        for failure in ('adapter','workload','selection','span','count','capability_hash','capability_adapters','serialization','summary'):
            with self.subTest(failure=failure),tempfile.TemporaryDirectory() as td:
                args,rows,selection,plan=artifacts(Path(td))
                if failure=='adapter': (args.adapter_root/'user_a'/'adapter_model.safetensors').write_bytes(b'changed')
                if failure=='workload': args.snips.write_text(args.snips.read_text()+'\n')
                if failure=='selection': args.selection.write_text(args.selection.read_text()+'\n')
                if failure=='span':
                    selection['episodes'][0]['target_start']=0; b1.write_json(args.selection,selection)
                if failure=='count':
                    selection['episodes'].pop(); b1.write_json(args.selection,selection)
                cap_path=args.capability_root/'capability_manifest.json'; cap=b2.read_json(cap_path)
                if failure=='capability_hash': cap['training_manifest_sha256']='bad'
                if failure=='capability_adapters': cap['adapter_hashes']={}
                if failure=='serialization': cap['label_serialization']='no leading space'
                if failure=='summary':
                    path=args.capability_root/'capability_summary.json'; summary=b2.read_json(path)
                    summary['accuracy']=1.0; b1.write_json(path,summary)
                    cap['output_hashes']['capability_summary.json']=b1.file_hash(path)
                b1.write_json(cap_path,cap)
                with patch.object(b2,'verify_profile'),self.assertRaises(ValueError): b2.prepare(args)

    def test_mode_map_and_one_build_one_decode_seven_candidates(self):
        b2.check_modes()
        episode=select(workload(),'snips',1)[0]
        hashes=[]
        for mode in b2.MODES:
            backend=SyntheticBackend()
            case=b2.evaluate_case(backend,episode,mode)
            self.assertEqual(backend.builds,[(episode['episode_id'],mode)])
            self.assertEqual(backend.candidates,list(b1.LABELS))
            self.assertEqual(len(set(backend.contexts)),1)
            reuse,_,storage=b2.MODES[mode]
            self.assertEqual(case['source_forward_count'],int(reuse))
            self.assertEqual(case['storage_decode_count'],int(storage))
            self.assertEqual(case['selected_hit_count'],int(reuse))
            if reuse: hashes.append(case['logical_event_hash'])
            else:
                self.assertIsNone(case['logical_event_hash']); self.assertEqual(case['resident_payload_bytes'],0)
        self.assertEqual(len(set(hashes)),1)

    def test_event_changes_and_double_decode_fail(self):
        episode=select(workload(),'snips',1)[0]
        backend=SyntheticBackend(); original=backend.score
        def corrupt(context,episode,label):
            value=original(context,episode,label); context.identity='changed'; return value
        backend.score=corrupt
        with self.assertRaisesRegex(ValueError,'event'): b2.evaluate_case(backend,episode,'RAW_SEMCACHE')
        backend=SyntheticBackend(); original=backend.greedy
        def twice(context,episode):
            backend.counts['storage_decode_count']+=1
            return original(context,episode)
        backend.greedy=twice
        with self.assertRaisesRegex(ValueError,'execution counts'): b2.evaluate_case(backend,episode,'STORAGE_KV_COMP')

    def test_paired_flips_deltas_and_secondary_greedy(self):
        cases=[]
        for episode in select(workload(),'snips',2):
            for mode in b2.MODES:
                cases.append(b2.evaluate_case(SyntheticBackend(),episode,mode))
        wrong=next(l for l in b1.LABELS if l!=cases[0]['true_label'])
        cases[2].update(predicted_label=wrong,correct=False,classification_margin=-2.,correct_label_mean_log_probability=-4.)
        cases[5].update(predicted_label=next(l for l in b1.LABELS if l!=cases[5]['true_label']),correct=False)
        pairs=b2.paired_rows(cases); summary=b2.summarize(cases,pairs)
        storage=next(p for p in pairs if p['effect']=='storage')
        self.assertTrue(storage['correct_to_wrong']); self.assertTrue(storage['prediction_changed'])
        self.assertEqual(storage['delta_classification_margin'],-4.)
        self.assertEqual(storage['delta_correct_label_log_probability'],-3.)
        reuse=next(e for e in summary['causal_comparisons'] if e['effect']=='semantic_reuse')
        self.assertEqual(reuse['wrong_to_correct'],1)
        effect=next(e for e in summary['causal_comparisons'] if e['effect']=='storage')
        self.assertEqual(effect['delta_accuracy_percentage_points'],-50.)
        self.assertEqual(effect['mean_delta_classification_margin'],-2.)
        self.assertEqual(effect['median_delta_classification_margin'],-2.)
        primary=summary['modes'][2]
        self.assertEqual(primary['accuracy'],.5); self.assertEqual(primary['greedy_exact_label_accuracy'],1.)
        self.assertEqual(primary['primary_task_metric'],b1.SCOPE['task_metric'])

    def test_physical_accounting_and_no_profile_double_charge(self):
        q=b2.RAW_ENTRY_BYTES//3
        raw=b2.byte_accounting('RAW_SEMCACHE',dict(resident_q_bytes=q,resident_payload_bytes=b2.RAW_ENTRY_BYTES))
        self.assertEqual(raw['kv_compression_ratio'],1.)
        compressed=b2.byte_accounting('FULL_PIPELINE',dict(resident_q_bytes=q,resident_kv_frame_bytes=1000,
            resident_payload_bytes=q+1000,resident_local_metadata_bytes=100))
        self.assertEqual(compressed['resident_payload_bytes'],q+1000)
        self.assertEqual(compressed['kv_compression_ratio'],2*q/1000)
        self.assertAlmostEqual(compressed['byte_reduction_percentage'],100*(1-(q+1000)/(3*q)))

    def test_shared_b1_scoring_ties_and_length_normalization(self):
        self.assertEqual(b1.mean_label_log_probability([-2.,-2.]),b1.mean_label_log_probability([-2.]))
        scores={l:-2. for l in b1.LABELS}
        self.assertEqual(b1.classify(scores,b1.LABELS[-1])['predicted_label'],b1.LABELS[0])

    def test_dry_run_no_model_codec_cuda_imports(self):
        with tempfile.TemporaryDirectory() as td:
            args,_,_,_=artifacts(Path(td)); out=Path(td)/'out'
            original=builtins.__import__
            def guard(name,*a,**kw):
                if name.startswith(('torch','peft','transformers','lmcache')) or name.endswith(('cachegen_codec','physical_storage','c6b2_runtime')):
                    raise AssertionError('Forbidden dry-run import '+name)
                return original(name,*a,**kw)
            argv=['--snips',str(args.snips),'--selection',str(args.selection),'--adapter-root',str(args.adapter_root),
                '--capability-root',str(args.capability_root),'--profile-path',str(args.profile_path),
                '--output-dir',str(out),'--max-episodes','4','--device','cpu','--dry-run']
            with patch.object(b2,'verify_profile'),patch('builtins.__import__',guard),redirect_stdout(io.StringIO()) as stdout:
                b2.main(argv)
            manifest=b2.read_json(out/'manifest.json')
            self.assertEqual(manifest['status'],'DRY_RUN'); self.assertTrue(manifest['subset'])
            self.assertEqual(manifest['capability']['summary_sha256'],b1.file_hash(args.capability_root/'capability_summary.json'))
            self.assertIn('model_codec_loaded',stdout.getvalue())


if __name__=='__main__': unittest.main()
