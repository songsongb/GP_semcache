"""C6-B1 CPU tests; tiny real PEFT tests are optional, never download assets."""
import builtins
import copy
import importlib.util
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from semcache.experiments.cachegen import c6b_snips as b1
from semcache.experiments.cachegen.c6_quality import select
from semcache.models.task_adapters import ADAPTER_CONFIG, validate_task_config

HAS_TORCH = importlib.util.find_spec('torch') is not None
HAS_PEFT = all(importlib.util.find_spec(n) is not None for n in ('torch','transformers','peft'))


def workload():
    return [dict(dataset='snips',source_id=f'row{i}',query_text=f'utterance {i}',
        reference_text=b1.LABELS[(i//2)%7],domain_or_intent=b1.LABELS[(i//2)%7],
        token_ids=[2,100+i,500,10,11,12,600+i],cluster_id=0,
        model_id=b1.MODEL_ID,model_revision=b1.MODEL_REVISION,
        tokenizer_id=f'{b1.MODEL_ID}@{b1.MODEL_REVISION}',semantic_execution_provenance='MEASURED',
        semantic_assignment_source=b1.assignment_source('snips')) for i in range(85)]


def selection(rows):
    return dict(episodes=select(rows,'snips',7))


class Tokenizer:
    eos_token_id=2
    all_special_ids=[0,1,2]
    def __call__(self,text,*,add_special_tokens,truncation):
        if text.startswith('utterance'):
            i=int(text.split()[-1])
            return dict(input_ids=[2,100+i,500,10,11,12,600+i])
        if not text.startswith(' ') or text.startswith('  ') or add_special_tokens:
            raise AssertionError('Expected separately tokenized single-space completion')
        return dict(input_ids=[20+b1.LABELS.index(text[1:]),30])


class DataTests(unittest.TestCase):
    def test_holdout_assignment_cap_and_no_duplicates(self):
        rows=workload(); frozen=selection(rows)
        plan=b1.build_plan(rows,frozen,2,42)
        self.assertEqual(plan,b1.build_plan(rows,frozen,2,42))
        holdout={e[k] for e in frozen['episodes'] for k in ('source_id','target_id')}
        self.assertEqual(set(plan['holdout_ids']),holdout)
        self.assertFalse(set(plan['train_row_ids'])&holdout)
        self.assertEqual(len(plan['train_row_ids']),len(set(plan['train_row_ids'])))
        eligible=[r for r in rows if r['source_id'] not in holdout]
        assigned=b1.logical_user_assignment(eligible,2,42,'snips')
        expected=[]; counts={u:{l:0 for l in b1.LABELS} for u in b1.USERS}
        for row,user in zip(eligible,assigned):
            user=b1.USER_MAP[user]; label=row['reference_text']
            if counts[user][label]<2:
                expected.append(row['source_id']); counts[user][label]+=1
        self.assertEqual(plan['train_row_ids'],expected)
        self.assertEqual(plan['counts'],counts)
        changed=copy.deepcopy(rows)
        for r in changed: r.update(generated_text='ignored',bleu=999)
        self.assertEqual(plan,b1.build_plan(changed,frozen,2,42))

    def test_selection_and_workload_fail_closed(self):
        for failure in ('duplicate','reference','missing_holdout','span','reference_hash','index','user'):
            with self.subTest(failure=failure):
                rows=workload(); frozen=selection(rows)
                if failure=='duplicate': rows.append(copy.deepcopy(rows[0]))
                if failure=='reference': rows[-1]['reference_text']=None
                if failure=='missing_holdout': frozen['episodes'][0]['source_id']='missing'
                if failure=='span': frozen['episodes'][0]['target_start']=0
                if failure=='reference_hash': frozen['episodes'][0]['reference_sha256']='bad'
                if failure=='index': frozen['episodes'][0]['target_index']=80
                if failure=='user': frozen['episodes'][0]['target_user']='unknown'
                with self.assertRaises(ValueError): b1.build_plan(rows,frozen)

    def test_shortfalls_no_oversampling(self):
        rows=workload(); plan=b1.build_plan(rows,selection(rows),200)
        self.assertEqual(len(plan['train_rows']),len(rows)-len(plan['holdout_ids']))
        self.assertTrue(all(n>0 for counts in plan['shortfalls'].values() for n in counts.values()))

    def test_explicit_boundary_prompt_mask_and_eos(self):
        row=workload()[0]
        result=b1.encode_example(Tokenizer(),row,64)
        n=len(row['token_ids']); completion=[20,30,2]
        self.assertEqual(result['input_ids'],row['token_ids']+completion)
        self.assertEqual(result['labels'],[-100]*n+completion)
        self.assertEqual(result['labels'][n:].count(2),1)
        with self.assertRaisesRegex(ValueError,'no silent truncation'):
            b1.encode_example(Tokenizer(),row,8)
        bad=copy.deepcopy(row); bad['token_ids']=[999]
        with self.assertRaisesRegex(ValueError,'canonical'): b1.encode_example(Tokenizer(),bad,64)

    def test_length_normalization_and_ties(self):
        self.assertEqual(b1.mean_label_log_probability([-2,-2]),b1.mean_label_log_probability([-2]))
        scores={l:-2.0 for l in b1.LABELS}
        self.assertEqual(b1.classify(scores,b1.LABELS[-1])['predicted_label'],b1.LABELS[0])
        scores[b1.LABELS[-1]]=-1.0
        result=b1.classify(scores,b1.LABELS[-1])
        self.assertTrue(result['correct']); self.assertEqual(result['classification_margin'],1.0)

    def test_leading_space_shared_by_training_and_candidate_scoring(self):
        class RecordingTokenizer(Tokenizer):
            def __init__(self): self.calls=[]
            def __call__(self,text,**kwargs):
                self.calls.append((text,kwargs))
                return super().__call__(text,**kwargs)
        tokenizer=RecordingTokenizer()
        for label in b1.LABELS:
            row=workload()[0].copy()
            row['reference_text']=label
            tokenizer.calls.clear()
            training=b1.encode_example(tokenizer,row,64)
            # Evaluation builds each candidate through this same label_ids API.
            candidate=b1.label_ids(tokenizer,label)
            self.assertEqual(tokenizer.calls[1:], [(' '+label,dict(add_special_tokens=False,truncation=False))]*2)
            self.assertEqual(training['input_ids'][:training['prompt_length']],row['token_ids'])
            self.assertEqual(training['labels'][:training['prompt_length']],[-100]*len(row['token_ids']))
            self.assertEqual(training['labels'][training['prompt_length']:],candidate+[tokenizer.eos_token_id])
            self.assertNotIn(tokenizer.eos_token_id,candidate)
            self.assertEqual(b1.normalized_label(' '+label+'\n'),label.casefold())

    @unittest.skipUnless(HAS_TORCH,'PyTorch unavailable')
    def test_causal_score_positions_exclude_prompt_and_eos(self):
        import torch
        logits=torch.zeros(6,40)
        logits[2,20]=5; logits[3,30]=7
        score=b1.score_candidate_logits(logits,3,[20,30])
        expected=(logits[2].log_softmax(-1)[20]+logits[3].log_softmax(-1)[30])/2
        self.assertAlmostEqual(score,expected.item(),places=6)
        logits[:2]=999; logits[4:]=-999
        self.assertAlmostEqual(score,b1.score_candidate_logits(logits,3,[20,30]),places=6)

    def test_confusion_and_accuracy(self):
        scores={l:-2 for l in b1.LABELS}; scores[b1.LABELS[0]]=-1
        cases=[dict(user=u,reference_text=label,greedy_exact_label_match=False,**b1.classify(scores,label))
               for u,label in zip(b1.USERS,b1.LABELS[:2])]
        summary=b1.aggregate(cases)
        self.assertEqual(summary['accuracy'],.5)
        self.assertEqual(summary['per_user']['user_a']['accuracy'],1)
        self.assertEqual(summary['per_user']['user_b']['accuracy'],0)
        self.assertEqual(summary['confusion_matrix'][b1.LABELS[1]][b1.LABELS[0]],1)
        self.assertIsNone(summary['threshold_passed'])
        self.assertIsNone(summary['per_intent'][b1.LABELS[-1]]['accuracy'])
        self.assertFalse(b1.aggregate(cases,.75)['threshold_passed'])

    def test_dry_run_has_no_model_import_and_schema(self):
        original=builtins.__import__
        def guarded(name,*args,**kwargs):
            if name.startswith(('torch','peft','transformers')) or name.endswith('c6b_runtime'):
                raise AssertionError('Dry run imported '+name)
            return original(name,*args,**kwargs)
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); data=root/'rows.jsonl'; selected=root/'selection.json'
            rows=workload(); data.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            selected.write_text(json.dumps(selection(rows)))
            for training in (True,False):
                out=root/str(training)
                args=['--snips',str(data),'--selection',str(selected),'--output-root',str(out),'--dry-run']
                if not training: args+=['--adapter-root',str(root/'unused')]
                with patch('builtins.__import__',guarded),redirect_stdout(io.StringIO()):
                    b1.main(training,args)
                name='training_manifest.json' if training else 'capability_manifest.json'
                manifest=json.loads((out/name).read_text())
                self.assertEqual(manifest['status'],'DRY_RUN')
                if training:
                    from semcache.experiments.cachegen.c6b_order import order_manifest
                    plan=json.loads((out/'train_selection.json').read_text())
                    for key,value in order_manifest(plan,42,3).items():
                        self.assertEqual(manifest[key],value)
                self.assertEqual(manifest['label_serialization'],b1.LABEL_SERIALIZATION)
                self.assertFalse(manifest['trained_adapter'])
                for flag in ('compression_enabled','semantic_reuse_enabled','paper_bleu_claimed'):
                    self.assertFalse(manifest[flag])
                self.assertEqual(manifest['train_selection_sha256'],b1.file_hash(out/'train_selection.json'))

    def test_adapter_config_rejects_variants(self):
        config=SimpleNamespace(**ADAPTER_CONFIG,peft_type='LORA',base_model_name_or_path=b1.MODEL_ID,revision=b1.MODEL_REVISION)
        validate_task_config(config)
        for field,value in [('r',16),('lora_alpha',16),('use_dora',True),('rank_pattern',{'q_proj':4})]:
            bad=copy.deepcopy(config); setattr(bad,field,value)
            with self.assertRaises(ValueError): validate_task_config(bad)

    def test_training_only_enables_current_adapter_and_evaluation_freezes_all(self):
        from semcache.models.task_adapters import select_training_user,activate_task_user
        class Parameter:
            requires_grad=True
            def requires_grad_(self,value): self.requires_grad=value
        class Model:
            peft_config={u:None for u in b1.USERS}
            def __init__(self):
                self.weights={n:Parameter() for n in ('q.base_layer.weight','q.lora_A.user_a.weight',
                    'q.lora_B.user_a.weight','q.lora_A.user_b.weight','q.lora_B.user_b.weight')}
            def named_parameters(self): return self.weights.items()
            def parameters(self): return self.weights.values()
            def set_adapter(self,name):
                self.active=name
                for p in self.parameters(): p.requires_grad_(True)
            def requires_grad_(self,value):
                for p in self.parameters(): p.requires_grad_(value)
            def eval(self): return self
        model=Model()
        for user in b1.USERS:
            params=select_training_user(model,user)
            self.assertEqual(len(params),2)
            self.assertTrue(all(p.requires_grad==(user in n) for n,p in model.named_parameters()))
            activate_task_user(model,user)
            self.assertFalse(any(p.requires_grad for p in model.parameters()))

    def test_training_artifact_hashes_and_leakage(self):
        from semcache.experiments.cachegen.c6b_runtime import verify_training_artifacts
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); rows=workload(); frozen=selection(rows); plan=b1.build_plan(rows,frozen)
            data=root/'rows.jsonl'; data.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            selected=root/'selection.json'; selected.write_text(json.dumps(frozen))
            b1.write_json(root/'train_selection.json',plan)
            for u in b1.USERS:
                (root/u).mkdir(); (root/u/'adapter_config.json').write_text('{}')
            trained=dict(**b1.SCOPE,base_qkv_unchanged=True,train_ids_sha256=plan['train_ids_sha256'],counts=plan['counts'],status='COMPLETE',trained_adapter=True,adapter_source='task_finetuned_snips',
                label_serialization=b1.LABEL_SERIALIZATION,
                base_model=b1.MODEL_ID,model_revision=b1.MODEL_REVISION,resolved_model_revision=b1.MODEL_REVISION,
                resolved_tokenizer_revision=b1.MODEL_REVISION,adapter_config=ADAPTER_CONFIG,
                workload_sha256=b1.file_hash(data),selection_sha256=b1.file_hash(selected),
                train_selection_sha256=b1.file_hash(root/'train_selection.json'),train_row_ids=plan['train_row_ids'],
                holdout_ids=plan['holdout_ids'],holdout_ids_sha256=plan['holdout_ids_sha256'],
                adapter_hashes={u:b1.file_hashes(root/u) for u in b1.USERS})
            b1.write_json(root/'training_manifest.json',trained)
            args=SimpleNamespace(adapter_root=root,snips=data,selection=selected)
            verify_training_artifacts(args,rows,plan)
            trained['train_row_ids'].append(plan['holdout_ids'][0])
            b1.write_json(root/'training_manifest.json',trained)
            with self.assertRaises(ValueError): verify_training_artifacts(args,rows,plan)


@unittest.skipUnless(HAS_PEFT,'PyTorch/Transformers/PEFT unavailable')
class TinyAdapterTests(unittest.TestCase):
    def test_two_local_adapters_base_unchanged_and_projection_compatible(self):
        import torch
        from transformers import OPTConfig, OPTForCausalLM
        from semcache.models.lora_fixtures import create_controlled_users,base_weight_fingerprint
        from semcache.models.task_adapters import load_two_task_users,activate_task_user
        from semcache.models.model_adapter import OPTModelAdapter
        from semcache.models.lora_decomposition import projection_parts
        from semcache.edgelora.mixed_projection import mixed_projection_path
        config=OPTConfig(vocab_size=64,hidden_size=16,ffn_dim=32,num_hidden_layers=2,
                         num_attention_heads=2,max_position_embeddings=32,dropout=0.0,attention_dropout=0.0)
        config._attn_implementation='eager'
        torch.manual_seed(3); base=OPTForCausalLM(config)
        state=copy.deepcopy(base.state_dict())
        fixtures,metadata=create_controlled_users(base)
        self.assertTrue(all(not m['trained_adapter'] for m in metadata.values()))
        with tempfile.TemporaryDirectory() as td:
            for c in fixtures.peft_config.values():
                c.base_model_name_or_path=b1.MODEL_ID; c.revision=b1.MODEL_REVISION
            fixtures.save_pretrained(td,safe_serialization=True,save_embedding_layers=False)
            fresh=OPTForCausalLM(config); fresh.load_state_dict(state)
            before=base_weight_fingerprint(OPTModelAdapter(fresh))
            loaded=load_two_task_users(fresh,Path(td)/'user_a',Path(td)/'user_b')
            adapter=OPTModelAdapter(loaded)
            self.assertEqual(before,base_weight_fingerprint(adapter))
            for user in b1.USERS:
                activate_task_user(loaded,user)
                self.assertFalse(any(p.requires_grad for p in loaded.parameters()))
                hidden=torch.randn(1,3,16)
                module=adapter.projection_modules(0)['q']
                _,_,total=projection_parts(module,hidden,user)
                torch.testing.assert_close(total,module(hidden))
                ids=torch.tensor([[2,10,11,12,13]])
                with torch.inference_mode():
                    native=loaded(input_ids=ids,use_cache=False).logits
                    with mixed_projection_path(adapter,user,[],5):
                        mixed=loaded(input_ids=ids,use_cache=False).logits
                torch.testing.assert_close(native,mixed)


if __name__=='__main__': unittest.main()
