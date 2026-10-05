import builtins
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from semcache.experiments.cachegen import c6b3_2_multiwoz as b


def selection_fixture():
    rows=[];episodes=[];spans=[];users=[]
    windows=list(range(27))+[0,0,1,1,2]
    for i,w in enumerate(windows):
        depth=i//8+1
        for side in ('source','target'):
            idx=len(rows);sid=str(idx);text='Dialogue:\nUser: words here please\nAssistant:'
            start=text.index('words');end=start+len('words here please')
            tokens=[2,10,20,100+w,200+w,300+w,1000+idx]
            row=dict(dataset='multiwoz',source_id=sid,conversation_id='c'+sid,source_split='train',
                token_ids=tokens,prompt_text=text,reference_text='reference',current_user_text='words here please',
                history_source_ids=['prior']*depth,turn_id=str(depth*2),cluster_id=0,
                semantic_execution_provenance='MEASURED',semantic_assignment_source=b.d.b.m9.assignment_source('multiwoz'),
                model_revision=b.d.b.m9.MODEL_REVISION,tokenizer_id=f'{b.d.b.m9.MODEL_ID}@{b.d.b.m9.MODEL_REVISION}')
            rows.append(row);spans.append(dict(source_id=sid,current_user_token_indices=[3,4,5],current_user_char_span=[start,end]))
            users.append('user_000' if side=='source' or i>=16 else 'user_001')
        si=2*i;ti=si+1
        e=dict(episode_id=str(i),source_index=si,target_index=ti,source_id=str(si),target_id=str(ti),
            source_user='user_a',target_user='user_b' if i<16 else 'user_a',cross_user=i<16,
            source_conversation_id='c'+str(si),target_conversation_id='c'+str(ti),history_depth=depth,
            source_start=3,target_start=3,token_ids=rows[ti]['token_ids'][3:6],cluster=0,selected_hit_count=1)
        e['cache_key']=[0,e['token_ids']]
        for side in ('source','target'):
            r=rows[e[side+'_index']];span=spans[e[side+'_index']]
            e[side+'_current_user_token_indices']=span['current_user_token_indices']
            e[side+'_current_user_char_span']=span['current_user_char_span']
            e[side+'_prompt_sha256']=b.d.b.digest(r['prompt_text']);e[side+'_reference_sha256']=b.d.b.digest(r['reference_text'])
        episodes.append(e)
    return rows,dict(episodes=episodes,selection_sha256=b.d.b.digest(episodes)),spans,users

class Tests(unittest.TestCase):
    def test_selection_fail_closed(self):
        rows,selection,spans,users=selection_fixture()
        with patch.object(b,'logical_user_assignment',return_value=users):
            self.assertEqual(b.audit_selection(rows,selection,spans)['unique_exact_w3_windows'],27)
            for mutate in (lambda s:s['episodes'].pop(),lambda s:s['episodes'][0].update(history_depth=2),
                           lambda s:s['episodes'][0].update(source_start=2),
                           lambda s:s['episodes'][0].update(source_current_user_token_indices=[4,5]),
                           lambda s:s['episodes'][0].update(cross_user=False)):
                bad=copy.deepcopy(selection);mutate(bad);bad['selection_sha256']=b.d.b.digest(bad['episodes'])
                with self.assertRaises(ValueError):b.audit_selection(rows,bad,spans)
            bad_spans=copy.deepcopy(spans);bad_spans[0]['current_user_token_indices']=[4,5]
            with self.assertRaisesRegex(ValueError,'Scaffold'):b.audit_selection(rows,selection,bad_spans)

    def test_modes_deltas_bytes(self):
        self.assertEqual(b.MODES,dict(FULL_RECOMPUTE=(False,False,False),RAW_SEMCACHE=(True,False,False),
            STORAGE_KV_COMP=(True,False,True),TRANSPORT_QKV_COMP=(True,True,False),FULL_PIPELINE=(True,True,True)))
        modes=[dict(mode=m,corpus_bleu={'value':v}) for m,v in zip(b.MODES,[3,2,1,2.5,.5])]
        self.assertEqual([r['delta_bleu'] for r in b.causal_comparisons(modes)],[-1,-1,.5,-2,-1.5])
        from semcache.experiments.cachegen.c6_runtime import RAW_ENTRY_BYTES
        q=RAW_ENTRY_BYTES//3
        for m in ('STORAGE_KV_COMP','FULL_PIPELINE'):
            account=b.byte_accounting(m,dict(resident_q_bytes=q,resident_kv_frame_bytes=100,resident_payload_bytes=q+100,resident_local_metadata_bytes=10))
            self.assertEqual(account['resident_payload_bytes'],q+100)
            with self.assertRaises(ValueError):b.byte_accounting(m,dict(resident_q_bytes=0,resident_kv_frame_bytes=100,resident_payload_bytes=100))

    def test_selected_adapter_hashes(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);weights={}
            for user in b.d.USERS:
                (root/user).mkdir();(root/user/'adapter_model.safetensors').write_bytes(user.encode())
                (root/user/'adapter_config.json').write_text(json.dumps(dict(peft_type='LORA',task_type='CAUSAL_LM',r=8,lora_alpha=8,lora_dropout=0,bias='none',target_modules=['q_proj','k_proj','v_proj'],base_model_name_or_path=b.d.b.m9.MODEL_ID,revision=b.d.b.m9.MODEL_REVISION)))
                weights[user]=b.d.b.m9.file_hash(root/user/'adapter_model.safetensors')
            trained=dict(status='COMPLETE',trained_adapter=True,base_qkv_unchanged=True,epochs=2,
                         adapter_hashes={u:b.d.file_hashes(root/u) for u in b.d.USERS})
            with patch.object(b,'WEIGHTS',weights):
                b.verify_adapters(root,trained)
                (root/'user_a'/'adapter_model.safetensors').write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError,'Adapter file hashes'):b.verify_adapters(root,trained)

    def test_import_canonical_cases_and_reject_reordered_identity(self):
        rows,selection,_,_=selection_fixture()
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            cases=[dict(source_id=e['target_id'],user=e['target_user'],conversation_id=e['target_conversation_id'],
                history_depth=e['history_depth'],reference_text='reference',generated_text='canonical original',
                generated_token_ids=[8,2],generated_length=2,normalized_edit_distance=.5,position_agreement=.5) for e in selection['episodes']]
            b.write_csv(root/'capability_per_case.csv',cases)
            imported=b.import_canonical_cases(root,rows,selection['episodes'])
            self.assertEqual(imported[0]['generated_text'],'canonical original')
            self.assertEqual(imported[0]['generated_token_ids'],[8,2])
            cases[0]['source_id']='wrong';b.write_csv(root/'capability_per_case.csv',cases)
            with self.assertRaisesRegex(ValueError,'Canonical case identity'):b.import_canonical_cases(root,rows,selection['episodes'])

    def test_protocol_and_guards(self):
        self.assertEqual(b.BLEU,dict(implementation='sacrebleu',scope='corpus',tokenizer='13a',smoothing='exp',effective_order=False,lowercase=False))
        with self.assertRaises(ValueError):b.check_fields({'training_manifest_sha256':'wrong'},{'training_manifest_sha256':'right'},'freeze')
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'profile';path.write_bytes(b'wrong')
            with self.assertRaises(ValueError):b.verify_profile(path)
            (Path(temp)/'manifest.json').write_text(json.dumps(dict(selection_version='current_user_content_v2',history_k=4,max_sequence_length=384,
                prompt_version=b.d.b.VERSION,model_revision=b.d.b.m9.MODEL_REVISION,bleu_protocol=b.BLEU,physical_safety_contract=b.CONTRACT['physical_safety_contract'])))
            (Path(temp)/'evaluation_selection.json').write_text('{}')
            from types import SimpleNamespace
            with self.assertRaisesRegex(ValueError,'selection SHA'):b.prepare(SimpleNamespace(plan_dir=Path(temp)))

    def test_dry_run_no_runtime(self):
        original=builtins.__import__
        def guarded(name,*args,**kwargs):
            if name.split('.')[0] in ('torch','peft','transformers','lmcache') or name.endswith('c6b3_2_runtime'):
                raise AssertionError('Dry run imported runtime: '+name)
            return original(name,*args,**kwargs)
        with tempfile.TemporaryDirectory() as temp, patch.object(b,'prepare',return_value=([],[],[],{'corpus_bleu':{'value':2.7}},dict(selection_audit={'episode_count':32}))),patch('builtins.__import__',side_effect=guarded):
            root=Path(temp)/'out';b.main(['--dry-run','--output-root',str(root)])
            self.assertEqual(json.loads((root/'manifest.json').read_text())['status'],'DRY_RUN')

    def test_canonical_task_texts_are_scored_as_supplied(self):
        cases=[dict(generated_text='frozen',reference_text='ref')]
        with patch.object(b,'aggregate',return_value=dict(case_count=1,corpus_bleu={'value':2.73},user_bleu={},history_depth_bleu_diagnostic={},nonempty_generation_count=1,generated_token_count=1,mean_normalized_edit_distance=.5,mean_position_agreement=.5)) as aggregate:
            result=b.mode_summary('FULL_RECOMPUTE',cases)
            aggregate.assert_called_once_with(cases);self.assertEqual(result['corpus_bleu']['value'],2.73)

if __name__=='__main__':unittest.main()
