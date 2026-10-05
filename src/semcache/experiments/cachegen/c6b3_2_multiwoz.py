"""Frozen MultiWOZ B3-2: immutable baseline import and causal reporting."""
import argparse
import csv
import json
import math
import subprocess
from collections import Counter
from pathlib import Path
from statistics import mean
from types import SimpleNamespace
from . import c6b3_train as d
from .c6_quality import MODES, BLEU, PROFILE, PROFILE_SHA, CONTRACT, verify_profile, generation_metrics, write_csv
from .c6b2_snips import PAIRS, byte_accounting
from .c6b3_capability import aggregate
from semcache.models.task_adapters import validate_task_config
from semcache.simulation.multi_user import logical_user_assignment

SELECTION_SHA='e1b5d2156cc548b712cfb09b805ad33ddab04dcbd7b8d55656333bf7b5ac3876'
WEIGHTS={'user_a':'f8159b61862dd4b18c0313959b8212a8421b794c6014dbcdc16462043582a1fe',
         'user_b':'f8935483e6b63fddf19eb74ed44e0a9d74997cf0208ca6666abca88bdcf2959f'}
GENERATION=dict(do_sample=False,num_beams=1,max_new_tokens=160,respect_eos=True)
SCOPE=dict(stage='C6-B3-2',trained_adapter=True,paper_bleu_claimed=False,protocol_provenance='REPRODUCTION_CHOICE',
    primary_task_metric='corpus SacreBLEU',quality_scope='frozen32 task-capable MultiWOZ reuse cohort; causal five-mode comparison; explicit SacreBLEU reproduction protocol; no exact paper reproduction')


def require(condition,message):
    if not condition:raise ValueError(message)


def hashed_outputs(root,manifest,required):
    hashes=manifest['output_hashes']
    for name in required:require(name in hashes,'Missing output hash: '+name)
    for name,sha in hashes.items():require(d.b.m9.file_hash(root/name)==sha,'Output hash mismatch: '+name)


def check_fields(obj,expected,label):
    for key,value in expected.items():require(obj.get(key)==value,label+' mismatch: '+key)


def verify_adapters(root,trained):
    check_fields(trained,dict(status='COMPLETE',trained_adapter=True,base_qkv_unchanged=True,epochs=2), 'Full training')
    for user in d.USERS:
        require(d.file_hashes(root/user)==trained['adapter_hashes'][user],'Adapter file hashes mismatch')
        require(d.b.m9.file_hash(root/user/'adapter_model.safetensors')==WEIGHTS[user],'Selected Full weights mismatch')
        validate_task_config(SimpleNamespace(**d.read(root/user/'adapter_config.json')))


def import_canonical_cases(root,rows,episodes):
    with (root/'capability_per_case.csv').open(encoding='utf8',newline='') as stream:
        baseline=list(csv.DictReader(stream))
    require(len(baseline)==32,'Canonical baseline must have32 cases')
    for case,e in zip(baseline,episodes):
        require(case['source_id']==e['target_id'] and case['user']==e['target_user'] and case['conversation_id']==e['target_conversation_id'] and int(case['history_depth'])==e['history_depth'],'Canonical case identity/order mismatch')
        require(case['reference_text']==rows[e['target_index']]['reference_text'],'Canonical reference mismatch')
        case['generated_token_ids']=json.loads(case['generated_token_ids'])
        require(all(type(t) is int and t>=0 for t in case['generated_token_ids']),'Invalid canonical token IDs')
        require(len(case['generated_token_ids'])==int(case['generated_length']) and 0<len(case['generated_token_ids'])<=160,'Invalid canonical generation length')
        case['history_depth']=int(case['history_depth']);case['generated_length']=int(case['generated_length'])
        for key in ('normalized_edit_distance','position_agreement'):case[key]=float(case[key]);require(math.isfinite(case[key]),'Invalid canonical diagnostic')
    return baseline


def audit_selection(rows,selection,spans):
    episodes=selection['episodes'];require(len(episodes)==32,'Need frozen32 episodes')
    require(selection['selection_sha256']==d.b.digest(episodes),'Selection content digest mismatch')
    require(len({e['episode_id'] for e in episodes})==32,'Duplicate episodes')
    require(len({e['target_conversation_id'] for e in episodes})==32,'Need32 distinct target conversations')
    require(Counter(min(e['history_depth'],4) for e in episodes)=={1:8,2:8,3:8,4:8},'Wrong history distribution')
    require(sum(e['cross_user'] for e in episodes)==16,'Wrong cross-user count')
    spanmap={r['source_id']:r for r in spans}
    require(len(spanmap)==len(rows),'Incomplete content span audit')
    users=logical_user_assignment(rows,2,42,'multiwoz')
    for e in episodes:
        require(e['source_index']<e['target_index'],'Source must precede target')
        require(e['source_id']!=e['target_id'],'Source/target must differ')
        require(e['selected_hit_count']==1,'Expected single hit')
        require(e['cache_key']==[e['cluster'],e['token_ids']],'Cache key mismatch')
        require(e['cross_user']==(e['source_user']!=e['target_user']),'User direction mismatch')
        source,target=[rows[e[s+'_index']] for s in ('source','target')]
        require(source['token_ids']!=target['token_ids'],'Distinct full prompts required')
        require(int(target['turn_id'])>0 and min(len(target['history_source_ids']),4)==e['history_depth'],'Invalid target history')
        for side in ('source','target'):
            r=rows[e[side+'_index']];p=e[side+'_start'];span=spanmap[r['source_id']]
            require(r['source_id']==e[side+'_id'] and r['conversation_id']==e[side+'_conversation_id'],'Frozen row identity mismatch')
            require(e[side+'_user']=={'user_000':'user_a','user_001':'user_b'}[users[e[side+'_index']]],'Frozen logical user assignment mismatch')
            require(r['cluster_id']==e['cluster'],'Semantic cluster mismatch')
            require(r['semantic_execution_provenance']=='MEASURED','Measured semantic input required')
            require(r['semantic_assignment_source']==d.b.m9.assignment_source('multiwoz'),'Semantic assignment provenance mismatch')
            require(r['model_revision']==d.b.m9.MODEL_REVISION and r['tokenizer_id']==f'{d.b.m9.MODEL_ID}@{d.b.m9.MODEL_REVISION}','Pinned tokenizer required')
            require(p>=3 and r['token_ids'][p:p+3]==e['token_ids'] and len(e['token_ids'])==3,'Exact-w3 mismatch')
            require(not ({0,1,2}|set(r.get('special_token_ids',[])))&set(e['token_ids']),'Special-token hit')
            require({p,p+1,p+2}<=set(span['current_user_token_indices']),'Scaffold/current-user span violation')
            require(e[side+'_current_user_token_indices']==span['current_user_token_indices'] and e[side+'_current_user_char_span']==span['current_user_char_span'],'Episode span audit mismatch')
            text=r['prompt_text'];suffix='User: '+r['current_user_text']+'\nAssistant:'
            start=len(text)-len(suffix)+6;end=start+len(r['current_user_text'])
            require(text.endswith(suffix) and span['current_user_char_span']==[start,end],'Content character span mismatch')
            require(e[side+'_prompt_sha256']==d.b.digest(text) and e[side+'_reference_sha256']==d.b.digest(r['reference_text']),'Prompt/reference hash mismatch')
            require(bool(r['reference_text'].strip()),'Missing reference')
    windows=Counter(tuple(e['token_ids']) for e in episodes)
    require(len(windows)==27 and max(windows.values())==3,'Frozen window diversity mismatch')
    audit=dict(episode_count=32,distinct_target_conversations=32,history_depth_distribution={'1':8,'2':8,'3':8,'4':8},
        cross_user_count=16,target_turn_zero_count=0,template_scaffold_window_count=0,
        unique_exact_w3_windows=len(windows),most_frequent_window_count=max(windows.values()))
    for k,v in audit.items():
        if k in selection.get('audit',{}):require(selection['audit'][k]==v,'Stored selection audit mismatch: '+k)
    return audit


def prepare(args):
    plan=d.read(args.plan_dir/'manifest.json')
    check_fields(plan,dict(selection_version='current_user_content_v2',history_k=4,max_sequence_length=384,
        prompt_version=d.b.VERSION,model_revision=d.b.m9.MODEL_REVISION,bleu_protocol=BLEU,
        physical_safety_contract=CONTRACT['physical_safety_contract']),'B3 plan')
    require(d.b.m9.file_hash(args.plan_dir/'evaluation_selection.json')==SELECTION_SHA,'Frozen selection SHA mismatch')
    for path,sha in plan['output_hashes'].items():require(d.b.m9.file_hash(path)==sha,'B3 artifact hash mismatch')
    for path in (args.semantic,args.plan_dir/'evaluation_selection.json',args.plan_dir/'current_user_spans.json',args.plan_dir/'training_plan.json'):
        require(plan['output_hashes'].get(str(path.resolve()))==d.b.m9.file_hash(path),'Unbound frozen artifact')
    require(d.b.m9.file_hash(args.source)==plan['source_sha256'],'Source hash mismatch')
    rows=d.b.m9.read_raw(args.semantic,'multiwoz')
    raw=d.b.m9.read_raw(args.source,'multiwoz');reconstructed=d.b.reconstruct(raw,4)
    require(len(rows)==len(reconstructed),'Workload row count mismatch')
    for row,expected in zip(rows,reconstructed):
        for key in ('source_id','conversation_id','prompt_text','query_text','reference_text','current_user_text','turn_id','history_source_ids','prompt_sha256','reference_sha256'):
            require(row[key]==expected[key],'Frozen source/prompt mismatch: '+key)
    selection=d.read(args.plan_dir/'evaluation_selection.json');spans=d.read(args.plan_dir/'current_user_spans.json')
    audit=audit_selection(rows,selection,spans)
    training_path=args.adapter_root/'training_manifest.json';trained=d.read(training_path)
    check_fields(trained,{**d.SCOPE,'status':'COMPLETE','trained_adapter':True,'base_qkv_unchanged':True,'epochs':2,
        'adapter_config':d.ADAPTER_CONFIG,'evaluation_selection_sha256':SELECTION_SHA,
        'plan_manifest_sha256':d.b.m9.file_hash(args.plan_dir/'manifest.json'),
        'training_plan_sha256':d.b.m9.file_hash(args.plan_dir/'training_plan.json'),
        'semantic_sha256':d.b.m9.file_hash(args.semantic),'source_sha256':d.b.m9.file_hash(args.source)},'Full training')
    check_fields(trained['model_tokenizer_provenance'],dict(resolved_model_revision=d.b.m9.MODEL_REVISION,
        resolved_tokenizer_revision=d.b.m9.MODEL_REVISION),'Resolved training revision')
    verify_adapters(args.adapter_root,trained)
    saved=d.read(args.adapter_root/'train_selection.json');capability=d.read(args.adapter_root/'capability_validation.json')
    require(saved['profile']=='full' and saved['epochs']==2,'Expected full training profile')
    require(d.b.digest(saved)==trained['plan_sha256'] and d.b.m9.file_hash(args.adapter_root/'train_selection.json')==trained['train_selection_file_sha256'],'Training plan hash mismatch')
    require(d.b.digest(capability)==trained['capability_validation_sha256'] and d.b.m9.file_hash(args.adapter_root/'capability_validation.json')==trained['capability_validation_file_sha256'],'Capability cohort hash mismatch')
    hold={e[side+'_conversation_id'] for e in selection['episodes'] for side in ('source','target')}
    require(not hold&set(capability['conversation_ids']),'Capability/final leakage')
    frozen_training=d.read(args.plan_dir/'training_plan.json')
    require(set(frozen_training['holdout_conversation_ids'])==hold,'Frozen training holdout mismatch')
    by_id={r['source_id']:r for r in rows}
    require(len(by_id)==len(rows),'Duplicate workload IDs')
    seen=set();seen_ids=set()
    for user in d.USERS:
        cids=set(saved['users'][user]['conversation_ids'])
        require(not cids&(hold|set(capability['conversation_ids'])|seen),'Training conversation leakage');seen|=cids
        ids=saved['users'][user]['row_ids']
        require(len(set(ids))==len(ids) and not seen_ids&set(ids),'Duplicate training IDs');seen_ids.update(ids)
        require({by_id[s]['conversation_id'] for s in ids}==cids,'Training conversation/row mismatch')
        expected={sid for sid in frozen_training['users'][user]['row_ids'] if by_id[sid]['conversation_id'] not in set(capability['conversation_ids'])}
        require(set(ids)==expected,'Full pool changed or downsampled')
    decision=d.read(args.freeze_decision);train_sha=d.b.m9.file_hash(training_path)
    require(decision['training_manifest_sha256']==train_sha,'Freeze training manifest mismatch')
    cap_path=Path(decision['capability_manifest_path']);cap=d.read(cap_path)
    require(d.b.m9.file_hash(cap_path)==decision['capability_manifest_sha256'],'Freeze capability hash mismatch')
    check_fields(cap,dict(**d.SCOPE,status='COMPLETE',cohort='capability64',mode='FULL_RECOMPUTE',trained_adapter=True,
        training_manifest_sha256=train_sha,adapter_hashes=trained['adapter_hashes'],evaluation_selection_sha256=SELECTION_SHA),'Capability evidence')
    hashed_outputs(cap_path.parent,cap,['capability_per_case.csv','capability_summary.json','capability_validation.json'])
    canonical=d.read(args.baseline_root/'capability_manifest.json')
    check_fields(canonical,{**d.SCOPE,'status':'COMPLETE','cohort':'frozen32','mode':'FULL_RECOMPUTE','trained_adapter':True,
        'training_manifest_sha256':train_sha,'adapter_hashes':trained['adapter_hashes'],
        'evaluation_selection_sha256':SELECTION_SHA,'semantic_sha256':d.b.m9.file_hash(args.semantic),
        'freeze_decision_sha256':d.b.m9.file_hash(args.freeze_decision),'generation':GENERATION},'Canonical baseline')
    require(Path(decision['final_output_root']).resolve()==args.baseline_root.resolve(),'Freeze baseline directory mismatch')
    hashed_outputs(args.baseline_root,canonical,['capability_per_case.csv','capability_summary.json','capability_validation.json'])
    require(d.read(args.baseline_root/'capability_validation.json')==capability,'Canonical capability cohort mismatch')
    baseline=import_canonical_cases(args.baseline_root,rows,selection['episodes'])
    summary=d.read(args.baseline_root/'capability_summary.json')
    check_fields(summary,dict(case_count=32,bleu_protocol=BLEU,paper_bleu_claimed=False),'Canonical summary')
    for key,val in BLEU.items():require(summary['corpus_bleu'][key]==val,'Canonical BLEU protocol mismatch')
    require(math.isfinite(summary['corpus_bleu']['value']),'Nonfinite baseline BLEU')
    verify_profile(args.profile_path)
    provenance=dict(selected_adapter_provenance=trained,adapter_hashes=trained['adapter_hashes'],training_manifest_sha256=train_sha,
        capability64_manifest_sha256=d.b.m9.file_hash(cap_path),freeze_decision_sha256=d.b.m9.file_hash(args.freeze_decision),
        canonical_frozen32_manifest_sha256=d.b.m9.file_hash(args.baseline_root/'capability_manifest.json'),
        canonical_frozen32_summary_sha256=d.b.m9.file_hash(args.baseline_root/'capability_summary.json'),
        evaluation_selection_sha256=SELECTION_SHA,semantic_workload_sha256=d.b.m9.file_hash(args.semantic),
        plan_manifest_sha256=d.b.m9.file_hash(args.plan_dir/'manifest.json'),selection_audit=audit,
        canonical_full_recompute_bleu=summary['corpus_bleu']['value'])
    return rows,selection['episodes'],baseline,summary,provenance


def causal_comparisons(modes):
    values={r['mode']:r['corpus_bleu']['value'] for r in modes}
    return [dict(effect=effect,baseline_mode=base,mode=mode,baseline_bleu=values[base],mode_bleu=values[mode],
                 delta_bleu=values[mode]-values[base]) for base,mode,effect in PAIRS]


def mode_summary(mode,cases):
    result=aggregate(cases)
    return dict(**SCOPE,mode=mode,case_count=result['case_count'],corpus_bleu=result['corpus_bleu'],
        user_a_bleu=result['user_bleu'].get('user_a'),user_b_bleu=result['user_bleu'].get('user_b'),
        history_depth_bleu=result['history_depth_bleu_diagnostic'],nonempty_generation_count=result['nonempty_generation_count'],
        generated_token_count=result['generated_token_count'],
        mean_normalized_edit_distance_to_reference=result['mean_normalized_edit_distance'],
        mean_position_agreement_to_reference=result['mean_position_agreement'])


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    defaults=dict(adapter_root='results/cachegen/c6b3/b1_full',freeze_decision='results/cachegen/c6b3/b1_full_freeze_decision.json',
        baseline_root='results/cachegen/c6b3/b1_full_frozen32',plan_dir='results/cachegen/c6b3/multiwoz_plan_v2',
        source='results/workloads/multiwoz.jsonl',semantic='results/workloads/c6b3_multiwoz_history_semantic.jsonl',profile_path=PROFILE)
    for name,path in defaults.items():p.add_argument('--'+name.replace('_','-'),type=Path,default=Path(path))
    p.add_argument('--storage-src',type=Path);p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--seed',type=int,default=42);p.add_argument('--dry-run',action='store_true')
    a=p.parse_args(argv);require(a.seed==42,'Frozen seed42 required')
    require(not a.output_root.exists(),'New output root required')
    rows,episodes,baseline,summary,provenance=prepare(a)
    a.max_sequence_length=384;a.max_new_tokens=160
    manifest=dict(**SCOPE,**provenance,status='DRY_RUN' if a.dry_run else 'STARTING',
        model=d.b.m9.MODEL_ID,tokenizer=d.b.m9.MODEL_ID,revision=d.b.m9.MODEL_REVISION,
        dtype='float16',device=a.device,seed=42,prompt_version=d.b.VERSION,history_k=4,max_sequence_length=384,
        storage_profile_sha256=PROFILE_SHA,storage_policy=CONTRACT['storage_policy'],storage_transform=CONTRACT['storage_transform'],
        physical_safety_contract=CONTRACT['physical_safety_contract'],five_mode_definitions=MODES,generation=GENERATION,bleu_protocol=BLEU,
        resident_q_policy='TOTAL Q FP16 uncompressed; Q compression deferred to C7',reference_count=1,bleu_unit='BLEU points, 0-100',
        arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
        git={k:subprocess.check_output(['git',*cmd],text=True).strip() for k,cmd in [('branch',['branch','--show-current']),('commit',['rev-parse','HEAD']),('status',['status','--porcelain'])]},
        output_hashes={},runtime_claim='diagnostic counters only; no production latency claim')
    d.b.write(a.output_root/'manifest.json',manifest)
    if a.dry_run:
        print(json.dumps(dict(status='DRY_RUN',case_count=32,modes=list(MODES),canonical_bleu=summary['corpus_bleu']['value'],selection_audit=provenance['selection_audit']),indent=2));return
    from .c6b3_2_runtime import run
    try:run(a,rows,episodes,baseline,summary,manifest)
    except Exception as exc:
        manifest.update(status='FAILED',error=str(exc));(a.output_root/'manifest.json').write_text(json.dumps(manifest,indent=2));raise
