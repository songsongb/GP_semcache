"""Frozen B3-1 data planning and explicit offline training entry point."""
import argparse
import json
import random
import subprocess
from collections import defaultdict
from pathlib import Path
from . import c6b3_multiwoz as b
from . import c6b3_selection as selection_v2
from semcache.models.task_adapters import USERS, ADAPTER_CONFIG
from .c6b_snips import file_hashes

SCOPE=dict(task_capability_stage='C6-B3-1',paper_bleu_claimed=False,
    protocol_provenance='REPRODUCTION_CHOICE',semantic_reuse_enabled=False,compression_enabled=False,
    model_id=b.m9.MODEL_ID,model_revision=b.m9.MODEL_REVISION,history_k=4,max_sequence_length=384,
    prompt_version=b.VERSION,bleu_protocol=b.BLEU,
    completion_serialization='one leading ASCII space + original reference response + EOS')


def read(path):return json.loads(Path(path).read_text())


def load_inputs(plan_dir, source, semantic):
    manifest=read(plan_dir/'manifest.json')
    expected=dict(selection_version='current_user_content_v2',history_k=4,max_sequence_length=384,
        prompt_version=b.VERSION,model_revision=b.m9.MODEL_REVISION,bleu_protocol=b.BLEU,
        physical_safety_contract='PHYSICAL_EXACT_W3_WITHIN_SEMANTIC_CLUSTER')
    for k,v in expected.items():
        if manifest.get(k)!=v:raise ValueError('Frozen B3 provenance mismatch: '+k)
    if b.m9.file_hash(source)!=manifest['source_sha256']:raise ValueError('Source changed')
    for path,sha in manifest['output_hashes'].items():
        if b.m9.file_hash(path)!=sha:raise ValueError('Frozen artifact changed: '+path)
    for path in (plan_dir/'training_plan.json',plan_dir/'evaluation_selection.json',plan_dir/'current_user_spans.json',semantic):
        if manifest['output_hashes'].get(str(path.resolve()))!=b.m9.file_hash(path):
            raise ValueError('Required artifact not bound to frozen manifest')
    rows=b.m9.read_raw(semantic,'multiwoz');raw=b.m9.read_raw(source,'multiwoz')
    expected_rows=b.reconstruct(raw,4)
    if len(rows)!=len(expected_rows):raise ValueError('Row count changed')
    for r,e in zip(rows,expected_rows):
        for k in ('source_id','conversation_id','prompt_text','reference_text','current_user_text','turn_id','history_source_ids','prompt_sha256','reference_sha256'):
            if r[k]!=e[k]:raise ValueError('Frozen row changed: '+k)
    spans=read(plan_dir/'current_user_spans.json')
    if [r['source_id'] for r in rows]!=[r['source_id'] for r in spans]:raise ValueError('Span identities changed')
    annotated=[dict(r,**{k:v for k,v in s.items() if k!='source_id'}) for r,s in zip(rows,spans)]
    episodes,_=selection_v2.select(annotated)
    frozen=read(plan_dir/'evaluation_selection.json')
    for new,old in zip(episodes,frozen['episodes']):
        if any(old.get(k)!=v for k,v in new.items()):raise ValueError('Frozen selection does not replay')
    if len(frozen['episodes'])!=32 or b.digest(frozen['episodes'])!=frozen['selection_sha256']:
        raise ValueError('Invalid frozen selection')
    training=read(plan_dir/'training_plan.json')
    hold={e[side+'_conversation_id'] for e in frozen['episodes'] for side in ('source','target')}
    if set(training['holdout_conversation_ids'])!=hold:raise ValueError('Evaluation holdout mismatch')
    by_id={r['source_id']:r for r in rows}
    all_ids=[];owners={}
    for user in USERS:
        ids=training['users'][user]['row_ids'];all_ids+=ids
        for sid in ids:
            cid=by_id[sid]['conversation_id']
            if cid in hold or (cid in owners and owners[cid]!=user):raise ValueError('Conversation leakage')
            owners[cid]=user
    if len(set(all_ids))!=len(all_ids) or set(all_ids)!={r['source_id'] for r in rows if r['conversation_id'] not in hold}:
        raise ValueError('Training pool is incomplete or duplicated')
    remaining=[r for r in rows if r['conversation_id'] not in hold]
    assignments=b.assign_users(remaining,2,42,'seeded_group_balanced')
    for user,logical in [('user_a','user_000'),('user_b','user_001')]:
        if training['users'][user]['row_ids']!=[r['source_id'] for r,u in zip(remaining,assignments) if u==logical]:
            raise ValueError('Frozen training user assignment mismatch')
    if set(training['train_row_ids'])!=set(all_ids) or set(training['training_conversation_ids'])!=set(owners):
        raise ValueError('Training identity lists mismatch')
    for key,values in [('train_row_ids',training['train_row_ids']),('training_conversation_ids',training['training_conversation_ids']),('holdout_conversation_ids',training['holdout_conversation_ids'])]:
        if b.digest(values)!=training[key+'_sha256']:raise ValueError('Training ID hash mismatch')
    return rows,training,frozen,dict(plan_manifest_sha256=b.m9.file_hash(plan_dir/'manifest.json'),
        training_plan_sha256=b.m9.file_hash(plan_dir/'training_plan.json'),
        evaluation_selection_sha256=b.m9.file_hash(plan_dir/'evaluation_selection.json'),
        source_sha256=b.m9.file_hash(source),semantic_sha256=b.m9.file_hash(semantic))


def capability_cohort(rows,training):
    by_id={r['source_id']:r for r in rows};chosen=[]
    for user in USERS:
        pools={k:{} for k in range(1,5)}
        for sid in training['users'][user]['row_ids']:
            r=by_id[sid];depth=min(int(r['turn_id'])//2,4)
            if depth:pools[depth].setdefault(r['conversation_id'],r)
        ordered={k:sorted(v,key=lambda cid:(b.digest(['b3cap',42,user,cid]),cid)) for k,v in pools.items()}
        owners={};assigned={}
        def match(slot,seen):
            for cid in ordered[slot[0]]:
                if cid in seen:continue
                seen.add(cid)
                if cid not in owners or match(owners[cid],seen):
                    owners[cid]=slot;assigned[slot]=pools[slot[0]][cid];return True
            return False
        for k in range(1,5):
            for i in range(8):
                if not match((k,i),set()):raise ValueError(f'Insufficient capability conversations for {user}: '+str({k:len(v) for k,v in pools.items()}))
        for slot in sorted(assigned):
            r=assigned[slot]
            chosen.append(dict(source_id=r['source_id'],conversation_id=r['conversation_id'],user=user,
                history_depth=slot[0],prompt_sha256=r['prompt_sha256'],reference_sha256=r['reference_sha256']))
    cids=[r['conversation_id'] for r in chosen]
    if len(set(cids))!=64 or set(cids)&set(training['holdout_conversation_ids']):raise ValueError('Capability overlap')
    return dict(**SCOPE,cohort='capability64',seed=42,examples=chosen,conversation_ids=cids,
                conversation_ids_sha256=b.digest(cids),examples_sha256=b.digest(chosen))


def training_subset(rows,training,capability,profile):
    by_id={r['source_id']:r for r in rows};excluded=set(capability['conversation_ids'])|set(training['holdout_conversation_ids'])
    users={}
    for user in USERS:
        groups=defaultdict(list)
        for sid in training['users'][user]['row_ids']:
            r=by_id[sid]
            if r['conversation_id'] not in excluded:groups[r['conversation_id']].append(sid)
        cids=list(groups)
        random.Random(int(b.digest(['b3subset',42,user]),16)).shuffle(cids)
        selected=[];used=[]
        for cid in cids:
            if profile=='pilot' and len(selected)+len(groups[cid])>5000:continue
            used.append(cid);selected.extend(groups[cid])
        if not selected:raise ValueError('Empty training user')
        users[user]=dict(row_ids=selected,conversation_ids=used,row_count=len(selected),conversation_count=len(used),
                         row_ids_sha256=b.digest(selected),conversation_ids_sha256=b.digest(used))
    return dict(**SCOPE,profile=profile,epochs=3 if profile=='pilot' else 2,users=users,
                excluded_conversation_ids=sorted(excluded),capability_validation_sha256=b.digest(capability),
                selection_rule='seed42 per-user conversation shuffle; whole groups; skip groups exceeding pilot5000 cap; full includes all')


def encode_example(tokenizer,row):
    prompt=tokenizer(row['prompt_text'],add_special_tokens=True,truncation=False)['input_ids']
    if prompt!=row['token_ids']:raise ValueError('Frozen prompt IDs changed')
    completion=tokenizer(' '+row['reference_text'],add_special_tokens=False,truncation=False)['input_ids']
    if tokenizer.eos_token_id is None or not completion:raise ValueError('Missing completion/EOS')
    completion=completion+[tokenizer.eos_token_id]
    if len(prompt)+len(completion)>384:raise ValueError('Sequence exceeds frozen384 bound')
    return dict(input_ids=prompt+completion,labels=[-100]*len(prompt)+completion,prompt_length=len(prompt))


def epoch_ids(plan,user,epoch):
    ids=list(plan['users'][user]['row_ids'])
    random.Random(int(b.digest(['b3epoch',42,user,epoch]),16)).shuffle(ids)
    return ids


def common_args(p):
    p.add_argument('--plan-dir',type=Path,default=Path('results/cachegen/c6b3/multiwoz_plan_v2'))
    p.add_argument('--source',type=Path,default=Path('results/workloads/multiwoz.jsonl'))
    p.add_argument('--semantic',type=Path,default=Path('results/workloads/c6b3_multiwoz_history_semantic.jsonl'))
    p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--dry-run',action='store_true')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);common_args(p)
    p.add_argument('--profile',choices=['pilot','full'],required=True)
    p.add_argument('--learning-rate',type=float,default=1e-4)
    p.add_argument('--gradient-accumulation',type=int,default=8)
    p.add_argument('--checkpoint-steps',type=int,default=500)
    p.add_argument('--resume',type=Path)
    a=p.parse_args(argv);a.seed=42
    if a.learning_rate<=0 or a.gradient_accumulation<1 or a.checkpoint_steps<1:raise ValueError('Invalid optimizer arguments')
    if a.output_root.exists():raise ValueError('Use a NEW output root, including for resume')
    rows,training,frozen,provenance=load_inputs(a.plan_dir,a.source,a.semantic)
    capability=capability_cohort(rows,training);plan=training_subset(rows,training,capability,a.profile)
    tokenizer,token_metadata=b.tokenizer_only()
    selected_ids={sid for v in plan['users'].values() for sid in v['row_ids']}
    encoded={r['source_id']:encode_example(tokenizer,r) for r in rows if r['source_id'] in selected_ids}
    for user,v in plan['users'].items():
        v.update(expected_tokens=sum(len(encoded[s]['input_ids']) for s in v['row_ids']),
                 supervised_tokens=sum(sum(t!=-100 for t in encoded[s]['labels']) for s in v['row_ids']))
    manifest=dict(**SCOPE,**provenance,status='DRY_RUN' if a.dry_run else 'STARTING',seed=42,
        adapter_source='task_finetuned_multiwoz',adapter_config=ADAPTER_CONFIG,tokenizer=token_metadata,
        plan_sha256=b.digest(plan),capability_validation_sha256=b.digest(capability),
        epochs=plan['epochs'],optimizer=dict(name='AdamW',learning_rate=a.learning_rate,weight_decay=0,
            batch_size=1,gradient_accumulation=a.gradient_accumulation,gradient_clip_norm=1.0,amp='fp16',gradient_checkpointing=True),
        checkpoint_steps=a.checkpoint_steps,epoch_order_hashes={u:[b.digest(epoch_ids(plan,u,e)) for e in range(plan['epochs'])] for u in USERS},
        git={k:subprocess.check_output(['git',*cmd],text=True).strip() for k,cmd in [('commit',['rev-parse','HEAD']),('branch',['branch','--show-current']),('status',['status','--porcelain'])]},
        final32_policy='not used for training-size decisions or stopping; manual adapter freeze before final32')
    for name,value in [('capability_validation.json',capability),('train_selection.json',plan),('training_manifest.json',manifest)]:b.write(a.output_root/name,value)
    print(json.dumps({u:{k:v for k,v in info.items() if not k.endswith('ids')} for u,info in plan['users'].items()},indent=2))
    if not a.dry_run:
        from .c6b3_training_runtime import train
        train(a,plan,encoded,manifest)
