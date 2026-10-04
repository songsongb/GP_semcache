"""FULL_RECOMPUTE response validation; capability64 and gated final32 are separate."""
import argparse
import math
from statistics import mean
from . import c6b3_train as d
from .c6_quality import generation_metrics,write_csv
from semcache.evaluation.bleu import compute_bleu


def aggregate(cases):
    def score(group):return compute_bleu([r['generated_text'] for r in group],[r['reference_text'] for r in group],**d.b.BLEU)
    return dict(**d.SCOPE,case_count=len(cases),corpus_bleu=score(cases),
        user_bleu={u:score([r for r in cases if r['user']==u]) for u in d.USERS if any(r['user']==u for r in cases)},
        history_depth_bleu_diagnostic={str(k):score([r for r in cases if r['history_depth']==k]) for k in range(1,5) if any(r['history_depth']==k for r in cases)},
        nonempty_generation_count=sum(bool(r['generated_text'].strip()) for r in cases),
        generated_token_count=sum(r['generated_length'] for r in cases),
        mean_normalized_edit_distance=mean(r['normalized_edit_distance'] for r in cases),
        mean_position_agreement=mean(r['position_agreement'] for r in cases))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);d.common_args(p)
    p.add_argument('--adapter-root',type=d.Path)
    p.add_argument('--base',action='store_true')
    p.add_argument('--cohort',choices=['capability64','frozen32'],default='capability64')
    p.add_argument('--freeze-decision',type=d.Path,help='Manual JSON binding training manifest and completed capability manifest hashes')
    a=p.parse_args(argv);a.seed=42
    if a.output_root.exists():raise ValueError('New evaluation output root required')
    if a.base==bool(a.adapter_root):raise ValueError('Choose --base OR --adapter-root')
    rows,training,frozen,provenance=d.load_inputs(a.plan_dir,a.source,a.semantic)
    capability=d.capability_cohort(rows,training)
    trained=None
    if a.adapter_root:
        trained=d.read(a.adapter_root/'training_manifest.json')
        for k,v in {**d.SCOPE,**provenance}.items():
            if trained.get(k)!=v:raise ValueError('Training provenance mismatch: '+k)
        if trained.get('status')!='COMPLETE' or trained.get('base_qkv_unchanged') is not True:raise ValueError('Completed frozen-base adapters required')
        if d.read(a.adapter_root/'capability_validation.json')!=capability:raise ValueError('Capability cohort mismatch')
        if d.b.m9.file_hash(a.adapter_root/'capability_validation.json')!=trained['capability_validation_file_sha256']:
            raise ValueError('Capability cohort file changed')
        for user in d.USERS:
            if d.file_hashes(a.adapter_root/user)!=trained['adapter_hashes'][user]:raise ValueError('Adapter hashes changed')
        saved_plan=d.read(a.adapter_root/'train_selection.json')
        replay=d.training_subset(rows,training,capability,saved_plan['profile'])
        for user in d.USERS:
            for key,value in replay['users'][user].items():
                if saved_plan['users'][user].get(key)!=value:raise ValueError('Training subset mismatch')
        if d.b.digest(saved_plan)!=trained['plan_sha256']:raise ValueError('Training plan digest mismatch')
        if trained.get('adapter_config')!=d.ADAPTER_CONFIG or trained.get('trained_adapter') is not True:
            raise ValueError('Task-trained adapter configuration mismatch')
        if d.b.m9.file_hash(a.adapter_root/'train_selection.json')!=trained['train_selection_file_sha256']:raise ValueError('Training plan changed')
    if a.cohort=='frozen32':
        if a.base or not a.freeze_decision:raise ValueError('Final32 requires manually frozen trained adapter decision')
        decision=d.read(a.freeze_decision)
        if decision['training_manifest_sha256']!=d.b.m9.file_hash(a.adapter_root/'training_manifest.json'):raise ValueError('Manual adapter freeze mismatch')
        if d.Path(decision['final_output_root']).resolve()!=a.output_root.resolve():
            raise ValueError('Use the single output root bound to the manual final32 decision')
        cap_path=d.Path(decision['capability_manifest_path']);cap=d.read(cap_path)
        if (decision['capability_manifest_sha256']!=d.b.m9.file_hash(cap_path) or cap['status']!='COMPLETE'
            or cap['cohort']!='capability64' or cap['training_manifest_sha256']!=decision['training_manifest_sha256']):
            raise ValueError('Completed capability64 decision evidence required')
        for name,sha in cap['output_hashes'].items():
            if d.b.m9.file_hash(cap_path.parent/name)!=sha:raise ValueError('Capability output changed')
        examples=[dict(source_id=e['target_id'],user=e['target_user'],history_depth=e['history_depth'],
                       conversation_id=e['target_conversation_id']) for e in frozen['episodes']]
    else:examples=capability['examples']
    manifest=dict(**d.SCOPE,**provenance,status='DRY_RUN' if a.dry_run else 'STARTING',cohort=a.cohort,
        mode='FULL_RECOMPUTE',trained_adapter=not a.base,capability_validation_sha256=d.b.digest(capability),
        training_manifest_sha256=d.b.m9.file_hash(a.adapter_root/'training_manifest.json') if a.adapter_root else None,
        adapter_hashes=trained['adapter_hashes'] if trained else None,
        final32_policy='manual capability64 review -> adapter freeze -> final32 once; never stopping criterion',
        freeze_decision_sha256=d.b.m9.file_hash(a.freeze_decision) if a.freeze_decision else None,
        generation=dict(do_sample=False,num_beams=1,max_new_tokens=160,respect_eos=True),
        fidelity_reference='tokenized leading-space reference response plus EOS; not compression fidelity')
    d.b.write(a.output_root/'capability_validation.json',capability)
    d.b.write(a.output_root/'capability_manifest.json',manifest)
    if a.dry_run:return
    import torch
    from .c6b_runtime import load_base
    from semcache.models.task_adapters import load_two_task_users,activate_task_user
    model,tokenizer,metadata=load_base(a)
    if not a.base:model=load_two_task_users(model,a.adapter_root/'user_a',a.adapter_root/'user_b')
    model.requires_grad_(False);model.eval()
    by_id={r['source_id']:r for r in rows};cases=[]
    with torch.inference_mode():
        for target in examples:
            row=by_id[target['source_id']];encoded=d.encode_example(tokenizer,row)
            prompt=encoded['input_ids'][:encoded['prompt_length']]
            if len(prompt)+160>model.config.max_position_embeddings:raise ValueError('Model context exceeded')
            if not a.base:activate_task_user(model,target['user'])
            inputs=torch.tensor([prompt],device=a.device)
            generated=model.generate(input_ids=inputs,attention_mask=torch.ones_like(inputs),do_sample=False,num_beams=1,
                max_new_tokens=160,use_cache=True,eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
            tokens=generated[0,len(prompt):].tolist()
            loss=model(input_ids=torch.tensor([encoded['input_ids']],device=a.device),
                labels=torch.tensor([encoded['labels']],device=a.device),use_cache=False).loss.item()
            if not math.isfinite(loss):raise ValueError('Nonfinite teacher-forced NLL')
            cases.append(dict(**target,reference_text=row['reference_text'],generated_text=tokenizer.decode(tokens,skip_special_tokens=True),
                generated_token_ids=tokens,**generation_metrics(encoded['input_ids'][len(prompt):],tokens),
                response_nll=loss,response_perplexity=math.exp(loss) if loss<700 else None))
    summary=aggregate(cases)
    write_csv(a.output_root/'capability_per_case.csv',cases);d.b.write(a.output_root/'capability_summary.json',summary)
    manifest.update(status='COMPLETE',model_tokenizer_provenance=metadata,
        software=dict(torch=torch.__version__,cuda=torch.version.cuda),
        output_hashes={name:d.b.m9.file_hash(a.output_root/name) for name in ['capability_per_case.csv','capability_summary.json','capability_validation.json']})
    from .c6b3_training_runtime import save_json
    save_json(a.output_root/'capability_manifest.json',manifest)
