"""Explicitly invoked C6-B1 training/evaluation. Never imported by data-plan dry runs."""
import json
import random
from statistics import mean

from semcache.models.task_adapters import (USERS, ADAPTER_CONFIG, load_two_task_users,
                                          activate_task_user, validate_task_config, select_training_user)
from semcache.models.lora_fixtures import base_weight_fingerprint, assert_frozen_base
from .c6b_snips import (MODEL_ID, MODEL_REVISION, LABELS, LABEL_SERIALIZATION, SCOPE, encode_example, label_ids,
    score_candidate_logits, classify, aggregate, file_hashes, file_hash, digest,
    write_json, write_csv, normalized_label, build_plan)


def load_base(args):
    import torch
    from semcache.models.loader import load_model
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.use_deterministic_algorithms(True)
    model, tokenizer, metadata=load_model(dict(name=MODEL_ID,tokenizer=MODEL_ID,
        revision=MODEL_REVISION,tokenizer_revision=MODEL_REVISION,dtype='float16',device=args.device,
        local_files_only=True,attention_implementation='eager'))
    if (metadata['resolved_model_revision'] != MODEL_REVISION
            or metadata['resolved_tokenizer_revision'] != MODEL_REVISION):
        raise ValueError('Resolved model/tokenizer revision mismatch')
    return model,tokenizer,metadata


def train(args, rows, plan, manifest):
    import torch
    from peft import LoraConfig, get_peft_model
    from semcache.models.model_adapter import OPTModelAdapter
    if not args.device.startswith('cuda'):
        raise ValueError('Real FP16 training requires CUDA; use --dry-run for CPU data planning')
    if any(not sum(plan['counts'][u].values()) for u in USERS):
        raise ValueError('Both users need at least one training example')
    base,tokenizer,metadata=load_base(args)
    before=base_weight_fingerprint(OPTModelAdapter(base))
    by_id={r['source_id']:r for r in rows}
    encoded={r['source_id']:encode_example(tokenizer,by_id[r['source_id']],args.max_sequence_length)
             for r in plan['train_rows']}
    # Select training examples only from the frozen plan; never truncate/filter by loss.
    holdout=set(plan['holdout_ids'])
    if holdout.intersection(encoded):
        raise ValueError('Holdout leakage before training')
    configs={u:LoraConfig(**ADAPTER_CONFIG,revision=MODEL_REVISION) for u in USERS}
    torch.manual_seed(args.seed)
    model=get_peft_model(base,configs['user_a'],adapter_name='user_a')
    torch.manual_seed(args.seed+1)
    model.add_adapter('user_b',configs['user_b'])
    for config in model.peft_config.values():
        config.base_model_name_or_path=MODEL_ID
        config.revision=MODEL_REVISION
        validate_task_config(config)
    model.config.use_cache=False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        model.enable_input_require_grads()
    # Adam moments and trainable weights stay FP32; frozen OPT remains FP16.
    for name,p in model.named_parameters():
        if '.lora_A.' in name or '.lora_B.' in name:
            p.data=p.data.float()
    history={}
    for user_index,user in enumerate(USERS):
        torch.manual_seed(args.seed+user_index)
        parameters=select_training_user(model,user)
        model.train()
        optimizer=torch.optim.AdamW(parameters,lr=args.learning_rate,weight_decay=0.0)
        scaler=torch.cuda.amp.GradScaler()
        user_ids=[r['source_id'] for r in plan['train_rows'] if r['user']==user]
        losses=[]; steps=0; skipped_steps=0
        for epoch in range(args.epochs):
            epoch_losses=[]
            # Stable workload order each epoch; only training RNG affects dropout.
            for offset in range(0,len(user_ids),args.gradient_accumulation):
                group=user_ids[offset:offset+args.gradient_accumulation]
                optimizer.zero_grad(set_to_none=True)
                for sid in group:
                    example=encoded[sid]
                    with torch.autocast(device_type='cuda',dtype=torch.float16):
                        output=model(input_ids=torch.tensor([example['input_ids']],device=args.device),
                            labels=torch.tensor([example['labels']],device=args.device),use_cache=False)
                        loss=output.loss
                    if not torch.isfinite(loss):
                        raise ValueError(f'Nonfinite training loss: {user}:{sid}')
                    epoch_losses.append(float(loss.detach()))
                    scaler.scale(loss/len(group)).backward()
                    del output,loss
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(parameters,1.0)
                previous_scale=scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale()<previous_scale:
                    skipped_steps+=1
                else:
                    steps+=1
            losses.append(dict(epoch=epoch+1,mean_example_completion_loss=mean(epoch_losses)))
            print(f'{user}: epoch {epoch+1}/{args.epochs} complete',flush=True)
        if steps==0:
            raise ValueError(f'No successful optimizer updates for {user}')
        history[user]=dict(epochs=losses,optimizer_steps=steps,skipped_overflow_steps=skipped_steps,
                           initialization_seed=args.seed+user_index,training_seed=args.seed+user_index)
        model.save_pretrained(str(args.output_root),selected_adapters=[user],safe_serialization=True,save_embedding_layers=False)
        if not (args.output_root/user/'adapter_config.json').is_file():
            raise ValueError('PEFT did not save the expected standard per-user adapter directory')
        # Release gradients and optimizer states before training the other user.
        optimizer.zero_grad(set_to_none=True)
        del optimizer,scaler,parameters
    assert_frozen_base(model)
    if before != base_weight_fingerprint(OPTModelAdapter(model)):
        raise ValueError('Training changed frozen base QKV weights')
    manifest.update(status='COMPLETE',trained_adapter=True,model_tokenizer_provenance=metadata,
        resolved_model_revision=metadata['resolved_model_revision'],resolved_tokenizer_revision=metadata['resolved_tokenizer_revision'],
        adapter_hashes={u:file_hashes(args.output_root/u) for u in USERS},history=history,
        adapter_validation='standard local PEFT adapters saved; task capability not yet measured',
        optimizer=dict(name='AdamW',learning_rate=args.learning_rate,weight_decay=0.0,gradient_clip_norm=1.0,
                       adapter_parameter_dtype='float32',base_dtype='float16',batch_size=1,
                       gradient_accumulation=args.gradient_accumulation,amp='float16 GradScaler',
                       incomplete_accumulation_group='divide by actual group length'),
        base_qkv_fingerprint_sha256=digest(before),base_qkv_unchanged=True)
    write_json(args.output_root/'training_manifest.json',manifest)


def verify_training_artifacts(args, rows, plan):
    manifest_path=args.adapter_root/'training_manifest.json'
    trained=json.loads(manifest_path.read_text())
    saved_plan=json.loads((args.adapter_root/'train_selection.json').read_text())
    if trained.get('label_serialization') != LABEL_SERIALIZATION:
        raise ValueError('Training/candidate label serialization mismatch')
    if (trained.get('status')!='COMPLETE' or trained.get('trained_adapter') is not True
            or trained.get('adapter_source')!='task_finetuned_snips'
            or trained.get('base_model')!=MODEL_ID or trained.get('model_revision')!=MODEL_REVISION
            or trained.get('resolved_model_revision')!=MODEL_REVISION
            or trained.get('resolved_tokenizer_revision')!=MODEL_REVISION
            or trained.get('adapter_config')!=ADAPTER_CONFIG
            or any(trained.get(k)!=v for k,v in SCOPE.items())
            or trained.get('base_qkv_unchanged') is not True):
        raise ValueError('Completed pinned task-trained adapter provenance required')
    if (trained.get('workload_sha256')!=file_hash(args.snips)
            or trained.get('selection_sha256')!=file_hash(args.selection)
            or trained.get('train_selection_sha256')!=file_hash(args.adapter_root/'train_selection.json')):
        raise ValueError('Training workload/selection/plan hash mismatch')
    expected=build_plan(rows,json.loads(args.selection.read_text()),saved_plan['per_intent_per_user'],saved_plan['seed'])
    if (saved_plan!=expected or trained.get('train_row_ids')!=expected['train_row_ids']
            or trained.get('train_ids_sha256')!=expected['train_ids_sha256']
            or trained.get('counts')!=expected['counts']):
        raise ValueError('Training plan does not reproduce deterministic held-out selection')
    if (set(trained['train_row_ids'])&set(plan['holdout_ids'])
            or trained.get('holdout_ids')!=plan['holdout_ids']
            or trained.get('holdout_ids_sha256')!=plan['holdout_ids_sha256']):
        raise ValueError('Training/evaluation holdout leakage or mismatch')
    for user in USERS:
        if not (args.adapter_root/user/'adapter_config.json').is_file():
            raise ValueError('Missing per-user adapter')
        if file_hashes(args.adapter_root/user)!=trained['adapter_hashes'][user]:
            raise ValueError(f'Adapter files changed after training: {user}')
    return trained,expected


def evaluate(args, rows, plan, manifest):
    # Verify actual training provenance before loading a model.
    trained,training_plan=verify_training_artifacts(args,rows,plan)
    import torch
    model,tokenizer,metadata=load_base(args)
    model=load_two_task_users(model,args.adapter_root/'user_a',args.adapter_root/'user_b')
    candidates={label:label_ids(tokenizer,label) for label in LABELS}
    by_id={r['source_id']:r for r in rows}
    cases=[]
    with torch.inference_mode():
        for target in plan['targets']:
            row=by_id[target['source_id']]
            # Validate the same explicit boundary and canonical prompt IDs as training.
            encoded=encode_example(tokenizer,row,args.max_sequence_length)
            prompt=encoded['input_ids'][:encoded['prompt_length']]
            if len(prompt)+max(max(map(len,candidates.values())),args.max_new_tokens)>min(args.max_sequence_length,model.config.max_position_embeddings):
                raise ValueError('Evaluation sequence exceeds length bound; no truncation')
            activate_task_user(model,target['user'])
            scores={}
            for label,ids in candidates.items():
                output=model(input_ids=torch.tensor([prompt+ids],device=args.device),use_cache=False)
                scores[label]=score_candidate_logits(output.logits[0],len(prompt),ids)
                del output
            input_ids=torch.tensor([prompt],device=args.device)
            generated=model.generate(input_ids=input_ids,attention_mask=torch.ones_like(input_ids),
                max_new_tokens=args.max_new_tokens,do_sample=False,num_beams=1,use_cache=True,
                eos_token_id=tokenizer.eos_token_id,pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
            tokens=generated[0,len(prompt):].tolist()
            text=tokenizer.decode(tokens,skip_special_tokens=True)
            cases.append(dict(**target,**SCOPE,mode='FULL_RECOMPUTE',trained_adapter=True,
                adapter_source='task_finetuned_snips',**classify(scores,row['reference_text']),
                generated_token_ids=tokens,generated_text=text,generated_length=len(tokens),
                greedy_exact_label_match=normalized_label(text)==normalized_label(row['reference_text'])))
    summary=aggregate(cases,args.min_accuracy)
    summary.update(trained_adapter=True,adapter_source='task_finetuned_snips')
    write_csv(args.output_root/'capability_per_case.csv',cases)
    write_json(args.output_root/'capability_summary.json',summary)
    # Report the actual training plan, even when evaluator seed/cap flags differ.
    write_json(args.output_root/'train_selection.json',training_plan)
    manifest.update(status='BELOW_MIN_ACCURACY' if summary['threshold_passed'] is False else 'COMPLETE',
        trained_adapter=True,adapter_validation='both hashed local task adapters validated; base QKV unchanged',
        model_tokenizer_provenance=metadata,
        resolved_model_revision=metadata['resolved_model_revision'],resolved_tokenizer_revision=metadata['resolved_tokenizer_revision'],
        training_manifest_sha256=file_hash(args.adapter_root/'training_manifest.json'),
        train_selection_sha256=file_hash(args.output_root/'train_selection.json'),
        train_row_ids=training_plan['train_row_ids'],train_ids_sha256=training_plan['train_ids_sha256'],counts=training_plan['counts'],
        adapter_hashes=trained['adapter_hashes'],threshold_provenance=summary['threshold_provenance'],
        threshold_passed=summary['threshold_passed'],
        output_hashes={name:file_hash(args.output_root/name) for name in ('train_selection.json','capability_per_case.csv','capability_summary.json')})
    write_json(args.output_root/'capability_manifest.json',manifest)
    if summary['threshold_passed'] is False:
        raise ValueError('Configured REPRODUCTION_CHOICE minimum accuracy was not met; measured outputs were saved')
