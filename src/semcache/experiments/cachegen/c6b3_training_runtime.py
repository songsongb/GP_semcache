"""Explicit CUDA training, shared pinned loader/adapter guards, restartable state."""
import json
from . import c6b3_train as d
from .c6b_runtime import load_base
from semcache.models.task_adapters import ADAPTER_CONFIG, USERS, select_training_user, validate_task_config
from semcache.models.lora_fixtures import base_weight_fingerprint, assert_frozen_base


def save_json(path,value):
    path.write_text(json.dumps(value,indent=2,sort_keys=True))


def train(args,plan,encoded,manifest):
    import torch
    from peft import LoraConfig,get_peft_model
    from semcache.models.model_adapter import OPTModelAdapter
    if not args.device.startswith('cuda'):raise ValueError('Real FP16 training requires CUDA')
    base,tokenizer,metadata=load_base(args)
    before=base_weight_fingerprint(OPTModelAdapter(base))
    model=get_peft_model(base,LoraConfig(**ADAPTER_CONFIG,revision=d.b.m9.MODEL_REVISION),adapter_name='user_a')
    torch.manual_seed(43)
    model.add_adapter('user_b',LoraConfig(**ADAPTER_CONFIG,revision=d.b.m9.MODEL_REVISION))
    for config in model.peft_config.values():
        config.base_model_name_or_path=d.b.m9.MODEL_ID;config.revision=d.b.m9.MODEL_REVISION
        validate_task_config(config)
    model.config.use_cache=False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.enable_input_require_grads()
    for name,p in model.named_parameters():
        if '.lora_A.' in name or '.lora_B.' in name:p.data=p.data.float()
    resume=None
    if args.resume:
        audit=d.read(args.resume/'checkpoint.json')
        if audit['plan_sha256']!=manifest['plan_sha256'] or audit['optimizer']!=manifest['optimizer'] or audit['inputs']!={k:manifest[k] for k in ('plan_manifest_sha256','source_sha256','semantic_sha256')}:
            raise ValueError('Resume configuration mismatch')
        if d.b.m9.file_hash(args.resume/'state.pt')!=audit['state_sha256']:raise ValueError('Checkpoint hash mismatch')
        # Locally produced tensor-only state; no arbitrary checkpoint pickle execution.
        resume=torch.load(args.resume/'state.pt',map_location='cpu',weights_only=True)
        params=dict(model.named_parameters())
        expected={n for n in params if '.lora_A.' in n or '.lora_B.' in n}
        if set(resume['adapters'])!=expected:raise ValueError('Incomplete checkpoint adapter state')
        with torch.no_grad():
            for name,tensor in resume['adapters'].items():
                if '.lora_A.' not in name and '.lora_B.' not in name:raise ValueError('Non-LoRA checkpoint tensor')
                params[name].copy_(tensor)
    steps=0
    for ui,user in enumerate(USERS):
        if resume and ui<resume['user_index']:continue
        parameters=select_training_user(model,user);model.train()
        for name,p in model.named_parameters():
            if p.requires_grad and (not any('.'+q+'.' in name for q in ('q_proj','k_proj','v_proj')) or '.lora_' not in name):
                raise ValueError('Unexpected trainable base/non-QKV parameter')
        optimizer=torch.optim.AdamW(parameters,lr=args.learning_rate,weight_decay=0)
        scaler=torch.cuda.amp.GradScaler()
        user_start_steps=steps
        start_epoch=0;start_offset=0
        if resume and ui==resume['user_index']:
            optimizer.load_state_dict(resume['optimizer']);scaler.load_state_dict(resume['scaler'])
            start_epoch=resume['epoch'];start_offset=resume['offset'];steps=resume['steps']
            torch.set_rng_state(resume['cpu_rng']);torch.cuda.set_rng_state_all(resume['cuda_rng'])
        else:torch.manual_seed(42+ui)
        for epoch in range(start_epoch,plan['epochs']):
            ids=d.epoch_ids(plan,user,epoch)
            for offset in range(start_offset if epoch==start_epoch else 0,len(ids),args.gradient_accumulation):
                group=ids[offset:offset+args.gradient_accumulation];optimizer.zero_grad(set_to_none=True)
                for sid in group:
                    example=encoded[sid]
                    with torch.autocast(device_type='cuda',dtype=torch.float16):
                        loss=model(input_ids=torch.tensor([example['input_ids']],device=args.device),
                            labels=torch.tensor([example['labels']],device=args.device),use_cache=False).loss
                    if not torch.isfinite(loss):raise ValueError('Nonfinite completion loss')
                    scaler.scale(loss/len(group)).backward()
                scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(parameters,1.0)
                scale=scaler.get_scale();scaler.step(optimizer);scaler.update()
                updated=scaler.get_scale()>=scale
                if updated:steps+=1
                if updated and steps%args.checkpoint_steps==0:
                    checkpoint=args.output_root/'checkpoints'/f'{user}-step-{steps}'
                    checkpoint.mkdir(parents=True,exist_ok=False)
                    model.save_pretrained(str(checkpoint),safe_serialization=True,save_embedding_layers=False)
                    state=dict(adapters={n:p.detach().cpu() for n,p in model.named_parameters() if '.lora_A.' in n or '.lora_B.' in n},
                        optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),user_index=ui,epoch=epoch,
                        offset=offset+len(group),steps=steps,cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all())
                    torch.save(state,checkpoint/'state.pt')
                    d.b.write(checkpoint/'checkpoint.json',dict(plan_sha256=manifest['plan_sha256'],optimizer=manifest['optimizer'],
                        inputs={k:manifest[k] for k in ('plan_manifest_sha256','source_sha256','semantic_sha256')},
                        state_sha256=d.b.m9.file_hash(checkpoint/'state.pt')))
            print(f'{user}: epoch {epoch+1}/{plan["epochs"]}',flush=True)
        if steps==user_start_steps and not resume:raise ValueError('No successful optimizer updates')
        optimizer.zero_grad(set_to_none=True)
        del optimizer,scaler,parameters
    assert_frozen_base(model)
    if before!=base_weight_fingerprint(OPTModelAdapter(model)):raise ValueError('Base QKV changed')
    model.save_pretrained(str(args.output_root),safe_serialization=True,save_embedding_layers=False)
    manifest.update(status='COMPLETE',trained_adapter=True,base_qkv_unchanged=True,
        base_qkv_fingerprint_sha256=d.b.digest(before),model_tokenizer_provenance=metadata,
        adapter_hashes={u:d.file_hashes(args.output_root/u) for u in USERS},optimizer_steps=steps,
        capability_validation_file_sha256=d.b.m9.file_hash(args.output_root/'capability_validation.json'),
        train_selection_file_sha256=d.b.m9.file_hash(args.output_root/'train_selection.json'),
        software=dict(torch=torch.__version__,cuda=torch.version.cuda),gpu=torch.cuda.get_device_name(args.device))
    save_json(args.output_root/'training_manifest.json',manifest)
