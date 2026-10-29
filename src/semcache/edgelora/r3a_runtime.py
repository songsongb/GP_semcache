"""Opt-in R3-A GPU preparation; never imported Torch at module import time."""
import os
from pathlib import Path
import time

from .communication import digest, digest_bytes, require
from .r3a_artifacts import ENDPOINT, THRESHOLDS, flags, full_base_fingerprint, process_environment, sha, tensor_bytes, versions, write


def runtime_setup(device):
    require(str(device).startswith('cuda'), 'OPT physical/reference phase requires CUDA')
    require(os.environ.get('HF_HUB_OFFLINE') == os.environ.get('TRANSFORMERS_OFFLINE') == '1', 'offline environment required')
    import torch
    require(torch.cuda.is_available(), 'allocated CUDA device unavailable')
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    torch.cuda.set_device(device)


def memory(device):
    import torch
    free,total=torch.cuda.mem_get_info(device); props=torch.cuda.get_device_properties(device)
    return dict(name=props.name,uuid=str(getattr(props,'uuid','UNAVAILABLE')),device=str(device),
        free_bytes=free,total_bytes=total,allocated_bytes=torch.cuda.memory_allocated(device),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
        reserved_bytes=torch.cuda.memory_reserved(device))


def load_bare(identity,device):
    from semcache.models.loader import load_model
    model,tokenizer,meta=load_model(dict(name=identity['model'],tokenizer=identity['tokenizer_source'],
        revision=identity['base_revision'],tokenizer_revision=identity['tokenizer_revision'],
        device=device,dtype='float16',local_files_only=True,attention_implementation='eager'))
    require(meta['resolved_model_revision'] == identity['base_revision'] and
            meta['resolved_tokenizer_revision'] == identity['tokenizer_revision'] and
            meta['tokenizer_source_id'] == identity['tokenizer_source'], 'INVALID: model/tokenizer snapshots mismatch')
    require(not getattr(model,'is_quantized',False) and model.config._attn_implementation == 'eager', 'unquantized eager OPT required')
    return model,tokenizer,meta


def prepare_reference(config,output):
    """One PEFT model in an isolated child; process exit releases its CUDA state."""
    from semcache.models.loader import load_model
    from semcache.models.model_adapter import OPTModelAdapter
    from semcache.models.lora_fixtures import initialize_fixture
    from semcache.models.lora_decomposition import projection_parts, validate_projection
    output=Path(output);device=config['device'];runtime_setup(device)
    import torch
    from peft import LoraConfig,get_peft_model
    torch.manual_seed(config['seed'])
    begin=time.monotonic_ns()
    model,tokenizer,meta=load_model(dict(name=config['model'],tokenizer=config['model'],device=device,dtype='float16',
        local_files_only=True,attention_implementation='eager'))
    require(not getattr(model,'is_quantized',False), 'quantized base forbidden')
    base_identity=full_base_fingerprint(model)
    layers,hidden=len(OPTModelAdapter(model).layers),model.config.hidden_size
    require((layers,hidden) == ((32,2560) if config['model'].endswith('2.7b') else (12,768)), 'unexpected OPT layout')
    from semcache.models.tokenizer_provenance import is_snapshot_commit
    require(is_snapshot_commit(meta['resolved_model_revision']) and is_snapshot_commit(meta['resolved_tokenizer_revision']),
            'resolved immutable model/tokenizer snapshots required')
    text='The edge server and user device jointly evaluate a controlled language model request. '*16
    tokens=tokenizer(text,add_special_tokens=True,truncation=False)['input_ids'][:32]
    require(len(tokens)==32, 'tokenizer did not supply 32 tokens')
    model=get_peft_model(model,LoraConfig(r=8,lora_alpha=8,lora_dropout=0.,bias='none',
        target_modules=['q_proj','k_proj','v_proj'],task_type='CAUSAL_LM'),adapter_name='user_a',autocast_adapter_dtype=True)
    adapter=OPTModelAdapter(model)
    initialize_fixture(adapter,'user_a',config['seed'],scale=.01)
    model.set_adapter('user_a');model.eval()
    require(full_base_fingerprint(model.get_base_model(),peft_wrapped=True)['sha256']==base_identity['sha256'],
            'INVALID: base state changed during PEFT adapter preparation')
    records=[];blob=bytearray()
    for layer in range(layers):
        for role,module in adapter.projection_modules(layer).items():
            validate_projection(module,'user_a')
            require(module.scaling['user_a']==1.,'unexpected PEFT scaling')
            for kind,matrix in (('A',module.lora_A['user_a']),('B',module.lora_B['user_a'])):
                require(matrix.weight.dtype==torch.float32 and matrix.bias is None,'ARCHITECTURE_FAIL: expected actual FP32 PEFT A/B')
                raw=tensor_bytes(matrix.weight)
                require(bool(matrix.weight.count_nonzero()),'zero controlled fixture matrix')
                records.append(dict(layer=layer,role=role,matrix=kind,dtype='float32',shape=list(matrix.weight.shape),
                    offset=len(blob),sha256=digest_bytes(raw)))
                blob.extend(raw)
    (output/'r3a_adapter.bin').write_bytes(blob)
    identity=dict(model=config['model'],dtype='float16',layers=layers,hidden_size=hidden,length=32,
        hidden_dtype='float16',delta_dtype='float32',adapter='user_a',rank=8,seed=config['seed'],
        adapter_matrix_sha256=sha(output/'r3a_adapter.bin'),base_weight_sha256=base_identity['sha256'],
        base_revision=meta['resolved_model_revision'],tokenizer_source=meta['tokenizer_source_id'],
        tokenizer_revision=meta['resolved_tokenizer_revision'],token_ids=tokens,token_ids_sha256=digest(tokens),
        endpoint=dict(ENDPOINT),versions=versions(),
        precision='native FP16 base + actual FP32 PEFT A/B; CPU F.linear FP32; bundled FP32 delta wire; (FP16 base + FP32 delta).to(FP16)')
    write(output/'r3a_base_fingerprint.json',base_identity)
    write(output/'r3a_adapter_manifest.json',dict(identity=identity,adapter='user_a',rank=8,scale=1.,dtypes=['float32'],
        all_matrices_nonzero=True,trained_adapter=False,adapter_source='controlled_untrained_fixture',
        initialization='existing initialize_fixture, CPU float32 normal std=.01; seed from CLI',matrices=records,
        matrix_file='r3a_adapter.bin',matrix_sha256=identity['adapter_matrix_sha256']))
    seen={};handles=[];hidden_seals={};intermediates={}
    def intermediate_hook(key):
        def capture(module,args,out):
            intermediates[key]=dict(input_dtype=str(args[0].dtype),output_dtype=str(out.dtype))
        return capture
    def hook(layer,role):
        def capture(module,args,out):
            key=f'{layer}.{role}'
            require(key not in seen,'duplicate native reference projection')
            if layer not in hidden_seals: hidden_seals[layer]=args[0].detach().clone()
            require(torch.equal(hidden_seals[layer],args[0]),'ARCHITECTURE_FAIL: native QKV inputs differ')
            base,delta,combined=projection_parts(module,args[0],'user_a')
            require(torch.equal(out,combined) and bool(delta.count_nonzero()),'native PEFT decomposition/nonzero gate failed')
            seen[key]=dict(hidden_sha256=digest_bytes(tensor_bytes(args[0])),delta_sha256=digest_bytes(tensor_bytes(delta)),
                native_output_sha256=digest_bytes(tensor_bytes(out)),input_dtype=str(args[0].dtype),
                A_dtype=str(module.lora_A['user_a'].weight.dtype),B_dtype=str(module.lora_B['user_a'].weight.dtype),
                A_intermediate=intermediates[key+'.A'],B_intermediate=intermediates[key+'.B'],
                intermediate_dtype=str(delta.dtype),base_dtype=str(base.dtype),final_dtype=str(out.dtype),nonzero_delta=True)
        return capture
    try:
        for layer in range(layers):
            for role,module in adapter.projection_modules(layer).items():
                key=f'{layer}.{role}'
                handles.append(module.lora_A['user_a'].register_forward_hook(intermediate_hook(key+'.A')))
                handles.append(module.lora_B['user_a'].register_forward_hook(intermediate_hook(key+'.B')))
                handles.append(module.register_forward_hook(hook(layer,role)))
        write(output/'r3a_prepare_execution.json',dict(gpu_inference_performed=True,stage='native_reference_forward_entered'))
        with torch.inference_mode():
            result=model(input_ids=torch.tensor([tokens],device=device),use_cache=False,output_attentions=True)
            torch.cuda.synchronize(device)
        require(len(seen)==3*layers,'incomplete native reference projections')
        require(full_base_fingerprint(model.get_base_model(),peft_wrapped=True)['sha256']==base_identity['sha256'],
                'INVALID: base state changed during native reference')
        raw=tensor_bytes(result.logits.float())
        (output/'r3a_reference_logits.bin').write_bytes(raw)
        write(output/'r3a_reference.json',dict(identity=identity,thresholds=THRESHOLDS,
            logits_file='r3a_reference_logits.bin',logits_sha256=sha(output/'r3a_reference_logits.bin'),
            logits_dtype='float32',native_logits_dtype=str(result.logits.dtype),logits_shape=list(result.logits.shape),native_projection_outputs=len(seen),
            projections=seen,tokenizer_metadata=meta,reference_is_task_quality_evidence=False,
            reference_clock='isolated preparation; never part of distributed request wall'))
    finally:
        for handle in handles: handle.remove()
    return dict(identity=identity,identity_sha256=digest(identity),gpu=memory(device),versions=versions(),environment=process_environment(),
        preparation_wall_ms=(time.monotonic_ns()-begin)/1e6,**flags(gpu=True))
