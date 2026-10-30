"""Bounded actual-tensor Native/ES triage, separate from serving timing."""
from contextlib import contextmanager
from pathlib import Path

from .communication import digest_bytes,require
from .r3a_artifacts import check_parity,read,sha,tensor_bytes,tensor_from_bytes,unpack,write


@contextmanager
def capture(adapter,directory):
    import torch
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True);handles=[];records=[];size=0
    def hook(key):
        def observe(module,args,kwargs,out):
            nonlocal size
            value=out[0] if isinstance(out,tuple) else out
            x=args[0] if args else kwargs.get('hidden_states');require(torch.is_tensor(value) and torch.is_tensor(x),'unsupported component output/input')
            item=dict(key=key)
            for name,t in (('input',x),('output',value)):
                blob=tensor_bytes(t);size+=len(blob);require(size<=128*2**20,'bounded triage capture exhausted')
                file=f'{key}_{name}.bin';(directory/file).write_bytes(blob)
                item[name]=dict(file=file,sha256=digest_bytes(blob),shape=list(t.shape),dtype=str(t.dtype).removeprefix('torch.'))
            records.append(item)
            return None
        return observe
    try:
        for layer,block in enumerate(adapter.layers):
            for role,module in adapter.projection_modules(layer).items():handles.append(module.register_forward_hook(hook(f'L{layer:02d}_{role}'),with_kwargs=True))
            for name,module in (('attention',block.self_attn),('ffn',block.fc2),('layer',block)):
                handles.append(module.register_forward_hook(hook(f'L{layer:02d}_{name}'),with_kwargs=True))
        yield records
        require(len(records)==32*6,'incomplete native/ES component observations')
    finally:
        for h in handles:h.remove()
        write(directory/'components.json',dict(records=records,bytes=size,instrumentation_excluded_from_measurements=True))


def compare(native_dir,es_dir,es_logits,shape,*,fixture=False):
    native_dir,es_dir=Path(native_dir),Path(es_dir)
    native=read(native_dir/'forward.json');values=unpack((native_dir/native['logits_file']).read_bytes(),'float32')
    parity=check_parity(es_logits,values,shape)
    if fixture:return dict(strict_metrics=parity,scope='TEST_FIXTURE_ONLY_CPU_CONTROL_NOT_NATIVE_PEFT',components='NOT_RUN')
    from .parity_forensics_probe import tensor_metrics
    a=read(native_dir/'components.json')['records'];b=read(es_dir/'components.json')['records']
    require([r['key'] for r in a]==[r['key'] for r in b],'native/ES component call order differs')
    comparisons=[]
    for ref,observed in zip(a,b):
        tensors=[]
        for directory,spec in ((es_dir,observed['output']),(native_dir,ref['output'])):
            path=directory/spec['file'];require(sha(path)==spec['sha256'],'triage tensor SHA mismatch')
            tensors.append(tensor_from_bytes(path.read_bytes(),spec['dtype'],spec['shape']))
        comparisons.append(dict(component=ref['key'],input_bitwise_equal=ref['input']['sha256']==observed['input']['sha256'],
            output_bitwise_equal=ref['output']['sha256']==observed['output']['sha256'],metrics=tensor_metrics(*tensors),
            interpretation='same-input component contrast' if ref['input']['sha256']==observed['input']['sha256'] else
                'changed input; downstream divergence is not independent arithmetic attribution'))
    first=next((r for r in comparisons if not r['output_bitwise_equal']),None)
    first_qkv=next((r for r in comparisons if r['component'].endswith(('_q','_k','_v')) and not r['output_bitwise_equal']),None)
    findings=[]
    for probe in native.get('probes',[]):
        key=f'L{probe["layer"]:02d}_{probe["role"]}'
        r=next((r for r in b if r['key']==key),None)
        if r is not None and probe['hidden']['sha256']==r['input']['sha256']:
            checks=probe['comparisons']
            if (checks['native_vs_gpu_reconstruction']['unequal_elements']==0 and checks['gpu_delta_vs_cpu_delta']['unequal_elements']>0 and
                probe['predicted_cpu_assembled']['sha256']==r['output']['sha256']):
                findings.append(dict(component=key,label='SAME_INPUT_FIRST_LAYER_OUTPUT_CONSISTENT_WITH_CPU_GPU_DELTA_NUMERICS',
                    gpu_reconstruction_exact=True,es_matches_predicted_cpu_assembly=True,fp16_flips=probe['fp16_output_flip_count'],
                    scope='measured first-layer component evidence; not a full-logit causal sufficiency result'))
    es_forward=read(es_dir/'forward.json')
    return dict(strict_metrics=parity,matched_model_logits_dtype=native['model_logits_dtype']==es_forward['model_logits_dtype'],
        first_divergent_component=first,first_divergent_qkv=first_qkv,components=comparisons,
        findings=findings,cause='UNRESOLVED_FULL_TRACE; measured same-input evidence is separate from downstream propagation')
