"""DIAGNOSTIC_ONLY: one bare OPT load, original FP16 Linear and temporary FP32 operands."""
import os
import math
from pathlib import Path
import time

from .communication import digest_bytes,require
from .r3a_artifacts import full_base_fingerprint,tensor_bytes,tensor_from_bytes,versions
from .r3a_runtime import load_bare,memory
from .r5_i5_evidence import seal
from .r5_i5_gpu import diagnostics
from .r5_i5r2b_capture import backend
from .r5_i5r2_tensors import Evidence,file_sha
from .r5_i5r2bs1_plan import CORE,VERSION,validate_plan,historical,fresh,layout_storage,verify_history_inputs
from .r5_perf_evidence import Writer


class Store:
    def __init__(self,root,budget=512*2**20,max_files=256):
        self.root=Path(root);self.writer=Writer(root,budget,create=True);self.max_files=max_files
    def put(self,name,value):
        require(self.writer.status()['data_files']<self.max_files,'S1 evidence file bound exhausted')
        self.writer.write(self.root/name,value)
    def tensor(self,name,data,shape,dtype,layout=None):
        require(dtype in ('float16','float32') and shape and all(type(n) is int and n>0 for n in shape) and
            len(data)==math.prod(shape)*(2 if dtype=='float16' else 4) and len(data)<=32*2**20,'S1 tensor shape/dtype/size bound differs')
        require(self.writer.status()['data_files']<self.max_files,'S1 evidence file bound exhausted')
        self.writer.bytes(self.root/name,data,exclusive=True)
        return dict(file=name,sha256=digest_bytes(data),shape=list(shape),dtype=dtype,layout=layout or {})


def layout(t):
    return dict(shape=list(t.shape),stride=list(t.stride()),storage_offset=t.storage_offset(),contiguous=t.is_contiguous(),
        dtype=str(t.dtype).removeprefix('torch.'),device=str(t.device))


def reconstruct(t,record):
    import torch
    shape=list(t.shape);stride=record['observed_stride'];offset=record['storage_offset']
    n=layout_storage(shape,stride,offset,width=t.element_size())
    storage=torch.empty(n,dtype=t.dtype,device=t.device);view=storage.as_strided(shape,stride,offset)
    view.copy_(t);require(torch.equal(view,t),'historical strided reconstruction changed values');return view


def parameters(module):
    return parameter_values(module.weight,module.bias)


def parameter_values(weight,bias):
    return {name:None if value is None else dict(shape=list(value.shape),dtype=str(value.dtype).removeprefix('torch.'),
        sha256=digest_bytes(tensor_bytes(value))) for name,value in (('weight',weight),('bias',bias))}


def apply_recorded_backend(recorded):
    """Declared replay of the historical baseline, never a pass-seeking override."""
    import torch
    require(all(k in recorded for k in CORE) and recorded['autocast_enabled'] is False,'historical core backend unavailable/unsupported')
    require(os.environ.get('CUBLAS_WORKSPACE_CONFIG')==recorded['CUBLAS_WORKSPACE_CONFIG'],'set the recorded CUBLAS_WORKSPACE_CONFIG before Python')
    torch.set_float32_matmul_precision(recorded['float32_matmul_precision'])
    torch.backends.cuda.matmul.allow_tf32=recorded['TF32_matmul']
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=recorded['FP16_reduced_precision_reduction']
    torch.backends.cudnn.allow_tf32=recorded['cudnn_TF32'];torch.backends.cudnn.benchmark=recorded['cudnn_benchmark']
    torch.backends.cudnn.deterministic=recorded['cudnn_deterministic']
    torch.use_deterministic_algorithms(recorded['deterministic_algorithms']);torch.set_num_threads(recorded['cpu_threads'])
    actual=backend();require(all(actual[k]==recorded[k] for k in CORE),'effective baseline backend differs from recorded settings')
    return actual


def measure(schedule,invoke,verify,save):
    """One invocation per scheduled operation; no simulated GPU path in the CLI."""
    rows=[]
    for step in schedule:
        before=verify(step['length']);output=invoke(step['length']);after=verify(step['length'])
        require(before==after,'input/weight/backend state changed across Linear call')
        if not step['warmup']:rows.append(dict(step,receipt=before,output=save(output,step)))
    return rows


def execute(experiment,historical_run,historical_review,base_prepared,output,source_commit):
    plan=validate_plan(experiment);require(not plan['TEST_FIXTURE_ONLY'],'real S1 GPU execution rejects CPU fixtures')
    require(source_commit==plan['source']['commit'] and os.environ.get('S1_ENABLE_GPU')=='1' and os.environ.get('SLURM_JOB_ID'),
        'manual Slurm allocation, S1_ENABLE_GPU=1 and pinned commit required')
    require(Path(historical_run).resolve()==Path(plan['historical_run_dir']) and Path(historical_review).resolve()==Path(plan['historical_review_dir']) and
        Path(base_prepared).resolve()==Path(plan['base_prepared']),'mandatory input paths differ from sealed plan')
    history=historical(historical_run,historical_review,expected_commit=plan['historical_source_commit'])
    require(history['run_seal']==plan['historical_run_seal'] and history['review_seal']==plan['historical_review_seal'], 'historical input inventories changed')
    from .r3a_artifacts import validate_prepared
    pm,_,_=validate_prepared(base_prepared)
    require(pm['identity']==plan['model_identity'] and file_sha(Path(base_prepared)/'r3a_prepare_manifest.json')==plan['base_manifest_sha256'],
        'prepared model identity changed')
    require(os.environ.get('HF_HUB_OFFLINE')==os.environ.get('TRANSFORMERS_OFFLINE')=='1','offline checkpoints required')
    require(int(os.environ.get('SLURM_CPUS_PER_TASK','0'))>=4 and int(os.environ.get('SLURM_MEM_PER_NODE','0'))>=65536,'four cores and 64 GiB Slurm allocation required')
    from importlib.util import spec_from_file_location,module_from_spec
    spec=spec_from_file_location('s1_i5_preflight',Path(__file__).resolve().parents[3]/'scripts/119_run_r5_i5.py')
    preflight=module_from_spec(spec);spec.loader.exec_module(preflight)
    require(preflight.cpu_mem_available_bytes()>=plan['bounds']['cpu_reserve_bytes'],'CPU memory reserve unavailable')
    import shutil
    require(shutil.disk_usage(Path(output).parent if Path(output).parent.exists() else Path(experiment).parent).free>2*plan['bounds']['evidence_bytes']+2**30,
        'insufficient bounded diagnostic disk headroom')
    import torch
    require(torch.cuda.is_available() and torch.cuda.device_count()==1,'one allocated CUDA-visible GPU required')
    torch.cuda.set_device('cuda:0');gpu=diagnostics(pm);require(gpu['eligible'],'allocated compatible RTX 3090 required: '+str(gpu))
    require(gpu['gpu']['free_bytes']>plan['bounds']['gpu_estimate_bytes'],'estimated GPU memory unavailable')
    initial_backend=backend();effective=apply_recorded_backend(plan['historical_backend'])
    require(all(effective[k]==plan['historical_backend'][k] for k in ('torch','transformers','cuda_compiled_version')),'historical/runtime backend versions differ')
    require(versions()==pm['identity']['versions'],'prepared/runtime software compatibility differs')
    out=fresh(output,(experiment,historical_run,historical_review,base_prepared));store=Store(out,plan['bounds']['evidence_bytes'],plan['bounds']['max_files'])
    started=time.monotonic();status=dict(status='FAILED',GPU_NOT_RUN=True);records=[];pe=Evidence(experiment);pe.verify()
    verify_history_inputs(plan,pe,history)
    try:
        model,tokenizer,meta=load_bare(pm['identity'],'cuda:0')  # Exactly one model load; never model.forward().
        from semcache.models.model_adapter import OPTModelAdapter
        from semcache.system_cost.es_base_profile import assert_no_lora
        adapter=OPTModelAdapter(model);assert_no_lora(model,adapter);fp=full_base_fingerprint(model)
        require(fp['sha256']==pm['identity']['base_weight_sha256'],'loaded base fingerprint mismatch')
        from .r5_multi_artifacts import tokenizer_identity
        require(tokenizer_identity(pm,tokenizer)==history['model']['tokenizer'], 'tokenizer differs')
        require(all(backend()[k]==effective[k] for k in CORE),'loader changed the declared backend')
        store.put('i5r2bs1_plan.json',plan);store.put('i5r2bs1_base_fingerprint.json',fp)
        weights={};inputs={};module_names={id(m):name for name,m in model.named_modules()}
        with torch.inference_mode():
            for role,module in adapter.projection_modules(0).items():
                require(type(module) is torch.nn.Linear and 'forward' not in module.__dict__ and not module._forward_hooks and not module._forward_pre_hooks and
                    module.in_features==module.out_features==2560 and module.weight.dtype==torch.float16 and
                    (module.bias is None or module.bias.dtype==torch.float16),'actual bare Layer-0 FP16 OPT Linear required')
                name=module_names[id(module)];original=parameters(module);weight=module.weight;bias=module.bias
                require(all(value is None or fp['tensors'][name+'.'+key]['sha256']==value['sha256'] for key,value in original.items()),'projection/base fingerprint mismatch')
                saved={k:None if t is None else store.tensor('weights/'+role+'_'+k+'.bin',tensor_bytes(t),list(t.shape),'float16',layout(t)) for k,t in (('weight',weight),('bias',bias))}
                weights[role]=dict(module=name,module_class=type(module).__name__,before=original,saved=saved)
                hidden=plan['roles'][role]['hidden'];long_half=tensor_from_bytes(pe.tensor('',plan['blobs'][role+'/input_long']),'float16',[1,256,hidden]).to('cuda:0')
                for precision in plan['precision_conditions']:
                    dtype=torch.float16 if precision=='NATIVE_FP16' else torch.float32
                    # Temporary FP32 selected projection only; no model parameter is cast or replaced.
                    if dtype==torch.float32:
                        require(torch.cuda.mem_get_info()[0]>2*weight.numel()*4+64*2**20,'temporary FP32 projection memory unavailable')
                    w32=None if dtype==torch.float16 else weight.detach().to(torch.float32)
                    b32=None if dtype==torch.float16 or bias is None else bias.detach().to(torch.float32)
                    active_parameters=original if dtype==torch.float16 else parameter_values(w32,b32)
                    import torch.nn.functional as F
                    call=module.forward if dtype==torch.float16 else lambda x:F.linear(x,w32,b32)
                    for control in plan['roles'][role]['layouts']:
                        long=torch.empty((1,256,hidden),dtype=dtype,device='cuda:0');long.copy_(long_half)
                        short=torch.empty((1,64,hidden),dtype=dtype,device='cuda:0');short.copy_(long[:,:64])
                        if control=='HISTORICAL_STRIDED':
                            cap=plan['roles'][role]['captures']
                            short=reconstruct(short,cap['source_projection_input']['item']);long=reconstruct(long,cap['target_projection_input']['item'])
                        xs={64:short,256:long};prefix=tensor_bytes(short);require(prefix==tensor_bytes(long[:,:64]),'input prefix bytes unequal')
                        require(prefix==tensor_bytes(tensor_from_bytes(pe.tensor('',plan['blobs'][role+'/source_projection_input']),'float16',[1,64,hidden]).to(dtype)),
                            'historical prefix values changed')
                        group=role+'/'+precision+'/'+control
                        inputs[group]={str(n):store.tensor('inputs/'+group.replace('/','_')+'_'+str(n)+'.bin',tensor_bytes(t),list(t.shape),str(dtype).removeprefix('torch.'),layout(t)) for n,t in xs.items()}
                        fixed={n:dict(input_sha256=digest_bytes(tensor_bytes(t)),input_prefix_sha256=prefix and digest_bytes(prefix),
                            weight=original,active_parameters=active_parameters,gpu_uuid=gpu['execution_gpu_uuid'],backend={k:effective[k] for k in CORE}) for n,t in xs.items()}
                        def verify(n):
                            require(time.monotonic()-started<plan['bounds']['runtime_seconds'],'S1 diagnostic runtime bound exceeded')
                            current=memory('cuda:0');require(current['uuid']==gpu['execution_gpu_uuid'],'execution GPU changed')
                            receipt=dict(input_sha256=digest_bytes(tensor_bytes(xs[n])),input_prefix_sha256=digest_bytes(tensor_bytes(xs[n][:,:64])),
                                weight=parameters(module),active_parameters=parameters(module) if dtype==torch.float16 else parameter_values(w32,b32),
                                gpu_uuid=current['uuid'],backend={k:backend()[k] for k in CORE})
                            require(receipt==fixed[n],'input/weight/backend pre-call identity differs');return receipt
                        def invoke(n):
                            torch.cuda.synchronize();status.update(GPU_NOT_RUN=False,gpu_projection_attempted=True)
                            result=call(xs[n]);torch.cuda.synchronize()
                            require(result.dtype==dtype and list(result.shape)==[1,n,hidden],'native/FP32 Linear output contract differs');return result
                        def save(t,step):
                            tag=group.replace('/','_')+'_'+str(step['length'])+'_'+str(step['round'])
                            return store.tensor('outputs/'+tag+'.bin',tensor_bytes(t),list(t.shape),str(dtype).removeprefix('torch.'),layout(t))
                        rows=measure(plan['schedule'],invoke,verify,save)
                        records.extend(dict(row,role=role,precision=precision,control=control) for row in rows)
                        del xs,short,long
                    weights[role][precision+'_temporary_parameter_hashes']=active_parameters
                    del w32,b32,call
                weights[role]['after']=parameters(module);require(weights[role]['after']==original,'native projection parameters mutated')
                del long_half
        after=full_base_fingerprint(model);require(after==fp,'full model state mutated')
        require(validate_plan(experiment)==plan,'pinned source/plan changed during diagnostic')
        status=dict(status='COMPLETE',GPU_NOT_RUN=False,DIAGNOSTIC_ONLY=True,experiment_id=plan['experiment_id'],source_commit=source_commit,
            model_load_count=1,production_forward_calls=0,LoRA_calls=0,TCP_frames=0,cache_reads=0,generation_tokens=0,base_fingerprint_before=fp['sha256'],base_fingerprint_after=after['sha256'])
        store.put('i5r2bs1_records.json',records);store.put('i5r2bs1_weights.json',weights);store.put('i5r2bs1_inputs.json',inputs)
        store.put('i5r2bs1_backend.json',dict(initial=initial_backend,effective=effective,after=backend(),policy=plan['backend_policy'],gpu=gpu,
            compute_capability=list(torch.cuda.get_device_capability(0)),cuBLAS_version='UNKNOWN: public API does not expose the loaded cuBLAS version',model_loader=meta,
            historical_gpu=plan['historical_gpu'],original_layout_reproduction='Recorded strides/offsets only; allocator alignment and scheduling are unobserved'))
    except BaseException as exc:
        status['reason']=type(exc).__name__+': '+str(exc);raise
    finally:
        store.put('i5r2bs1_status.json',status);store.writer.reconcile();seal(out)
    return status
