"""Explicitly enabled causal interventions; never a physical latency endpoint.

Only one GPU model per isolated child. Imports stay model-free until a worker
is explicitly called. Production source_identity() excludes this filename.
"""
from contextlib import contextmanager
import os
from pathlib import Path
import platform
import time
import uuid

from .communication import Channel, connect, digest, digest_bytes, header, require
from .r3a_artifacts import (append, full_base_fingerprint, load_matrices, read, sha, tensor_bytes,
    tensor_from_bytes, unpack, write)
from .parity_causal_evidence import SERIALIZATION, validate_identity
from .parity_forensics_probe import (load_native, logit_metrics, passive_hashes, same_hidden_probes,
    setup, tensor_info, tensor_metrics)


def exact_affinity(expected):
    affinity = expected['cpu_affinity']
    require(isinstance(affinity, list) and affinity and len(set(affinity)) == len(affinity) and
            all(type(v) is int and v >= 0 for v in affinity) and
            set(affinity).issubset(os.sched_getaffinity(0)),
            'allocation must allow original UD CPU affinity; request matching CPU resources, do not bypass')
    os.sched_setaffinity(0, affinity)
    require(sorted(os.sched_getaffinity(0)) == affinity, 'original CPU affinity could not be restored')


def cpu_runtime():
    import torch
    model = next((line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                  if line.startswith('model name')), platform.machine())
    return dict(cpu_model=model, machine=platform.machine(), torch_config_sha256=digest_bytes(torch.__config__.show().encode()),
        torch_threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads(),
        affinity=sorted(os.sched_getaffinity(0)), mkldnn_enabled=torch.backends.mkldnn.enabled,
        environment={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS',
            'MKL_CBWR','OMP_DYNAMIC','MKL_DYNAMIC','ONEDNN_MAX_CPU_ISA')})


def start(plan):
    require(plan.get('enable_gpu_probes') is True, 'explicit GPU intervention opt-in required')
    from .r3a_artifacts import ROOT
    for name,expected in plan['diagnostic_source_hashes'].items():
        path=(ROOT/name).resolve()
        require(path.is_relative_to(ROOT) and sha(path)==expected,'causal diagnostic source changed during execution')
    exact_affinity(plan['comparison_identity'])
    try:
        pm, ref, am, metadata = setup(plan)  # unchanged immutable-prepare/version/UUID gates
    except ValueError as exc:
        if 'matching original GPU UUID' in str(exc):
            from .r3a_runtime import memory
            raise ValueError('GPU UUID mismatch: original='+str(plan['comparison_identity']['gpu_uuid'])+
                ', allocated='+memory(plan['device'])['uuid']+
                '; request the original physical GPU on moana-y6 through Slurm; do not rewrite prepare or bypass the gate') from exc
        raise
    validate_identity(plan['comparison_identity'], plan['comparison_identity'])
    require(pm['identity'] == plan['comparison_identity']['identity'] and
            sha(Path(plan['prepared'])/'r3a_prepare_manifest.json') == plan['comparison_identity']['prepared_manifest_sha256'],
            'child prepared identity changed')
    require(metadata['effective_affinity'] == plan['comparison_identity']['cpu_affinity'], 'setup affinity differs')
    metadata['cpu_runtime'] = cpu_runtime()
    metadata['monotonic_start_ns'] = time.monotonic_ns()
    return pm, ref, am, metadata


def forward(model, plan, output, label, ref):
    import torch
    write(output/'gpu_execution.json', dict(stage=label, gpu_inference_performed=True, pid=os.getpid()))
    with torch.inference_mode():
        result = model(input_ids=torch.tensor([plan['comparison_identity']['identity']['token_ids']],device=plan['device']),
                       use_cache=False, output_attentions=True)
        torch.cuda.synchronize(plan['device'])
    require(list(result.logits.shape) == ref['logits_shape'] and str(result.logits.dtype) == ref['native_logits_dtype'],
            'native all-token endpoint dtype/shape mismatch')
    raw = tensor_bytes(result.logits.float())
    descriptor = dict(sha256=digest_bytes(raw), dtype='float32', shape=list(result.logits.shape),
        bytes=len(raw), serialization=SERIALIZATION)
    if plan.get('capture_logits'):
        require(len(raw) <= 16*1024*1024, 'bounded diagnostic logit capture exceeds 16 MiB')
        target = output/(label+'_logits.f32.bin')
        require(not target.exists(), 'refusing existing diagnostic tensor capture')
        target.write_bytes(raw)
    values = unpack(raw, 'float32')
    original = unpack((Path(plan['prepared'])/ref['logits_file']).read_bytes(), ref['logits_dtype'])
    return descriptor, logit_metrics(values, original, ref['logits_shape'])


def native(plan, output):
    pm, ref, am, metadata = start(plan)
    model, adapter, matrices = load_native(plan, pm, am)
    controls = []
    desc, metrics = forward(model, plan, output, 'native_no_hooks', ref)
    controls.append(dict(label='native_no_hooks', logits=desc, vs_original=metrics,
                         projection_hashes={}, original_projection_mismatches=[]))
    with passive_hashes(adapter) as (hashes, order):
        desc, metrics = forward(model, plan, output, 'native_passive_hashes', ref)
    mismatches = [dict(projection=k,field=f,observed=v[f],original=ref['projections'][k][f])
        for k,v in hashes.items() for f in ('hidden_sha256','native_output_sha256') if v[f] != ref['projections'][k][f]]
    controls.append(dict(label='native_passive_hashes',logits=desc,vs_original=metrics,
                         projection_hashes=hashes,original_projection_mismatches=mismatches,role_order=order))
    # These probes do not replace native outputs. They also quantify all 96
    # same-native-hidden CPU/GPU deltas, rather than generalizing six samples.
    with same_hidden_probes(adapter, matrices, pm['identity'], set(range(32)), output,
                           fp64_control=plan.get('cpu_fp64_control',False)) as (points, order):
        desc, metrics = forward(model, plan, output, 'native_gpu_delta_control', ref)
    hashes = {f"{p['layer']}.{p['role']}":dict(hidden_sha256=p['hidden']['sha256'],
        native_output_sha256=p['native_output']['sha256']) for p in points}
    mismatches = [dict(projection=k,field=f,observed=v[f],original=ref['projections'][k][f])
        for k,v in hashes.items() for f in ('hidden_sha256','native_output_sha256') if v[f] != ref['projections'][k][f]]
    controls.append(dict(label='native_gpu_delta_control',logits=desc,vs_original=metrics,
        projection_hashes=hashes,original_projection_mismatches=mismatches,role_order=order))
    require(full_base_fingerprint(model.get_base_model(),peft_wrapped=True)['sha256'] == pm['identity']['base_weight_sha256'],
            'native base weights changed')
    records = {(r['layer'],r['role'],r['matrix']):r for r in am['matrices']}
    for layer in range(32):
        for role,module in adapter.projection_modules(layer).items():
            for kind,part in (('A',module.lora_A['user_a']),('B',module.lora_B['user_a'])):
                require(digest_bytes(tensor_bytes(part.weight)) == records[layer,role,kind]['sha256'], 'native A/B changed')
    return dict(execution_status='COMPLETE',process_metadata=metadata,comparison_identity=plan['comparison_identity'],
        controls=controls, all_gpu_reconstructions_exact=all(p['gpu_reconstructed']['sha256'] == p['native_output']['sha256']
            and p['comparisons']['native_vs_observed_parts_sum']['bitwise_unequal_elements'] == 0 for p in points),
        gpu_reconstruction_count=len(points), same_hidden_projections=points,
        gpu_delta_control_scope='diagnostic native PEFT/GPU reconstruction only; not a CPU User Device implementation',
        weights_verified_after=True)


@contextmanager
def intervention_path(adapter, matrices, identity, output, *, session=None, capture=False, local_reference=None):
    """All corrections from CPU; real TCP reply drives output when session exists."""
    import torch
    from .user_device import cpu_deltas
    originals = []; handles = []; records = []; bundle = {}; pending = {}; order = []; total_capture = 0
    local_map = {f"{p['layer']}.{p['role']}":p for p in (local_reference or {}).get('projections',[])}

    def replacement(layer, role, original):
        def routed(x, *args, **kwargs):
            require(not args and not kwargs and x.dtype == torch.float16 and list(x.shape) == [1,32,2560] and
                    x.device.type == 'cuda', 'unsupported intervention projection input')
            remote = session.correction(layer,role,x) if session is not None else None
            if not bundle:
                cpu_x = tensor_from_bytes(tensor_bytes(x), 'float16', list(x.shape))
                precision = {}
                deltas = cpu_deltas(cpu_x,matrices,layer,identity['hidden_size'],torch_backend=True,precision_out=precision)
                require(deltas.dtype == torch.float32 and deltas.device.type == 'cpu', 'oracle delta precision changed')
                bundle.update(layer=layer, hidden=x.detach().clone(), deltas=deltas, roles=set(), precision=precision)
            require(bundle['layer'] == layer and role not in bundle['roles'] and torch.equal(bundle['hidden'],x),
                    'stale/duplicate or mismatched QKV hidden in oracle')
            cpu_delta = bundle['deltas']['qkv'.index(role)]
            delta = cpu_delta.to(x.device) if session is None else remote
            torch.cuda.synchronize(x.device)  # same synchronized transfer boundary as production correction()
            require(delta.dtype == torch.float32 and delta.device == x.device, 'actual correction dtype/device changed')
            base = original(x)
            require(base.dtype == torch.float16, 'base projection dtype changed')
            assembled = (base+delta).to(base.dtype)
            predicted = (base.detach().clone()+delta.detach().clone()).to(base.dtype)
            predicted_cpu = (base.detach().clone()+cpu_delta.to(base.device)).to(base.dtype)
            pending[layer,role] = (x,base,cpu_delta,delta,predicted,predicted_cpu,bundle['precision'][role])
            bundle['roles'].add(role)
            if bundle['roles'] == set('qkv'): bundle.clear()
            return assembled
        return routed

    def observer(layer,role):
        def observe(module,args,out):
            nonlocal total_capture
            x,base,cpu_delta,delta,predicted,predicted_cpu,precision = pending.pop((layer,role))
            original_ref = identity['_reference_projections'][f'{layer}.{role}']
            p = dict(layer=layer,role=role,call_index=len(order),hidden=tensor_info(x),base=tensor_info(base),
                local_cpu_delta=tensor_info(cpu_delta),received_delta=tensor_info(delta),assembled=tensor_info(predicted),
                observed_output=tensor_info(out), observed_output_matches_assembly=tensor_bytes(out) == tensor_bytes(predicted),
                assembly_comparison=tensor_metrics(out,predicted),wire_matches_local_cpu=tensor_bytes(delta) == tensor_bytes(cpu_delta),
                assembly_vs_local_cpu=tensor_metrics(out,predicted_cpu),
                cpu_verification_comparison=tensor_metrics(delta,cpu_delta), cpu_precision=precision,
                correction_source='LOCAL_CPU_NO_SOCKET' if session is None else 'PHYSICAL_CPU_UD_TCP',
                hidden_matches_original_native=digest_bytes(tensor_bytes(x)) == original_ref['hidden_sha256'],
                output_matches_original_native=digest_bytes(tensor_bytes(out)) == original_ref['native_output_sha256'])
            tensors=dict(hidden=x,base=base,cpu_delta=cpu_delta,qkv=out)
            if capture:
                (output/'comparison_tensors').mkdir(exist_ok=True);p['captures']={}
                for name,tensor in tensors.items():
                    raw=tensor_bytes(tensor);total_capture+=len(raw)
                    require(total_capture <= 128*1024*1024, 'bounded comparison capture exceeds 128 MiB')
                    target=output/'comparison_tensors'/f'L{layer}_{role}_{name}.bin'
                    require(not target.exists(),'refusing existing tensor capture');target.write_bytes(raw)
                    p['captures'][name]=dict(file=str(target.relative_to(output)),sha256=digest_bytes(raw),
                        dtype=str(tensor.dtype).removeprefix('torch.'),shape=list(tensor.shape))
            reference = local_map.get(f'{layer}.{role}')
            if reference and reference.get('captures'):
                same_hidden = p['hidden']['sha256'] == reference['hidden']['sha256']
                p['local_oracle_numeric_comparisons'] = dict(same_hidden=same_hidden,metrics={})
                for name,tensor in tensors.items():
                    spec=reference['captures'][name];path=Path(local_reference['_directory'])/spec['file']
                    require(path.resolve().is_relative_to(Path(local_reference['_directory']).resolve()) and
                            sha(path)==spec['sha256'], 'local oracle capture path/SHA mismatch')
                    previous=tensor_from_bytes(path.read_bytes(),spec['dtype'],spec['shape'])
                    p['local_oracle_numeric_comparisons']['metrics'][name]=tensor_metrics(tensor,previous)
                p['local_oracle_numeric_comparisons']['interpretation'] = ('identical-input arithmetic contrast' if same_hidden
                    else 'downstream different inputs; descriptive magnitude only, not matched-input attribution')
            order.append(dict(layer=layer,role=role));records.append(p);append(output/'causal_qkv_trace.jsonl',p)
            return None
        return observe
    try:
        for layer in range(32):
            for role,module in adapter.projection_modules(layer).items():
                require(type(module) is torch.nn.Linear and 'forward' not in module.__dict__ and
                        not module._forward_hooks and not module._forward_pre_hooks, 'unmodified bare Linear required')
                originals.append(module);module.forward=replacement(layer,role,module.forward)
                handles.append(module.register_forward_hook(observer(layer,role)))
        yield records,order
        require(len(records)==96 and not bundle and not pending,'incomplete 32-layer oracle')
        if session is not None:session.complete()
    finally:
        for handle in handles:handle.remove()
        for module in originals:del module.forward
        bundle.clear();pending.clear()


def bare(plan,output,*,physical=False):
    pm,ref,am,metadata=start(plan);identity=pm['identity']
    from .r3a_runtime import load_bare
    from .edge_server import ProjectionSession
    from semcache.models.model_adapter import OPTModelAdapter
    from semcache.system_cost.es_base_profile import assert_no_lora
    model,tokenizer,meta=load_bare(identity,plan['device']);adapter=OPTModelAdapter(model);assert_no_lora(model,adapter)
    require(full_base_fingerprint(model)['sha256']==identity['base_weight_sha256'],'bare base fingerprint mismatch')
    matrices=load_matrices(plan['prepared'],am,torch_backend=True)
    require(all(t.device.type=='cpu' for t in matrices.values()),'local diagnostic matrices must remain CPU-only')
    result={};channel=None;ud=None;session=None
    try:
        if physical:
            # Same OS process group as ES: parent timeout kills both. UD imports
            # original serve(), not model loading or PEFT, and hides CUDA.
            from .parity_causal_process import launch
            from .collaborative_inference import wait_ready
            session_id=uuid.uuid4().hex;ud_plan=dict(plan,session_id=session_id,output=str(output))
            write(output/'ud_plan.json',ud_plan)
            ud=launch('ud',output/'ud_plan.json',output,new_session=False)
            ready=wait_ready(ud[0],output/'r3a_ud_ready.json')
            require(ready['pid']==ud[0].pid,'UD PID mismatch')
            channel=Channel(connect(ready['port']),session_id,am['matrix_sha256'],
                trace=lambda r:append(output/'diagnostic_es_wire.jsonl',r))
            channel.send(header('HELLO','handshake',-1,pid=os.getpid(),identity_sha256=pm['identity_sha256']))
            reply,_,_=channel.recv(kind='READY',request_id='handshake',layer_id=-1)
            require(reply['metadata']['pid']==ready['pid'] and reply['metadata']['identity_sha256']==pm['identity_sha256'] and
                    reply['metadata']['user_device_cuda_used'] is False,'UD handshake identity/placement mismatch')
            session=ProjectionSession(channel,identity,'causal_diagnostic_000000',torch_backend=True)
        local_reference=read(plan['local_report']) if physical else None
        if local_reference:local_reference['_directory']=str(Path(plan['local_report']).parent)
        observed_identity=dict(identity,_reference_projections=ref['projections'])
        import torch
        with torch.inference_mode(),intervention_path(adapter,matrices,observed_identity,output,session=session,
                capture=plan.get('capture_comparison_tensors',False) and not physical,
                local_reference=local_reference) as (records,order):
            desc,metrics=forward(model,plan,output,'tcp_replay' if physical else 'local_cpu_delta_oracle',ref)
        assert_no_lora(model,adapter)
        require(full_base_fingerprint(model)['sha256']==identity['base_weight_sha256'],'bare weights changed during intervention')
        # Re-read the immutable binary and compare actual CPU matrix bytes too.
        entries={(r['layer'],r['role'],r['matrix']):r for r in am['matrices']}
        require(all(digest_bytes(tensor_bytes(t))==entries[k]['sha256'] for k,t in matrices.items()),'CPU A/B mutated')
        result=dict(execution_status='COMPLETE',comparison_identity=plan['comparison_identity'],process_metadata=metadata,
            cpu_runtime=cpu_runtime(),logits=desc,vs_original_logits=metrics,projections=records,role_order=order,
            expected_layer_count=32,observed_qkv_outputs=len(records),gpu_lora_on_es=False,weights_verified_after=True,
            actual_network_socket_used=physical,actual_remote_network_measured=False,distinct_ud_process_used=physical,
            cpu_delta_is_forward_source=True,cache_reuse_enabled=False,performance_timing_eligible=False,
            placement_scope='diagnostic local CPU oracle, not production physical UD' if not physical else 'diagnostic real CPU UD/TCP replay')
        if physical:
            session.complete();result['layers']=session.trace
            channel.send(header('STOP','shutdown',-1));channel.recv(kind='BYE',request_id='shutdown',layer_id=-1)
            ud[0].wait(timeout=10.)
            require(ud[0].returncode==0,'causal replay UD failed')
            report=read(output/'r3a_ud_result.json');runtime=read(output/'causal_ud_metadata.json')
            require(report['execution_status']=='COMPLETE' and runtime['cuda_initialized'] is False and
                runtime['pid']==report['pid']==ud[0].pid and runtime['cpu_runtime']==result['cpu_runtime'],
                'physical UD CPU/runtime isolation mismatch')
            result['ud_process_metadata']=dict(pid=report['pid'],cpu=report['cpu'],environment=report['environment'],
                user_device_cuda_used=False,cpu_runtime=runtime['cpu_runtime'])
            result['wire_accounting']=dict(hidden_frames=len(session.trace),delta_frames=len(session.trace),
                hidden_payload_bytes=sum(t['sent_frame']['header']['payload_bytes'] for t in session.trace),
                delta_payload_bytes=sum(t['received_frame']['header']['payload_bytes'] for t in session.trace),
                application_wire_bytes=sum(t[f]['wire_bytes'] for t in session.trace for f in ('sent_frame','received_frame')))
        return result
    finally:
        if channel is not None:channel.close()
        if ud is not None:
            from .parity_causal_process import finish
            result['ud_cleanup']=finish(ud,group=False)
        write(output/'bare_partial_result.json',result)
