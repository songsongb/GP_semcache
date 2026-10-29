"""Bare OPT ES execution and one real hidden/bundled-delta exchange per layer."""
from contextlib import contextmanager
import os
from pathlib import Path
import time

from .communication import Channel, connect, digest_bytes, header, require
from .r3a_artifacts import append, check_parity, flags, full_base_fingerprint, pack, process_environment, read, tensor_bytes, tensor_from_bytes, unpack, validate_prepared, versions, write


class ProjectionSession:
    """Coordinate actual q/k/v calls without relying on a particular call order."""
    def __init__(self, channel, identity, request_id, *, torch_backend):
        self.channel,self.identity,self.request_id = channel,identity,request_id
        self.torch_backend = torch_backend
        self.next_layer,self.active,self.trace = 0,None,[]

    def correction(self, layer, role, hidden):
        identity = self.identity
        require(layer == self.next_layer and role in ('q','k','v'), 'out-of-order projection layer')
        shape = [1,identity['length'],identity['hidden_size']]
        if self.active is None:
            start = time.monotonic_ns()
            if self.torch_backend:
                import torch
                require(list(hidden.shape) == shape and hidden.dtype == getattr(torch,identity['hidden_dtype']) and hidden.is_cuda,
                        'actual OPT hidden dtype/shape/device mismatch')
                seal = hidden.detach().clone()
                payload = tensor_bytes(hidden)
            else:
                seal = list(hidden); payload = pack(hidden,identity['hidden_dtype'])
            copied = time.monotonic_ns()
            sent = self.channel.send(header('HIDDEN',self.request_id,layer,identity['hidden_dtype'],shape),payload)
            reply,data,received = self.channel.recv(kind='DELTAS',request_id=self.request_id,layer_id=layer,
                dtype=identity['delta_dtype'],shape=[3]+shape)
            require(reply['metadata']['hidden_payload_sha256'] == digest_bytes(payload) and
                    reply['metadata']['user_device_cuda_used'] is False, 'delta/hidden provenance mismatch')
            decode_start = time.monotonic_ns()
            deltas = tensor_from_bytes(data,identity['delta_dtype'],[3]+shape) if self.torch_backend else unpack(data,identity['delta_dtype'])
            done = time.monotonic_ns()
            self.active = dict(seal=seal,deltas=deltas,seen=set(),record=dict(layer_id=layer,request_id=self.request_id,
                first_projection=role, hidden_payload_sha256=digest_bytes(payload), delta_payload_sha256=reply['payload_sha256'],
                es_hidden_copy_cast_ms=(copied-start)/1e6, es_delta_deserialization_ms=(done-decode_start)/1e6,
                exchange_host_wall_ms=(decode_start-copied)/1e6, exchange_is_pure_rtt=False,
                sent_frame=sent, received_frame=received, ud=reply['metadata'], qkv_hidden_input_equal={},
                es_delta_cpu_to_gpu_ms={},es_base_add_self_time_status='UNMEASURED_NESTED_IN_REQUEST_WALL'))
        active = self.active
        require(role not in active['seen'], 'duplicate projection call')
        if self.torch_backend:
            import torch
            equal = bool(torch.equal(hidden,active['seal']))
        else:
            equal = hidden == active['seal']
        require(equal, 'ARCHITECTURE_FAIL: Q/K/V hidden inputs differ or were mutated')
        active['seen'].add(role)
        active['record']['qkv_hidden_input_equal'][role] = equal
        if self.torch_backend:
            copy_start = time.monotonic_ns()
            result = active['deltas']['qkv'.index(role)].to(hidden.device)
            # Complete the copy so this host span is explicitly synchronized.
            torch.cuda.synchronize(hidden.device)
            active['record']['es_delta_cpu_to_gpu_ms'][role] = (time.monotonic_ns()-copy_start)/1e6
        else:
            n = identity['length']*identity['hidden_size']; idx = 'qkv'.index(role)
            result = active['deltas'][idx*n:(idx+1)*n]
        if len(active['seen']) == 3:
            self.trace.append(active['record']); self.active = None; self.next_layer += 1
        return result

    def complete(self):
        require(self.active is None and self.next_layer == self.identity['layers'] and len(self.trace) == self.identity['layers'],
                'not all real forward layers/projections were observed')
        require(sum(len(r['qkv_hidden_input_equal']) for r in self.trace) == 3*self.identity['layers'], 'incomplete QKV outputs')


@contextmanager
def transported_projection_path(adapter, session):
    import torch
    from semcache.system_cost.es_base_profile import assert_no_lora
    assert_no_lora(adapter.model,adapter)
    originals = []
    def replacement(layer,role,native):
        def forward(hidden,*args,**kwargs):
            require(not args and not kwargs, 'ARCHITECTURE_FAIL: unsupported projection arguments')
            delta = session.correction(layer,role,hidden)
            # Preserve native Linear bias and the actual PEFT promotion/add/cast order.
            base = native(hidden)
            return (base+delta).to(base.dtype)
        return forward
    try:
        for layer in range(len(adapter.layers)):
            for role,module in adapter.projection_modules(layer).items():
                require(type(module) is torch.nn.Linear and 'forward' not in module.__dict__ and
                        not module._forward_hooks and not module._forward_pre_hooks, 'bare uninstrumented Linear required')
                originals.append(module)
                module.forward = replacement(layer,role,module.forward)
        yield session
        session.complete()
    finally:
        for module in originals: del module.forward


def execute(prepared, output, session_id, device, warmup, measured):
    from .r3a_runtime import load_bare, memory, runtime_setup
    output = Path(output); pm,ref,am = validate_prepared(prepared)
    fixture = pm.get('TEST_FIXTURE_ONLY') is True; identity = pm['identity']
    start = time.monotonic_ns(); model=adapter=tokenizer=None
    if fixture:
        from .r3a_tiny import forward as tiny_forward
        from .communication import digest
        base = read(Path(prepared)/'r3a_tiny_base.json')
        require(digest(base) == identity['base_weight_sha256'], 'tiny base fingerprint mismatch')
    else:
        runtime_setup(device)
        model,tokenizer,metadata = load_bare(identity,device)
        from semcache.models.model_adapter import OPTModelAdapter
        from semcache.system_cost.es_base_profile import assert_no_lora
        adapter=OPTModelAdapter(model); assert_no_lora(model,adapter)
        require(full_base_fingerprint(model)['sha256'] == identity['base_weight_sha256'], 'INVALID: base weights differ from reference')
        require(versions() == identity['versions'], 'runtime version mismatch with reference')
    loaded=time.monotonic_ns()
    logits_ref = unpack((Path(prepared)/ref['logits_file']).read_bytes(),ref['logits_dtype'])
    ready=read(output/'r3a_ud_ready.json'); channel=None; rows=[]; parities=[]; traces=[]
    result=dict(execution_status='RUNNING',pid=os.getpid(),parent_pid=os.getppid(),identity_sha256=pm['identity_sha256'],
        base_only=True,adapter_parameters_on_es=0,model_load_and_identity_ms=(loaded-start)/1e6,
        versions=versions(),device=device,environment=process_environment(),**flags(fixture=fixture))
    if not fixture:
        result['gpu']=memory(device)
    try:
        handshake_start=time.monotonic_ns()
        result['reference_artifact_load_ms']=(handshake_start-loaded)/1e6
        channel=Channel(connect(ready['port']),session_id,am['matrix_sha256'],
                        trace=lambda r:append(output/'r3a_wire_trace.jsonl',r))
        channel.send(header('HELLO','handshake',-1,pid=os.getpid(),identity_sha256=pm['identity_sha256']))
        response,_,_=channel.recv(kind='READY',request_id='handshake',layer_id=-1)
        require(response['metadata']['pid'] == ready['pid'] != os.getpid() and
                response['metadata']['identity_sha256'] == pm['identity_sha256'] and
                response['metadata']['user_device_cuda_used'] is False, 'process/handshake identity mismatch')
        result['actual_network_socket_used']=True; result['ud_pid']=ready['pid']
        result['socket_handshake_ms']=(time.monotonic_ns()-handshake_start)/1e6
        input_preparation_start=time.monotonic_ns()
        if not fixture:
            import torch
            inputs=dict(input_ids=torch.tensor([identity['token_ids']],device=device),use_cache=False)
        result['input_tensor_preparation_ms']=(time.monotonic_ns()-input_preparation_start)/1e6
        for phase,count in (('warmup',warmup),('measured',measured)):
            for repeat in range(count):
                request_id=f'{phase}_{repeat:06d}'
                if not fixture:
                    torch.cuda.synchronize(device)
                    begin_event,end_event=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                begin=time.monotonic_ns()
                projection=ProjectionSession(channel,identity,request_id,torch_backend=not fixture)
                if fixture:
                    logits=tiny_forward(identity['token_ids'],base,lambda l,r,x:projection.correction(l,r,x))
                    projection.complete(); cuda_ms=None
                else:
                    result['gpu_inference_performed']=True
                    with torch.inference_mode(),transported_projection_path(adapter,projection):
                        begin_event.record()
                        outputs=model(**inputs,output_attentions=True)
                        end_event.record()
                        torch.cuda.synchronize(device)
                    cuda_ms=begin_event.elapsed_time(end_event)
                end=time.monotonic_ns()
                if not fixture:
                    require(str(outputs.logits.dtype)==ref['native_logits_dtype'],'INVALID: native reference/distributed logit dtype mismatch')
                    result['native_logits_dtype']=str(outputs.logits.dtype)
                    logits=unpack(tensor_bytes(outputs.logits.float()),'float32'); del outputs
                parity=check_parity(logits,logits_ref,ref['logits_shape'])
                parity.update(request_id=request_id,phase=phase,repeat=repeat)
                parities.append(parity)
                layer_records=projection.trace
                frames=[f for r in layer_records for f in (r['sent_frame'],r['received_frame'])]
                row=dict(request_id=request_id,phase=phase,repeat=repeat,es_pid=os.getpid(),ud_pid=ready['pid'],
                    es_start_monotonic_ns=begin,es_end_monotonic_ns=end,es_synchronized_wall_ms=(end-begin)/1e6,
                    es_cuda_elapsed_ms=cuda_ms,ud_cpu_lora_ms=sum(r['ud']['cpu_lora_ms'] for r in layer_records),
                    es_to_ud_frames=sum(f['operation']=='send' for f in frames),ud_to_es_frames=sum(f['operation']=='recv' for f in frames),
                    es_to_ud_payload_bytes=sum(f['header']['payload_bytes'] for f in frames if f['operation']=='send'),
                    ud_to_es_payload_bytes=sum(f['header']['payload_bytes'] for f in frames if f['operation']=='recv'),
                    es_to_ud_wire_bytes=sum(f['wire_bytes'] for f in frames if f['operation']=='send'),
                    ud_to_es_wire_bytes=sum(f['wire_bytes'] for f in frames if f['operation']=='recv'),
                    es_copy_cast_ms=sum(r['es_hidden_copy_cast_ms']+sum(r['es_delta_cpu_to_gpu_ms'].values()) for r in layer_records),
                    es_send_socket_ms=sum(f['socket_call_wall_ms'] for f in frames if f['operation']=='send'),
                    es_recv_socket_ms=sum(f['socket_call_wall_ms'] for f in frames if f['operation']=='recv'),
                    es_framing_serialization_ms=sum(f.get('framing_serialization_ms',0)+f.get('framing_deserialization_ms',0) for f in frames),
                    ud_input_deserialization_ms=sum(r['ud']['input_deserialization_ms'] for r in layer_records),
                    ud_output_serialization_ms=sum(r['ud']['output_serialization_ms'] for r in layer_records),
                    layers_observed=len(layer_records),qkv_outputs_observed=3*len(layer_records),
                    parity_passed=parity['passed'],identity_sha256=pm['identity_sha256'],TEST_FIXTURE_ONLY=fixture)
                rows.append(row);traces.append(dict(request_id=request_id,layers=layer_records))
                append(output/'r3a_iterations.jsonl',dict(row=row,parity=parity,layers=layer_records))
                if not parity['passed']: raise RuntimeError('PARITY_FAIL: predeclared native logit tolerance failed; stop before further benchmarking')
        channel.send(header('STOP','shutdown',-1))
        channel.recv(kind='BYE',request_id='shutdown',layer_id=-1)
        result.update(execution_status='COMPLETE',rows=rows,parities=parities,per_layer_trace=traces)
    except BaseException as exc:
        result.update(execution_status='FAILED',error=f'{type(exc).__name__}: {exc}',rows=rows,parities=parities,per_layer_trace=traces)
        raise
    finally:
        if channel is not None: channel.close()
        if not fixture: result['gpu']=memory(device)
        result['closed_monotonic_ns']=time.monotonic_ns()
        write(output/'r3a_es_result.json',result)
