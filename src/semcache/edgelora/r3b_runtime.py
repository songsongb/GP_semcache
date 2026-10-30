"""Physical R3-B ES, staged gates and bounded application measurements.

Bare GPU OPT + unchanged CPU UD serve() + unchanged ProjectionSession/Channel.
"""
from contextlib import contextmanager
import os
from pathlib import Path
import time
import uuid

from .communication import Channel,connect,digest,digest_bytes,header,require
from .r3a_artifacts import (append,check_parity,full_base_fingerprint,pack,process_environment,read,
    sha,tensor_bytes,unpack,validate_prepared,versions,write)
from .r3b_analysis import MODES,flags,summarize,validate_row
from .r3b_cache import ExactRawCache,invalid_guard_checks,raw,request_context,spec


def total_projection(entry,layer,role,miss):
    """This is the actual shared HIT branch: MISS callback is never invoked."""
    return entry.tensors[layer]['qkv'.index(role)].clone() if entry is not None else miss()


@contextmanager
def physical_path(adapter,session,entry,captures,records):
    """Full HIT avoids the native Linear AND correction callback entirely."""
    import torch
    originals=[];seen=set()
    def route(layer,role,native):
        def forward(x,*args,**kwargs):
            key=(layer,role);require(key not in seen and not args and not kwargs,'duplicate/unsupported QKV step')
            def miss():
                delta=session.correction(layer,role,x)
                base=native(x);return (base+delta).to(base.dtype)
            # Never expose writable cache storage to later attention operations.
            output=total_projection(entry,layer,role,miss)
            # Owned passive captures avoid later in-place attention scaling.
            captures.setdefault(layer,{})[role]=output.detach().clone()
            records.append(dict(layer=layer,role=role,reused=entry is not None))
            seen.add(key);return output
        return forward
    try:
        for layer in range(len(adapter.layers)):
            for role,module in adapter.projection_modules(layer).items():
                require(type(module) is torch.nn.Linear and 'forward' not in module.__dict__ and
                        not module._forward_hooks and not module._forward_pre_hooks,'bare uninstrumented projection required')
                originals.append(module);module.forward=route(layer,role,module.forward)
        yield
        require(len(seen)==3*len(adapter.layers),'incomplete actual QKV forward')
        if entry is None:session.complete()
    finally:
        for module in originals:del module.forward


def wire_totals(records):
    data=[r for r in records if r['complete'] and r['header']['kind'] in ('HIDDEN','DELTAS')]
    require(len(data)==len(records),'incomplete or unexpected frame inside request')
    return dict(hidden_frames=sum(r['header']['kind']=='HIDDEN' for r in data),
        delta_frames=sum(r['header']['kind']=='DELTAS' for r in data),
        hidden_payload_bytes=sum(r['header']['payload_bytes'] for r in data if r['header']['kind']=='HIDDEN'),
        delta_payload_bytes=sum(r['header']['payload_bytes'] for r in data if r['header']['kind']=='DELTAS'),
        application_wire_bytes=sum(r['wire_bytes'] for r in data),
        send_blocking_ms=sum(r['socket_call_wall_ms'] for r in data if r['operation']=='send'),
        receive_blocking_ms=sum(r['socket_call_wall_ms'] for r in data if r['operation']=='recv'),
        framing_serialization_ms=sum(r.get('framing_serialization_ms',0)+r.get('framing_deserialization_ms',0) for r in data))


class Experiment:
    def __init__(self,pm,ref,cache,channel,output,*,model=None,adapter=None,device='cpu',fixture_base=None):
        self.pm,self.ref,self.cache,self.channel,self.output=pm,ref,cache,channel,Path(output)
        self.model,self.adapter,self.device,self.fixture_base=model,adapter,device,fixture_base
        self.fixture=fixture_base is not None;self.rows=[];self.reference=None;self.sequence=0
        self.native_raw=(Path(cache.context['prepared_directory'])/ref['logits_file']).read_bytes() if 'prepared_directory' in cache.context else None
        self.stages={s:dict(status='NOT_RUN') for s in ('B1','B2','B3')}

    def save_stages(self):
        write(self.output/'r3b_stage_gates.json',self.stages)
        for name,value in self.stages.items():write(self.output/f'{name}_gate.json',value)

    def start_stage(self,name):
        self.stages[name]=dict(status='RUNNING',started_monotonic_ns=time.monotonic_ns())
        self.save_stages();print('R3B '+name+' START',flush=True)

    def request(self,mode,stage,phase,repeat):
        before=len(self.rows)
        try:return self._request(mode,stage,phase,repeat)
        except BaseException as exc:
            if len(self.rows)==before:
                failed=dict(request_id=getattr(self,'active_request_id','FAILED_BEFORE_FORWARD'),stage=stage,
                    phase=phase,repeat=repeat,mode=mode,status='FAILED',reason=f'{type(exc).__name__}: {exc}',**flags(self.fixture))
                self.rows.append(failed);append(self.output/'r3b_requests.jsonl',failed)
            raise

    def _request(self,mode,stage,phase,repeat):
        from .edge_server import ProjectionSession
        identity=self.pm['identity'];request_id=f'{stage}_{phase}_{repeat:03d}_{mode}_{self.sequence:04d}';self.sequence+=1
        self.active_request_id=request_id
        event_start=len(self.cache.events);reset_ms=0.
        if mode=='COLD_MISS':
            reset_start=time.monotonic_ns();self.cache.reset('cold-miss experimental precondition')
            reset_ms=(time.monotonic_ns()-reset_start)/1e6
        if not self.fixture:
            import torch
            torch.cuda.synchronize(self.device)
        begin=time.monotonic_ns();cache_before=self.cache.bytes;frame_start=len(self.channel.records)
        lookup_start=time.monotonic_ns();entry=None
        if mode!='NO_REUSE':entry=self.cache.lookup(self.cache.context)
        lookup_ms=(time.monotonic_ns()-lookup_start)/1e6
        require((entry is not None)==(mode=='WARM_HIT'),'expected HIT/MISS precondition not satisfied')
        session=ProjectionSession(self.channel,identity,request_id,torch_backend=not self.fixture)
        captures={};observed=[];insert_ms=0.;admitted=False
        forward_start=time.monotonic_ns();gpu_ms=None
        if self.fixture:
            from .r3b_fixture import ArrayTensor,forward
            from .r3a_tiny import f32
            def project(layer,role,x,native):
                def miss():
                    delta=session.correction(layer,role,x)
                    return ArrayTensor([f32(a+b) for a,b in zip(native(),delta)],[1,identity['length'],identity['hidden_size']])
                values=total_projection(entry,layer,role,miss).values
                captures.setdefault(layer,{})[role]=ArrayTensor(values,[1,identity['length'],identity['hidden_size']])
                observed.append(dict(layer=layer,role=role,reused=entry is not None));return values
            values=forward(identity['token_ids'],self.fixture_base,project)
            if entry is None:session.complete()
            logits_raw=pack(values,'float32')
        else:
            begin_event,end_event=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            with torch.inference_mode(),physical_path(self.adapter,session,entry,captures,observed):
                begin_event.record()
                result=self.model(input_ids=self.input_tensor,use_cache=False,output_attentions=True)
                end_event.record();torch.cuda.synchronize(self.device)
            gpu_ms=begin_event.elapsed_time(end_event)
            require(str(result.logits.dtype)==self.ref['native_logits_dtype'] and list(result.logits.shape)==self.ref['logits_shape'],
                    'native endpoint dtype/shape changed')
            require(bool(torch.isfinite(result.logits).all()),'nonfinite request logits; no cache insertion')
        forward_ms=(time.monotonic_ns()-forward_start)/1e6
        require(len(observed)==3*identity['layers'],'incomplete QKV captures')
        blocks={layer:tuple(captures[layer][role] for role in 'qkv') for layer in range(identity['layers'])}
        if mode=='COLD_MISS':
            insertion_start=time.monotonic_ns()
            admitted=self.cache.insert(self.cache.context,blocks,request_id,completed=True)
            insert_ms=(time.monotonic_ns()-insertion_start)/1e6
            require(admitted,'cold MISS admission failed; no WARM HIT or performance claim')
        elif entry is not None:self.cache.reused(entry)
        if not self.fixture:torch.cuda.synchronize(self.device)
        end=time.monotonic_ns()
        frames=self.channel.records[frame_start:]
        # Full-logit/QKV quality export follows the synchronized request wall.
        # Cache integrity hashing/cloning/admission already happened INSIDE it.
        if not self.fixture:logits_raw=tensor_bytes(result.logits.float());del result
        qkv={f'{layer}.{role}':spec(t)['sha256'] for layer,block in blocks.items() for role,t in zip('qkv',block)}
        hashes=dict(logits_sha256=digest_bytes(logits_raw),qkv_sha256=qkv)
        if self.reference is None:
            require(mode=='NO_REUSE','first physical reference must be NO_REUSE')
            self.reference=hashes
            (self.output/'r3b_no_reuse_logits.f32.bin').write_bytes(logits_raw)
            write(self.output/'r3b_no_reuse_reference.json',dict(hashes,logits_shape=self.ref['logits_shape'],
                logits_dtype='float32',scope='same physical CPU-UD/ES endpoint; separate from Native PEFT'))
        bitwise=hashes==self.reference
        native=check_parity(unpack(logits_raw,'float32'),unpack(self.native_raw,'float32'),self.ref['logits_shape'])
        totals=wire_totals(frames)
        row=dict(request_id=request_id,stage=stage,phase=phase,repeat=repeat,mode=mode,
            cache_status='HIT' if entry is not None else 'MISS' if mode=='COLD_MISS' else 'BYPASS',
            reused_qkv_outputs=sum(r['reused'] for r in observed),observed_qkv_outputs=len(observed),
            es_pid=os.getpid(),ud_pid=self.ud_pid,request_start_monotonic_ns=begin,request_end_monotonic_ns=end,
            request_wall_ms=(end-begin)/1e6,cache_lookup_ms=lookup_ms,cache_insert_ms=insert_ms,
            gpu_forward_elapsed_ms=gpu_ms,gpu_interval_scope='CUDA event includes stream idle/socket gaps; not kernel self-time',
            forward_host_ms=forward_ms,cache_reset_ms_outside_request=reset_ms,cache_bytes_before=cache_before,
            cache_bytes_after=self.cache.bytes,cache_admitted=admitted,
            cache_validation_cpu_read_bytes=entry.size_bytes if entry is not None else 0,
            cache_integrity_transfer_hash_self_time_status='NOT_SEPARATELY_MEASURED; included in cache_lookup_ms/cache_insert_ms',
            cache_gpu_clone_self_time_status='NOT_SEPARATELY_MEASURED; included in forward_host_ms/CUDA interval/cache_insert_ms',
            ud_cpu_lora_ms=sum(l['ud']['cpu_lora_ms'] for l in session.trace),
            es_hidden_copy_cast_ms=sum(l['es_hidden_copy_cast_ms'] for l in session.trace),
            es_delta_cpu_to_gpu_ms=sum(sum(l['es_delta_cpu_to_gpu_ms'].values()) for l in session.trace),
            es_delta_deserialization_ms=sum(l['es_delta_deserialization_ms'] for l in session.trace),
            ud_input_deserialization_ms=sum(l['ud']['input_deserialization_ms'] for l in session.trace),
            ud_output_serialization_ms=sum(l['ud']['output_serialization_ms'] for l in session.trace),
            cache_events=self.cache.events[event_start:],cache_bitwise_passed=bitwise,native_parity=native,
            **totals,**hashes,**flags(self.fixture))
        self.rows.append(row);append(self.output/'r3b_requests.jsonl',row)
        append(self.output/f'{stage}_projection_trace.jsonl',dict(request_id=request_id,qkv=qkv,observed=observed,
            cache_events=row['cache_events'],physical_layers=session.trace))
        if not bitwise:self.cache.reset('failed cache parity; stop before any further request')
        validate_row(row,identity['layers'])
        return row

    def run(self,warmup,measured):
        self.start_stage('B1')
        self.request('NO_REUSE','B1','correctness',0)
        self.request('COLD_MISS','B1','correctness',0)
        self.stages['B1']=dict(status='PASS',full_completed_entry=True,owned_cache_bytes=self.cache.bytes,completed_monotonic_ns=time.monotonic_ns())
        self.save_stages();print('R3B B1 PASS',flush=True)
        self.start_stage('B2')
        warm=self.request('WARM_HIT','B2','correctness',0)
        guards=invalid_guard_checks(self.cache)
        self.stages['B2']=dict(status='PASS',cold_vs_no_reuse_bitwise=True,warm_vs_no_reuse_bitwise=True,
            invalid_guards=guards,warm_reused_outputs=warm['reused_qkv_outputs'],warm_data_frames=0,
            native_parity_is_separate=True,completed_monotonic_ns=time.monotonic_ns())
        self.save_stages();print('R3B B2 PASS',flush=True)
        self.start_stage('B3')
        for phase,count in (('warmup',warmup),('measured',measured)):
            for repeat in range(count):
                # COLD_MISS empties and repopulates before this round's WARM_HIT.
                for mode in MODES:self.request(mode,'B3',phase,repeat)
        summary=summarize(self.rows,measured,self.pm['identity']['layers'])
        self.stages['B3']=dict(status='PASS',summary=summary,warmup_per_mode=warmup,measured_per_mode=measured,
            order=list(MODES),order_randomized=False,completed_monotonic_ns=time.monotonic_ns())
        self.save_stages();print('R3B B3 PASS',flush=True)
        return summary


def execute(plan,output):
    """One ES model process, one independent original CPU UD process."""
    from .collaborative_inference import launch,wait_ready,cleanup
    from .r3a_runtime import load_bare,runtime_setup,memory
    pm,ref,am=validate_prepared(plan['prepared']);fixture=pm.get('TEST_FIXTURE_ONLY') is True
    require(fixture==plan.get('fixture',False),'fixture/production gate mismatch')
    identity=pm['identity'];model=adapter=base=None;gpu=None
    require(sha(Path(plan['prepared'])/'r3a_prepare_manifest.json')==plan['prepared_manifest_sha256'],'immutable prepare changed')
    if fixture:
        base=read(Path(plan['prepared'])/'r3a_tiny_base.json');require(digest(base)==identity['base_weight_sha256'],'fixture base SHA mismatch')
    else:
        import torch
        require(identity['model']=='facebook/opt-2.7b' and identity['length']==32 and identity['layers']==32,
                'R3-B paper-facing endpoint requires OPT-2.7B L32 with 32 layers')
        runtime_setup(plan['device']);torch.manual_seed(identity['seed'])
        available=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,available[:4]);torch.set_num_threads(min(4,len(available)))
        torch.set_num_interop_threads(1)
        gpu=memory(plan['device']);require(gpu['uuid']==pm['gpu']['uuid'],'fresh prepared/ES GPU UUID mismatch')
        require('3090' in gpu['name'],'this R3-B cohort requires an allocated RTX 3090')
        require(versions()==identity['versions'],'fresh prepared/ES versions mismatch')
        model,tokenizer,meta=load_bare(identity,plan['device'])
        from semcache.models.model_adapter import OPTModelAdapter
        from semcache.system_cost.es_base_profile import assert_no_lora
        adapter=OPTModelAdapter(model);assert_no_lora(model,adapter)
        require(full_base_fingerprint(model)['sha256']==identity['base_weight_sha256'],'bare ES fingerprint mismatch')
    context=request_context(pm,plan['prepared_manifest_sha256'],'TEST_FIXTURE_ONLY' if fixture else gpu['uuid'])
    # Local artifact locator is included in the sealed provenance, never sent as network payload.
    context['prepared_directory']=plan['prepared']
    cache=ExactRawCache(context,plan['capacity_bytes'],'cpu' if fixture else plan['device'],fixture=fixture)
    result=dict(execution_status='RUNNING',pid=os.getpid(),gpu=gpu,base_only=True,adapter_parameters_on_es=0,
        environment=process_environment(),versions=versions(),**flags(fixture))
    children=[];channel=None;experiment=None
    try:
        session=uuid.uuid4().hex;write(output/'ud_plan.json',dict(prepared=plan['prepared'],output=str(output),session_id=session))
        ud=launch('ud',output/'ud_plan.json',output);children.append(('ud',ud))
        ready=wait_ready(ud[0],output/'r3a_ud_ready.json')
        for stage in ('B1','B2','B3'):(output/f'{stage}_wire_trace.jsonl').touch()
        def wire_trace(record):
            append(output/'r3b_wire_trace.jsonl',record)
            stage=record.get('header',{}).get('request_id','').split('_')[0]
            if stage in ('B1','B2','B3'):append(output/f'{stage}_wire_trace.jsonl',record)
        channel=Channel(connect(ready['port']),session,am['matrix_sha256'],trace=wire_trace)
        channel.send(header('HELLO','handshake',-1,pid=os.getpid(),identity_sha256=pm['identity_sha256']))
        response,_,_=channel.recv(kind='READY',request_id='handshake',layer_id=-1)
        require(response['metadata']['pid']==ready['pid']==ud[0].pid!=os.getpid() and
            response['metadata']['identity_sha256']==pm['identity_sha256'] and response['metadata']['user_device_cuda_used'] is False,
            'actual CPU UD process/identity isolation failed')
        result.update(actual_network_socket_used=True,user_device_cuda_used=False,ud_pid=ud[0].pid)
        experiment=Experiment(pm,ref,cache,channel,output,model=model,adapter=adapter,
            device=plan['device'],fixture_base=base);experiment.ud_pid=ud[0].pid
        if not fixture:experiment.input_tensor=torch.tensor([identity['token_ids']],device=plan['device'])
        write(output/'r3b_gpu_execution.json',dict(gpu_inference_performed=not fixture,TEST_FIXTURE_ONLY=fixture))
        summary=experiment.run(plan['warmup'],plan['measured'])
        channel.send(header('STOP','shutdown',-1));channel.recv(kind='BYE',request_id='shutdown',layer_id=-1)
        ud[0].wait(timeout=10.);require(ud[0].returncode==0,'UD exit failure')
        ud_result=read(output/'r3a_ud_result.json')
        require(ud_result['user_device_cuda_used'] is False and ud_result['full_model_loaded'] is False,
                'UD loaded full model or used CUDA')
        expected_requests=sum(r['hidden_frames']>0 for r in experiment.rows)
        require(ud_result['completed_requests']==expected_requests and len(ud_result['layers'])==expected_requests*identity['layers'],
                'UD did work for HIT or request accounting differs')
        result.update(execution_status='COMPLETE',summary=summary,stages=experiment.stages,
            native_strict_passes=sum(r['native_parity']['passed'] for r in experiment.rows),request_count=len(experiment.rows),
            ud_result_sha256=sha(output/'r3a_ud_result.json'),ud_cpu=ud_result['cpu'])
        if not fixture:
            assert_no_lora(model,adapter)
            require(full_base_fingerprint(model)['sha256']==identity['base_weight_sha256'],'ES base mutated during experiment')
        return result
    finally:
        if channel is not None:channel.close()
        result['cleanup']=cleanup(children)
        if experiment is not None:
            for stage,value in experiment.stages.items():
                if value['status']=='RUNNING':value.update(status='FAIL',reason='request/guard/measurement exception; see worker traceback')
            result['stages']=experiment.stages
            experiment.save_stages()
        write(output/'r3b_es_partial.json',result)
