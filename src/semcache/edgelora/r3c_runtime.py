"""Persistent bare ES + CPU UD + calibrated transparent TCP proxy workload."""
import os
from pathlib import Path
import time
import uuid

from .communication import Channel,header,require,digest
from .r3a_artifacts import (append,full_base_fingerprint,process_environment,read,sha,
    validate_prepared,versions,write)
from .r3b_runtime import Experiment
from .r3b_cache import ExactRawCache,request_context
from .r3c_analysis import flags
from .r3c_network_proxy import control,connect
from .r3c_calibration import BW_REL_TOL,RTT_ABS_TOL_MS
from .r3c_process import spawn,ready,reap


def offline_model_preflight():
    """Check the exact loader's offline .bin snapshot without a CUDA context."""
    from huggingface_hub import try_to_load_from_cache
    from .r3a_artifacts import read
    model='facebook/opt-2.7b'
    def cached(name):
        value=try_to_load_from_cache(model,name,revision='main')
        require(isinstance(value,str) and Path(value).is_file(),'offline model cache missing: '+name)
        return Path(value)
    config=cached('config.json');settings=read(config)
    require(settings['model_type']=='opt' and settings['num_hidden_layers']==32 and settings['hidden_size']==2560,
            'offline OPT-2.7B architecture mismatch')
    revision=config.parent.name
    require(len(revision)==40 and all(c in '0123456789abcdef' for c in revision),'immutable offline snapshot required')
    files=[config]
    tokenizer=try_to_load_from_cache(model,'tokenizer.json',revision='main')
    if isinstance(tokenizer,str) and Path(tokenizer).is_file():files.append(Path(tokenizer))
    else:files.extend((cached('vocab.json'),cached('merges.txt')))
    single=try_to_load_from_cache(model,'pytorch_model.bin',revision='main')
    if isinstance(single,str) and Path(single).is_file():files.append(Path(single))
    else:
        index=cached('pytorch_model.bin.index.json');files.append(index)
        names=set(read(index)['weight_map'].values());require(names,'empty model shard index')
        for name in sorted(names):
            require(Path(name).name==name and name.endswith('.bin'),'unsafe model shard name')
            files.append(cached(name))
    require(all(p.parent.name==revision and p.stat().st_size>0 for p in files),'offline snapshot files use mixed revisions/empty files')
    return dict(status='PASS',model=model,revision=revision,local_files_only=True,no_cuda_context_created=True,
        files=[dict(path=str(p),bytes=p.stat().st_size) for p in files],
        scope='cache presence/config only; original 95 prepare independently fingerprints actual loaded weights and tokenizer')


def proxy_request_metrics(frames,profile):
    result=dict(delayed_frames=sum(f['delayed_frames'] for f in frames),frames=len(frames),
        one_way_delay_wait_ms=sum(f['one_way_delay_wait_ms'] for f in frames),
        pacing_wait_ms=sum(f['pacing_wait_ms'] for f in frames),
        application_queue_wait_ms=sum(f['application_queue_wait_ms'] for f in frames),
        timing_scope='nested proxy clocks; never added to request wall',directions={},valid=True)
    for f in frames:
        require(f['complete'] and f['condition_id']==profile['id'] and f['wire_bytes_read']==f['wire_bytes_forwarded'],
                'proxy incomplete, wrong profile, or unequal bytes')
        require(f['delayed_frames']==int(profile['rtt_ms']>0),'frame delay charged more than once or omitted')
        delay=profile['rtt_ms']/2
        if delay:
            result['valid']=result['valid'] and delay<=f['one_way_delay_wait_ms']<=delay+max(RTT_ABS_TOL_MS,delay*.35)
    for direction in ('es_to_ud','ud_to_es'):
        selected=[f for f in frames if f['direction']==direction]
        meaningful=[f for f in selected if f['header']['payload_bytes']>=128*1024]
        elapsed=sum(f['serialization_ms'] for f in meaningful)
        rate=sum(f['wire_bytes_forwarded'] for f in meaningful)*.008/elapsed if elapsed else None
        requested=profile[direction+'_mbps'];ok=requested is None or rate is None or abs(rate-requested)/requested<=BW_REL_TOL
        result['valid']=result['valid'] and ok
        result['directions'][direction]=dict(wire_bytes=sum(f['wire_bytes_forwarded'] for f in selected),frames=len(selected),
            requested_mbps=requested,realized_mbps_for_large_frames=rate,valid=ok,
            rate_status='MEASURED' if rate is not None else 'NOT_ESTIMABLE_MODEL_FRAMES_TOO_SMALL_OR_NO_DATA')
    return result


class NetworkExperiment(Experiment):
    def __init__(self,*args,plan,**kwargs):
        super().__init__(*args,**kwargs);self.plan=plan;self.network_rows=[];self.execution_order=0
        self.deadline=time.monotonic()+plan['budget']['max_duration_seconds'];self.condition_gates={}
        self.proxy=plan['proxy'];self.token=plan['proxy_token'];self.index=None

    def checkpoint(self,**extra):
        return control(self.proxy['control_port'],self.token,'DRAIN',
            expected=dict(es_to_ud=self.channel.sent,ud_to_es=self.channel.received),**extra)

    def one(self,profile,arm,phase,ordinal,mode,hit_fraction=None,arm_reset_ms=0.):
        require(time.monotonic()<self.deadline,'bounded model experiment time exhausted')
        before=self.checkpoint();index=before['next_index']
        cal=next(c for c in self.plan['calibration']['conditions'] if c['condition']['id']==profile['id'])
        require(cal['status']=='VALID','INVALID_IMPAIRMENT: calibration failed')
        prior_rows=len(self.network_rows)
        try:
            original=super().request(mode,profile['id'],phase,ordinal)
            after=self.checkpoint(since=index);frames=after['records']
            require(len(frames)==original['hidden_frames']+original['delta_frames'] and
                all(f['header']['request_id']==original['request_id'] for f in frames),'proxy request frame count/identity mismatch')
            metric=proxy_request_metrics(frames,profile)
            require(sum(f['wire_bytes_forwarded'] for f in frames)==original['application_wire_bytes'],'relay/ES request wire bytes differ')
            entry=self.cache.pool.entries.get(self.cache.key(self.cache.context))
            row=dict(original,**{k:v for k,v in flags(self.fixture).items() if k not in original})
            row.update(flags(self.fixture))
            row.update(condition_id=profile['id'],condition=profile,arm=arm,hit_fraction=hit_fraction,ordinal=ordinal,
                execution_order=self.execution_order,proxy=metric,proxy_frames=frames,proxy_pid=self.proxy['pid'],
                calibration_status=cal['status'],calibration_round_trip_residual_ms=cal['round_trip_residual_ms'],
                calibration=cal,arm_reset_ms_outside_request=arm_reset_ms,
                forced_miss_provenance='explicit empty-cache reset outside wall, real lookup/insertion inside' if mode=='COLD_MISS' else None,
                fill_source_request_id=None if entry is None else entry.qkv_metadata['source_request_id'],
                reference_identity=self.plan['reference_identity'],seed=self.plan['seed'],
                cache_residency='TEST_FIXTURE_ONLY_CPU' if self.fixture else 'ES_GPU',cache_capacity_bytes=self.cache.pool.capacity_bytes,
                application_network_emulation=True,actual_network_socket_used=True,
                workload_scope='resident HIT' if hit_fraction==1 and phase=='measured' else 'controlled HIT/MISS mixture',
                impairment_status='VALID' if metric['valid'] else 'INVALID_IMPAIRMENT')
            self.network_rows.append(row);append(self.output/'r3c_requests.jsonl',row);self.execution_order+=1
            require(metric['valid'],'INVALID_IMPAIRMENT: model-session delivery materially differs from configured profile')
            return row
        except BaseException as exc:
            if len(self.network_rows)==prior_rows:
                failure=dict(status='FAILED',request_id=getattr(self,'active_request_id','NOT_STARTED'),
                    condition_id=profile['id'],condition=profile,arm=arm,phase=phase,ordinal=ordinal,mode=mode,
                    reason=f'{type(exc).__name__}: {exc}',reference_identity=self.plan['reference_identity'],**flags(self.fixture))
                self.network_rows.append(failure);append(self.output/'r3c_requests.jsonl',failure)
            append(self.output/'r3c_failures.jsonl',dict(condition_id=profile['id'],arm=arm,phase=phase,ordinal=ordinal,
                mode=mode,reason=f'{type(exc).__name__}: {exc}',execution_order=self.execution_order))
            raise

    def run_grid(self):
        order=[]
        for arm in self.plan['schedule']:
            if arm['condition']['id'] not in order:order.append(arm['condition']['id'])
        for cid in order:
            profile=next(p for p in self.plan['profiles'] if p['id']==cid)
            control(self.proxy['control_port'],self.token,'PROFILE',condition=profile,
                expected=dict(es_to_ud=self.channel.sent,ud_to_es=self.channel.received))
            self.condition_gates[cid]=dict(status='RUNNING',started_monotonic_ns=time.monotonic_ns())
            write(self.output/'r3c_condition_gates.json',self.condition_gates)
            for i,mode in enumerate(('NO_REUSE','COLD_MISS','WARM_HIT')):self.one(profile,'GATE','gate',i,mode)
            self.condition_gates[cid]=dict(status='PASS',completed_monotonic_ns=time.monotonic_ns(),
                cache_no_reuse_bitwise_equal=True,native_parity_separate=True)
            write(self.output/'r3c_condition_gates.json',self.condition_gates)
            for task in (a for a in self.plan['schedule'] if a['condition']['id']==cid):
                start=time.monotonic_ns();self.cache.reset('declared arm precondition')
                if not self.fixture:
                    import torch
                    torch.cuda.synchronize(self.device)
                reset_ms=(time.monotonic_ns()-start)/1e6
                append(self.output/'r3c_preconditions.jsonl',dict(condition_id=cid,arm=task['arm'],kind='ARM_RESET',wall_ms=reset_ms,
                    reset_event=self.cache.events[-1],cache_bytes_after=self.cache.bytes,monotonic_ns=time.monotonic_ns()))
                for i,mode in enumerate(task['warmup_modes']):
                    self.one(profile,task['arm'],'warmup',i,mode,task['hit_fraction'],arm_reset_ms=reset_ms if i==0 else 0.)
                for i,mode in enumerate(task['measured_modes']):self.one(profile,task['arm'],'measured',i,mode,task['hit_fraction'])
            print('R3C condition '+cid+' COMPLETE',flush=True)


def execute(plan,output):
    from .r3a_runtime import load_bare,runtime_setup,memory
    pm,ref,am=validate_prepared(plan['prepared']);fixture=plan['fixture'];model=adapter=base=gpu=None
    require(pm.get('TEST_FIXTURE_ONLY',False)==fixture,'TEST_FIXTURE_ONLY data forbidden in GPU cohort')
    require(sha(Path(plan['prepared'])/'r3a_prepare_manifest.json')==plan['prepared_manifest_sha256'],'fresh prepared SHA mismatch')
    identity=pm['identity']
    if fixture:
        base=read(Path(plan['prepared'])/'r3a_tiny_base.json');require(digest(base)==identity['base_weight_sha256'],'fixture base mismatch')
    else:
        import torch
        runtime_setup(plan['device']);torch.manual_seed(identity['seed'])
        cpus=sorted(os.sched_getaffinity(0))[:4];os.sched_setaffinity(0,cpus);torch.set_num_threads(len(cpus));torch.set_num_interop_threads(1)
        gpu=memory(plan['device'])
        require(gpu['uuid']!='UNAVAILABLE' and gpu['uuid']==pm['gpu']['uuid'] and '3090' in gpu['name'],
                'same-allocation verified RTX3090 UUID required; unavailable UUID is not accepted')
        require(identity['model']=='facebook/opt-2.7b' and identity['length']==32 and versions()==identity['versions'],
                'model/length/version mismatch')
        model,tokenizer,meta=load_bare(identity,plan['device'])
        from semcache.models.model_adapter import OPTModelAdapter
        from semcache.system_cost.es_base_profile import assert_no_lora
        adapter=OPTModelAdapter(model);assert_no_lora(model,adapter)
        require(full_base_fingerprint(model)['sha256']==identity['base_weight_sha256'],'bare base fingerprint mismatch')
    ctx=request_context(pm,plan['prepared_manifest_sha256'],'TEST_FIXTURE_ONLY' if fixture else gpu['uuid']);ctx['prepared_directory']=plan['prepared']
    cache=ExactRawCache(ctx,64*1024*1024,'cpu' if fixture else plan['device'],fixture=fixture)
    ud=None;channel=None;experiment=None;result={}
    try:
        session=uuid.uuid4().hex
        ud=spawn('ud',dict(plan,session_id=session),output,new_session=False)
        ud_ready=ready(ud,output/'r3a_ud_ready.json')
        require(ud_ready['pid']==ud[0].pid!=os.getpid(),'UD PID mismatch')
        control(plan['proxy']['control_port'],plan['proxy_token'],'TARGET',port=ud_ready['port'])
        control(plan['proxy']['control_port'],plan['proxy_token'],'PROFILE',condition=plan['profiles'][0])
        channel=Channel(connect(plan['proxy']['data_port']),session,am['matrix_sha256'],timeout=120.,
            trace=lambda r:append(output/'es_wire.jsonl',r))
        channel.send(header('HELLO','handshake',-1,pid=os.getpid(),identity_sha256=pm['identity_sha256']))
        response,_,_=channel.recv(kind='READY',request_id='handshake',layer_id=-1)
        require(response['metadata']['pid']==ud[0].pid and response['metadata']['identity_sha256']==pm['identity_sha256'] and
            response['metadata']['user_device_cuda_used'] is False,'original UD identity/isolation mismatch')
        experiment=NetworkExperiment(pm,ref,cache,channel,output,model=model,adapter=adapter,
            fixture_base=base,device=plan['device'],plan=plan);experiment.ud_pid=ud[0].pid
        if not fixture:experiment.input_tensor=torch.tensor([identity['token_ids']],device=plan['device'])
        write(output/'model_execution.json',dict(gpu_inference_performed=not fixture,TEST_FIXTURE_ONLY=fixture))
        experiment.run_grid()
        channel.send(header('STOP','shutdown',-1));channel.recv(kind='BYE',request_id='shutdown',layer_id=-1);channel.close();channel=None
        ud[0].wait(timeout=10.);require(ud[0].returncode==0,'CPU UD exited unsuccessfully')
        control(plan['proxy']['control_port'],plan['proxy_token'],'DRAIN',disconnected=True)
        if not fixture:
            assert_no_lora(model,adapter)
            require(full_base_fingerprint(model)['sha256']==identity['base_weight_sha256'],'ES base mutated')
        result=dict(execution_status='COMPLETE',pid=os.getpid(),ud_pid=ud[0].pid,proxy_pid=plan['proxy']['pid'],session_id=session,
            gpu=gpu,environment=process_environment(),condition_gates=experiment.condition_gates,requests=len(experiment.network_rows),
            adapter_parameters_on_es=0,user_device_cuda_used=False,**flags(fixture))
        return result
    finally:
        if channel is not None:channel.close()
        if ud is not None:result['ud_lifecycle']=reap(ud,output/'ud_lifecycle.json',group=False)
        if experiment is not None:
            result['condition_gates']=experiment.condition_gates
            for value in result['condition_gates'].values():
                if value['status']=='RUNNING':value.update(status='FAIL',reason='network/correctness/worker failure; see stderr')
            write(output/'r3c_condition_gates.json',result['condition_gates'])
        write(output/'r3c_es_partial.json',result)
