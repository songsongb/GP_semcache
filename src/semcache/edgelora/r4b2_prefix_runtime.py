"""Physical suffix-prefill ES and independent B2 CPU UD lifecycle."""
from copy import deepcopy
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

from .communication import Channel,connect,digest,digest_bytes,header,require
from .r3a_artifacts import append,full_base_fingerprint,pack,process_environment,read,tensor_bytes,unpack,write
from .r3c_process import ready,reap
from .r4a_evidence import load_prepared
from .r4a_parity import capture,compare
from .r4a_runtime import setup
from .r4b2_cache import PrefixCache
from .r4b2_generation import PrefixDriver,generate
from .r4b2_workload import schedule,validate_suite,tokenizer_configuration


def flags(fixture):
    return dict(TEST_FIXTURE_ONLY=fixture,PRIMARY_PERFORMANCE_ELIGIBLE=False,correctness_qualified_speedup=False,
        NATIVE_PEFT_STRICT_PARITY_PASS=False,native_parity_deferred=True,semantic_matching=False,compression=False,
        actual_remote_network_measured=False,concurrent_serving=False)


def load_inputs(prepared):
    prepared=Path(prepared);m=read(prepared/'r4b2_prepared.json')
    from .r4a_evidence import verify_sources
    verify_sources(m['sources'])
    for name,h in m['hashes'].items():
        p=(prepared/name).resolve();require(p.is_relative_to(prepared.resolve()) and digest_bytes(p.read_bytes())==h,'B2 prepared input changed')
    native,suite_old,pm,ref,am=load_prepared(m['native_prepared'])
    require(native['sources']==m['sources'] and m['base_prepared']==native['base_prepared'],'B2/native preparation source identity differs')
    suite=read(prepared/'r4b2_suite.json');validate_suite(suite)
    return m,suite,native,pm,ref,am


def launch_ud(plan,output):
    write(output/'ud_plan.json',dict(plan,output=str(output)))
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',HIP_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
    stdout=(output/'ud.stdout.log').open('w');stderr=(output/'ud.stderr.log').open('w')
    try:p=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve().parents[3]/'scripts/104_run_r4b2_shared_prefix.py'),
        '--worker','ud','--plan',str(output/'ud_plan.json'),'--test-fixture-only' if plan['fixture'] else '--enable-gpu-experiment'],
        cwd=Path(__file__).resolve().parents[3],env=env,stdout=stdout,stderr=stderr)
    except BaseException:stdout.close();stderr.close();raise
    return p,stdout,stderr


def negative_controls(driver,cache,suite):
    """Expected rejections are separate records, never recomputed HIT rows."""
    lookup={p['sample_id']:p for p in suite['prompts']};first=suite['conditions'][0];p=lookup[first['targets'][0]];results=[]
    def reject(name,target=None,**options):
        target=deepcopy(p if target is None else target);rid='NEG_'+name
        # A changed token list has a new honest digest; do not conflate exact
        # mismatch with a deliberately stale digest in these controls.
        if name not in ('bad_digest',):target['prompt_sha256']=digest(dict(tokens=target['token_ids'],mask=target['attention_mask']))
        before=len(driver.channel.records)
        try:generate(driver,target,suite['policy'],rid,cache,'PARTIAL_PREFIX_HIT',**options)
        except (ValueError,KeyError) as exc:
            frames=driver.channel.records[before:];data=[r for r in frames if r['header']['kind'] in ('HIDDEN','DELTAS')]
            require(not data,'invalid prefix HIT secretly recomputed')
            results.append(dict(case=name,request_id=rid,status='EXPECTED_REJECTION',reason=str(exc),data_frames=0,
                control_frames=len(frames),recomputed=False,alternative_model_or_adapter_executed=False))
        else:raise ValueError('invalid prefix HIT accepted: '+name)
    changed=deepcopy(p);changed['token_ids']=[5]*len(p['token_ids']);reject('no_matching_prefix',changed)
    changed=deepcopy(p);changed['token_ids'][first['prefix_length']//2]=(changed['token_ids'][first['prefix_length']//2]+1)%driver.vocab
    reject('one_token_difference',changed)
    changed=deepcopy(p);changed['token_ids']=[5]+p['token_ids'][:-1];reject('nonprefix_offset',changed)
    reject('adapter_mismatch',cohort='0'*64);reject('model_mismatch',cohort='1'*64)
    changed=deepcopy(p);changed['attention_mask'][0]=0;reject('mask_mismatch',changed)
    changed=deepcopy(p);changed['positions'][0]=99;reject('position_mismatch',changed)
    changed=deepcopy(p);changed['prompt_sha256']='0'*64;reject('bad_digest',changed)
    reject('unsupported_length',expected_prefix=16)
    key=next(iter(cache.entries));saved=deepcopy(cache.proofs[key]);cache.proofs[key]['prefix_length']+=1
    reject('corrupt_proof');results[-1].update(entry_id=key,original_proof_sha256=digest(saved),corrupted_proof_sha256=digest(cache.proofs[key]))
    cache.proofs[key]=saved
    # A real completed source is offered to an actually undersized owner. No
    # tensor is cloned and no grant is registered after denied admission.
    owner=cache.entries[key];entry=next(iter(owner.pool.entries.values()));proof=cache.proofs[key]
    source=next(x for x in suite['prompts'] if x['token_ids']==owner.context['token_ids'])
    small=PrefixCache(cache._pm,cache.prepared_sha,cache.gpu_uuid,cache.device,1,tokenizer=cache.tokenizer,sources=cache.sources,
        fixture=cache.fixture,contract=cache.contract,observed_cache_type=cache.cache_type,synchronize=driver.synchronize)
    denied=small.admit(source,entry.tensors,proof,entry.qkv_metadata['source_request_id'],completed=True)
    require(denied is None and small.bytes==0,'insufficient cache capacity was ignored')
    results.append(dict(case='insufficient_capacity',status='EXPECTED_REJECTION',capacity_bytes=1,source_request_id=entry.qkv_metadata['source_request_id'],
        events=small.events,occupancy_bytes=small.bytes,recomputed=False,data_frames=0))
    cache.request_id='NEG_EVICT';cache.remove(key,'EVICT','negative_control_explicit_eviction');reject('evicted_prefix')
    remaining=next(iter(cache.entries),None)
    if remaining is not None:
        owner=cache.entries[remaining];entry=next(iter(owner.pool.entries.values()))
        target=next(x for x in suite['prompts'] if x['role']=='target' and x['token_ids'][:len(owner.context['token_ids'])]==owner.context['token_ids'])
        tensor=entry.tensors[0][0]
        from .r3b_cache import spec
        before=spec(tensor)
        if driver.fixture:tensor.values[0]+=1.
        else:
            import torch
            with torch.no_grad():tensor.view(torch.uint8).reshape(-1)[0].bitwise_xor_(1)
        after=driver.evidence.save(tensor,'NEG_CORRUPTED_CACHE_Q');del tensor
        reject('corrupt_tensor',target);results[-1].update(entry_id=remaining,original_q=before,corrupted_q=after)
    write(driver.output/'r4b2_negative_controls.json',results);return results


def execute(plan,output):
    output=Path(output);m,suite,native,pm,ref,am=load_inputs(plan['prepared']);require(m['sources']==plan['sources'],'B2 run source revision differs')
    pm,ref,am,old,tokenizer,gpu=setup(plan,native=False)
    driver=PrefixDriver(old.identity,old.device,model=old.model,adapter=old.adapter,tiny=old.tiny,native=False)
    if driver.fixture:driver.tiny.matrices=None
    else:driver.model_tokenizer=tokenizer
    driver.configure(output);fixture=driver.fixture
    if not fixture:tokenizer.truncation_side='right'
    tok=dict(vocab_sha256='TEST_FIXTURE_ONLY' if fixture else digest(tokenizer.get_vocab()),special_tokens=suite['special_tokens'],
        source=pm['identity']['tokenizer_source'],revision=pm['identity']['tokenizer_revision'],configuration=tokenizer_configuration(tokenizer,fixture=fixture))
    require(tok['vocab_sha256']==native['native_result']['tokenizer_vocab_sha256'],'B2 actual tokenizer differs from prepared tokenizer')
    require(tok['configuration']==suite['tokenizer_configuration'],'actual ES tokenizer configuration differs from prepared B2 suite')
    if not fixture:require({k:getattr(tokenizer,k) for k in tok['special_tokens']}==tok['special_tokens'],'actual special token IDs differ')
    uuid_gpu='TEST_FIXTURE_ONLY' if fixture else gpu['uuid'];channel=ud=None;control_count=0;rows=[];result={};start=time.monotonic()
    def control(message):
        nonlocal control_count
        rid=f'PREFIX_CONTROL_{control_count:04d}';control_count+=1
        channel.send(header('HELLO',rid,-1,**message));h,_,_=channel.recv(kind='READY',request_id=rid,layer_id=-1)
        require(h['metadata']['control_sha256']==digest(message) and h['metadata']['user_device_cuda_used'] is False,'UD prefix grant acknowledgement differs')
        append(output/'es_prefix_controls.jsonl',dict(message,request_id=rid))
    cache=PrefixCache(pm,native['base_manifest_sha256'],uuid_gpu,driver.device,plan['capacity_bytes'],tokenizer=tok,sources=plan['sources'],fixture=fixture,
        contract=driver.contract,observed_cache_type=None,synchronize=driver.synchronize,control=control,
        trace=lambda e:append(output/'r4b2_cache_events.jsonl',e))
    write(output/'r4b2_opt_contract.json',dict(implementation=driver.contract,cohort_sha256=cache.cohort,tokenizer=tok,gpu_uuid=uuid_gpu))
    try:
        session=uuid.uuid4().hex;ud=launch_ud(dict(plan,session_id=session,cohort_sha256=cache.cohort),output)
        info=ready(ud,output/'r4b2_ud_ready.json');require(info['pid']==ud[0].pid!=os.getpid(),'UD PID isolation failed')
        channel=Channel(connect(info['port']),session,am['matrix_sha256'],timeout=120.,trace=lambda r:append(output/'es_wire.jsonl',r));driver.channel=channel
        channel.send(header('HELLO','handshake',-1,application='R4B2_PREFIX_V2',cohort_sha256=cache.cohort,identity_sha256=pm['identity_sha256']))
        h,_,_=channel.recv(kind='READY',request_id='handshake',layer_id=-1)
        require(h['metadata']['cohort_sha256']==cache.cohort and h['metadata']['pid']==ud[0].pid and h['metadata']['user_device_cuda_used'] is False,'B2 UD handshake failed')
        triage={}
        for use in (False,True):
            directory=output/f'triage_es_{int(use)}';directory.mkdir();driver.sample_id='ORIGINAL_R3A_INPUT'
            if not fixture:write(output/'gpu_attempt.json',dict(gpu_inference_attempted=True))
            out,record,_,_,_=driver.b2_forward(f'ORIGINAL_TRIAGE_{int(use)}',0,pm['identity']['token_ids'],pm['identity']['token_ids'],None,1,
                'NO_REUSE',cache.cohort,use_cache=use,observer=None if fixture else lambda a,d=directory:capture(a,d))
            if not fixture:write(output/'gpu_execution.json',dict(gpu_inference_performed=True))
            blob=pack(out.logits,'float32') if fixture else tensor_bytes(out.logits.float());shape=out.shape if fixture else list(out.logits.shape)
            (directory/'logits.f32.bin').write_bytes(blob)
            write(directory/'forward.json',dict(logits_file='logits.f32.bin',logits_sha256=digest_bytes(blob),shape=shape,use_cache=use,
                model_logits_dtype=record['model_logits_dtype']))
            triage[str(use)]=compare(Path(m['native_prepared'])/f'triage_native_{int(use)}',directory,unpack(blob,'float32'),shape,fixture=fixture)
            append(output/'triage_passes.jsonl',record);del out
        cache.cache_type=driver.cache_type
        write(output/'r4b2_parity_diagnostics.json',dict(original_input_use_cache_false=triage['False'],original_input_use_cache_true=triage['True'],
            native_B2_generation='NOT_RUN_DEFERRED',frozen_thresholds_unchanged=True,**flags(fixture)))
        tasks=schedule(suite);write(output/'r4b2_schedule.json',tasks);prompts={p['sample_id']:p for p in suite['prompts']}
        for task in tasks:
            require(time.monotonic()-start<7200,'bounded B2 worker time exhausted')
            row=generate(driver,prompts[task['sample_id']],suite['policy'],task['request_id'],cache,task['mode'],
                expected_prefix=task['prefix_length'] if task['mode']=='PARTIAL_PREFIX_HIT' else None)
            blob=row.pop('logits_blob');name=f'logits/{row["request_id"]}.f32.bin';(output/'logits').mkdir(exist_ok=True);(output/name).write_bytes(blob)
            row.update(task,logits_file=name,es_pid=os.getpid(),ud_pid=ud[0].pid,cohort_sha256=cache.cohort,**flags(fixture))
            if not fixture:
                from .r3a_runtime import memory
                row['gpu_memory_outside_wall']=memory(driver.device)
            rows.append(row);append(output/'r4b2_requests.jsonl',row);append(output/'r4b2_generation_outputs.jsonl',row)
        negatives=negative_controls(driver,cache,suite)
        channel.send(header('STOP','shutdown',-1));channel.recv(kind='BYE',request_id='shutdown',layer_id=-1);channel.close();channel=None
        ud[0].wait(timeout=10.);require(ud[0].returncode==0,'B2 UD failed')
        if not fixture:
            from semcache.system_cost.es_base_profile import assert_no_lora
            from .r3a_runtime import memory
            assert_no_lora(driver.model,driver.adapter);require(full_base_fingerprint(driver.model)['sha256']==pm['identity']['base_weight_sha256'],'ES base weights changed')
            gpu=memory(driver.device)
        result=dict(execution_status='COMPLETE',pid=os.getpid(),ud_pid=ud[0].pid,session_id=session,cohort_sha256=cache.cohort,heads=driver.heads,
            gpu=gpu,gpu_inference_performed=not fixture,adapter_parameters_on_es=0,user_device_cuda_used=False,tokenizer=tok,
            requests=len(rows),cache_final=cache.snapshot(),tensor_evidence_bytes=driver.evidence.bytes,negative_controls=len(negatives),
            environment=process_environment(),**flags(fixture));return result
    finally:
        if channel is not None:channel.close()
        if ud is not None:result['ud_lifecycle']=reap(ud,output/'ud_lifecycle.json',group=False)
        result['partial_rows']=len(rows);write(output/'r4b2_partial.json',result)
