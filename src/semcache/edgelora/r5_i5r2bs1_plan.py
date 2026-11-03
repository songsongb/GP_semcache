"""Read-only historical operands and a sealed isolated Linear diagnostic plan."""
import math
from pathlib import Path

from .communication import digest,digest_bytes,require
from .r3a_artifacts import ROOT,pack,unpack,write,validate_prepared
from .r5_i5_evidence import seal
from .r5_i5r1_plan import source_identity
from .r5_i5r2_analysis import source_provenance
from .r5_i5r2_tensors import Evidence,file_sha,slice_axis,DTYPES,UNKNOWN,safe_path

VERSION='I5R2B_S1_NATIVE_LINEAR_V1'
NAMESPACE=ROOT/'results/r5/i5r/numerical_fidelity/linear_shape_control'
CORE=('TF32_matmul','FP16_reduced_precision_reduction','float32_matmul_precision','cudnn_TF32',
      'cudnn_benchmark','cudnn_deterministic','deterministic_algorithms','cpu_threads','autocast_enabled','CUBLAS_WORKSPACE_CONFIG')


def fresh(path,protected=()):
    raw=Path(path);require(not raw.is_symlink(),'unsafe output symlink');p=raw.resolve()
    require(not p.is_relative_to(ROOT) or p.is_relative_to(NAMESPACE),'S1 output must use linear_shape_control namespace')
    require(all(not p.is_relative_to(Path(x).resolve()) and not Path(x).resolve().is_relative_to(p) for x in protected),'output overlaps input')
    require(not p.exists(),'fresh output directory required');p.mkdir(parents=True,mode=0o700);return p


def schedule(repetitions=6,warmups=2):
    require(type(repetitions) is int and repetitions in (6,8) and warmups==2,'bounded balanced 6/8 repeats and two warmups required')
    return [dict(warmup=warm,round=i,length=n) for warm,count in ((True,warmups),(False,repetitions))
            for i in range(count) for n in ((64,256) if i%2==0 else (256,64))]


def suffix(prefix,hidden,seed=42):
    """Versioned integer xorshift32, dyadic [-1,1) values exactly representable in FP16."""
    require(0<seed<2**32,'nonzero uint32 suffix seed required');state=seed;values=[]
    for _ in range(192*hidden):
        state^=(state<<13)&0xffffffff;state^=state>>17;state^=(state<<5)&0xffffffff;state&=0xffffffff
        values.append(((state&2047)-1024)/1024.)
    return prefix+pack(values,'float16')


def layout_storage(shape,stride,offset,*,width=2,limit=16*2**20):
    require(isinstance(stride,list) and len(stride)==len(shape) and type(offset) is int and offset>=0 and
        all(type(s) is int and s>=0 for s in stride),'invalid historical layout')
    extent=1
    for s,n in sorted((s,n) for s,n in zip(stride,shape) if n>1):
        require(s>=extent,'overlapping/unsupported historical layout');extent+=(n-1)*s
    size=offset+sum((n-1)*s for n,s in zip(shape,stride))+1
    require(size*width<=limit,'historical layout exceeds bounded storage');return size


def historical(run,review,*,expected_commit,fixture=False):
    ev=Evidence(run,max_input_bytes=4*2**30);run_seal=ev.verify()
    require(ev.json('i5_block_status.json')['status']=='COMPLETE','historical block incomplete')
    cfg=ev.json('i5_execution_config.json');require(cfg['fixture']==fixture,'historical fixture scope differs')
    provenance=source_provenance(cfg,expected_commit=expected_commit,fixture=fixture)
    pe=Evidence(cfg['experiment_prepared'],max_input_bytes=128*2**20);ps=pe.verify();plan=pe.json('i5_workload_manifest.json')
    p=dict(plan);eid=p.pop('experiment_id')
    require(digest(p)==eid==cfg['experiment_id'] and ps['manifest_sha256']==cfg['experiment_manifest_sha256'] and
        plan['blocks']==[cfg['block']] and plan['i5r2b']==cfg['i5r2b'] and cfg['i5r2b']['source']['commit']==cfg['source_commit'] and
        cfg['i5r2b']['source']['files']==cfg['sources'],'historical plan/source chain differs')
    re=Evidence(review,max_input_bytes=128*2**20);rs=re.verify('i5r2b_output_manifest.json')
    inv=re.json('i5r2b_capture_inventory.json');verdict=re.json('i5r2b_first_divergence.json')
    require(inv['config']==cfg['i5r2b'] and inv['source_commit']==cfg['source_commit'] and
        inv['independent_I5_audit']['status']=='PASS' and inv['independent_I5_audit']['TEST_FIXTURE_ONLY']==fixture and
        verdict['TEST_FIXTURE_ONLY']==fixture and (fixture or verdict['GPU_EVIDENCE_COMPLETE']=='PASS'),'historical exported independent audit incomplete/mismatched')
    tasks=cfg['block']['tasks'];require(len(tasks)==2,'one historical source/target pair required');s,t=tasks
    require(cfg['block']['dataset']=='qasper' and s['adapter_id']==t['adapter_id']=='Adapter_A' and
        s['user_id']==t['user_id']=='User_01' and len(s['token_ids'])==64 and len(t['token_ids'])==256 and
        s['token_ids']==t['token_ids'][:64],'historical exact-prefix pair differs')
    base='FULL_RECOMPUTE/Adapter_A/es';cap=ev.json(base+'/i5r2b_capture.json')
    require(cap==inv['inventories']['FULL_RECOMPUTE/ES'] and cap['status']=='COMPLETE','historical capture/export differs')
    backend=ev.json(base+'/i5r2b_backend.json');roles={};payloads={}
    for r in 'qkv':
        captures={}
        for label,task,n in (('source',s,64),('target',t,256)):
            ctx=cap['contexts'][task['request_id']];observed=ctx['actual_context']
            require(observed['input_token_ids']==task['token_ids'] and observed['past_length']==0 and observed['pass_index']==0 and
                observed['positions']==list(range(n)) and observed['attention_mask']==[1]*n,'historical input/position alignment differs')
            require(ctx['backend_before']==ctx['backend_after'] and (fixture or all(k in ctx['backend_before'] for k in CORE)),
                'historical backend settings unavailable/changed')
            if label=='source':source_backend=ctx['backend_before']
            else:require(ctx['backend_before']==source_backend,'historical S/F backend settings differ')
            for boundary in ('projection_input','base'):
                item=cap['entries'][task['request_id']+'/'+boundary+'_'+r]
                require(item['status']=='CAPTURED' and item['dtype']=='float16' and item['token_axis']==1 and len(item['shape'])==3 and len(item['observed_shape'])==3 and
                    item['shape'][:2] in ([1,64],[1,n]) and item['semantic_positions']==[0,item['shape'][1]] and
                    item['observed_shape'][:2]==[1,n],'missing/misaligned historical tensor: '+boundary)
                data=ev.tensor(base,item);unpack(data,'float16')  # Reject nonfinite recorded operands.
                prefix,shape=slice_axis(data,item['shape'],1,0,64,2)
                captures[label+'_'+boundary]=dict(item=item,prefix_sha256=digest_bytes(prefix))
                payloads[r+'/'+label+'_'+boundary]=prefix
                if label=='target' and boundary=='projection_input' and item['shape'][1]==256:payloads[r+'/full_input']=data
        require(payloads[r+'/source_projection_input']==payloads[r+'/target_projection_input'],'historical source/full input prefixes unequal')
        hidden=captures['source_projection_input']['item']['shape'][-1]
        require(all(x['item']['shape'][-1]==hidden for x in captures.values()) and (fixture or hidden==2560),'historical hidden dimension differs')
        roles[r]=dict(captures=captures,hidden=hidden,full_suffix_available=r+'/full_input' in payloads)
    return dict(config=cfg,roles=roles,payloads=payloads,backend=source_backend,provenance=provenance,
        model=ev.json('FULL_RECOMPUTE/Adapter_A/identity.json'),
        run_seal=run_seal,review_seal=rs,plan_seal=ps,verification_scope='Sealed raw bytes, historical Git blobs and exported independent audits; no fresh historical GPU audit')


def prepare(run,review,base_prepared,output,*,expected_commit,repetitions=6,seed=42,fixture=False):
    history=historical(run,review,expected_commit=expected_commit,fixture=fixture);cfg=history['config']
    pm,_,_=validate_prepared(base_prepared)
    require(Path(base_prepared).resolve()==Path(cfg['base_prepared']).resolve() and pm.get('TEST_FIXTURE_ONLY',False)==fixture,'base preparation scope/path differs')
    for key in ('model','base_revision','base_weight_sha256','tokenizer_source','tokenizer_revision','hidden_size','layers','dtype'):
        require(history['model']['model_identity'][key]==pm['identity'][key],'historical/prepared model identity differs: '+key)
    require(fixture or (pm['identity']['model']=='facebook/opt-2.7b' and pm['identity']['dtype']=='float16' and pm['identity']['hidden_size']==2560),
        'actual OPT-2.7B FP16 preparation required')
    if not fixture:
        from .r3a_artifacts import versions
        from .r4c_prepare import opt_tokenizer
        require(versions()==pm['identity']['versions'],'prepared/software dependency versions differ; use the verified semcache environment')
        require(opt_tokenizer(pm).identity==history['model']['tokenizer'],'offline tokenizer/configuration differs from historical run')
        require(1<=history['backend']['cpu_threads']<=4,'historical thread count exceeds this four-core diagnostic')
    require(all(file_sha(Path(base_prepared)/name)==sha for name,sha in pm['artifact_hashes'].items()),'prepared artifacts changed')
    records={};longs={}
    for r,info in history['roles'].items():
        prefix=history['payloads'][r+'/source_projection_input'];h=info['hidden']
        longs[r]=history['payloads'][r+'/full_input'] if info['full_suffix_available'] else suffix(prefix,h,seed)
        records[r]=dict(info,suffix_method='RECORDED_FULL_INPUT' if info['full_suffix_available'] else 'SYNTHETIC_XORSHIFT32_DYADIC_V1',suffix_seed=seed,
            layouts=['CONTIGUOUS'],layout_limit_bytes=16*2**20)
        original=[info['captures'][x+'_projection_input']['item'] for x in ('source','target')]
        if any(x.get('original_contiguous') is False or x.get('storage_offset',0)!=0 for x in original):
            for x in original:layout_storage(x['observed_shape'],x.get('observed_stride'),x.get('storage_offset'),width=4)
            records[r]['layouts'].append('HISTORICAL_STRIDED')
    current_source=source_identity(fixture)  # Fail before any writes if the reviewed source is not committed.
    out=fresh(output,(run,review,base_prepared,cfg['experiment_prepared']))
    blobs={}
    for r in 'qkv':
        for name,data in [('input_long',longs[r])]+[(n,history['payloads'][r+'/'+n]) for n in
                ('source_projection_input','target_projection_input','source_base','target_base')]:
            path='inputs/'+r+'_'+name+'.bin';safe_path(out,path).parent.mkdir(exist_ok=True)
            safe_path(out,path).write_bytes(data);blobs[r+'/'+name]=dict(file=path,sha256=digest_bytes(data),dtype='float16',
                shape=[1,256 if name=='input_long' else 64,records[r]['hidden']])
    plan=dict(version=VERSION,DIAGNOSTIC_ONLY=True,TEST_FIXTURE_ONLY=fixture,source=current_source,
        historical_pair=dict(dataset='qasper',adapter_id='Adapter_A',user_id='User_01',
            requests=[dict(request_id=t['request_id'],token_ids=t['token_ids']) for t in cfg['block']['tasks']]),
        historical_run_dir=str(Path(run).resolve()),historical_review_dir=str(Path(review).resolve()),base_prepared=str(Path(base_prepared).resolve()),
        historical_source_commit=cfg['source_commit'],historical_run_seal=history['run_seal'],historical_review_seal=history['review_seal'],
        historical_plan_seal=history['plan_seal'],historical_source=history['provenance'],historical_backend=history['backend'],
        historical_gpu=Evidence(run).json('FULL_RECOMPUTE/gpu_identity.json'),model_identity=pm['identity'],
        base_manifest_sha256=file_sha(Path(base_prepared)/'r3a_prepare_manifest.json'),roles=records,blobs=blobs,
        repetitions=repetitions,warmups=2,schedule=schedule(repetitions),
        backend_policy='EXPLICIT_REPLAY_RECORDED_CORE_SETTINGS; initial/effective values recorded; no alternate backend sweep',
        bounds=dict(evidence_bytes=512*2**20,max_files=256,gpu_estimate_bytes=5400000000+5*2**30,
            cpu_reserve_bytes=32*2**30,runtime_seconds=900),precision_conditions=['NATIVE_FP16','DERIVED_FP32'],
        scope='No LoRA/TCP/cache/attention/generation; no performance claims')
    plan['experiment_id']=digest(plan);write(out/'i5r2bs1_plan.json',plan);seal(out);return plan


def validate_plan(path,*,check_source=True):
    ev=Evidence(path,max_input_bytes=128*2**20);ev.verify();p=ev.json('i5r2bs1_plan.json');q=dict(p);eid=q.pop('experiment_id')
    require(digest(q)==eid and p['version']==VERSION and p['DIAGNOSTIC_ONLY'],'S1 plan identity differs')
    require(p['precision_conditions']==['NATIVE_FP16','DERIVED_FP32'] and p['schedule']==schedule(p['repetitions'],p['warmups']) and set(p['roles'])==set('qkv') and p['bounds']==
        dict(evidence_bytes=512*2**20,max_files=256,gpu_estimate_bytes=5400000000+5*2**30,cpu_reserve_bytes=32*2**30,runtime_seconds=900),
        'S1 configuration bounds differ')
    if check_source:require(p['source']==source_identity(p['TEST_FIXTURE_ONLY']),'S1 pinned execution source changed')
    for record in p['blobs'].values():ev.tensor('',record)
    for r,info in p['roles'].items():
        require(info['layouts'] in (['CONTIGUOUS'],['CONTIGUOUS','HISTORICAL_STRIDED']),'unknown layout control')
        require(ev.tensor('',p['blobs'][r+'/source_projection_input'])==ev.tensor('',p['blobs'][r+'/target_projection_input'])==
            ev.tensor('',p['blobs'][r+'/input_long'])[:64*info['hidden']*2],'sealed prefix bytes differ')
        require(info['suffix_method']==('RECORDED_FULL_INPUT' if info['full_suffix_available'] else 'SYNTHETIC_XORSHIFT32_DYADIC_V1'),'suffix provenance label differs')
        if not info['full_suffix_available']:
            require(ev.tensor('',p['blobs'][r+'/input_long'])==suffix(ev.tensor('',p['blobs'][r+'/source_projection_input']),info['hidden'],info['suffix_seed']),
                'deterministic suffix values differ')
    return p


def verify_history_inputs(plan,evidence,history):
    """Bind prepared operand/reference bytes back to their original sealed captures."""
    for role in 'qkv':
        for name in ('source_projection_input','target_projection_input','source_base','target_base'):
            require(evidence.tensor('',plan['blobs'][role+'/'+name])==history['payloads'][role+'/'+name],'prepared/historical tensor values differ')
        if plan['roles'][role]['full_suffix_available']:
            require(evidence.tensor('',plan['blobs'][role+'/input_long'])==history['payloads'].get(role+'/full_input'),'recorded full input differs')
