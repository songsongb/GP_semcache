"""Sealed CPU planning/simulation for an explicitly approved I5R-4 campaign."""
from copy import deepcopy
from collections import Counter
from pathlib import Path
import json,math,os,random,shutil,subprocess

from .communication import require,digest,digest_bytes
from .r3a_artifacts import ROOT,read,write,sha,THRESHOLDS
from .r5_i5_evidence import fresh,seal,sources
from .r5_i5r1_plan import source_identity
from .r5_i5r2_tensors import Evidence
from .r5_i5r3b_plan import history,shared_pairs,original_link,archived_source
from .r5_i5r3a import inputs
from .r5_i5_workload import prepare_prompt,routing,requests,resources,CONDITIONS
from .r5_multi_artifacts import validate,sharing
from .r5_i5r3b_transport import preflight as transport
from .r5_i5r4_policy import Engine,POLICIES,CAPACITY,parameters

SCHEMA='I5R4_CAMPAIGN_V1'
HISTORICAL_BE='f7157c9b8b97fdce56f87dcbc3566cb677404bb8'
NAMESPACE=ROOT/'results/r5/i5r/controlled_hit/i5r4'
GPU_SECONDS=68400

def output(path,protected=()):
    p=Path(path).resolve()
    require(not p.is_relative_to(ROOT) or p.is_relative_to(NAMESPACE),'I4 fresh namespace required')
    require(not any((x/'i5_output_manifest.json').exists() or (x/'i4_bundle_manifest.json').exists() for x in p.parents),'cannot write within sealed evidence')
    return fresh(path,protected)

def reviewed(commit):
    actual=source_identity()
    require(commit==actual['commit'] and commit!=HISTORICAL_BE,'pin a newly reviewed I4 commit; historical BE SHA is read-only')
    require('src/semcache/edgelora/r5_i5r4_policy.py' in actual['files'],'I4 implementation absent from source inventory')
    test='tests/test_r5_i5r4.py';blob=subprocess.run(['git','show',commit+':'+test],cwd=ROOT,capture_output=True)
    require(blob.returncode==0 and digest_bytes(blob.stdout)==sha(ROOT/test),'commit reviewed CPU validation tests too')
    return actual

def baseline(be_experiment,be_run,be_audit,be_verification,nf1_audit):
    # Existing NF1 checks source Git blobs, archived independent audit bindings,
    # real completed sources, physical HITs, tensors and selected original data.
    from .r5_i5r_nf1 import read_inputs,A,C,S,IDS
    d=read_inputs(be_experiment,be_run,be_audit,be_verification,HISTORICAL_BE)
    nf=Evidence(nf1_audit,max_input_bytes=64*2**20);nr=nf.verify();nd=nf.json('nf1_decision.json')
    require(nd['status']=='FORENSICS_COMPLETE' and nd['CPU_ONLY'] and nd['source_cache_integrity_reproduced'] and nf.json('nf1_evidence_availability.json')['input_inventories']==d['receipts'],'completed NF1 binding required')
    archived_source(nf.json('nf1_source_provenance.json')['analysis_source'])
    positives=[r for r in d['economics']['positive_target_savings_ms']]
    saving=max(0.,sum(r['savings_ms'] for r in positives)/len(positives))
    rows=[d['rows'][c,r] for c in (A,C) for r in IDS]
    rate=dict(prefill_seconds_per_token=max(r['prefill_ms']/r['input_tokens']/1000 for r in rows),
              decode_seconds_per_token=max((r['decode_ms']/max(1,r['generated_tokens']-1)/1000 for r in rows)),
              admission_seconds=max(r['cache_admission_eviction_ms']/1000 for r in rows),
              other_request_seconds=max(max(0,r['application_e2e_ms']-r['model_execution_e2e_ms']-r['cache_admission_eviction_ms'])/1000 for r in rows))
    return dict(paths=dict(be_experiment=str(Path(be_experiment).resolve()),be_run=str(Path(be_run).resolve()),be_audit=str(Path(be_audit).resolve()),be_verification=str(Path(be_verification).resolve()),nf1_audit=str(Path(nf1_audit).resolve())),
                inventories=dict(d['receipts'],nf1=nr),historical_source=d['provenance'],
                saving_ms=saving,admission_ms=d['rows'][C,S]['cache_admission_eviction_ms'],rates=rate,
                calibration_contexts=sorted({digest_bytes(t['prepared']['sample']['context'].encode()) for t in d['tasks'].values()}),
                quality_numerical_handoff=d['old']['verdicts'],economics=d['economics'],
                limitation='One context calibrates costs only, never a general reuse-probability model; expected reuse is a declared optimistic prior with explicit exploration. Admission interval includes instrumentation, not isolated GPU-copy cost.')

def choose(prompts,users,count,excluded):
    pairs=[p for p in shared_pairs(prompts,users) if p['context_sha256'] not in excluded and p['a']['route']['user_id']!=p['b']['route']['user_id']]
    picked=[];used=set(excluded);used_prefix=set();availability={a:dict(contexts=len({p['context_sha256'] for p in pairs if p['adapter']==a}),distinct_prefixes=len({digest(p['a']['token_ids'][:512]) for p in pairs if p['adapter']==a})) for a in ('Adapter_A','Adapter_B','Adapter_C')}
    for _ in range(count):
        options=[]
        for aid in ('Adapter_A','Adapter_B','Adapter_C'):
            groups={}
            for p in pairs:
                if p['adapter']==aid and p['context_sha256'] not in used:groups.setdefault(p['context_sha256'],p)
            distinct=[];prefixes=set(used_prefix)
            for g in sorted(groups.values(),key=lambda p:p['context_sha256']):
                h=digest(g['a']['token_ids'][:512])
                if h not in prefixes:distinct.append(g);prefixes.add(h)
            if len(distinct)>=3:options.append((len(distinct),aid,distinct[:3]))
        require(options,'INSUFFICIENT_DISJOINT_REAL_GROUPS: need three authorized cross-user contexts per trace; availability='+str(availability)+'; explicitly resize before sealing')
        _,aid,group=sorted(options,key=lambda x:(-x[0],x[1]))[0]
        used.update(p['context_sha256'] for p in group);used_prefix.update(digest(g['a']['token_ids'][:512]) for g in group);picked.append(group)
    for group in picked:
        aid=group[0]['adapter'];prefixes=[p['a']['token_ids'][:512] for p in group]
        require(len({digest(p) for p in prefixes})==3 and all(len(p)==512 for p in prefixes),'competing sources need distinct exact prefixes')
        negatives=[p for p in prompts if p['route']['adapter_id']==aid and digest_bytes(p['sample']['context'].encode()) not in used and all(p['token_ids'][:512]!=x for x in prefixes)]
        require(negatives,'genuine same-adapter mismatched-context negative missing')
        group[0]['negative']=sorted(negatives,key=lambda p:p['sample']['id'])[0]
        used.add(digest_bytes(group[0]['negative']['sample']['context'].encode()))
        foreign=[p for p in prompts if p['route']['adapter_id']!=aid and p['token_ids'][:512] in prefixes and len(p['token_ids'])>512 and digest_bytes(p['sample']['context'].encode()) in {g['context_sha256'] for g in group}]
        group[0]['foreign']=sorted(foreign,key=lambda p:p['sample']['id'])[0] if foreign else None
    return picked,availability

def build_trace(group,tok,special,seed,quality_tokens):
    src=[];targets=[]
    for i,pair in enumerate(group):
        pair_targets=[]
        for j,p in enumerate((pair['a'],pair['b'])):
            p=deepcopy(p);p.update(strict=False,max_new_tokens=quality_tokens)
            t=requests([p],track='NATURAL',prefix=None,offset=i*2+j,special=special,routing_seed=seed)[0];pair_targets.append(t)
        p=deepcopy(pair['a']);p.update(strict=False,max_new_tokens=1,source_semantic_text=tok.decode(p['token_ids'][:512],skip_special_tokens=False,clean_up_tokenization_spaces=False))
        s=requests([p],track='CONTROLLED',prefix=512,offset=i,special=special,routing_seed=seed)[0];s['role']='CONTROLLED_STANDALONE_SOURCE';src.append(s);targets.append(pair_targets)
    p=deepcopy(group[0]['negative']);p.update(strict=False,max_new_tokens=quality_tokens)
    neg=requests([p],track='NATURAL',prefix=None,offset=6,special=special,routing_seed=seed)[0];neg['role']='NEGATIVE_TARGET'
    # All arrivals are predeclared workload only. The Engine never receives this
    # list or its future context labels. Six genuine repeated negative requests
    # create request-index idle gaps, without synthetic filler or policy hints.
    trace=[src[0],*targets[0],src[1],targets[1][0],src[2],targets[2][0],*[neg]*6,targets[1][1],targets[2][1],src[0],*targets[0],neg]
    if group[0].get('foreign'):
        p=deepcopy(group[0]['foreign']);p.update(strict=False,max_new_tokens=quality_tokens)
        foreign=requests([p],track='NATURAL',prefix=None,offset=7,special=special,routing_seed=seed)[0];foreign['role']='CROSS_ADAPTER_NEGATIVE';trace.append(foreign)
    result=[]
    tag='I4'+digest([g['context_sha256'] for g in group])[:12]
    for i,t in enumerate(trace):
        t=deepcopy(t);t.update(request_id=f'{tag}R{i:03d}',request_sequence_number=i);result.append(t)
    return result

def check_trace(tasks,users,seed,sentinel=False):
    require(3<=len(tasks)<=20 and len({t['request_id'] for t in tasks})==len(tasks),'bounded unique request identities required')
    require([t['request_sequence_number'] for t in tasks]==list(range(len(tasks))),'nonsequential requests')
    priming=[t for t in tasks if not t['scored']];require(priming and all(len(t['token_ids'])==512 and t['policy']['max_new_tokens']==1 for t in priming),'source contract differs')
    if not sentinel:require(len({digest(t['token_ids']) for t in priming})==3,'three competing 512-token standalone sources required')
    for t in tasks:
        route=routing(t['prepared']['sample'],seed);require(all(t[k]==v for k,v in route.items()),'original routing modified')
        require(t['sample_id']==t['prepared']['sample']['id'] and t['group_id']==t['prepared']['sample']['group_id'],'original sample identity differs')
        require(t['token_ids']==(t['prepared']['token_ids'] if t['scored'] else t['prepared']['token_ids'][:512]),'token prefix/source slicing differs')
        require(len(t['token_ids'])+t['policy']['max_new_tokens']<=2048 and len(t['token_ids'])<=1280,'position/transport bound exceeded')
        if t['scored']:require(len(t['token_ids'])==1280,'fixed1280 target shape required; no synthetic suffix or short-target substitution')
        require(not any(k in t for k in ('forced_hit','selected_entry_id','cache_grant','future_uses')),'oracle/forced cache instruction prohibited')
        require(t['strict']==sentinel,'instrumentation segment mixing forbidden')
        if not t['scored']:continue
        positives=[s for s in priming if s['token_ids']==t['token_ids'][:512] and s['adapter_id']==t['adapter_id']]
        if t['role'] not in ('NEGATIVE_TARGET','CROSS_ADAPTER_NEGATIVE'):
            require(positives and any(s['request_sequence_number']<t['request_sequence_number'] and sharing(users,s['user_id'],t['user_id'],s['adapter_id'],t['adapter_id']) is None for s in positives),'no completed-before-target authorized source opportunity')
        else:require(not positives,'negative target would be an own-adapter exact-prefix opportunity')
    return dict(total_requests=len(tasks),scored_requests=sum(t['scored'] for t in tasks),unique_scored_samples=len({t['sample_id'] for t in tasks if t['scored']}),
                priming_requests=len(priming),distinct_source_prefixes=len({digest(t['token_ids']) for t in priming}),
                repeated_scored_observations=sum(t['scored'] for t in tasks)-len({t['sample_id'] for t in tasks if t['scored']}),
                independent_positive_contexts=len({digest_bytes(t['prepared']['sample']['context'].encode()) for t in priming}),
                foreign_negative='REAL_ROUTED_REQUEST' if any(t['role']=='CROSS_ADAPTER_NEGATIVE' for t in tasks) else 'NO_AVAILABLE_REAL_MATCH; existing unauthorized registry probes remain required')

def task_block(tasks,trace,rotation,segment):
    return dict(dataset='qasper',track='CONTROLLED',prefix_length=512,repetition=rotation,scale='I5R4',tasks=tasks,
                trace_unit=trace,segment=segment,cache_resets_per_condition=1,cache_lifetime_requests=len(tasks),physical_requests_per_condition=len(tasks),
                resources=resources(tasks,capacity=CAPACITY,evidence_budget=16*2**30))

def schedule(traces,sentinel,rotations,seed):
    rng=random.Random(seed);base=list(POLICIES);rng.shuffle(base);jobs=[]
    for rotation in range(rotations):
        for trace,ts in enumerate(traces):
            offset=trace%6
            order=base[offset:]+base[:offset]
            if rotation%2:order=list(reversed(order))
            for policy in order:
                tasks=deepcopy(ts if policy!='P5' else [t for t in ts if t['scored']])
                for i,t in enumerate(tasks):t['request_sequence_number']=i
                b=task_block(tasks,trace,rotation,'MEASUREMENT');b['block_id']=digest(b)
                jobs.append(dict(child_id=f'm_t{trace}_r{rotation}_{policy}',policy=policy,block=b))
    for policy in base:
        tasks=deepcopy(sentinel if policy!='P5' else [t for t in sentinel if t['scored']])
        for i,t in enumerate(tasks):t['request_sequence_number']=i
        b=task_block(tasks,0,0,'STRICT_SENTINEL');b['block_id']=digest(b);jobs.append(dict(child_id='s_'+policy,policy=policy,block=b))
    return jobs

def resource_plan(jobs,rates,safety=2.,startup_seconds=180):
    require(1.5<=safety<=4 and 120<=startup_seconds<=600,'conservative planning factors required')
    rows=[]
    for job in jobs:
        ts=job['block']['tasks'];r=job['block']['resources']
        seconds=startup_seconds+safety*(sum(len(t['token_ids'])*rates['prefill_seconds_per_token']+(t['policy']['max_new_tokens']-1)*rates['decode_seconds_per_token']+rates['admission_seconds']+rates['other_request_seconds'] for t in ts)+3*16*rates['prefill_seconds_per_token'])
        require(seconds<=10800,'individual bounded condition exceeds three hours')
        rows.append(dict(child_id=job['child_id'],duration_seconds=math.ceil(seconds),maximum_forwards=r['expected_maximum_forwards'],
                         evidence_budget_bytes=r['evidence_budget_bytes'],estimated_evidence_bytes=r['estimated_evidence_bytes'],
                         logit_bytes=sum(t['policy']['max_new_tokens'] for t in ts)*50272*4,strict_requests=sum(t['strict'] for t in ts)))
    children_seconds=sum(x['duration_seconds'] for x in rows);total=children_seconds+900;budget=sum(x['evidence_budget_bytes'] for x in rows)
    require(total<=GPU_SECONDS,'GPU campaign conservative bound exceeds 19h; explicitly reduce trace units/rotations before sealing; no silent dropping')
    gpu=max(j['block']['resources']['gpu_estimate_bytes'] for j in jobs);require(gpu<22*2**30,'GPU peak estimate exceeds 22GiB')
    host_peak=8*2**30+6*max(j['block']['resources']['expected_maximum_forwards']*2*32*(32768+j['block']['resources']['maximum_input_tokens']*100) for j in jobs)
    require(host_peak<56*2**30,'bounded legacy per-child CPU replay exceeds 56GiB estimate; explicitly resize token/request workload')
    estimated=sum(x['estimated_evidence_bytes'] for x in rows)
    # Transparent CPU planning assumptions, enforced again by allocation timeouts.
    verify_seconds=900+estimated/(128*2**20)+sum(x['maximum_forwards'] for x in rows)*.06
    require(verify_seconds<=3600,'estimated independent verification exceeds one hour; explicitly resize campaign')
    return dict(children=rows,model_loads=len(jobs),ES_process_launches=len(jobs),CPU_UD_process_launches=3*len(jobs),GPU_preflight_processes=1,total_worker_launches=4*len(jobs)+1,
                total_maximum_forwards=sum(x['maximum_forwards'] for x in rows),GPU_phase_seconds=total,GPU_phase_limit_seconds=GPU_SECONDS,
                GPU_children_seconds=children_seconds,provenance_preflight_reserve_seconds=900,estimated_CPU_verification_seconds=verify_seconds,
                CPU_analysis_assumptions='128MiB/s sealed-file read/hash plus .06s per forward receipt and 900s archive/input margin; must be checked against actual filesystem, timeout remains fail-closed',
                analysis_limit_seconds=10800,verification_limit_seconds=3600,report_limit_seconds=1200,allocation_seconds=85800,
                inspect_limit_seconds=900,timeout_kill_after_reserve_seconds=5*30,allocation_headroom_seconds=85800-GPU_SECONDS-10800-3600-1200-900-150,
                evidence_budget_bytes=budget,estimated_evidence_bytes=sum(x['estimated_evidence_bytes'] for x in rows),
                logit_bytes=sum(x['logit_bytes'] for x in rows),disk_reserve_bytes=32*2**30,required_free_disk_bytes=budget+32*2**30,
                GPU_peak_estimate_bytes=gpu,GPU_limit_bytes=22*2**30,host_RAM_required_bytes=64*2**30,host_available_reserve_bytes=32*2**30,CPU_audit_peak_estimate_bytes=host_peak,
                uncertainty='BE maximum observed per-token costs x pinned safety factor, no HIT speed credit; startup floor includes full fingerprint/session warmups. This is a conservative estimate, not a runtime guarantee. Timeout/incomplete is failure, never automatic retry.')

def prepare(output_dir,source_commit,plan_3a,audit_3a,verification_3a,expected_3a_commit,be_experiment,be_run,be_audit,be_verification,nf1_audit,
            trace_units=4,rotations=2,max_prompt_tokens=1280,quality_tokens=32,quality_margin_f1=2.,seed=42,
            expected_reuses=2.,retention_ms_per_mib_request=1.,ttl_idle_requests=4,exploration_every=3,runtime_safety_factor=2.,startup_seconds=180):
    require(2<=trace_units<=4 and 1<=rotations<=2,'predeclare two..four independent multi-source units and one..two rotations')
    require(max_prompt_tokens==1280 and 32<=quality_tokens<=64 and 0<=quality_margin_f1<=2,'fixed transport/quality contract required')
    src=reviewed(source_commit);be=baseline(be_experiment,be_run,be_audit,be_verification,nf1_audit)
    old,result,proof=history(plan_3a,audit_3a,verification_3a,expected_3a_commit)
    tok,users,prov,data,pm,m=inputs(old['registry_path'],old['multi_prepared'])
    from .r5_i5_preflight import readiness
    ready=readiness({'qasper':prov['qasper']['path']},base_prepared=m['base_prepared'],multi_prepared=old['multi_prepared'],count=1,budgets={'qasper':quality_tokens},input_limit=1280,allow_truncation=True)
    require(ready['status']=='READY_FOR_MANUAL_GPU_SMOKE','offline assets/dependencies not ready: '+json.dumps(ready['runtime']))
    prompts=[];exclusions=[]
    for s in data['qasper']:
        try:p=prepare_prompt(s,tok.tokenizer,2048,quality_tokens,allow_truncation=True,input_limit=max_prompt_tokens)
        except ValueError as exc:exclusions.append(dict(id=s['id'],reason=str(exc)));continue
        if len(p['token_ids'])!=1280:exclusions.append(dict(id=s['id'],reason='SHORTER_THAN_FIXED_1280_TARGET'));continue
        record=next(r for r in result['qasper']['samples'] if r['id']==s['id'])
        p.update(route=routing(s,old['seed']),original_3a=original_link(s,record));require(p['route']==record['route'],'original routing changed');prompts.append(p)
    groups,availability=choose(prompts,users,trace_units,set(be['calibration_contexts']))
    special={k:getattr(tok.tokenizer,k) for k in ('bos_token_id','eos_token_id','pad_token_id')}
    traces=[build_trace(g,tok.tokenizer,special,old['seed'],quality_tokens) for g in groups]
    require(len({t['request_id'] for ts in traces for t in ts})==sum(map(len,traces)),'trace request identity collision')
    checks=[check_trace(t,users,old['seed']) for t in traces]
    sentinel=deepcopy([traces[0][i] for i in (0,2,7)])
    for i,t in enumerate(sentinel):t.update(request_id=f'I4S{i:03d}',request_sequence_number=i,strict=True);t['policy']['max_new_tokens']=1 if not t['scored'] else 8
    check_trace(sentinel,users,old['seed'],True)
    jobs=schedule(traces,sentinel,rotations,seed);_,_,specs,_=validate(old['multi_prepared'])
    plans={j['child_id']:dict(blocks=[j['block']],conditions=[CONDITIONS[0] if j['policy']=='P5' else CONDITIONS[2]]) for j in jobs}
    wire=transport(plans,pm,specs,m['tokenizer'],users);rp=resource_plan(jobs,be['rates'],runtime_safety_factor,startup_seconds)
    dest=Path(output_dir).resolve();parent=next((p for p in (dest,*dest.parents) if p.exists()),None)
    require(parent and shutil.disk_usage(parent).free>=rp['required_free_disk_bytes'],'campaign evidence budget plus 32GiB disk reserve unavailable')
    params=parameters(be['saving_ms'],be['admission_ms'],expected_reuses=expected_reuses,retention_ms_per_mib_request=retention_ms_per_mib_request,ttl_idle_requests=ttl_idle_requests,exploration_every=exploration_every)
    p=dict(schema=SCHEMA,source=src,source_commit=source_commit,historical_BE_source=HISTORICAL_BE,baseline=be,
           input_3a=dict(paths=dict(plan_3a=str(Path(plan_3a).resolve()),audit_3a=str(Path(audit_3a).resolve()),verification_3a=str(Path(verification_3a).resolve())),expected_commit=expected_3a_commit,proof=proof),
           multi_prepared=old['multi_prepared'],multi_inventory=Evidence(old['multi_prepared']).verify('r5_output_manifest.json'),
           datasets={'qasper':dict(prov['qasper']['manifest'],prepared_path=prov['qasper']['path'],prepared_manifest_sha256=sha(Path(prov['qasper']['path'])/'i5_output_manifest.json'))},
           model_identity=pm['identity'],adapters=specs,registry=users,tokenizer_identity=m['tokenizer'],routing_seed=old['seed'],seed=seed,
           label='CONTROLLED_PROACTIVE_MULTI_SOURCE_POLICY_CAMPAIGN_TRANSPORT_LIMITED',natural_hit_status='NOT_EVALUATED',
           policy_parameters=params,policies=list(POLICIES),clusters=1,capacity_per_adapter_bytes=CAPACITY,max_entries=64,
           trace_units=trace_units,rotations=rotations,trace_checks=checks,availability=availability,exclusions=exclusions,jobs=jobs,
           independent_positive_contexts=3*trace_units,context_groups=[[g['context_sha256'] for g in unit] for unit in groups],
           resources=rp,resources_safety_factor=runtime_safety_factor,startup_seconds=startup_seconds,transport=wire,readiness=ready,max_prompt_tokens=1280,quality_tokens=quality_tokens,validation_test_sha256=sha(ROOT/'tests/test_r5_i5r4.py'),
           quality=dict(metric='Qasper QA_F1',scale='0..100',margin_f1_points=quality_margin_f1,confidence=.95,
                        inference='Paired context/trace cluster bootstrap, repeats and rotations collapsed first; fewer than 8 independent traces DESCRIPTIVE_ONLY',
                        minimum_independent_traces_for_confirmatory=8,policy_target_noninferiority_reference='P0 and P5 separately',
                        correctness_qualified_speedup='Requires frozen numerical and token/text gates plus independently supported quality and performance; small campaign NOT_ESTABLISHED'),
           numerical_thresholds=deepcopy(THRESHOLDS),preparation_disk_observation=dict(path=str(parent),free_bytes=shutil.disk_usage(parent).free),instrumentation=dict(measurement='SCALABLE_EVALUATION uniform',sentinel='STRICT_AUDIT separate empty ES/cache/model/UD; excluded from performance'),
           source_cost='All priming/reentry requests charged to source-inclusive span; source forward is never incremental admission cost in policy score',GPU_NOT_RUN=True)
    p['plan_id']=digest(p);protected=[*be['paths'].values(),*p['input_3a']['paths'].values(),old['multi_prepared'],*[x['path'] for x in prov.values()],*[x['manifest']['source_path'] for x in prov.values()]]
    out=output(output_dir,protected);write(out/'i4_plan.json',p);seal(out)
    return dict(status='SEALED_CPU_PREPARED_GPU_NOT_RUN',plan_id=p['plan_id'],resources=rp,trace_checks=checks)

def load(experiment,source_commit=None,recheck_inputs=True):
    require(recheck_inputs is True,'immutable input verification cannot be disabled')
    ev=Evidence(experiment,max_input_bytes=128*2**20);receipt=ev.verify();p=ev.json('i4_plan.json');q=dict(p);pid=q.pop('plan_id')
    require(p['schema']==SCHEMA and digest(q)==pid,'plan identity differs');require(p['source']==reviewed(source_commit or p['source_commit']),'reviewed implementation source differs')
    require(p['numerical_thresholds']==THRESHOLDS and p['capacity_per_adapter_bytes']==CAPACITY and p['max_entries']==64 and p['clusters']==1 and p['policies']==list(POLICIES),'frozen constraints differ')
    archived_source(p['source'])
    if recheck_inputs:
        b=p['baseline'];require(b==baseline(**b['paths']),'historical BE/NF1 evidence changed')
        i=p['input_3a'];old,rs3a,proof=history(**{dict(plan_3a='plan',audit_3a='audit',verification_3a='verification')[k]:v for k,v in i['paths'].items()},expected_commit=i['expected_commit'])
        require(proof==i['proof'],'3A evidence changed')
        m,pm,a,r=validate(p['multi_prepared']);require(Evidence(p['multi_prepared']).verify('r5_output_manifest.json')==p['multi_inventory'] and pm['identity']==p['model_identity'] and a==p['adapters'] and r==p['registry'] and m['tokenizer']==p['tokenizer_identity'],'adapter/model registry provenance changed')
        from .r5_i5_data import load_dataset
        for manifest in p['datasets'].values():
            _,dm=load_dataset(manifest['prepared_path']);require(all(manifest[k]==v for k,v in dm.items()) and sha(Path(manifest['prepared_path'])/'i5_output_manifest.json')==manifest['prepared_manifest_sha256'],'dataset changed')
        # Independently reconstruct selection/prompt truncation and every arrival
        # from immutable normalized data, rather than trusting sealed task labels.
        tok,users,_,data,_,_=inputs(old['registry_path'],old['multi_prepared']);prompts=[]
        for s in data['qasper']:
            try:v=prepare_prompt(s,tok.tokenizer,2048,p['quality_tokens'],allow_truncation=True,input_limit=1280)
            except ValueError:continue
            if len(v['token_ids'])!=1280:continue
            record=next(x for x in rs3a['qasper']['samples'] if x['id']==s['id']);v.update(route=routing(s,old['seed']),original_3a=original_link(s,record));prompts.append(v)
        groups,_=choose(prompts,users,p['trace_units'],set(p['baseline']['calibration_contexts']))
        special={k:getattr(tok.tokenizer,k) for k in ('bos_token_id','eos_token_id','pad_token_id')};traces=[build_trace(g,tok.tokenizer,special,old['seed'],p['quality_tokens']) for g in groups]
        sentinel=deepcopy([traces[0][n] for n in (0,2,7)])
        for n,t in enumerate(sentinel):t.update(request_id=f'I4S{n:03d}',request_sequence_number=n,strict=True);t['policy']['max_new_tokens']=1 if not t['scored'] else 8
        require(schedule(traces,sentinel,p['rotations'],p['seed'])==p['jobs'],'independent real-data workload reconstruction differs')
    for j in p['jobs']:
        b=j['block'];q=dict(b);bid=q.pop('block_id');require(digest(q)==bid,'child block identity differs')
        if j['policy']!='P5':check_trace(b['tasks'],p['registry'],p['routing_seed'],b['segment']=='STRICT_SENTINEL')
    require(resource_plan(p['jobs'],p['baseline']['rates'],p['resources_safety_factor'],p['startup_seconds'])==p['resources'],'runtime/resource budget differs')
    require(sha(ROOT/'tests/test_r5_i5r4.py')==p['validation_test_sha256'],'CPU invariant tests changed')
    return p,receipt

def simulate(tasks,policy,params,registry,capacity=CAPACITY):
    """Behavior oracle sees supplied arrivals; Engine sees one request at a time.
    Not TinyBERT, not a physical HIT measurement, and not a forecast oracle.
    """
    engines={};metadata={};rows=[]
    for t in tasks:
        aid=t['adapter_id'];e=engines.setdefault(aid,Engine(policy,params,capacity));stored=metadata.setdefault(aid,{})
        expired=e.begin(t['request_id'])
        for k in expired:e.remove(k);stored.pop(k)
        possible=[k for k,s in stored.items() if len(s['token_ids'])<len(t['token_ids']) and t['token_ids'][:len(s['token_ids'])]==s['token_ids'] and sharing(registry,s['user_id'],t['user_id'],aid,aid) is None]
        key=sorted(possible,key=lambda k:(-len(stored[k]['token_ids']),k))[0] if possible and policy!='P5' else None
        victims=[];admitted=None
        if key:e.hit(key)
        elif policy!='P5':
            size=len(t['token_ids'])*491520;token_hash=digest(t['token_ids']);d=e.admission(token_hash,size)
            if d['allowed']:
                victims=e.victims(size,token_hash)
                if token_hash in stored:e.remove(token_hash);stored.pop(token_hash)
                for k in victims:e.remove(k);stored.pop(k)
                e.admitted(token_hash,token_hash,size);stored[token_hash]=dict(token_ids=t['token_ids'],user_id=t['user_id']);admitted=token_hash
        rows.append(dict(request_id=t['request_id'],adapter_id=aid,expired=expired,victims=victims,simulated_prefix_opportunity=key,admitted=admitted,state=e.snapshot(),scope='CPU_BEHAVIOR_ORACLE_ONLY_NOT_RUNTIME_HIT'))
    return rows

def invariant_tests():
    # Run real tensor-owner/controller invariant tests, not simulator alone.
    import unittest,sys
    sys.path.insert(0,str(ROOT/'tests'))
    suite=unittest.defaultTestLoader.loadTestsFromNames(['test_r5_i5r4.PolicyOwnerTests','test_r5_i5r4.PolicyTests'])
    outcome=unittest.TestResult();suite.run(outcome)
    require(outcome.testsRun>0 and not outcome.failures and not outcome.errors and not outcome.skipped,'required no-Torch policy invariant suite failed')
    return dict(run=outcome.testsRun,passed=outcome.testsRun,failed=0,skipped=0)

def cpu_validate(experiment,output_dir):
    p,receipt=load(experiment);results={j['child_id']:simulate(j['block']['tasks'],j['policy'],p['policy_parameters'],p['registry']) for j in p['jobs']}
    tests=invariant_tests()
    out=output(output_dir,(experiment,p['multi_prepared'],*[x['prepared_path'] for x in p['datasets'].values()],*[x['source_path'] for x in p['datasets'].values()]))
    write(out/'i4_cpu_audit.json',dict(status='PASS',source=p['source'],plan_id=p['plan_id'],plan_inventory=receipt,simulations=results,
          policy_tests=tests,GPU_NOT_RUN=True,scope='Real CPU owner invariants plus arrival-only simulation; no actual semantic/GPU HIT guarantee'))
    seal(out);return dict(status='PASS',plan_id=p['plan_id'],GPU_NOT_RUN=True,tests=tests['run'])

def cpu_receipt(p,plan_receipt,path):
    e=Evidence(path,max_input_bytes=128*2**20);receipt=e.verify();a=e.json('i4_cpu_audit.json')
    require(a['status']=='PASS' and a['source']==p['source'] and a['plan_id']==p['plan_id'] and a['plan_inventory']==plan_receipt and a['GPU_NOT_RUN'],'CPU validation does not bind exact source/plan')
    require(a['policy_tests']==invariant_tests(),'CPU owner-invariant receipt differs from independent rerun')
    require(a['simulations']=={j['child_id']:simulate(j['block']['tasks'],j['policy'],p['policy_parameters'],p['registry']) for j in p['jobs']},'CPU simulations changed')
    return receipt

def inspect(experiment,cpu_audit,source_commit,require_cpu_validation=False):
    require(require_cpu_validation,'--require-cpu-validation required')
    p,r=load(experiment,source_commit);cpu=cpu_receipt(p,r,cpu_audit)
    return dict(status='PREAPPROVED_SEALED_CAMPAIGN_GPU_NOT_RUN',plan_id=p['plan_id'],source=p['source']['commit'],cpu_audit=cpu,
                resources=p['resources'],execution_order=[j['child_id'] for j in p['jobs']],trace_checks=p['trace_checks'],selection=[dict(trace=j['block']['trace_unit'],request_id=t['request_id'],sample_id=t['sample_id'],context_sha256=digest_bytes(t['prepared']['sample']['context'].encode()),user=t['user_id'],adapter=t['adapter_id'],input_tokens=len(t['token_ids']),prefix_sha256=digest(t['token_ids'][:512]),role=t['role']) for j in p['jobs'] if j['policy']=='P0' and j['block']['repetition']==0 for t in j['block']['tasks']],transport_maximum=p['transport']['maximum_header_bytes'])
