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

class ResourceBudgetError(ValueError):
    """Numerical rejection receipt; never a partially sealed experiment."""
    def __init__(self,report):
        self.report=report
        failures=report.get('failures',[])
        reason='; '.join(f"{x['metric']}: estimate={x['estimate']} bound={x['bound']} excess={x['excess']} {x['unit']}" for x in failures)
        super().__init__(reason or 'campaign resource preflight rejected')

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

def resource_plan(jobs,rates,safety=2.,startup_seconds=180,*,diagnostic_only=False):
    require(1.5<=safety<=4 and 120<=startup_seconds<=600,'conservative planning factors required')
    require(jobs and all(k in rates and type(rates[k]) in (int,float) and math.isfinite(rates[k]) and rates[k]>=0 for k in ('prefill_seconds_per_token','decode_seconds_per_token','admission_seconds','other_request_seconds')),'finite nonnegative historical rate envelope required')
    rows=[];failures=[]
    def limit(metric,estimate,bound,unit,child_id=None,exclusive=False):
        if estimate>bound or exclusive and estimate==bound:failures.append(dict(metric=metric,estimate=estimate,bound=bound,excess=estimate-bound,unit=unit,child_id=child_id,comparison='<' if exclusive else '<='))
    for job in jobs:
        ts=job['block']['tasks'];r=job['block']['resources']
        # Exact original estimator: no credited HITs or early EOS, same 2x
        # safety when requested. Decomposition exposes rather than discounts it.
        raw=dict(prefill=(sum(len(t['token_ids']) for t in ts)+3*16)*rates['prefill_seconds_per_token'],
                 decoding=sum(t['policy']['max_new_tokens']-1 for t in ts)*rates['decode_seconds_per_token'],
                 admission=len(ts)*rates['admission_seconds'],other_overhead=len(ts)*rates['other_request_seconds'])
        seconds=startup_seconds+safety*sum(raw.values());rounded=math.ceil(seconds)
        scaled=dict(model_startup=startup_seconds,**{k:safety*v for k,v in raw.items()},rounding=rounded-seconds)
        limit('child_duration_seconds',rounded,10800,'seconds',job['child_id'])
        rows.append(dict(child_id=job['child_id'],policy=job.get('policy','UNSPECIFIED'),segment=job['block'].get('segment','UNSPECIFIED'),
                         duration_seconds=rounded,duration_hours=rounded/3600,raw_components_seconds=raw,conservative_components_seconds=scaled,
                         safety_factor=safety,safety_factor_contribution_seconds=(safety-1)*sum(raw.values()),
                         scored_target_executions=sum(t['scored'] for t in ts),maximum_decode_steps=sum(t['policy']['max_new_tokens']-1 for t in ts),
                         maximum_forwards=r['expected_maximum_forwards'],
                         evidence_budget_bytes=r['evidence_budget_bytes'],estimated_evidence_bytes=r['estimated_evidence_bytes'],
                         logit_bytes=sum(t['policy']['max_new_tokens'] for t in ts)*50272*4,strict_requests=sum(t['strict'] for t in ts)))
    children_seconds=sum(x['duration_seconds'] for x in rows);total=children_seconds+900;budget=sum(x['evidence_budget_bytes'] for x in rows)
    limit('GPU_phase_seconds',total,GPU_SECONDS,'seconds')
    gpu=max(j['block']['resources']['gpu_estimate_bytes'] for j in jobs)
    limit('GPU_peak_estimate_bytes',gpu,22*2**30,'bytes',exclusive=True)
    host_peak=8*2**30+6*max(j['block']['resources']['expected_maximum_forwards']*2*32*(32768+j['block']['resources']['maximum_input_tokens']*100) for j in jobs)
    limit('CPU_audit_peak_estimate_bytes',host_peak,56*2**30,'bytes',exclusive=True)
    estimated=sum(x['estimated_evidence_bytes'] for x in rows)
    # Transparent CPU planning assumptions, enforced again by allocation timeouts.
    verify_seconds=900+estimated/(128*2**20)+sum(x['maximum_forwards'] for x in rows)*.06
    limit('estimated_CPU_verification_seconds',verify_seconds,3600,'seconds')
    totals={k:sum(x['conservative_components_seconds'][k] for x in rows) for k in ('model_startup','prefill','decoding','admission','other_overhead','rounding')};totals['provenance_preflight']=900
    by_policy={}
    for policy in (*POLICIES,'UNSPECIFIED'):
        xs=[x for x in rows if x['policy']==policy]
        if not xs:continue
        by_policy[policy]=dict(children=len(xs),duration_seconds=sum(x['duration_seconds'] for x in xs),
            duration_hours=sum(x['duration_seconds'] for x in xs)/3600,
            components_seconds={k:sum(x['conservative_components_seconds'][k] for x in xs) for k in ('model_startup','prefill','decoding','admission','other_overhead','rounding')})
    report=dict(children=rows,model_loads=len(jobs),ES_process_launches=len(jobs),CPU_UD_process_launches=3*len(jobs),GPU_preflight_processes=1,total_worker_launches=4*len(jobs)+1,
                estimator='UNCHANGED_BE_COMPONENT_MAXIMA_NO_HIT_OR_EOS_CREDIT',rates=deepcopy(rates),safety_factor=safety,startup_seconds=startup_seconds,
                components_seconds=totals,by_policy=by_policy,
                safety_factor_contribution_seconds=sum(x['safety_factor_contribution_seconds'] for x in rows),
                GPU_phase_hours=total/3600,GPU_bound_hours=GPU_SECONDS/3600,GPU_excess_seconds=max(0,total-GPU_SECONDS),GPU_headroom_seconds=GPU_SECONDS-total,
                dominant_contributing_factors=sorted([dict(component=k,seconds=v,hours=v/3600) for k,v in totals.items()],key=lambda x:(-x['seconds'],x['component'])),
                accepted=not failures,failures=failures,
                total_maximum_forwards=sum(x['maximum_forwards'] for x in rows),GPU_phase_seconds=total,GPU_phase_limit_seconds=GPU_SECONDS,
                GPU_children_seconds=children_seconds,provenance_preflight_reserve_seconds=900,estimated_CPU_verification_seconds=verify_seconds,
                CPU_analysis_assumptions='128MiB/s sealed-file read/hash plus .06s per forward receipt and 900s archive/input margin; must be checked against actual filesystem, timeout remains fail-closed',
                analysis_limit_seconds=10800,verification_limit_seconds=3600,report_limit_seconds=1200,allocation_seconds=85800,
                inspect_limit_seconds=900,timeout_kill_after_reserve_seconds=5*30,allocation_headroom_seconds=85800-GPU_SECONDS-10800-3600-1200-900-150,
                evidence_budget_bytes=budget,estimated_evidence_bytes=sum(x['estimated_evidence_bytes'] for x in rows),
                logit_bytes=sum(x['logit_bytes'] for x in rows),disk_reserve_bytes=32*2**30,required_free_disk_bytes=budget+32*2**30,
                GPU_peak_estimate_bytes=gpu,GPU_limit_bytes=22*2**30,host_RAM_required_bytes=64*2**30,host_available_reserve_bytes=32*2**30,CPU_audit_peak_estimate_bytes=host_peak,
                uncertainty='BE maximum observed per-token costs x pinned safety factor, no HIT speed credit; startup floor includes full fingerprint/session warmups. This is a conservative estimate, not a runtime guarantee. Timeout/incomplete is failure, never automatic retry.')
    report['pessimism_review']=[
        'Component maxima come from different archived rows; their sum is an envelope, not an observed joint request.',
        'The maximum admission interval is charged even to P5 and oversized target requests; this can overcharge, but has NOT been discounted.',
        'Strict historical instrumentation costs are retained. No unmeasured scalable-mode speed credit or guaranteed HIT/EOS is used.',
        '32-token decoding extrapolates an 8-token historical run. The declared factor is retained; neither kernels nor run-to-run variance are guaranteed.',
        'Original four-rate estimator is preserved, not represented as a formal runtime guarantee; actual stage timeouts remain fail-closed.']
    if failures and not diagnostic_only:raise ResourceBudgetError(report)
    return report

def compare_budgets(make_jobs,rates,safety,startup_seconds,available_disk_bytes,requested):
    """Diagnose only; never resize the selected campaign or write a plan."""
    cases={};profiles=list(dict.fromkeys([requested,(2,2),(2,1),(3,1)]))
    for units,rotations in profiles:
        name=f'{units}x{rotations}'
        try:
            jobs=make_jobs(units,rotations)
            rp=resource_plan(jobs,rates,safety,startup_seconds,diagnostic_only=True)
            failures=deepcopy(rp['failures'])
            if available_disk_bytes<rp['required_free_disk_bytes']:
                failures.append(dict(metric='required_free_disk_bytes',estimate=rp['required_free_disk_bytes'],bound=available_disk_bytes,
                                     excess=rp['required_free_disk_bytes']-available_disk_bytes,unit='bytes',child_id=None))
            cases[name]=dict(trace_units=units,rotations=rotations,accepted=not failures,failures=failures,resources=rp)
        except ValueError as exc:
            cases[name]=dict(trace_units=units,rotations=rotations,accepted=False,failures=[dict(metric='workload_or_transport_eligibility',estimate=None,bound=None,excess=None,unit='NOT_EVALUATED',reason=str(exc))])
    # Preserve two rotations when possible. If 2x2 cannot fit, the user asked to
    # prioritize 2x1; 3x1 is still evaluated and never silently substituted.
    recommendation=next((dict(trace_units=u,rotations=r) for u,r in ((2,2),(2,1),(3,1)) if cases[f'{u}x{r}']['accepted']),None)
    selected=cases[f'{requested[0]}x{requested[1]}']
    return dict(schema='I5R4_BUDGET_DIAGNOSTIC_V1',scope='CPU_ONLY_HISTORICAL_BE_ENVELOPE_NOT_A_BENCHMARK',
                requested=dict(trace_units=requested[0],rotations=requested[1]),accepted=selected['accepted'],failures=selected['failures'],
                available_disk_bytes=available_disk_bytes,configurations=cases,recommended_profile=recommendation,
                automatic_resize=False,stage_budgets_changed=False,plan_sealed=False,
                interpretation='Recommendation is conditional on verified inputs and declared bounds, not a runtime guarantee. Explicitly re-run prepare with chosen counts; no policy/task is dropped.')

def prepare(output_dir,source_commit,plan_3a,audit_3a,verification_3a,expected_3a_commit,be_experiment,be_run,be_audit,be_verification,nf1_audit,
            trace_units=4,rotations=2,max_prompt_tokens=1280,quality_tokens=32,quality_margin_f1=2.,seed=42,
            expected_reuses=2.,retention_ms_per_mib_request=1.,ttl_idle_requests=4,exploration_every=3,runtime_safety_factor=2.,startup_seconds=180,
            budget_only=False,disk_path=None):
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
    wire=transport(plans,pm,specs,m['tokenizer'],users)
    require(budget_only or output_dir is not None,'prepare requires a fresh output directory')
    require(budget_only or disk_path is None,'prepare checks the actual output filesystem; --disk-path is diagnostic-only')
    dest=Path(disk_path or output_dir or ROOT/'results').resolve();parent=next((p for p in (dest,*dest.parents) if p.exists()),None)
    require(parent is not None,'budget disk filesystem unavailable');free=shutil.disk_usage(parent).free
    def alternatives(units,repeat):
        if units==trace_units and repeat==rotations:return jobs
        gs,_=choose(prompts,users,units,set(be['calibration_contexts']))
        ats=[build_trace(g,tok.tokenizer,special,old['seed'],quality_tokens) for g in gs]
        for ts in ats:check_trace(ts,users,old['seed'])
        ss=deepcopy([ats[0][i] for i in (0,2,7)])
        for i,t in enumerate(ss):t.update(request_id=f'I4S{i:03d}',request_sequence_number=i,strict=True);t['policy']['max_new_tokens']=1 if not t['scored'] else 8
        check_trace(ss,users,old['seed'],True)
        js=schedule(ats,ss,repeat,seed)
        ps={j['child_id']:dict(blocks=[j['block']],conditions=[CONDITIONS[0] if j['policy']=='P5' else CONDITIONS[2]]) for j in js}
        transport(ps,pm,specs,m['tokenizer'],users)
        return js
    report=compare_budgets(alternatives,be['rates'],runtime_safety_factor,startup_seconds,free,(trace_units,rotations))
    report.update(source=src,baseline_inventories=be['inventories'],historical_rate_source=be['historical_source']['recorded_source_commit'],quality_tokens=quality_tokens)
    if budget_only:
        if report['recommended_profile'] is None:raise ResourceBudgetError(report)
        return dict(status='CPU_BUDGET_DIAGNOSTIC_COMPLETE',**report)
    if not report['accepted']:raise ResourceBudgetError(report)
    # Hard guard remains enabled for preparation and load/run/analysis. The
    # diagnostic-only calculation above cannot seal or authorize execution.
    rp=resource_plan(jobs,be['rates'],runtime_safety_factor,startup_seconds)
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
           resources=rp,budget_comparison=report,resources_safety_factor=runtime_safety_factor,startup_seconds=startup_seconds,transport=wire,readiness=ready,max_prompt_tokens=1280,quality_tokens=quality_tokens,validation_test_sha256=sha(ROOT/'tests/test_r5_i5r4.py'),
           quality=dict(metric='Qasper QA_F1',scale='0..100',margin_f1_points=quality_margin_f1,confidence=.95,
                        inference='Paired context/trace cluster bootstrap, repeats and rotations collapsed first; fewer than 8 independent traces DESCRIPTIVE_ONLY',
                        minimum_independent_traces_for_confirmatory=8,policy_target_noninferiority_reference='P0 and P5 separately',
                        correctness_qualified_speedup='Requires frozen numerical and token/text gates plus independently supported quality and performance; small campaign NOT_ESTABLISHED'),
           numerical_thresholds=deepcopy(THRESHOLDS),preparation_disk_observation=dict(path=str(parent),free_bytes=shutil.disk_usage(parent).free),instrumentation=dict(measurement='SCALABLE_EVALUATION uniform',sentinel='STRICT_AUDIT separate empty ES/cache/model/UD; excluded from performance'),
           source_cost='All priming/reentry requests charged to source-inclusive span; source forward is never incremental admission cost in policy score',GPU_NOT_RUN=True)
    p['plan_id']=digest(p);protected=[*be['paths'].values(),*p['input_3a']['paths'].values(),old['multi_prepared'],*[x['path'] for x in prov.values()],*[x['manifest']['source_path'] for x in prov.values()]]
    out=output(output_dir,protected);write(out/'i4_plan.json',p);seal(out)
    return dict(status='SEALED_CPU_PREPARED_GPU_NOT_RUN',plan_id=p['plan_id'],resources=rp,trace_checks=checks)

def budget(**kwargs):
    return prepare(output_dir=None,budget_only=True,**kwargs)

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
    engines={};metadata={};completed={};rows=[]
    for t in tasks:
        aid=t['adapter_id'];e=engines.setdefault(aid,Engine(policy,params,capacity));stored=metadata.setdefault(aid,{})
        previous=completed.setdefault(aid,{})
        def removal(key):return dict(entry_id=key,source_request_id=stored[key]['source_request_id'],bytes=e.entries[key]['bytes'])
        expired=e.begin(t['request_id'])
        expirations=[removal(k) for k in expired]
        for k in expired:e.remove(k);stored.pop(k)
        # Reporting oracle only: all previously completed, capacity-eligible
        # standalone requests, including denied/evicted sources. It is NEVER
        # passed to Engine, nor evidence of actual completion or a physical HIT.
        completed_matches=[k for k,s in previous.items() if len(s['token_ids'])<len(t['token_ids']) and t['token_ids'][:len(s['token_ids'])]==s['token_ids'] and sharing(registry,s['user_id'],t['user_id'],aid,aid) is None]
        possible=[k for k,s in stored.items() if len(s['token_ids'])<len(t['token_ids']) and t['token_ids'][:len(s['token_ids'])]==s['token_ids'] and sharing(registry,s['user_id'],t['user_id'],aid,aid) is None]
        key=sorted(possible,key=lambda k:(-len(stored[k]['token_ids']),k))[0] if possible and policy!='P5' else None
        victims=[];victim_details=[];admitted=None;d=None
        if key:e.hit(key)
        elif policy!='P5':
            size=len(t['token_ids'])*491520;token_hash=digest(t['token_ids']);d=e.admission(token_hash,size)
            if d['allowed']:
                victims=e.victims(size,token_hash)
                victim_details=[removal(k) for k in victims]
                if token_hash in stored:e.remove(token_hash);stored.pop(token_hash)
                for k in victims:e.remove(k);stored.pop(k)
                e.admitted(token_hash,token_hash,size);stored[token_hash]=dict(token_ids=t['token_ids'],user_id=t['user_id'],source_request_id=t['request_id']);admitted=token_hash
        if policy!='P5' and key is None and len(t['token_ids'])*491520<=capacity and len(t['token_ids'])<2048:
            previous[digest(t['token_ids'])]=dict(token_ids=t['token_ids'],user_id=t['user_id'])
        rows.append(dict(request_id=t['request_id'],adapter_id=aid,input_tokens=len(t['token_ids']),scored=t['scored'],
                         expired=expired,expiration_details=expirations,victims=victims,capacity_eviction_details=victim_details,
                         admission_decision=d,exploration_admission=policy in ('P1','P4') and d is not None and d['allowed'] and d['exploration'],
                         completed_source_prefix_opportunities=sorted(completed_matches),
                         simulated_prefix_opportunity=key,admitted=admitted,state=e.snapshot(),scope='CPU_BEHAVIOR_ORACLE_ONLY_NOT_RUNTIME_HIT'))
    return rows

def discrimination(p,simulations):
    """Read-only decision/coverage comparison, never tuning or a physical HIT rate.

    Scores alone, policy names and unused exploration flags do not discriminate
    policies. Compare observable simulated actions on identical arrivals. P5's
    different source traffic is never labeled an admission/TTL/eviction ablation.
    """
    groups={};children={}
    for job in p['jobs']:
        b=job['block'];key=(b['segment'],b['trace_unit'],b['repetition']);group=groups.setdefault(key,{})
        require(job['policy'] not in group,'duplicate policy discrimination unit');group[job['policy']]=job['child_id']
        rs=simulations[job['child_id']];require([r['request_id'] for r in rs]==[t['request_id'] for t in b['tasks']],'simulation/request coverage differs')
        evaluated=[r for r in rs if r['admission_decision'] is not None];eligible=[r for r in evaluated if r['input_tokens']==512]
        def counts(xs):return dict(candidates=len(xs),admissions=sum(r['admitted'] is not None for r in xs),denials=sum(not r['admission_decision']['allowed'] for r in xs),exploration_admissions=sum(r['exploration_admission'] for r in xs))
        opportunities=[r for r in rs if r['completed_source_prefix_opportunities']]
        children[job['child_id']]=dict(policy=job['policy'],segment=b['segment'],trace_unit=b['trace_unit'],rotation=b['repetition'],
            admissions=dict(all_candidates=counts(evaluated),eligible_512=counts(eligible),
                decisions=[dict(request_id=r['request_id'],input_tokens=r['input_tokens'],admitted=r['admitted'],exploration_admission=r['exploration_admission'],decision=r['admission_decision']) for r in evaluated]),
            TTL_expirations=[dict(at_request_id=r['request_id'],**x) for r in rs for x in r['expiration_details']],
            capacity_eviction_victims=[dict(at_request_id=r['request_id'],**x) for r in rs for x in r['capacity_eviction_details']],
            exact_prefix_opportunities=dict(denominator='Requests with a previously completed capacity-eligible exact source under the CPU sequential-completion assumption',
                requests=len(opportunities),retained=sum(r['simulated_prefix_opportunity'] is not None for r in opportunities),
                lost=sum(r['simulated_prefix_opportunity'] is None for r in opportunities),
                retained_request_ids=[r['request_id'] for r in rs if r['simulated_prefix_opportunity'] is not None],
                lost_request_ids=[r['request_id'] for r in opportunities if r['simulated_prefix_opportunity'] is None]))
    pairwise=[]
    def actions(r):return dict(admitted=r['admitted'],expired=r['expired'],victims=r['victims'],retained_prefix=r['simulated_prefix_opportunity'])
    for key,group in sorted(groups.items()):
        require(set(group)==set(POLICIES),'incomplete policy discrimination unit')
        for i,a in enumerate(POLICIES):
            for b in POLICIES[i+1:]:
                item=dict(segment=key[0],trace_unit=key[1],rotation=key[2],reference=a,policy=b)
                if b=='P5':
                    pairwise.append(dict(item,status='NOT_COMPARABLE_DIFFERENT_SOURCE_TRAFFIC',reason='Target-only economic control, not an identical-arrival policy ablation'));continue
                ar,br=simulations[group[a]],simulations[group[b]]
                require([r['request_id'] for r in ar]==[r['request_id'] for r in br],'policy arrivals differ')
                changed=[]
                for x,y in zip(ar,br):
                    ax,by=actions(x),actions(y);fields=[f for f in ax if ax[f]!=by[f]]
                    if fields:changed.append(dict(request_id=x['request_id'],fields=fields,reference=ax,candidate=by))
                lost=[x['request_id'] for x,y in zip(ar,br) if x['simulated_prefix_opportunity'] is not None and y['simulated_prefix_opportunity'] is None]
                gained=[x['request_id'] for x,y in zip(ar,br) if x['simulated_prefix_opportunity'] is None and y['simulated_prefix_opportunity'] is not None]
                pairwise.append(dict(item,status='DISCRIMINATED_BY_WORKLOAD' if changed else 'NOT_DISCRIMINATED_BY_WORKLOAD',changed_request_count=len(changed),changes=changed,exact_prefix_opportunities_lost_vs_reference=lost,exact_prefix_opportunities_gained_vs_reference=gained))
    summary={};ablations={}
    for policy in POLICIES:
        cs=[c for c in children.values() if c['policy']==policy and c['segment']=='MEASUREMENT']
        def totals(name):return {k:sum(c['admissions'][name][k] for c in cs) for k in ('candidates','admissions','denials','exploration_admissions')}
        summary[policy]=dict(measurement_child_runs=len(cs),trace_units=len({c['trace_unit'] for c in cs}),rotations=len({c['rotation'] for c in cs}),admission_decisions=totals('all_candidates'),eligible_512_admission_decisions=totals('eligible_512'),
            TTL_expirations=sum(len(c['TTL_expirations']) for c in cs),capacity_evictions=sum(len(c['capacity_eviction_victims']) for c in cs),
            exact_prefix_opportunities={k:sum(c['exact_prefix_opportunities'][k] for c in cs) for k in ('requests','retained','lost')})
        if policy in ('P0','P5'):continue
        ps=[x for x in pairwise if x['segment']=='MEASUREMENT' and x['reference']=='P0' and x['policy']==policy]
        differences=sum(x['changed_request_count'] for x in ps)
        relevant={'P1':('admitted',),'P2':('expired',),'P3':('victims',),'P4':('admitted','expired','victims','retained_prefix')}[policy]
        mechanism=sum(any(k in c['fields'] for k in relevant) for x in ps for c in x['changes'])
        ablations[policy]=dict(status='DISCRIMINATED_BY_WORKLOAD' if differences else 'NOT_DISCRIMINATED_BY_WORKLOAD',
            changed_requests_vs_P0=differences,mechanism_exercised=mechanism>0,mechanism_changed_requests=mechanism,
            interpretation='Simulated action difference, not benefit, runtime semantic retrieval or physical reuse')
    return dict(schema='I5R4_CPU_DISCRIMINATION_V1',scope='STATIC_ONLY',physical_HIT_status='NOT_EVALUATED',
                parameters=deepcopy(p['policy_parameters']),policy_summary=summary,ablations=ablations,
                not_discriminated_ablations=[policy for policy,a in ablations.items() if a['status']=='NOT_DISCRIMINATED_BY_WORKLOAD'],children=children,pairwise=pairwise,
                limitations=['Repeated requests/rotations are dependent observations; counts are decisions, not independent samples.',
                            'Previously completed sources are a CPU trace assumption, not measured runtime completion/admission.',
                            'Sentinels are reported separately and excluded from ablation discrimination summaries.',
                            'Identical decisions are reported honestly; no automatic parameter/workload adjustment.'])

def invariant_tests():
    # Run real tensor-owner/controller invariant tests, not simulator alone.
    import unittest,sys
    sys.path.insert(0,str(ROOT/'tests'))
    suite=unittest.defaultTestLoader.loadTestsFromNames(['test_r5_i5r4.PolicyOwnerTests','test_r5_i5r4.PolicyTests','test_r5_i5r4.DiscriminationTests'])
    outcome=unittest.TestResult();suite.run(outcome)
    require(outcome.testsRun>0 and not outcome.failures and not outcome.errors and not outcome.skipped,'required no-Torch policy invariant suite failed')
    return dict(run=outcome.testsRun,passed=outcome.testsRun,failed=0,skipped=0)

def cpu_validate(experiment,output_dir):
    p,receipt=load(experiment);results={j['child_id']:simulate(j['block']['tasks'],j['policy'],p['policy_parameters'],p['registry']) for j in p['jobs']}
    tests=invariant_tests()
    report=discrimination(p,results)
    out=output(output_dir,(experiment,p['multi_prepared'],*[x['prepared_path'] for x in p['datasets'].values()],*[x['source_path'] for x in p['datasets'].values()]))
    write(out/'i4_policy_discrimination.json',report)
    write(out/'i4_cpu_audit.json',dict(status='PASS',source=p['source'],plan_id=p['plan_id'],plan_inventory=receipt,simulations=results,policy_discrimination_sha256=digest(report),
          policy_tests=tests,GPU_NOT_RUN=True,scope='Real CPU owner invariants plus arrival-only simulation; no actual semantic/GPU HIT guarantee'))
    seal(out);return dict(status='PASS',plan_id=p['plan_id'],GPU_NOT_RUN=True,tests=tests['run'],policy_discrimination=report)

def cpu_receipt(p,plan_receipt,path):
    e=Evidence(path,max_input_bytes=128*2**20);receipt=e.verify();a=e.json('i4_cpu_audit.json')
    require(a['status']=='PASS' and a['source']==p['source'] and a['plan_id']==p['plan_id'] and a['plan_inventory']==plan_receipt and a['GPU_NOT_RUN'],'CPU validation does not bind exact source/plan')
    require(a['policy_tests']==invariant_tests(),'CPU owner-invariant receipt differs from independent rerun')
    expected={j['child_id']:simulate(j['block']['tasks'],j['policy'],p['policy_parameters'],p['registry']) for j in p['jobs']}
    require(a['simulations']==expected,'CPU simulations changed')
    report=discrimination(p,expected)
    require(a['policy_discrimination_sha256']==digest(report) and e.json('i4_policy_discrimination.json')==report,'CPU policy discrimination changed')
    return receipt

def inspect(experiment,cpu_audit,source_commit,require_cpu_validation=False):
    require(require_cpu_validation,'--require-cpu-validation required')
    p,r=load(experiment,source_commit);cpu=cpu_receipt(p,r,cpu_audit)
    return dict(status='PREAPPROVED_SEALED_CAMPAIGN_GPU_NOT_RUN',plan_id=p['plan_id'],source=p['source']['commit'],cpu_audit=cpu,
                policy_discrimination=Evidence(cpu_audit).json('i4_policy_discrimination.json'),
                resources=p['resources'],execution_order=[j['child_id'] for j in p['jobs']],trace_checks=p['trace_checks'],selection=[dict(trace=j['block']['trace_unit'],request_id=t['request_id'],sample_id=t['sample_id'],context_sha256=digest_bytes(t['prepared']['sample']['context'].encode()),user=t['user_id'],adapter=t['adapter_id'],input_tokens=len(t['token_ids']),prefix_sha256=digest(t['token_ids'][:512]),role=t['role']) for j in p['jobs'] if j['policy']=='P0' and j['block']['repetition']==0 for t in j['block']['tasks']],transport_maximum=p['transport']['maximum_header_bytes'])
