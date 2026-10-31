"""Independent block replay, paired fidelity, negative controls and cluster bootstrap."""
from collections import defaultdict
import math
from pathlib import Path
import random
import sqlite3
import statistics

from .communication import digest,digest_bytes,require
from .r3a_artifacts import read,unpack,write,check_parity
from .r3a_evidence import jsonl
from .r4c_metrics import numerical
from .r5_analysis import audit as audit_r5
from .r5_perf_evidence import RESERVE
from .r5_perf_schedule import validate_schedule


def accounting_audit(root):
    root=Path(root).resolve()
    with sqlite3.connect(f'file:{root}/accounting/ledger.sqlite?mode=ro',uri=True) as c:
        total,count,failed,budget=c.execute('SELECT * FROM meta').fetchone()
        expected={p:(s,h) for p,s,h in c.execute('SELECT * FROM files')}
    actual={}
    for p in root.rglob('*'):
        require(not p.is_symlink(),'evidence symlink')
        if p.is_file() and not p.is_relative_to(root/'accounting'):
            b=p.read_bytes();actual[str(p.relative_to(root))]=(len(b),digest_bytes(b))
    require(actual==expected and failed is None,'independent accounting inventory/hash mismatch')
    managed=list((root/'accounting').iterdir())
    require(all(p.name=='ledger.sqlite' for p in managed) and sum(p.stat().st_size for p in managed)<=RESERVE,'accounting namespace differs')
    require(count==len(actual) and total==sum(s for s,h in actual.values()) and total+RESERVE<=budget,'independent accounting occupancy/budget differs')
    return dict(status='PASS',data_files=count,data_bytes=total,reserved_accounting_bytes=RESERVE,budget_bytes=budget)


def describe(xs):
    return dict(n=len(xs),mean=statistics.mean(xs) if xs else None,median=statistics.median(xs) if xs else None,
        standard_deviation=statistics.stdev(xs) if len(xs)>1 else None)


def bootstrap(xs,seed=42,resamples=2000):
    result=describe(xs)
    if len(xs)<2:return dict(result,ci95=None,median_ci95=None,status='INSUFFICIENT_INDEPENDENT_REPETITIONS')
    rng=random.Random(seed);samples=[rng.choices(xs,k=len(xs)) for _ in range(resamples)]
    means=sorted(statistics.mean(x) for x in samples);medians=sorted(statistics.median(x) for x in samples)
    return dict(result,ci95=[means[int(.025*(resamples-1))],means[int(.975*(resamples-1))]],
        median_ci95=[medians[int(.025*(resamples-1))],medians[int(.975*(resamples-1))]],
        status='EXPLORATORY_CLUSTER_BOOTSTRAP' if len(xs)<8 else 'CLUSTER_BOOTSTRAP',resamples=resamples)


def trend(points):
    if len(points)<2:return dict(n=len(points),slope=None,correlation=None)
    x,y=zip(*points);mx=statistics.mean(x);my=statistics.mean(y)
    xx=sum((v-mx)**2 for v in x);yy=sum((v-my)**2 for v in y);xy=sum((a-mx)*(b-my) for a,b in points)
    return dict(n=len(points),slope=xy/xx if xx else None,correlation=xy/math.sqrt(xx*yy) if xx and yy else None,
        interpretation='descriptive association, not causal attribution')


def flags():
    return dict(R5_I3P_TIMING_ATTRIBUTION_PASS=False,R5_I3P_EVIDENCE_ACCOUNTING_PASS=False,R5_I3P_STATE_ISOLATION_PASS=False,
        R5_I3P_ORDER_BALANCED_PASS=False,R5_I3P_INDEPENDENT_AUDIT_PASS=False,R5_I3P_GPU_EXECUTION_PASS=False,
        R5_I3P_PERFORMANCE_QUALIFIED=False,STATISTICALLY_SUPPORTED_TTFT_REDUCTION=False,STATISTICALLY_SUPPORTED_E2E_REDUCTION=False,
        NATIVE_PEFT_STRICT_PARITY_PASS=False,STRICT_NUMERICAL_FIDELITY_PASS=False,correctness_qualified_speedup=False)


def analyze(root,*,sealed=True):
    root=Path(root)
    if sealed:
        from .r5_evidence import verify
        verify(root)
    config=read(root/'r5_perf_config.json');order=validate_schedule(read(root/'r5_perf_order.json'))
    decision=flags();blocks=[];identities=[];sessions=[];pids=[];data={};negative=[];pairs=[];perstep=[];warmup_reference=None
    actual_order=read(root/'r5_perf_execution_order.json')
    require(len(actual_order)==len(order['blocks']),'incomplete benchmark blocks')
    for sequence,task in enumerate(order['blocks']):
        bid=task['block_id'];path=root/'blocks'/bid;life=read(path/'es/lifecycle.json');plan=read(path/'es/es_plan.json')
        require(actual_order[sequence]['block_id']==bid and actual_order[sequence]['condition']==task['condition'] and
            actual_order[sequence]['start_ns']<actual_order[sequence]['end_ns'] and
            (sequence==0 or actual_order[sequence-1]['end_ns']<=actual_order[sequence]['start_ns']),'actual block order differs')
        require(life['returncode']==0 and not life['forced_termination'] and not life['pipe_errors'],'unclean ES lifecycle')
        require(all(plan[k]==config[k] for k in ('evidence_profile','generation_budget','generation_mode','warmup_requests','seed','legacy_scan','reconcile_every','semantic_revision','fixture')),'condition settings differ')
        require(plan['sources']==read(root/'r5_perf_sources.json'),'execution code differs between blocks')
        a=audit_r5(path,performance_block=True);account=accounting_audit(path)
        final_reconciliation=read(path/'r5_final_reconciliation.json')
        require(final_reconciliation['status']=='PASS' and final_reconciliation['scans'],'missing final full reconciliation')
        initial=read(path/'r5_initial_state.json');ud_initial=read(path/'es/ud_order_initial.json');execution=read(path/'r5_block_execution.json')
        require(initial['cache']['occupancy_bytes']==0 and initial['cache']['entry_count']==0 and not initial['index_entries'] and
            initial['cluster']['centroids']==[] and initial['cluster']['queries']==0 and not initial['seen'] and initial['driver_cache_type_is_none'] and not initial['active_kv_history'],'serving state leaked')
        require(ud_initial==dict(grants=[],parents=[],finished=[],layer=0,current=None),'UD grant/history state leaked')
        identity=read(path/'r5_identity.json')
        invariant={k:v for k,v in identity.items() if k!='gpu'}
        invariant['gpu_hardware']=None if identity['gpu'] is None else {k:identity['gpu'][k] for k in ('name','uuid')}
        identities.append(invariant)
        worker=read(path/'es/es_result.json');pids.extend([worker['pid'],worker['ud_pid']]);sessions.append(worker['session_id'])
        if config['warmup_requests']:
            warm=execution['phases'][0];require(warm['warmup'] and warm['requests']==config['warmup_requests'] and warm['cache_final']['occupancy_bytes']==0 and
                warm['ud_pid']!=worker['ud_pid'] and warm['session_id']!=worker['session_id'],'warmup state not isolated')
            warm_rows=jsonl(path/'warmup/r5_generation_outputs.jsonl')
            require(all(r['source'] is None and r['admission'] is None for r in warm_rows),'warmup introduced reuse/admission')
            warm_requests=[e['request'] for e in jsonl(path/'warmup/r5_requests.jsonl') if e['kind']=='ARRIVAL']
            warm_identity=[{k:r[k] for k in ('input_text','token_ids','attention_mask','positions','generation_policy','adapter_id','physical_identity')} for r in warm_requests]
            if warmup_reference is None:warmup_reference=warm_identity
            require(warm_identity==warmup_reference and len(warm_requests)==config['warmup_requests'],'warmup inputs/settings differ')
            warm_ud=read(path/'warmup/es/ud_result.json')
            require(warm_ud['execution_status']=='COMPLETE' and not warm_ud['registered_entries'] and warm_ud['parents']==config['warmup_requests'],'warmup UD did not complete cleanly without grants')
        rows=jsonl(path/'r5_generation_outputs.jsonl');requests=[e['request'] for e in jsonl(path/'r5_requests.jsonl') if e['kind']=='ARRIVAL']
        require(all(r['evidence_profile']==config['evidence_profile'] for r in rows),'actual request evidence profile differs')
        steps=jsonl(path/'es/r5_tokens.jsonl');timings=jsonl(path/'es/r5_perf_steps.jsonl')
        require(len(timings)==len(steps),'missing step timing')
        require([r['request_sequence_number'] for r in requests]==list(range(10)) and all(r['request_index']==i for i,r in enumerate(rows)),'internal request order changed')
        require((a['hits']==0 and a['admissions']==0 and a['evictions']==0) if task['condition']=='A' else (a['hits']==4 and a['admissions']==4 and a['evictions']==1),'block lifecycle differs')
        for step,timing in zip(steps,timings):
            f=step['perf_forward'];require(step['request_id']==timing['request_id'] and step['step']==timing['step'] and
                timing['exclusive_forward_ms']==f['exclusive_ms'] and timing['wall_to_availability_ms']==step['wall_ms'] and
                timing['post_forward_ms']==step['perf_post_forward_ms'] and f['c_forward_ms']>0 and
                'evidence_budget_check' in f['exclusive_ms'] and 'forward_preflight' in f['exclusive_ms'],'timing attribution missing/different')
            require(all(v>=0 for v in f['exclusive_ms'].values()),'negative attribution')
            perstep.append(dict(block_id=bid,condition=task['condition'],**timing,host_forward_and_evidence_ms=step['host_forward_and_evidence_ms'],
                cpu_ud_ms=f['cpu_ud_ms'],tcp_send_ms=f['tcp_send_ms'],tcp_receive_ms=f['tcp_receive_ms'],cuda_event_ms=f['cuda_event_ms']))
        summary=dict(**task,actual_start_ns=actual_order[sequence]['start_ns'],actual_end_ns=actual_order[sequence]['end_ns'],
            identity_sha256=digest(invariant),model_fingerprint=identity['physical_identity']['base_fingerprint'],
            adapter_matrix_fingerprint=identity['physical_identity']['adapter_matrix_fingerprint'],gpu=identity['gpu'],evidence_profile=config['evidence_profile'],initial=initial,final=worker['cache_final'],
            audit=a,accounting=account,setup_ms=execution['setup_ms'],warmup=execution['phases'][:-1],requests=len(rows),physical_hits=a['hits'],hit_rate=a['hits']/len(rows),
            generated_tokens=sum(r['generated_tokens'] for r in rows),admissions=a['admissions'],evictions=a['evictions'])
        sum_fields=('application_e2e_ms','application_ttft_ms','model_execution_e2e_ms','prefill_ms','decode_ms','semantic_encoding_ms',
            'cache_lookup_and_kv_ms','cache_admission_eviction_ms','evidence_scan_ms','evidence_writer_ms','total_request_wire_bytes')
        summary.update({k:sum(r[k] for r in rows) for k in sum_fields})
        summary['prefill_wire_bytes']=sum(r['prefill']['application_wire_bytes'] for r in rows)
        summary['decode_wire_bytes']=sum(r['decode']['application_wire_bytes'] for r in rows)
        summary['cpu_ud_ms']=sum(s['perf_forward']['cpu_ud_ms'] for s in steps)
        summary['tcp_send_ms']=sum(s['perf_forward']['tcp_send_ms'] for s in steps);summary['tcp_receive_ms']=sum(s['perf_forward']['tcp_receive_ms'] for s in steps)
        summary['gpu_memory']=[s['perf_forward'].get('gpu_memory') for s in steps]
        summary['ud_evidence_writer_ms']=read(path/'es/ud_accounting.json')['writer_io_ms']
        summary['admitted_source_application_ms']=sum(r['application_e2e_ms'] for r in rows if r['admission'])
        summary['admission_per_realized_hit_ms']=sum(r['cache_admission_eviction_ms'] for r in rows)/a['hits'] if a['hits'] else None
        summary['amortization_scope']='actual workload total already includes all source generations/admissions; source application subtotal must not be added again'
        blocks.append(summary);data[bid]=(path,rows,requests,steps)
    require(len(set(sessions))==len(sessions) and len(set(pids))==len(pids),'ES/UD instances reused across blocks')
    require(all(i==identities[0] for i in identities),'model/adapter/tokenizer/encoder/GPU identity mismatch')
    grouped=defaultdict(list)
    for b in order['blocks']:grouped[b['independent_pair']].append(b)
    paired_blocks=[];block_by={b['block_id']:b for b in blocks}
    for pair_id,tasks in sorted(grouped.items()):
        aa=next(t for t in tasks if t['condition']=='A');bb=next(t for t in tasks if t['condition']=='B')
        ap,ar,aq,asteps=data[aa['block_id']];bp,br,bq,bsteps=data[bb['block_id']]
        paired_blocks.append(dict(pair_id=pair_id,repetition=aa['repetition'],order=aa['pair_order'],A_block=aa['block_id'],B_block=bb['block_id'],
            **{f+'_delta':block_by[bb['block_id']][f]-block_by[aa['block_id']][f] for f in sum_fields}))
        for i,(a,b,ra,rb) in enumerate(zip(ar,br,aq,bq)):
            require(all(ra[k]==rb[k] for k in ('input_text','token_ids','attention_mask','positions','generation_policy','adapter_id','user_id','physical_identity')),'unmatched input/policy/adapter')
            ab=(ap/'es'/a['logits_file']).read_bytes();bl=(bp/'es'/b['logits_file']).read_bytes();width=a['logits_shape'][-1]
            av,bv=unpack(ab,'float32'),unpack(bl,'float32')
            numeric=numerical([av[j:j+width] for j in range(0,len(av),width)],[bv[j:j+width] for j in range(0,len(bv),width)],a['generated_token_ids'],b['generated_token_ids'])
            tolerance=[]
            for step in range(min(a['generated_tokens'],b['generated_tokens'])):
                if a['generated_token_ids'][:step]!=b['generated_token_ids'][:step]:break
                tolerance.append(check_parity(bv[step*width:(step+1)*width],av[step*width:(step+1)*width],[1,1,width]))
            hit=b['source'] is not None
            contrast=dict(pair_id=pair_id,repetition=aa['repetition'],order=aa['pair_order'],request_index=i,physical_hit=hit,
                token_agreement=a['generated_token_ids']==b['generated_token_ids'],text_agreement=a['generated_text']==b['generated_text'],
                bitwise_logits=ab==bl,numerical=numeric,frozen_numerical_checks=tolerance,
                frozen_tolerance_pass=bool(tolerance) and len(tolerance)==a['generated_tokens']==b['generated_tokens'] and all(t['passed'] for t in tolerance),
                A_block=aa['block_id'],B_block=bb['block_id'])
            for field in sum_fields:contrast[field+'_delta']=b[field]-a[field]
            contrast['prefill_wire_bytes_delta']=b['prefill']['application_wire_bytes']-a['prefill']['application_wire_bytes']
            contrast['decode_wire_bytes_delta']=b['decode']['application_wire_bytes']-a['decode']['application_wire_bytes']
            ass=[s for s in asteps if s['request_id']==a['request_id']];bss=[s for s in bsteps if s['request_id']==b['request_id']]
            if not hit:
                require([s['context']['input_token_ids'] for s in ass]==[s['context']['input_token_ids'] for s in bss] or not contrast['token_agreement'],'non-HIT inputs differ despite matching generation')
                require(ass[0]['executed_projection_token_rows']==bss[0]['executed_projection_token_rows'] and
                    a['prefill']['hidden_payload_bytes']==b['prefill']['hidden_payload_bytes'] and a['prefill']['delta_payload_bytes']==b['prefill']['delta_payload_bytes'],
                    'non-HIT hidden computation/payload mismatch')
                forward=lambda ss:sum(s['perf_forward']['nested_ms'].get('actual_opt_distributed_forward_nested',0.) for s in ss)
                contrast['miss_actual_forward_delta_ms']=forward(bss)-forward(ass)
                contrast['miss_adjusted_e2e_delta_ms']=contrast['application_e2e_ms_delta']-contrast['cache_admission_eviction_ms_delta']-contrast['cache_lookup_and_kv_ms_delta']
                negative.append(contrast)
            pairs.append(contrast)
    # Resample whole ABBA/BAAB repetitions: retain both within-cycle orders and
    # the dependence among requests. Never bootstrap 10 requests as replicates.
    statistics_report={}
    for label,selected in [('all',pairs),('hit',[p for p in pairs if p['physical_hit']]),('non_hit',negative)]:
        byrep=defaultdict(list)
        for p in selected:byrep[p['repetition']].append(p)
        metrics={}
        fields=['application_e2e_ms_delta','application_ttft_ms_delta','model_execution_e2e_ms_delta','prefill_ms_delta','decode_ms_delta']
        if label=='non_hit':fields+=['miss_actual_forward_delta_ms','miss_adjusted_e2e_delta_ms']
        for field in fields:
            values=[statistics.mean(x[field] for x in group) for rep,group in sorted(byrep.items())]
            metrics[field]=bootstrap(values,config['seed'])
        statistics_report[label]=metrics
    order_contrasts={}
    for direction in ('AB','BA'):
        byrep=defaultdict(list)
        for p in pairs:
            if p['order']==direction:byrep[p['repetition']].append(p['application_e2e_ms_delta'])
        order_contrasts[direction]=bootstrap([statistics.mean(xs) for _,xs in sorted(byrep.items())],config['seed'])
    request_contrasts={str(i):bootstrap([statistics.mean(p['application_e2e_ms_delta'] for p in pairs if p['request_index']==i and p['repetition']==r)
        for r in range(order['repetitions'])],config['seed']) for i in range(10)}
    workload_by_rep=[]
    for r in range(order['repetitions']):workload_by_rep.append(sum(p['application_e2e_ms_delta'] for p in pairs if p['repetition']==r)/2)
    workload=bootstrap(workload_by_rep,config['seed'])
    intervals=[statistics_report['non_hit'][k]['ci95'] for k in ('miss_actual_forward_delta_ms','miss_adjusted_e2e_delta_ms')]
    drift=None if any(ci is None for ci in intervals) else any(not (ci[0]<=0<=ci[1]) for ci in intervals)
    sufficient=order['repetitions']>=8
    ttft_ci=statistics_report['hit']['application_ttft_ms_delta']['ci95'];e2e_ci=workload['ci95']
    decision.update(R5_I3P_TIMING_ATTRIBUTION_PASS=True,R5_I3P_EVIDENCE_ACCOUNTING_PASS=True,R5_I3P_STATE_ISOLATION_PASS=True,
        R5_I3P_ORDER_BALANCED_PASS=True,R5_I3P_INDEPENDENT_AUDIT_PASS=True,R5_I3P_GPU_EXECUTION_PASS=not config['fixture'])
    evidence_ok=not config['fixture'] and sufficient and drift is False and all(p['token_agreement'] and p['frozen_tolerance_pass'] for p in pairs)
    decision['R5_I3P_PERFORMANCE_QUALIFIED']=bool(evidence_ok and not config['legacy_scan'])
    decision['STATISTICALLY_SUPPORTED_TTFT_REDUCTION']=bool(evidence_ok and ttft_ci and ttft_ci[1]<0)
    decision['STATISTICALLY_SUPPORTED_E2E_REDUCTION']=bool(evidence_ok and e2e_ci and e2e_ci[1]<0)
    # Qualification is confined to an instrumented within-distributed comparison
    # with actual physical proofs, exact output tokens and frozen logit tolerance.
    # It does not establish Native/full-tensor strict parity or qualified speedup.
    drift_report=dict(miss_only_unexplained_drift=drift,criterion='95% cycle-bootstrap CI excluding zero for actual non-HIT forward or admission/lookup-adjusted application delta; no arbitrary millisecond cutoff',
        chronological_application_latency=trend([(j,row['application_e2e_ms']) for j,(bid,row) in enumerate((bid,row) for bid in data for row in data[bid][1])]),
        accumulated_files_step_latency=trend([(s['evidence']['data_files'],s['step_through_json_ms']) for s in perstep]),
        chronological_step_latency=trend([(j,s['step_through_json_ms']) for j,s in enumerate(perstep)]))
    condition_statistics={condition:{field:bootstrap([statistics.mean(b[field] for b in blocks if b['condition']==condition and b['repetition']==r)
        for r in range(order['repetitions'])],config['seed']) for field in ('application_e2e_ms','application_ttft_ms','model_execution_e2e_ms','prefill_ms','decode_ms')}
        for condition in ('A','B')}
    drift_report['within_block']={b['block_id']:dict(chronological_application=trend([(r['request_index'],r['application_e2e_ms']) for r in data[b['block_id']][1]]),
        files_vs_step_latency=trend([(s['evidence']['data_files'],s['step_through_json_ms']) for s in perstep if s['block_id']==b['block_id']])) for b in blocks}
    return dict(schema='R5_PERF_ANALYSIS_V1',decision=decision,audit=dict(status='PASS',blocks=len(blocks),requests=len(blocks)*10,
        fixture=config['fixture'],inference_settings_identical=True,fresh_processes_verified=True),blocks=blocks,paired_blocks=paired_blocks,paired_requests=pairs,negative_controls=negative,
        statistics=dict(principal_unit='independent four-block repetition; clustered paired bootstrap',independent_repetitions=order['repetitions'],
            minimum_design_repetitions=8,minimum_is_not_significance_guarantee=True,scopes=statistics_report,workload_e2e_delta=workload,
            order_specific=order_contrasts,per_request=request_contrasts,conditions=condition_statistics),drift=drift_report,steps=perstep,
        limitations=dict(GPU_NOT_RUN=config['fixture'],historical='HISTORICAL_GPU_EVIDENCE_UNAVAILABLE_LOCALLY',
            old_55_1_percent='CONFOUNDED_NOT_A_BENCHMARK',profile=config['evidence_profile'],
            strict_tensor_evidence=config['evidence_profile']=='STRICT_AUDIT',native_parity='HISTORICAL_UNRESOLVED',
            qualification_scope='instrumented distributed A/B with physical audit, exact token output and unchanged R3-A logit tolerances; not Native PEFT or full strict numerical parity',
            performance='instrumented localhost, ordered controlled text; no uninstrumented/native-qualified speedup',
            warmup='separate no-reuse UD session; no measured cache/index/cluster/history survives',
            statistical='cold processes alone do not prove absence of hardware/time dependence; repetitions and drift diagnostics required'))


def export(root,result):
    root=Path(root)
    for key,name in [('decision','r5_perf_decision.json'),('audit','r5_perf_audit.json'),('blocks','r5_perf_blocks.json'),
        ('paired_blocks','r5_perf_paired_blocks.json'),
        ('paired_requests','r5_perf_fidelity.json'),('negative_controls','r5_perf_negative_controls.json'),('statistics','r5_perf_statistics.json'),
        ('drift','r5_perf_drift.json'),('limitations','r5_perf_limitations.json'),('steps','r5_perf_step_summary.json')]:write(root/name,result[key])
