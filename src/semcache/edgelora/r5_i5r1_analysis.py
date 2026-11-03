"""R1 assertions around the existing strict I5 Raw-QKV/TCP audit, never a toy cache."""
from collections import defaultdict
import csv
import math
from pathlib import Path

from .communication import require,digest
from .r3a_artifacts import read,write,unpack,THRESHOLDS
from .r4b2_analysis import tensor_data
from .r5_i5_evidence import fresh,seal,jsonl
from .r5_i5_analysis import analyze,distribution,grouped_ci,METRICS
from .r5_multi_analysis import fidelity
from .r5_i5r1_plan import validate_plan
from .r5_i5_workload import CONDITIONS

COMPARISONS=((CONDITIONS[0],CONDITIONS[1]),(CONDITIONS[1],CONDITIONS[2]),(CONDITIONS[0],CONDITIONS[2]))


def delta(reference,candidate):
    return dict(reference=reference,candidate=candidate,absolute_delta=candidate-reference,
                percentage_delta=100*(candidate-reference)/reference if reference else None)


def costs(rows,initialization_ms,warmup_ms,lifecycle):
    """Scopes are alternatives, never added to nested model/semantic timers."""
    targets=[r for r in rows if r['scored']];sources=[r for r in rows if not r['scored']]
    require(targets and len(sources)==1,'one priming source and nonempty targets required')
    require(rows[0] is sources[0] and all(r['completion_ns']<=s['arrival_ns'] for r,s in zip(rows,rows[1:])),
            'nonsequential or reordered controlled workload')
    target_ms=sum(r['application_e2e_ms'] for r in targets);request_ms=sum(r['application_e2e_ms'] for r in rows)
    span=(rows[-1]['completion_ns']-rows[0]['arrival_ns'])/1e6
    require(span+1e-6>=request_ms,'request span excludes measured work')
    require(lifecycle['execution_start_ns']<=rows[0]['arrival_ns']<=rows[-1]['completion_ns']<=lifecycle['execution_end_ns'] and
        lifecycle['process_wall_ms']==(lifecycle['execution_end_ns']-lifecycle['execution_start_ns'])/1e6,'condition process timing differs')
    return dict(targets=len(targets),priming_sources=len(sources),warm_target_mean_ms=target_ms/len(targets),
        target_interval_sum_ms=target_ms,source_request_ms=sum(r['application_e2e_ms'] for r in sources),
        source_admission_eviction_ms=sum(r['cache_admission_eviction_ms'] for r in sources),
        source_to_last_target_span_ms=span,request_interval_sum_ms=request_ms,inter_request_gap_ms=span-request_ms,
        amortized_sequence_ms_per_target=span/len(targets),model_session_initialization_ms=initialization_ms,warmup_ms=warmup_ms,
        process_lifecycle_ms=lifecycle['process_wall_ms'],process_ms_per_target=lifecycle['process_wall_ms']/len(targets),
        scored_targets_per_second_sequence=len(targets)/(span/1000),scored_targets_per_second_lifecycle=len(targets)/(lifecycle['process_wall_ms']/1000),
        scopes='Sequence includes source, admission, targets and between-request work, excludes startup/warmup and final target score/serialization. Process lifecycle includes initialization, warmup, final scoring/serialization and ES/UD teardown; excludes parent reconciliation, offline audit and scheduler queue.')


def layer_diagnostics(root,a,b,steps_a,steps_b):
    """Last prefill position only; localizes an observed capture, not numerical causality."""
    out=[];order=('q','k','v','attention','ffn','layer')
    for layer in range(a['physical_identity']['layers']):
        for role in order:
            kind='qkv' if role in 'qkv' and len(role)==1 else 'components';key=f'{layer}.{role}'
            if key not in steps_a[kind] or key not in steps_b[kind]:continue
            sa=steps_a[kind][key];sb=steps_b[kind][key];require(sa['dtype']==sb['dtype'],'component precision differs')
            width=sa['shape'][-1];size=2 if sa['dtype']=='float16' else 4
            x=unpack(tensor_data(root/a['condition']/a['adapter_id']/'es',sa)[-width*size:],sa['dtype'])
            y=unpack(tensor_data(root/b['condition']/b['adapter_id']/'es',sb)[-width*size:],sb['dtype'])
            errors=[abs(v-u) for u,v in zip(x,y)];relative=math.sqrt(sum(e*e for e in errors)/max(sum(u*u for u in x),1e-30))
            violations=sum(e>THRESHOLDS['atol']+THRESHOLDS['rtol']*abs(u) for e,u in zip(errors,x))
            out.append(dict(layer=layer,operation=role,phase='PREFILL_LAST_POSITION',max_abs=max(errors),relative_l2=relative,
                tolerance_violations=violations,bitwise_equal=x==y,
                diagnostic_tolerance_pass=violations==0 and max(errors)<=THRESHOLDS['max_abs'] and relative<=THRESHOLDS['relative_l2']))
    return dict(sample_scope='All available strict captures at the final prefill position; activation checks are diagnostic, the frozen logit verdict is separate',
        checks=out,first_observed_bitwise_divergence=next((x for x in out if not x['bitwise_equal']),None),
        first_observed_tolerance_divergence=next((x for x in out if not x['diagnostic_tolerance_pass']),None),
        causality='UNRESOLVED: FP16 shape dependence, reconstruction or execution ordering require R5-I5R-2 investigation; this capture does not establish the cause')


def analyze_block(experiment_path,root):
    root=Path(root);plan=validate_plan(experiment_path);report=analyze(root);cfg=report['config'];block=cfg['block']
    require(cfg['source_commit']==plan['i5r1']['source']['commit'] and cfg['sources']==plan['i5r1']['source']['files'],
            'R1 execution differs from frozen preparation source')
    require(cfg['experiment_id']==plan['experiment_id'] and cfg['fixture']==plan['TEST_FIXTURE_ONLY'],'R1 matched plan/scope differs')
    require(Path(cfg['multi_prepared']).resolve()==Path(plan['i5r1']['multi_prepared']),'R1 execution adapter preparation differs')
    rows={(r['condition'],r['request_id']):r for r in report['request_metrics']};by_steps={};reqs={};condition_costs={}
    for c in CONDITIONS:
        reqs[c]={x['request']['request_id']:x['request'] for x in jsonl(root/c/'requests.jsonl') if x['kind']=='ARRIVAL'}
        for aid in plan['i5r1']['adapter_fingerprints']:
            for step in jsonl(root/c/aid/'es/r5_tokens.jsonl'):
                by_steps.setdefault((c,step['request_id']),[]).append(step)
        subset=[rows[c,t['request_id']] for t in block['tasks']]
        init=report['independent_audit']['conditions'][c]['initialization']['model_and_session_initialization_ms']
        warm=report['independent_audit']['conditions'][c]['warmup_total_ms']
        condition_costs[c]=costs(subset,init,warm,read(root/c/'lifecycle.json'))
    hits=[];numerical=[];layers=[];target_ids={t['request_id'] for t in block['tasks'] if t['scored']}
    for proof in block['pair_checks']:
        sid=proof['source_request_id'];rid=proof['target_request_id'];a=rows[CONDITIONS[0],rid];c=rows[CONDITIONS[2],rid]
        source=rows[CONDITIONS[2],sid];grant=c['source'];p=proof['source_tokens'];first=by_steps[CONDITIONS[2],rid][0]
        expected_skip=by_steps[CONDITIONS[0],rid][0]['executed_projection_token_rows']-first['executed_projection_token_rows']
        events=[e['event'] for e in report['cache_events'] if e['condition']==CONDITIONS[2]]
        hit_events=[e for e in events if e['kind']=='PHYSICAL_REUSE' and e['request_id']==rid]
        correction_rows={}
        for condition in (CONDITIONS[0],CONDITIONS[2]):
            fid=by_steps[condition,rid][0]['forward_id']
            hidden=[f for f in jsonl(root/condition/c['adapter_id']/'es/wire.jsonl')
                    if f['header']['kind']=='HIDDEN' and f['header']['request_id']==fid]
            require(len(hidden)==c['physical_identity']['layers'],'missing prefill CPU correction receipts')
            correction_rows[condition]=3*sum(f['header']['shape'][-2] for f in hidden)
        omitted_correction_rows=correction_rows[CONDITIONS[0]]-correction_rows[CONDITIONS[2]]
        checks=dict(actual_grant=grant is not None,source_admitted=source['admission'] is not None,
            controller_physical_hit=c['decision']=='PHYSICAL_PREFIX_HIT',physical_reuse_event=len(hit_events)==1,
            suffix_only=first['context']['input_token_ids']==reqs[CONDITIONS[2]][rid]['token_ids'][p:],
            projection_omission=c['skipped_prefix_projection_token_rows']==expected_skip==p*3*c['physical_identity']['layers'],
            CPU_correction_omission=omitted_correction_rows==expected_skip,
            cached_Q_not_consumed=c['cached_prefix_q_consumed'] is False,
            strict_capture=c['evidence_profile']==source['evidence_profile']=='STRICT_AUDIT',
            source_completed_before_target=source['completion_ns']<c['arrival_ns'])
        if grant:
            checks.update(source_identity=grant['source_request_id']==sid and grant['entry_id']==source['admission']['entry_id'],
                reused_tokens=c['reused_prefix_tokens']==p,
                resident_at_lookup=grant['entry_id'] in c['cache_before']['lru_to_mru'],
                exact_token_prefix=reqs[CONDITIONS[2]][sid]['token_ids']==reqs[CONDITIONS[2]][rid]['token_ids'][:p],
                positions=reqs[CONDITIONS[2]][sid]['positions']==reqs[CONDITIONS[2]][rid]['positions'][:p],
                mask=reqs[CONDITIONS[2]][sid]['attention_mask']==reqs[CONDITIONS[2]][rid]['attention_mask'][:p])
        functional=all(checks.values())
        hits.append(dict(proof,checks=checks,functional_checks_pass=functional,actual_reused_tokens=c['reused_prefix_tokens'],
            admission=source['admission'],cache_before=c['cache_before'],cache_after=c['cache_after'],evictions=c['eviction_count'],
            actual_projection_token_role_layer_rows_omitted=expected_skip,
            actual_CPU_correction_token_role_layer_rows_omitted=omitted_correction_rows,
            CPU_correction_basis='Audited first-forward HIDDEN/DELTAS shapes at every CPU UD layer; correction call count remains one per layer',
            prefill_hidden_payload_bytes_omitted=a['prefill']['hidden_payload_bytes']-c['prefill']['hidden_payload_bytes'],
            prefill_delta_payload_bytes_omitted=a['prefill']['delta_payload_bytes']-c['prefill']['delta_payload_bytes'],
            total_request_wire=delta(a['total_request_wire_bytes'],c['total_request_wire_bytes']),CPU_LoRA_ms=delta(a['cpu_lora_ms'],c['cpu_lora_ms']),
            execution_gpu_uuid=c['physical_identity']['gpu_uuid'],preparation_gpu_uuid=report['independent_audit']['conditions'][CONDITIONS[2]]['gpu_identity']['preparation_gpu_uuid']))
        layers.append(dict(request_id=rid,**layer_diagnostics(root,a,c,by_steps[CONDITIONS[0],rid][0],first)))
    # Include priming requests as well as all scored targets in numerical comparisons.
    for ca,cb in COMPARISONS:
        for t in block['tasks']:
            ra=rows[ca,t['request_id']];rb=rows[cb,t['request_id']]
            numerical.append(dict(comparison=ca+'_vs_'+cb,scored=t['scored'],**fidelity(ra,rb,root/ca/ra['adapter_id']/'es',root/cb/rb['adapter_id']/'es')))
    summaries={}
    for a,b in COMPARISONS:
        summaries[a+'_vs_'+b]={k:delta(condition_costs[a][k],condition_costs[b][k]) for k in
            ('warm_target_mean_ms','source_to_last_target_span_ms','amortized_sequence_ms_per_target','process_lifecycle_ms','process_ms_per_target')}
    return dict(root=str(root),block=block,config=cfg,report=report,hits=hits,numerical=numerical,layers=layers,
        costs=condition_costs,cost_comparisons=summaries,functional_assertions='PASS' if hits and all(h['functional_checks_pass'] for h in hits) else 'FAIL',
        nonempty_target_outputs=all(rows[c,rid]['generated_token_ids'] and rows[c,rid]['generated_text'].strip() for c in CONDITIONS for rid in target_ids))


def decide(*,fixture,functional,cache_integrity,numerical,tokens,nonempty,warm_delta,amortized_delta):
    """Scientific verdicts remain separate; CPU data never produces a GPU PASS."""
    real=lambda ok:'CPU_FIXTURE_ONLY' if fixture else 'PASS' if ok else 'FAIL'
    return dict(FUNCTIONAL_PHYSICAL_REUSE=real(functional),PHYSICAL_HIT_FUNCTIONAL=real(functional),
        CACHE_INTEGRITY=real(cache_integrity),RAW_QKV_INTEGRITY=real(cache_integrity),
        NUMERICAL_FIDELITY=real(numerical),NUMERICAL_TOLERANCE=real(numerical),TOKEN_TEXT_FIDELITY=real(tokens),
        NONEMPTY_TARGET_OUTPUTS=real(nonempty),NATIVE_PEFT_STRICT_PARITY='UNESTABLISHED',
        WARM_TARGET_BENEFIT=real(warm_delta<0),AMORTIZED_E2E_BENEFIT=real(amortized_delta<0),
        CORRECTNESS_QUALIFIED_SPEEDUP='UNESTABLISHED',GPU_NOT_RUN=fixture,
        qualification='Native PEFT strict parity remains unestablished; physical success alone or matching tokens cannot establish numerical or performance success')


def summarize(experiment_path,runs,output,*,block_only=False):
    plan=validate_plan(experiment_path);expected={b['block_id'] for b in plan['blocks']}
    require(runs and (not block_only or len(runs)==1),'select one block for block-only review, or every block')
    blocks=[];seen=set()
    for path in runs:
        r=analyze_block(experiment_path,path);bid=r['block']['block_id']
        require(bid in expected and bid not in seen,'unmatched/duplicate controlled block');seen.add(bid);blocks.append(r)
    require(block_only or seen==expected,'MISSING_BLOCKS: R1 complete comparison requires all planned blocks')
    blocks.sort(key=lambda r:next(i for i,b in enumerate(plan['blocks']) if b['block_id']==r['block']['block_id']))
    grouped=defaultdict(list)
    for r in blocks:grouped[r['block']['dataset'],r['block']['i5r1_case'],r['block']['i5r1_target_repeats']].append(r)
    groups=[];table=[]
    for (dataset,case,repeats),rs in sorted(grouped.items()):
        comparisons={};a,c=CONDITIONS[0],CONDITIONS[2]
        for ca,cb in COMPARISONS:
            label=ca+'_vs_'+cb;metrics={}
            for k in rs[0]['cost_comparisons'][label]:
                paired=[dict(absolute_delta=r['cost_comparisons'][label][k]['absolute_delta'],independent_block_group=r['block']['i5r1_source_group']) for r in rs]
                metrics[k]=dict(absolute_delta=distribution([x['absolute_delta'] for x in paired]),
                    paired_percentage_delta=distribution([r['cost_comparisons'][label][k]['percentage_delta'] for r in rs if r['cost_comparisons'][label][k]['percentage_delta'] is not None]),
                    total_scope_delta=delta(sum(r['costs'][ca][k] for r in rs),sum(r['costs'][cb][k] for r in rs)),
                    confidence_interval=grouped_ci(paired))
            comparisons[label]=metrics
        warm=comparisons[a+'_vs_'+c]['warm_target_mean_ms']['total_scope_delta']['absolute_delta']
        amort=comparisons[a+'_vs_'+c]['source_to_last_target_span_ms']['total_scope_delta']['absolute_delta']
        functional=all(r['functional_assertions']=='PASS' for r in rs)
        decisions=decide(fixture=plan['TEST_FIXTURE_ONLY'],functional=functional,cache_integrity=functional,
            numerical=all(n['frozen_tolerance_pass'] for r in rs for n in r['numerical']),
            tokens=all(n['generated_tokens_equal'] and n['generated_text_equal'] for r in rs for n in r['numerical']),
            nonempty=all(r['nonempty_target_outputs'] for r in rs),warm_delta=warm,amortized_delta=amort)
        independent=len({r['block']['i5r1_source_group'] for r in rs})
        decisions['STATISTICAL_EVIDENCE']='CPU_FIXTURE_ONLY' if plan['TEST_FIXTURE_ONLY'] else 'DESCRIPTIVE_ONLY_FEWER_THAN_THREE_GROUPS' if independent<3 else 'PAIRED_SOURCE_GROUP_BOOTSTRAP_REPORTED; inspect intervals, not a significance declaration'
        groups.append(dict(dataset=dataset,case=case,target_repeats=repeats,independent_source_groups=independent,comparisons=comparisons,decisions=decisions))
    for r in blocks:
        for c,cost in r['costs'].items():table.append(dict(dataset=r['block']['dataset'],case=r['block']['i5r1_case'],
            target_repeats=r['block']['i5r1_target_repeats'],block_id=r['block']['block_id'],source_group=r['block']['i5r1_source_group'],
            condition=c,TEST_FIXTURE_ONLY=plan['TEST_FIXTURE_ONLY'],**{k:v for k,v in cost.items() if not isinstance(v,str)}))
    result=dict(status='CPU_FIXTURE_AUDIT_PASS' if plan['TEST_FIXTURE_ONLY'] else 'INDEPENDENT_EVIDENCE_AUDIT_PASS',
        functional_assertions='PASS' if all(r['functional_assertions']=='PASS' for r in blocks) else 'FAIL',
        GPU_NOT_RUN=plan['TEST_FIXTURE_ONLY'],complete_study=seen==expected,complete_blocks=len(seen),expected_blocks=len(expected),
        experiment_id=plan['experiment_id'],groups=groups,source=plan['i5r1']['source'],
        history='R5-I5R-0 zero HITs diagnosed; HIT numerical root cause ROOT_CAUSE_UNRESOLVED; synthetic rank-8 adapters, localhost TCP, thirty logical users, three physical CPU UD processes per condition',
        warnings=['Warm-target benefit excludes priming. Sequence and process lifecycle costs have separate scopes.',
            'Nested timing components are not additive. Admission is not isolated GPU-copy time.',
            'Shared-source targets and repeated cases are dependent; CIs stay within dataset/case/repeat groups.',
            'No interpolation or estimated-token break-even; only observed repeat counts are reported.'])
    result['request_counts']={c:dict(scored_requests=sum(row['scored'] for r in blocks for row in r['report']['request_metrics'] if row['condition']==c),
        priming_requests=sum(not row['scored'] for r in blocks for row in r['report']['request_metrics'] if row['condition']==c),
        actual_physical_grants=sum(row['source'] is not None for r in blocks for row in r['report']['request_metrics'] if row['condition']==c),
        scored_misses=sum(row['source'] is None and row['scored'] for r in blocks for row in r['report']['request_metrics'] if row['condition']==c)) for c in CONDITIONS}
    result['independently_verified_target_reuse_checks']=sum(h['functional_checks_pass'] for r in blocks for h in r['hits'])
    result['verified_reuse_scope']='CPU_FIXTURE_ONLY' if plan['TEST_FIXTURE_ONLY'] else 'REAL_GPU_STRICT_RAW_QKV_AND_CONSUMED_KV_AUDIT'
    result['observed_amortization_points']=[dict(dataset=g['dataset'],case=g['case'],target_repeats=g['target_repeats'],
        independent_source_groups=g['independent_source_groups'],
        sequence_total_delta=g['comparisons'][CONDITIONS[0]+'_vs_'+CONDITIONS[2]]['source_to_last_target_span_ms']['total_scope_delta'],
        lifecycle_total_delta=g['comparisons'][CONDITIONS[0]+'_vs_'+CONDITIONS[2]]['process_lifecycle_ms']['total_scope_delta']) for g in groups]
    result['break_even_interpretation']='Only observed repeat counts; negative deltas indicate measured benefit at that count. No interpolated crossing or statistical claim from fewer than three source groups. CPU fixture points are not performance evidence.'
    out=fresh(output,(*runs,experiment_path))
    write(out/'i5r1_decision.json',result)
    write(out/'i5r1_independent_audit.json',dict(status='PASS',TEST_FIXTURE_ONLY=plan['TEST_FIXTURE_ONLY'],scope='Existing I5 full strict tensor, TCP and physical/cache replay plus R1 pair and frozen-source checks',
        blocks=[dict(block_id=r['block']['block_id'],root=r['root'],source_commit=r['config']['source_commit'],legacy_audit=r['report']['independent_audit']) for r in blocks]))
    write(out/'i5r1_cost_comparisons.json',[dict(block_id=r['block']['block_id'],costs=r['costs'],comparisons=r['cost_comparisons']) for r in blocks])
    for name,records in (
        ('physical_hits',[dict(block_id=r['block']['block_id'],**h) for r in blocks for h in r['hits']]),
        ('numerical_fidelity',[dict(block_id=r['block']['block_id'],**n) for r in blocks for n in r['numerical']]),
        ('layer_diagnostics',[dict(block_id=r['block']['block_id'],**n) for r in blocks for n in r['layers']]),
        ('request_metrics',[dict(row) for r in blocks for row in r['report']['request_metrics']]),
        ('quality_per_sample',[dict(row) for r in blocks for row in r['report']['quality_per_sample']])):
        write(out/f'i5r1_{name}.json',records)
    write(out/'i5r1_legacy_summaries.json',[dict(block_id=r['block']['block_id'],latency=r['report']['latency_summary'],
        network=r['report']['network_summary'],quality=r['report']['quality_summary'],cache=r['report']['cache_summary']) for r in blocks])
    with (out/'i5r1_cost_table.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    text='# R5-I5R-1 controlled physical prefix reuse\n\n'
    text+=f"GPU_NOT_RUN={plan['TEST_FIXTURE_ONLY']}; complete blocks {len(seen)}/{len(expected)}.\n\n"
    text+='See i5r1_decision.json for separate functional, integrity, token, numerical, warm and amortized verdicts. Numerical failures are preserved. No correctness-qualified speedup or Native PEFT parity is inferred.\n\n'
    for g in groups:text+=f"* {g['dataset']} {g['case']}, repeats={g['target_repeats']}, independent groups={g['independent_source_groups']}: {g['decisions']}\n"
    text+='\nAll model timings include instrumentation. Measured TCP is localhost. Sequence costs include one complete standalone source; process lifecycle additionally includes model load, warmup and shutdown.\n'
    (out/'i5r1_report.md').write_text(text);seal(out)
    return result
