"""Three-condition audit using existing wire/value/policy replay unchanged by default."""
from collections import Counter
from pathlib import Path
import json,tempfile
from .communication import require,digest
from .r3a_artifacts import unpack,check_parity,THRESHOLDS
from .r5_i5r2_tensors import Evidence
from .r5_subsequence_natural_io import output,put,seal,verify as tree_verify
from .r5_subsequence_natural_audit import child
from .r5_natural_approx import approval,execution,CONDITIONS


def row_categories(rows,tasks):
    by_id={t['request_id']:t for t in tasks};counts=Counter({c:0 for c in 'ABCD'})
    for rid,row in rows.items():
        target=by_id[rid]['token_ids'];details=[]
        for record in row['by_layer_role']:
            for item in record['consumed_windows']:
                source=by_id[item['source_request_id']]['token_ids'];lo,hi=item['source_positions'];start,end=item['target_positions']
                require(hi-lo==end-start==3 and source[lo:hi]==target[start:end],'consumed w3 differs')
                prefix=0
                for a,b in zip(source,target):
                    if a!=b:break
                    prefix+=1
                category=('A' if end<=prefix else 'B') if [lo,hi]==[start,end] else ('D' if end<=prefix else 'C')
                counts[category]+=end-start
                details.append(dict(layer=record['layer'],role=record['role'],category=category,**item))
        require(sum(x['target_positions'][1]-x['target_positions'][0] for x in details)==row['reused_projection_rows'],'category/consumed row sum differs')
        row['physical_reuse_categories']=details
    return dict(counts)


def numerical(reference,actual,arow,crow,width):
    require(width>0 and len(reference)%width==len(actual)%width==0,'logit dimensions differ')
    gates=[]
    for step in range(min(len(reference),len(actual))//width):
        gate=check_parity(actual[step*width:(step+1)*width],reference[step*width:(step+1)*width],[1,1,width])
        gates.append(dict(step=step,input_histories_equal=arow['generated_token_ids'][:step]==crow['generated_token_ids'][:step],**gate))
    same_length=len(reference)==len(actual)
    return dict(numerical_tolerance='PASS' if gates and same_length and all(g['input_histories_equal'] and g['passed'] for g in gates) else 'FAIL',
        full_vocabulary_per_step=gates,bitwise_logits_equal=__import__('struct').pack('<'+'f'*len(reference),*reference)==__import__('struct').pack('<'+'f'*len(actual),*actual),
        generation_length_equal=same_length,generated_token_ids_equal=arow['generated_token_ids']==crow['generated_token_ids'],
        generated_text_equal=arow['generated_text']==crow['generated_text'],
        final_output_logit_comparison=check_parity(actual[-width:],reference[-width:],[1,1,width]),
        final_output_histories_matched=same_length and arow['generated_token_ids'][:-1]==crow['generated_token_ids'][:-1],
        note='after generated-history divergence, cross-output logit differences are diagnostic, not matched-input equivalence')


def analyze(experiment,run,cpu_audit,output_dir):
    p,pr,rest,ar=approval(experiment,cpu_audit);block,m,pm,specs,users,tok=rest
    run=Path(run).resolve();inventory=tree_verify(run);status=json.loads((run/'status.json').read_bytes())
    require(status['status']=='COMPLETE' and status['source']==p['implementation']['source'] and status['plan_inventory']==pr and
        status['cpu_audit_inventory']==ar and status['cpu_audit_path']==str(Path(cpu_audit).resolve()) and
        [c['condition'] for c in status['children']]==list(CONDITIONS) and all(c['returncode']==0 for c in status['children']),'failed/unapproved run')
    reports={c:child(run/c,p,pr,block,tok,_conditions=CONDITIONS,_context_mode=execution(c)[0],_allow_approximate=execution(c)[1]) for c in CONDITIONS}
    require(all(reports[c]['config']['condition']==c for c in CONDITIONS),'condition directory/identity differs')
    reference=reports[CONDITIONS[0]]
    from .r5_i5_gpu import verify_receipt
    from .r5_multi_artifacts import physical,runtime_view
    for report in reports.values():
        require(report['config']['gpu']['uuid']==reference['config']['gpu']['uuid'] and report['config']['backend']==reference['config']['backend'] and
            report['identities'].keys()==reference['identities'].keys(),'condition GPU/backend/adapter set differs')
        verify_receipt(report['config']['gpu_identity_receipt'],pm,report['config']['gpu'],'SAME_COMPATIBLE_GPU_CLASS',fixture=False)
        for aid,cfg in report['identities'].items():
            require(all(cfg[k]==reference['identities'][aid][k] for k in ('physical','contract','model_identity','registry')),'condition model/adapter/registry differs')
            native=dict(cfg['contract']);native.pop('r5_multi')
            expected,contract=physical(runtime_view(pm,specs[aid]),p['tokenizer_identity'],cfg['physical']['device'],cfg['physical']['gpu_uuid'],native,specs[aid],users)
            require(cfg['physical']==expected and cfg['contract']==contract and cfg['registry']==users and cfg['model_identity']==runtime_view(pm,specs[aid])['identity'],'prepared native identity differs')
    comparisons=[];totals={};out=output(output_dir,(experiment,run,cpu_audit,p['multi'],p['original_experiment'],*p['dataset_paths'].values()))
    for condition,report in reports.items():
        categories=row_categories(report['requests'],p['tasks'])
        computed=sum(r['costs']['native_projection_rows'] for r in report['requests'].values())
        skipped=sum(r['costs']['physically_reused_projection_rows'] for r in report['requests'].values())
        totals[condition]=dict(application_e2e_ms=sum(r['timing']['application_e2e_ms'] for r in report['requests'].values()),
            application_ttft_sum_ms=sum(r['timing']['application_ttft_ms'] for r in report['requests'].values()),
            request_sequence_span_ms=(report['sequence'][-1]['completion_ns']-report['sequence'][0]['arrival_ns'])/1e6,
            wire_bytes=sum(r['network']['es_to_ud_wire_bytes']+r['network']['ud_to_es_wire_bytes'] for r in report['requests'].values()),
            native_and_cpu_lora_projection_rows=computed,projection_rows_skipped=skipped,
            projection_rows_skipped_by_category=categories,
            physical_qkv_rows_consumed_by_category=categories,physical_hit_requests=sum(r['physical_hit'] for r in report['requests'].values()),
            quality=[r['quality'] for t in p['tasks'] if t['scored'] for r in [report['requests'][t['request_id']]]],
            policy=report['policy'],initialization_ms=report['config']['initialization_ms'])
        component_names={x['component'] for r in report['requests'].values() for x in r['diagnostics']['components']}
        totals[condition]['cache_and_inference_component_inclusive_ms']={name:sum(x['inclusive_wall_ms'] for r in report['requests'].values()
            for x in r['diagnostics']['components'] if x['component']==name) for name in sorted(component_names)}
        totals[condition]['component_scope']='inclusive wall scopes overlap; compare each named component, never sum nested components'
        for rid,row in report['requests'].items():put(out/condition/(rid+'.json'),row)
        for task in p['tasks']:
            if condition==CONDITIONS[0]:continue
            rid=task['request_id'];a=reference['requests'][rid];c=report['requests'][rid]
            av=unpack(Evidence(run/CONDITIONS[0]/task['adapter_id'],max_input_bytes=32*2**30).tensor('es',a['logits']),'float32')
            cv=unpack(Evidence(run/condition/task['adapter_id'],max_input_bytes=32*2**30).tensor('es',c['logits']),'float32')
            require(a['logits_shape'][-1]==c['logits_shape'][-1]==50272,'full OPT vocabulary differs')
            comparison=dict(condition=condition,request_id=rid,scored=task['scored'],
                exact_equivalence_claim='NOT_CLAIMED',**numerical(av,cv,a,c,50272),
                quality_delta=None if not task['scored'] else c['quality']['score']-a['quality']['score'])
            name=condition+'/'+rid+'_numerical.json';put(out/name,comparison)
            comparisons.append(dict(file=name,condition=condition,request_id=rid,numerical_tolerance=comparison['numerical_tolerance'],
                generated_token_ids_equal=comparison['generated_token_ids_equal'],quality_delta=comparison['quality_delta']))
    for condition,summary in totals.items():
        summary['delta_e2e_ms_vs_full']=summary['application_e2e_ms']-totals[CONDITIONS[0]]['application_e2e_ms']
        summary['delta_e2e_percent_vs_full']=100*summary['delta_e2e_ms_vs_full']/totals[CONDITIONS[0]]['application_e2e_ms']
        quality=summary['quality'];aq=totals[CONDITIONS[0]]['quality']
        summary['task_quality_score_mean']=sum(r['score'] for r in quality)/len(quality)
        summary['task_quality_delta_vs_full']=sum(c['score']-a['score'] for a,c in zip(aq,quality))/len(quality)
        summary['observed_instrumented_latency_reduction_with_nonprefix_skipping']=(summary['delta_e2e_ms_vs_full']<0 and
            summary['projection_rows_skipped']>0 and sum(summary['physical_qkv_rows_consumed_by_category'][c] for c in 'BC')>0)
    positive=totals[CONDITIONS[2]]['physical_qkv_rows_consumed_by_category']['B']+totals[CONDITIONS[2]]['physical_qkv_rows_consumed_by_category']['C']>0
    result=dict(schema=p['schema'],status='EVIDENCE_INTEGRITY_PASS',track=p['track'],plan_id=p['plan_id'],
        plan_inventory=pr,run_inventory=inventory,cpu_audit_inventory=ar,thresholds=THRESHOLDS,conditions=totals,comparisons=comparisons,
        controlled_nonprefix_consumption=('PASS' if positive else 'FAIL_NO_NONPREFIX_CONSUMPTION') if p['track']=='CONTROLLED' else 'NOT_APPLICABLE',
        NATURAL_zero_hit_is_valid=p['track']=='NATURAL' and not positive,expansion='REQUIRES_SEPARATE_DECISION',
        correctness_qualified_speedup='NOT_ESTABLISHED',measurement='instrumented strict evidence E2E; source requests included; initialization/final seals separate; changed generation length can affect timing; no causal/statistical speedup claim')
    put(out/'approx_audit.json',result)
    (out/'approx_report.md').write_text('# D5 '+p['track']+' A/EXACT/APPROX pilot\n\nEvidence integrity: PASS. Numerical tolerance, generation agreement and task quality are independent fields.\n\n'+
        'Controlled non-prefix consumption: '+result['controlled_nonprefix_consumption']+'.\n\n'+
        '\n\n'.join(c+': '+json.dumps(s,sort_keys=True) for c,s in totals.items())+
        '\n\nZero natural hits are valid. No exact-equivalence, correctness-qualified or statistical speedup is claimed. Controlled source costs are included.\n')
    seal(out);return {k:v for k,v in result.items() if k not in ('comparisons','conditions')}


def verify(experiment,run,cpu_audit,audit,output_dir):
    p,_,_,_=approval(experiment,cpu_audit)
    expected=tree_verify(Path(audit).resolve())
    with tempfile.TemporaryDirectory() as tmp:
        fresh=Path(tmp)/'audit';analyze(experiment,run,cpu_audit,fresh)
        require(tree_verify(fresh)==expected,'independent approximate audit replay differs')
    out=output(output_dir,(experiment,run,cpu_audit,audit,p['multi'],p['original_experiment'],*p['dataset_paths'].values()));put(out/'approx_verification.json',dict(status='PASS',audit_inventory=expected));seal(out)
    return dict(status='PASS')
