#!/usr/bin/env python3
"""B3 FULL-relative paired latency, explicit target workloads and source amortization."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from semcache.diagnostics.r1b3_b3_runtime import Runtime,common_arguments,config
from semcache.diagnostics.r1b3_break_even import (create_run,failure,schedule,target_variants,workload_target,summarize_pairs,MissingEvidence,ParityFailure)
from semcache.diagnostics.r1b3_fp16_divergence import logit_metrics
from semcache.edgelora.batched_mixed_projection import compare_projections
from semcache.metrics.r1 import atomic_json,raw_only_guard,digest
from semcache.metrics.r1b2 import coverage_value,csv_write


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__); common_arguments(p)
    p.add_argument('--profile',choices=('smoke','pilot','formal'),default='smoke')
    p.add_argument('--coverages',default='max'); p.add_argument('--workloads',default='warm',help='warm,cold,mixed,noncontiguous')
    p.add_argument('--executors',choices=('primary','all'),default='primary',help='primary = Legacy and Contiguous; Full always paired')
    p.add_argument('--instrumentation',choices=('minimal','breakdown'),default='minimal')
    p.add_argument('--warmup-runs',type=int); p.add_argument('--measured-runs',type=int)
    p.add_argument('--order-seed',type=int,default=101)
    a=p.parse_args(argv); a.lengths=[int(x) for x in a.prompt_lengths.split(',')]
    a.coverage_values=[coverage_value(x) for x in a.coverages.split(',')]; a.workload_values=a.workloads.split(',')
    a.modes=['legacy','v1','v2','contiguous'] if a.executors=='all' else ['legacy','contiguous']
    a.warmup_runs=a.warmup_runs if a.warmup_runs is not None else (1 if a.profile=='smoke' else 3)
    a.measured_runs=a.measured_runs if a.measured_runs is not None else (2 if a.profile=='smoke' else 10 if a.profile=='pilot' else 30)
    if (min(a.lengths)<4 or any(len(v)!=len(set(v)) for v in (a.lengths,a.coverage_values,a.workload_values)) or
        any(w not in ('warm','cold','mixed','noncontiguous') for w in a.workload_values) or a.warmup_runs<1 or a.measured_runs<1 or a.capacity_mib<1): p.error('Invalid cases/counts')
    if a.profile!='smoke' and (a.warmup_runs<3 or a.measured_runs<(30 if a.profile=='formal' else 3)): p.error('Use >=3 warmups; formal requires >=30 measured pairs')
    if a.cache_device=='cuda' and not a.device.startswith('cuda'): p.error('GPU cache requires CUDA model')
    return a


def reports(root,rows):
    fields=('row_id','length','requested_coverage','actual_coverage','workload','executor','phase','repeat','order','instrumentation',
        'independent_run_id','order_seed','source_snapshot_id','target_identity','adapter','dtype','seed','code_identity','model_revision','namespace','weight_identity')
    def export(name,extras): csv_write(root/name,[{k:r.get(k) for k in (*fields,*extras)} for r in rows])
    export('latency_comparison.csv',('full_ms','full_prefill_ms','reuse_ms','selection_ms','mapping_ms','context_setup_ms','executor_parity','correctness_status'))
    export('paired_differences.csv',('paired_difference_ms','speedup_vs_full','paired_win','executor_parity','correctness_status'))
    export('coverage_sweep.csv',('reused_tokens','recomputed_tokens','skipped_projection_rows','native_projection_rows','selection_hash','selected_spans','prefix_valid','assembly_path'))
    export('cold_warm_mapping.csv',('mapping_cache_hit','mapping_ms','context_setup_ms','cache_statistics'))
    export('source_cost_accounting.csv',('source_costs',))
    export('memory_accounting.csv',('memory',))
    export('correctness_comparison.csv',('quality','executor_parity','qkv_validation','correctness_status'))
    export('executor_breakdown.csv',('setup_breakdown','executor_breakdown'))
    summary=summarize_pairs(rows)
    atomic_json(root/'break_even_summary.json',dict(groups=summary,baseline='FULL_RECOMPUTE',classification='ORACLE_DIAGNOSTIC'))
    atomic_json(root/'summary.json',dict(groups=summary,rows=len(rows),outliers_removed=False,
        independent_run_inference='Use 81 aggregator; one run is descriptive only'))


def run(a,manifest):
    import torch
    runtime=Runtime(a); manifest['runtime']=runtime.provenance; rows=[]; gpu=[]
    for length in a.lengths:
        for coverage in a.coverage_values:
            for workload in a.workload_values:
                case=f'{length}-{coverage}-{workload}-{a.instrumentation}'
                manifest['active_case']=dict(case=case,phase='source_build')
                runtime.source(length)
                warm=dict(text=runtime.text,ids=runtime.ids,identity=digest(runtime.ids))
                count=a.warmup_runs+a.measured_runs
                cold=target_variants(runtime.tokenizer,runtime.ids,runtime.text,count,a.seed+length,runtime.accepts) if workload in ('cold','mixed') else []
                targets=[workload_target(workload,i,warm,cold) for i in range(count)]
                orders=(schedule(['full',*a.modes],a.warmup_runs,a.order_seed+length) +
                        schedule(['full',*a.modes],a.measured_runs,a.order_seed+length+1000003))
                manifest.setdefault('cases',{})[case]=dict(source_ids=runtime.ids,source_snapshot_id=runtime.snapshot,memory_preflight=runtime.preflight,
                    fixtures={t['identity']:dict(ids=t['ids'],text=t['text']) for t in targets},orders=orders,
                    fixture_generation='offline tokenizer roundtrip + source-cluster preflight; actual semantic processing remains timed',
                    filter='runner-only every-second selected window' if workload=='noncontiguous' else None)
                atomic_json(a.output_dir/'manifest.json',manifest)
                gpu.append(dict(case=case,phase='before',state=runtime.gpu_state())); atomic_json(a.output_dir/'gpu_state.json',gpu)
                block=[]
                for iteration,(target,order) in enumerate(zip(targets,orders)):
                    phase='warmup' if iteration<a.warmup_runs else 'measured'; repeat=iteration if phase=='warmup' else iteration-a.warmup_runs
                    outputs={}; executions={}
                    for mode in order:
                        manifest['active_case']=dict(case=case,phase=phase,repeat=repeat,executor=mode,target_identity=target['identity'])
                        if mode=='full':
                            full=runtime.full(target['text'],detailed=a.instrumentation=='breakdown')
                            if full['ids']!=target['ids']: raise MissingEvidence('Paired Full tokens differ from target fixture')
                            outputs[mode]=full['logits']
                        else:
                            outputs[mode],executions[mode]=runtime.request(target['text'],target['ids'],mode,coverage,workload,instrumentation=a.instrumentation)
                    base=executions['legacy']
                    for mode in a.modes:
                        e=executions[mode]
                        parity=bool(torch.equal(outputs['legacy'],outputs[mode]) and base['projection_audit']==e['projection_audit'] and base['selection_hash']==e['selection_hash'])
                        if workload=='cold' and e['reused_tokens'] and mode in ('v2','contiguous') and e['mapping_cache_hit'] is not False:
                            raise MissingEvidence('Previously unseen cold target unexpectedly reused its mapping')
                        if e['reused_tokens'] and mode in ('v2','contiguous'):
                            expected_hit=target['identity']==warm['identity'] and iteration>0
                            if e['mapping_cache_hit'] is not expected_hit:
                                raise MissingEvidence('Actual mapping hit/miss does not match the exact-target workload schedule')
                        if workload in ('cold','mixed') and coverage!=0 and not e['reused_tokens']: raise MissingEvidence('Cold valid fixture lost all feasible reuse')
                        q=logit_metrics(outputs['full'],outputs[mode]); full_ms=full['timing']['request_wall_ms']; reuse_ms=e['request_ms']
                        temporary=e['memory']['temporary']
                        if mode=='contiguous' and e['reused_tokens'] and temporary.get('fallback_group_count') is None:
                            raise MissingEvidence('Contiguous executor did not provide assembly-path audit counters')
                        assembly=('NO_REUSE' if not e['reused_tokens'] else 'WINDOW_LOOP' if mode=='legacy' else
                            'GATHER_SCATTER' if mode in ('v1','v2') else
                            'GATHER_FALLBACK_USED' if temporary.get('fallback_group_count',0)>0 else 'CONTIGUOUS_ONLY')
                        if workload=='noncontiguous' and mode=='contiguous' and len(e['selected_provenance'])>1 and assembly!='GATHER_FALLBACK_USED':
                            raise MissingEvidence('Noncontiguous multi-window fixture did not exercise gather fallback')
                        costs=dict(runtime.costs)
                        if mode in ('legacy','v1'): costs['incremental_build_ms']-=costs['descriptor_ms']
                        record=dict(code_identity=manifest['code']['sha256'],model_revision=runtime.meta['resolved_model_revision'],namespace=runtime.namespace,weight_identity=runtime.weights,
                            row_id=f'{case}-{phase}-{repeat}-{mode}',length=length,requested_coverage=coverage,actual_coverage=e['actual_coverage'],
                            workload=workload,executor=mode,phase=phase,repeat=repeat,order=order,instrumentation=a.instrumentation,
                            independent_run_id=a.independent_run_id,order_seed=a.order_seed,seed=a.seed,dtype=a.dtype,adapter='user_a',
                            source_snapshot_id=case+':'+runtime.snapshot,target_identity=target['identity'],
                            full_ms=full_ms,full_prefill_ms=full['timing']['prefill_wall_ms'],reuse_ms=reuse_ms,paired_difference_ms=full_ms-reuse_ms,
                            speedup_vs_full=full_ms/reuse_ms,paired_win=full_ms>reuse_ms,executor_parity=parity,qkv_validation='PENDING_UNTIMED_VALIDATION',
                            quality=q,correctness_status=q['correctness_status'],reused_tokens=e['reused_tokens'],recomputed_tokens=length-e['reused_tokens'],
                            skipped_projection_rows=e['projection_audit']['skipped_projection_rows_all_layers_roles'],native_projection_rows=e['projection_audit']['native_projection_rows_all_layers_roles'],
                            selection_hash=e['selection_hash'],selected_spans=[dict(source=p['source'],target=p['target'],
                                source_query_id=p['query'],prefix_token_ids_equal=p['prefix_token_ids_equal'],
                                adapter_identity_match=p['adapter_identity_match']) for p in e['selected_provenance']],
                            prefix_valid=all(p['prefix_token_ids_equal'] for p in e['selected_provenance']),assembly_path=assembly,source_costs=costs,
                            **{k:e[k] for k in ('mapping_cache_hit','mapping_ms','context_setup_ms','selection_ms','memory','cache_statistics','setup_breakdown','executor_breakdown')})
                        block.append(record)
                    outputs.clear()
                    atomic_json(a.output_dir/'checkpoints'/f'{case}.json',dict(identity=manifest['identity'],status='RUNNING',rows=block,checksum=digest(block)))
                gpu.append(dict(case=case,phase='after',state=runtime.gpu_state())); atomic_json(a.output_dir/'gpu_state.json',gpu)
                # All passive QKV copies occur after the complete timing block. Each
                # unique target is checked with the exact same source snapshot/hits.
                for target in {t['identity']:t for t in targets}.values():
                    manifest['active_case']=dict(case=case,phase='untimed_qkv_validation',target_identity=target['identity'])
                    baseline,ref=runtime.request(target['text'],target['ids'],'legacy',coverage,workload,capture_projections=True)
                    refq=ref.pop('_qkv'); mask=[False]*length
                    for p in ref['selected_provenance']: mask[p['target'][0]:p['target'][1]]=[True]*(p['target'][1]-p['target'][0])
                    for mode in a.modes:
                        actual,e=runtime.request(target['text'],target['ids'],mode,coverage,workload,capture_projections=True)
                        qkv=compare_projections(refq,e.pop('_qkv'),mask)
                        valid=bool(qkv['passed'] and torch.equal(baseline,actual) and e['selection_hash']==ref['selection_hash'] and e['projection_audit']==ref['projection_audit'])
                        for r in block:
                            if r['target_identity']==target['identity'] and r['executor']==mode:
                                r['qkv_validation']='PASS' if valid else 'FAIL'; r['executor_parity']=r['executor_parity'] and valid
                    refq.clear()
                rows.extend(block)
                atomic_json(a.output_dir/'checkpoints'/f'{case}.json',dict(identity=manifest['identity'],status='COMPLETE',rows=block,checksum=digest(block)))
                reports(a.output_dir,rows)
                if any(not r['executor_parity'] for r in block): raise ParityFailure(f'Executor parity failed in {case}; all timing rows preserved')
    manifest.update(execution_status='COMPLETE',correctness_status='STRICT_PASS' if all(r['correctness_status']=='STRICT_PASS' for r in rows) else 'STRICT_FAIL')
    atomic_json(a.output_dir/'manifest.json',manifest)


def main(argv=None):
    a=parse_args(argv); manifest=create_run(ROOT,a.output_dir,config(a),'R1-B3-B3 break-even')
    reports(a.output_dir,[]); atomic_json(a.output_dir/'gpu_state.json',[])
    try:
        with raw_only_guard() as guard:
            run(a,manifest); manifest['raw_only']=guard.evidence()
        atomic_json(a.output_dir/'manifest.json',manifest); return 0
    except (Exception,KeyboardInterrupt) as exc: return failure(a.output_dir,manifest,exc)
if __name__=='__main__': raise SystemExit(main())
