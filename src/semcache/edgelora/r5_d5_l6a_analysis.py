"""Descriptive long replay analysis from three verified, plan-bound Light receipts."""
import csv
import io
import json
from statistics import mean

from .communication import require
from .r5_d5_l6a import CYCLES, output
from .r5_d5_lightperf_analysis import compare, verify_run
from .r5_d5_lightperf_io import read_json
from .r5_d5_lightperf_replay_analysis import measured, segment
from .r5_d5_lightperf_workload import CONDITIONS


def cache_snapshots(rows):
    # Each row samples its active owner. Idle owners persist across requests/cycles.
    states = {aid: (0,0) for aid in {r['adapter_id'] for r in rows}}
    snapshots = []
    for r in rows:
        count, size = r['cache_entry_count'], r['cache_occupancy_bytes']
        require(type(count) is int and 0 <= count <= 64 and type(size) is int and
                0 <= size <= 512*2**20, 'per-adapter cache bound differs')
        states[r['adapter_id']] = count, size
        snapshots.append(states.copy())
    return snapshots


def occupancy(samples):
    def stats(values):
        return dict(mean=mean(values),maximum=max(values),final=values[-1])
    def metrics(states):
        return {key:stats([s[index] for s in states])
                for index,key in enumerate(('entry_count','physical_bytes'))}
    return dict(per_adapter={aid:metrics([s[aid] for s in samples]) for aid in sorted(samples[0])},
                total=metrics([tuple(map(sum,zip(*s.values()))) for s in samples]))


def summarize(runs, stream, *, fixture=False):
    require([r['condition'] for r in runs]==list(CONDITIONS),'condition coverage/order differs')
    full = runs[0]
    for r in runs:
        require(r['scale']==len(r['requests'])==50 and r['tasks_sha256']==stream['tasks_sha256']
                and [x['replay_provenance'] for x in r['requests']]==stream['occurrences']
                and [x['request_id'] for x in r['requests']]==[t['request_id'] for t in stream['tasks']],
                'long run workload/count differs')
    for r in runs[1:]:
        pair=compare(full,r)
        require(pair['status']=='DESCRIPTIVE_MATCHED_IDENTITY','NOT_COMPARABLE: '+str(pair.get('mismatches')))
        require(r['condition']!=CONDITIONS[1] or pair['generated_token_agreements']==50,'EXACT token mismatch')
    report=dict(scope='FIXTURE/NOT_MEASURED' if fixture else 'DESCRIPTIVE_SINGLE_TRIAL',
                correctness_qualified_speedup='NOT_ESTABLISHED',quality_preservation='NOT_ESTABLISHED',
                sequence_span_ms={r['condition']:r['sequence_span_ms'] for r in runs},
                cycle_metrics=[],cumulative_metrics=[])
    for run in runs:
        rows=run['requests']
        snapshots=cache_snapshots(rows)
        for cycle in CYCLES:
            stop=cycle['last']+1
            for group,start in (('cycle_metrics',cycle['first']),('cumulative_metrics',0)):
                subset=rows[start:stop]
                ref=full['requests'][start:stop]
                metrics=segment(subset)
                del metrics['cache_occupancy_bytes']  # replace active-owner samples with end-of-request snapshots
                metric_names={r['quality']['metric'] for r in subset+ref}
                require(len(metric_names)==1,'quality metric differs')
                delta=[b['application_e2e_ms']-a['application_e2e_ms'] for a,b in zip(ref,subset)]
                denominator=sum(r['application_e2e_ms'] for r in subset)
                metrics.update(scope=report['scope'], condition=run['condition'],cycle=cycle['cycle'],
                    first=start,last=stop-1,cache=occupancy(snapshots[start:stop]),
                    paired_condition_minus_full_e2e_ms=delta,mean_delta_e2e_ms=mean(delta),
                    descriptive_full_over_condition_e2e=sum(r['application_e2e_ms'] for r in ref)/denominator
                        if denominator else None,
                    generated_token_agreements=sum(a['generated_token_ids']==b['generated_token_ids'] for a,b in zip(ref,subset)),
                    attention_impact_ms=measured([(r.get('attention_impact') or {}).get('elapsed_ms') for r in subset]),
                    integrity={k:measured([(r.get('integrity_validation') or {}).get(k) for r in subset])
                               for k in sorted({k for r in subset for k in (r.get('integrity_validation') or {})
                                                if k.endswith(('_bytes','_ms','_count'))})},
                    quality_metric=next(iter(metric_names)),quality_mean=mean(r['quality']['score'] for r in subset),
                    wire_bytes_per_request={op:n/len(subset) for op,n in metrics['wire_bytes'].items()},
                    wire_pacing_ms={op:measured([r['wire'][op].get('added_pacing_ms') for r in subset])
                                    for op in ('send','recv')})
                report[group].append(metrics)
    return report


def analyze(plan,args):
    require(plan['entry_limit']==64,'ENTRY128_GPU_NOT_APPROVED')
    light=read_json(args.light_plan/'plan.json')
    require(light.get('l6_plan_id')==plan['plan_id'] and light.get('entry_limit')==64 and
            all(light[k]==plan[k] for k in ('source','stream','resources','dataset','count','thresholds',
                'validation_mode','conditions','seed','performance_knobs','impact_implementation','transport',
                'base_prepared','multi_inventory','original_inventory','semantic_identity','tokenizer_identity',
                'dataset_inventories','workload_mode','enable_continuous_replay','warmup_requests')),
            'Light plan not bound to reviewed L6 contract')
    paths=[args.full,args.exact,args.approx]
    verified=[verify_run(path,args.light_plan,args.light_audit) for path in paths]
    report=summarize([r for r,_ in verified],plan['stream'])
    report.update(plan_id=plan['plan_id'],bindings=[b for _,b in verified])
    return write_report(report,args.output,[args.plan,args.light_plan,args.light_audit,*paths])


def write_report(report,path,protected):
    w=output(path,protected)
    w.write('long_replay_analysis.json',report)
    for group in ('cycle_metrics','cumulative_metrics'):
        rows=report[group]
        buf=io.StringIO()
        writer=csv.DictWriter(buf,fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k:json.dumps(v,allow_nan=False) if isinstance(v,(dict,list)) else v for k,v in row.items()})
        w.write_text(group+'.csv',buf.getvalue())
    return dict(status='DESCRIPTIVE_ANALYSIS_COMPLETE',inventory=w.seal())
