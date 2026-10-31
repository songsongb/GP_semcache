"""Consolidation requires all planned immutable blocks; CI units preserve dependence."""
from collections import defaultdict
import csv
import json
from pathlib import Path
from .communication import require
from .r3a_artifacts import read,write
from .r5_i5_evidence import fresh,seal,verify
from .r5_i5_analysis import analyze,distribution,grouped_ci,METRICS
from .r5_i5_quality import aggregate
from .r5_i5_workload import CONDITIONS


def dependency_clusters(rows):
    """Merge block units that share a conversation; repetitions remain dependent."""
    parent={r['independent_block_group']:r['independent_block_group'] for r in rows};seen={}
    def root(x):
        while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
        return x
    for r in rows:
        b=r['independent_block_group'];g=r['group_id']
        if g in seen:parent[root(b)]=root(seen[g])
        else:seen[g]=b
    return [dict(r,independent_block_group=root(r['independent_block_group'])) for r in rows]


def summarize(experiment_path, runs, output):
    verify(experiment_path);plan=read(Path(experiment_path)/'i5_workload_manifest.json');expected={b['block_id'] for b in plan['blocks']}
    require(len(expected)==len(plan['blocks']), 'duplicate planned block identity')
    reports=[];seen=set()
    for path in runs:
        r=analyze(path);block=r['config']['block'];bid=block['block_id']
        require(r['config']['experiment_id']==plan['experiment_id'] and bid in expected and bid not in seen,'unmatched/duplicate block; explicitly select one complete retry')
        seen.add(bid);reports.append(r)
    require(seen==expected,f'MISSING_BLOCKS: expected={len(expected)} complete={len(seen)}; missing={sorted(expected-seen)}')
    out=fresh(output,(*runs,experiment_path));groups=defaultdict(list);pairs=defaultdict(list);qd=defaultdict(list);flat=[]
    for r in reports:
        b=r['config']['block'];key=(b['dataset'],b['track'],b['prefix_length'],b['scale'])
        groups[key].append(r)
        for p in r['paired_latency']:pairs[key+(p['comparison'],)].append(p)
        for q in r['quality_deltas']:qd[key+(q['comparison'],)].append(q)
        qmap={(q['condition'],q['request_id']):q for q in r['quality_per_sample']}
        for row in r['request_metrics']:
            if not row['scored']:continue
            q=qmap[row['condition'],row['request_id']];baseline=qmap[CONDITIONS[0],row['request_id']]
            flat.append({k:row[k] for k in ('block_id','dataset','condition','track','scale','repetition','request_id','group_id',
                'sample_id','prefix_length_target','input_tokens','reused_prefix_tokens','reused_input_fraction','application_ttft_ms',
                'model_ttft_ms','application_e2e_ms','model_execution_e2e_ms','total_request_wire_bytes','truncated','TEST_FIXTURE_ONLY')} |
                dict(hit=bool(row['source']),task_metric=q['metric'],task_score=q['score'],quality_delta_from_A=q['score']-baseline['score']))
    tables=[];delta_tables=[];break_even=[]
    for key,reps in groups.items():
        dataset,track,prefix,scale=key
        for c in CONDITIONS:
            rows=[x for r in reps for x in r['request_metrics'] if x['condition']==c and x['scored']]
            quality=[q for r in reps for q in r['quality_per_sample'] if q['condition']==c]
            tables.append(dict(dataset=dataset,track=track,prefix_length=prefix,scale=scale,condition=c,blocks=len(reps),TEST_FIXTURE_ONLY=plan['TEST_FIXTURE_ONLY'],
                unique_samples=len({x['sample_id'] for x in rows}),repeated_observations=len(rows),
                quality=aggregate(quality),latency={m:distribution([x[m] for x in rows]) for m in METRICS},
                hit_rate=sum(bool(x['source']) for x in rows)/len(rows),wire_bytes=sum(x['total_request_wire_bytes'] for x in rows),
                per_repetition={str(rep):dict(requests=sum(x['repetition']==rep for x in rows),
                    application_e2e_ms=distribution([x['application_e2e_ms'] for x in rows if x['repetition']==rep])) for rep in sorted({x['repetition'] for x in rows})}))
        if track=='CONTROLLED':
            a=CONDITIONS[0];c=CONDITIONS[2]
            warm=[r['latency_summary'][c]['target_wall_ms']/r['latency_summary'][c]['target_requests']-
                  r['latency_summary'][a]['target_wall_ms']/r['latency_summary'][a]['target_requests'] for r in reps]
            amort=[r['latency_summary'][c]['amortized_ms_per_target']-r['latency_summary'][a]['amortized_ms_per_target'] for r in reps]
            break_even.append(dict(dataset=dataset,scale=scale,prefix_length=prefix,measured_blocks=len(reps),
                warm_target_mean_delta_ms=sum(warm)/len(warm),amortized_mean_delta_ms=sum(amort)/len(amort),
                observed_amortized_benefit=sum(amort)<0,interpolated=False,statistical_break_even_established=False))
    for key,ps in pairs.items():
        ps=dependency_clusters(ps);quality=dependency_clusters(qd[key]);entry=dict(dataset=key[0],track=key[1],prefix_length=key[2],scale=key[3],comparison=key[4],metrics={})
        for metric in METRICS:
            values=[dict(p,absolute_delta=p['metrics'][metric]['absolute_delta']) for p in ps]
            entry['metrics'][metric]=dict(delta=distribution([x['absolute_delta'] for x in values]),
                percentage_delta=distribution([p['metrics'][metric]['percentage_delta'] for p in ps if p['metrics'][metric]['percentage_delta'] is not None]),
                confidence_interval=grouped_ci(values))
        entry['quality_delta']=dict(delta=distribution([q['absolute_delta'] for q in quality]),confidence_interval=grouped_ci(quality),
            by_target_outcome={label:distribution([q['absolute_delta'] for q in quality if q['hit']==hit]) for label,hit in (('HIT',True),('MISS',False))})
        delta_tables.append(entry)
    write(out/'i5_consolidated_tables.json',tables);write(out/'i5_paired_comparisons.json',delta_tables)
    write(out/'i5_break_even.json',dict(points=break_even,interpretation='only measured points; no inferred/interpolated crossing; instrumented localhost timing'))
    write(out/'i5_independent_audit.json',dict(status='PASS',complete_blocks=len(seen),expected_blocks=len(expected),
        experiment_id=plan['experiment_id'],GPU_NOT_RUN=plan['TEST_FIXTURE_ONLY'],historical_numerical_root_cause='ROOT_CAUSE_UNRESOLVED',
        source_block_manifests={str(p):read(Path(p)/'i5_output_manifest.json') for p in runs}))
    with (out/'i5_plot_data.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    with (out/'i5_consolidated_tables.csv').open('x',newline='') as f:
        rows=[dict(dataset=t['dataset'],track=t['track'],prefix_length=t['prefix_length'],scale=t['scale'],condition=t['condition'],TEST_FIXTURE_ONLY=plan['TEST_FIXTURE_ONLY'],
            quality_metric=t['quality']['metric'],quality_score=t['quality']['score'],hit_rate=t['hit_rate'],wire_bytes=t['wire_bytes'],
            mean_app_ttft_ms=t['latency']['application_ttft_ms']['mean'],mean_app_e2e_ms=t['latency']['application_e2e_ms']['mean']) for t in tables]
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    seal(out);return out
