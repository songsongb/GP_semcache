"""Read-only V5 aggregation; official identity checks remain authoritative."""
import csv
from pathlib import Path

from .communication import require
from .r5_d5_lightperf_io import read_json, verify, no_symlinks
from .r5_d5_lightperf_analysis import verify_run, compare, summarize
from .r5_d5_lightperf_replay_analysis import segment, run_segments, compare_segments
from .r5_d5_lightperf_workload import CONDITIONS
from .r5_d5_impact import implementation

LIMITATION = ('EXPLORATORY_SINGLE_TRIAL; 8-token cap and historically low F1 limit output correctness conclusions; '
              'APPROX output changes do not establish quality preservation; correctness-qualified speedup NOT_ESTABLISHED; '
              'retest promising cells in >=3 independent allocations with balanced orders; inclusive timers overlap')


def review_runs(runs, analysis, positions, bindings):
    require(len(runs)==3 and {r['condition'] for r in runs}==set(CONDITIONS), 'three distinct conditions required')
    require(analysis['bindings']==bindings, 'official analysis/run inventory binding differs')
    by={r['condition']:r for r in runs};full=by[CONDITIONS[0]]
    expected_positions=[dict(run_segments(r),impact_implementation=implementation(r)) for r in runs]
    require(positions['runs']==expected_positions, 'position_analysis differs from validated receipts')
    comparisons={}
    for condition in CONDITIONS[1:]:
        actual=compare(full,by[condition])
        stored=[c for c in analysis['comparisons'] if c.get('condition')==condition]
        # NOT_COMPARABLE records have no condition in the core compare return;
        # official analyze still records position comparisons by condition.
        if actual['status']=='NOT_COMPARABLE':
            stored=[c for c in analysis['comparisons'] if c.get('status')=='NOT_COMPARABLE' and
                    all(c.get(k)==v for k,v in actual.items())]
        require(stored and any(all(c.get(k)==v for k,v in actual.items()) for c in stored),
                'official comparison differs from validated runs')
        comparisons[condition]=actual
    expected_position_comparisons=[dict(dataset=by[c]['dataset'],scale=by[c]['scale'],condition=c,
        impact_implementation=implementation(by[c]),**compare_segments(full,by[c])) for c in CONDITIONS[1:]]
    require(positions['comparisons']==expected_position_comparisons, 'position comparison differs')
    require(all(len(r['requests'])==20 for r in runs), '20-request sweep only')
    exact=comparisons[CONDITIONS[1]]
    matched=all(c['status']=='DESCRIPTIVE_MATCHED_IDENTITY' for c in comparisons.values())
    gate='MATCHED_COMPLETE' if matched and exact.get('generated_token_agreements')==20 else 'PARTIAL_BLOCKED'
    return by,comparisons,gate


def validated_cell(attempt, expected_source=None, expected_tasks=None):
    attempt=no_symlinks(Path(attempt).absolute())
    registry=read_json(attempt/'registry.json')
    require(type(registry) is list and len(registry)==3,'cell registry must contain exactly three runs')
    require(all(set(t)=={'run','plan','cpu_audit'} for t in registry),'explicit run/plan/audit triples required')
    loaded=[verify_run(t['run'],t['plan'],t['cpu_audit']) for t in registry]
    runs=[r for r,b in loaded];bindings=[b for r,b in loaded]
    verify(attempt/'analysis')
    analysis=read_json(attempt/'analysis/analysis.json')
    positions=read_json(attempt/'analysis/position_analysis.json')
    by,comparisons,gate=review_runs(runs,analysis,positions,bindings)
    if expected_source is not None:
        require(all(r['source']==expected_source for r in runs),'cell source differs from sweep source')
    if expected_tasks is not None:
        require(all(r['tasks_sha256']==expected_tasks for r in runs),'cell workload differs from frozen sweep')
    submission=read_json(attempt/'submission.json')
    allocation=read_json(attempt/'allocation.json')
    from .r5_d5_sweep import cells
    expected=next(c['performance_knobs'] for c in cells() if c['id']==submission['cell'])
    require(all(r['performance_knobs']==expected for r in runs),'run knob/cell ID differs')
    # Official compare checks GPU/backend/cpu_concurrency exactly; also bind the
    # allocation manifest without requiring GPU identity across different cells.
    require(all(r['gpu']['uuid'].removeprefix('GPU-').lower()==allocation['gpu']['uuid'].removeprefix('GPU-').lower()
                and r['cpu_concurrency']['affinity']==allocation['affinity'] for r in runs),'allocation/run provenance differs')
    rows=[];details=[]
    for condition in CONDITIONS:
        run=by[condition];metrics=summarize(run)
        comparison=comparisons.get(condition)
        comparable=all(c['status']=='DESCRIPTIVE_MATCHED_IDENTITY' for c in comparisons.values())
        qualified=comparable and (condition!=CONDITIONS[1] or gate=='MATCHED_COMPLETE')
        agreements=20 if condition==CONDITIONS[0] else comparison.get('generated_token_agreements')
        common=dict(cell=submission['cell'],trial=submission['trial'],order=allocation['order'],condition=condition,
            bandwidth_mbps=expected['bandwidth_mbps'],ud_factor=expected['ud_throughput_factor'],
            gpu_uuid=run['gpu']['uuid'],node=allocation['node'],affinity=allocation['affinity'],
            cpu_concurrency=run['cpu_concurrency'],backend=run['backend'],
            source_commit=run['source']['commit'],source_fingerprint=run['source']['content_fingerprint'],
            comparability_status='DESCRIPTIVE_MATCHED_IDENTITY' if comparable else 'NOT_COMPARABLE',
            cell_gate=gate,exploratory=True,correctness_qualified_speedup='NOT_ESTABLISHED')
        whole=segment(run['requests'])
        row=dict(common,segment='all_20',requests=20,e2e_mean_ms=metrics['e2e_ms']['mean'],
            e2e_median_ms=metrics['e2e_ms']['median'],e2e_p95_ms=metrics['e2e_ms']['p95'],ttft_mean_ms=metrics['ttft_ms']['mean'],
            session_span_ms=run['sequence_span_ms'],full_session_span_ms=full_span(by),
            paired_session_span_delta_ms=0 if comparison is None else comparison.get('sequence_span_delta_ms') if qualified else None,
            paired_mean_e2e_delta_ms=0 if comparison is None else comparison.get('delta_e2e_ms')/20 if qualified else None,
            measured_descriptive_speedup=None if comparison is None or not qualified else comparison.get('speedup'),
            qkv_skip_ratio=whole['qkv_skip_ratio'],physical_hit_rate=whole['hit_rate'],
            send_bytes=whole['wire_bytes']['send'],recv_bytes=whole['wire_bytes']['recv'],
            quality_metric=metrics['quality']['metric'],quality_mean=metrics['quality']['mean'],
            generated_token_agreements=agreements,generated_token_agreement_rate=None if agreements is None else agreements/20,
            limitation=LIMITATION)
        rows.append(row)
        bins=[]
        for lo,hi,label in ((0,10,'first_10'),(10,20,'repeat_10')):
            subset=run['requests'][lo:hi];value=segment(subset)
            entry=dict(common,segment=label,requests=10,e2e_mean_ms=value['e2e_ms']['mean'],
                e2e_median_ms=value['e2e_ms']['median'],e2e_p95_ms=value['e2e_ms']['p95'],
                ttft_mean_ms=value['ttft_ms']['mean'],qkv_skip_ratio=value['qkv_skip_ratio'],physical_hit_rate=value['hit_rate'],
                send_bytes=value['wire_bytes']['send'],recv_bytes=value['wire_bytes']['recv'],
                cache_insert=value['policy_counts']['INSERT'],cache_evict=value['policy_counts']['EVICT'])
            for target,source in (('tcp_send_inclusive_ms','tcp_send_inclusive_ms'),('tcp_recv_inclusive_ms','tcp_recv_inclusive_ms'),
                                  ('owner_completion_ms','owner_completion_ms'),('prefill_ms','prefill_ms'),('decode_ms','decode_ms')):
                measured=value['timings'][source]
                entry[target]=None if measured['statistics'] is None else measured['statistics']['mean']
                entry[target+'_measurement_status']=measured['status']
            rows.append(entry);bins.append(dict(segment=label,statistics=value))
        details.append(dict(condition=condition,whole_session=whole,bins=bins))
    return dict(gate=gate,cell=submission['cell'],trial=submission['trial'],attempt=str(attempt),rows=rows,
                details=details,comparisons=comparisons,bindings=bindings,limitation=LIMITATION)


def full_span(by):
    return by[CONDITIONS[0]]['sequence_span_ms']


def aggregate(state, output):
    from .r5_d5_sweep import write_once,cells,SCHEMA
    state=no_symlinks(Path(state).absolute());output=no_symlinks(Path(output).absolute())
    require(not output.exists() and output.parent==state.parent and not output.is_relative_to(state),
            'aggregate output must be a fresh sibling of input state')
    definition=read_json(state/'definition.json')
    require(definition['schema']==SCHEMA,'unknown sweep definition')
    reviews=[];blocked=[]
    for payload in sorted((state/'attempts').glob('*/submission.json')):
        attempt=payload.parent
        try:
            reviews.append(validated_cell(attempt,definition['source'],definition['reference']['tasks_sha256']))
        except (ValueError,OSError,KeyError,TypeError) as exc:
            blocked.append(dict(attempt=str(attempt),status='PARTIAL_BLOCKED',reason=str(exc)))
    output.mkdir(parents=True)
    rows=[r for review in reviews for r in review['rows']]
    for segment_name,filename in (('all_20','grid.csv'),('bins','first10_repeat10.csv')):
        selected=[r for r in rows if (r['segment']=='all_20')==(segment_name=='all_20')]
        fields=sorted({k for r in selected for k in r})
        with (output/filename).open('x',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
            for row in selected:
                writer.writerow({k:__import__('json').dumps(v,sort_keys=True) if isinstance(v,(dict,list)) else v for k,v in row.items()})
    report=dict(schema=SCHEMA,status='EXPLORATORY_AGGREGATE',matrix=cells(),cells=reviews,blocked=blocked,
                unmeasured_cells=[c['id'] for c in cells() if not any(r['cell']==c['id'] and r['gate']=='MATCHED_COMPLETE' for r in reviews)],
                limitation=LIMITATION,correctness_qualified_speedup='NOT_ESTABLISHED')
    write_once(output/'sweep.json',report)
    return dict(status=report['status'],validated_attempts=len(reviews),blocked=len(blocked),output=str(output))
