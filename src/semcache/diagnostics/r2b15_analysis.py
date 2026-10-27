"""Scalar-only B1.5 evidence gates and disjoint timing summaries."""
from collections import defaultdict
from semcache.diagnostics.r2b1_prefix_cache import PrefixError
from semcache.metrics.r1b2 import stats

REQUEST_MODES=('FULL_USE_CACHE','WARM_NORMAL','WARM_PREPARED_DIAGNOSTIC')


def resolve_layers(value,count):
    if value=='all': return list(range(count))
    result=[count-1 if x.strip()=='last' else int(x) for x in value.split(',')]
    if not result or len(set(result))!=len(result) or any(x<0 or x>=count for x in result):
        raise PrefixError('REJECT_CAPTURE_LAYERS','invalid or duplicate layer indices')
    return result


def alignment(shape,kind,start,stop,kv_stop=None):
    """Declared token axes, never broadcast or infer token axis from tensor length."""
    if kind=='tokens':
        if len(shape)!=3 or shape[0]!=1: raise PrefixError('REJECT_ALIGNMENT','expected [1,tokens,features]')
        if not 0<=start<stop<=shape[1]: raise PrefixError('REJECT_ALIGNMENT','invalid token slice')
        return (1,start,stop,None)
    if kind=='attention':
        if len(shape)!=4 or shape[0]!=1 or not 0<=start<stop<=shape[2]: raise PrefixError('REJECT_ALIGNMENT','expected [1,heads,query,key]')
        if kv_stop is None or not 0<kv_stop<=shape[3]: raise PrefixError('REJECT_ALIGNMENT','invalid KV extent')
        return (2,start,stop,kv_stop)
    raise PrefixError('REJECT_ALIGNMENT','unknown tensor coordinate convention')


def first_findings(rows):
    result={}
    for comparison in dict.fromkeys(r['comparison'] for r in rows):
        group=sorted((r for r in rows if r['comparison']==comparison),key=lambda r:r['execution_order'])
        result[comparison]=dict(first_bitwise_difference=next((r for r in group if not r.get('bitwise_equal',r['exact_equal'])),None),first_nonzero=next((r for r in group if not r['exact_equal']),None),
            first_tolerance_failure=next((r for r in group if not r['allclose']),None),
            status='OBSERVED_DIFFERENCE' if any(not r['exact_equal'] for r in group) else 'NO_DIFFERENCE_AT_CAPTURED_BOUNDARIES',
            scope='first observed hook boundary, actual chronological order; not an identified CUDA kernel')
    return result


def timing_summary(rows):
    pairs=defaultdict(dict); grouped=defaultdict(list)
    for r in rows:
        if r['phase']!='measured': continue
        if r['mode'] not in REQUEST_MODES or r['instrumentation']!='minimal':
            raise PrefixError('REJECT_TIMING_MIX','component/instrumented samples cannot enter minimal paired statistics')
        if r['wall_ms']<=0: raise PrefixError('REJECT_TIMING_MIX','positive measured request wall required')
        key=(r['case'],r['repeat'])
        if r['mode'] in pairs[key]: raise PrefixError('REJECT_TIMING_MIX','duplicate paired sample')
        pairs[key][r['mode']]=r; grouped[r['case']].append(r)
    out=[]
    for case,block in grouped.items():
        values=[p for (c,_),p in sorted(pairs.items()) if c==case]
        if any(set(p)!=set(REQUEST_MODES) for p in values): raise PrefixError('MISSING_PAIRED_EVIDENCE','incomplete Full/Normal/Prepared pair')
        for mode in REQUEST_MODES:
            data=[p[mode]['wall_ms'] for p in values]
            base=[p['FULL_USE_CACHE']['wall_ms'] for p in values]
            delta=[a-b for a,b in zip(base,data)]
            isolation=[p['WARM_NORMAL']['wall_ms']-p['WARM_PREPARED_DIAGNOSTIC']['wall_ms'] for p in values]
            out.append(dict(case=case,mode=mode,count=len(data),**stats(data),full_minus_mode=stats(delta),
                full_relative_wins=sum(d>0 for d in delta),normal_minus_prepared=stats(isolation),observed_full_ratio=sum(base)/sum(data),
                prepared_is_ideal_diagnostic=mode=='WARM_PREPARED_DIAGNOSTIC',correctness_qualified_speedup=False,
                inference='one exploratory run; no independence or service speedup inference',
                isolation_scope='Normal-Prepared changes preparation placement and request-entry state, not clone-only causal cost'))
    return out


def cost_decisions(summary):
    groups=defaultdict(dict)
    for r in summary: groups[r['case']][r['mode']]=r
    result={}
    for case,modes in groups.items():
        p=modes['WARM_PREPARED_DIAGNOSTIC']; normal=modes['WARM_NORMAL']
        delta=p['normal_minus_prepared']['mean']; gap=normal['mean']-modes['FULL_USE_CACHE']['mean']
        result[case]={
            'B-H1':dict(status='UNRESOLVED',normal_minus_prepared_ms=delta,
                reason='Prepared contrast bundles cache-entry/scheduling effects; standalone preparation measured separately'),
            'B-H2':dict(status='SUPPORTED' if p['full_minus_mode']['mean']<=0 else 'CONTRADICTED',
                reason='descriptive: prepared request remains >=Full, or is lower, in this one measured session; not CPU/GPU attribution',
                ideal_diagnostic_only=True),
            'B-H3':dict(status='UNRESOLVED',reason='Shape/launch mechanism not isolated by request wall'),
            'B-H4':dict(status='UNRESOLVED',reason='No output-attention OFF ablation; native forward bundles output KV and LM head'),
            'unattributed_gap_ms':None,'observed_normal_minus_full_ms':gap,
            'correctness_qualified_speedup':False}
    return result
