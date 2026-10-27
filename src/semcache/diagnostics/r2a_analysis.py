"""R2-A scalar contracts. No profiler/minimal or inclusive/exclusive summation."""
from collections import defaultdict
import statistics
from semcache.metrics.r1b2 import stats
from semcache.edgelora.batched_mapping_optimized import paired_statistics
from semcache.diagnostics.r1b3_break_even import schedule

MODES=('FULL_NATIVE','LOOKUP_NO_REUSE','CONTIGUOUS_WARM','ORACLE_PRESELECTED')
BRIDGES=(('CONTIGUOUS_WARM','FULL_NATIVE'),('LOOKUP_NO_REUSE','FULL_NATIVE'),
         ('CONTIGUOUS_WARM','ORACLE_PRESELECTED'),('ORACLE_PRESELECTED','FULL_NATIVE'))


def summaries(rows,seed=101):
    groups=defaultdict(list); pairs=defaultdict(dict)
    for r in rows:
        if r['phase']!='measured': continue
        if r['profiler_enabled']: raise ValueError('Profiler samples cannot enter minimal analysis')
        key=(r['length'],r['repeat'])
        if r['mode'] in pairs[key]: raise ValueError('Duplicate pair member')
        pairs[key][r['mode']]=r['wall_ms']; groups[(r['length'],r['mode'])].append(r['wall_ms'])
    summary=[dict(length=k[0],mode=k[1],**stats(v)) for k,v in groups.items()]
    bridge=[]
    for length in sorted({k[0] for k in pairs}):
        block=[v for k,v in pairs.items() if k[0]==length]
        if any(set(v)!=set(MODES) for v in block): raise ValueError('Incomplete measured pair')
        for x,y in BRIDGES:
            s=paired_statistics([v[x] for v in block],[v[y] for v in block],seed)
            bridge.append(dict(length=length,x=x,y=y,definition='D(X,Y)=X-Y; positive X slower',
                mean_ms=s['mean'],p50_ms=s['median'],ci95=s['bootstrap_mean_ci95'],
                y_faster_count=s['optimized_win_count'],count=len(block),
                differences_ms=[v[x]-v[y] for v in block],scope='same-session minimal; no outliers removed'))
    return summary,bridge


def decisions(bridges):
    result=[]
    for length in sorted({r['length'] for r in bridges}):
        by={(r['x'],r['y']):r for r in bridges if r['length']==length}
        h={k:dict(status='UNRESOLVED',reason='Requires attributed GPU/host component evidence; not inferred from row coverage') for k in ('H-A','H-B','H-C')}
        for name,key in [('H-D',BRIDGES[1]),('H-E',BRIDGES[3])]:
            r=by[key]; ci=r['ci95']
            status='SUPPORTED' if ci and ci[0]>0 else 'CONTRADICTED' if ci and ci[1]<0 else 'UNRESOLVED'
            h[name]=dict(status=status,evidence=r,scope='exploratory one-run fixture only; Oracle retains executor validation')
        result.append(dict(length=length,hypotheses=h,unattributed_residual_ms=None,
            residual_reason='Overlapping profiler/component costs cannot be subtracted from minimal wall',safe_reuse_claimed=False))
    return result


def union_us(intervals):
    """Union GPU activity intervals, never sum overlapping kernels across streams."""
    total=0.; end=None
    for start,stop in sorted(intervals):
        if stop<start: raise ValueError('Invalid interval')
        total+=max(0.,stop-max(start,end if end is not None else start)); end=max(stop,end if end is not None else stop)
    return total
