#!/usr/bin/env python3
"""Independent B3 runs: FULL-relative hierarchical bootstrap and correctness gates."""
import argparse,json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from semcache.diagnostics.r1b3_break_even import hierarchical
from semcache.metrics.r1 import atomic_json,digest
from semcache.metrics.r1b2 import csv_write,stats


def load_run(path):
    manifest=json.loads((path/'manifest.json').read_text())
    if manifest['execution_status']!='COMPLETE' or manifest['classification']!='ORACLE_DIAGNOSTIC': raise ValueError('Completed ORACLE run required')
    files=sorted((path/'checkpoints').glob('*.json'))
    if {p.stem for p in files}!=set(manifest['cases']): raise ValueError('Missing case checkpoint')
    rows=[]; seen=set()
    for checkpoint in files:
        bundle=json.loads(checkpoint.read_text())
        if bundle['identity']!=manifest['identity'] or bundle['status']!='COMPLETE' or bundle['checksum']!=digest(bundle['rows']): raise ValueError('Incomplete/corrupt checkpoint')
        for row in bundle['rows']:
            if row['row_id'] in seen: raise ValueError('Duplicate sample row')
            if row['independent_run_id']!=manifest['config']['independent_run_id'] or row['order_seed']!=manifest['config']['order_seed']: raise ValueError('Run identity mismatch')
            seen.add(row['row_id']); rows.append(row)
    if not rows: raise ValueError('No measured evidence')
    return manifest,rows


def identity(m):
    cfg={k:v for k,v in m['config'].items() if k not in ('order_seed','independent_run_id')}
    runtime=m['runtime']
    return digest(dict(config=cfg,code=m['code']['sha256'],model=runtime['model'],weights=runtime['weight_identity'],
        encoder=runtime['encoder'],gpu_uuid=runtime['gpu'].get('gpu_uuid'),torch=runtime['torch'],cuda=runtime['cuda'],
        fixtures={k:v['fixtures'] for k,v in m['cases'].items()}))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dirs',type=Path,nargs='+',required=True); p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--seed',type=int,default=9107); p.add_argument('--bootstrap-samples',type=int,default=5000)
    a=p.parse_args(argv)
    if len(set(path.resolve() for path in a.run_dirs))!=len(a.run_dirs): p.error('Duplicate run directories')
    loaded=[load_run(path) for path in a.run_dirs]
    if len({identity(m) for m,_ in loaded})!=1: raise ValueError('Code/model/adapter/workload/hardware/instrumentation mismatch')
    runids=[m['config']['independent_run_id'] for m,_ in loaded]; seeds=[m['config']['order_seed'] for m,_ in loaded]
    if len(set(runids))!=len(runids) or len(set(seeds))!=len(seeds): raise ValueError('Independent runs require distinct run IDs AND order seeds')
    groups={}; differences=[]
    for manifest,rows in loaded:
        for row in rows:
            if row['phase']!='measured': continue
            key=(row['length'],str(row['requested_coverage']),row['workload'],row['executor'],row['instrumentation'])
            groups.setdefault(key,{}).setdefault(row['independent_run_id'],[]).append(row)
            differences.append({k:row[k] for k in ('row_id','independent_run_id','order_seed','length','requested_coverage','workload','executor','repeat','full_ms','reuse_ms','paired_difference_ms','correctness_status','executor_parity')})
    hardware_verified=all(m['runtime']['gpu'].get('gpu_uuid') for m,_ in loaded)
    summaries=[]
    for key,runs in groups.items():
        if set(runs)!=set(runids): raise ValueError('Unmatched case coverage between independent runs')
        for run_id,rows in runs.items():
            expected=loaded[runids.index(run_id)][0]['config']['measured_runs']
            if len(rows)!=expected or {r['repeat'] for r in rows}!=set(range(expected)): raise ValueError('Missing/duplicate paired iterations')
        boot=hierarchical({r:[x['paired_difference_ms'] for x in rows] for r,rows in runs.items()},a.seed,a.bootstrap_samples)
        flat=[r for rows in runs.values() for r in rows]
        parity=all(r['executor_parity'] and r['qkv_validation']=='PASS' for r in flat)
        strict=all(r['correctness_status']=='STRICT_PASS' for r in flat)
        enough=len(runs)>=3 and all(len(v)>=30 for v in runs.values())
        measured=key[4]=='minimal'; observed=measured and parity and boot['mean']>0
        stable=observed and enough and hardware_verified and boot['ci95'][0]>0
        full=[r['full_ms'] for r in flat]; reuse=[r['reuse_ms'] for r in flat]
        summaries.append(dict(length=key[0],requested_coverage=key[1],workload=key[2],executor=key[3],instrumentation=key[4],
            full=dict(stats(full),min=min(full),max=max(full)),reuse=dict(stats(reuse),min=min(reuse),max=max(reuse)),hardware_identity_verified=bool(hardware_verified),speedup=sum(full)/sum(reuse),paired_win_count=sum(r['paired_difference_ms']>0 for r in flat),
            paired_win_rate=sum(r['paired_difference_ms']>0 for r in flat)/len(flat),bootstrap=boot,executor_parity=parity,
            correctness_status='STRICT_PASS' if strict else 'STRICT_FAIL',observed_break_even=observed,reproducible_break_even=stable,
            correctness_qualified_break_even=stable and strict and any(r['reused_tokens'] for r in flat),
            decision='REPRODUCIBLE_OBSERVED_BREAK_EVEN' if stable else 'OBSERVED_NOT_ESTABLISHED_ACROSS_RUNS' if observed else 'NO_BREAK_EVEN_OBSERVED'))
    a.output_dir.mkdir(parents=True,exist_ok=False)
    csv_write(a.output_dir/'independent_run_differences.csv',differences); csv_write(a.output_dir/'overall_summary.csv',summaries)
    atomic_json(a.output_dir/'break_even_decisions.json',dict(groups=summaries,baseline='FULL_RECOMPUTE',classification='ORACLE_DIAGNOSTIC',outliers_removed=False))
    atomic_json(a.output_dir/'aggregate_manifest.json',dict(input_runs=[str(x.resolve()) for x in a.run_dirs],run_ids=runids,order_seeds=seeds,
        identity=identity(loaded[0][0]),bootstrap_seed=a.seed,bootstrap_samples=a.bootstrap_samples))
    return 0
if __name__=='__main__': raise SystemExit(main())
