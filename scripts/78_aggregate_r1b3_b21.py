#!/usr/bin/env python3
"""Aggregate independent R1-B3-B2.1 output directories with run-level bootstrap."""
import argparse, importlib.util, json, random, statistics, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from semcache.metrics.r1 import atomic_json,digest
from semcache.metrics.r1b2 import csv_write

def b21():
    spec=importlib.util.spec_from_file_location('b21_aggregate',ROOT/'scripts/77_benchmark_r1b3_b21_setup.py')
    m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

def args(argv=None):
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--run-dirs',required=True,help='Comma-separated independent run output dirs')
    p.add_argument('--output-dir',type=Path,required=True); p.add_argument('--seed',type=int,default=9107); p.add_argument('--bootstrap-samples',type=int,default=10000)
    a=p.parse_args(argv); a.paths=[Path(x) for x in a.run_dirs.split(',')]
    if len(a.paths)<2 or len({str(x.resolve()) for x in a.paths})!=len(a.paths): p.error('Provide at least two distinct independent run directories')
    if a.bootstrap_samples<100: p.error('Use at least 100 bootstrap samples')
    return a

def checkpoints(directory):
    manifest=json.loads((directory/'manifest.json').read_text())
    rows=[]; seen=set()
    for path in sorted((directory/'checkpoints').glob('*.json')):
        item=json.loads(path.read_text())
        if item.get('identity')!=manifest.get('identity') or item.get('status')!='COMPLETE' or digest(item.get('rows',[]))!=item.get('checksum'):
            raise ValueError(f'Incomplete/invalid checkpoint {path}')
        for row in item['rows']:
            if row['row_id'] in seen: raise ValueError(f'Duplicate row {row["row_id"]}')
            seen.add(row['row_id']); rows.append(row)
    return manifest,rows

def main(argv=None):
    a=args(argv); runs=[]
    for directory in a.paths:
        manifest,rows=checkpoints(directory)
        if manifest.get('status') not in ('COMPLETE','PARITY_FAILED'):
            raise ValueError('Run is not complete: '+str(directory)+': '+str(manifest.get('status')))
        if manifest.get('cache_classification')!='ORACLE_DIAGNOSTIC': raise ValueError('Unexpected cache classification')
        runs.append((manifest,rows))
    identity={(m['runtime']['gpu_uuid'],m['runtime']['model'].get('resolved_model_revision'),m['runtime'].get('namespace'),
               m['runtime'].get('weight_identity'),m['config'].get('model'),m['config'].get('dtype'),m['config'].get('seed'),
               tuple(m['config'].get('lengths',[])),m['config'].get('cache_device'),m['config'].get('instrumentation'),
               m['config'].get('tokenizer_revision'),m['config'].get('encoder_revision')) for m,_ in runs}
    if len(identity)!=1: raise ValueError('Runs differ in GPU/model/dtype/length/storage/instrumentation provenance')
    groups={}
    for run_index,(manifest,rows) in enumerate(runs):
        by={}
        for row in rows:
            if row['phase']!='measured': continue
            by.setdefault((row['length'],row['storage'],str(row['coverage']),row['instrumentation'],row['repeat']),{})[row['mode']]=row
        run_id=manifest['config']['independent_run_id']
        for key,values in by.items():
            length,storage,cov,instr,repeat=key
            for mode,row in values.items():
                if mode in ('legacy','full'): continue
                base=values.get('legacy')
                if base is None: continue
                group=(length,storage,cov,instr,mode)
                groups.setdefault(group,{}).setdefault(run_id,[]).append(dict(repeat=repeat,
                    difference_ms=base['target_warm_ms']-row['target_warm_ms'],
                    legacy_ms=base['target_warm_ms'],optimized_ms=row['target_warm_ms'],
                    parity=bool(row.get('executor_parity',{}).get('passed'))))
    summary=[]; differences=[]; rng=random.Random(a.seed)
    for group,by_run in groups.items():
        run_ids=sorted(by_run); run_means={r:statistics.mean(x['difference_ms'] for x in by_run[r]) for r in run_ids}
        boot=[]
        for _ in range(a.bootstrap_samples):
            sampled_runs=[run_ids[rng.randrange(len(run_ids))] for _ in run_ids]; vals=[]
            for run_id in sampled_runs:
                samples=by_run[run_id]
                vals.extend(samples[rng.randrange(len(samples))]['difference_ms'] for _ in samples)
            boot.append(statistics.mean(vals))
        boot.sort()
        ci=[boot[int(.025*(len(boot)-1))],boot[int(.975*(len(boot)-1))]]
        flat=[x for run in by_run.values() for x in run]; mean_diff=statistics.mean(x['difference_ms'] for x in flat)
        wins=sum(x['difference_ms']>0 for x in flat); parity=all(x['parity'] for x in flat)
        item=dict(length=group[0],storage=group[1],coverage=group[2],instrumentation=group[3],optimized_mode=group[4],
            independent_run_count=len(run_ids),paired_iteration_count=len(flat),run_mean_differences_ms=run_means,
            mean_difference_ms=mean_diff,median_difference_ms=statistics.median(x['difference_ms'] for x in flat),
            between_run_sd_ms=statistics.stdev(run_means.values()) if len(run_ids)>1 else None,
            bootstrap_hierarchical_ci95_ms=ci,bootstrap_seed=a.seed,bootstrap_samples=a.bootstrap_samples,
            optimized_win_count=wins,optimized_win_rate=wins/len(flat),parity_pass=parity,
            reproducible_warm_improvement=bool(len(run_ids)>=3 and parity and ci[0]>0),
            statistical_status='INDEPENDENT_RUN_HIERARCHICAL_BOOTSTRAP' if len(run_ids)>=3 else 'INSUFFICIENT_INDEPENDENT_RUNS')
        summary.append(item)
        for run_id,samples in by_run.items():
            for row in samples: differences.append(dict(group=str(group),independent_run_id=run_id,**row))
    if not groups: raise ValueError('No complete paired Legacy comparisons found')
    a.output_dir.mkdir(parents=True,exist_ok=False)
    csv_write(a.output_dir/'independent_run_differences.csv',differences)
    csv_write(a.output_dir/'overall_summary.csv',summary)
    atomic_json(a.output_dir/'aggregate_manifest.json',dict(experiment='R1-B3-B2.1 aggregate',run_dirs=[str(x.resolve()) for x in a.paths],
        independent_run_count=len(runs),seed=a.seed,bootstrap_samples=a.bootstrap_samples,
        provenance_identity=list(next(iter(identity))),classification='ORACLE_DIAGNOSTIC',no_safety_claim=True))
    atomic_json(a.output_dir/'summary.json',dict(groups=summary,independent_run_count=len(runs)))
    return 0
if __name__=='__main__': raise SystemExit(main())
