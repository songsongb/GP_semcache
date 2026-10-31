#!/usr/bin/env python3
"""CPU-only real dataset readiness and read-only sealed experiment inspection."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from semcache.edgelora.r5_i5_preflight import readiness,inspect_plan
from semcache.edgelora.r5_i5_evidence import fresh,seal
from semcache.edgelora.r3a_artifacts import read,write
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('phase',choices=['readiness','inspect']);p.add_argument('--registry',type=Path);p.add_argument('--experiment',type=Path)
p.add_argument('--base-prepared',type=Path);p.add_argument('--multi-prepared',type=Path);p.add_argument('--count',type=int,default=10)
p.add_argument('--budgets',type=Path);p.add_argument('--input-limit',type=int,default=512);p.add_argument('--allow-context-truncation',action='store_true');p.add_argument('--output-dir',type=Path)
a=p.parse_args()
if a.phase=='inspect':
    if not a.experiment:p.error('--experiment required')
    report=inspect_plan(a.experiment)
else:
    report=readiness(read(a.registry) if a.registry else {},base_prepared=a.base_prepared,multi_prepared=a.multi_prepared,count=a.count,
        budgets=read(a.budgets) if a.budgets else None,input_limit=a.input_limit,allow_truncation=a.allow_context_truncation)
if a.output_dir:
    out=fresh(a.output_dir,(a.registry,a.experiment,a.base_prepared,a.multi_prepared));write(out/'i5_preflight.json',report);seal(out)
print(json.dumps(report,indent=2));sys.exit(1 if report.get('status')=='NOT_READY' else 0)
