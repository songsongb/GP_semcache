#!/usr/bin/env python3
"""Opt-in integrated padding plans, bounded execution, independent analysis."""
import argparse,json,sys,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from semcache.edgelora.r5_i5r2c2 import prepare,validate,analyze,MODES
from semcache.edgelora.r5_i5_evidence import fresh,seal
p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
a=sub.add_parser('prepare')
for n in ('historical-plan','multi-prepared','r2a-review','c1-plan','c1-run','c1-audit','output-dir'):a.add_argument('--'+n,type=Path,required=True)
a=sub.add_parser('run')
for n in ('experiment','multi-prepared','output-dir'):a.add_argument('--'+n,type=Path,required=True)
a.add_argument('--source-commit',required=True);g=a.add_mutually_exclusive_group(required=True);g.add_argument('--enable-gpu-experiment',action='store_true');g.add_argument('--test-fixture-only',action='store_true')
a=sub.add_parser('analyze')
for n in ('experiment','run-dir','output-dir'):a.add_argument('--'+n,type=Path,required=True)
if __name__=='__main__':
 a=p.parse_args()
 if a.command=='prepare':prepare(a.historical_plan,a.multi_prepared,a.r2a_review,a.c1_plan,a.c1_run,a.c1_audit,a.output_dir);print('PREPARED_GPU_NOT_RUN')
 elif a.command=='run':
  out=fresh(a.output_dir,(a.experiment,a.multi_prepared))
  gpu_uuid=None
  for mode in MODES:
   plan=validate(a.experiment/mode)
   if plan['i5r2b']['source']['commit']!=a.source_commit:raise ValueError('pinned source differs')
   cmd=[sys.executable,str(ROOT/'scripts/119_run_r5_i5.py'),'--experiment',str(a.experiment/mode),'--multi-prepared',str(a.multi_prepared),'--output-dir',str(out/mode),'--source-commit',a.source_commit,'--runtime-budget-seconds','1000','--enable-arithmetic-diagnostic','--test-fixture-only' if a.test_fixture_only else '--enable-gpu-experiment']
   subprocess.run(cmd,check=True)
   from semcache.edgelora.r3a_artifacts import read
   actual=read(out/mode/'SEMCACHE_INTEGRATED/gpu_identity.json')['execution_gpu_uuid']
   if mode!='OFF' and actual!=gpu_uuid:raise ValueError('GPU changed across mode comparisons')
   gpu_uuid=actual
   analyze(a.experiment/mode,out/mode,out/(mode+'_c2_audit'))
  seal(out)
 else:
  out=fresh(a.output_dir,(a.experiment,a.run_dir))
  from semcache.edgelora.r3a_artifacts import read
  from semcache.edgelora.r5_i5_evidence import jsonl
  identities=[read(a.run_dir/m/'SEMCACHE_INTEGRATED/gpu_identity.json')['execution_gpu_uuid'] for m in MODES]
  if len(set(identities))!=1:raise ValueError('GPU changed across modes')
  for condition in ('FULL_RECOMPUTE','SEMANTIC_NO_REUSE'):
   signatures=[[(r['matched_identity_sha256'],r['logits_sha256'],r['generated_token_ids']) for r in jsonl(a.run_dir/m/condition/'generation.jsonl')] for m in MODES]
   if signatures[0]!=signatures[1] or signatures[1]!=signatures[2]:raise ValueError('A/B baseline changed across modes')
  result={m:analyze(a.experiment/m,a.run_dir/m,out/m) for m in MODES};seal(out);print(json.dumps({m:{k:r[k] for k in ('PHYSICAL_REUSE_PRESERVED','STRICT_FIDELITY_RESTORED','PERFORMANCE_BENEFIT')} for m,r in result.items()},indent=2))
