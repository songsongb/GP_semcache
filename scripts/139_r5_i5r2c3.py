#!/usr/bin/env python3
"""CPU C2 coverage audit; conditional one-layer native attention replay."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from semcache.edgelora.r5_i5r2c3 import stage_a,prepare

def main():
 p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
 for command in ('stage-a','prepare'):
  a=sub.add_parser(command)
  for n in ('experiment','run-dir','audit-dir','output-dir'):a.add_argument('--'+n,type=Path,required=True)
  a.add_argument('--expected-historical-commit',required=True)
  if command=='stage-a':a.add_argument('--allow-incomplete',action='store_true')
  else:
   a.add_argument('--base-prepared',type=Path,required=True)
   for n in ('companion-experiment','companion-run','companion-audit'):a.add_argument('--'+n,type=Path)
 a=sub.add_parser('run')
 for n in ('experiment','output-dir'):a.add_argument('--'+n,type=Path,required=True)
 a.add_argument('--source-commit',required=True)
 a=sub.add_parser('analyze')
 for n in ('experiment','run-dir','output-dir'):a.add_argument('--'+n,type=Path,required=True)
 args=p.parse_args()
 try:
  if args.command=='stage-a':result=stage_a(args.experiment,args.run_dir,args.audit_dir,args.output_dir,args.expected_historical_commit,args.allow_incomplete)
  elif args.command=='prepare':
   values=(args.companion_experiment,args.companion_run,args.companion_audit)
   if any(values) and not all(values):raise ValueError('all three explicit B0 companion paths required')
   companion=dict(zip(('experiment','run','audit'),values)) if all(values) else None
   result=prepare(args.experiment,args.run_dir,args.audit_dir,args.base_prepared,args.output_dir,args.expected_historical_commit,companion=companion)
  else:
   from semcache.edgelora.r5_i5r2c3_runtime import execute,analyze
   result=execute(args.experiment,args.output_dir,args.source_commit) if args.command=='run' else analyze(args.experiment,args.run_dir,args.output_dir)
  print(json.dumps(result,indent=2));return 0
 except (ValueError,FileNotFoundError,KeyError) as exc:
  print(json.dumps(dict(status='REJECTED',reason=str(exc))),file=sys.stderr);return 2
if __name__=='__main__':sys.exit(main())
