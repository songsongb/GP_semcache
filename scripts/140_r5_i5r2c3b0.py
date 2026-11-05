#!/usr/bin/env python3
"""Prepare, capture and independently audit a bounded C3 B0 companion."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from semcache.edgelora.r5_i5r2c3b0 import prepare,analyze

def main():
 p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
 a=sub.add_parser('prepare')
 for n in ('c2-experiment','c2-run','c2-audit','multi-prepared','output-dir'):a.add_argument('--'+n,type=Path,required=True)
 a.add_argument('--expected-historical-commit',required=True)
 a=sub.add_parser('run')
 for n in ('experiment','output-dir'):a.add_argument('--'+n,type=Path,required=True)
 a.add_argument('--source-commit',required=True)
 a=sub.add_parser('analyze')
 for n in ('experiment','run-dir','output-dir'):a.add_argument('--'+n,type=Path,required=True)
 args=p.parse_args()
 try:
  if args.command=='prepare':result=prepare(args.c2_experiment,args.c2_run,args.c2_audit,args.multi_prepared,args.output_dir,args.expected_historical_commit)
  elif args.command=='run':
   from semcache.edgelora.r5_i5r2c3b0_runtime import execute
   result=execute(args.experiment,args.output_dir,args.source_commit)
  else:result=analyze(args.experiment,args.run_dir,args.output_dir)
  print(json.dumps(result,indent=2));return 0
 except (ValueError,FileNotFoundError,KeyError) as exc:
  print(json.dumps(dict(status='REJECTED',reason=str(exc))),file=sys.stderr);return 2
if __name__=='__main__':sys.exit(main())
