#!/usr/bin/env python3
"""C4 prepare/run/analyze/final-assessment. No automatic job submission."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from semcache.edgelora.r5_i5r2c4 import prepare,analyze,final_assessment

def main():
 p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='command',required=True);a=s.add_parser('prepare')
 for stage in ('c2','b0','c3'):
  for role in ('experiment','run','audit'):a.add_argument('--'+stage+'-'+role,type=Path,required=True)
  a.add_argument('--expected-'+stage+'-commit',required=True)
 a.add_argument('--base-prepared',type=Path,required=True);a.add_argument('--output-dir',type=Path,required=True)
 a=s.add_parser('run');a.add_argument('--source-commit',required=True)
 for role in ('experiment','output-dir'):a.add_argument('--'+role,type=Path,required=True)
 for command in ('analyze','final-assessment'):
  a=s.add_parser(command)
  for role in ('experiment','run-dir','output-dir'):a.add_argument('--'+role,type=Path,required=True)
  if command=='final-assessment':a.add_argument('--audit-dir',type=Path,required=True)
 args=p.parse_args()
 try:
  if args.command=='prepare':
   inputs={stage+'_'+role:getattr(args,stage+'_'+role) for stage in ('c2','b0','c3') for role in ('experiment','run','audit')}
   result=prepare(inputs,{stage:getattr(args,'expected_'+stage+'_commit') for stage in ('c2','b0','c3')},args.base_prepared,args.output_dir)
  elif args.command=='run':
   from semcache.edgelora.r5_i5r2c4_runtime import execute
   result=execute(args.experiment,args.output_dir,args.source_commit)
  elif args.command=='analyze':result=analyze(args.experiment,args.run_dir,args.output_dir)
  else:result=final_assessment(args.experiment,args.run_dir,args.audit_dir,args.output_dir)
  print(json.dumps(result,indent=2));return 0
 except (ValueError,KeyError,FileNotFoundError) as exc:
  print(json.dumps(dict(status='REJECTED',reason=str(exc))),file=sys.stderr);return 2
if __name__=='__main__':sys.exit(main())
