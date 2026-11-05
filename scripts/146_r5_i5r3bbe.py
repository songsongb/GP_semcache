#!/usr/bin/env python3
"""BE CPU offline/prepare/inspect/analyze/verify and explicitly opted-in GPU run."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from semcache.edgelora import r5_i5r3bbe as api

def main():
 p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True)
 for name in ('offline','prepare'):
  a=sub.add_parser(name)
  for key in ('experiment','run','audit','verification','output-dir'):a.add_argument('--'+key,required=True)
  a.add_argument('--expected-commit',default=api.BASE_COMMIT)
  if name=='prepare':
   a.add_argument('--positive-requests',type=int,default=3);a.add_argument('--repetitions',type=int,default=1);a.add_argument('--validated-be-audit')
 a=sub.add_parser('inspect');a.add_argument('--experiment',required=True)
 a=sub.add_parser('run')
 for key in ('experiment','output-dir','source-commit'):a.add_argument('--'+key,required=True)
 a.add_argument('--block-index',type=int,default=0);a.add_argument('--enable-gpu-experiment',action='store_true',required=True)
 for name in ('analyze','verify'):
  a=sub.add_parser(name);a.add_argument('--experiment',required=True);a.add_argument('--runs',nargs='+',required=True);a.add_argument('--output-dir',required=True)
  if name=='verify':a.add_argument('--audit',required=True)
 args=vars(p.parse_args());action=args.pop('action')
 if action=='run':
  from semcache.edgelora.r5_i5r3bbe_runtime import run
  args.pop('enable_gpu_experiment');fn=run
 elif action in ('analyze','verify'):
  from semcache.edgelora import r5_i5r3bbe_analysis
  fn=getattr(r5_i5r3bbe_analysis,action)
 else:fn=getattr(api,action)
 try:print(json.dumps(fn(**args),indent=2));return 0
 except (ValueError,KeyError,FileNotFoundError,ImportError) as exc:print(json.dumps(dict(status='REJECTED',reason=str(exc))),file=sys.stderr);return 2
if __name__=='__main__':sys.exit(main())
