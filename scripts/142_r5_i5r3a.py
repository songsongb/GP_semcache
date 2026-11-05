#!/usr/bin/env python3
"""Offline I5R-3A prepare/analyze/verify; no inference or submission."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from semcache.edgelora import r5_i5r3a as api

def main():
 p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True)
 a=sub.add_parser('prepare');a.add_argument('--registry',required=True);a.add_argument('--multi-prepared',required=True);a.add_argument('--output-dir',required=True);a.add_argument('--block-size',type=int,default=20);a.add_argument('--input-limit',type=int);a.add_argument('--seed',type=int,default=42)
 for name in ('analyze','verify'):
  a=sub.add_parser(name);a.add_argument('--plan',required=True);a.add_argument('--output-dir',required=True)
  if name=='verify':a.add_argument('--audit',required=True)
 args=vars(p.parse_args());action=args.pop('action');args['output']=args.pop('output_dir')
 try:print(json.dumps(getattr(api,action)(**args),indent=2));return 0
 except (ValueError,KeyError,FileNotFoundError,ImportError) as exc:print(json.dumps(dict(status='REJECTED',reason=str(exc))),file=sys.stderr);return 2
if __name__=='__main__':sys.exit(main())
