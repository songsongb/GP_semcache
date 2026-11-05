#!/usr/bin/env python3
"""CPU strict analysis/independent recomputation; functional failures preserved."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from semcache.edgelora import r5_i5r3b_analysis as api

def main():
 p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='action',required=True)
 for action in ('analyze','verify'):
  a=s.add_parser(action)
  for key in ('experiment','run','output'):a.add_argument('--'+key,required=True)
  if action=='verify':a.add_argument('--audit',required=True)
 args=vars(p.parse_args());action=args.pop('action')
 try:print(json.dumps(getattr(api,action)(**args),indent=2));return 0
 except (ValueError,KeyError,FileNotFoundError) as exc:print(json.dumps(dict(status='REJECTED',reason=str(exc))),file=sys.stderr);return 2
if __name__=='__main__':sys.exit(main())
