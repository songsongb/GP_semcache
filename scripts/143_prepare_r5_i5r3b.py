#!/usr/bin/env python3
"""CPU-only 3B preparation/inspection of sealed, unoptimized I5 blocks."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from semcache.edgelora import r5_i5r3b_plan as api

def main():
 p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='action',required=True);a=s.add_parser('prepare')
 for key in ('plan-3a','audit-3a','verification-3a','expected-3a-commit','output'):a.add_argument('--'+key,required=True)
 a.add_argument('--profile',choices=('B1','B2_SMOKE','B2_THREE_ADAPTER'),default='B2_SMOKE');a.add_argument('--adapter',choices=('Adapter_A','Adapter_B','Adapter_C'),default='Adapter_A');a.add_argument('--historical-r1-audit')
 a.add_argument('--max-prompt-tokens',type=int,help='B2 only; default 1280, context-only derived prompt budget')
 a=s.add_parser('inspect');a.add_argument('--experiment',required=True)
 args=vars(p.parse_args());action=args.pop('action')
 try:print(json.dumps(api.inspect(args['experiment']) if action=='inspect' else api.prepare(**args),indent=2));return 0
 except (ValueError,KeyError,FileNotFoundError,ImportError) as exc:print(json.dumps(dict(status='REJECTED',reason=str(exc))),file=sys.stderr);return 2
if __name__=='__main__':sys.exit(main())
