#!/usr/bin/env python3
"""D5 NATURAL: CPU preparation and ONE manually gated original I5 block."""
import argparse,json,sys,signal
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='action',required=True)
    a=sub.add_parser('prepare')
    for k in ('original-experiment','multi','source-commit','output-dir'):a.add_argument('--'+k,required=True)
    a.add_argument('--block-index',type=int,required=True)
    for action in ('cpu-validate','inspect','run','analyze','verify','_worker'):
        a=sub.add_parser(action);a.add_argument('--experiment',required=True)
        if action!='cpu-validate':a.add_argument('--cpu-audit',required=True)
        if action!='inspect':a.add_argument('--output-dir',required=True)
        if action=='run':
            a.add_argument('--source-commit',required=True);a.add_argument('--enable-gpu-experiment',action='store_true',required=True)
        if action in ('analyze','verify'):a.add_argument('--run',required=True)
        if action=='verify':a.add_argument('--audit',required=True)
        if action=='_worker':a.add_argument('--condition',required=True);a.add_argument('--expected-gpu-uuid')
    a=sub.add_parser('_ud');a.add_argument('--config',required=True);a.add_argument('--output-dir',required=True)
    args=vars(parser.parse_args(argv));action=args.pop('action')
    try:
        if action in ('run','_worker','_ud'):from semcache.edgelora import r5_subsequence_natural_runtime as api
        elif action in ('analyze','verify'):from semcache.edgelora import r5_subsequence_natural_audit as api
        else:from semcache.edgelora import r5_subsequence_natural as api
        result=getattr(api,action.lstrip('_').replace('-','_'))(**args);print(json.dumps(result,indent=2));return 0
    except (ValueError,KeyError,FileNotFoundError,ImportError) as exc:
        print(json.dumps(dict(status='REJECTED',reason=str(exc))),file=sys.stderr);return 2

def terminated(signum,frame):raise TimeoutError('NATURAL worker terminated by signal '+str(signum))
if __name__=='__main__':
    signal.signal(signal.SIGTERM,terminated);sys.exit(main())
