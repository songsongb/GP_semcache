#!/usr/bin/env python3
"""Explicit existing D5 APPROX pilot: CPU prepare/approve, manually gated A/B/C run."""
import argparse,json,sys,signal
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True)
    a=sub.add_parser('prepare')
    for name in ('original-experiment','multi','source-commit','output-dir'):a.add_argument('--'+name,required=True)
    a.add_argument('--track',choices=('CONTROLLED','NATURAL'),required=True)
    a.add_argument('--enable-approximate-experiment',action='store_true',required=True)
    for name in ('cpu-validate','inspect','run','analyze','verify','_worker'):
        a=sub.add_parser(name);a.add_argument('--experiment',required=True)
        if name!='cpu-validate':a.add_argument('--cpu-audit',required=True)
        if name!='inspect':a.add_argument('--output-dir',required=True)
        if name=='run':
            a.add_argument('--source-commit',required=True)
            a.add_argument('--enable-gpu-experiment',action='store_true',required=True)
            a.add_argument('--enable-approximate-experiment',action='store_true',required=True)
        if name in ('analyze','verify'):a.add_argument('--run',required=True)
        if name=='verify':a.add_argument('--audit',required=True)
        if name=='_worker':a.add_argument('--condition',required=True);a.add_argument('--expected-gpu-uuid')
    a=sub.add_parser('_ud');a.add_argument('--config',required=True);a.add_argument('--output-dir',required=True)
    args=vars(p.parse_args(argv));action=args.pop('action').lstrip('_').replace('-','_')
    try:
        if action in ('analyze','verify'):from semcache.edgelora import r5_natural_approx_audit as api
        else:from semcache.edgelora import r5_natural_approx as api
        print(json.dumps(getattr(api,action)(**args),indent=2));return 0
    except (ValueError,KeyError,FileNotFoundError,ImportError) as exc:
        print(json.dumps(dict(status='REJECTED',reason=str(exc))),file=sys.stderr);return 2


def terminated(signum,frame):raise TimeoutError('bounded APPROX worker terminated by signal '+str(signum))
if __name__=='__main__':
    signal.signal(signal.SIGTERM,terminated);sys.exit(main())
