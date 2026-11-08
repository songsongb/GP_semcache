#!/usr/bin/env python3
"""Bounded real Qasper subsequence generation pilot. Never submits jobs."""
import argparse,json,sys,signal
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='action',required=True)
    a=s.add_parser('prepare')
    for k in ('registry','multi','source-commit','output-dir'):a.add_argument('--'+k,required=True)
    a.add_argument('--approximate',action='store_true')
    a=s.add_parser('run')
    for k in ('experiment','source-commit','cpu-audit','output-dir'):a.add_argument('--'+k,required=True)
    a.add_argument('--enable-gpu-experiment',action='store_true',required=True)
    for name in ('analyze','verify'):
        a=s.add_parser(name)
        for k in ('experiment','run','output-dir'):a.add_argument('--'+k,required=True)
        if name=='verify':a.add_argument('--audit',required=True)
    a=s.add_parser('cpu-validate');a.add_argument('--experiment',required=True);a.add_argument('--output-dir',required=True)
    a=s.add_parser('_worker')
    for k in ('experiment','output-dir','condition','cpu-audit'):a.add_argument('--'+k,required=True)
    a.add_argument('--expected-gpu-uuid')
    a=s.add_parser('_ud');a.add_argument('--config',required=True);a.add_argument('--output-dir',required=True)
    args=vars(p.parse_args(argv));action=args.pop('action')
    try:
        if action in ('analyze','verify'):from semcache.edgelora import r5_subsequence_audit as api
        else:from semcache.edgelora import r5_subsequence_pilot as api
        result=getattr(api,action.lstrip('_').replace('-','_'))(**args)
        print(json.dumps(result,indent=2));return 0
    except (ValueError,KeyError,FileNotFoundError,ImportError) as exc:
        print(json.dumps(dict(status='REJECTED',reason=str(exc))),file=sys.stderr);return 2

def terminated(signum,frame):
    # Raising lets worker/run finally blocks preserve FAILED evidence and reap UD.
    raise TimeoutError('sparse diagnostic terminated by signal '+str(signum))

if __name__=='__main__':
    signal.signal(signal.SIGTERM,terminated)
    sys.exit(main())
