#!/usr/bin/env python3
"""I5R-4 sealed CPU planning and manually opted-in sequential GPU campaign."""
import argparse,json,sys,signal
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))

def parser():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True)
    for name in ('prepare','budget'):
        a=sub.add_parser(name)
        for key in ('source-commit','plan-3a','audit-3a','verification-3a','expected-3a-commit','be-experiment','be-run','be-audit','be-verification','nf1-audit'):a.add_argument('--'+key,required=True)
        if name=='prepare':a.add_argument('--output-dir',required=True)
        else:a.add_argument('--disk-path',help='Read-only filesystem free-space probe; defaults to repository results')
        for key,value in (('trace-units',4 if name=='prepare' else 2),('rotations',2),('max-prompt-tokens',1280),('quality-tokens',32),('seed',42),('ttl-idle-requests',4),('exploration-every',3),('startup-seconds',180)):a.add_argument('--'+key,type=int,default=value)
        for key,value in (('quality-margin-f1',2.),('expected-reuses',2.),('retention-ms-per-mib-request',1.),('runtime-safety-factor',2.)):a.add_argument('--'+key,type=float,default=value)
    a=sub.add_parser('cpu-validate')
    for key in ('experiment','output-dir'):a.add_argument('--'+key,required=True)
    a=sub.add_parser('inspect')
    for key in ('experiment','cpu-audit','source-commit'):a.add_argument('--'+key,required=True)
    a.add_argument('--require-cpu-validation',action='store_true',required=True)
    a=sub.add_parser('run')
    for key in ('experiment','source-commit','output-dir','cpu-audit'):a.add_argument('--'+key,required=True)
    a.add_argument('--runtime-budget-seconds',type=int,default=68400);a.add_argument('--enable-gpu-experiment',action='store_true',required=True)
    for name in ('analyze','verify','report'):
        a=sub.add_parser(name)
        for key in ('experiment','run','output-dir'):a.add_argument('--'+key,required=True)
        if name!='report':a.add_argument('--cpu-audit',required=True)
        if name in ('verify','report'):a.add_argument('--audit',required=True)
        if name=='report':a.add_argument('--verification',required=True)
    for name in ('_worker','_gpu-preflight'):
        a=sub.add_parser(name,help=argparse.SUPPRESS);a.add_argument('--plan',required=True)
    return p

def main(argv=None):
    import os
    os.umask(0o077);args=vars(parser().parse_args(argv));action=args.pop('action')
    try:
        if action.startswith('_'):
            from semcache.edgelora import r5_i5r4_runtime as runtime
            if action=='_worker':runtime.worker(args['plan']);return 0
            from semcache.edgelora.r3a_artifacts import read
            d=read(args['plan']);result=dict(execution_gpu_uuid=runtime.preflight(d['plan'],Path(d['output']),d['remaining']))
        else:
            if action=='run':from semcache.edgelora import r5_i5r4_runtime as api
            elif action in ('analyze','verify','report'):from semcache.edgelora import r5_i5r4_analysis as api
            else:from semcache.edgelora import r5_i5r4 as api
            result=getattr(api,action.replace('-','_'))(**args)
        print(json.dumps(result,indent=2,allow_nan=False));return 0
    except (ValueError,KeyError,FileNotFoundError,ImportError) as exc:
        receipt=dict(status='REJECTED',action=action,reason=str(exc))
        if hasattr(exc,'report'):receipt['resource_diagnostics']=exc.report
        print(json.dumps(receipt,indent=2,allow_nan=False),file=sys.stderr);return 2

if __name__=='__main__':
    signal.signal(signal.SIGTERM,lambda s,f:sys.exit(128+s));sys.exit(main())
