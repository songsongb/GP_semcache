#!/usr/bin/env python3
"""Independent CPU natural audit and manually gated controlled I5 GPU pilots."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='action',required=True)
    for name in ('cpu','prepare'):
        a=s.add_parser(name)
        for k in ('registry','multi','output-dir'):a.add_argument('--'+k,required=True)
        a.add_argument('--input-limit',type=int,default=1280)
        if name=='cpu':a.add_argument('--block-size',type=int,default=5);a.add_argument('--seed',type=int,default=42);a.add_argument('--max-samples',type=int,default=200)
        else:
            a.add_argument('--dataset',choices=('qasper','2wikimqa','multi_news'),required=True);a.add_argument('--source-commit',required=True)
            a.add_argument('--contexts',type=int,choices=(1,2,3),default=1);a.add_argument('--reuses',type=int,choices=(1,2,3,4),default=3)
    a=s.add_parser('cpu-verify');a.add_argument('--audit',required=True);a.add_argument('--output-dir',required=True)
    a=s.add_parser('inspect');a.add_argument('--experiment',required=True)
    a=s.add_parser('run')
    for k in ('experiment','output-dir','source-commit'):a.add_argument('--'+k,required=True)
    a.add_argument('--block-index',type=int,default=0);a.add_argument('--enable-gpu-experiment',action='store_true',required=True)
    for name in ('analyze','verify'):
        a=s.add_parser(name)
        for k in ('experiment','output-dir'):a.add_argument('--'+k,required=True)
        a.add_argument('--runs',nargs='+',required=True)
        if name=='verify':a.add_argument('--audit',required=True)
    args=vars(p.parse_args(argv));action=args.pop('action')
    try:
        if action in ('cpu','cpu-verify'):
            from semcache.edgelora import r5_i5r_parallel as api
        else:from semcache.edgelora import r5_i5r_parallel_gpu as api
        if action=='inspect':
            plan,plans,inventory=api.load(**args);r=dict(plan=plan,inventory=inventory,blocks={k:len(v['blocks']) for k,v in plans.items()})
        else:r=getattr(api,action.replace('-','_'))(**args)
        print(json.dumps(r,indent=2));return 0
    except (ValueError,KeyError,FileNotFoundError,ImportError) as exc:
        print(json.dumps(dict(status='REJECTED',reason=str(exc))),file=sys.stderr);return 2

if __name__=='__main__':sys.exit(main())
