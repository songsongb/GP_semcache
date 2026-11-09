#!/usr/bin/env python3
"""Explicit one-dataset Phase3A D5 A/EXACT/APPROX validation; no automatic sweep."""
import argparse,json,signal,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from semcache.edgelora import r5_natural_phase3a as api


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='action',required=True)
    a=s.add_parser('discover')
    a.add_argument('--search-root',dest='search_roots',action='append',default=[])
    a.add_argument('--registry',dest='registries',action='append',default=[])
    a.add_argument('--original-experiment',dest='original_experiments',action='append',default=[])
    a.add_argument('--prepared-dataset',dest='prepared_datasets',action='append',default=[])
    a.add_argument('--multi')
    a=s.add_parser('prepare')
    for name in ('original-experiment','multi','source-commit','output-dir'):a.add_argument('--'+name,required=True)
    a.add_argument('--dataset',choices=api.DATASETS,required=True);a.add_argument('--block-index',type=int,required=True)
    a.add_argument('--enable-approximate-experiment',action='store_true',required=True)
    for name in ('cpu-validate','inspect','run','analyze','verify','_worker'):
        a=s.add_parser(name);a.add_argument('--experiment',required=True)
        if name!='cpu-validate':a.add_argument('--cpu-audit',required=True)
        if name!='inspect':a.add_argument('--output-dir',required=True)
        if name=='run':
            a.add_argument('--source-commit',required=True)
            a.add_argument('--enable-gpu-experiment',action='store_true',required=True)
            a.add_argument('--enable-approximate-experiment',action='store_true',required=True)
        if name in ('analyze','verify'):a.add_argument('--run',required=True)
        if name=='verify':a.add_argument('--audit',required=True)
        if name=='_worker':a.add_argument('--condition',choices=api.base.CONDITIONS,required=True);a.add_argument('--expected-gpu-uuid')
    a=s.add_parser('_ud');a.add_argument('--config',required=True);a.add_argument('--output-dir',required=True)
    a=s.add_parser('aggregate');a.add_argument('--results-registry',required=True);a.add_argument('--output-dir',required=True)
    args=vars(p.parse_args(argv));action=args.pop('action').lstrip('_').replace('-','_')
    try:print(json.dumps(getattr(api,action)(**args),indent=2));return 0
    except (ValueError,KeyError,OSError,ImportError,TypeError) as exc:
        print(json.dumps(dict(status='BLOCKED',reason=str(exc),GPU_status='GPU_NOT_RUN')),file=sys.stderr);return 2


def terminated(signum,frame):raise TimeoutError('bounded Phase3A worker terminated by signal '+str(signum))
if __name__=='__main__':
    signal.signal(signal.SIGTERM,terminated);sys.exit(main())
