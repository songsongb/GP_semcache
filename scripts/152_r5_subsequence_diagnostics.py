#!/usr/bin/env python3
"""CPU-only sealed historical forensics or preparation of an opt-in revalidation.

This program never runs inference, accesses CUDA or submits jobs.
"""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))

def main(argv=None):
    from semcache.edgelora import r5_subsequence_diagnostics as api
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('experiment','run','audit','output-dir'):p.add_argument('--'+name,required=True)
    p.add_argument('--expected-source-commit',default=api.REFERENCE_COMMIT)
    p.add_argument('--prepare-revalidation',action='store_true')
    p.add_argument('--source-commit');p.add_argument('--cuda-events',action='store_true')
    args=vars(p.parse_args(argv));prepare=args.pop('prepare_revalidation');source=args.pop('source_commit');cuda=args.pop('cuda_events')
    try:
        if prepare:
            if not source:p.error('--prepare-revalidation requires --source-commit')
            result=api.prepare_revalidation(**args,source_commit=source,cuda_events=cuda)
        else:
            if source or cuda:p.error('--source-commit/--cuda-events require --prepare-revalidation')
            result=api.diagnose(**args)
        print(json.dumps(result,indent=2));return 0
    except (ValueError,KeyError,OSError,ImportError) as exc:
        print(json.dumps(dict(status='REJECTED',reason=str(exc))),file=sys.stderr);return 2

if __name__=='__main__':sys.exit(main())
