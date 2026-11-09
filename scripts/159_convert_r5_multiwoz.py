#!/usr/bin/env python3
"""Offline M-stage MultiWOZ2.2 JSONL conversion. Never downloads data or schemas."""
import argparse,json,sys,subprocess
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from semcache.edgelora import r5_multiwoz_native as api


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True)
    c=sub.add_parser('convert')
    for name in ('source','source-sha256','output-dir','ontology','schema-repository','schema-revision','schema-sha256'):
        c.add_argument('--'+name,required=True)
    c=sub.add_parser('verify');c.add_argument('--conversion-dir',required=True)
    args=vars(p.parse_args(argv));action=args.pop('action')
    try:print(json.dumps(getattr(api,action)(**args),indent=2));return 0
    except (ValueError,KeyError,OSError,TypeError,subprocess.TimeoutExpired) as exc:
        print(json.dumps(dict(status='BLOCKED',reason=str(exc),GPU_status='GPU_NOT_RUN')),file=sys.stderr);return 2


if __name__=='__main__':sys.exit(main())
