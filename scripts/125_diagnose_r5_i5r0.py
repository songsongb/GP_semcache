#!/usr/bin/env python3
"""CPU-only historical I5 diagnostics; reads sealed run trees or audit ZIPs."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from semcache.edgelora.r5_i5r0 import run


def main():
    p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--runs-root',type=Path);g.add_argument('--audit-zip',type=Path)
    p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--allow-partial',action='store_true',help='Explicitly permit missing optional evidence/unknown authorization, positions or timing fields')
    p.add_argument('--expected-blocks',type=int,default=12,help='Required completed block count; defaults to this historical twelve-block study')
    a=p.parse_args()
    try:
        r=run(runs_root=a.runs_root,audit_zip=a.audit_zip,output_dir=a.output_dir,allow_partial=a.allow_partial,expected_blocks=a.expected_blocks)
        print(json.dumps(dict(status='PARTIAL_DIAGNOSTIC' if r['coverage']['partial_observability'] else 'DIAGNOSTIC_COMPLETE',
            output=str(a.output_dir),historical_observations=r['workload']['historical_handoff_comparison']['recalculated'],GPU_EXECUTION_REQUIRED=False),indent=2));return 0
    except (ValueError,OSError,KeyError,TypeError) as exc:
        print(json.dumps(dict(status='DIAGNOSTIC_REJECTED',reason=str(exc))),file=sys.stderr);return 1


if __name__=='__main__':sys.exit(main())
