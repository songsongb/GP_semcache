#!/usr/bin/env python3
"""Read-only historical I5R-1 tensor diagnosis. No model loading or GPU execution."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from semcache.edgelora.r5_i5r2_analysis import run


def main():
    p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--run-dir',type=Path);g.add_argument('--runs-root',type=Path)
    p.add_argument('--experiment',type=Path,help='Relocated immutable original plan; defaults to the recorded path')
    p.add_argument('--audit-dir',type=Path,help='Historical exported I5 audit; defaults to block directory plus _audit')
    p.add_argument('--r1-audit-dir',type=Path,help='Optional historical I5R-1 export; auto-detects block directory plus _i5r1_audit')
    p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--expected-source-commit')
    p.add_argument('--layers',nargs='+',type=int,help='Explicit layer subset; default all recorded layers')
    p.add_argument('--max-tensor-mib',type=int,default=32);p.add_argument('--max-input-gib',type=int,default=32)
    p.add_argument('--allow-unavailable-historical-git',action='store_true',help='Explicit partial provenance only; never waives available-blob mismatches')
    p.add_argument('--test-fixture-only',action='store_true')
    a=p.parse_args()
    try:
        result=run(output_dir=a.output_dir,run_dir=a.run_dir,runs_root=a.runs_root,experiment=a.experiment,audit_dir=a.audit_dir,r1_audit_dir=a.r1_audit_dir,
            expected_commit=a.expected_source_commit,fixture=a.test_fixture_only,layers=a.layers,allow_unavailable=a.allow_unavailable_historical_git,
            max_tensor_bytes=a.max_tensor_mib*2**20,max_input_bytes=a.max_input_gib*2**30)
        print(json.dumps(result,indent=2));return 0
    except (ValueError,OSError,KeyError,TypeError) as exc:
        print(json.dumps(dict(status='DIAGNOSIS_REJECTED',reason=str(exc))),file=sys.stderr);return 1


if __name__=='__main__':sys.exit(main())
