#!/usr/bin/env python3
"""One explicitly opted-in isolated native GPU Linear diagnostic, never a submission."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from semcache.edgelora.r5_i5r2bs1_runtime import execute

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('experiment','historical-run-dir','historical-review-dir','base-prepared','output-dir'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--source-commit',required=True);p.add_argument('--enable-gpu-diagnostic',action='store_true',required=True);a=p.parse_args()
    result=execute(a.experiment,a.historical_run_dir,a.historical_review_dir,a.base_prepared,a.output_dir,a.source_commit)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
