#!/usr/bin/env python3
"""CPU-only preparation from verified historical native Linear operands; no download."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from semcache.edgelora.r5_i5r2bs1_plan import prepare,validate_plan

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--historical-run-dir',type=Path,required=True);p.add_argument('--historical-review-dir',type=Path,required=True)
    p.add_argument('--expected-historical-source-commit',required=True);p.add_argument('--base-prepared',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--repetitions',type=int,choices=(6,8),default=6)
    p.add_argument('--suffix-seed',type=int,default=42);a=p.parse_args()
    result=prepare(a.historical_run_dir,a.historical_review_dir,a.base_prepared,a.output_dir,
        expected_commit=a.expected_historical_source_commit,repetitions=a.repetitions,seed=a.suffix_seed)
    validate_plan(a.output_dir)
    print(json.dumps(dict(status='PREPARED_GPU_NOT_RUN',source_commit=result['source']['commit'],repetitions=result['repetitions'],
        suffix={r:x['suffix_method'] for r,x in result['roles'].items()},bounds=result['bounds'],
        backend_policy=result['backend_policy'],recorded_backend=result['historical_backend']),indent=2))

if __name__=='__main__':main()
