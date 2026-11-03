#!/usr/bin/env python3
"""Read-only independent audit of a completed, pinned arithmetic diagnostic."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from semcache.edgelora.r5_i5r2b_analysis import analyze

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--experiment',type=Path,required=True);p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
    r=analyze(a.experiment,a.run_dir,a.output_dir)
    print(json.dumps({k:r[k] for k in ('GPU_NOT_RUN','GPU_EVIDENCE_COMPLETE','cases','ROOT_CAUSE','ROOT_CAUSE_CONFIRMED')},indent=2))
    return 1 if r['GPU_EVIDENCE_COMPLETE']=='FAIL' else 0

if __name__=='__main__':sys.exit(main())
