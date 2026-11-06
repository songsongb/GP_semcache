#!/usr/bin/env python3
"""CPU-only immutable NF1 forensics; no inference or job submission."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from semcache.edgelora.r5_i5r_nf1 import analyze,COMMIT

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for key in ('experiment','run','audit','verification','output-dir'):p.add_argument('--'+key,required=True)
 p.add_argument('--expected-commit',default=COMMIT);p.add_argument('--deadline-seconds',type=int,default=7200)
 args=vars(p.parse_args())
 try:print(json.dumps(analyze(**args),indent=2));return 0
 except (ValueError,KeyError,FileNotFoundError) as exc:print(json.dumps(dict(status='REJECTED',reason=str(exc))),file=sys.stderr);return 2
if __name__=='__main__':sys.exit(main())
