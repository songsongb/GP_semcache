#!/usr/bin/env python3
"""Fresh read-only I5 tensor/TCP audit and R1 warm/amortized comparison export."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from semcache.edgelora.r5_i5r1_analysis import summarize


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--experiment',type=Path,required=True)
    p.add_argument('--runs',type=Path,nargs='+',required=True);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--block-only',action='store_true',help='Explicit one-block review of a larger plan; never a complete-study claim')
    a=p.parse_args();result=summarize(a.experiment,a.runs,a.output_dir,block_only=a.block_only)
    print(json.dumps(result,indent=2));return 0 if result['functional_assertions']=='PASS' else 2


if __name__=='__main__':sys.exit(main())
