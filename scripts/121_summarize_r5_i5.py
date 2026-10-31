#!/usr/bin/env python3
"""Audit and join every planned block; rejects missing blocks and duplicate retries."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from semcache.edgelora.r5_i5_summary import summarize

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--experiment',type=Path,required=True)
    p.add_argument('--runs',nargs='+',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();print(summarize(a.experiment,a.runs,a.output_dir))
