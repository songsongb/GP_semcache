#!/usr/bin/env python3
"""Independent read-only I5 block audit into a new sealed directory."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from semcache.edgelora.r5_i5_analysis import analyze,export
from semcache.edgelora.r5_i5_evidence import fresh,seal
from semcache.edgelora.r3a_artifacts import write


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--inputs',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
    out=fresh(a.output_dir,(a.inputs,))
    try:export(out,analyze(a.inputs))
    except BaseException as exc:
        write(out/'i5_independent_audit.json',dict(status='FAIL',reason=str(exc)));seal(out);raise
    seal(out)


if __name__=='__main__':main()
