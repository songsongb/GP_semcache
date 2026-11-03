#!/usr/bin/env python3
"""CPU-only independent read-only analysis of a completed isolated Linear run."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from semcache.edgelora.r5_i5r2bs1_analysis import analyze

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('experiment','run-dir','output-dir'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();result=analyze(a.experiment,a.run_dir,a.output_dir)
    print(json.dumps({k:result[k] for k in ('case','SAME_SHAPE_REPEATABILITY','SHAPE_DEPENDENCE_DEMONSTRATED','ROOT_CAUSE','NUMERICAL_FIDELITY_RESTORED')},indent=2))

if __name__=='__main__':main()
