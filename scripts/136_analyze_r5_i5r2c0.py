#!/usr/bin/env python3
"""CPU-only cast analysis of explicit sealed historical S1 inputs."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from semcache.edgelora.r5_i5r2c0_analysis import analyze
p=argparse.ArgumentParser(description=__doc__)
for name in ('experiment','run-dir','audit-dir','output-dir'):p.add_argument('--'+name,type=Path,required=True)
if __name__=='__main__':
 a=p.parse_args();print(json.dumps(analyze(a.experiment,a.run_dir,a.audit_dir,a.output_dir),indent=2))
