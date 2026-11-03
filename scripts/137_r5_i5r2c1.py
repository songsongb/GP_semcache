#!/usr/bin/env python3
"""Prepare, execute or independently audit a diagnostic-only fixed-shape control."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from semcache.edgelora.r5_i5r2c1 import prepare,execute,analyze
p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='mode',required=True)
a=s.add_parser('prepare')
for n in ('s1-experiment','s1-run','s1-audit','output-dir'):a.add_argument('--'+n,type=Path,required=True)
a=s.add_parser('run')
for n in ('experiment','output-dir'):a.add_argument('--'+n,type=Path,required=True)
a.add_argument('--source-commit',required=True);a.add_argument('--enable-gpu-diagnostic',action='store_true',required=True)
a=s.add_parser('analyze')
for n in ('experiment','run-dir','output-dir'):a.add_argument('--'+n,type=Path,required=True)
if __name__=='__main__':
 a=p.parse_args()
 if a.mode=='prepare':r=prepare(a.s1_experiment,a.s1_run,a.s1_audit,a.output_dir);r={'experiment_id':r['experiment_id'],'source':r['source']['commit'],'GPU_NOT_RUN':True}
 elif a.mode=='run':execute(a.experiment,a.output_dir,a.source_commit);r={'status':'COMPLETE'}
 else:r=analyze(a.experiment,a.run_dir,a.output_dir)
 print(json.dumps(r,indent=2))
