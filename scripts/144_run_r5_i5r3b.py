#!/usr/bin/env python3
"""One manual paired block plus separate target-only control; existing I5 workers."""
import argparse,json,os,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from semcache.edgelora.r5_i5r3b_plan import validate_plan,ROOT,new_output
from semcache.edgelora.r5_i5_evidence import fresh,seal
from semcache.edgelora.r3a_artifacts import write
from semcache.edgelora.communication import require

def command(experiment,multi,index,commit,seconds,output):
 return [sys.executable,str(ROOT/'scripts/119_run_r5_i5.py'),'--experiment',str(experiment),'--multi-prepared',str(multi),'--block-index',str(index),'--source-commit',commit,'--runtime-budget-seconds',str(seconds),'--output-dir',str(output),'--enable-gpu-experiment']

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for key in ('experiment','output-dir','source-commit'):p.add_argument('--'+key,required=True)
 p.add_argument('--block-index',type=int,default=0);p.add_argument('--runtime-budget-seconds',type=int,default=4800)
 p.add_argument('--enable-gpu-experiment',action='store_true',required=True);p.add_argument('--include-a0-target-only',action='store_true',required=True)
 a=p.parse_args();protocol,plans=validate_plan(a.experiment)
 require(a.source_commit==protocol['source']['commit'] and 0<=a.block_index<len(plans['abc']['blocks']),'pinned source/block mismatch')
 require(a.enable_gpu_experiment and os.environ.get('R5_I5_ENABLE_GPU')==os.environ.get('R5_ENABLE_GPU')=='1' and os.environ.get('SLURM_JOB_ID'),'manual opted-in allocation required')
 require(60<=a.runtime_budget_seconds<=4800 and int(os.environ.get('SLURM_CPUS_PER_TASK','0'))>=4 and int(os.environ.get('SLURM_MEM_PER_NODE','0'))>=65536,'bounded 4 CPU / 64 GiB allocation required')
 import shutil
 reserve=sum(3*plans[k]['blocks'][a.block_index]['resources']['evidence_budget_bytes'] for k in ('abc','a0'))
 require(shutil.disk_usage(Path(a.experiment).parent).free>reserve+2**30,'combined ABC+A0 worst-case disk reserve unavailable')
 out=new_output(a.output_dir,(a.experiment,protocol['multi_prepared']));status=dict(status='FAILED',source_commit=a.source_commit,block_index=a.block_index,with_a0=True)
 primary=None
 try:
  for name in ('abc','a0'):
   result=subprocess.run(command(Path(a.experiment)/name,protocol['multi_prepared'],a.block_index,a.source_commit,a.runtime_budget_seconds,out/(name+'_run')),cwd=ROOT)
   require(result.returncode==0,'unchanged I5 '+name+' failed; preserve attempt; no automatic retry')
  status['status']='COMPLETE'
 except BaseException as exc:primary=exc;status['reason']=type(exc).__name__+': '+str(exc)[:512];raise
 finally:
  try:write(out/'i5r3b_attempt_status.json',status);seal(out)
  except BaseException as secondary:
   if primary is None:raise
   if hasattr(primary,'add_note'):primary.add_note('3B finalization failed: '+str(secondary))
 print(json.dumps(status));return 0
if __name__=='__main__':sys.exit(main())
