"""Manual bounded execution; delegate actual ES/UD inference unchanged to I5."""
import importlib.util,os,subprocess,sys,shutil
from pathlib import Path
from .communication import require
from .r3a_artifacts import ROOT,write
from .r5_i5_evidence import seal
from .r5_i5r3bbe import load,output

def run(experiment,output_dir,source_commit,block_index=0):
 p,plans,base=load(experiment)
 require(source_commit==p['source']['commit'] and 0<=block_index<p['repetitions'],'BE source/block differs')
 require(os.environ.get('R5_I5_ENABLE_GPU')==os.environ.get('R5_ENABLE_GPU')=='1' and os.environ.get('SLURM_JOB_ID'),'explicit manual GPU allocation/opt-in required')
 require(int(os.environ.get('SLURM_CPUS_PER_TASK','0'))>=4 and int(os.environ.get('SLURM_MEM_PER_NODE','0'))>=65536,'four CPUs /64 GiB required')
 reserve=sum(3*plans[k]['blocks'][block_index]['resources']['evidence_budget_bytes'] for k in plans)
 require(shutil.disk_usage(Path(experiment).parent).free>reserve+2**30,'combined worst-case evidence disk unavailable')
 spec=importlib.util.spec_from_file_location('be_legacy_command',ROOT/'scripts/144_run_r5_i5r3b.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
 out=output(output_dir,(experiment,p['multi_prepared'],*base['paths'].values()));status=dict(status='FAILED',source_commit=source_commit,block_index=block_index,execution_order=p['execution_order'][block_index]);primary=None
 try:
  for control in status['execution_order']:
   result=subprocess.run(old.command(Path(experiment)/control,p['multi_prepared'],block_index,source_commit,4800,out/(control+'_run')),cwd=ROOT)
   require(result.returncode==0,'I5 '+control+' failed; no automatic retry')
  status['status']='COMPLETE'
 except BaseException as exc:primary=exc;status['reason']=type(exc).__name__+': '+str(exc)[:512];raise
 finally:
  try:write(out/'i5r3bbe_attempt_status.json',status);seal(out)
  except BaseException as secondary:
   if primary is None:raise
   if hasattr(primary,'add_note'):primary.add_note('BE finalization failed: '+str(secondary))
 return status
