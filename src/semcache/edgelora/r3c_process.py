"""Isolated R3-C workers; socket workers never load model dependencies."""
import os
from pathlib import Path
import subprocess
import sys
import time

from .communication import require
from .r3a_artifacts import ROOT,read,write
from .parity_causal_process import finish


def spawn(kind,plan,output,*,new_session=True):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    write(output/(kind+'_plan.json'),dict(plan,output=str(output)))
    env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',
        TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
    if kind in ('proxy','echo','ud'):env.update(CUDA_VISIBLE_DEVICES='',HIP_VISIBLE_DEVICES='')
    out=(output/(kind+'.stdout.log')).open('w');err=(output/(kind+'.stderr.log')).open('w')
    try:
        process=subprocess.Popen([sys.executable,'-u',str(ROOT/'scripts/100_run_r3c_network_one_shot.py'),
            '--worker',kind,'--plan',str(output/(kind+'_plan.json')),
            '--test-fixture-only' if plan['fixture'] else '--enable-gpu-experiment'],cwd=ROOT,env=env,
            stdout=out,stderr=err,start_new_session=new_session)
    except BaseException:out.close();err.close();raise
    return process,out,err


def ready(child,path,timeout=30.):
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        if Path(path).is_file():return read(path)
        require(child[0].poll() is None,'worker exited before readiness; inspect stderr')
        time.sleep(.01)
    raise TimeoutError('worker readiness deadline')


def reap(child,path,*,group=True):
    result=finish(child,group=group);write(path,result);return result
