"""Bounded diagnostic child lifecycle. No GPU/model imports or Slurm calls."""
import os
from pathlib import Path
import signal
import subprocess
import sys

from .r3a_artifacts import ROOT


def launch(kind, plan, output, *, new_session=True):
    env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
             CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
    if kind=='ud':env.update(CUDA_VISIBLE_DEVICES='',HIP_VISIBLE_DEVICES='')
    stdout=(Path(output)/(kind+'.stdout.log')).open('w');stderr=(Path(output)/(kind+'.stderr.log')).open('w')
    try:
        process=subprocess.Popen([sys.executable,'-u',str(ROOT/'scripts/97_attribute_r3a_cpu_delta.py'),
            '--worker',kind,'--plan',str(plan),'--enable-gpu-probes'],cwd=ROOT,env=env,
            stdout=stdout,stderr=stderr,start_new_session=new_session)
    except BaseException:
        stdout.close();stderr.close();raise
    return process,stdout,stderr


def finish(child, *, group=True):
    process,stdout,stderr=child;forced=process.poll() is None;descendant_cleanup=False
    try:
        if forced:
            try:
                if group:os.killpg(process.pid,signal.SIGTERM)
                else:process.terminate()
            except ProcessLookupError:pass
            try:process.wait(timeout=5.)
            except subprocess.TimeoutExpired:
                if group:os.killpg(process.pid,signal.SIGKILL)
                else:process.kill()
                process.wait(timeout=5.)
        else:process.wait(timeout=1.)
        if group:
            # An ES can exit unexpectedly while its inherited-group CPU UD is
            # still alive. Clean that group even after the leader was reaped.
            try:
                os.killpg(process.pid,signal.SIGKILL)
                descendant_cleanup=True
            except ProcessLookupError:pass
    finally:
        stdout.close();stderr.close()
    return dict(pid=process.pid,returncode=process.returncode,forced_termination=forced,
                process_group=process.pid if group else 'inherited diagnostic ES group',
                descendant_group_cleanup_applied=descendant_cleanup,
                all_children_exited=process.returncode is not None)
