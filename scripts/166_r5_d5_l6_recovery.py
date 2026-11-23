#!/usr/bin/env python3
"""APPROX-only recovery control. CPU commands never query CUDA."""
import argparse
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode=True
os.environ.update(PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
if sys.argv[1:2]!=['_job']:
    os.environ.update(CUDA_VISIBLE_DEVICES='',HIP_VISIBLE_DEVICES='')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from semcache.edgelora import r5_d5_l6_recovery as recovery
from semcache.edgelora.r5_d5_lightperf_workload import CONDITIONS


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='cmd',required=True)
    for name in ('check','prepare-cpu','inspect','dry-run','submit','analyze','status','reconcile','_registered','_job'):
        cmd=sub.add_parser(name)
        if name!='check':cmd.add_argument('--trial',required=True)
        if name=='prepare-cpu':
            cmd.add_argument('--reviewed-commit',required=True)
            cmd.add_argument('--reviewed-fingerprint',required=True)
        if name=='submit':cmd.add_argument('--enable-approximate-experiment',action='store_true',required=True)
    a=parser.parse_args()
    try:
        if a.cmd!='_job':recovery.l6.cpu_guard()
        if a.cmd=='check':
            data=recovery.original(recovery.ORIGINAL)
            result=dict(status='HISTORICAL_RECEIPTS_VERIFIED',evidence={k:v for k,v in data.items() if k!='plan'})
        else:
            root=recovery.runner.state(a.trial)
            if root==recovery.ORIGINAL:raise ValueError('original trial is read-only')
            if a.cmd=='prepare-cpu':result=recovery.prepare(root,a.reviewed_commit,a.reviewed_fingerprint)
            elif a.cmd=='inspect':
                p=recovery.runner.ready(root)
                if 'recovery' not in p:raise ValueError('recovery plan required')
                result=dict(status='FRESH_RECOVERY_NATIVE_CPU_APPROVAL',plan_id=p['plan_id'])
            elif a.cmd in ('submit','dry-run'):result=recovery.submit(root,dry_run=a.cmd=='dry-run')
            elif a.cmd=='analyze':result=recovery.analyze(root)
            elif a.cmd=='_registered':
                if 'recovery' not in recovery.read_json(root/'plan/plan.json'):raise ValueError('recovery plan required')
                result=recovery.runner.allocation(root,CONDITIONS[2])
            elif a.cmd=='_job':
                if 'recovery' not in recovery.read_json(root/'plan/plan.json'):raise ValueError('recovery plan required')
                result=recovery.runner.worker(root,CONDITIONS[2],recovery_mode=True)
            else:result=recovery.runner.scheduler_status(root,record_failures=a.cmd=='reconcile')
        print(json.dumps(result));return 0
    except Exception as exc:
        message=dict(status='BLOCKED_MISSING_REFERENCE' if 'BLOCKED_MISSING_REFERENCE' in str(exc) else 'PARTIAL_BLOCKED',stage=a.cmd,reason=str(exc))
        if a.cmd in ('submit','_job') and 'root' in locals() and root!=recovery.ORIGINAL and root.is_dir():
            from semcache.edgelora.r5_d5_sweep import event
            event(root,'RECOVERY_FAILED',**message)
        print(json.dumps(message),file=sys.stderr);return 2

if __name__=='__main__':sys.exit(main())
