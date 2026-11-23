"""One explicit trial, three independent allocations, immutable registration."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from types import SimpleNamespace

from . import r5_d5_l6a as l6
from . import r5_d5_lightperf as light
from .communication import digest, require
from .r3a_artifacts import ROOT
from .r5_d5_l6b import NS, native_plan
from .r5_d5_lightperf_io import SummaryWriter, fresh, no_symlinks, read_json, verify
from .r5_d5_lightperf_runtime import gpu_uuid_matches
from .r5_d5_lightperf_workload import CONDITIONS
from .r5_d5_sweep import D5_NODES, GPU_PROBE, official, write_once

JOB=ROOT/'jobs/d5_l6b.sh'


def state(tag):
    require(re.fullmatch(r'[A-Za-z0-9_-]{1,80}',tag),'invalid trial ID')
    return no_symlinks(NS/tag)


def paths(root):
    return dict(plan=l6.NS/(root.name+'_plan'),audit=l6.NS/(root.name+'_cpu'),
                light_plan=root/'plan',light_audit=root/'cpu')


def prepare(root,commit,fingerprint, *, recovery=None):
    l6.cpu_guard()
    offline=assets()
    os.environ['HF_HUB_CACHE']=str(Path(offline['HF_HOME'])/'hub')
    args=SimpleNamespace(command='prepare-cpu',reviewed_commit=commit,reviewed_fingerprint=fingerprint,
                         entry_limit=64,worker_deadline_seconds=9000,reference=l6.REFERENCE)
    l6.source(commit,fingerprint)
    from .r5_d5_l6b import L6A_HEAD
    subprocess.run(['git','merge-base','--is-ancestor',L6A_HEAD,'HEAD'],cwd=ROOT,check=True)
    root.mkdir(parents=True,exist_ok=False)
    loc=paths(root)
    args.output=loc['plan']
    require(l6.execute(args)['status']=='STATIC_BOUNDS_PASS_RUNTIME_CHECKS_PENDING','L6 preflight blocked')
    args.command,args.plan,args.output='cpu-validate',loc['plan'],loc['audit']
    require(l6.execute(args)['status']=='CPU_APPROVED','fresh L6-A approval required')
    l6.checked(loc['plan'])
    p=native_plan(loc['plan'],loc['audit'])
    if recovery is not None:
        from .r5_d5_l6_recovery import bind_plan
        p=bind_plan(p,recovery)
    w=SummaryWriter(fresh(loc['light_plan'],[loc['plan'],loc['audit']],namespace=NS))
    w.write('plan.json',p)
    w.seal()
    # Start the second approval suite in a fresh interpreter after the L6-A suite.
    official(['cpu-validate','--experiment',loc['light_plan'],'--output-dir',loc['light_audit']])
    _,pi,ai=light.approval(loc['light_plan'],loc['light_audit'])
    write_once(root/'ready.json',dict(plan=pi,cpu=ai,native_id=p['plan_id'],python=sys.executable,assets=offline))
    return 'CPU_APPROVED'


def ready(root):
    offline=assets()
    os.environ['HF_HUB_CACHE']=str(Path(offline['HF_HOME'])/'hub')
    loc=paths(root)
    p,pi,ai=light.approval(loc['light_plan'],loc['light_audit'])
    require(read_json(root/'ready.json')==dict(plan=pi,cpu=ai,native_id=p['plan_id'],python=sys.executable,assets=offline)
            and p['l6_binding']['plan']==str(loc['plan']) and p['l6_binding']['audit']==str(loc['audit']),
            'trial/Python/native approval differs')
    return p


def command(argv):
    result=subprocess.run(argv,text=True,capture_output=True)
    require(result.returncode==0, f'{argv[0]} exited {result.returncode}: {result.stderr[-2000:]}')
    return result.stdout.strip()


def assets():
    values={}
    for key in ('HF_HOME','TMPDIR'):
        raw=os.environ.get(key)
        require(raw and Path(raw).is_absolute(),'set absolute '+key)
        path=no_symlinks(Path(raw))
        require(path.is_dir(),key+' absent')
        values[key]=str(path)
    require(os.access(values['TMPDIR'],os.W_OK|os.X_OK),'TMPDIR is not writable')
    require((Path(values['HF_HOME'])/'hub').is_dir(),'offline HF hub absent')
    return values


def batch_args(root,node,c,previous):
    args=['sbatch','--parsable','--hold','--no-requeue','--kill-on-invalid-dep=yes',
          '--partition=batch_eebme_ugrad','--nodelist='+node,'--nodes=1-1','--ntasks=1',
          '--gres=gpu:1','--cpus-per-task=4','--mem=64G','--time=04:30:00','--chdir='+str(ROOT),
          '--output='+str(root/'%j.out'),'--error='+str(root/'%j.err')]
    if previous:
        args.append('--dependency=afterok:'+previous)
    return args+[str(JOB),sys.executable,root.name,c]


def submit(root,node):
    require(node in D5_NODES and str(ROOT)=='/data/khuss/repos/GP_semcache_lightperf','SERAPH node/repo required')
    p=ready(root)
    require('recovery' not in p,'use APPROX-only recovery submit')
    trial=dict(node=node,assets=assets(),plan_id=p['plan_id'],ready=digest(read_json(root/'ready.json')),
               python=sys.executable,order=list(CONDITIONS))
    write_once(root/'trial.json',trial)
    ids=[]
    for c in CONDITIONS:
        args=batch_args(root,node,c,ids[-1] if ids else None)
        command([args[0],'--test-only',*args[1:]])
        job=command(args).split(';')[0]
        require(job.isdecimal() and job not in ids,'invalid Slurm job registration')
        ids.append(job)
        write_once(root/(c+'.job.json'),dict(job=job,condition=c,trial=digest(trial)))
    for job in ids:
        command(['scontrol','release',job])
    return ids


def allocation(root,c):
    t=read_json(root/'trial.json')
    reg=read_json(root/(c+'.job.json'))
    env=os.environ
    require(reg==dict(job=env['SLURM_JOB_ID'],condition=c,trial=digest(t)) and
            t['node']==env['SLURMD_NODENAME'] and t['node'] in D5_NODES and t['python']==sys.executable and
            not env.get('SLURM_ARRAY_TASK_ID') and int(env.get('SLURM_RESTART_COUNT',0))==0 and
            int(env['SLURM_JOB_NUM_NODES'])==1 and int(env['SLURM_CPUS_PER_TASK'])==4 and
            int(env['SLURM_MEM_PER_NODE'])>=65536,'unregistered allocation/node/resources')
    return t


def budget(p, *, record=None):
    job=os.environ['SLURM_JOB_ID']
    fields=dict(x.split('=',1) for x in command(['scontrol','show','job','--oneliner',job]).split() if '=' in x)
    require(fields['JobId']==job and fields['Partition']=='batch_eebme_ugrad' and fields['Requeue']=='0' and
            fields['JobState']=='RUNNING' and fields['TimeLimit']=='04:30:00' and
            fields['NodeList']==os.environ['SLURMD_NODENAME'],
            'scheduler identity/requeue differs')
    now=time.time()
    remaining=l6.allocation_remaining(p['resources'],datetime.fromisoformat(fields['EndTime']).timestamp(),now)
    if record is not None:
        write_once(record,dict(job=job,end_time=fields['EndTime'],observed_at=now,remaining_seconds=remaining,
                               time_limit=fields['TimeLimit'],deadline_seconds=p['resources']['worker_deadline_seconds']))
    return remaining


def native_gate(p,experiment,audit,c,out,uuid):
    root=no_symlinks(Path(experiment)).parent
    require(root==state(root.name) and Path(experiment)==root/'plan' and Path(audit)==root/'cpu' and
            no_symlinks(Path(out))==root/c,'L6 run path differs')
    trial=allocation(root,c)
    if 'recovery' in p:
        require(c==CONDITIONS[2], 'recovery is APPROX only')
    record=read_json(root/(c+'.allocation.json'))
    require(digest(read_json(root/'ready.json'))==trial['ready'] and p['plan_id']==trial['plan_id'] and gpu_uuid_matches(uuid,record['gpu']['uuid']) and
            sorted(os.sched_getaffinity(0))==record['affinity'],'L6 execution binding differs')
    return budget(p)


def matched(first,current):
    require(gpu_uuid_matches(first['gpu']['uuid'],current['gpu']['uuid']) and
            first['node']==current['node'] and first['affinity']==current['affinity'],
            'NOT_COMPARABLE: allocated GPU/node/CPU affinity differs')


def worker(root,c, *, recovery_mode=False):
    trial=allocation(root,c)
    cpus=sorted(os.sched_getaffinity(0))
    require(os.environ.get('CUDA_VISIBLE_DEVICES') and len(cpus)>=4,'bound Slurm step required')
    require(not recovery_mode or c==CONDITIONS[2],'recovery is APPROX only')
    pinned=cpus[:4] if c==CONDITIONS[0] or recovery_mode else read_json(root/(CONDITIONS[0]+'.allocation.json'))['affinity']
    require(len(set(pinned))==4 and set(pinned)<=set(cpus),'NOT_COMPARABLE: bound CPU set differs')
    os.sched_setaffinity(0,pinned)
    allocation_env=dict(os.environ)
    os.environ.update(CUDA_VISIBLE_DEVICES='',HIP_VISIBLE_DEVICES='',**trial['assets'])
    p=ready(root)
    require(digest(read_json(root/'ready.json'))==trial['ready'] and p['plan_id']==trial['plan_id'] and assets()==trial['assets'],'trial assets/plan differ')
    recovery=p.get('recovery')
    require(bool(recovery)==recovery_mode,'explicit recovery mode must match approved plan')
    prior=[] if recovery is not None else CONDITIONS[:CONDITIONS.index(c)]
    for prev in prior:
        verified(root,prev)
    env=dict(allocation_env)
    env.update(trial['assets'],TORCH_HOME=str(Path(trial['assets']['TMPDIR'])/'torch'),
               CUDA_CACHE_PATH=str(Path(trial['assets']['TMPDIR'])/'cuda_cache'),HF_HUB_CACHE=str(Path(trial['assets']['HF_HOME'])/'hub'),R5_ENABLE_GPU='1',
               R5_I5_ENABLE_GPU='1',D5L_ENABLE_GPU='1',D5N_NODE=trial['node'],PYTHONHASHSEED='42',
               CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
    budget(p)
    if recovery is not None:
        from .r5_d5_l6_recovery import original, topology, probe_and_validate
        evidence=original(Path(recovery['original']))
        gpu=probe_and_validate(root,evidence['allocations'][0],env,
                              allow_compatible_gpu=recovery.get('gpu_policy')=='SAME_RTX3090_SPEC')
    else:
        probe=subprocess.run([sys.executable,'-c',GPU_PROBE],env=dict(env,PYTHONPATH=str(ROOT/'src')),
                             check=True,text=True,capture_output=True)
        gpu=json.loads(probe.stdout)
    record=dict(job=os.environ['SLURM_JOB_ID'],node=trial['node'],gpu=gpu,affinity=sorted(os.sched_getaffinity(0)))
    if recovery is not None:
        record['cpu_topology']=topology(env)
    write_once(root/(c+'.allocation.json'),record)
    if prior:
        matched(read_json(root/(prior[0]+'.allocation.json')),record)
    env['D5L_ALLOCATION_REMAINING_SECONDS']=str(budget(p,record=root/(c+'.budget.json')))
    argv=['run','--experiment',root/'plan','--cpu-audit',root/'cpu','--output-dir',root/c,
          '--source-commit',p['source']['commit'],'--expected-gpu-uuid',gpu['uuid'],'--condition',c,
          '--enable-gpu-experiment','--enable-continuous-replay','--enable-extended-experiment']
    if c==CONDITIONS[2]:
        argv.append('--enable-approximate-experiment')
    official(argv,gpu=True,environment=env)
    result=official(['verify','--experiment',root/'plan','--cpu-audit',root/'cpu','--run',root/c])
    require(result['status']=='LIGHT_RECEIPT_INTEGRITY_PASS','official verification failed')
    run=read_json(root/c/'run.json')
    matched(record,dict(gpu=run['gpu'],node=trial['node'],affinity=run['cpu_concurrency']['affinity']))
    if prior:
        from .r5_d5_lightperf_analysis import compare
        pair=compare(read_json(root/CONDITIONS[0]/'run.json'),run)
        require(pair['status']=='DESCRIPTIVE_MATCHED_IDENTITY','NOT_COMPARABLE: '+str(pair.get('mismatches')))
        require(c!=CONDITIONS[1] or pair['generated_token_agreements']==50,'EXACT token mismatch')
    write_once(root/(c+'.complete.json'),dict(allocation=digest(record),binding=result['binding']))
    return 'CONDITION_COMPLETE'


def verified(root,c):
    from .r5_d5_lightperf_analysis import verify_run
    r,b=verify_run(root/c,root/'plan',root/'cpu')
    record=read_json(root/(c+'.allocation.json'))
    reg=read_json(root/(c+'.job.json'))
    trial=read_json(root/'trial.json')
    require(r['condition']==c and len(r['requests'])==50 and record['job']==reg['job'] and
            reg['condition']==c and reg['trial']==digest(trial) and record['node']==trial['node'] and
            read_json(root/(c+'.complete.json'))==dict(allocation=digest(record),binding=b),'incomplete condition')
    matched(record,dict(gpu=r['gpu'],node=trial['node'],affinity=r['cpu_concurrency']['affinity']))
    return record


def aggregate(root):
    from .r5_d5_l6a_analysis import analyze
    ready(root)
    records=[verified(root,c) for c in CONDITIONS]
    require(len({r['job'] for r in records})==3,'independent jobs required')
    for r in records[1:]:
        matched(records[0],r)
    loc=paths(root)
    p,_=l6.checked(loc['plan'])
    args=SimpleNamespace(**loc,full=root/CONDITIONS[0],exact=root/CONDITIONS[1],approx=root/CONDITIONS[2],
                         output=l6.NS/(root.name+'_analysis'))
    result=analyze(p,args)
    report=read_json(args.output/'long_replay_analysis.json')
    require(all(len(report[k])==15 for k in ('cycle_metrics','cumulative_metrics')), 'incomplete cycle analysis')
    write_once(root/'matched.json',dict(status='MATCHED_COMPLETE',scope='SINGLE_TRIAL_DESCRIPTIVE',
               correctness_qualified_speedup='NOT_ESTABLISHED',trial=digest(read_json(root/'trial.json')),
               allocations=records,analysis=str(args.output),inventory=result['inventory']))
    return result


def scheduler_status(root, *, record_failures=False):
    """Read scheduler accounting. Only explicit reconcile publishes terminal failures."""
    require(root.is_dir(),'unknown trial')
    result={}
    for c in CONDITIONS:
        path=root/(c+'.job.json')
        if not path.exists():
            result[c]='UNREGISTERED'
            continue
        job=read_json(path)['job']
        rows=command(['sacct','-n','-P','-X','-j',job,'--format=JobIDRaw,State,ExitCode']).splitlines()
        values=[row.split('|')[1:3] for row in rows if row.split('|')[0]==job]
        require(len(values)<=1,'ambiguous job accounting')
        state,code=values[0] if values else ('PENDING_ACCOUNTING','')
        result[c]=dict(job=job,state=state,exit_code=code,completion_record_present=(root/(c+'.complete.json')).is_file())
        terminal=state.split()[0] in ('CANCELLED','FAILED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL',
                                     'BOOT_FAIL','PREEMPTED','DEADLINE','REVOKED')
        dest=root/(c+'.scheduler_failure.json')
        if record_failures and terminal and not dest.exists():
            write_once(dest,dict(status='PARTIAL_BLOCKED',**result[c]))
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='cmd',required=True)
    for name in ('check','prepare-cpu','inspect','submit','status','reconcile','_registered','_job','analyze'):
        cmd=sub.add_parser(name)
        if name!='check':
            cmd.add_argument('--trial',required=True)
        if name=='prepare-cpu':
            cmd.add_argument('--reviewed-commit',required=True)
            cmd.add_argument('--reviewed-fingerprint',required=True)
        if name=='submit':
            cmd.add_argument('--enable-approximate-experiment',action='store_true',required=True)
            cmd.add_argument('--node',choices=D5_NODES,default='moana-y3')
        if name in ('_registered','_job'):
            cmd.add_argument('--condition',choices=CONDITIONS,required=True)
    a=parser.parse_args(argv)
    try:
        if a.cmd!='_job':
            l6.cpu_guard()
        if a.cmd=='check':
            p=l6.build(l6.REFERENCE,l6.source(None,None),64,9000)
            require(p['stream']['statistics']['per_adapter_requests']=={'Adapter_A':20,'Adapter_C':30} and
                    p['resources']['forecast_seconds']==7824.54483968, 'L6-B reference routing/forecast differs')
            result=p['resources']
        else:
            root=state(a.trial)
            if a.cmd=='prepare-cpu':
                result=prepare(root,a.reviewed_commit,a.reviewed_fingerprint)
            elif a.cmd=='inspect':
                p=ready(root)
                result=dict(status='FRESH_NATIVE_CPU_APPROVAL',native_plan_id=p['plan_id'],
                            l6_plan_id=p['l6_plan_id'],tests=read_json(root/'cpu/cpu_validation.json')['tests'],
                            source_commit=p['source']['commit'],fingerprint=p['source']['content_fingerprint'],
                            worker_deadline_seconds=p['resources']['worker_deadline_seconds'])
            elif a.cmd=='submit':
                result=submit(root,a.node)
            elif a.cmd=='_registered':
                allocation(root,a.condition)
                result='REGISTERED_ALLOCATION'
            elif a.cmd=='_job':
                result=worker(root,a.condition)
            elif a.cmd=='analyze':
                result=aggregate(root)
            else:
                result=scheduler_status(root,record_failures=a.cmd=='reconcile')
        print(json.dumps(result))
        return 2 if isinstance(result,dict) and result.get('status')=='BLOCKED_INFEASIBLE' else 0
    except Exception as exc:
        message=dict(status='PARTIAL_BLOCKED',stage=a.cmd,reason=str(exc))
        if 'NOT_COMPARABLE' in str(exc):
            message['comparability']='NOT_COMPARABLE'
        if a.cmd=='check' and 'BLOCKED_MISSING_REFERENCE' in str(exc):
            message['status']='BLOCKED_MISSING_REFERENCE'
        if a.cmd in ('submit','_job') and 'root' in locals() and root.is_dir() and (root/'trial.json').is_file():
            from .r5_d5_sweep import event
            event(root,'L6_BLOCKED',condition=getattr(a,'condition',None),**message)
        print(json.dumps(message),file=sys.stderr)
        return 2
