"""Explicit APPROX-only recovery. Historical receipts remain in their original trial."""
import ast
import hashlib
import json
import os
import re
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

from .communication import digest, require
from .r3a_artifacts import ROOT
from . import r5_d5_l6a as l6
from . import r5_d5_l6b_runner as runner
from .r5_d5_lightperf_io import read_json, verify
from .r5_d5_lightperf_analysis import verify_run, compare
from .r5_d5_lightperf_workload import CONDITIONS
from .r5_d5_lightperf_runtime import gpu_uuid_matches, native_versions

BASE='513b9610488c11ee8aa9a538f7829ec5d1490360'
ORIGINAL=runner.NS/'long50_20261123T061210Z'
CONTROL={'src/semcache/edgelora/r5_d5_l6b.py','src/semcache/edgelora/r5_d5_l6b_runner.py'}
LIGHT='src/semcache/edgelora/r5_d5_lightperf.py'


def original(root):
    require(root==ORIGINAL and root.is_dir(),'BLOCKED_MISSING_REFERENCE: original recovery trial absent')
    p=read_json(root/'plan/plan.json')
    require(p['source']['commit']==BASE and p['entry_limit']==64 and p['count']==50 and
            p['resources']['worker_deadline_seconds']==9000,'original experiment differs')
    runs=[];allocations=[]
    for c,job in zip(CONDITIONS[:2],('223161','223163')):
        record=runner.verified(root,c)
        require(record['job']==job and record['affinity']==[40,41,88,89] and gpu_uuid_matches(record['gpu']['uuid'],'0dff7717-a97a-b957-b478-f26213d1b38e'),
                'original job/GPU differs')
        run,_=verify_run(root/c,root/'plan',root/'cpu')
        require(run['versions']==native_versions(),'historical/native package versions differ')
        runs.append(run);allocations.append(record)
    pair=compare(*runs)
    require(pair['status']=='DESCRIPTIVE_MATCHED_IDENTITY' and pair['generated_token_agreements']==50,
            'original FULL/EXACT not verified')
    return dict(plan=p,allocations=allocations,bindings={
        'plan':verify(root/'plan'),'cpu':verify(root/'cpu'),
        **{c:verify(root/c) for c in CONDITIONS[:2]}},
        controls={f:digest(read_json(root/f)) for c in CONDITIONS[:2]
                  for f in (c+'.job.json',c+'.allocation.json',c+'.complete.json')},
        trial_sha256=digest(read_json(root/'trial.json')))


def dependencies(old,new):
    """Verify all executable algorithm dependencies, including additions/deletions."""
    changed=[]
    for name in sorted(set(old)|set(new)):
        if name.startswith(('docs/','tests/')) or name in ('README.md','jobs/d5_l6_recovery.sh','scripts/166_r5_d5_l6_recovery.py'):
            continue
        if name.startswith('src/semcache/edgelora/r5_d5_l6_recovery'):
            continue
        if old.get(name)==new.get(name):
            continue
        require(name in CONTROL or name==LIGHT,'unreviewed inference dependency changed: '+name)
        if name==LIGHT:
            def body(data):
                node=ast.parse(data)
                node.body=[n for n in node.body if not isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) or n.name!='approval_modules']
                return ast.dump(node,include_attributes=False)
            before=subprocess.check_output(['git','show',BASE+':'+name],cwd=ROOT).decode()
            require(body(before)==body((ROOT/name).read_text()),'Light inference changed outside approval modules')
        changed.append(name)
    return dict(status='ALGORITHM_DEPENDENCIES_IDENTICAL',control_files_changed=changed,
                original_dependencies_sha256=digest(old),current_dependencies_sha256=digest(new))


def bind_plan(p,recovery):
    require(set(recovery) in ({'original','evidence','dependencies'},
            {'original','evidence','dependencies','gpu_policy'}), 'unknown recovery contract')
    require(recovery.get('gpu_policy','STRICT_UUID') in ('STRICT_UUID','SAME_RTX3090_SPEC'),
            'unknown recovery GPU policy')
    evidence=original(Path(recovery['original']))
    old=evidence['plan']
    variable={'source','plan_id','l6_plan_id','l6_binding','recovery'}
    require({k:v for k,v in old.items() if k not in variable}==
            {k:v for k,v in p.items() if k not in variable},'recovery workload/model/policy differs')
    dep=dependencies(old['source']['files'],p['source']['files'])
    require(recovery['evidence']=={k:v for k,v in evidence.items() if k!='plan'} and
            recovery['dependencies']==dep,'recovery historical evidence changed')
    result=dict(p,recovery=recovery)
    result.pop('plan_id')
    result['plan_id']=digest(result)
    return result


def topology(environment=None):
    cpus=sorted(os.sched_getaffinity(0))
    rows=[]
    for cpu in cpus:
        base=Path('/sys/devices/system/cpu')/('cpu'+str(cpu))
        rows.append(dict(cpu=cpu,package=(base/'topology/physical_package_id').read_text().strip(),
                         core=(base/'topology/core_id').read_text().strip(),
                         numa=sorted(p.name for p in base.glob('node[0-9]*')),
                         siblings=(base/'topology/thread_siblings_list').read_text().strip()))
    info=dict(line.split(':',1) for line in Path('/proc/cpuinfo').read_text().split('\n\n')[0].splitlines() if ':' in line)
    model={k.strip():v.strip() for k,v in info.items() if k.strip() in
           ('vendor_id','model name','cpu family','model','stepping','flags')}
    return dict(cpu_model=model,kernel=os.uname().release,cpu_count=len(cpus),affinity=cpus,cpus=rows,
                cpuinfo_sha256=hashlib.sha256(Path('/proc/cpuinfo').read_bytes()).hexdigest(),
                environment={k:(os.environ if environment is None else environment).get(k) for k in
                    ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','CUBLAS_WORKSPACE_CONFIG','CUDA_VISIBLE_DEVICES')},
                node=os.uname().nodename)


def gpu_node_match(old,new, *, allow_compatible_gpu=False, require_spec=False):
    require(old['node']==new['node'],'NOT_COMPARABLE: recovery node differs')
    require(len(new['affinity'])==4,'NOT_COMPARABLE: recovery CPU count differs')
    same=gpu_uuid_matches(old['gpu']['uuid'],new['gpu']['uuid'])
    require(same or allow_compatible_gpu,'NOT_COMPARABLE: recovery GPU UUID differs')
    if allow_compatible_gpu or require_spec:
        for gpu in (old['gpu'],new['gpu']):
            require(re.fullmatch(r'NVIDIA (?:GeForce )?RTX 3090',gpu.get('name','')) is not None,
                    'NOT_COMPARABLE: recovery GPU model differs from RTX 3090')
        require(type(new['gpu'].get('total_bytes')) is int and
                new['gpu']['total_bytes']==old['gpu'].get('total_bytes') and
                23*2**30 <= new['gpu']['total_bytes'] <= 24*2**30,
                'NOT_COMPARABLE: recovery GPU total memory differs')
        require(new['gpu'].get('device_count')==1,'recovery requires exactly one CUDA device')
        require(new['gpu'].get('free_bytes',0)>=12*2**30,'recovery GPU free memory below 12GiB')


RECOVERY_PROBE = """import json
result={}
try:
    import torch
    from semcache.edgelora.r5_d5_lightperf_runtime import native_versions
    count=torch.cuda.device_count()
    result.update(device_count=count,versions=native_versions(),cuda_compiled_version=torch.version.cuda)
    if count:
        p=torch.cuda.get_device_properties(0)
        result.update(uuid=str(p.uuid),name=p.name,total_bytes=p.total_memory,free_bytes=torch.cuda.mem_get_info(0)[0],
                      cudnn_version=torch.backends.cudnn.version())
except Exception as exc:
    result['probe_error']=str(exc)
print(json.dumps(result,allow_nan=False))
"""


def probe_and_validate(root,old,environment, *, allow_compatible_gpu=False):
    """Publish diagnostics before raising or returning permission to infer."""
    try:
        result=runner.subprocess.run([sys.executable,'-c',RECOVERY_PROBE],
            env=dict(environment,PYTHONPATH=str(ROOT/'src')),text=True,capture_output=True,timeout=120)
    except (OSError,subprocess.TimeoutExpired) as exc:
        result=SimpleNamespace(returncode=-1,stdout='',stderr=str(exc))
    gpu={}
    reasons=[]
    try:
        gpu=json.loads(result.stdout)
        require(isinstance(gpu,dict),'GPU probe must return an object')
    except (ValueError,TypeError) as exc:
        gpu={}
        reasons.append('invalid GPU probe output: '+str(exc))
    if result.returncode or gpu.get('probe_error'):
        reasons.append('GPU probe failed: '+str(gpu.get('probe_error',result.stderr[-2000:])))
    current=dict(node=environment.get('SLURMD_NODENAME'),gpu=gpu,affinity=sorted(os.sched_getaffinity(0)))
    try:
        gpu_node_match(old,current,allow_compatible_gpu=allow_compatible_gpu,require_spec=True)
    except (ValueError,KeyError,TypeError) as exc:
        reasons.append(str(exc))
    try:
        historical=read_json(ORIGINAL/CONDITIONS[0]/'run.json')
        require(gpu.get('versions')==historical['versions']==native_versions() and
                gpu.get('cuda_compiled_version')==historical['backend']['cuda_compiled_version'],
                'recovery GPU software versions differ')
        if 'cudnn_version' in historical['backend']:
            require(gpu.get('cudnn_version')==historical['backend']['cudnn_version'],
                    'recovery cuDNN version differs')
    except (ValueError,KeyError,TypeError) as exc:
        reasons.append(str(exc))
    diagnostic=dict(status='BLOCKED' if reasons else 'GPU_GATE_PASS',reasons=reasons,
        policy='SAME_RTX3090_SPEC' if allow_compatible_gpu else 'STRICT_UUID',
        expected=dict(old,cpu_count=len(old['affinity'])),actual=dict(current,cpu_count=len(current['affinity'])),environment={k:environment.get(k) for k in
            ('CUDA_VISIBLE_DEVICES','SLURM_JOB_GPUS','SLURM_STEP_GPUS')},
        device_count=gpu.get('device_count'),probe_returncode=result.returncode,
        probe_stderr=result.stderr[-2000:])
    runner.write_once(root/(CONDITIONS[2]+'.gpu_diagnostic.json'),diagnostic)
    require(not reasons,'recovery GPU gate blocked: '+'; '.join(reasons))
    return gpu



def prepare(root,commit,fingerprint, *, allow_compatible_gpu=False):
    identity=l6.source(commit,fingerprint)
    evidence=original(ORIGINAL)
    recovery=dict(original=str(ORIGINAL),evidence={k:v for k,v in evidence.items() if k!='plan'},
                  dependencies=dependencies(evidence['plan']['source']['files'],identity['files']),
                  gpu_policy='SAME_RTX3090_SPEC' if allow_compatible_gpu else 'STRICT_UUID')
    return runner.prepare(root,commit,fingerprint,recovery=recovery)


def submit(root, *, dry_run=False, allow_compatible_gpu=False):
    p=runner.ready(root)
    require('recovery' in p,'APPROX-only recovery plan required')
    require(allow_compatible_gpu == (p['recovery'].get('gpu_policy','STRICT_UUID')=='SAME_RTX3090_SPEC'),
            'GPU opt-in must match the newly approved recovery PLAN')
    evidence=original(ORIGINAL)
    node=evidence['allocations'][0]['node']
    require(node in runner.D5_NODES,'unsupported recovery node')
    trial=dict(node=node,assets=runner.assets(),plan_id=p['plan_id'],ready=digest(read_json(root/'ready.json')),
               python=sys.executable,order=[CONDITIONS[2]])
    args=runner.batch_args(root,node,CONDITIONS[2],None)
    args[-4]=str(ROOT/'jobs/d5_l6_recovery.sh')
    if dry_run:
        return dict(status='CPU_SLURM_DRY_RUN',gpu_policy=p['recovery'].get('gpu_policy','STRICT_UUID'),argv=args,test_only=[args[0],'--test-only',*args[1:]])
    require(str(ROOT)=='/data/khuss/repos/GP_semcache_lightperf','SERAPH repo required')
    runner.write_once(root/'trial.json',trial)
    runner.command([args[0],'--test-only',*args[1:]])
    job=runner.command(args).split(';')[0]
    require(job.isdecimal(),'invalid job ID')
    runner.write_once(root/(CONDITIONS[2]+'.job.json'),dict(job=job,condition=CONDITIONS[2],trial=digest(trial)))
    runner.command(['scontrol','release',job])
    return dict(status='RECOVERY_APPROX_SUBMITTED',job=job)


def analyze(root):
    from .r5_d5_l6a_analysis import cache_snapshots,occupancy,write_report
    from .r5_d5_lightperf_replay_analysis import segment,measured
    p=runner.ready(root)
    require('recovery' in p,'recovery plan required')
    evidence=original(ORIGINAL)
    require(p['recovery']['evidence']=={k:v for k,v in evidence.items() if k!='plan'},'historical inputs changed')
    alloc=runner.verified(root,CONDITIONS[2])
    gpu_node_match(evidence['allocations'][0],alloc,
                   allow_compatible_gpu=p['recovery'].get('gpu_policy')=='SAME_RTX3090_SPEC')
    runs=[verify_run(ORIGINAL/c,ORIGINAL/'plan',ORIGINAL/'cpu')[0] for c in CONDITIONS[:2]]
    runs.append(verify_run(root/CONDITIONS[2],root/'plan',root/'cpu')[0])
    require(all(len(r['requests'])==50 and r['tasks_sha256']==p['stream']['tasks_sha256'] and
                [x['replay_provenance'] for x in r['requests']]==p['stream']['occurrences'] for r in runs),
            'recovery workload/request coverage differs')
    require(len({row['quality']['metric'] for run in runs for row in run['requests']})==1,
            'quality metrics differ')
    report=dict(scope='CROSS_TRIAL_RECOVERY_DESCRIPTIVE',status='LIMITED_COMPARABILITY',
                correctness_qualified_speedup='NOT_ESTABLISHED',quality_preservation='NOT_ESTABLISHED',
                limitation='Different source/control code, allocation and CPU cores. Historical NUMA topology not captured.',
                strict_comparison=compare(runs[0],runs[2]),
                physical_gpu_uuid_changed=not gpu_uuid_matches(runs[0]['gpu']['uuid'],runs[2]['gpu']['uuid']),recovery_contract=p['recovery'],allocation=alloc,
                sequence_span_ms={r['condition']:r['sequence_span_ms'] for r in runs},
                input_inventories=dict(original=evidence['bindings'],
                    recovery_plan=verify(root/'plan'),recovery_cpu=verify(root/'cpu'),
                    recovery_run=verify(root/CONDITIONS[2])),
                cycle_metrics=[],cumulative_metrics=[])
    for run in runs:
        snapshots=cache_snapshots(run['requests'])
        for cycle in l6.CYCLES:
            for name,start in (('cycle_metrics',cycle['first']),('cumulative_metrics',0)):
                stop=cycle['last']+1
                rows=run['requests'][start:stop];full=runs[0]['requests'][start:stop]
                metrics=segment(rows)
                metrics.update(scope=report['scope'],condition=run['condition'],cycle=cycle['cycle'],first=start,last=stop-1,
                    quality_metric=rows[0]['quality']['metric'],
                    quality_mean=sum(r['quality']['score'] for r in rows)/len(rows),
                    attention_impact_ms=measured([(r.get('attention_impact') or {}).get('elapsed_ms') for r in rows]),
                    cache=occupancy(snapshots[start:stop]),descriptive_full_over_condition_e2e=None,
                    descriptive_speedup_status='NOT_COMPARABLE_CPU_SOURCE_ALLOCATION',
                    generated_token_agreements=sum(a['generated_token_ids']==b['generated_token_ids'] for a,b in zip(full,rows)),
                    observed_condition_minus_full_e2e_ms=[b['application_e2e_ms']-a['application_e2e_ms'] for a,b in zip(full,rows)])
                report[name].append(metrics)
    dest=l6.NS/(root.name+'_recovery_analysis')
    return write_report(report,dest,[root,ORIGINAL])
