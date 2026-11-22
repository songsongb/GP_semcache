"""V5 CPU-first sweep control. Only explicit submit and allocated _job run Slurm/GPU.

State/receipts are append-only. Historical reference approval is evidence of its
own plan, never authorization for this source. No retry/overwrite path.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import uuid

from .communication import digest, require
from .r3a_artifacts import ROOT
from .r5_d5_impact import MASK_INDEX_REUSE
from .r5_d5_lightperf_io import ENGINE, MODE, SCHEMA as LIGHT_SCHEMA, no_symlinks, read_json, source_identity, verify
from .r5_d5_lightperf_workload import CONDITIONS, knobs
from .r5_d5_lightperf_replay import validate_replay_stream
from .r5_subsequence_natural_runtime import D5_NODES

CLI = ROOT / 'scripts/159_r5_d5_lightperf.py'
JOB = ROOT / 'jobs/d5_l2e_bw_ud_sweep_v5.sh'
FROZEN_PLAN = ROOT / 'results/r5/d5_lightperf/l2e_ud025_20261121T083703Z_l2e_plan/plan.json'
FROZEN_CPU = ROOT / 'results/r5/d5_lightperf/l2e_ud025_20261121T083703Z_l2e_cpu'
V4_RUNNER = Path('/data/khuss/tmp/d5_l2e_matched_repeat_v4.sh')
CANARIES = ('bw050_ud025', 'bw050_ud010')
SCHEMA = 'D5_L2E_SWEEP_V5'
MAN_NODELIST_TEXT = 'a lower node or processor count may only require a subset of the supplied node list'


def cells():
    return [dict(id=f'bw{bw:03d}_ud{percent:03d}', performance_knobs=knobs(bw, factor),
                 order=[CONDITIONS[(j + i) % 3] for j in range(3)])
            for i, (bw, percent, factor) in enumerate(
                (b, p, f) for b in (50, 100, 200) for p, f in ((100, 1.), (50, .5), (25, .25), (10, .10)))]


def select(text, *, submit=False):
    ids = [c['id'] for c in cells()]
    chosen = ids if text == 'all' and not submit else text.split(',')
    require(chosen and len(chosen) == len(set(chosen)) and all(c in ids for c in chosen), 'explicit valid unique cell IDs required')
    require(not submit or len(chosen) < 12, 'submit never queues the entire 12-cell grid')
    return chosen


def write_once(path, value):
    path = no_symlinks(Path(path).absolute())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def event(root, kind, **fields):
    write_once(Path(root) / 'events' / f'{time.time_ns()}_{uuid.uuid4().hex}.json',
               dict(schema=SCHEMA, kind=kind, **fields))


def events(root):
    return [read_json(p) for p in sorted((Path(root) / 'events').glob('*.json'))]


@contextmanager
def control_lock(root):
    # A persistent empty synchronization file, not a mutable result/receipt.
    with (Path(root) / '.control.lock').open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def cpu_env():
    return dict(os.environ, CUDA_VISIBLE_DEVICES='', HIP_VISIBLE_DEVICES='',
                HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
                PYTHONDONTWRITEBYTECODE='1', PYTHONHASHSEED='42')


def official(arguments, *, gpu=False, environment=None):
    result = subprocess.run([sys.executable, str(CLI), *map(str, arguments)], cwd=ROOT,
                            env=environment if gpu else cpu_env(), capture_output=True, text=True)
    require(result.returncode == 0, f'official {arguments[0]} failed: {(result.stderr or result.stdout)[-3000:]}')
    return json.loads(result.stdout)


def historical_reference(plan_path, cpu_path):
    plan_path, cpu_path = no_symlinks(Path(plan_path).absolute()), no_symlinks(Path(cpu_path).absolute())
    require(plan_path.is_file() and (cpu_path / 'cpu_validation.json').is_file(), 'BLOCKED_MISSING_REFERENCE: frozen PLAN/CPU audit absent')
    pi, ai = verify(plan_path.parent), verify(cpu_path)
    p, a = read_json(plan_path), read_json(cpu_path / 'cpu_validation.json')
    body = dict(p); pid = body.pop('plan_id')
    require(p['engine']==a['engine']==ENGINE and p['schema']==LIGHT_SCHEMA and p['mode']==MODE and
            p['source']['approval_scope']=='CLEAN_REVIEWED_SOURCE' and
            digest(body) == pid and a['plan_inventory'] == pi and a['source'] == p['source'], 'historical plan/audit binding differs')
    require(a['status'] == 'PASS' and a['failed'] == a['errors'] == a['skipped'] == 0 and
            a['scope'] == 'CLEAN_NATIVE_CPU_APPROVAL', 'invalid historical CPU evidence')
    require(p['dataset'] == '2wikimqa' and p['count'] == 20 and p['workload_mode'] == 'CONTINUOUS_REPLAY' and
            p['enable_continuous_replay'] is True and p['impact_implementation'] == MASK_INDEX_REUSE and
            p['performance_knobs'] == knobs(50, .25), 'wrong frozen 2WikiMQA replay reference')
    require(p['validation_mode']=='REQUEST_SCOPED_VERIFIED' and p['conditions']==list(CONDITIONS) and
            p['warmup_requests']==0 and p['approx_requires_explicit_run_opt_in'] is True,
            'historical safety contract differs')
    validate_replay_stream(p['stream'])
    require(p['seed'] == p['stream']['seed'], 'reference seed differs')
    require(len(p['stream']['source_pool'])==10 and
            all(o['occurrence_index']==i//10 for i,o in enumerate(p['stream']['occurrences'])),
            'reference is not original first10/repeated10 replay')
    require(all(t['policy']['max_new_tokens']==8 for t in p['stream']['tasks']), 'frozen 8-token cap differs')
    for key in ('original_experiment', 'multi', 'continuous_contract', 'strict_pilot', 'baseline_plan', 'ud_operation_seconds'):
        require(key in p, 'missing reference field: ' + key)
    for key in ('original_experiment', 'multi', 'strict_pilot', 'baseline_plan'):
        value = p[key]
        if value is not None:
            require(isinstance(value, str) and Path(value).is_absolute() and Path(value).is_dir(), 'missing original asset path: ' + key)
    require(p['original_experiment'] is not None and p['multi'] is not None, 'required asset path is null')
    contract = p['continuous_contract']
    if isinstance(contract, str):
        require(Path(contract).is_absolute() and Path(contract).is_file(), 'missing contract path')
        contract = read_json(contract)
    require(contract is None or isinstance(contract, dict), 'invalid optional contract type')
    # The official replay implementation itself rejects non-null contracts; do not invent/clear one.
    summary = dict(plan=str(plan_path), cpu_audit=str(cpu_path), plan_inventory=pi, cpu_inventory=ai,
                   seed=p['seed'], tasks_sha256=p['stream']['tasks_sha256'], stream_sha256=digest(p['stream']),
                   request_order_sha256=digest([t['request_id'] for t in p['stream']['tasks']]),
                   source=p['source'], fields={k:p[k] for k in ('original_experiment','multi','continuous_contract',
                       'strict_pilot','baseline_plan','ud_operation_seconds','workload_mode','enable_continuous_replay','impact_implementation')})
    return p, contract, summary


def same_workload(p, frozen):
    keys = ('dataset','count','seed','stream','impact_implementation','workload_mode','enable_continuous_replay',
            'original_experiment','multi','strict_pilot','baseline_plan','ud_operation_seconds','continuous_contract',
            'original_inventory','multi_inventory','base_prepared','tokenizer_identity','semantic_identity','transport',
            'dataset_inventories','dataset_paths','dataset_revision','dataset_split','strict_pilot_inventory',
            'conditions','validation_mode','thresholds','warmup_requests','approx_requires_explicit_run_opt_in')
    require(all(k in p and k in frozen and p[k] == frozen[k] for k in keys), 'new plan changed frozen workload/model/generation identity')


def check(plan_path, cpu_path, node):
    result = dict(schema=SCHEMA, phase='A_B_ONLY', gpu='NOT_RUN', matrix=cells(), v4_runner_exists=V4_RUNNER.is_file())
    try:
        _, _, reference = historical_reference(plan_path, cpu_path)
        # Report no token/input payload. Historical source is provenance, not new approval.
        result.update(status='REFERENCE_READ_ONLY_VALID', reference={k:v for k,v in reference.items() if k != 'source'})
    except (ValueError, OSError, KeyError) as exc:
        result.update(status='BLOCKED_MISSING_REFERENCE' if not Path(plan_path).is_file() or
                      not (Path(cpu_path)/'cpu_validation.json').is_file() else 'PARTIAL_BLOCKED', reason=str(exc))
    if node != 'auto':
        require(node in D5_NODES, 'unapproved node')
    else:
        try:
            auto_node_arguments(local_sbatch_manual())
            result['auto_nodes'] = 'LOCAL_DOCUMENTED_SUBSET_SEMANTICS'
        except (ValueError, OSError) as exc:
            result['auto_nodes'] = 'BLOCKED_UNCONFIRMED_SLURM_SEMANTICS'
            result['auto_reason'] = str(exc)
    return result


def definition(root):
    d = read_json(Path(root) / 'definition.json')
    require(d['schema'] == SCHEMA, 'unknown sweep definition')
    current = source_identity(reviewed_commit=d['source']['commit'], reviewed_fingerprint=d['source']['content_fingerprint'])
    require(current == d['source'] and str(Path(sys.executable).absolute()) == d['python'], 'source/Python changed')
    return d


def protect_reference_output(root, p, reference):
    protected=[Path(reference['plan']).parent,Path(reference['cpu_audit'])]
    protected += [Path(p[k]) for k in ('original_experiment','multi','strict_pilot','baseline_plan','base_prepared')
                  if isinstance(p.get(k),str)]
    protected += [Path(v) for v in p.get('dataset_paths',{}).values() if isinstance(v,str)]
    if isinstance(p.get('continuous_contract'),str):protected.append(Path(p['continuous_contract']))
    for path in protected:
        original=no_symlinks(path.absolute())
        require(not (root==original or root.is_relative_to(original) or original.is_relative_to(root)),
                'new sweep state overlaps read-only reference/assets')


def prepare_cpu(args):
    # Both explicit review values are required; no --development path.
    source = source_identity(reviewed_commit=args.reviewed_commit, reviewed_fingerprint=args.reviewed_fingerprint)
    p, contract, reference = historical_reference(args.frozen_plan, args.frozen_cpu)
    root = no_symlinks(Path(args.state).absolute())
    protect_reference_output(root,p,reference)
    chosen = select(args.cells)
    if not root.exists():
        root.mkdir(parents=True)
        write_once(root / 'definition.json', dict(schema=SCHEMA, source=source, reference=reference,
                    python=str(Path(sys.executable).absolute()), matrix=cells()))
        if contract is not None:
            write_once(root / 'continuous_contract.json', contract)
    d = definition(root)
    require(d['reference'] == reference, 'different historical reference')
    with control_lock(root):
        for cid in chosen:
            entry = next(c for c in cells() if c['id'] == cid)
            base = root / 'prepared' / cid
            require(not base.exists(), 'fresh cell required; no overwrite/retry: ' + cid)
            base.mkdir(parents=True)
            stage = 'prepare'
            try:
                argv = ['prepare','--original-experiment',p['original_experiment'],'--multi',p['multi'],
                        '--dataset',p['dataset'],'--count',str(p['count']),'--output-dir',base/'plan',
                        '--bandwidth-mbps',str(entry['performance_knobs']['bandwidth_mbps']),
                        '--ud-throughput-factor',str(entry['performance_knobs']['ud_throughput_factor']),
                        '--reviewed-commit',source['commit'],'--reviewed-fingerprint',source['content_fingerprint'],
                        '--workload-mode',p['workload_mode'],'--enable-continuous-replay',
                        '--impact-implementation',p['impact_implementation']]
                for key, flag in (('strict_pilot','--strict-pilot'),('baseline_plan','--baseline-plan'),
                                  ('ud_operation_seconds','--ud-operation-seconds')):
                    if p[key] is not None:
                        argv += [flag, str(p[key])]
                if contract is not None:
                    argv += ['--continuous-contract', root/'continuous_contract.json']
                official(argv)
                new = read_json(base/'plan/plan.json')
                same_workload(new, p)
                require(new['source'] == source and new['performance_knobs'] == entry['performance_knobs'], 'source/knob binding differs')
                stage = 'preflight'
                pre = official(['preflight','--experiment',base/'plan'])
                write_once(base/'preflight.json', pre)
                require(new['resources']['status'] == 'STATIC_BOUNDS_PASS_RUNTIME_CHECKS_PENDING',
                        'BLOCKED_INFEASIBLE: ' + str(new['resources']['reasons']))
                stage = 'cpu-validate'
                official(['cpu-validate','--experiment',base/'plan','--output-dir',base/'cpu'])
                stage = 'inspect'
                inspected = official(['inspect','--experiment',base/'plan','--cpu-audit',base/'cpu'])
                write_once(base/'inspect.json', inspected)
                write_once(base/'ready.json', dict(schema=SCHEMA, cell=cid, source=source,
                    plan_inventory=verify(base/'plan'), cpu_inventory=verify(base/'cpu'), performance_knobs=entry['performance_knobs']))
            except BaseException as exc:
                event(root,'PREPARE_FAILED',cell=cid,status='PARTIAL_BLOCKED',stage=stage,reason=str(exc))
                raise
    return dict(status='CPU_PREPARED_GPU_NOT_SUBMITTED', cells=chosen)


def registered(root, cid, d):
    from .r5_d5_lightperf import approval
    base = Path(root)/'prepared'/cid
    ready = read_json(base/'ready.json')
    require(ready['schema']==SCHEMA and ready['cell']==cid,'wrong ready record')
    p, pi, ai = approval(base/'plan', base/'cpu')
    require(ready['plan_inventory'] == pi and ready['cpu_inventory'] == ai and ready['source'] == d['source'] == p['source'],
            'cell approval/source/inventory differs')
    expected = next(c['performance_knobs'] for c in cells() if c['id'] == cid)
    require(p['performance_knobs'] == ready['performance_knobs'] == expected, 'wrong registered knob cell')
    frozen, _, reference = historical_reference(d['reference']['plan'], d['reference']['cpu_audit'])
    require(reference == d['reference'], 'reference evidence changed')
    same_workload(p, frozen)
    return p, base


def local_sbatch_manual():
    r = subprocess.run(['man','-P','cat','sbatch'], capture_output=True, text=True)
    require(r.returncode == 0, 'local sbatch manual unavailable; select one explicit approved node')
    return r.stdout


def auto_node_arguments(manual):
    normalized = re.sub(r'\s+', ' ', re.sub('.\x08', '', manual)).lower()
    require(MAN_NODELIST_TEXT in normalized, 'local Slurm does not document nodelist subset semantics; auto blocked')
    return ['--nodelist='+','.join(D5_NODES)]


def remaining_seconds(job_text, deadline, now=None):
    match = re.search(r'\bEndTime=([^\s]+)', job_text)
    require(match is not None, 'scontrol EndTime missing')
    end = datetime.fromisoformat(match.group(1)).timestamp()
    remaining = math.floor(end - (time.time() if now is None else now))
    require(0 < remaining <= 16200 and 0 < deadline <= 7200 and remaining > deadline + 120,
            'insufficient actual allocation remaining time; no timeout extension')
    return remaining


def sbatch_arguments(payload, node, dependency, log_dir):
    require(node == 'auto' or node in D5_NODES, 'unapproved requested node')
    nodes = auto_node_arguments(local_sbatch_manual()) if node == 'auto' else ['--nodelist='+node]
    argv = ['sbatch','--parsable','--hold','--no-requeue','--partition=batch_eebme_ugrad','--nodes=1-1','--ntasks=1',
            '--gres=gpu:1','--cpus-per-task=4','--mem=64G','--time=04:30:00',
            '--chdir='+str(ROOT),'--output='+str(log_dir/'%j.out'),'--error='+str(log_dir/'%j.err'), *nodes]
    if dependency is not None:
        require(re.fullmatch(r'[0-9]+', dependency) is not None, 'invalid predecessor job ID')
        argv += ['--dependency=afterok:'+dependency]
    return argv + [str(JOB),'_job','--payload',str(payload),'--payload-sha256',digest(read_json(payload))]


def submit(args):
    root = no_symlinks(Path(args.state).absolute()); d = definition(root)
    chosen = select(args.cells, submit=True)
    require(args.trial >= 0, 'nonnegative balanced trial index required')
    for path in (args.tmpdir, args.hf_home):
        require(path is not None and no_symlinks(Path(path).absolute()).is_dir(), 'existing offline TMPDIR/HF_HOME required')
    with control_lock(root):
        history = events(root)
        from .r5_d5_sweep_analysis import validated_cell
        completed = set()
        for e in history:
            if e['kind']=='CELL_COMPLETE' and e.get('status')=='MATCHED_COMPLETE':
                checked=validated_cell(e['attempt'],d['source'],d['reference']['tasks_sha256'])
                require(checked['gate']=='MATCHED_COMPLETE' and checked['cell']==e['cell'], 'canary completion evidence changed')
                completed.add(e['cell'])
        if not set(CANARIES) <= completed:
            require(set(chosen) <= set(CANARIES), 'both new-source canaries must pass before other cells')
            if CANARIES[1] in chosen:
                require(CANARIES[0] in completed or CANARIES[0] in chosen and
                        chosen.index(CANARIES[0]) < chosen.index(CANARIES[1]), '0.25 canary must precede 0.10')
        # Validate the entire selected subset BEFORE queuing its first job.
        for cid in chosen:
            registered(root, cid, d)
            require(not any(e['kind']=='SUBMITTED' and e['cell']==cid and e['trial']==args.trial for e in history),
                    'cell/trial already submitted; no retry')
        previous = [e for e in history if e['kind']=='SUBMITTED']
        last = max(previous, key=lambda e:e['ordinal'])['job_id'] if previous else None
        for cid in chosen:
            tag = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:12]
            attempt = root/'attempts'/tag
            attempt.mkdir(parents=True)
            base_order = next(c['order'] for c in cells() if c['id']==cid)
            shift = args.trial % 3
            order = base_order[shift:]+base_order[:shift]
            payload = attempt/'submission.json'
            write_once(payload,dict(schema=SCHEMA,state=str(root),cell=cid,trial=args.trial,order=order,
                requested_node=args.node,tmpdir=str(Path(args.tmpdir).absolute()),hf_home=str(Path(args.hf_home).absolute()),
                source=d['source'],python=d['python']))
            scheduler_env = {k:v for k,v in os.environ.items() if not k.startswith('SBATCH_') and k not in ('CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES')}
            scheduler_env['D5_SWEEP_PYTHON'] = sys.executable
            argv = sbatch_arguments(payload,args.node,last,attempt)
            stage = 'sbatch_hold'
            try:
                # --test-only checks scheduler feasibility without submitting or reserving a GPU.
                test = [x for x in argv if x not in ('--hold','--parsable')]
                test.insert(1,'--test-only')
                result = subprocess.run(test,env=scheduler_env,capture_output=True,text=True)
                require(result.returncode == 0, 'Slurm test-only rejected resource request: '+result.stderr[-2000:])
                result = subprocess.run(argv,env=scheduler_env,capture_output=True,text=True)
                require(result.returncode == 0, 'sbatch failed: '+result.stderr[-2000:])
                job = result.stdout.strip().split(';')[0]
                require(re.fullmatch(r'[0-9]+',job) is not None, 'unparseable sbatch job ID; no retry')
                event(root,'SUBMITTED',cell=cid,trial=args.trial,attempt=str(attempt),job_id=job,dependency=last,
                      ordinal=len(previous),payload_sha256=digest(read_json(payload)),order=order)
                previous.append(dict(job_id=job));last=job
                stage = 'release_registered_job'
                released = subprocess.run(['scontrol','release',job],capture_output=True,text=True)
                require(released.returncode == 0,'held job release failed: '+released.stderr[-2000:])
            except BaseException as exc:
                event(root,'SUBMISSION_FAILED',cell=cid,trial=args.trial,attempt=str(attempt),status='PARTIAL_BLOCKED',stage=stage,reason=str(exc))
                raise
    return dict(status='EXPLICIT_SUBSET_SUBMITTED', cells=chosen, last_job_id=last)


GPU_PROBE = '''import json,torch
from semcache.edgelora.r5_d5_lightperf_runtime import gpu_uuid_matches
assert torch.cuda.device_count()==1, 'exactly one allocated CUDA device required'
p=torch.cuda.get_device_properties(0)
u=str(p.uuid)
assert gpu_uuid_matches(u,u), 'invalid actual UUID'
assert torch.cuda.mem_get_info(0)[0]>=12*2**30, '12GiB GPU reserve required'
print(json.dumps(dict(uuid=u,name=p.name,total_bytes=p.total_memory)))
'''


def job(args):
    payload = no_symlinks(Path(args.payload).absolute()); data=read_json(payload)
    require(digest(data)==args.payload_sha256, 'submission payload changed')
    root=Path(data['state']); stage='allocation'; cid=data['cell']; attempt=payload.parent; trusted=False
    try:
        d=definition(root)
        require(data['source']==d['source'] and os.environ.get('SLURM_JOB_ID') and not os.environ.get('SLURM_ARRAY_TASK_ID') and
                int(os.environ.get('SLURM_RESTART_COUNT','0'))==0,
                'unbound/non-Slurm/array allocation')
        bound=[e for e in events(root) if e['kind']=='SUBMITTED' and e['attempt']==str(attempt)]
        require(len(bound)==1 and bound[0]['job_id']==os.environ['SLURM_JOB_ID'] and
                bound[0]['payload_sha256']==args.payload_sha256, 'job not registered by explicit submit')
        trusted=True
        node=os.environ.get('SLURMD_NODENAME')
        require(node in D5_NODES and int(os.environ.get('SLURM_JOB_NUM_NODES','0'))==1 and
                int(os.environ.get('SLURM_CPUS_PER_TASK','0'))==4 and
                int(os.environ.get('SLURM_MEM_PER_NODE','0'))>=65536, 'unapproved node or allocation')
        require(data['requested_node']=='auto' or node==data['requested_node'], 'allocated node differs')
        for value in (ROOT,root,attempt,Path(data['tmpdir']),Path(data['hf_home'])):
            require(no_symlinks(Path(value).absolute()).is_relative_to('/data/khuss'), 'job paths must remain under /data/khuss')
        require(Path(data['tmpdir']).is_dir() and Path(data['hf_home']).is_dir(), 'offline cache/temp absent')
        visible=os.environ.get('CUDA_VISIBLE_DEVICES');hip=os.environ.get('HIP_VISIBLE_DEVICES')
        require(visible is not None and visible!='', 'Slurm GPU visibility missing')
        os.environ.update(cpu_env())
        p,base=registered(root,cid,d)
        os.environ['CUDA_VISIBLE_DEVICES']=visible
        if hip is None:os.environ.pop('HIP_VISIBLE_DEVICES',None)
        else:os.environ['HIP_VISIBLE_DEVICES']=hip
        cpus=sorted(os.sched_getaffinity(0));require(len(cpus)>=4,'srun CPU affinity too small')
        pinned=cpus[:4];os.sched_setaffinity(0,pinned)
        environment=dict(os.environ,D5N_NODE=node,R5_ENABLE_GPU='1',R5_I5_ENABLE_GPU='1',D5L_ENABLE_GPU='1',
            HF_HOME=data['hf_home'],HF_HUB_CACHE=str(Path(data['hf_home'])/'hub'),TMPDIR=data['tmpdir'],
            TORCH_HOME=str(Path(data['tmpdir'])/'torch'),CUDA_CACHE_PATH=str(Path(data['tmpdir'])/'cuda_cache'),
            HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
            OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONHASHSEED='42')
        def budget():
            control=subprocess.run(['scontrol','show','job','--oneliner',os.environ['SLURM_JOB_ID']],capture_output=True,text=True)
            require(control.returncode==0,'scontrol EndTime query failed')
            require('Partition=batch_eebme_ugrad' in control.stdout.split() and
                    'JobId='+os.environ['SLURM_JOB_ID'] in control.stdout.split(),'scheduler job/partition differs')
            return remaining_seconds(control.stdout,p['resources']['worker_deadline_seconds']),control.stdout
        stage='remaining_before_probe';budget()
        stage='allocated_gpu_probe'
        probe=subprocess.run([sys.executable,'-c',GPU_PROBE],cwd=ROOT,
            env=dict(environment,PYTHONPATH=str(ROOT/'src')),capture_output=True,text=True)
        require(probe.returncode==0,'allocated PyTorch GPU probe failed: '+probe.stderr[-2000:])
        gpu=json.loads(probe.stdout)
        write_once(attempt/'allocation.json',dict(job_id=os.environ['SLURM_JOB_ID'],node=node,gpu=gpu,
                   affinity=pinned,order=data['order'],requested_node=data['requested_node']))
        registry=[]
        for condition in data['order']:
            stage='remaining_'+condition
            remaining,control=budget()
            require(sorted(os.sched_getaffinity(0))==pinned,'CPU affinity changed')
            write_once(attempt/(condition+'_budget.json'),dict(remaining_seconds=remaining,scontrol=control))
            output=attempt/condition
            require(not output.exists(),'fresh condition result required')
            environment['D5L_ALLOCATION_REMAINING_SECONDS']=str(remaining)
            stage='run_'+condition
            argv=['run','--experiment',base/'plan','--cpu-audit',base/'cpu','--condition',condition,
                  '--source-commit',d['source']['commit'],'--expected-gpu-uuid',gpu['uuid'],'--output-dir',output,
                  '--enable-gpu-experiment','--enable-extended-experiment','--enable-continuous-replay']
            if condition==CONDITIONS[2]:argv+=['--enable-approximate-experiment']
            official(argv,gpu=True,environment=environment)
            stage='verify_'+condition
            official(['verify','--experiment',base/'plan','--cpu-audit',base/'cpu','--run',output])
            record=read_json(output/'run.json')
            require(record['gpu']['uuid']==gpu['uuid'] or
                    record['gpu']['uuid'].removeprefix('GPU-').lower()==gpu['uuid'].removeprefix('GPU-').lower(),'actual run GPU differs')
            require(record['cpu_concurrency']['affinity']==pinned,'actual worker CPU affinity differs')
            registry.append(dict(run=str(output),plan=str(base/'plan'),cpu_audit=str(base/'cpu')))
        write_once(attempt/'registry.json',registry)
        stage='official_analyze'
        official(['analyze','--registry',attempt/'registry.json','--output-dir',attempt/'analysis'])
        from .r5_d5_sweep_analysis import validated_cell
        stage='matched_identity_and_exact_tokens'
        review=validated_cell(attempt)
        if review['gate']!='MATCHED_COMPLETE':
            reasons={c:{k:v.get(k) for k in ('status','mismatches','generated_token_agreements')}
                     for c,v in review.get('comparisons',{}).items()}
            raise ValueError('cell comparison gate failed: '+json.dumps(reasons,sort_keys=True))
        event(root,'CELL_COMPLETE',cell=cid,trial=data['trial'],attempt=str(attempt),status='MATCHED_COMPLETE',
              job_id=os.environ['SLURM_JOB_ID'],order=data['order'],gpu=gpu,affinity=pinned)
        return dict(status='MATCHED_COMPLETE',cell=cid,correctness_qualified_speedup='NOT_ESTABLISHED')
    except BaseException as exc:
        if trusted:
            event(root,'CELL_FAILED',cell=cid,attempt=str(attempt),status='PARTIAL_BLOCKED',stage=stage,reason=str(exc))
        raise ValueError(f'PARTIAL_BLOCKED stage={stage}: {exc}') from exc


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    for command in ('check','prepare-cpu','submit','status','aggregate','_job'):
        p=sub.add_parser(command)
        if command in ('check','prepare-cpu'):
            p.add_argument('--frozen-plan',type=Path,default=FROZEN_PLAN)
            p.add_argument('--frozen-cpu',type=Path,default=FROZEN_CPU)
        if command not in ('check','_job'):p.add_argument('--state',type=Path,required=True)
        if command in ('check','submit'):p.add_argument('--node',default='auto')
        if command=='prepare-cpu':
            p.add_argument('--cells',default='all');p.add_argument('--reviewed-commit',required=True);p.add_argument('--reviewed-fingerprint',required=True)
        if command=='submit':
            p.add_argument('--cells',required=True);p.add_argument('--trial',type=int,default=0)
            p.add_argument('--tmpdir',type=Path,default=os.environ.get('D5L_TMPDIR'))
            p.add_argument('--hf-home',type=Path,default=os.environ.get('HF_HOME'))
        if command=='aggregate':p.add_argument('--output-dir',type=Path,required=True)
        if command=='_job':p.add_argument('--payload',type=Path,required=True);p.add_argument('--payload-sha256',required=True)
    args=parser.parse_args(argv)
    try:
        if args.command!='_job':
            require(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and os.environ.get('HIP_VISIBLE_DEVICES')=='',
                    'CPU control commands require hidden CUDA/HIP; use the Bash entry point')
        if args.command=='check':result=check(args.frozen_plan,args.frozen_cpu,args.node)
        elif args.command=='prepare-cpu':result=prepare_cpu(args)
        elif args.command=='submit':result=submit(args)
        elif args.command=='_job':result=job(args)
        elif args.command=='status':
            result=dict(schema=SCHEMA,status='READ_ONLY',matrix=cells(),events=events(args.state),gpu='NOT_QUERIED')
        else:
            from .r5_d5_sweep_analysis import aggregate
            result=aggregate(args.state,args.output_dir)
        print(json.dumps(result,indent=2,allow_nan=False))
        return 2 if result.get('status') in ('PARTIAL_BLOCKED','BLOCKED_MISSING_REFERENCE') else 0
    except (ValueError,OSError,KeyError,TypeError) as exc:
        print(json.dumps(dict(status='PARTIAL_BLOCKED',stage=args.command,reason=str(exc),gpu_action='NO_RETRY')),file=sys.stderr)
        return 2
