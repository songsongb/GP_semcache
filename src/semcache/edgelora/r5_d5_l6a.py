"""L6-A CPU plans. These records have no Light GPU authorization scope."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

from .communication import canonical, digest, require
from .r3a_artifacts import ROOT, THRESHOLDS
from .r5_d5_impact import resource_bounds
from .r5_d5_lightperf import material, approval_modules
from .r5_d5_lightperf_runtime import native_versions
from .r5_d5_lightperf_io import MAX_JSON, SummaryWriter, fresh, no_symlinks, read_json, verify, source_identity
from .r5_d5_lightperf_replay import validate_replay_stream
from .r5_d5_sweep import historical_reference, protect_reference_output

BASE = '40717c1f92cb99ea51dd180e58320441a99c1acb'
BRANCH = 'exp/d5-l1-l5-lightperf'
NS = ROOT / 'results/r5/d5_l6a'
REFERENCE = ROOT / 'results/r5/d5_lightperf/sweep_v5_20261122T141858Z/prepared/bw050_ud025'
CYCLES = [dict(cycle=i+1, first=i*10, last=i*10+9) for i in range(5)]


def cpu_guard():
    import torch
    require(all(os.environ.get(k) == v for k, v in dict(CUDA_VISIBLE_DEVICES='', HIP_VISIBLE_DEVICES='',
            HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1').items())
            and not torch.cuda.is_initialized(), 'offline CPU environment required')


def source(commit, fingerprint):
    require(bool(commit) == bool(fingerprint), 'paired L6 review values required')
    subprocess.run(['git','merge-base','--is-ancestor',BASE,'HEAD'], cwd=ROOT, check=True)
    return source_identity(expected_branch=BRANCH, development=not commit, reviewed_commit=commit,
                           reviewed_fingerprint=fingerprint)


def check_stream(stream, previous):
    validate_replay_stream(previous)
    validate_replay_stream(stream)
    require(previous['request_count'] == 20 and stream['request_count'] == 50 and
            stream['dataset'] == previous['dataset'] == '2wikimqa' and
            stream['seed'] == previous['seed'] and len(stream['source_pool']) == 10 and
            stream['source_pool'] == previous['source_pool'] and
            all(stream[k][:20] == previous[k] for k in ('tasks','occurrences')) and
            all(t['policy']['max_new_tokens'] == 8 for t in stream['tasks']), 'long replay prefix/pool differs')


def build(reference, identity, limit):
    cpu_guard()
    require(type(limit) is int and limit in (64,128), 'invalid entry_limit')
    frozen, contract, ref = historical_reference(reference/'plan/plan.json', reference/'cpu')
    require(frozen['source']['commit'] == BASE and frozen['thresholds'] == THRESHOLDS,
            'reviewed v5 source/thresholds differ')
    fields = material(frozen['original_experiment'], frozen['multi'], '2wikimqa', 50, frozen['performance_knobs'],
                      continuous_contract=contract, strict_pilot=frozen['strict_pilot'],
                      ud_operation_seconds=frozen['ud_operation_seconds'], workload_mode='CONTINUOUS_REPLAY',
                      enable_continuous_replay=True, baseline_plan=frozen['baseline_plan'])
    check_stream(fields['stream'], frozen['stream'])
    require(all(value == frozen[key] for key,value in fields.items() if key not in ('stream','transport','resources')),
            'frozen assets or model identity differs')
    transport = dict(fields['transport'], rows=fields['transport']['rows'][:20])
    require(transport == frozen['transport'], 'frozen transport prefix differs')
    fields = dict(fields, resources=resource_bounds(fields['resources'], frozen['impact_implementation']))
    p = dict(schema='D5_L6A_CPU_V1', source=identity, reference=ref, count=50, dataset='2wikimqa',
             entry_limit=limit, cycles=CYCLES, phase='CPU_STRUCTURAL_ONLY' if limit==128 else 'LONG_REPLAY_CPU',
             reviewed_lineage_base=BASE, capacity_bytes_per_adapter=512*2**20,
             gpu_status='NOT_GPU_APPROVED', **fields)
    for key in ('impact_implementation','performance_knobs','conditions','thresholds','validation_mode',
                'warmup_requests','enable_continuous_replay','workload_mode','ud_operation_seconds'):
        p[key] = frozen[key]
    p['plan_id'] = digest(p)
    require(len(canonical(p)) <= MAX_JSON, 'BLOCKED_INFEASIBLE: PLAN_JSON_BOUND')
    cpu_guard()
    return p


def output(path, protected):
    path = no_symlinks(path)
    require(path.parent == NS, 'use a new direct child of the L6 namespace')
    return SummaryWriter(fresh(path, protected, namespace=NS))


def checked(path):
    inventory = verify(path)
    p = read_json(path/'plan.json')
    ref = Path(p['reference']['plan']).parent.parent
    s = p['source']
    reviewed = s['approval_scope'] == 'CLEAN_REVIEWED_SOURCE'
    current = source(s['commit'] if reviewed else None, s['content_fingerprint'] if reviewed else None)
    require(build(ref, current, p['entry_limit']) == p, 'L6 source/reference/plan differs')
    return p, inventory


def execute(a):
    cpu_guard()
    if a.command in ('check','prepare-cpu','preflight'):
        identity = source(a.reviewed_commit, a.reviewed_fingerprint)
        p = build(a.reference, identity, a.entry_limit)
        if a.command == 'prepare-cpu':
            protect_reference_output(no_symlinks(a.output), read_json(Path(p['reference']['plan'])), p['reference'])
            w = output(a.output,[a.reference])
            w.write('plan.json',p)
            w.seal()
        return dict(status=p['resources']['status'], plan_id=p['plan_id'], resources=p['resources'],
                    entry_limit=p['entry_limit'], source_scope=identity['approval_scope'])
    require(no_symlinks(a.plan).parent == NS, 'L6 plan outside L6 namespace')
    p, inventory = checked(a.plan)
    if a.command == 'inspect':
        status = 'NO_CPU_AUDIT'
        if a.audit:
            require(no_symlinks(a.audit).parent == NS, 'L6 audit outside L6 namespace')
            verify(a.audit)
            record = read_json(a.audit/'cpu_validation.json')
            require(record['plan_inventory']==inventory and record['plan_id']==p['plan_id'] and
                    record['source']==p['source'] and record['tests']>0 and record['modules']==modules(p)
                    and record['failed']==record['errors']==record['skipped']==0 and
                    record['status']==audit_status(p), 'L6 audit binding differs')
            status = record['status']
        return dict(plan_id=p['plan_id'], status=status, resources=p['resources'], gpu='NOT_GPU_APPROVED')
    if a.command == 'analyze':
        from .r5_d5_l6a_analysis import analyze
        return analyze(p,a)
    require(p['resources']['status']=='STATIC_BOUNDS_PASS_RUNTIME_CHECKS_PENDING','BLOCKED_INFEASIBLE')
    sys.path.insert(0,str(ROOT/'tests'))
    suite = unittest.defaultTestLoader.loadTestsFromNames(modules(p))
    result = unittest.TextTestRunner().run(suite)
    require(result.wasSuccessful() and not result.skipped and result.testsRun, 'L6 CPU validation failed/skipped')
    cpu_guard()
    require(checked(a.plan)==(p,inventory),'L6 inputs changed during validation')
    record=dict(status=audit_status(p),source=p['source'], plan_id=p['plan_id'],plan_inventory=inventory,
                modules=modules(p),versions=native_versions(),tests=result.testsRun, failed=0, errors=0, skipped=0, gpu='NOT_GPU_APPROVED')
    w=output(a.output,[a.plan])
    w.write('cpu_validation.json',record)
    w.seal()
    return record


def audit_status(p):
    return 'CPU_APPROVED' if p['source']['approval_scope']=='CLEAN_REVIEWED_SOURCE' else 'DEVELOPMENT_TESTS_PASS'


def modules(p):
    return approval_modules(p)+['test_r5_d5_l6a']


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    for name in ('check','prepare-cpu','preflight','cpu-validate','inspect','analyze'):
        cmd=sub.add_parser(name)
        if name in ('check','prepare-cpu','preflight'):
            cmd.add_argument('--reference',type=Path,default=REFERENCE)
            cmd.add_argument('--entry-limit',type=int,choices=(64,128),default=64)
            cmd.add_argument('--reviewed-commit')
            cmd.add_argument('--reviewed-fingerprint')
        else:
            cmd.add_argument('--plan',type=Path,required=True)
        if name in ('prepare-cpu','cpu-validate','analyze'):
            cmd.add_argument('--output',type=Path,required=True)
        if name=='inspect':
            cmd.add_argument('--audit',type=Path)
        if name=='analyze':
            for field in ('light-plan','light-audit','full','exact','approx'):
                cmd.add_argument('--'+field,type=Path,required=True)
    try:
        result=execute(parser.parse_args(argv))
        cpu_guard()
        print(json.dumps(result,sort_keys=True,allow_nan=False))
        return 2 if result.get('status')=='BLOCKED_INFEASIBLE' else 0
    except (ValueError,KeyError,OSError,ImportError,subprocess.CalledProcessError) as exc:
        reason = str(exc)
        status = next((s for s in ('BLOCKED_INFEASIBLE','BLOCKED_MISSING_REFERENCE') if s in reason), 'BLOCKED')
        print(json.dumps(dict(status=status,reason=reason,gpu='GPU_NOT_RUN')),file=sys.stderr)
        return 2
