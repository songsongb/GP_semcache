"""Native L6-B plan binding, also used by the isolated UD and offline verifier."""
import sys
from pathlib import Path

from . import r5_d5_l6a as l6
from .communication import require, digest
from .r5_d5_lightperf_io import read_json, verify, no_symlinks, NAMESPACE
from .r5_d5_lightperf_workload import CONDITIONS
from .r5_d5_lightperf_runtime import native_versions
from .r5_d5_sweep import historical_reference

NS = NAMESPACE / 'l6b'
L6A_HEAD = '044f22542a9983a8ad73e1f05913b1b93d7ca1ad'


def native_plan(plan, audit):
    plan, audit = no_symlinks(plan), no_symlinks(audit)
    require(plan.parent == audit.parent == l6.NS, 'L6-A inputs outside namespace')
    pi, ai = verify(plan), verify(audit)
    a = read_json(plan/'plan.json')
    cpu = read_json(audit/'cpu_validation.json')
    require(a['schema']=='D5_L6A_CPU_V2' and a['entry_limit']==64 and a['count']==50 and
            a['capacity_bytes_per_adapter']==512*2**20 and a['phase']=='LONG_REPLAY_CPU' and
            a['source']['approval_scope']=='CLEAN_REVIEWED_SOURCE' and a['source']['commit']!=L6A_HEAD and
            digest({k:v for k,v in a.items() if k!='plan_id'})==a['plan_id'], 'new reviewed L6-A V2 entry64 required')
    require(cpu['plan_inventory']==pi and cpu['plan_id']==a['plan_id'] and cpu['source']==a['source'] and
            cpu['modules']==l6.modules(a) and cpu['versions']==native_versions() and cpu['tests']>0 and
            cpu['failed']==cpu['errors']==cpu['skipped']==0 and cpu['status']=='CPU_APPROVED','L6-A CPU approval differs')
    frozen, _, ref = historical_reference(l6.REFERENCE/'plan/plan.json',l6.REFERENCE/'cpu')
    require(ref==a['reference'] and frozen['source']['commit']==l6.BASE,'frozen reference differs')
    l6.check_stream(a['stream'],frozen['stream'])
    require(a['transport']['rows'][:20]==frozen['transport']['rows'] and len(a['transport']['rows'])==50 and
            a['stream']['statistics']['per_adapter_requests']=={'Adapter_A':20,'Adapter_C':30} and
            a['cycles']==l6.CYCLES and a['allocations']==[dict(condition=c,worker_deadline_seconds=9000,
            requests=50,reset='EMPTY_ONCE_PER_CONDITION') for c in CONDITIONS],'L6 allocation/workload differs')
    resources=a['resources']
    from .r5_d5_lightperf_workload import preflight
    from .r5_d5_impact import resource_bounds
    expected=l6.long_deadline(resource_bounds(preflight(a['stream']['tasks'],frozen['performance_knobs'],
        ud_operation_seconds=frozen['ud_operation_seconds']),frozen['impact_implementation']))
    require(resources==expected and resources['status']=='STATIC_BOUNDS_PASS_RUNTIME_CHECKS_PENDING' and
            resources['forecast_seconds']==7824.54483968,'L6 forecast or resources differ')
    variable={'source','stream','count','resources','transport','plan_id'}
    require(all(a[k]==frozen[k] for k in (a.keys() & frozen.keys()) - variable - {'schema'}),'frozen Light fields differ')
    p=dict(frozen, **{k:a[k] for k in variable-{'plan_id'}})
    p.update(l6_plan_id=a['plan_id'],entry_limit=64,l6_binding=dict(python=sys.executable,plan=str(plan),audit=str(audit),
             plan_inventory=pi,audit_inventory=ai))
    del p['plan_id']
    p['plan_id']=digest(p)
    return p


def validate_binding(p):
    link=p.get('l6_binding')
    require(isinstance(link,dict) and native_plan(Path(link['plan']),Path(link['audit']))==p,
            'native L6-B plan binding differs')
