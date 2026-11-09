"""Explicit bounded A/EXACT/APPROX experiment; existing D5 operator only."""
from copy import deepcopy
from pathlib import Path
import json,os,sys,time,shutil,unittest
from .communication import require,digest
from .r3a_artifacts import ROOT,THRESHOLDS
from .r5_i5r1_plan import source_identity
from .r5_i5r2_tensors import Evidence,file_sha
from .r5_subsequence_natural import inputs,resources,transport,check_block
from .r5_subsequence_natural_io import output,put,seal,verify
from .r5_subsequence_physical import EXACT,APPROX
from .r5_subsequence_integrity import SCOPED

SCHEMA='D5_NATURAL_APPROX_PILOT_V1'
CONDITIONS=('FULL_RECOMPUTE','SEMCACHE_D5_EXACT','SEMCACHE_D5_APPROX')
SCRIPT=ROOT/'scripts/157_r5_natural_approx.py'
FILES=('scripts/157_r5_natural_approx.py','jobs/r5_natural_approx.sh','tests/test_r5_natural_approx.py')


def implementation():
    src=source_identity()
    # source_identity pins all execution modules; also require the new entry
    # points/tests/job in the reviewed commit, never just a historical SHA.
    import subprocess
    for name in FILES:
        r=subprocess.run(['git','show',src['commit']+':'+name],cwd=ROOT,capture_output=True)
        require(r.returncode==0 and __import__('hashlib').sha256(r.stdout).hexdigest()==file_sha(ROOT/name),'commit reviewed APPROX CLI/job/tests before prepare')
    return dict(source=src,files={name:file_sha(ROOT/name) for name in FILES})


def execution(condition):
    require(condition in CONDITIONS,'unknown approximate pilot condition')
    return (APPROX,True) if condition==CONDITIONS[2] else (EXACT,False)


def ordering(track):
    return 'original Block0 five tasks; original routing/order; no priming' if track=='NATURAL' else 'CONTROLLED standalone token-slice source, then original target; both costs included; not NATURAL'


def track_tasks(workload,track,tokenizer):
    require(track in ('NATURAL','CONTROLLED'),'explicit bounded track required')
    block=workload['blocks'][0];original=check_block(block)
    require(block['dataset']=='qasper' and len(original)==5,'first original five-request Qasper block required; no alternative selection')
    if track=='NATURAL':return deepcopy(block),dict(original_block_index=0,original_block_id=block['block_id'],priming_requests=0)
    target=deepcopy(original[0]);require(len(target['token_ids'])>=35,'controlled fixed source slice [32:35] unavailable; no replacement')
    source=deepcopy(target);ids=target['token_ids'][32:35]
    require(ids!=target['token_ids'][:3],'controlled fixed slice is also a prefix; no replacement or engineered fallback')
    require(target['request_id']!='CTRL_SOURCE_000','controlled source identifier collides with original target')
    from .r4c_workload import policy
    source.update(request_id='CTRL_SOURCE_000',token_ids=list(ids),
        input_text=tokenizer.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False),
        request_sequence_number=0,role='CONTROLLED_SOURCE',scored=False,policy=policy(workload['special_tokens'],'fixed',1))
    target['request_sequence_number']=1
    controlled=dict(dataset=block['dataset'],track='CONTROLLED',tasks=[source,target])
    return controlled,dict(original_block_index=0,original_block_id=block['block_id'],
        original_target_sha256=digest(original[0]),source_slice=[32,35],source_is_standalone=True,
        classification='CONTROLLED_NONPREFIX_SLICE_NOT_NATURAL',priming_requests=1)


def preflight(tasks,network=None):
    r=resources(tasks)  # Preserve every original memory/evidence guard.
    r=dict(r);r['estimated_run_file_count']=3*(r['estimated_run_file_count']-2048)//2+3072
    # B and C can persist one authenticated consumed tensor for each of 64
    # selected spans, 32 layers and three roles in every prefill. Do not assume
    # cold misses when bounding file count; byte forecast already bounds dense
    # full-row evidence, which dominates mixed fresh/cached tensor bytes.
    r['estimated_run_file_count']+=2*len(tasks)*64*32*3
    r['run_evidence_forecast_bytes']=3*r['condition_evidence_forecast_bytes']
    ud={}
    for t in tasks:
        # Actual dense UD persistence: one FP16 hidden, three FP32 deltas.
        aid=t['adapter_id'];rows=len(t['token_ids'])+t['policy']['max_new_tokens']-1
        ud[aid]=ud.get(aid,0)+rows*32*2560*14
    r['ud_dense_payload_forecast_bytes']=ud
    require(r['estimated_run_file_count']<=500000 and r['run_evidence_forecast_bytes']<64*2**30,'three-condition evidence/file guard failed; no automatic shrink or limit increase')
    require(all(v<4*2**30 for v in ud.values()),'UD payload alone exceeds unchanged4GiB bound')
    if network is not None:
        headers={row['request_id']:row['maximum_header_bytes'] for row in network['rows']}
        full=dict(ud)
        for t in tasks:
            # Two wire records and one UD layer record per exchange, with a
            # fourth header allowance, 8KiB framing/record margin, and controls.
            full[t['adapter_id']]+=t['policy']['max_new_tokens']*32*4*(headers[t['request_id']]+8192)+16*2**20
        require(all(v<4*2**30 for v in full.values()),'UD strict payload/ledger forecast exceeds unchanged4GiB bound')
        r['ud_payload_and_ledger_forecast_bytes']=full
    r['planning_seconds_per_forward']=5.
    r['planning_three_condition_seconds']=3*5*sum(t['policy']['max_new_tokens'] for t in tasks)
    r['planning_scope']='unmeasured I5 5s/forward assumption; excludes initialization/strict evidence overhead'
    r['deadline_scope']='shared14460s total guard; each worker <=7200s and remaining shared time'
    return r


def protected(p):return (p['multi'],p['original_experiment'],*p['dataset_paths'].values())


def prepare(original_experiment,multi,source_commit,track,output_dir,enable_approximate_experiment=False):
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU preparation requires hidden CUDA')
    require(enable_approximate_experiment,'explicit approximate experiment opt-in required')
    impl=implementation();require(source_commit==impl['source']['commit'],'reviewed source commit differs')
    w,d,receipt,inventories,m,pm,specs,users,tok=inputs(original_experiment,multi)
    block,provenance=track_tasks(w,track,tok.tokenizer)
    network=transport(block['tasks'],pm,specs,tok.identity,users)
    plan=dict(schema=SCHEMA,implementation=impl,original_experiment=str(Path(original_experiment).resolve()),
        multi=str(Path(multi).resolve()),base_prepared=m['base_prepared'],tokenizer_identity=tok.identity,
        original_inventory=receipt,dataset_inventories=inventories,multi_inventory=Evidence(multi).verify('r5_output_manifest.json'),
        dataset_paths={name:v['prepared_path'] for name,v in d.items()},track=track,provenance=provenance,
        tasks=block['tasks'],tasks_sha256=digest(block['tasks']),conditions=list(CONDITIONS),condition_order=list(CONDITIONS),
        execution={c:dict(mode=execution(c)[0],allow_approximate=execution(c)[1]) for c in CONDITIONS},
        enable_approximate_experiment=True,validation_mode=SCOPED,thresholds=THRESHOLDS,warmup_requests=0,ordering=ordering(track),
        resources=preflight(block['tasks'],network),transport=network,
        expansion='NO_AUTOMATIC_EXPANSION; one bounded track per run; separate approval required')
    plan['plan_id']=digest(plan)
    out=output(output_dir,(original_experiment,multi,*plan['dataset_paths'].values()));put(out/'approx_plan.json',plan);seal(out)
    return dict(status='CPU_PREPARED_GPU_NOT_RUN',track=track,plan_id=plan['plan_id'],resources=plan['resources'])


def load(experiment):
    require(not Path(experiment).is_symlink(),'plan root symlink forbidden');experiment=Path(experiment).resolve()
    receipt=verify(experiment);p=json.loads((experiment/'approx_plan.json').read_bytes());body=dict(p);pid=body.pop('plan_id')
    require(p['schema']==SCHEMA and digest(body)==pid and p['implementation']==implementation(),'pinned approximate plan/source differs')
    require(p['conditions']==p['condition_order']==list(CONDITIONS) and p['enable_approximate_experiment'] is True and
        p['execution']=={c:dict(mode=execution(c)[0],allow_approximate=execution(c)[1]) for c in CONDITIONS} and
        p['validation_mode']==SCOPED and p['thresholds']==THRESHOLDS and p['warmup_requests']==0 and p['ordering']==ordering(p['track']) and
        p['expansion']=='NO_AUTOMATIC_EXPANSION; one bounded track per run; separate approval required','approximate safety/scope differs')
    w,d,original,inv,m,pm,specs,users,tok=inputs(p['original_experiment'],p['multi'])
    block,provenance=track_tasks(w,p['track'],tok.tokenizer)
    network=transport(block['tasks'],pm,specs,tok.identity,users)
    require(original==p['original_inventory'] and inv==p['dataset_inventories'] and
        Evidence(p['multi']).verify('r5_output_manifest.json')==p['multi_inventory'] and
        p['dataset_paths']=={name:v['prepared_path'] for name,v in d.items()},'original input inventories differ')
    require(p['provenance']==provenance and p['tasks']==block['tasks'] and p['tasks_sha256']==digest(block['tasks']) and
        p['base_prepared']==m['base_prepared'] and p['tokenizer_identity']==tok.identity and
        p['resources']==preflight(block['tasks'],network) and p['transport']==network,'bounded workload/resource reconstruction differs')
    return p,receipt,[block,m,pm,specs,users,tok]


def cpu_validate(experiment,output_dir):
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','hide CUDA for CPU validation')
    p,receipt,_=load(experiment)
    import torch,transformers,numpy
    require(not torch.cuda.is_initialized(),'CPU validation initialized CUDA')
    sys.path.insert(0,str(ROOT/'tests'))
    suite=unittest.defaultTestLoader.loadTestsFromNames(['test_r5_natural_approx','test_r5_subsequence_generation',
        'test_r5_subsequence_validation','test_r5_subsequence_audit','test_r5_sparse_physical'])
    result=unittest.TestResult();suite.run(result)
    require(not result.errors and not result.failures and not result.skipped and not torch.cuda.is_initialized(),
        'native CPU approval failed/skipped: '+str(result.errors+result.failures+result.skipped))
    from .r5_i5_quality import validate_scorers
    validate_scorers({t['dataset'] for t in p['tasks'] if t['scored']})
    out=output(output_dir,(experiment,*protected(p)));record=dict(status='PASS',implementation=p['implementation'],plan_inventory=receipt,
        tests=result.testsRun,failed=0,skipped=0);put(out/'cpu_validation.json',record);seal(out);return record


def approval(experiment,cpu_audit):
    require(not Path(cpu_audit).is_symlink(),'CPU audit root symlink forbidden')
    p,receipt,rest=load(experiment);ar=verify(Path(cpu_audit).resolve());a=json.loads((Path(cpu_audit)/'cpu_validation.json').read_bytes())
    require(a['status']=='PASS' and a['failed']==a['skipped']==0 and a['implementation']==p['implementation'] and a['plan_inventory']==receipt,'CPU approval differs')
    return p,receipt,rest,ar


def inspect(experiment,cpu_audit):
    p,_,_,_=approval(experiment,cpu_audit)
    return dict(status='READY_FOR_MANUAL_BOUNDED_GPU_PILOT_NOT_RUN',track=p['track'],plan_id=p['plan_id'],resources=p['resources'])


class Profile:
    conditions=CONDITIONS;script=SCRIPT
    approval=staticmethod(approval);execution=staticmethod(execution)
    @staticmethod
    def check_run_budget(root,r):
        files=[p for p in Path(root).rglob('*') if p.is_file()]
        require(len(files)<=500000 and sum(p.stat().st_size for p in files)<r['run_limit_bytes'],'shared three-condition run budget exceeded')


def ud(config,output_dir):
    from .r5_subsequence_natural_runtime import ud as native_ud
    return native_ud(config,output_dir,_implementation=implementation)


def worker(experiment,output_dir,condition,cpu_audit,expected_gpu_uuid=None):
    from .r5_subsequence_natural_runtime import worker as native_worker
    return native_worker(experiment,output_dir,condition,cpu_audit,expected_gpu_uuid,_profile=Profile())


def run(experiment,output_dir,source_commit,cpu_audit,enable_gpu_experiment=False,enable_approximate_experiment=False):
    require(enable_gpu_experiment and enable_approximate_experiment,'explicit GPU and APPROX opt-ins required')
    require(os.environ.get('SLURM_JOB_ID') and not os.environ.get('SLURM_ARRAY_TASK_ID') and
        os.environ.get('R5_ENABLE_GPU')==os.environ.get('R5_I5_ENABLE_GPU')=='1','manual non-array opted-in allocation required')
    p,receipt,_,ar=approval(experiment,cpu_audit);require(source_commit==p['implementation']['source']['commit'],'reviewed commit differs')
    require(shutil.disk_usage(Path(output_dir).resolve().parent).free>=p['resources']['required_free_disk_bytes'],'unchanged96GiB free disk required')
    from .r5_subsequence_pilot import wait_worker
    out=output(output_dir,(experiment,cpu_audit,*protected(p)));begin=time.monotonic();gpu=None
    status=dict(status='FAILED',source=p['implementation']['source'],plan_inventory=receipt,cpu_audit_inventory=ar,
        cpu_audit_path=str(Path(cpu_audit).resolve()),children=[])
    try:
        for c in CONDITIONS:
            remaining=p['resources']['total_run_timeout_seconds']-(time.monotonic()-begin)
            require(remaining>0,'shared run deadline exhausted; no limit extension')
            argv=[sys.executable,str(SCRIPT),'_worker','--experiment',str(Path(experiment).resolve()),'--cpu-audit',str(Path(cpu_audit).resolve()),
                '--condition',c,'--output-dir',str(out/c)]
            if gpu:argv+=['--expected-gpu-uuid',gpu]
            start=time.monotonic_ns();rc=wait_worker(argv,timeout=min(remaining,p['resources']['worker_timeout_seconds']))
            status['children'].append(dict(condition=c,returncode=rc,process_lifecycle_ms=(time.monotonic_ns()-start)/1e6))
            require(rc==0,'condition failed; no retry/fallback');verify(out/c);Profile.check_run_budget(out,p['resources'])
            gpu=json.loads((out/c/'identity.json').read_bytes())['gpu']['uuid']
        status['status']='COMPLETE'
    except BaseException as exc:status['reason']=repr(exc);raise
    finally:put(out/'status.json',status);seal(out)
    return status
