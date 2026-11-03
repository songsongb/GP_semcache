"""One fresh diagnostic plan derived from immutable real tokenizer IDs, not tensors."""
from copy import deepcopy
from pathlib import Path

from .communication import digest,require
from .r3a_artifacts import read,write,sha
from .r4c_workload import policy
from .r5_i5_evidence import fresh,seal,verify
from .r5_i5_workload import resources,CONDITIONS
from .r5_i5r1_plan import source_identity
from .r5_i5r2_tensors import Evidence
from .r5_multi_artifacts import validate

VERSION='I5R2B_EARLIEST_ARITHMETIC_V1'


def prepare(historical_plan,multi_prepared,output,*,r2a_review=None,fixture=False,generation_tokens=2,capture_bytes=32*2**20,max_files=128):
    require(1<=generation_tokens<=2 and 2**20<=capture_bytes<=64*2**20 and 32<=max_files<=256,'bounded diagnostic configuration required')
    olde=Evidence(historical_plan,max_input_bytes=128*2**20);oldseal=olde.verify();old=olde.json('i5_workload_manifest.json')
    copy=dict(old);oid=copy.pop('experiment_id');require(digest(copy)==oid and len(old['blocks'])==1,'one sealed historical smoke block required')
    require(old['TEST_FIXTURE_ONLY']==fixture,'real/fixture plan differs')
    historical_review=dict(status='TEST_FIXTURE_ONLY')
    if not fixture:
        require(r2a_review is not None,'completed R2A historical review required before GPU preparation')
        re=Evidence(r2a_review,max_input_bytes=128*2**20);rs=re.verify('i5r2_output_manifest.json')
        provenance=re.json('i5r2_source_provenance.json')
        require(provenance['historical_source_verification']=='ALL_RECORDED_BLOBS_MATCH' and
                provenance['recorded_source_commit']==old['i5r1']['source']['commit'],'R2A source provenance differs/unavailable')
        historical_review=dict(status='VERIFIED_EXPORTED_R2A',manifest_sha256=rs['manifest_sha256'],path=str(Path(r2a_review).resolve()),
            observations=re.json('i5r2_hypothesis_matrix.json'),first_divergence=re.json('i5r2_first_divergence.json'))
    m,pm,adapters,registry=validate(multi_prepared);current=source_identity(fixture)
    require(m['stage']=='expanded' and len(registry['users'])==30 and m['TEST_FIXTURE_ONLY']==fixture,'expanded thirty-user preparation required')
    require(old['i5r1']['model_identity']==pm['identity'] and old['tokenizer_identity']==m['tokenizer'] and
        old['i5r1']['adapter_fingerprints']=={a:s['matrix_sha256'] for a,s in adapters.items()},'historical model/tokenizer/adapters changed')
    block=deepcopy(old['blocks'][0]);block.pop('block_id');tasks=block['tasks']
    require(len(tasks)==2 and [t['role'] for t in tasks]==['PRIMING_SOURCE','SCORED_TARGET'],'one standalone source-target pair required')
    s,t=tasks
    require(len(s['token_ids'])==64 and len(t['token_ids'])==256 and s['token_ids']==t['token_ids'][:64],'exact 64/256-token historical pair required')
    require(all(x['adapter_id']=='Adapter_A' and x['user_id']=='User_01' for x in tasks),'initial same-user Adapter_A scope required')
    require(block['dataset']=='qasper' and block['track']=='CONTROLLED' and s['scored'] is False and t['scored'] is True,'controlled Qasper roles differ')
    t['policy']=policy(old['special_tokens'],'quality',generation_tokens)
    for x in tasks:x['strict']=False
    block.update(i5r2b_diagnostic=True,resources=resources(tasks,capacity=block['resources']['per_adapter_capacity_bytes'],evidence_budget=512*2**20),
        condition_order=list(CONDITIONS))
    block['block_id']=digest(block)
    config=dict(version=VERSION,DIAGNOSTIC_ONLY=True,layer=0,token_interval=[0,64],capture_bytes=capture_bytes,max_files=max_files,target_generation_tokens=generation_tokens,
        source_request_id=s['request_id'],target_request_id=t['request_id'],source=current,TEST_FIXTURE_ONLY=fixture,
        historical_plan_manifest_sha256=oldseal['manifest_sha256'],historical_experiment_id=old['experiment_id'],r2a_review=historical_review,
        backend_policy='Observe only; no precision/backend override; not a performance benchmark',
        optional_shape_control='NOT_IMPLEMENTED_PENDING_FIRST_GPU_DIAGNOSTIC',same_shape_repeat='NOT_RUN',instrumentation_effect_control='NOT_RUN')
    plan=deepcopy(old);plan.pop('experiment_id');plan.update(blocks=[block],i5r2b=config)
    plan['i5r1']['source']=current;plan['i5r1']['generation_tokens']=generation_tokens
    plan['i5r1']['multi_prepared']=str(Path(multi_prepared).resolve());plan['i5r1']['multi_manifest_sha256']=sha(Path(multi_prepared)/'multi_prepare.json')
    plan['experiment_id']=digest(plan)
    out=fresh(output,(historical_plan,multi_prepared,r2a_review));write(out/'i5_workload_manifest.json',plan)
    write(out/'i5_dataset_manifest.json',olde.json('i5_dataset_manifest.json'));write(out/'i5r2b_protocol.json',config);seal(out)
    return plan


def validate_plan(path,multi_prepared=None):
    path=Path(path);verify(path);plan=read(path/'i5_workload_manifest.json');p=dict(plan);eid=p.pop('experiment_id');cfg=plan['i5r2b']
    require(digest(p)==eid and read(path/'i5r2b_protocol.json')==cfg and cfg['version']==VERSION and cfg['DIAGNOSTIC_ONLY'] is True,'diagnostic plan identity differs')
    require(cfg['source']==source_identity(plan['TEST_FIXTURE_ONLY']),'diagnostic pinned source changed')
    require(len(plan['blocks'])==1 and cfg['layer']==0 and cfg['token_interval']==[0,64] and
            2**20<=cfg['capture_bytes']<=64*2**20 and 32<=cfg['max_files']<=256,'unbounded diagnostic plan')
    multi_prepared=multi_prepared or plan['i5r1']['multi_prepared'];m,pm,adapters,registry=validate(multi_prepared)
    require(Path(multi_prepared).resolve()==Path(plan['i5r1']['multi_prepared']) and m['TEST_FIXTURE_ONLY']==plan['TEST_FIXTURE_ONLY'] and
        sha(Path(multi_prepared)/'multi_prepare.json')==plan['i5r1']['multi_manifest_sha256'] and m['stage']=='expanded' and len(registry['users'])==30 and
        pm['identity']==plan['i5r1']['model_identity'] and m['tokenizer']==plan['tokenizer_identity'] and
        {a:s['matrix_sha256'] for a,s in adapters.items()}==plan['i5r1']['adapter_fingerprints'],'diagnostic preparation identity differs')
    b=plan['blocks'][0];bc=dict(b);bid=bc.pop('block_id');s,t=b['tasks']
    require(digest(bc)==bid and len(s['token_ids'])==64 and len(t['token_ids'])==256 and s['token_ids']==t['token_ids'][:64], 'diagnostic prefix/order identity differs')
    require(s['request_id']==cfg['source_request_id'] and t['request_id']==cfg['target_request_id'] and not s['scored'] and t['scored'] and
        all(not x['strict'] and x['adapter_id']=='Adapter_A' and x['user_id']=='User_01' for x in (s,t)), 'diagnostic task scope differs')
    require(b['dataset']=='qasper' and [x['request_sequence_number'] for x in (s,t)]==[0,1] and b['condition_order']==list(CONDITIONS),'diagnostic dataset/order differs')
    require(s['policy']['max_new_tokens']==1 and 1<=t['policy']['max_new_tokens']==cfg['target_generation_tokens']<=2 and len(t['token_ids'])+t['policy']['max_new_tokens']<=plan['context_limit']<=2048,
            'diagnostic generation/context bound differs')
    return plan
