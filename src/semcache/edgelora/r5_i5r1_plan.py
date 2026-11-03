"""Bounded controlled I5 plans. No inference, tensor creation, download or submission."""
from copy import deepcopy
from pathlib import Path
import subprocess
import tempfile

from .communication import digest, require
from .r3a_artifacts import ROOT, read, write, sha
from .r5_i5_evidence import fresh, seal, verify, sources, verify_sources
from .r5_i5_workload import prepare_experiment, resources, CONDITIONS, ORDERS
from .r5_multi_artifacts import validate, sharing

SCHEMA='I5R1_CONTROLLED_PHYSICAL_REUSE_V1'
DATASETS=('qasper','2wikimqa','multi_news')
CASES={'64:256':(64,256),'128:256':(128,256),'128:512':(128,512),'256:512':(256,512)}
PROFILES={
    'SMOKE':dict(datasets=('qasper',),cases=('64:256',),groups=1,adapters=('Adapter_A',),repeats=(1,)),
    'COMPARISON':dict(datasets=('qasper',),cases=tuple(CASES),groups=3,adapters=('Adapter_A','Adapter_B','Adapter_C'),repeats=(1,)),
    'DATASETS':dict(datasets=DATASETS,cases=('64:256','128:512'),groups=3,adapters=('Adapter_A','Adapter_B','Adapter_C'),repeats=(1,)),
    'AMORTIZATION':dict(datasets=('qasper',),cases=('128:512',),groups=3,adapters=('Adapter_A','Adapter_B','Adapter_C'),repeats=(1,2,4)),
}


def source_identity(fixture=False):
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()
    require(branch=='exp/raw-semcache-r5-integration','required branch mismatch')
    if not fixture:
        require(not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True).strip(),
                'commit reviewed tracked changes before preparation')
        tracked=set(subprocess.check_output(['git','ls-files'],cwd=ROOT,text=True).splitlines())
        require(set(sources())<=tracked,'commit all I5/I5R execution sources before preparation')
    return dict(commit=head,branch=branch,files=sources())


def r0_receipt(path=None):
    if path is None:
        return dict(status='USER_VERIFIED_HANDOFF_NOT_LOCALLY_REAUDITED',location='results/r5/i5r/diagnosis/raw_review_01',
                    implication='Equal 512-token non-prefix sources did not authorize any physical HIT; design uses genuine standalone shorter sources')
    path=Path(path)
    manifest=read(path/'i5r0_output_manifest.json')['files']
    require(set(manifest)=={str(p.relative_to(path)) for p in path.rglob('*') if p.is_file() and p.name!='i5r0_output_manifest.json'},'R0 inventory differs')
    require(not any(p.is_symlink() for p in path.rglob('*')),'R0 evidence symlink forbidden')
    require(all(sha(path/n)==h for n,h in manifest.items()),'R0 evidence changed')
    coverage=read(path/'i5r0_evidence_coverage.json');prov=read(path/'i5r0_source_provenance.json')
    require(not coverage['partial_observability'] and all(b['verification']=='ALL_RECORDED_BLOBS_MATCH' for b in prov['blocks']),
            'requested completed R0 diagnostic is not fully verified')
    return dict(status='COMPLETED_R0_INVENTORY_VERIFIED',path=str(path.resolve()),manifest_sha256=sha(path/'i5r0_output_manifest.json'),
                observations=read(path/'i5r0_workload_summary.json')['historical_handoff_comparison']['recalculated'])


def pair_checks(block, registry, context_limit):
    """Static observations only; no HIT flag reaches the controller."""
    tasks=block['tasks'];require(len({t['request_id'] for t in tasks})==len(tasks),'duplicate controlled request')
    require([t['request_sequence_number'] for t in tasks]==list(range(len(tasks))),'source/target order differs')
    require(len(tasks)<=5 and block['track']=='CONTROLLED' and all(t['strict'] for t in tasks),'bounded all-strict controlled block required')
    require(tasks[0]['role']=='PRIMING_SOURCE' and not tasks[0]['scored'],'standalone priming source must execute first')
    require(sum(not t['scored'] for t in tasks)==1,'one genuine standalone source per R1 block')
    source=tasks[0];p=len(source['token_ids']);checks=[]
    require(source['token_ids']==source['prepared']['token_ids'][:p],'source includes suffix or differs from prepared token slice')
    require(source['policy']['max_new_tokens']==1,'priming generates one real token')
    for t in tasks:
        require(t['policy']['max_new_tokens']+len(t['token_ids'])<=context_limit<=2048,'OPT position/generation budget exceeded')
        require(t['policy']['max_new_tokens']<=16,'R1 conservative output budget exceeded')
    for target in tasks[1:]:
        require(target['scored'] and target['role']=='SCORED_TARGET','unexpected controlled task role')
        require(target['token_ids']==target['prepared']['token_ids'],'target token identity differs')
        require(0<p<len(target['token_ids']) and source['token_ids']==target['token_ids'][:p],'source is not an exact strictly shorter full prefix')
        require(source['adapter_id']==target['adapter_id'] and sharing(registry,source['user_id'],target['user_id'],source['adapter_id'],target['adapter_id']) is None,
                'unauthorized controlled source/target sharing')
        require(target['sample_id']==source['sample_id'] and target['prepared']['sample']==source['prepared']['sample'],'unmatched dataset source/target')
        checks.append(dict(source_request_id=source['request_id'],target_request_id=target['request_id'],adapter_id=target['adapter_id'],
            source_user=source['user_id'],target_user=target['user_id'],source_tokens=p,target_tokens=len(target['token_ids']),suffix_tokens=len(target['token_ids'])-p,
            source_token_ids_sha256=digest(source['token_ids']),target_token_ids_sha256=digest(target['token_ids']),exact_full_prefix=True,
            expected_positions='zero-based source and target',expected_mask='all ones',runtime_HIT_required=True,
            runtime_HIT_authorized_by_static_check=False))
    require(p==block['prefix_length'],'prefix label differs')
    return checks


def prepare(registry,tok,tokenizer_identity,context_limit,special,output,*,multi_prepared,profile='SMOKE',
            datasets=None,cases=None,groups=None,adapters=None,repeats=None,generation_tokens=8,sharing_mode='SAME_USER',
            seed=42,capacity=512*2**20,evidence_budget=8*2**30,seconds_per_forward=5.,fixture=False,r0_diagnostic=None):
    require(profile in PROFILES,'unknown R1 profile');defaults=PROFILES[profile]
    datasets=tuple(defaults['datasets'] if datasets is None else datasets);cases=tuple(defaults['cases'] if cases is None else cases)
    groups=defaults['groups'] if groups is None else groups
    adapters=tuple(defaults['adapters'] if adapters is None else adapters);repeats=tuple(defaults['repeats'] if repeats is None else repeats)
    require(datasets and set(datasets)<=set(DATASETS) and len(set(datasets))==len(datasets),'unsupported/duplicate dataset')
    require(cases and set(cases)<=set(CASES) and len(set(cases))==len(cases),'unsupported/duplicate prefix/input case')
    require(type(groups) is int and 1<=groups<=6 and set(repeats)<={1,2,4} and repeats and len(set(repeats))==len(repeats),'bounded independent groups/repeats required')
    require(1<=generation_tokens<=16 and sharing_mode in ('SAME_USER','AUTHORIZED_CROSS_USER'),'invalid conservative generation/sharing policy')
    require(all(d in registry for d in datasets),'DATASET_NOT_READY: required real dataset registry key missing')
    identity=source_identity(fixture);m,pm,specs,users=validate(multi_prepared)
    require(m['stage']=='expanded' and len(users['users'])==30 and m['TEST_FIXTURE_ONLY']==fixture,'three adapters/thirty logical users required')
    require(adapters and set(adapters)<=set(specs) and len(set(adapters))==len(adapters),'unprepared/duplicate adapter')
    require(m['tokenizer']==tokenizer_identity,'prepared adapter/tokenizer identity differs')
    if profile=='SMOKE':
        require(len(datasets)==len(cases)==groups==len(adapters)==1 and repeats==(1,) and sharing_mode=='SAME_USER',
                'SMOKE is exactly one dataset, one pair/block, one adapter, same user')
    protocol=dict(schema=SCHEMA,profile=profile,source=identity,model_identity=pm['identity'],tokenizer=tokenizer_identity,
        multi_prepared=str(Path(multi_prepared).resolve()),multi_manifest_sha256=sha(Path(multi_prepared)/'multi_prepare.json'),
        multi_inventory_sha256=sha(Path(multi_prepared)/'r5_output_manifest.json') if (Path(multi_prepared)/'r5_output_manifest.json').exists() else None,
        adapter_fingerprints={a:s['matrix_sha256'] for a,s in specs.items()},registry_sha256=users['sha256'],
        datasets=list(datasets),cases=list(cases),groups=groups,adapters=list(adapters),target_repeats=list(repeats),generation_tokens=generation_tokens,
        sharing_mode=sharing_mode,seed=seed,r0=r0_receipt(r0_diagnostic),TEST_FIXTURE_ONLY=fixture,
        note='Controlled token relationships only; no candidate choice or HIT grant supplied to inference; cached Q is stored/accounted, never consumed')
    blocks=[];dataset_records={}
    # Existing sealed temporary plans are read and retained in memory, never edited/resealed.
    with tempfile.TemporaryDirectory(prefix='i5r1_prepare_') as temporary:
        for case_index,case in enumerate(cases):
            prefix,limit=CASES[case]
            old=prepare_experiment({d:registry[d] for d in datasets},tok,tokenizer_identity,context_limit,special,Path(temporary)/str(case_index),
                overrides={d:groups for d in datasets},budgets={d:generation_tokens for d in datasets},tracks=('CONTROLLED',),prefixes=(prefix,),
                repetitions=1,block_size=1,seed=seed,strict_per_block=1,capacity=capacity,evidence_budget=evidence_budget,
                allow_truncation=True,fixture=fixture,input_limit=limit,seconds_per_forward=seconds_per_forward)
            records=read(Path(temporary)/str(case_index)/'i5_dataset_manifest.json')
            for d,rec in records.items():
                if d not in dataset_records:dataset_records[d]=deepcopy(rec)
                else:dataset_records[d]['selections'].extend(rec['selections'])
            per_dataset={d:0 for d in datasets};seen_groups={d:set() for d in datasets}
            for original in old['blocks']:
                d=original['dataset'];ordinal=per_dataset[d];per_dataset[d]+=1
                group=original['tasks'][0]['group_id'];require(group not in seen_groups[d],'independent source groups unavailable')
                seen_groups[d].add(group);aid=adapters[ordinal%len(adapters)]
                authorized=sorted(u for u,v in users['users'].items() if v['adapter_id']==aid)
                for repeat in repeats:
                    b=deepcopy(original);b.pop('block_id');source,target=b['tasks'];source.update(user_id=authorized[0],adapter_id=aid)
                    target.update(user_id=authorized[0 if sharing_mode=='SAME_USER' else 1],adapter_id=aid)
                    b['tasks']=[source]+[dict(deepcopy(target),request_id=target['request_id']+f'_R{r:02d}') for r in range(repeat)]
                    for i,t in enumerate(b['tasks']):t.update(request_sequence_number=i,strict=True)
                    b.update(i5r1_case=case,i5r1_source_group=digest([d,group]),i5r1_target_repeats=repeat,i5r1_group_ordinal=ordinal,
                        scored_unique_samples=1,repeated_evaluation_samples=repeat-1,priming_requests=1,cache_lifetime_requests=1+repeat,
                        physical_requests_per_condition=1+repeat,requested_input_limit=limit,
                        condition_order=[CONDITIONS[i] for i in ORDERS[len(blocks)%len(ORDERS)]])
                    b['resources']=resources(b['tasks'],capacity=capacity,evidence_budget=evidence_budget,seconds_per_forward=seconds_per_forward)
                    b['pair_checks']=pair_checks(b,users,context_limit);b['block_id']=digest(b);blocks.append(b)
    plan=dict(old,blocks=blocks,routing_algorithm=None,i5r1=protocol,
        controlled_routing_algorithm='I5R1_EXPLICIT_AUTHORIZED_ADAPTER_USER_V1',
        scale_summary=dict(unique_scored_samples=len({(b['dataset'],t['sample_id']) for b in blocks for t in b['tasks'] if t['scored']}),
            scored_observations_per_condition=sum(len(b['tasks'])-1 for b in blocks),independent_blocks=len(blocks),
            cache_resets_all_conditions=3*len(blocks),maximum_cache_lifetime_requests=max(len(b['tasks']) for b in blocks)))
    plan.pop('experiment_id');plan['experiment_id']=digest(plan)
    out=fresh(output,(*registry.values(),multi_prepared,r0_diagnostic))
    write(out/'i5_workload_manifest.json',plan);write(out/'i5_dataset_manifest.json',dataset_records)
    write(out/'i5r1_protocol.json',protocol);seal(out)
    return plan


def validate_plan(path,multi_prepared=None):
    path=Path(path);verify(path);plan=read(path/'i5_workload_manifest.json');copy=dict(plan);identity=copy.pop('experiment_id')
    require(digest(copy)==identity and plan['i5r1']['schema']==SCHEMA,'R1 experiment identity differs')
    protocol=plan['i5r1'];require(read(path/'i5r1_protocol.json')==protocol,'R1 protocol differs')
    require(protocol['source']==source_identity(plan['TEST_FIXTURE_ONLY']),'R1 pinned source/commit differs')
    if multi_prepared is None:multi_prepared=protocol['multi_prepared']
    require(Path(multi_prepared).resolve()==Path(protocol['multi_prepared']),'R1 adapter preparation path differs')
    m,pm,adapters,registry=validate(multi_prepared)
    require(sha(Path(multi_prepared)/'multi_prepare.json')==protocol['multi_manifest_sha256'] and registry['sha256']==protocol['registry_sha256'] and
        {a:s['matrix_sha256'] for a,s in adapters.items()}==protocol['adapter_fingerprints'] and pm['identity']==protocol['model_identity'] and m['tokenizer']==protocol['tokenizer'],
        'R1 model/tokenizer/adapter identity differs')
    require(len(plan['blocks'])==len(protocol['datasets'])*len(protocol['cases'])*protocol['groups']*len(protocol['target_repeats']),
            'R1 planned block count differs')
    seen=set()
    for block in plan['blocks']:
        b=dict(block);identity=b.pop('block_id');require(digest(b)==identity and identity not in seen,'invalid/duplicate R1 block');seen.add(identity)
        require(pair_checks(block,registry,plan['context_limit'])==block['pair_checks'],'controlled pair proof differs')
        aid=protocol['adapters'][block['i5r1_group_ordinal']%len(protocol['adapters'])]
        authorized=sorted(u for u,v in registry['users'].items() if v['adapter_id']==aid)
        for t in block['tasks']:
            expected=authorized[0 if not t['scored'] or protocol['sharing_mode']=='SAME_USER' else 1]
            require(t['adapter_id']==aid and t['user_id']==expected,'R1 explicit user routing differs')
            require(t['policy']['max_new_tokens']==(protocol['generation_tokens'] if t['scored'] else 1),'R1 generation policy differs')
    return plan


def inspect(path):
    plan=validate_plan(path);blocks=plan['blocks'];p=plan['i5r1']
    return dict(status='CPU_STATIC_VALIDATED_GPU_NOT_RUN',experiment_id=plan['experiment_id'],source_commit=p['source']['commit'],
        blocks=len(blocks),valid_array=f'0-{len(blocks)-1}%1',datasets=p['datasets'],cases=p['cases'],
        scored_observations_per_condition=sum(len(b['tasks'])-1 for b in blocks),priming_per_condition=len(blocks),
        maximum_physical_requests_per_condition=max(len(b['tasks']) for b in blocks),
        resource_totals=dict(max_gpu_estimate_bytes=max(b['resources']['gpu_estimate_bytes'] for b in blocks),
            max_evidence_budget_per_condition=max(b['resources']['evidence_budget_bytes'] for b in blocks),
            total_evidence_budget_all_conditions=sum(3*b['resources']['evidence_budget_bytes'] for b in blocks),
            maximum_forwards_all_conditions=sum(3*b['resources']['expected_maximum_forwards'] for b in blocks),
            estimated_seconds_all_conditions=sum(3*b['resources']['duration_estimate_seconds'] for b in blocks)),
        block_table=[dict(index=i,dataset=b['dataset'],case=b['i5r1_case'],adapter=b['tasks'][0]['adapter_id'],
            target_repeats=b['i5r1_target_repeats'],source_group=b['i5r1_source_group'],pairs=b['pair_checks']) for i,b in enumerate(blocks)])
