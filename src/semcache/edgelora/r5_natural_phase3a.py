"""Five additional bounded NATURAL pilots; existing D5 execution and audit only."""
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
import importlib.util,json,os,subprocess
from .communication import require,digest
from .r3a_artifacts import ROOT,THRESHOLDS
from .r5_i5_data import DATASETS as I5_DATASETS
from .r5_i5_workload import TEMPLATE,turn_order
from .r4c_workload import policy
from .r5_i5_quality import validate_scorers
from .r5_i5r3a import provenance
from .r5_i5r2_tensors import Evidence,file_sha
from .r5_subsequence_natural import inputs,check_block,transport
from .r5_subsequence_natural_io import output,put,seal,verify as tree_verify
from .r5_subsequence_integrity import SCOPED
from . import r5_natural_approx as base

DATASETS=('coqa','snips','multiwoz','2wikimqa','multi_news')
SCHEMA='D5_PHASE3A_FIVE_DATASET_NATURAL_V1'
SCRIPT=ROOT/'scripts/158_r5_natural_phase3a.py'
FILES=('scripts/158_r5_natural_phase3a.py','jobs/r5_natural_phase3a.sh','tests/test_r5_natural_phase3a.py')
ORDERING='one original first-occurrence five-request NATURAL block; original routing/order; no priming'


def implementation():
    impl=base.implementation()
    for name in FILES:
        r=subprocess.run(['git','show',impl['source']['commit']+':'+name],cwd=ROOT,capture_output=True)
        require(r.returncode==0 and __import__('hashlib').sha256(r.stdout).hexdigest()==file_sha(ROOT/name),
            'commit reviewed Phase3A CLI/job/tests before prepare')
    return dict(source=impl['source'],files={**impl['files'],**{n:file_sha(ROOT/n) for n in FILES}})


def select_block(w,dataset,block_index,samples):
    require(dataset in DATASETS,'Phase3A accepts only five additional datasets; Qasper is excluded')
    first=next((i for i,b in enumerate(w['blocks']) if b['dataset']==dataset and b['track']=='NATURAL' and b['repetition']==0),None)
    require(first is not None,'MISSING_ORIGINAL_NATURAL_BLOCK: '+dataset)
    require(block_index==first,'must select first original NATURAL repetition0 block; no overlap-based selection')
    b=w['blocks'][block_index];tasks=check_block(b)
    require(len(tasks)==5,'ORIGINAL_BLOCK_NOT_FIVE: preserve original grouping; no slicing, combining or replacement')
    require(not b.get('continuity_breaks'),'DIALOGUE_GROUP_SPLIT_AT_BLOCK_BOUNDARY')
    require(w['template_version']==TEMPLATE and b['routing_seed']==w['seed'],'original template/seed differs')
    for t in tasks:
        budget=t['prepared']['max_new_tokens']
        require(type(budget) is int and 1<=budget<=96 and type(t['policy']['max_new_tokens']) is int and
            t['policy']['max_new_tokens']==budget,'sealed request/prepared generation budget differs or exceeds I5 bounds')
        require(t['policy']==policy(w['special_tokens'],'quality',budget),'sealed generation policy differs')
    if dataset in ('coqa','multiwoz'):
        all_groups=defaultdict(list);chosen=defaultdict(list)
        for s in samples:all_groups[s['group_id']].append(s)
        seen=[]
        for t in tasks:
            g=t['group_id'];chosen[g].append(t['sample_id'])
            if not seen or seen[-1]!=g:
                require(g not in seen,'dialogue group is not contiguous');seen.append(g)
        for g,ids in chosen.items():
            require(ids==[s['id'] for s in sorted(all_groups[g],key=turn_order)],
                'DIALOGUE_GROUP_INCOMPLETE_OR_TURN_ORDER_CHANGED: '+g+'; no partial conversations')
    return deepcopy(b)


def original_plan(path):
    ev=Evidence(path);receipt=ev.verify();w=ev.json('i5_workload_manifest.json');d=ev.json('i5_dataset_manifest.json')
    body=dict(w);eid=body.pop('experiment_id')
    require(w['schema']=='I5_EXPERIMENT_V1' and not w['TEST_FIXTURE_ONLY'] and digest(body)==eid,'real original sealed I5 plan required')
    return w,d,receipt


def discover(search_roots=(),registries=(),original_experiments=(),prepared_datasets=(),multi=None):
    """Read-only local/SERAPH discovery. Prints records; never prepares data or imports a model."""
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU discovery requires hidden CUDA')
    candidates=set();declared={};plans={Path(p).resolve() for p in original_experiments};registry_results=[];roots=[]
    for item in prepared_datasets:
        name,sep,raw=item.partition('=');require(sep and name in DATASETS and raw,'prepared dataset must be DATASET=existing_path')
        path=Path(raw).resolve();candidates.add(path);declared[path]=name
    for raw in search_roots:
        root=Path(raw).resolve();roots.append(dict(path=str(root),exists=root.is_dir()))
        if root.is_dir():
            candidates.update(p.parent for p in root.rglob('i5_dataset_manifest.json'))
            plans.update(p.parent for p in root.rglob('i5_workload_manifest.json'))
    for raw in registries:
        path=Path(raw).resolve();record=dict(path=str(path),status='MISSING')
        if path.is_file():
            try:
                mapping=json.loads(path.read_text());require(isinstance(mapping,dict),'registry must map datasets to prepared paths')
                record.update(status='READ',datasets=list(mapping))
                for name,value in mapping.items():
                    require(name in I5_DATASETS and isinstance(value,str),'invalid dataset registry')
                    # Existing I5 registry paths are interpreted from the repo
                    # working directory, not from the registry's parent.
                    p=Path(value);p=(ROOT/p).resolve() if not p.is_absolute() else p
                    candidates.add(p);declared[p]=name
            except (ValueError,OSError,TypeError) as exc:record.update(status='INVALID',reason=str(exc))
        registry_results.append(record)
    plan_records=[];plan_data=[]
    for path in sorted(plans):
        row=dict(path=str(path))
        try:
            w,d,receipt=original_plan(path);row.update(status='SEALED_REAL_PLAN',inventory=receipt)
            for value in d.values():candidates.add(Path(value['prepared_path']).resolve())
            plan_data.append((path,w,d))
        except (ValueError,OSError,KeyError,TypeError) as exc:row.update(status='REJECTED',reason=str(exc))
        plan_records.append(row)
    datasets={n:dict(prepared=[],declared_prepared_paths=[str(p) for p,v in declared.items() if v==n],original_blocks=[],blockers=[]) for n in DATASETS};rejected=[]
    for path in sorted(candidates):
        try:
            m=json.loads((path/'i5_dataset_manifest.json').read_text());name=m['dataset']
            if name not in DATASETS:continue
            require(path not in declared or declared[path]==name,'registry/prepared dataset identity differs')
            samples,m=provenance(path,name)
            datasets[name]['prepared'].append(dict(path=str(path),inventory=Evidence(path).verify(),revision=m['revision'],
                split=m['split'],samples=len(samples),version=m['version'],source=m['source'],source_path=m['source_path']))
        except (ValueError,OSError,KeyError,TypeError) as exc:rejected.append(dict(path=str(path),dataset=declared.get(path),reason=str(exc)))
    for path,w,d in plan_data:
        for name in DATASETS:
            if name not in d:continue
            row=dict(plan=str(path))
            try:
                samples,manifest=provenance(d[name]['prepared_path'],name)
                require(file_sha(Path(d[name]['prepared_path'])/'i5_output_manifest.json')==d[name]['prepared_manifest_sha256'] and
                    all(d[name][k]==v for k,v in manifest.items()),'plan/dataset manifest differs')
                index=next(i for i,b in enumerate(w['blocks']) if b['dataset']==name and b['track']=='NATURAL' and b['repetition']==0)
                row['block_index']=index;b=select_block(w,name,index,samples)
                row.update(block_id=b['block_id'],requests=len(b['tasks']),status='STATIC_BLOCK_VALID',
                    resources=base.preflight(b['tasks']),transport='NOT_CHECKED_UNTIL_PREPARE')
            except (ValueError,OSError,KeyError,TypeError,StopIteration) as exc:row.update(status='BLOCKED',reason=str(exc))
            datasets[name]['original_blocks'].append(row)
    for name,record in datasets.items():
        if not record['prepared']:record['blockers'].append('MISSING_VALID_SEALED_I5_PREPARATION: '+name)
        if not any(r['status']=='STATIC_BLOCK_VALID' for r in record['original_blocks']):record['blockers'].append('MISSING_COMPATIBLE_ORIGINAL_FIVE_REQUEST_BLOCK: '+name)
        try:validate_scorers({name});record['scorer']='AVAILABLE'
        except (ValueError,ImportError) as exc:record['scorer']='BLOCKED';record['blockers'].append(str(exc))
        record['status']='BLOCKED' if record['blockers'] else 'ASSETS_STATICALLY_AVAILABLE_CPU_APPROVAL_PENDING'
    adapter=dict(path=str(Path(multi).resolve()) if multi else None,status='NOT_CHECKED')
    if multi:
        try:
            from .r5_multi_artifacts import validate
            m,_,specs,users=validate(multi)
            require(not m['TEST_FIXTURE_ONLY'] and m['stage']=='expanded' and len(users['users'])==30 and set(specs)=={'Adapter_A','Adapter_B','Adapter_C'},'real expanded30-user/3-adapter preparation required')
            adapter.update(status='VALID',inventory=Evidence(multi).verify('r5_output_manifest.json'))
        except (ValueError,OSError,KeyError,ImportError) as exc:adapter.update(status='BLOCKED',reason=str(exc))
    return dict(schema=SCHEMA,status='CPU_DISCOVERY_ONLY',GPU_status='GPU_NOT_RUN',scope='current host only; no remote existence inferred',
        search_roots=roots,registries=registry_results,plans=plan_records,datasets=datasets,rejected_candidates=rejected,multi=adapter,
        native_dependencies={n:importlib.util.find_spec(n) is not None for n in ('torch','transformers','numpy')})


def material(original_experiment,multi,dataset,block_index):
    w,d,receipt,inv,m,pm,specs,users,tok=inputs(original_experiment,multi,_datasets=I5_DATASETS)
    require(m['stage']=='expanded','existing expanded adapter preparation required')
    require(dataset in d,'MISSING_DATASET_IN_ORIGINAL_PLAN: '+dataset)
    samples,_=provenance(d[dataset]['prepared_path'],dataset)
    b=select_block(w,dataset,block_index,samples)
    validate_scorers({dataset})
    special={k:getattr(tok.tokenizer,k) for k in ('bos_token_id','eos_token_id','pad_token_id')}
    require(w['special_tokens']==special,'original special token policy differs')
    for t in b['tasks']:
        require(t['prepared']['template_version']==TEMPLATE,'original template differs')
    network=transport(b['tasks'],pm,specs,tok.identity,users)
    fields=dict(original_experiment=str(Path(original_experiment).resolve()),multi=str(Path(multi).resolve()),
        original_inventory=receipt,dataset_inventories=inv,multi_inventory=Evidence(multi).verify('r5_output_manifest.json'),
        dataset_paths={n:v['prepared_path'] for n,v in d.items()},base_prepared=m['base_prepared'],tokenizer_identity=tok.identity,
        dataset=dataset,block_index=block_index,block_id=b['block_id'],tasks=b['tasks'],tasks_sha256=digest(b['tasks']),
        resources=base.preflight(b['tasks'],network),transport=network)
    return fields,[b,m,pm,specs,users,tok]


def prepare(original_experiment,multi,source_commit,dataset,block_index,output_dir,enable_approximate_experiment=False):
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU preparation requires hidden CUDA')
    require(enable_approximate_experiment,'explicit approximate opt-in required')
    impl=implementation();require(source_commit==impl['source']['commit'],'reviewed source commit differs')
    fields,_=material(original_experiment,multi,dataset,block_index)
    p=dict(fields,schema=SCHEMA,implementation=impl,track='NATURAL',ordering=ORDERING,conditions=list(base.CONDITIONS),
        condition_order=list(base.CONDITIONS),execution={c:dict(mode=base.execution(c)[0],allow_approximate=base.execution(c)[1]) for c in base.CONDITIONS},
        enable_approximate_experiment=True,validation_mode=SCOPED,thresholds=THRESHOLDS,warmup_requests=0,
        expansion='ONE_DATASET_ONE_ALLOCATION_NO_AUTOMATIC_EXPANSION',GPU_status='GPU_NOT_RUN')
    p['plan_id']=digest(p)
    out=output(output_dir,(original_experiment,multi,*p['dataset_paths'].values()));put(out/'approx_plan.json',p);seal(out)
    return dict(status='CPU_PREPARED_GPU_NOT_RUN',dataset=dataset,plan_id=p['plan_id'],resources=p['resources'])


def load(experiment):
    require(not Path(experiment).is_symlink(),'plan symlink forbidden');experiment=Path(experiment).resolve()
    receipt=tree_verify(experiment);p=json.loads((experiment/'approx_plan.json').read_text());body=dict(p);pid=body.pop('plan_id')
    require(p['schema']==SCHEMA and digest(body)==pid and p['implementation']==implementation(),'pinned Phase3A plan/source differs')
    require(p['conditions']==p['condition_order']==list(base.CONDITIONS) and p['track']=='NATURAL' and p['ordering']==ORDERING and
        p['execution']=={c:dict(mode=base.execution(c)[0],allow_approximate=base.execution(c)[1]) for c in base.CONDITIONS} and
        p['enable_approximate_experiment'] is True and p['validation_mode']==SCOPED and p['thresholds']==THRESHOLDS and
        p['warmup_requests']==0 and p['expansion']=='ONE_DATASET_ONE_ALLOCATION_NO_AUTOMATIC_EXPANSION' and p['GPU_status']=='GPU_NOT_RUN','Phase3A scope/safety differs')
    fields,rest=material(p['original_experiment'],p['multi'],p['dataset'],p['block_index'])
    require(all(p[k]==v for k,v in fields.items()),'original selected block/assets/routing/resources differ')
    return p,receipt,rest


def approval(experiment,cpu_audit):
    require(not Path(cpu_audit).is_symlink(),'CPU approval symlink forbidden')
    p,receipt,rest=load(experiment);ar=tree_verify(Path(cpu_audit).resolve());a=json.loads((Path(cpu_audit)/'cpu_validation.json').read_text())
    require(a['status']=='PASS' and a['failed']==a['skipped']==0 and a['implementation']==p['implementation'] and a['plan_inventory']==receipt,'zero-skip Phase3A CPU approval differs')
    return p,receipt,rest,ar


class Profile(base.Profile):
    script=SCRIPT;tests=['test_r5_natural_phase3a']
    load=staticmethod(load);approval=staticmethod(approval)


def cpu_validate(experiment,output_dir):return base.cpu_validate(experiment,output_dir,_profile=Profile())
def inspect(experiment,cpu_audit):return base.inspect(experiment,cpu_audit,_profile=Profile())
def run(**kwargs):return base.run(**kwargs,_profile=Profile())


def worker(experiment,output_dir,condition,cpu_audit,expected_gpu_uuid=None):
    from .r5_subsequence_natural_runtime import worker as native_worker
    return native_worker(experiment,output_dir,condition,cpu_audit,expected_gpu_uuid,_profile=Profile())


def ud(config,output_dir):
    from .r5_subsequence_natural_runtime import ud as native_ud
    return native_ud(config,output_dir,_implementation=implementation)


def analyze(**kwargs):
    from .r5_natural_approx_audit import analyze as audit
    return audit(**kwargs,_approval=approval,_request_policy=True)


def verify(**kwargs):
    from .r5_natural_approx_audit import verify as audit
    return audit(**kwargs,_approval=approval,_request_policy=True)


def summary_rows(dataset,plan,audit,audit_path):
    """Read existing independently audited request summaries; no new numerical auditor."""
    from .r5_i5_quality import aggregate as quality_aggregate
    rows=[]
    for c in base.CONDITIONS:
        total=audit['conditions'][c]
        requests=[json.loads((Path(audit_path)/c/(t['request_id']+'.json')).read_text()) for t in plan['tasks']]
        n=len(requests);require(n==5,'five audited requests required')
        comparisons=[] if c==base.CONDITIONS[0] else [json.loads((Path(audit_path)/c/(t['request_id']+'_numerical.json')).read_text()) for t in plan['tasks']]
        rows.append(dict(dataset=dataset,condition=c,requests=n,evidence_integrity='PASS',
            physical_hit_requests=total['physical_hit_requests'],physical_hit_rate=total['physical_hit_requests']/n,
            reused_qkv_projection_rows=total['projection_rows_skipped'],skipped_cpu_lora_rows=total['projection_rows_skipped'],
            categories=total['physical_qkv_rows_consumed_by_category'],policy=total['policy'],
            request_cache_occupancy_bytes=[r['cache_bytes'] for r in requests],
            request_policy_counts={t['request_id']:r['cache_policy_counts'] for t,r in zip(plan['tasks'],requests)},
            total_application_e2e_ms=total['application_e2e_ms'],mean_e2e_ms=total['application_e2e_ms']/n,
            mean_ttft_ms=total['application_ttft_sum_ms']/n,wire_bytes=total['wire_bytes'],
            mean_prefill_ms=sum(r['model_timings']['prefill_ms'] for r in requests)/n,
            mean_decode_ms=sum(r['model_timings']['decode_ms'] for r in requests)/n,
            components_inclusive_ms=total['cache_and_inference_component_inclusive_ms'],
            component_scope=total['component_scope'],quality=quality_aggregate(total['quality']),
            quality_delta_vs_full=total['task_quality_delta_vs_full'],delta_e2e_ms_vs_full=total['delta_e2e_ms_vs_full'],
            numerical_fidelity='REFERENCE' if not comparisons else ('PASS' if all(x['numerical_tolerance']=='PASS' for x in comparisons) else 'FAIL'),
            generated_token_agreements=n if not comparisons else sum(x['generated_token_ids_equal'] for x in comparisons),
            generated_text_agreements=n if not comparisons else sum(x['generated_text_equal'] for x in comparisons),
            correctness_qualified_speedup='NOT_ESTABLISHED'))
    return rows


def aggregate(results_registry,output_dir):
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU aggregation requires hidden CUDA')
    path=Path(results_registry).resolve();mapping=json.loads(path.read_text())
    require(set(mapping)==set(DATASETS),'exactly five dataset-specific audited results required; Qasper excluded')
    rows=[];bindings={};protected=[path]
    for dataset in DATASETS:
        record=mapping[dataset]
        require(set(record)=={'experiment','run','cpu_audit','audit','verification'},'explicit plan/run/CPU/audit/verification paths required')
        p,pr,_,ar=approval(record['experiment'],record['cpu_audit'])
        require(p['dataset']==dataset,'result registry dataset/plan differs')
        ai=tree_verify(Path(record['audit']).resolve());vi=tree_verify(Path(record['verification']).resolve());ri=tree_verify(Path(record['run']).resolve())
        a=json.loads((Path(record['audit'])/'approx_audit.json').read_text())
        v=json.loads((Path(record['verification'])/'approx_verification.json').read_text())
        require(v['status']=='PASS' and v['audit_inventory']==ai and a['status']=='EVIDENCE_INTEGRITY_PASS' and
            a['schema']==SCHEMA and a['track']=='NATURAL' and a['plan_id']==p['plan_id'] and
            a['plan_inventory']==pr and a['cpu_audit_inventory']==ar and a['run_inventory']==ri and
            set(a['conditions'])==set(base.CONDITIONS),'independent audit/verification binding differs')
        rows.extend(summary_rows(dataset,p,a,record['audit']))
        bindings[dataset]=dict(plan_inventory=pr,cpu_inventory=ar,run_inventory=ri,audit_inventory=ai,verification_inventory=vi)
        protected.extend([*record.values(),*base.protected(p)])
    out=output(output_dir,protected)
    put(out/'phase3a_results.json',dict(schema=SCHEMA,status='AUDITED_RESULTS_AGGREGATED',rows=rows,bindings=bindings,
        scope='five independent instrumented pilots; no pooled task metric or causal speedup claim'))
    lines=['# Five-dataset D5 A/EXACT/APPROX validation','',
        '| Dataset | Condition | HITs/5 | QKV rows skipped | A/B/C/D rows | Mean TTFT ms | Mean E2E ms | Quality | Delta quality | Numerical | Tokens agree/5 | Text agree/5 |',
        '|---|---|---:|---:|---|---:|---:|---:|---:|---|---:|---:|']
    for r in rows:
        lines.append('| '+ ' | '.join(map(str,[r['dataset'],r['condition'],r['physical_hit_requests'],r['reused_qkv_projection_rows'],
            '/'.join(str(r['categories'][k]) for k in 'ABCD'),round(r['mean_ttft_ms'],3),round(r['mean_e2e_ms'],3),
            round(r['quality']['score'],6),round(r['quality_delta_vs_full'],6),r['numerical_fidelity'],
            r['generated_token_agreements'],r['generated_text_agreements']]))+' |')
    lines+=['','Evidence integrity PASS is independent of numerical FAIL. Inclusive profiler components overlap and must not be summed to reconstruct E2E. Quality metrics differ by dataset; no pooled quality average. No correctness-qualified speedup is claimed.']
    (out/'phase3a_results.md').write_text('\n'.join(lines)+'\n');seal(out)
    return dict(status='AUDITED_RESULTS_AGGREGATED',datasets=list(DATASETS),rows=len(rows))
