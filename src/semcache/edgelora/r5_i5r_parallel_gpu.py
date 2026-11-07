"""Controlled exact-prefix pilots: unchanged I5 forwards and independent audits."""
from copy import deepcopy
from pathlib import Path
import os,subprocess,sys,tempfile
from .communication import digest,digest_bytes,require
from .r3a_artifacts import ROOT,read,write,sha,THRESHOLDS
from .r5_i5_evidence import seal,sources
from .r5_i5r2_tensors import Evidence
from .r5_i5r_parallel import assets,provenance,output
from .r5_i5_workload import select,requests,resources,prepare_experiment,BUDGETS,CONDITIONS
from .r5_i5r1_plan import source_identity
from .r5_i5r3b_transport import preflight

def repeated_tasks(source,target,negative,n):
    require(n in (1,2,3,4) and source['token_ids']==target['token_ids'][:512] and len(source['token_ids'])==512<len(target['token_ids']),'strict controlled prefix required')
    require(source['adapter_id']==target['adapter_id'] and source['user_id']==target['user_id'],'positive must preserve original same-user adapter routing')
    require(negative['token_ids'][:512]!=source['token_ids'] and negative['adapter_id']==source['adapter_id'],'compatible genuine negative required')
    ts=[deepcopy(source)]+[dict(deepcopy(target),request_id=target['request_id']+f'_R{i+1:02d}') for i in range(n)]+[deepcopy(negative)]
    for i,t in enumerate(ts):t['request_sequence_number']=i
    return ts

def prepare(registry,multi,dataset,output_dir,source_commit,contexts=1,reuses=3,input_limit=1280):
    require(dataset in ('qasper','2wikimqa','multi_news') and contexts in (1,2,3) and reuses in (1,2,3,4),'bounded controlled pilot required')
    require(512<input_limit<=1280,'explicit transport-limited prompt bound required')
    src=source_identity();require(src['commit']==source_commit and not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True).strip(),'reviewed clean committed source required')
    tracked=set(subprocess.check_output(['git','ls-files'],cwd=ROOT,text=True).splitlines());require(set(sources())<=tracked,'commit new implementation sources before preparation')
    paths,m,pm,specs,users,tok=assets(registry,multi);require(dataset in paths,'DATASET_NOT_READY')
    samples,dm=provenance(paths[dataset],dataset)
    from .r5_i5_quality import validate_scorers
    validate_scorers({dataset:dm})
    eligible=[]
    from .r5_i5_workload import prepare_prompt,routing
    for s in samples:
        try:p=prepare_prompt(s,tok.tokenizer,2048,BUDGETS[dataset],allow_truncation=True,input_limit=input_limit)
        except ValueError:continue
        if len(p['token_ids'])>512:p['route']=routing(s);eligible.append(p)
    eligible.sort(key=lambda p:digest(p['sample']['id']));chosen=[];seen=set()
    for p in eligible:
        key=digest_bytes(p['sample']['context'].encode())
        if key not in seen:chosen.append(p);seen.add(key)
        if len(chosen)==contexts:break
    require(len(chosen)==contexts,'insufficient independent real contexts')
    special={k:getattr(tok.tokenizer,k) for k in ('bos_token_id','eos_token_id','pad_token_id')}
    blocks=[]
    for i,p in enumerate(chosen):
        p=deepcopy(p);p.update(strict=True,source_semantic_text=tok.tokenizer.decode(p['token_ids'][:512],skip_special_tokens=False,clean_up_tokenization_spaces=False))
        source,target=requests([p],track='CONTROLLED',prefix=512,offset=i*2,special=special)
        neg=next((q for q in eligible if q['route']['adapter_id']==target['adapter_id'] and q['sample']['context']!=p['sample']['context'] and q['token_ids'][:512]!=source['token_ids']),None)
        require(neg is not None,'no compatible real negative target; no synthetic fallback')
        neg=deepcopy(neg);neg['strict']=True
        negative=requests([neg],track='NATURAL',prefix=None,offset=i*2+1,special=special)[0];negative['role']='NEGATIVE_TARGET'
        ts=repeated_tasks(source,target,negative,reuses)
        blocks.append(ts)
    # Use a real standard I5 template, then seal only NEW derived child plans.
    with tempfile.TemporaryDirectory() as tmp:
        base=prepare_experiment({dataset:paths[dataset]},tok.tokenizer,tok.identity,2048,special,Path(tmp)/'template',overrides={dataset:1},tracks=('CONTROLLED',),prefixes=(512,),repetitions=1,block_size=1,strict_per_block=1,allow_truncation=True,input_limit=input_limit,evidence_budget=16*2**30)
        manifests=read(Path(tmp)/'template/i5_dataset_manifest.json')
    plans={}
    for control in ('abc','a0'):
        plan=deepcopy(base);plan['blocks']=[]
        for i,ts in enumerate(blocks):
            b=deepcopy(base['blocks'][0]);tasks=deepcopy(ts if control=='abc' else ts[1:])
            for j,t in enumerate(tasks):t['request_sequence_number']=j
            b.update(tasks=tasks,cache_lifetime_requests=len(tasks),physical_requests_per_condition=len(tasks),priming_requests=1 if control=='abc' else 0,scored_unique_samples=2,repeated_evaluation_samples=reuses-1,
                     resources=resources(tasks,evidence_budget=16*2**30),condition_order=CONDITIONS[i%3:]+CONDITIONS[:i%3],classification='CONTROLLED_EXACT_PREFIX',independent_source_context=digest_bytes(ts[0]['prepared']['sample']['context'].encode()))
            if control=='a0':b['track']='NATURAL'
            b.pop('block_id',None);b['block_id']=digest(b);plan['blocks'].append(b)
        plan.update(block_size=reuses+2,classification='CONTROLLED_EXACT_PREFIX',conditions=list(CONDITIONS),scale_summary=dict(unique_scored_samples=len({t['sample_id'] for ts in blocks for t in ts if t['scored']}),independent_source_contexts=contexts,maximum_cache_lifetime_requests=reuses+2))
        plan.pop('experiment_id',None);plan['experiment_id']=digest(plan);plans[control]=plan
    transport=preflight(plans,pm,specs,m['tokenizer'],users)
    require(all(b['resources']['duration_estimate_seconds']*3+180<10800 for plan in plans.values() for b in plan['blocks']),'three-hour child estimate exceeded; no silent shrinking')
    manifests[dataset]['selections']=[dict(source_sample_id=ts[0]['sample_id'],target_sample_id=ts[1]['sample_id'],negative_sample_id=ts[-1]['sample_id'],prefix_sha256=digest(ts[0]['token_ids']),target_sha256=digest(ts[1]['token_ids']),original_context_sha256=digest_bytes(ts[0]['prepared']['sample']['context'].encode()),route={k:ts[0][k] for k in ('user_id','adapter_id')},classification='CONTROLLED_EXACT_PREFIX') for ts in blocks]
    p=dict(source=src,registry_path=str(Path(registry).resolve()),registry_sha256=sha(registry),input_limit=input_limit,multi_prepared=str(Path(multi).resolve()),multi_inventory=Evidence(multi).verify('r5_output_manifest.json'),dataset_path=str(Path(paths[dataset]).resolve()),dataset_inventory=Evidence(paths[dataset]).verify(),dataset=dataset,contexts=contexts,reuses=reuses,plans={k:v['experiment_id'] for k,v in plans.items()},transport=transport,thresholds=THRESHOLDS,
           note='N repeated target executions, one unique positive sample per independent context; one extra real negative. A0 auxiliary B/C execute because unchanged I5 runner requires ABC; economics use A0 FULL only.',GPU_NOT_RUN=True)
    out=output(output_dir,(registry,multi,*paths.values()))
    for control,plan in plans.items():
        child=out/control;child.mkdir();write(child/'i5_workload_manifest.json',plan);write(child/'i5_dataset_manifest.json',manifests);seal(child)
    write(out/'experiment3_plan.json',p);seal(out);return dict(status='SEALED_CONTROLLED_GPU_NOT_RUN',contexts=contexts,reuses=reuses,transport=transport)

def load(experiment):
    ev=Evidence(experiment);receipt=ev.verify();p=ev.json('experiment3_plan.json');require(p['source']==source_identity(),'pinned reviewed source differs')
    require(p['thresholds']==THRESHOLDS,'thresholds changed')
    require(Evidence(p['multi_prepared']).verify('r5_output_manifest.json')==p['multi_inventory'] and Evidence(p['dataset_path']).verify()==p['dataset_inventory'],'prepared inputs changed')
    from .r3a_artifacts import sha
    require(sha(p['registry_path'])==p['registry_sha256'],'registry changed')
    paths,m,pm,specs,users,tok=assets(p['registry_path'],p['multi_prepared'])
    samples,_=provenance(p['dataset_path'],p['dataset']);original={s['id']:s for s in samples}
    from .r5_i5_workload import prepare_prompt,routing
    plans={}
    for c,key in p['plans'].items():
        e=Evidence(Path(experiment)/c);e.verify();q=e.json('i5_workload_manifest.json');d=dict(q);identifier=d.pop('experiment_id');require(identifier==digest(d)==key and q['tokenizer_identity']==tok.identity,'child plan/tokenizer changed');plans[c]=q
        for block in q['blocks']:
            for t in block['tasks']:
                sample=original[t['sample_id']];require(t['prepared']['sample']==sample,'original sample/reference differs')
                prompt=prepare_prompt(sample,tok.tokenizer,2048,BUDGETS[p['dataset']],allow_truncation=True,input_limit=p['input_limit'])
                expected=prompt['token_ids'] if t['scored'] else prompt['token_ids'][:512]
                require(t['token_ids']==expected and all(t[k]==v for k,v in routing(sample).items()),'prompt/routing differs')
                require(t['policy']['max_new_tokens']==(BUDGETS[p['dataset']] if t['scored'] else 1),'generation budget changed')
    require(preflight(plans,pm,specs,m['tokenizer'],users)==p['transport'],'transport envelope changed')
    return p,plans,receipt

def run(experiment,source_commit,output_dir,block_index=0,enable_gpu_experiment=False):
    p,plans,_=load(experiment);require(enable_gpu_experiment and source_commit==p['source']['commit'] and 0<=block_index<p['contexts'],'explicit bounded source/block opt-in required')
    require(os.environ.get('SLURM_JOB_ID') and os.environ.get('R5_I5_ENABLE_GPU')==os.environ.get('R5_ENABLE_GPU')=='1','manual opted-in GPU allocation required')
    import shutil
    reserve=sum(3*q['blocks'][block_index]['resources']['evidence_budget_bytes'] for q in plans.values())+32*2**30
    require(shutil.disk_usage(Path(experiment).parent).free>=reserve,'combined ABC/A0 evidence disk reserve unavailable')
    out=output(output_dir,(experiment,p['multi_prepared'],p['dataset_path']));status=dict(status='FAILED',block_index=block_index,source_commit=source_commit)
    primary=None
    try:
        for c in ('abc','a0'):
            cmd=[sys.executable,str(ROOT/'scripts/119_run_r5_i5.py'),'--experiment',str(Path(experiment)/c),'--multi-prepared',p['multi_prepared'],'--block-index',str(block_index),'--source-commit',source_commit,'--runtime-budget-seconds','10800','--output-dir',str(out/(c+'_run')),'--enable-gpu-experiment']
            require(subprocess.run(cmd,cwd=ROOT).returncode==0,'I5 failed; preserve attempt, no retry')
        status['status']='COMPLETE'
    except BaseException as exc:primary=exc;status['reason']=str(exc)[:512];raise
    finally:
        try:write(out/'experiment3_status.json',status);seal(out)
        except BaseException as exc:
            if primary is None:raise
            if hasattr(primary,'add_note'):primary.add_note(str(exc))
    return status

def compute(experiment,runs):
    p,plans,pr=load(experiment);seen=set();results=[]
    from .r5_i5_analysis import analyze
    from .r5_i5r3bbe import economics,sequence
    from .r5_i5r3bbe_analysis import physical_checks
    from .r5_i5r3b_analysis import scope_cost
    for path in runs:
        e=Evidence(path,max_input_bytes=128*2**30);receipt=e.verify();s=e.json('experiment3_status.json');index=s['block_index']
        require(s['status']=='COMPLETE' and s['source_commit']==p['source']['commit'] and index not in seen and 0<=index<p['contexts'],'duplicate/failed/invalid block');seen.add(index)
        reports={c:analyze(Path(path)/(c+'_run')) for c in plans}
        for c,r in reports.items():
            cfg=r['config'];require(cfg['block']==plans[c]['blocks'][index] and cfg['experiment_id']==p['plans'][c] and cfg['source_commit']==p['source']['commit'] and cfg['sources']==p['source']['files'] and not cfg['fixture'],'execution/plan identity differs')
        require(reports['abc']['config']['expected_execution_gpu_uuid']==reports['a0']['config']['expected_execution_gpu_uuid'],'unmatched actual GPU')
        sessions=[a['session_id'] for r in reports.values() for info in r['independent_audit']['conditions'].values() for a in info['adapters'].values()]
        require(len(sessions)==len(set(sessions)),'ABC/A0 shared a CPU UD session')
        lives={c:[read(Path(path)/(c+'_run')/cond/'lifecycle.json') for cond in CONDITIONS] for c in reports}
        require(max(x['execution_end_ns'] for x in lives['abc'])<=min(x['execution_start_ns'] for x in lives['a0']),'paired control order differs')
        rows={c:{cond:[next(x for x in r['request_metrics'] if x['condition']==cond and x['request_id']==t['request_id']) for t in plans[c]['blocks'][index]['tasks']] for cond in CONDITIONS} for c,r in reports.items()}
        cr=rows['abc']['SEMCACHE_INTEGRATED'];ar=rows['a0']['FULL_RECOMPUTE'];econ=economics(cr,ar);checks=physical_checks(plans['abc']['blocks'][index]['tasks'],cr)
        result=dict(block_index=index,inventory=receipt,economics=econ,physical_checks=checks,all_metrics=rows,quality=reports['abc']['quality_per_sample'],numerical=reports['abc']['numerical_fidelity'],network={c:r.get('network_summary',r.get('network','NOT_OBSERVABLE_IN_SUMMARY; see all_metrics and sealed I5 wire records')) for c,r in reports.items()},decision=reports['abc']['decision'],
                    warm_target_comparison=[dict(request_id=a['request_id'],A_ms=a['application_e2e_ms'],C_ms=c['application_e2e_ms'],difference_ms=c['application_e2e_ms']-a['application_e2e_ms']) for a,c in zip(rows['abc']['FULL_RECOMPUTE'][1:],cr[1:])],
                    PHYSICAL_HIT_VERIFIED=all(x['pass_'] for x in checks),ECONOMIC_BREAK_EVEN='PASS' if econ['D_ms']<0 else 'FAIL',NUMERICAL_FIDELITY=reports['abc']['decision']['numerical_tolerance'],CORRECTNESS_QUALIFIED_SPEEDUP='NOT_ESTABLISHED',independent_source_contexts=1,unique_positive_samples=1,repeated_positive_observations=p['reuses']-1)
        result['lifecycle_costs']={c:{cond:scope_cost(rows[c][cond],r['independent_audit']['conditions'][cond]['initialization']['model_and_session_initialization_ms'],r['independent_audit']['conditions'][cond]['warmup_total_ms'],read(Path(path)/(c+'_run')/cond/'lifecycle.json')) for cond in CONDITIONS} for c,r in reports.items()}
        result['generation_throughput']={c:{cond:dict(tokens_per_second=sum(x['generated_tokens'] for x in rs)/((rs[-1]['completion_ns']-rs[0]['arrival_ns'])/1e9),scope='Instrumented request sequence including gaps/admission; initialization/warmup excluded; source token included for ABC') for cond,rs in by.items()} for c,by in rows.items()}
        result['matched_source_inclusive_C_minus_A_ms']=sequence(cr)['span_ms']-sequence(rows['abc']['FULL_RECOMPUTE'])['span_ms']
        result['warm_pairwise']=[dict(request_id=cr[i]['request_id'],B_minus_A_e2e_ms=rows['abc']['SEMANTIC_NO_REUSE'][i]['application_e2e_ms']-rows['abc']['FULL_RECOMPUTE'][i]['application_e2e_ms'],C_minus_B_e2e_ms=cr[i]['application_e2e_ms']-rows['abc']['SEMANTIC_NO_REUSE'][i]['application_e2e_ms'],C_minus_A_wire_bytes=cr[i]['total_request_wire_bytes']-rows['abc']['FULL_RECOMPUTE'][i]['total_request_wire_bytes']) for i in range(1,len(cr))]
        results.append(result)
    require(seen==set(range(p['contexts'])),'missing context blocks; no partial pooling')
    return dict(plan_inventory=pr,source=p['source'],blocks=sorted(results,key=lambda r:r['block_index']),independent_source_contexts=p['contexts'],classification='CONTROLLED_EXACT_PREFIX',statistical_scope='DESCRIPTIVE_ONLY: repeats are not independent samples; one paired run per context',ROOT_CAUSE='UNRESOLVED')

def analyze(experiment,runs,output_dir):
    result=compute(experiment,runs);out=output(output_dir,(experiment,*runs));write(out/'experiment3_results.json',result)
    write(out/'experiment3_independent_audit.json',dict(status='PASS',scope='I5 tensor/TCP/logit evidence verified; physical/numerical/performance verdicts remain separate',source=result['source'],plan_inventory=result['plan_inventory'],run_inventories=[b['inventory'] for b in result['blocks']]))
    seal(out);return dict(status='INDEPENDENT_EVIDENCE_AUDIT_PASS',verdicts=[{k:r[k] for k in ('PHYSICAL_HIT_VERIFIED','ECONOMIC_BREAK_EVEN','NUMERICAL_FIDELITY','CORRECTNESS_QUALIFIED_SPEEDUP')} for r in result['blocks']])

def verify(experiment,runs,audit,output_dir):
    ev=Evidence(audit);receipt=ev.verify();result=compute(experiment,runs);a=ev.json('experiment3_independent_audit.json')
    require(ev.json('experiment3_results.json')==result and a['status']=='PASS' and a['source']==result['source'] and a['plan_inventory']==result['plan_inventory'] and a['run_inventories']==[b['inventory'] for b in result['blocks']],'independent audit differs')
    out=output(output_dir,(experiment,*runs,audit));write(out/'experiment3_verification.json',dict(status='PASS',audit_inventory=receipt));seal(out);return dict(status='PASS')
