"""Read-only I5 NATURAL import; one explicitly selected D5 block, never a sweep."""
from collections import Counter
from pathlib import Path
import os,sys,subprocess,unittest
from .communication import require,digest,canonical,header,MAX_HEADER,payload_size
from .r3a_artifacts import ROOT,THRESHOLDS,sha
from .r5_i5r2_tensors import Evidence
from .r5_i5r1_plan import source_identity
from .r5_i5r_parallel import grams,lcp
from .r5_multi_artifacts import sharing,validate,resolve
from .r5_subsequence_natural_io import output,put,seal,verify
from .r5_subsequence_integrity import SCOPED,THREAT_MODEL
from .r5_subsequence_physical import EXACT

CONDITIONS=('FULL_RECOMPUTE','SEMCACHE_D5_NATURAL')
SCHEMA='D5_NATURAL_ONE_BLOCK_V1'
UNRESOLVED='GPU_DEPENDENT_NOT_EVALUATED'
FILES=('scripts/153_r5_subsequence_natural.py','jobs/r5_subsequence_natural.sh','tests/test_r5_subsequence_natural.py')

def implementation():
    src=source_identity()
    for name in FILES:
        r=subprocess.run(['git','show',src['commit']+':'+name],cwd=ROOT,capture_output=True)
        require(r.returncode==0 and __import__('hashlib').sha256(r.stdout).hexdigest()==sha(ROOT/name),'commit reviewed NATURAL CLI/job/tests before prepare')
    return dict(source=src,files={n:sha(ROOT/n) for n in FILES})

def check_block(block):
    copy=dict(block);bid=copy.pop('block_id');require(digest(copy)==bid,'original I5 block identity differs')
    ts=block['tasks'];require(block['track']=='NATURAL' and block['priming_requests']==0 and 1<=len(ts)<=20,'original NATURAL block required')
    require(len({t['request_id'] for t in ts})==len(ts) and len({t['sample_id'] for t in ts})==len(ts),'duplicate NATURAL request/sample')
    require([t['request_sequence_number'] for t in ts]==list(range(len(ts))),'NATURAL request ordering differs')
    for t in ts:
        require(t['scored'] is True and t['role']=='SCORED_TARGET' and t['dataset']==block['dataset'],'priming/non-natural task forbidden')
        require(3<=len(t['token_ids'])<=512 and len(t['token_ids'])+t['policy']['max_new_tokens']<=2048,'unchanged512-token NATURAL resource profile required')
        require(1<=t['policy']['max_new_tokens']<=96,'original generation resource bound exceeded')
    return ts

def feasibility(block,registry):
    ts=check_block(block);pairs=[];requests=[]
    for i,t in enumerate(ts):
        resolve(registry,t['user_id'],t['adapter_id']);tg=grams(t['token_ids']);eligible=set();raw_positions=set()
        for s in ts[:i]:
            sg=grams(s['token_ids']);common=sg.keys()&tg.keys();prefix=lcp(s['token_ids'],t['token_ids'])
            auth=sharing(registry,s['user_id'],t['user_id'],s['adapter_id'],t['adapter_id'])
            counts=Counter(NO_EXACT_W3=len(tg.keys()-common));raw=sum(len(sg[g])*len(tg[g]) for g in common)
            for g in common:
                for p in tg[g]:
                    raw_positions.add(p)
                    same=p in sg[g]
                    if not same:counts['EXACT_POSITION_MISMATCH']+=1
                    if p+3>prefix:counts['EXACT_CAUSAL_CONTEXT_MISMATCH']+=1
                    if auth:counts[auth]+=1
                    if same and p+3<=prefix and auth is None:eligible.add(p)
            pairs.append(dict(source_request_id=s['request_id'],target_request_id=t['request_id'],source_adapter=s['adapter_id'],target_adapter=t['adapter_id'],
                unique_matching_w3=len(common),matching_occurrence_pairs=raw,common_prefix_tokens=prefix,diagnostic_rejections=dict(counts),
                source_shorter_requirement='NOT_APPLICABLE_TO_D5_SUBSEQUENCE',semantic_cluster_assignment=UNRESOLVED))
        chosen=[];end=-1
        for p in sorted(eligible):
            if p>=end and len(chosen)<64:chosen.append([p,p+3]);end=p+3
        requests.append(dict(request_id=t['request_id'],sample_id=t['sample_id'],user_id=t['user_id'],adapter_id=t['adapter_id'],input_tokens=len(t['token_ids']),generation_limit=t['policy']['max_new_tokens'],
            raw_matching_target_windows=len(raw_positions),candidate_limit=64,overlap_or_limit_excluded_static_windows=len(eligible)-len(chosen),static_exact_context_windows=len(eligible),optimistic_nonoverlap_windows=chosen,
            potential_token_positions=sorted({p for a,b in chosen for p in range(a,b)}),
            residency='UNKNOWN: actual Impact/F/A/I/S admission, duplicate-key ownership, eviction and candidate ranking may remove opportunities',
            per_layer_input_equivalence=UNRESOLVED,physical_hit='NOT_EVALUATED'))
    entry=3*3*32*2560*2
    return dict(block_id=block['block_id'],dataset=block['dataset'],requests=requests,pairs=pairs,request_count=len(ts),
        potential_w3_occurrence_pairs=sum(p['matching_occurrence_pairs'] for p in pairs),static_exact_context_windows=sum(r['static_exact_context_windows'] for r in requests),
        scope='STATIC_OPTIMISTIC_UPPER_BOUND; all previous requests assumed completed; no semantic encoding, policy simulation or physical grant',
        independent_contexts=len({digest(t['prepared']['sample']['context']) for t in ts}),
        capacity=dict(per_adapter_bytes=512*2**20,max_entries=64,actual_all_layer_qkv_bytes_per_w3_entry=entry,max_64_entries_bytes=64*entry,entry_fits=entry<=512*2**20),
        policy='Original D5 F/A/I/S/CHU; PBR every100 adapter requests, therefore not triggered within <=20-request blocks',
        semantic_candidates=UNRESOLVED,actual_admissions=UNRESOLVED,actual_evictions=UNRESOLVED,actual_gpu_physical_hits='NOT_EVALUATED')

def resources(tasks):
    n=max(len(t['token_ids']) for t in tasks);total=max(len(t['token_ids'])+t['policy']['max_new_tokens'] for t in tasks)
    probability=32*n*n*2;working=probability+32*n*n*8+32*3*n*8
    # Dense worst case: input/output/base FP16 + delta FP32, all32x3 roles;
    # C input sharing saves bytes but is NOT assumed. Include headers+ledger reserve.
    by_adapter=Counter()
    for t in tasks:
        rows=len(t['token_ids'])+t['policy']['max_new_tokens']-1
        by_adapter[t['adapter_id']]+=rows*32*2560*44+32*32*len(t['token_ids'])**2*2+t['policy']['max_new_tokens']*(50272*4+4*2**20)+64*2**20
    peak=5400000000+3*512*2**20+4*2**30+2*32*2560*total*2+n*491520+working+n*50272*4
    forecast=sum(by_adapter.values())
    file_forecast=2*sum(t['policy']['max_new_tokens']*(32*14+4)+64 for t in tasks)+2048
    require(file_forecast<=500000,'NATURAL file inventory forecast exceeds500000')
    require(probability<=128*2**20 and working<=2**30 and peak<22*2**30,'NATURAL GPU/impact resource estimate exceeded')
    require(all(v<16*2**30 for v in by_adapter.values()) and forecast<32*2**30,'NATURAL strict evidence forecast exceeded; no automatic truncation/drop')
    return dict(max_input_tokens=n,max_context_tokens=total,max_forwards=sum(t['policy']['max_new_tokens'] for t in tasks),
        probability_layer_bytes=probability,impact_layer_working_estimate_bytes=working,gpu_estimate_bytes=peak,gpu_limit_bytes=22*2**30,
        estimated_run_file_count=file_forecast,maximum_run_files=500000,condition_evidence_forecast_bytes=forecast,per_adapter_forecast_bytes=dict(by_adapter),per_adapter_es_budget_bytes=16*2**30,per_adapter_ud_budget_bytes=4*2**30,
        condition_limit_bytes=32*2**30,run_limit_bytes=64*2**30,required_free_disk_bytes=96*2**30,host_available_required_bytes=32*2**30,
        worker_timeout_seconds=7200,total_run_timeout_seconds=14460,allocation='04:30:00',warmup_requests=0,
        timing='uniform strict value evidence and wall-only profiler; diagnostic observer effect, not production performance')

def transport(tasks,pm,specs,tok,registry):
    from .r5_i5r3b_transport import predicted_request
    from .r5_i5_limits import wire_limits
    from .r5_sparse_qkv import envelope
    rows=[]
    for t in tasks:
        req=predicted_request(t,pm,specs[t['adapter_id']],tok)
        wire_limits(req,registry,'0'*64,2048)
        sizes=[]
        for step in range(t['policy']['max_new_tokens']):
            history=[50271]*step
            for positions in ((list(range(len(req['token_ids']))) if not step else [0]),[]):
                c=envelope(req,31,positions,req['request_id']+f'_s{step:03d}',history)
                h=header('HIDDEN' if positions else 'HELLO',c['pass_id'],31 if positions else -1,req['physical_identity']['dtype'] if positions else 'none',[1,len(positions),2560] if positions else [],sparse=c)
                h.update(session_id='0'*32,adapter_sha256='0'*64,sequence=999999,payload_bytes=payload_size(h['dtype'],h['shape']),payload_sha256='0'*64)
                sizes.append(len(canonical(h)))
        # Completion includes the full unchanged request and generated history.
        control=dict(request=req,generated_token_ids=[50271]*t['policy']['max_new_tokens'],termination='LENGTH_CAP')
        h=header('HELLO',req['request_id']+'_COMPLETE',-1,sparse_complete=control)
        h.update(session_id='0'*32,adapter_sha256='0'*64,sequence=999999,payload_bytes=0,payload_sha256='0'*64);sizes.append(len(canonical(h)))
        maximum=max(sizes);require(maximum+8192<=MAX_HEADER,'NATURAL sparse header plus8192 headroom exceeds65536: '+t['request_id'])
        rows.append(dict(request_id=t['request_id'],maximum_header_bytes=maximum,headroom_bytes=MAX_HEADER-maximum))
    return dict(rows=rows,header_limit=MAX_HEADER,required_headroom_bytes=8192,scope='CPU predicted; cold/all-fresh, all-reused controls, decode and completion; no fake grants')

def inputs(original_experiment,multi,*,_datasets=None):
    ev=Evidence(original_experiment);receipt=ev.verify();w=ev.json('i5_workload_manifest.json');d=ev.json('i5_dataset_manifest.json')
    c=dict(w);eid=c.pop('experiment_id');require(digest(c)==eid and not w['TEST_FIXTURE_ONLY'] and w['clusters']==1,'original real I5 plan differs')
    m,pm,specs,users=validate(multi);require(not m['TEST_FIXTURE_ONLY'] and len(users['users'])==30 and set(specs)=={'Adapter_A','Adapter_B','Adapter_C'},'real30-user/3-adapter assets required')
    from .r4c_prepare import opt_tokenizer
    from .r5_i5r3a import provenance
    from .r5_i5_workload import routing,TEMPLATE
    tok=opt_tokenizer(pm);require(tok.identity==w['tokenizer_identity']==m['tokenizer'] and tok.supported_context_length==2048 and w['template_version']==TEMPLATE,'original model/tokenizer/template differs')
    inventories={};samples={}
    for name,record in d.items():
        require(name in (('qasper','2wikimqa','multi_news') if _datasets is None else _datasets),'original dataset unsupported')
        rows,manifest=provenance(record['prepared_path'],name);de=Evidence(record['prepared_path']);inventories[name]=de.verify()
        require(sha(Path(record['prepared_path'])/'i5_output_manifest.json')==record['prepared_manifest_sha256'] and all(manifest[k]==record[k] for k in manifest),'original dataset manifest differs')
        samples[name]={s['id']:s for s in rows}
    for b in w['blocks']:
        for t in check_block(b):
            s=samples[b['dataset']][t['sample_id']]
            require(t['prepared']['sample']==s and t['group_id']==s['group_id'] and all(t[k]==v for k,v in routing(s,b['routing_seed']).items()),'original sample/routing differs')
            require(t['token_ids']==t['prepared']['token_ids']==list(tok.tokenizer.encode(t['input_text'],add_special_tokens=False)) and t['input_text']==t['prepared']['input_text'] and t['policy']['max_new_tokens']==t['prepared']['max_new_tokens'],'original prompt/policy differs')
            resolve(users,t['user_id'],t['adapter_id'])
    return w,d,receipt,inventories,m,pm,specs,users,tok

def prepare(original_experiment,multi,source_commit,block_index,output_dir):
    impl=implementation();require(impl['source']['commit']==source_commit,'reviewed new commit required')
    w,d,receipt,inventories,m,pm,specs,users,tok=inputs(original_experiment,multi)
    require(0<=block_index<len(w['blocks']),'selected block index invalid')
    block=w['blocks'][block_index];tasks=block['tasks'];res=resources(tasks);network=transport(tasks,pm,specs,tok.identity,users)
    plan=dict(schema=SCHEMA,implementation=impl,original_experiment=str(Path(original_experiment).resolve()),original_inventory=receipt,dataset_inventories=inventories,
        multi=str(Path(multi).resolve()),multi_inventory=Evidence(multi).verify('r5_output_manifest.json'),base_prepared=m['base_prepared'],
        block_index=block_index,block_id=block['block_id'],tasks_sha256=digest(tasks),tokenizer_identity=tok.identity,conditions=list(CONDITIONS),
        condition_order=[c for c in (('SEMCACHE_D5_NATURAL','FULL_RECOMPUTE') if block['repetition']%2 else CONDITIONS)],
        validation_mode=SCOPED,context_mode=EXACT,allow_approximate=False,thresholds=THRESHOLDS,resources=res,transport=network,
        phase3='PENDING_EXPLICIT_USER_APPROVAL; no full-workload GPU action',historical_results='immutable separate no-HIT exact-prefix reference; not matched timing',warmup_requests=0)
    plan['plan_id']=digest(plan)
    out=output(output_dir,(original_experiment,multi,*[x['prepared_path'] for x in d.values()]));put(out/'natural_plan.json',plan)
    reports=[feasibility(b,users) for b in w['blocks']]
    put(out/'natural_feasibility.json',dict(scope='STATIC_ONLY_GPU_NOT_RUN',blocks=reports))
    lines=['# D5 NATURAL static feasibility','','No GPU physical HITs measured. Counts are optimistic static opportunities, before actual semantic assignment/admission/residency/input-byte checks.','','| Original block | Dataset | Requests | Raw w3 occurrence pairs | Static exact-context windows |','|---:|---|---:|---:|---:|']
    for i,r in enumerate(reports):lines.append(f"| {i} | {r['dataset']} | {r['request_count']} | {r['potential_w3_occurrence_pairs']} | {r['static_exact_context_windows']} |")
    lines+=['','Each block preserves its original routing/order and starts empty. Per-request positions, overlapping rejection categories, independent context counts and capacity accounting are in natural_feasibility.json. PBR interval100 cannot trigger in <=20-request blocks. No policy admission, semantic encoding, tensor reuse or numerical fidelity has been measured. Phase3 remains pending.']
    (out/'natural_feasibility.md').write_text('\n'.join(lines)+'\n')
    seal(out);return dict(status='CPU_PREPARED_GPU_NOT_RUN',plan_id=plan['plan_id'],block_index=block_index,resources=res,transport=network)

def load(experiment):
    receipt=verify(experiment);p=__import__('json').loads((Path(experiment)/'natural_plan.json').read_bytes());copy=dict(p);pid=copy.pop('plan_id')
    require(p['schema']==SCHEMA and digest(copy)==pid and p['implementation']==implementation(),'pinned NATURAL source/plan differs')
    require(p['conditions']==list(CONDITIONS) and p['validation_mode']==SCOPED and p['context_mode']==EXACT and p['allow_approximate'] is False and p['thresholds']==THRESHOLDS,'sealed safety configuration differs')
    w,d,rr,inv,m,pm,specs,users,tok=inputs(p['original_experiment'],p['multi']);b=w['blocks'][p['block_index']]
    require(rr==p['original_inventory'] and inv==p['dataset_inventories'] and Evidence(p['multi']).verify('r5_output_manifest.json')==p['multi_inventory'],'input inventories differ')
    expected_order=list(('SEMCACHE_D5_NATURAL','FULL_RECOMPUTE') if b['repetition']%2 else CONDITIONS)
    require(p['condition_order']==expected_order and p['warmup_requests']==0 and p['base_prepared']==m['base_prepared'] and p['tokenizer_identity']==tok.identity and p['phase3']=='PENDING_EXPLICIT_USER_APPROVAL; no full-workload GPU action','sealed execution scope/order differs')
    require(b['block_id']==p['block_id'] and digest(b['tasks'])==p['tasks_sha256'] and p['resources']==resources(b['tasks']) and p['transport']==transport(b['tasks'],pm,specs,tok.identity,users),'original task/resource reconstruction differs')
    feasibility_record=__import__('json').loads((Path(experiment)/'natural_feasibility.json').read_bytes())
    require(feasibility_record==dict(scope='STATIC_ONLY_GPU_NOT_RUN',blocks=[feasibility(x,users) for x in w['blocks']]),'static report differs')
    return p,receipt,b,m,pm,specs,users,tok

def cpu_validate(experiment,output_dir):
    p,pr,*_=load(experiment)
    import torch,transformers,numpy
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_initialized(),'hide CUDA for CPU validation')
    sys.path.insert(0,str(ROOT/'tests'));suite=unittest.defaultTestLoader.loadTestsFromNames(['test_r5_subsequence_natural','test_r5_subsequence_generation','test_r5_subsequence_validation','test_r5_subsequence_diagnostics'])
    result=unittest.TestResult();suite.run(result);failed=len(result.errors)+len(result.failures);skipped=len(result.skipped)
    require(not failed and not skipped and not torch.cuda.is_initialized(),'native CPU validation failed/skipped: '+str(result.errors+result.failures+result.skipped))
    out=output(output_dir,(experiment,));record=dict(status='PASS',plan_inventory=pr,implementation=p['implementation'],tests=result.testsRun,failed=failed,skipped=skipped)
    put(out/'cpu_validation.json',record);seal(out);return record

def approval(experiment,cpu_audit):
    p,pr,*rest=load(experiment);ar=verify(cpu_audit);a=__import__('json').loads((Path(cpu_audit)/'cpu_validation.json').read_bytes())
    require(a['status']=='PASS' and a['failed']==a['skipped']==0 and a['implementation']==p['implementation'] and a['plan_inventory']==pr,'CPU approval differs')
    return p,pr,rest,ar

def inspect(experiment,cpu_audit):
    p,pr,rest,ar=approval(experiment,cpu_audit)
    return dict(status='READY_FOR_MANUAL_ONE_BLOCK_ONLY_GPU_NOT_RUN',plan_id=p['plan_id'],block_id=p['block_id'],resources=p['resources'],transport=p['transport'],CPU_audit=ar,phase3=p['phase3'])


from .r5_subsequence_profile import Profiler
class NaturalProfiler(Profiler):
    """All wall calls, all prefill eligibility; repeated decode layouts summarized.

    Per-step actual input tensors and wire shapes remain fully saved/audited.
    No live arithmetic or timing scope changes.
    """
    def __init__(self,**kwargs):
        super().__init__(**kwargs);self.decode_layout_counts={};self.decode_layouts={}
    def detail(self,record):
        if record.get('boundary')=='native_projection_layout' and record.get('phase')=='DECODE':
            key=str(record['layer'])+':'+record['role'];self.decode_layout_counts[key]=self.decode_layout_counts.get(key,0)+1
            if key in self.decode_layouts:
                require(record==self.decode_layouts[key],'decode projection layout changed; no silent sampling');return
            self.decode_layouts[key]=record
        super().detail(record)
    def finish(self):
        return dict(super().finish(),decode_layout_counts=self.decode_layout_counts,layout_coverage='every prefill layout; one verified invariant decode layout per layer/role plus exact count; full per-step tensors saved')
