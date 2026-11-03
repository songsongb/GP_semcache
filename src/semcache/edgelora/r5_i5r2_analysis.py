"""Historical I5R-1 numerical review, independent of the current I5 source inventory.

This is new CPU tensor analysis over sealed data and historical audit exports,
not a fresh execution of the historical GPU independent analyzer.
"""
from collections import defaultdict
import hashlib
import math
from pathlib import Path
import re
import subprocess

from .communication import digest,require
from .r3a_artifacts import ROOT,THRESHOLDS,check_parity,unpack,write
from .r4b2_analysis import raw_head_bytes
from .r5_i5_contracts import validate_request
from .r5_i5_protocol import context,validate_context
from .r5_multi_artifacts import sharing,validate_registry
from .r5_i5r2_tensors import Evidence,UNKNOWN,DTYPES,align,compare,metrics,slice_axis,first,verdict,file_sha,safe_path

CONDITIONS=('FULL_RECOMPUTE','SEMANTIC_NO_REUSE','SEMCACHE_INTEGRATED')


def unique(rows,key):
    result={}
    for row in rows:
        k=key(row);require(k not in result,'duplicate evidence identity: '+str(k));result[k]=row
    return result


def locate(run_dir=None,runs_root=None):
    require((run_dir is None)!=(runs_root is None),'select --run-dir or --runs-root')
    if run_dir is not None:
        require(not Path(run_dir).is_symlink(),'unsafe block symlink');return Path(run_dir).resolve()
    root=Path(runs_root);require(root.is_dir() and not root.is_symlink(),'historical runs-root is not available/unsafe: '+str(root))
    candidates=[p for p in root.iterdir() if re.fullmatch(r'block_\d+_\d+',p.name)]
    require(len(candidates)==1,'missing/ambiguous original blocks; select exactly one completed --run-dir')
    require(not candidates[0].is_symlink(),'unsafe block symlink');return candidates[0].resolve()


def source_provenance(config,*,expected_commit,fixture,allow_unavailable=False):
    commit=config['source_commit'];inventory=config['sources']
    require(fixture or expected_commit is not None,'real historical analysis requires --expected-source-commit')
    require(expected_commit is None or commit==expected_commit,'recorded historical source commit mismatch')
    require(fixture or re.fullmatch('[0-9a-f]{40}',commit or '') and inventory,'missing pinned historical source inventory')
    available=bool(commit and subprocess.run(['git','cat-file','-e',commit+'^{commit}'],cwd=ROOT,capture_output=True).returncode==0)
    checks={}
    if available:
        for name,expected in sorted(inventory.items()):
            safe_path(ROOT,name)
            blob=subprocess.run(['git','show',commit+':'+name],cwd=ROOT,capture_output=True)
            checks[name]='PASS' if blob.returncode==0 and hashlib.sha256(blob.stdout).hexdigest()==expected else 'FAIL'
        require(checks and all(v=='PASS' for v in checks.values()),'historical source SHA-256 mismatch or missing Git blob')
    else:require(fixture or allow_unavailable,'historical Git object unavailable; use the recorded checkout/object, or explicitly request partial provenance with --allow-unavailable-historical-git')
    current=[Path(__file__),ROOT/'src/semcache/edgelora/r5_i5r2_tensors.py',ROOT/'scripts/129_analyze_r5_i5r2.py']
    return dict(recorded_source_commit=commit,recorded_source_files=inventory,historical_git_object_available=available,blob_checks=checks,
        historical_source_verification='ALL_RECORDED_BLOBS_MATCH' if available else 'TEST_FIXTURE_ONLY' if fixture else 'HISTORICAL_OBJECT_UNAVAILABLE',
        current_diagnostic=dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip(),
            tracked_status=subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True),
            files={str(p.relative_to(ROOT)):file_sha(p) for p in current if p.exists()}),
        verification_scope='Historical Git blobs and archived audit exports plus new read-only tensor comparisons. Current sources are never substituted for historical sources.',
        fresh_original_source_GPU_audit_performed=False)


def read_run(root,*,experiment=None,audit_dir=None,r1_audit_dir=None,expected_commit=None,fixture=False,allow_unavailable=False,
             max_tensor_bytes=32*2**20,max_input_bytes=32*2**30):
    evidence=Evidence(root,max_tensor_bytes=max_tensor_bytes,max_input_bytes=max_input_bytes);seal=evidence.verify()
    require(evidence.json('i5_block_status.json')['status']=='COMPLETE','incomplete/failed historical block')
    cfg=evidence.json('i5_execution_config.json');require(cfg['fixture']==fixture,'explicit real/fixture scope mismatch')
    provenance=source_provenance(cfg,expected_commit=expected_commit,fixture=fixture,allow_unavailable=allow_unavailable)
    block=cfg['block'];b=dict(block);bid=b.pop('block_id');require(digest(b)==bid,'historical block identity differs')
    require(block['track']=='CONTROLLED','R2 requires a controlled exact-prefix block')
    tasks=unique(block['tasks'],lambda t:t['request_id']);sources=[t for t in tasks.values() if not t['scored']]
    require(len(sources)==1 and sources[0]['role']=='PRIMING_SOURCE','one genuine standalone source required')
    plan_root=Path(experiment or cfg['experiment_prepared']);pe=Evidence(plan_root,max_input_bytes=128*2**20);plan_seal=pe.verify()
    require(plan_seal['manifest_sha256']==cfg['experiment_manifest_sha256'],'historical experiment inventory identity differs')
    plan=pe.json('i5_workload_manifest.json');p=dict(plan);eid=p.pop('experiment_id')
    require(digest(p)==eid==cfg['experiment_id'] and block in plan['blocks'],'historical experiment/block chain differs')
    require(plan['i5r1']['source']['commit']==cfg['source_commit'] and plan['i5r1']['source']['files']==cfg['sources'],'historical plan source snapshot differs')
    ae=Evidence(audit_dir or Path(root).with_name(Path(root).name+'_audit'),max_input_bytes=max_input_bytes)
    audit_seal=ae.verify();audit=ae.json('i5_independent_audit.json')
    require(audit['status']=='PASS' and audit['TEST_FIXTURE_ONLY']==fixture and ae.json('i5_execution_config.json')==cfg,'historical exported audit/configuration differs')
    old_numerical=ae.json('i5_numerical_fidelity.json')
    require(old_numerical['thresholds']==THRESHOLDS,'historical frozen thresholds differ from diagnostic thresholds')
    r1_path=Path(r1_audit_dir or Path(root).with_name(Path(root).name+'_i5r1_audit'));r1_export=dict(status='NOT_AVAILABLE')
    if r1_audit_dir is not None or r1_path.exists():
        r1e=Evidence(r1_path,max_input_bytes=max_input_bytes);r1_seal=r1e.verify();r1_audit=r1e.json('i5r1_independent_audit.json')
        r1_decision=r1e.json('i5r1_decision.json')
        require(r1_audit['status']=='PASS' and r1_audit['TEST_FIXTURE_ONLY']==fixture and r1_decision['experiment_id']==cfg['experiment_id'],
                'historical R1 audit identity/scope differs')
        matching=[b for b in r1_audit['blocks'] if b['block_id']==block['block_id']]
        require(len(matching)==1 and matching[0]['source_commit']==cfg['source_commit'],'historical R1 block source differs')
        r1_export=dict(status='HISTORICAL_EXPORT_VERIFIED',inventory=r1_seal,decision=r1_decision,
            last_position_layer_diagnostics=r1e.json('i5r1_layer_diagnostics.json'),scope='Previous last-prefill-position captures only, not all-token causal localization')
    registry=evidence.json('i5_adapter_registry.json');users=validate_registry(registry['registry'],registry['adapters'])
    rows={};steps={};requests={};extra={};settings={};cpu_receipts={}
    for c in CONDITIONS:
        cp=evidence.json(c+'/plan.json');result=evidence.json(c+'/result.json');life=evidence.json(c+'/lifecycle.json')
        require(cp['sources']==cfg['sources'] and cp['source_commit']==cfg['source_commit'] and cp['block']==block and cp['condition']==c,'mixed historical execution snapshot/configuration')
        require(result['status']=='COMPLETE' and result['gpu_inference_performed']==(not fixture) and life['returncode']==0 and not life['pipe_errors'] and not life['forced_termination'],'condition execution incomplete/scope differs')
        rs=unique(evidence.jsonl(c+'/generation.jsonl'),lambda r:r['request_id']);require(list(rs)==list(tasks),'missing/reordered historical requests')
        arrivals=[x for x in evidence.jsonl(c+'/requests.jsonl') if x['kind']=='ARRIVAL']
        rq=unique([validate_request(x['request']) for x in arrivals],lambda r:r['request_id'])
        require(list(rq)==list(tasks),'request arrival coverage/order differs')
        events=evidence.jsonl(c+'/events.jsonl',optional=c==CONDITIONS[0])
        if c==CONDITIONS[0]:require(not events,'FULL_RECOMPUTE contains semantic/cache events')
        for aid in registry['adapters']:
            es=c+'/'+aid+'/es';identity=evidence.json(c+'/'+aid+'/identity.json')
            settings[c,aid]=dict(physical_identity=identity['physical_identity'],contract=identity['contract'],gpu=identity['gpu'])
            ud=c+'/'+aid+'/ud';cpu_result=evidence.json(ud+'/result.json') if ud+'/result.json' in evidence.files else None
            if cpu_result is not None:require(not cpu_result['user_device_cuda_used'],'historical UD CUDA isolation differs')
            receipts=evidence.jsonl(ud+'/layers.jsonl',optional=True)
            cpu_receipts[c,aid]=dict(receipt_count=len(receipts),result=cpu_result or UNKNOWN,
                actual_precision_records=[dict(forward_id=r['forward_id'],layer=r['layer'],precision=r['metrics'].get('actual_precision',UNKNOWN),
                    hidden_payload_sha256=r['metrics']['hidden_payload_sha256']) for r in receipts],
                value_evidence='Hashes/precision receipts are not numerical tensor values; no CPU delta array is inferred')
            for s in evidence.jsonl(es+'/r5_tokens.jsonl'):
                if s['request_id'] in tasks:
                    key=(c,s['request_id']);steps.setdefault(key,[]).append(s)
            for s in evidence.jsonl(es+'/fidelity_steps.jsonl',optional=True):
                key=(c,s['request_id'],s['step']);require(key not in extra,'duplicate optional forensic capture');extra[key]=s
        for rid,row in rs.items():
            t=tasks[rid];req=rq[rid];seq=steps.get((c,rid),[])
            require(req['TEST_FIXTURE_ONLY']==fixture,'request real/fixture scope differs')
            require(row['status']=='COMPLETE' and row['adapter_id']==t['adapter_id'] and row['user_id']==t['user_id'] and row['scored']==t['scored'],'historical request identity/completion differs')
            require(req['token_ids']==t['token_ids'] and req['input_text']==t['input_text'] and req['generation_policy']==t['policy'] and req['request_sha256']==row['request_sha256'],'historical prepared/actual request differs')
            require(row['physical_identity']==req['physical_identity']==settings[c,t['adapter_id']]['physical_identity'],'request physical identity differs')
            require(row['matched_identity_sha256']==digest(dict(task=t,adapter_sha256=registry['adapters'][t['adapter_id']]['matrix_sha256'],tokenizer=row['physical_identity']['tokenizer'])),
                    'matched immutable request hash differs')
            require(seq and len(seq)==row['generated_tokens']==len(row['generated_token_ids']) and row['generated_tokens']>0,'missing/empty generation evidence')
            for i,s in enumerate(seq):
                tokens=t['token_ids'][row['reused_prefix_tokens']:] if i==0 else [row['generated_token_ids'][i-1]]
                old=row['reused_prefix_tokens'] if i==0 else len(t['token_ids'])+i-1
                expected=context(rid,i,tokens,t['token_ids'],old,t['policy'],row['mode'],s['context']['cohort_sha256'],row['source'],context_limit=s['context']['context_limit'])
                require(s['step']==i and s['context']==expected,'token/mask/position/prefill-decode context differs')
                require(s['position_observation']['positions']==expected['positions'] and s['position_observation']['past_length']==old,'actual position observation differs')
                ident=row['physical_identity'];heads=settings[c,t['adapter_id']]['contract'].get('heads',s['kv'][0]['k']['shape'][1])
                require(ident['hidden_size']%heads==0,'invalid historical head layout')
                require(all(x['shape']==[1,len(tokens),ident['hidden_size']] and x['dtype']==ident['dtype'] and x['device']==ident['device'] for x in s['qkv'].values()),
                        'actual QKV shape/dtype/device differs from executed input rows')
                require(all(x[role]['shape']==[1,heads,old+len(tokens),ident['hidden_size']//heads] and
                    x[role]['dtype']==ident['dtype'] and x[role]['device']==ident['device'] for x in s['kv'] for role in 'kv'),'actual KV shape/dtype/device differs')
                es=c+'/'+t['adapter_id']+'/es'
                for spec in (*s['qkv'].values(),*s['components'].values(),*(x[role] for x in s['kv'] for role in 'kv')):
                    evidence.validate_tensor_reference(es,spec)
                if (c,rid,i) in extra:
                    require(extra[c,rid,i]['context']==s['context'],'optional forensic context differs')
                    for spec in extra[c,rid,i]['tensors'].values():evidence.validate_tensor_reference(es,spec)
            require(len(seq[0]['qkv'])==3*row['physical_identity']['layers'] and len(seq[0]['kv'])==row['physical_identity']['layers'],'partial prefill QKV/KV layer coverage')
            if row['source']:
                source_id=row['source']['source_request_id'];require(source_id in rq and rs[source_id]['completion_ns']<row['arrival_ns'],'source incomplete or missing')
                require(sharing(users,rq[source_id]['user_id'],req['user_id'],rq[source_id]['adapter_id'],req['adapter_id']) is None,'unauthorized physical source')
            for spec in row.get('strict_cached_qkv',{}).values():evidence.validate_tensor_reference(c+'/'+t['adapter_id']+'/es',spec)
            rows[c,rid]=row;requests[c,rid]=req
    for rid in tasks:
        require(len({rows[c,rid]['matched_identity_sha256'] for c in CONDITIONS})==1,'unmatched A/B/C request')
        require(all(rows[c,rid]['physical_identity']==rows[CONDITIONS[0],rid]['physical_identity'] for c in CONDITIONS),'A/B/C execution GPU/model/adapter differs')
    return dict(evidence=evidence,config=cfg,block=block,tasks=tasks,source_id=sources[0]['request_id'],rows=rows,steps=steps,requests=requests,
        extra=extra,settings=settings,cpu_receipts=cpu_receipts,provenance=provenance,old_numerical=old_numerical,exported_audit=audit,r1_export=r1_export,
        inventory=dict(raw=seal,experiment=plan_seal,exported_audit=audit_seal,verification_level='HISTORICAL_EXPORT_AUDIT_PLUS_NEW_CPU_TENSOR_COMPARISONS',
            historical_R1_export_status=r1_export['status'],TEST_FIXTURE_ONLY=fixture,fresh_original_source_independent_audit=False))


def analyze(data,layers=None):
    ev=data['evidence'];rows=data['rows'];steps=data['steps'];sid=data['source_id'];tasks=data['tasks'];records=[];missing=[];hashes=[]
    a,b,c=CONDITIONS;source=steps[c,sid][0];source_row=rows[c,sid];p=len(tasks[sid]['token_ids'])
    count=source_row['physical_identity']['layers'];layers=list(range(count)) if layers is None else list(layers)
    require(layers and len(set(layers))==len(layers) and all(type(l) is int and 0<=l<count for l in layers),'invalid/duplicate selected layers')
    def base(condition,rid):return condition+'/'+rows[condition,rid]['adapter_id']+'/es'
    def add(family,rid,layer,role,x,xb,y,yb,*,step=0,**kw):
        absent=[]
        for name,spec,root in (('reference',x,xb),('actual',y,yb)):
            if not spec or 'file' not in spec:absent.append(dict(side=name,base=root,tensor=role,reason='MISSING_CAPTURE' if not spec else 'HASH_ONLY_NO_TENSOR_BYTES'))
        if absent:
            missing.append(dict(family=family,request_id=rid,layer=layer,role=role,step=step,missing=absent));return None
        result=dict(family=family,request_id=rid,layer=layer,role=role,step=step,metrics=compare(ev,x,xb,y,yb,**kw));records.append(result);return result
    for target in [t for t in tasks.values() if t['scored']]:
        rid=target['request_id'];full=steps[a,rid][0];hit=steps[c,rid][0];shadow=steps[b,rid][0]
        require(rows[c,rid]['source'] is not None and rows[c,rid]['source']['source_request_id']==sid and rows[c,rid]['reused_prefix_tokens']==p,
                'historical target has no compatible physical source grant')
        prefix_ranges=align(full['context'],source['context'],relation='FULL_PREFIX_VS_SOURCE',prefix=p)
        align(full['context'],steps[a,sid][0]['context'],relation='FULL_PREFIX_VS_SOURCE',prefix=p)
        suffix_ranges=align(full['context'],hit['context'],relation='FULL_SUFFIX_VS_HIT',prefix=p)
        align(full['context'],shadow['context'],relation='MATCHED',prefix=0)
        for l in layers:
            for role in 'qkv':
                key=f'{l}.{role}';src=source['qkv'][key];ref=full['qkv'][key];actual=hit['qkv'][key]
                add('FULL_PREFIX_QKV_VS_SOURCE',rid,l,role,ref,base(a,rid),src,base(c,sid),range_a=prefix_ranges[0],range_b=prefix_ranges[1])
                add('WITHIN_A_FULL_PREFIX_VS_STANDALONE_QKV',rid,l,role,ref,base(a,rid),steps[a,sid][0]['qkv'][key],base(a,sid),range_a=prefix_ranges[0],range_b=prefix_ranges[1])
                add('FULL_SUFFIX_QKV_VS_HIT',rid,l,role,ref,base(a,rid),actual,base(c,rid),range_a=suffix_ranges[0],range_b=suffix_ranges[1])
                add('A_VS_B_FULL_QKV',rid,l,role,ref,base(a,rid),shadow['qkv'][key],base(b,rid))
                add('A_VS_C_STANDALONE_SOURCE_QKV',rid,l,role,steps[a,sid][0]['qkv'][key],base(a,sid),src,base(c,sid))
                cached=rows[c,rid]['strict_cached_qkv'].get(key)
                add('SOURCE_RAW_VS_CACHED_RAW',rid,l,role,src,base(c,sid),cached,base(c,rid))
                if role=='q':continue
                j='kv'.index(role);source_kv=source['kv'][l][role];full_kv=full['kv'][l][role];hit_kv=hit['kv'][l][role]
                add('FULL_PREFIX_KV_VS_SOURCE_KV',rid,l,role,full_kv,base(a,rid),source_kv,base(c,sid),axis_a=2,axis_b=2,range_a=(0,p),range_b=(0,p))
                add('FULL_PREFIX_KV_VS_HIT_PREFIX',rid,l,role,full_kv,base(a,rid),hit_kv,base(c,rid),axis_a=2,axis_b=2,range_a=(0,p),range_b=(0,p))
                add('SOURCE_KV_VS_HIT_PRESERVED_PREFIX',rid,l,role,source_kv,base(c,sid),hit_kv,base(c,rid),axis_a=2,axis_b=2,range_a=(0,p),range_b=(0,p))
                consumed=hit['input_kv'][l][j]
                hashes.append(dict(family='SOURCE_KV_VS_CONSUMED_INPUT_HASH',request_id=rid,layer=l,role=role,
                    reference_sha256=source_kv['sha256'],actual_sha256=consumed['sha256'],
                    bitwise_hash_identity=all(source_kv[k]==consumed[k] for k in ('sha256','shape','dtype','device')),
                    numerical_metrics=UNKNOWN,stride=UNKNOWN,scope='Recorded consumed-input hash/layout identity; no invented consumed-input tensor bytes'))
                if cached and 'file' in cached and 'file' in source_kv:
                    heads=source_kv['shape'][1];blob=ev.tensor(base(c,rid),cached)
                    transformed=raw_head_bytes(blob,cached['shape'],heads,DTYPES[cached['dtype']][1]);src_bytes=ev.tensor(base(c,sid),source_kv)
                    records.append(dict(family='SOURCE_KV_VS_CACHED_RAW_HEAD_TRANSFORM',request_id=rid,layer=l,role=role,step=0,
                        metrics=dict(coverage='ALL_ALIGNED_ELEMENTS',sampling=False,aligned_shape=source_kv['shape'],dtype=cached['dtype'],
                            transform='view(1,P,heads,dim).transpose(1,2).contiguous() byte permutation; not recomputed model inference',
                            **metrics(src_bytes,transformed,cached['dtype']))))
                else:missing.append(dict(family='SOURCE_KV_VS_CACHED_RAW_HEAD_TRANSFORM',request_id=rid,layer=l,role=role,missing='cached raw or source KV tensor bytes'))
            for role in ('attention','ffn','layer'):
                add('FULL_SUFFIX_COMPONENTS_VS_HIT',rid,l,role,full['components'].get(f'{l}.{role}'),base(a,rid),hit['components'].get(f'{l}.{role}'),base(c,rid),
                    range_a=suffix_ranges[0],range_b=suffix_ranges[1])
                add('FULL_PREFIX_COMPONENTS_VS_SOURCE',rid,l,role,full['components'].get(f'{l}.{role}'),base(a,rid),source['components'].get(f'{l}.{role}'),base(c,sid),
                    range_a=prefix_ranges[0],range_b=prefix_ranges[1])
                add('WITHIN_A_FULL_PREFIX_COMPONENTS',rid,l,role,full['components'].get(f'{l}.{role}'),base(a,rid),steps[a,sid][0]['components'].get(f'{l}.{role}'),base(a,sid),
                    range_a=prefix_ranges[0],range_b=prefix_ranges[1])
            full_extra=data['extra'].get((a,rid,0),{}).get('tensors',{});hit_extra=data['extra'].get((c,rid,0),{}).get('tensors',{})
            source_extra=data['extra'].get((c,sid,0),{}).get('tensors',{})
            for role in ('layer_input','norm1','projection_input','base_q','base_k','base_v','delta_q','delta_k','delta_v',
                         'attention_scores','attention_masked_scores','attention_probabilities','attention_context','attention_residual','norm2','fc1','activation','ffn_residual'):
                name=f'L{l}.{role}';x=full_extra.get(name);y=hit_extra.get(name)
                add('OPTIONAL_FULL_SUFFIX_INTERNALS',rid,l,role,x,base(a,rid),y,base(c,rid),
                    axis_a=x.get('token_axis',1) if x else 1,axis_b=y.get('token_axis',1) if y else 1,
                    range_a=suffix_ranges[0],range_b=suffix_ranges[1])
                z=source_extra.get(name)
                add('OPTIONAL_FULL_PREFIX_INTERNALS',rid,l,role,x,base(a,rid),z,base(c,sid),
                    axis_a=x.get('token_axis',1) if x else 1,axis_b=z.get('token_axis',1) if z else 1,
                    range_a=prefix_ranges[0],range_b=prefix_ranges[1],key_range=(0,p) if x and x.get('key_axis') is not None else None)
        for step in range(1,min(len(steps[a,rid]),len(steps[c,rid]))):
            left,right=steps[a,rid][step],steps[c,rid][step]
            if rows[a,rid]['generated_token_ids'][:step]!=rows[c,rid]['generated_token_ids'][:step]:
                missing.append(dict(family='DECODE_ALIGNMENT',request_id=rid,step=step,reason='UNMATCHED_GENERATION_HISTORY'));continue
            qa,qb=align(left['context'],right['context'],relation='FULL_SUFFIX_VS_HIT',prefix=p,
                history_reference=rows[a,rid]['generated_token_ids'],history_actual=rows[c,rid]['generated_token_ids'])
            for l in layers:
                for role in 'qkv':
                    add('DECODE_QKV_A_VS_C',rid,l,role,left['qkv'][f'{l}.{role}'],base(a,rid),right['qkv'][f'{l}.{role}'],base(c,rid),step=step,range_a=qa,range_b=qb)
                for role in ('attention','ffn','layer'):
                    add('DECODE_COMPONENTS_A_VS_C',rid,l,role,left['components'].get(f'{l}.{role}'),base(a,rid),right['components'].get(f'{l}.{role}'),base(c,rid),step=step,range_a=qa,range_b=qb)
    def status(family):
        rs=[r for r in records if r['family']==family]
        value=verdict(rs)
        return UNKNOWN if value!='FAIL' and any(m['family']==family for m in missing) else value
    numerical=logits(data)
    reconstruction_families=('SOURCE_RAW_VS_CACHED_RAW','SOURCE_KV_VS_CACHED_RAW_HEAD_TRANSFORM','SOURCE_KV_VS_HIT_PRESERVED_PREFIX')
    integrity=[status(f) for f in reconstruction_families]
    reconstruction='FAIL' if 'FAIL' in integrity or any(not h['bitwise_hash_identity'] for h in hashes) else 'PASS' if all(s=='PASS' for s in integrity) and hashes else UNKNOWN
    gates=dict(CACHE_SOURCE_INTEGRITY=status('SOURCE_RAW_VS_CACHED_RAW'),CACHE_RECONSTRUCTION_INTEGRITY=reconstruction,
        FULL_PREFIX_KV_EQUIVALENCE=status('FULL_PREFIX_KV_VS_SOURCE_KV'),ATTENTION_INPUT_EQUIVALENCE=UNKNOWN,
        ATTENTION_OUTPUT_EQUIVALENCE=verdict([r for r in records if r['family']=='FULL_SUFFIX_COMPONENTS_VS_HIT' and r['role']=='attention']),
        PREFILL_LOGIT_FIDELITY=numerical['prefill_verdict'],DECODE_LOGIT_FIDELITY=numerical['decode_verdict'],
        TOKEN_TEXT_FIDELITY=numerical['token_text_verdict'],FROZEN_NUMERICAL_TOLERANCE=numerical['frozen_verdict'],NATIVE_PEFT_STRICT_PARITY='UNESTABLISHED')
    paths={name:first([r for r in records if r['family'] in families]) for name,families in
        {'standalone_source_vs_full_prefix':('FULL_PREFIX_QKV_VS_SOURCE','FULL_PREFIX_COMPONENTS_VS_SOURCE','OPTIONAL_FULL_PREFIX_INTERNALS'),
         'within_full_recompute_source_vs_target_prefix':('WITHIN_A_FULL_PREFIX_VS_STANDALONE_QKV','WITHIN_A_FULL_PREFIX_COMPONENTS'),
         'full_target_vs_physical_hit_suffix':('FULL_SUFFIX_QKV_VS_HIT','FULL_SUFFIX_COMPONENTS_VS_HIT','OPTIONAL_FULL_SUFFIX_INTERNALS'),
         'cache_round_trip':reconstruction_families}.items()}
    paths['decode_full_vs_hit']=first([r for r in records if r['family'] in ('DECODE_QKV_A_VS_C','DECODE_COMPONENTS_A_VS_C')])
    if gates['ATTENTION_OUTPUT_EQUIVALENCE']!='FAIL' and any(m['family']=='FULL_SUFFIX_COMPONENTS_VS_HIT' and m['role']=='attention' for m in missing):
        gates['ATTENTION_OUTPUT_EQUIVALENCE']=UNKNOWN
    hypotheses=dict(H1=dict(status='PREFIX_COMPUTATIONS_DIFFER_BEFORE_CACHE' if status('FULL_PREFIX_QKV_VS_SOURCE')=='FAIL' else 'NO_DIFFERENCE_IN_RECORDED_PREFIX_QKV' if status('FULL_PREFIX_QKV_VS_SOURCE')=='PASS' else 'UNRESOLVED',
        baseline_controls=dict(A_vs_B_full_QKV=status('A_VS_B_FULL_QKV'),A_vs_C_standalone_source_QKV=status('A_VS_C_STANDALONE_SOURCE_QKV'),
            within_A_short_vs_full_prefix_QKV=status('WITHIN_A_FULL_PREFIX_VS_STANDALONE_QKV')),
        possible_cause='FP16 shape dependence remains a hypothesis; layer-input/native-projection/delta captures or controlled identical-input operation tests are needed to isolate it'),
        H2=dict(status='NO_ADDITIONAL_VALUE_ERROR_OBSERVED_IN_RECORDED_ROUND_TRIP' if reconstruction=='PASS' else 'RECORDED_ROUND_TRIP_DIFFERS' if reconstruction=='FAIL' else 'UNRESOLVED',
            limitation='Source integrity is distinct from full-target prefix equivalence. Historical strides are unobserved; consumed-input values may be hash-only.'),
        H3=dict(status='UNRESOLVED',reason='Different or unobserved attention inputs cannot isolate attention arithmetic. Actual scaled Q, K, V, additive mask, score/softmax/BMM operands must be compared before attributing a cause.'),
        H4=dict(status='UNRESOLVED',reason='Historical TCP hashes/precision receipts cannot reconstruct unrecorded CPU correction values. Need matched hidden, CPU input/cast, A intermediate, B output and transported delta.'),
        H5=dict(status='LAYERWISE_OBSERVATIONS_ONLY',reason='Aligned all-token per-layer QKV/attention/FFN/layer differences are reported without assuming propagation. Missing residual/norm boundaries prevent a complete causal chain.'))
    for record in records:
        record.update(phase='DECODE' if record['step'] else 'PREFILL',source_context_tokens=p,target_context_tokens=len(tasks[record['request_id']]['token_ids']))
        stored=record['metrics'].get('first_differing_stored_token_indices')
        if stored:
            offset_a=steps[a,record['request_id']][record['step']]['context']['past_length'] if record['step'] else 0
            offset_b=offset_a if record['step'] else p if record['family'] in ('FULL_SUFFIX_QKV_VS_HIT','FULL_SUFFIX_COMPONENTS_VS_HIT','OPTIONAL_FULL_SUFFIX_INTERNALS') else 0
            record['first_differing_semantic_positions']={side:(stored[side]+offset if stored[side] is not None else None) for side,offset in (('reference',offset_a),('actual',offset_b))}
    return dict(tensor_comparisons=records,hash_only_comparisons=hashes,missing=missing,gates=gates,numerical=numerical,
        first_divergence=dict(FIRST_OBSERVED_DIFFERENCE=paths,FIRST_CAUSALLY_LOCALIZED_DIFFERENCE='UNRESOLVED',ROOT_CAUSE='UNRESOLVED',
            distinction='First visible capture is not automatically the first arithmetic divergence. Parallel Q/K/V branches are causal peers.'),
        hypotheses=hypotheses,selected_layers=layers,
        next_action='Review this historical report before any new GPU experiment. If inputs remain unobserved, specify only those missing early-layer boundaries; no GPU experiment is automatically proposed or submitted.')


def logits(data):
    ev=data['evidence'];rows=data['rows'];comparisons=[];prefill=[];decode=[];tokens=[]
    for ca,cb in ((CONDITIONS[0],CONDITIONS[1]),(CONDITIONS[1],CONDITIONS[2]),(CONDITIONS[0],CONDITIONS[2])):
        for rid,t in data['tasks'].items():
            a=rows[ca,rid];b=rows[cb,rid];require(a['logits_shape'][-1]==b['logits_shape'][-1],'logit vocabulary differs')
            width=a['logits_shape'][-1];values=[];blobs=[]
            for condition,row in ((ca,a),(cb,b)):
                item=dict(file=row['logits_file'],shape=row['logits_shape'],dtype='float32',sha256=row['logits_sha256'])
                blob=ev.tensor(condition+'/'+row['adapter_id']+'/es',item);blobs.append(blob);values.append(unpack(blob,'float32'))
            checks=[];skipped=[]
            for step in range(max(a['generated_tokens'],b['generated_tokens'])):
                if step>=min(a['generated_tokens'],b['generated_tokens']) or a['generated_token_ids'][:step]!=b['generated_token_ids'][:step]:
                    skipped.append(dict(step=step,reason='UNMATCHED_GENERATION_HISTORY'));continue
                check=dict(step=step,phase='PREFILL' if step==0 else 'DECODE',bitwise_equal=blobs[0][step*width*4:(step+1)*width*4]==blobs[1][step*width*4:(step+1)*width*4],
                    **check_parity(values[1][step*width:(step+1)*width],values[0][step*width:(step+1)*width],[1,1,width]));checks.append(check)
                if t['scored']:
                    (prefill if step==0 else decode).append(check['passed'])
            equal=a['generated_token_ids']==b['generated_token_ids'] and a['generated_text']==b['generated_text'];tokens.append(equal)
            comparisons.append(dict(comparison=ca+'_vs_'+cb,request_id=rid,scored=t['scored'],steps=checks,skipped=skipped,
                generated_tokens_equal=a['generated_token_ids']==b['generated_token_ids'],generated_text_equal=a['generated_text']==b['generated_text'],
                frozen_tolerance_pass=bool(checks) and not skipped and all(s['passed'] for s in checks)))
    summaries={}
    for label in sorted({r['comparison'] for r in comparisons}):
        targets=[r for r in comparisons if r['comparison']==label and r['scored']];checks=[s for r in targets for s in r['steps']]
        summaries[label]=dict(scored_requests=len(targets),matched_steps=len(checks),failing_steps=sum(not s['passed'] for s in checks),
            bitwise_equal_steps=sum(s['bitwise_equal'] for s in checks),skipped_steps=sum(len(r['skipped']) for r in targets),
            worst_max_abs=max((s['max_abs'] for s in checks),default=None),worst_relative_l2=max((s['relative_l2'] for s in checks),default=None),
            worst_last_token_KL=max((s['last_token_kl'] for s in checks),default=None),argmax_agreement_steps=sum(s['last_token_argmax_agreement'] for s in checks))
    return dict(thresholds=THRESHOLDS,comparisons=comparisons,scored_target_summary=summaries,scope='Frozen logit checks; no new thresholds or replacement generation',
        prefill_verdict='PASS' if prefill and all(prefill) else 'FAIL',decode_verdict='FAIL' if any(x['skipped'] for x in comparisons if x['scored']) else 'PASS' if decode and all(decode) else 'FAIL' if decode else 'NOT_EXERCISED',
        token_text_verdict='PASS' if all(tokens) else 'FAIL',frozen_verdict='PASS' if all(x['frozen_tolerance_pass'] for x in comparisons) else 'FAIL')


def run(*,output_dir,run_dir=None,runs_root=None,experiment=None,audit_dir=None,r1_audit_dir=None,expected_commit=None,fixture=False,
        layers=None,allow_unavailable=False,max_tensor_bytes=32*2**20,max_input_bytes=32*2**30):
    root=locate(run_dir,runs_root);out=Path(output_dir).resolve()
    require(not out.exists() and not out.is_relative_to(root) and not root.is_relative_to(out),'fresh nonoverlapping R2 output required')
    require(not out.is_relative_to(ROOT) or out.is_relative_to(ROOT/'results/r5/i5r/numerical_fidelity'),'repository R2 outputs belong under results/r5/i5r/numerical_fidelity')
    for protected in (experiment,audit_dir,r1_audit_dir):
        if protected:require(not out.is_relative_to(Path(protected).resolve()) and not Path(protected).resolve().is_relative_to(out),'output overlaps historical inputs')
    data=read_run(root,experiment=experiment,audit_dir=audit_dir,r1_audit_dir=r1_audit_dir,expected_commit=expected_commit,fixture=fixture,allow_unavailable=allow_unavailable,
        max_tensor_bytes=max_tensor_bytes,max_input_bytes=max_input_bytes)
    report=analyze(data,layers=layers)
    out.mkdir(parents=True,mode=0o700)
    write(out/'i5r2_evidence_inventory.json',dict(data['inventory'],raw_root=str(root),selected_layers=report['selected_layers'],
        tensor_bound_bytes=max_tensor_bytes,input_bound_bytes=max_input_bytes,missing_comparisons=report['missing'],
        missing_attention_context=['actual scaled query and consumed K/V values at attention entry','actual additive causal mask at aligned rows','attention scores/probabilities if not recorded'],
        missing_CPU_math=['CPU hidden/cast input','actual A intermediate','B output/transported delta values'],
        sampling='All elements of each requested aligned tensor; optional --layers reduces layer coverage explicitly'))
    write(out/'i5r2_tensor_comparisons.json',dict(records=report['tensor_comparisons'],hash_only=report['hash_only_comparisons']))
    write(out/'i5r2_first_divergence.json',report['first_divergence']);write(out/'i5r2_hypothesis_matrix.json',report['hypotheses'])
    write(out/'i5r2_numerical_fidelity.json',dict(report['numerical'],gates=report['gates'],historical_export=data['old_numerical'],
        gate_layer_scope=report['selected_layers'],TEST_FIXTURE_ONLY=fixture,fresh_GPU_execution_performed=False))
    write(out/'i5r2_source_provenance.json',data['provenance'])
    write(out/'i5r2_runtime_contracts.json',{c+'/'+a:v for (c,a),v in data['settings'].items()})
    write(out/'i5r2_cpu_correction_receipts.json',{c+'/'+a:v for (c,a),v in data['cpu_receipts'].items()})
    write(out/'i5r2_historical_r1_export.json',data['r1_export'])
    import json
    text='# R5-I5R-2 historical numerical evidence review\n\n'
    text+=f"TEST_FIXTURE_ONLY={fixture}; no new GPU execution. Historical-export-based audit plus new CPU tensor comparisons.\n\n"
    text+='ROOT_CAUSE: UNRESOLVED. This report does not authorize cache reuse or claim performance improvement.\n\n'
    text+='## Independent numerical/integrity statuses\n\n'+json.dumps(report['gates'],indent=2)+'\n\n'
    text+='## First observed differences\n\n'+json.dumps(report['first_divergence'],indent=2)+'\n\n'
    text+='## Hypotheses\n\n'+json.dumps(report['hypotheses'],indent=2)+'\n\n'
    text+=f"Missing tensor comparisons: {len(report['missing'])}. Exact roles/layers are in i5r2_evidence_inventory.json. Original strides not recorded are {UNKNOWN}.\n\n"
    text+='Cache integrity proves fidelity to the standalone source, not equivalence to the corresponding full-target prefix. The old last-position trace cannot prove all-prefix equivalence. Source/full-prefix and full/HIT-suffix comparisons here use every aligned recorded token.\n\n'
    text+='Historical source blobs are checked against the recorded commit, never the current source collector. A fresh original-source independent GPU audit was not executed. Timing is excluded: this is not a serving performance benchmark.\n\n'
    text+=report['next_action']+'\n'
    (out/'i5r2_root_cause_report.md').write_text(text)
    write(out/'i5r2_output_manifest.json',dict(files={p.name:file_sha(p) for p in sorted(out.iterdir()) if p.is_file()}))
    return dict(status='CPU_FIXTURE_DIAGNOSIS_COMPLETE' if fixture else 'PARTIAL_HISTORICAL_PROVENANCE' if not data['provenance']['historical_git_object_available'] else 'HISTORICAL_TENSOR_DIAGNOSIS_COMPLETE',ROOT_CAUSE='UNRESOLVED',
        GPU_NOT_RUN=True,TEST_FIXTURE_ONLY=fixture,gates=report['gates'],tensor_comparisons=len(report['tensor_comparisons']),missing_comparisons=len(report['missing']),
        historical_source_verification=data['provenance']['historical_source_verification'],output=str(out))
