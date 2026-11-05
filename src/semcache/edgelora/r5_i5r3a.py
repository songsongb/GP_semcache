"""CPU static opportunities only; never executes encoder, model, or cache."""
import json,re
from collections import Counter,defaultdict
from pathlib import Path
from .communication import require,digest,digest_bytes
from .r3a_artifacts import ROOT,read,write,sha
from .r5_i5_data import load_dataset,SOURCES,VERSIONS
from .r5_i5_workload import prepare_prompt,routing,pack_groups,turn_order,select,BUDGETS,TEMPLATE
from .r5_multi_artifacts import validate,sharing,resolve
from .r5_i5_evidence import seal
from .r5_i5r1_plan import source_identity
from .r5_i5r2_tensors import Evidence
from .r5_i5r2_analysis import source_provenance

DATASETS=('qasper','2wikimqa','multi_news')
NAMESPACE=ROOT/'results/r5/i5r/shared_context_audit'
CAPACITY=512*2**20

def fresh(output,protected):
 p=Path(output).resolve()
 require(not Path(output).is_symlink() and not p.exists(),'fresh output required')
 require(not p.is_relative_to(ROOT) or p.is_relative_to(NAMESPACE),'wrong I5R3A namespace')
 require(all(not p.is_relative_to(Path(x).resolve()) and not Path(x).resolve().is_relative_to(p) for x in protected),'output overlaps inputs')
 p.mkdir(parents=True);return p

def provenance(path,name,fixture=False):
 rows,m=load_dataset(path,fixture=fixture)
 require(m['dataset']==name and m['version']==VERSIONS[name] and m['source']==SOURCES[name],'dataset provenance differs')
 require(m['split'] and re.fullmatch('[0-9a-f]{40,64}',m['revision']) and m['source_files'] and all(re.fullmatch('[0-9a-f]{64}',x) for x in m['source_files'].values()),'missing split/revision/source hashes')
 return rows,m

def pair_record(a,b,locations,registry,bpt,capacity=CAPACITY):
 x,y=a['token_ids'],b['token_ids'];s,t=a['sample'],b['sample'];ra=a['route'];rb=b['route']
 denied=sharing(registry,ra['user_id'],rb['user_id'],ra['adapter_id'],rb['adapter_id'])
 checks=dict(CONTEXT_SHARED=s['context']==t['context'],STRICT_TOKEN_PREFIX=0<len(x)<len(y) and x==y[:len(x)],SAME_OR_AUTHORIZED_ADAPTER=denied is None,SOURCE_CAPACITY_ELIGIBLE=len(x)<2048 and len(x)*bpt<=capacity,SOURCE_PRECEDES_TARGET=locations[s['id']][1]<locations[t['id']][1],WITHIN_SAME_CACHE_LIFETIME=locations[s['id']][0]==locations[t['id']][0])
 reasons=[k for k,v in checks.items() if not v and k!='CONTEXT_SHARED']
 if denied:reasons.append(denied)
 return dict(scope='STATIC_ONLY',category='NATURAL',source_id=s['id'],target_id=t['id'],shared_context_key=a['shared_context_key'],target_shared_context_key=b['shared_context_key'],source_route=ra,target_route=rb,source_prompt_sha256=digest(x),target_prompt_sha256=digest(y),prefix_tokens=len(x) if checks['STRICT_TOKEN_PREFIX'] else 0,suffix_tokens=len(y)-len(x) if checks['STRICT_TOKEN_PREFIX'] else None,source_raw_qkv_bytes=len(x)*bpt,source_location=locations[s['id']],target_location=locations[t['id']],**checks,STATICALLY_ELIGIBLE=not reasons,rejection_reasons=reasons,completion_before_target='REQUIRED_NOT_OBSERVED; proposed sequential completion barrier',residency='NOT_EVALUATED',static_scope='Necessary static conditions only; not admission/residency/semantic/physical proof')

def order_locations(prompts,block_size):
 blocks=pack_groups(prompts,block_size);locations={};sequence=0
 for block,items in enumerate(blocks):
  for p in items:locations[p['sample']['id']]=[block,sequence];sequence+=1
 return blocks,locations

def prefix_counts(pairs):
 prefixes=[p for p in pairs if p['STRICT_TOKEN_PREFIX']]
 return dict(scope='STATIC_ONLY',strict_prefix_pairs=len(prefixes),same_context_strict_prefix_pairs=sum(p['CONTEXT_SHARED'] for p in prefixes),cross_context_strict_prefix_pairs=sum(not p['CONTEXT_SHARED'] for p in prefixes),authorized_prefix_pairs=sum(p['SAME_OR_AUTHORIZED_ADAPTER'] for p in prefixes),capacity_eligible_pairs=sum(p['SOURCE_CAPACITY_ELIGIBLE'] for p in prefixes),source_precedes_target_pairs=sum(p['SOURCE_PRECEDES_TARGET'] for p in prefixes),within_lifetime_pairs=sum(p['WITHIN_SAME_CACHE_LIFETIME'] for p in prefixes),statically_eligible_pairs=sum(p['STATICALLY_ELIGIBLE'] for p in prefixes))

def audit_dataset(samples,tok,registry,*,seed=42,block_size=20,input_limit=None,bpt=491520,capacity=CAPACITY):
 require(1<=block_size<=20 and 0<len(samples)<=2000,'INSUFFICIENT_OR_UNBOUNDED_DATASET: requires 1..2000 unique records')
 require(len({s['id'] for s in samples})==len(samples),'duplicate IDs')
 prepared=[];records=[];excluded=[];groups=defaultdict(list)
 for ordinal,s in enumerate(samples):
  key=digest_bytes(s['context'].encode());groups[key].append(s['id']);cids=list(tok.encode(s['context'],add_special_tokens=False))
  row=dict(dataset=s['dataset'],version=s['version'],source_record_sha256=s['source_record_sha256'],original_ordinal=ordinal,id=s['id'],group_id=s['group_id'],shared_context_key=key,original_context_bytes=len(s['context'].encode()),original_context_sha256=key,original_context_token_sha256=digest(cids),route=routing(s,seed))
  resolve(registry,row['route']['user_id'],row['route']['adapter_id'])
  try:
   p=prepare_prompt(s,tok,2048,BUDGETS[s['dataset']],allow_truncation=True,input_limit=input_limit)
   p.update(route=row['route'],shared_context_key=key);prepared.append(p)
   row.update({k:p[k] for k in ('truncated','original_context_tokens','retained_context_tokens','removed_context_tokens','retained_prompt_tokens','token_ids_sha256','evaluation_label')});row['token_ids']=p['token_ids']
  except ValueError as exc:row['exclusion_reason']=str(exc);excluded.append(dict(id=s['id'],reason=str(exc)))
  records.append(row)
 # Production I5 selection/order over the same ALL eligible IDs; no historical
 # request-count selection or block configuration is inferred.
 original=[]
 if prepared:
  original,_=select(samples,tok,count=len(prepared),seed=seed,context_limit=2048,budget=BUDGETS[samples[0]['dataset']],allow_truncation=True,input_limit=input_limit)
 require({p['sample']['id']:p['token_ids'] for p in original}=={p['sample']['id']:p['token_ids'] for p in prepared},'order comparison changed eligible IDs or prompt tokens')
 original_blocks,original_locations=order_locations(original,block_size)
 # Proposed context-cohort order is separate from original IDs/group IDs/routing.
 prepared.sort(key=lambda p:(p['shared_context_key'],digest([seed,p['sample']['group_id']]),turn_order(p['sample'])))
 # Reuse production group packing without changing original sample grouping.
 blocks,locations=order_locations(prepared,block_size)
 pairs=[];original_pairs=[]
 for a in prepared:
  for b in prepared:
   if a is b:continue
   x,y=a['token_ids'],b['token_ids']
   if a['sample']['context']==b['sample']['context'] or len(x)<len(y) and x==y[:len(x)]:
    require(len(pairs)<500000,'pair evidence bound exceeded; no silent subsampling')
    pair=pair_record(a,b,locations,registry,bpt,capacity)
    old=pair_record(a,b,original_locations,registry,bpt,capacity)
    original_pairs.append(old)
    pair['original_i5_order']={k:old[k] for k in ('source_location','target_location','SOURCE_PRECEDES_TARGET','WITHIN_SAME_CACHE_LIFETIME','STATICALLY_ELIGIBLE','rejection_reasons')}
    pairs.append(pair)
 prefixes=[p for p in pairs if p['STRICT_TOKEN_PREFIX']];eligible=[p for p in prefixes if p['STATICALLY_ELIGIBLE']]
 # Connected source/target context components avoid counting shared dependencies twice.
 keys={p['sample']['id']:p['shared_context_key'] for p in prepared};parent={}
 def root(k):
  parent.setdefault(k,k)
  while parent[k]!=k:k=parent[k]
  return k
 for pair in eligible:
  a,b=root(keys[pair['source_id']]),root(keys[pair['target_id']]);parent[b]=a
 independent=len({root(k) for k in parent})
 block_stats=[dict(block=i,requests=len(items),cache_reset=True,adapters={aid:dict(requests=sum(p['route']['adapter_id']==aid for p in items),offered_raw_qkv_bytes=sum(len(p['token_ids'])*bpt for p in items if p['route']['adapter_id']==aid),occupancy='NOT_EVALUATED') for aid in ('Adapter_A','Adapter_B','Adapter_C')}) for i,items in enumerate(blocks)]
 shared=[dict(shared_context_key=k,original_ids=v,size=len(v),distinct_unordered_pairs=len(v)*(len(v)-1)//2,block_ordinals=sorted({locations[x][0] for x in v if x in locations}),eligible_samples=sum(x in locations for x in v)) for k,v in sorted(groups.items()) if len(v)>1]
 controlled=[]
 for p in prepared:
  if len(p['token_ids'])<=64:continue
  controlled.append(dict(category='CONTROLLED_DIAGNOSTIC',target_id=p['sample']['id'],original_group_id=p['sample']['group_id'],shared_context_key=p['shared_context_key'],route=p['route'],source_token_ids=p['token_ids'][:64],source_sha256=digest(p['token_ids'][:64]),target_sha256=digest(p['token_ids']),prefix_tokens=64,suffix_tokens=len(p['token_ids'])-64,proposed_order=['NEW_STANDALONE_SOURCE','ORIGINAL_TARGET'],source_capacity_eligible=64*bpt<=capacity,limitations='Constructed source, not a natural dataset request; real admission, semantic discovery, completed source and independent physical proof required'))
  if len(controlled)==3:break
 summary=dict(dataset=samples[0]['dataset'],total_samples=len(samples),full_context_eligible=sum(not p['truncated'] for p in prepared),context_limited_eligible=len(prepared),excluded_count=len(excluded),truncated_count=sum(p['truncated'] for p in prepared),shared_context_groups=len(shared),shared_context_request_pairs=sum(g['distinct_unordered_pairs'] for g in shared),strict_prefix_pairs=len(prefixes),authorized_prefix_pairs=sum(p['SAME_OR_AUTHORIZED_ADAPTER'] for p in prefixes),capacity_eligible_pairs=sum(p['SOURCE_CAPACITY_ELIGIBLE'] for p in prefixes),within_lifetime_pairs=sum(p['WITHIN_SAME_CACHE_LIFETIME'] for p in prefixes),statically_eligible_pairs=len(eligible),natural_actual_hit_status='NOT_EVALUATED',independent_context_groups=len(groups),independent_eligible_prefix_groups=independent,proposed_blocks=len(blocks),cache_resets=len(blocks),maximum_lifetime_requests=max(map(len,blocks),default=0),group_size_histogram={str(k):v for k,v in sorted(Counter(len(v) for v in groups.values()).items())},context_groups_split_by_reset=sum(len(g['block_ordinals'])>1 for g in shared),removed_tokens=sum(p['removed_context_tokens'] for p in prepared),truncation_rate=sum(p['truncated'] for p in prepared)/len(prepared) if prepared else None,exclusions_by_reason=dict(Counter(x['reason'] for x in excluded)),result='NATURAL_STATIC_CANDIDATES' if eligible else 'NO_NATURAL_ELIGIBLE_PAIR')
 summary.update(prefix_counts(pairs))
 comparison=dict(scope='STATIC_ONLY',basis='Same all-eligible original IDs, tokens, routing, block size and reset policy; not a historical execution replay',original_i5=prefix_counts(original_pairs),proposed_context_grouped=prefix_counts(pairs),original_blocks=len(original_blocks),gained_statically_eligible_pairs=sum(p['STRICT_TOKEN_PREFIX'] and p['STATICALLY_ELIGIBLE'] and not p['original_i5_order']['STATICALLY_ELIGIBLE'] for p in pairs),lost_statically_eligible_pairs=sum(p['STRICT_TOKEN_PREFIX'] and not p['STATICALLY_ELIGIBLE'] and p['original_i5_order']['STATICALLY_ELIGIBLE'] for p in pairs))
 return dict(scope='STATIC_ONLY',summary=summary,order_comparison=comparison,original_i5_order=[dict(id=k,block=v[0],sequence=v[1]) for k,v in sorted(original_locations.items(),key=lambda x:x[1][1])],proposed_block_statistics=block_stats,scenario='EXACT_CONTEXT_ORDER_THEN_PRODUCTION_ORIGINAL_GROUP_PACKING; not historical request order',samples=records,groups=shared,pairs=pairs,exclusions=excluded,pilot=dict(natural=eligible[:3],controlled=controlled,independent_groups_required_for_statistics=3,repeated_samples_independent=False),order=[dict(id=k,block=v[0],sequence=v[1]) for k,v in sorted(locations.items(),key=lambda x:x[1][1])])

def inputs(registry_path,multi):
 registry=read(registry_path);require(set(registry)==set(DATASETS),'explicit registry must contain exactly three dataset paths')
 # Relative paths follow existing I5 repository-root execution conventions.
 registry={k:str((ROOT/Path(v)).resolve()) for k,v in registry.items()}
 m,pm,specs,users=validate(multi);require(not m['TEST_FIXTURE_ONLY'] and len(users['users'])==30,'real expanded 30-user preparation required')
 from .r4c_prepare import opt_tokenizer
 tok=opt_tokenizer(pm);require(tok.supported_context_length==2048 and tok.identity['raw_cache_bytes_per_token']==491520,'OPT configuration differs')
 data={};prov={}
 for name,path in registry.items():
  data[name],manifest=provenance(path,name)
  original=Path(manifest['source_path'])
  upstream='UNAVAILABLE_LOCAL_UPSTREAM; recorded inventory sealed, normalized originals verified'
  if original.exists():
   from .r4c_dataset import file_hashes
   require(file_hashes(original)==manifest['source_files'],'available upstream source hash differs');upstream='AVAILABLE_UPSTREAM_SHA256_VERIFIED'
  prov[name]=dict(path=path,manifest=manifest,inventory=Evidence(path).verify(),upstream_verification=upstream)
 return tok,users,prov,data,pm,m

def prepare(registry,multi,output,block_size=20,input_limit=None,seed=42):
 require(1<=block_size<=20 and (input_limit is None or 1<=input_limit<=2048),'invalid audit bounds')
 tok,users,prov,data,pm,m=inputs(registry,multi)
 p=dict(schema='I5R3A_STATIC_V1',source=source_identity(),registry_path=str(Path(registry).resolve()),registry_sha256=sha(registry),multi_prepared=str(Path(multi).resolve()),multi_inventory=Evidence(multi).verify('r5_output_manifest.json'),datasets=prov,tokenizer_identity=tok.identity,model_identity=pm['identity'],users=users,template=TEMPLATE,budgets={k:BUDGETS[k] for k in DATASETS},block_size=block_size,input_limit=input_limit,seed=seed,capacity_bytes=CAPACITY,max_entries=64,clusters=1,cache_policy='LRU',context_limit=2048,group_stable_policy='NOT_EVALUATED: production per-sample retention only',TEST_FIXTURE_ONLY=False)
 p['plan_id']=digest(p);out=fresh(output,[registry,multi,*[x['path'] for x in prov.values()]]);write(out/'i5r3a_plan.json',p);seal(out);return dict(status='PREPARED',plan_id=p['plan_id'])

def compute(plan):
 ev=Evidence(plan);ev.verify();p=ev.json('i5r3a_plan.json');q=dict(p);ident=q.pop('plan_id');require(digest(q)==ident and p['schema']=='I5R3A_STATIC_V1','plan differs')
 require(p['capacity_bytes']==CAPACITY and p['max_entries']==64 and p['clusters']==1 and p['cache_policy']=='LRU' and p['context_limit']==2048 and p['TEST_FIXTURE_ONLY'] is False,'static runtime constraints changed')
 source_provenance(dict(source_commit=p['source']['commit'],sources=p['source']['files']),expected_commit=p['source']['commit'],fixture=False)
 require(sha(p['registry_path'])==p['registry_sha256'] and Evidence(p['multi_prepared']).verify('r5_output_manifest.json')==p['multi_inventory'],'input provenance changed')
 tok,users,prov,data,pm,m=inputs(p['registry_path'],p['multi_prepared'])
 require(prov==p['datasets'] and tok.identity==p['tokenizer_identity'] and pm['identity']==p['model_identity'] and users==p['users'] and p['template']==TEMPLATE and p['budgets']=={k:BUDGETS[k] for k in DATASETS},'inputs/template changed')
 result={k:audit_dataset(v,tok.tokenizer,users,seed=p['seed'],block_size=p['block_size'],input_limit=p['input_limit'],bpt=tok.identity['raw_cache_bytes_per_token']) for k,v in data.items()}
 return p,result

def summary_record(p,result,analysis_source):
 return dict(scope='STATIC_ONLY',datasets=[r['summary'] for r in result.values()],order_comparisons={k:r['order_comparison'] for k,r in result.items()},denominators='Context pairs: distinct unordered original IDs; prefix pairs: directed distinct complete prompts, independently filtered counts; context_limited_eligible includes untruncated eligible records. Not HIT rates.',runtime_semantics_admission_eviction_reuse_numerical_fidelity='NOT_EVALUATED',statistical_warning='Shared context/source/repeated samples are dependent; count independent groups',static_completion='Proposed sequential completion barrier; actual completion/admission not observed',residency='NOT_EVALUATED; no hypothetical LRU simulation',source=p['source'],analysis_source=analysis_source,plan_id=p['plan_id'])

def analyze(plan,output):
 p,result=compute(plan)
 require(len(json.dumps(result,ensure_ascii=False).encode())<=16*2**20,'audit JSON exceeds 16 MiB metadata bound; no partial result emitted')
 out=fresh(output,[plan,p['registry_path'],p['multi_prepared'],*[v['path'] for v in p['datasets'].values()]])
 write(out/'i5r3a_dataset_provenance.json',p['datasets']);write(out/'i5r3a_results.json',result)
 write(out/'i5r3a_summary.json',summary_record(p,result,source_identity()))
 seal(out);return dict(status='STATIC_AUDIT_WRITTEN',GPU_NOT_RUN=True)

def verify(plan,audit,output):
 p,result=compute(plan);ev=Evidence(audit);receipt=ev.verify()
 require(ev.json('i5r3a_results.json')==result and ev.json('i5r3a_dataset_provenance.json')==p['datasets'],'independent recomputation differs')
 summary=ev.json('i5r3a_summary.json');require(summary==summary_record(p,result,summary['analysis_source']),'summary differs')
 source_provenance(dict(source_commit=summary['analysis_source']['commit'],sources=summary['analysis_source']['files']),expected_commit=summary['analysis_source']['commit'],fixture=False)
 out=fresh(output,[plan,audit,p['registry_path'],p['multi_prepared'],*[v['path'] for v in p['datasets'].values()]])
 write(out/'i5r3a_independent_audit.json',dict(status='PASS',scope='CPU static recomputation only',inventory=receipt,plan_id=p['plan_id'],verification_source=source_identity(),real_runtime_hit_status='NOT_EVALUATED'));seal(out);return dict(status='PASS',scope='STATIC_ONLY')
