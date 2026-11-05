"""Read-only C2 attention operand audit and sealed, conditional replay preparation.

No missing suffix values or masks are generated. Historical exports are audited
against historical Git blobs, never against the current I5 execution inventory.
"""
import math
import struct
from pathlib import Path
from .communication import require,digest,digest_bytes
from .r3a_artifacts import ROOT,write,THRESHOLDS
from .r5_i5_evidence import seal
from .r5_i5r2_tensors import Evidence,DTYPES,slice_axis,metrics,file_sha
from .r5_i5r2_analysis import source_provenance
from .r5_i5r1_plan import source_identity
from .r5_i5r2bs1_analysis import compare_bytes
from .r5_multi_artifacts import runtime_view

VERSION='I5R2C3_ATTENTION_V1'
NAMESPACE=ROOT/'results/r5/i5r/numerical_fidelity/attention_localization'
BOUNDARIES=('scores','masked_scores','probabilities','pv','output_projection','module_output')


def fresh(path,protected=()):
 p=Path(path).resolve();require(not Path(path).is_symlink() and not p.exists(),'fresh output directory required')
 require(not p.is_relative_to(ROOT) or p.is_relative_to(NAMESPACE),'wrong C3 output namespace')
 require(all(not p.is_relative_to(Path(x).resolve()) and not Path(x).resolve().is_relative_to(p) for x in protected),'output overlaps evidence')
 p.mkdir(parents=True,mode=0o700);return p


def causal_mask(data,shape,dtype,queries,keys):
 """Validate actual additive causal entries, including all recorded future keys."""
 require(dtype in DTYPES and shape==[1,1,queries,keys] and queries<=keys,'causal mask shape differs')
 fmt,width=DTYPES[dtype];require(len(data)==queries*keys*width,'causal mask bytes differ')
 minimum=-65504. if dtype=='float16' else -3.4028234663852886e38
 for i,(v,) in enumerate(struct.iter_unpack(fmt,data)):
  q,k=divmod(i,keys)
  require(v==0. if k<=q else v in (minimum,-math.inf),'invalid actual causal mask/future-key visibility')
 return dict(queries=queries,keys=keys,causal=True,future_keys_masked=True,sha256=digest_bytes(data))


def aligned_bytes(data,shape,dtype,role):
 """Query prefix only; score/probability key comparisons use the shared 64 keys."""
 width=DTYPES[dtype][1]
 if role in ('scores','masked_scores','probabilities'):
  require(len(shape)==3 and shape[1]>=64 and shape[2]>=64,'invalid head/query/key axes')
  data,s=slice_axis(data,shape,1,0,64,width)
  return slice_axis(data,s,2,0,64,width)[0]
 require(role in ('pv','output_projection','module_output') and len(shape)==3 and shape[1]>=64,'invalid output token axes')
 return slice_axis(data,shape,1,0,64,width)[0]


def compare_aligned(a,b,dtype,role):
 x=aligned_bytes(a[0],a[1],dtype,role);y=aligned_bytes(b[0],b[1],dtype,role)
 if role=='masked_scores' and any(not math.isfinite(v[0]) for v in struct.iter_unpack(DTYPES[dtype][0],x+y)):
  m=metrics(x,y,dtype,allow_negative_infinity=True)
 else:m=compare_bytes(x,y,dtype)
 m['differing_percentage']=100*m['differing_values']/m['elements']
 return m


def verdict(comparisons,repeatable,reproduced,complete=True,fixture=False):
 first=next((n for n in BOUNDARIES if any(x['boundary']==n and not x['metrics']['bitwise_equal'] for x in comparisons)),None)
 established=bool(complete and repeatable and reproduced and not fixture)
 return dict(EVIDENCE_VALIDATED='CPU_FIXTURE_ONLY' if fixture else 'PASS',
  HISTORICAL_REPLAY_REPRODUCED=established,REPLAY_NOT_ESTABLISHED=not established,
  FIRST_OBSERVED_ISOLATED_DIFFERENCE=first,
  FIRST_DIVERGENCE_LOCALIZED=first if established and first else 'NOT_ESTABLISHED',
  INCOMPLETE_EVIDENCE=not complete,ROOT_CAUSE_CONFIRMED=False,
  STRICT_NUMERICAL_FIDELITY_RESTORED='NOT_EVALUATED',
  PHYSICAL_CACHE_REUSE='CPU_FIXTURE_ONLY' if fixture else 'HISTORICAL_C2_VERIFIED',
  scope='Native eager operations observed in isolated replay, not a performance benchmark or proven physical mechanism')


def history(experiment,run,audit,expected_commit,fixture=False):
 pe=Evidence(experiment,max_input_bytes=128*2**20);ps=pe.verify();p=pe.json('i5_workload_manifest.json');q=dict(p);eid=q.pop('experiment_id')
 require(digest(q)==eid,'historical plan identity differs')
 ev=Evidence(run,max_input_bytes=8*2**30);rs=ev.verify();cfg=ev.json('i5_execution_config.json')
 require(ev.json('i5_block_status.json')['status']=='COMPLETE' and cfg['fixture']==fixture,'historical completion/scope differs')
 require(cfg['experiment_id']==eid and ps['manifest_sha256']==cfg['experiment_manifest_sha256'] and cfg['block'] in p['blocks'] and cfg['i5r2c2']==p['i5r2c2'],'C2 plan/run chain differs')
 require(cfg['i5r2c2']['mode'] in ('SOURCE_ONLY','SOURCE_AND_SUFFIX','OFF'),'unsupported C2 mode')
 require(p['i5r2b']['source']['commit']==cfg['source_commit'] and p['i5r2b']['source']['files']==cfg['sources'],'historical plan/source inventory differs')
 provenance=source_provenance(cfg,expected_commit=expected_commit,fixture=fixture)
 provenance['current_attention_analyzer']={str(f.relative_to(ROOT)):file_sha(f) for f in (Path(__file__),ROOT/'src/semcache/edgelora/r5_i5r2c3_runtime.py',ROOT/'scripts/139_r5_i5r2c3.py') if f.exists()}
 ae=Evidence(audit,max_input_bytes=128*2**20);ats=ae.verify();ad=ae.json('i5r2c2_decision.json')
 require(ad['TEST_FIXTURE_ONLY']==fixture and ad['mode']==cfg['i5r2c2']['mode'] and ad['inventory']==rs and ad['independent_audit']['status']=='PASS' and ad['PHYSICAL_REUSE_PRESERVED']==('CPU_FIXTURE_ONLY' if fixture else 'PASS'),'C2 exported audit identity/inventory differs')
 oldnum=ae.json('i5r2c2_numerical.json')
 # C2 numerical exports embed the frozen I5 threshold record.
 require(oldnum['target']['thresholds']==THRESHOLDS,'historical frozen thresholds differ')
 s,t=cfg['block']['tasks'];require(len(s['token_ids'])==64 and len(t['token_ids'])==256 and s['token_ids']==t['token_ids'][:64] and s['adapter_id']==t['adapter_id']=='Adapter_A' and s['user_id']==t['user_id']=='User_01','historical source/target alignment differs')
 registry=ev.json('i5_adapter_registry.json');captures={};identities={};steps={};contexts={};backends={}
 spec=registry['adapters']['Adapter_A'];base_identity=p['i5r1']['model_identity']
 require(spec['adapter_id']=='Adapter_A' and spec['matrix_sha256']==p['i5r1']['adapter_fingerprints']['Adapter_A'] and
  spec['base_fingerprint']==base_identity['base_weight_sha256'] and spec['rank']==8 and spec['matrix_dtype']=='float32','historical model/adapter differs')
 expected_identity=runtime_view({'identity':base_identity},spec)['identity']
 for label,cond,task,n in (('S','SEMCACHE_INTEGRATED',s,64),('F','FULL_RECOMPUTE',t,256)):
  base=cond+'/Adapter_A/es';ident=ev.json(cond+'/Adapter_A/identity.json');identities[label]=ident
  require(ident['model_identity']==expected_identity and ident['adapter']==spec,'historical model/adapter differs')
  if not fixture:
   require(ident['physical_identity']['base_fingerprint']==p['i5r1']['model_identity']['base_weight_sha256'] and ident['physical_identity']['adapter_matrix_fingerprint']==ident['adapter']['matrix_sha256'],'physical/model/adapter fingerprint differs')
   method=ident['contract']['methods']['attention'];require(digest_bytes(method['source'].encode())==method['source_sha256'],'historical attention implementation text/hash differs')
  cap=ev.json(base+'/i5r2b_capture.json');require(cap['status']=='COMPLETE','incomplete capture')
  captures[label]=(base,cap);ctx=cap['contexts'][task['request_id']]['actual_context'];contexts[label]=ctx
  observation=cap['contexts'][task['request_id']].get('position_observation')
  require(observation and observation['positions']==list(range(n)) and observation['past_length']==0 and (fixture or observation['source']=='ACTUAL_OPT_POSITION_MODULE' and observation['lookup_output_equal'] and observation['attention_mask_sha256']==digest([1]*n)),'actual position-module observation missing/mismatched')
  require(ctx['pass_index']==0 and ctx['past_length']==0 and ctx['input_token_ids']==task['token_ids'] and ctx['positions']==list(range(n)) and ctx['attention_mask']==[1]*n,'actual positions/mask/token context differs')
  found=[x for x in ev.jsonl(base+'/r5_tokens.jsonl') if x['request_id']==task['request_id'] and x['step']==0];require(len(found)==1 and found[0]['context']==ctx,'missing/duplicate actual prefill')
  steps[label]=found[0]
  be=cap['contexts'][task['request_id']];require(be['backend_before']==be['backend_after'],'backend changed within forward');backends[label]=be['backend_before']
 require(identities['S']['physical_identity']==identities['F']['physical_identity'] and identities['S']['contract']==identities['F']['contract'] and backends['S']==backends['F'],'matched GPU/backend/contract differs')
 missing=[];inputs={};observations=[];coverage={};mask_checks={}
 for label,task,n in (('S',s,64),('F',t,256)):
  base,cap=captures[label];entries=cap['entries'];rid=task['request_id'];inputs[label]={};coverage[label]={}
  for role in 'qkv':
   bounded=entries.get(rid+'/raw_qkv_'+role)
   require(bounded and bounded['status']=='CAPTURED' and bounded['shape']==[1,64,identities[label]['model_identity']['hidden_size']] and bounded['semantic_positions']==[0,64],'missing/misaligned prefix QKV')
   require(bounded.get('observed_shape')==[1,n,identities[label]['model_identity']['hidden_size']] and (fixture or bounded['dtype']=='float16'),'projection observed shape/dtype differs')
   stride=bounded.get('observed_stride')
   require(stride is None or isinstance(stride,str) or isinstance(stride,list) and len(stride)==3 and all(type(v) is int and v>=0 for v in stride),'invalid historical QKV stride')
   ev.tensor(base,bounded);coverage[label]['raw_'+role]=bounded
   full=steps[label]['qkv']['0.'+role]
   selected=full if 'file' in full else bounded if bounded['shape'][1]==n else None
   if selected is None:missing.append(label+': full '+str(n)+'-row raw '+role+' values (hash-only does not supply suffix)')
   else:
    require(selected['shape']==[1,n,identities[label]['model_identity']['hidden_size']] and selected['dtype']==bounded['dtype'],'recorded full QKV shape/dtype differs')
    data=ev.tensor(base,selected);prefix,_=slice_axis(data,selected['shape'],1,0,64,DTYPES[selected['dtype']][1]);require(prefix==ev.tensor(base,bounded),'full/bounded QKV prefix differs');inputs[label][role]=(data,selected)
  for name in ('actual_attention_mask','attention_output'):
   item=entries.get(rid+'/'+name)
   if not item or item['status']!='CAPTURED':missing.append(label+': '+name+' values');continue
   data=ev.tensor(base,item);coverage[label][name]=item
   if name=='actual_attention_mask':
    require(item['shape'] in ([1,1,64,n],[1,1,n,n]),'unexpected actual additive mask shape')
    mask_checks[label]=causal_mask(data,item['shape'],item['dtype'],item['shape'][2],n)
    if item['shape']!=[1,1,n,n]:missing.append(label+': complete '+str(n)+'-query additive mask')
    else:inputs[label]['mask']=(data,item)
   else:
    require(item['shape']==[1,64,identities[label]['model_identity']['hidden_size']] and item['semantic_positions']==[0,64],'historical output position mismatch')
    inputs[label]['reference']=(data,item)
  for boundary in BOUNDARIES[:-1]:
   item=entries.get(rid+'/attention_'+boundary)
   if item and item['status']=='CAPTURED':ev.tensor(base,item)
   coverage[label]['internal_'+boundary]=item if item else 'NOT_CAPTURED'
 for role in 'qkv':
  a=coverage['S']['raw_'+role];b=coverage['F']['raw_'+role]
  require(a['dtype']==b['dtype'],'prefix QKV dtypes differ')
  observations.append(dict(boundary='raw_'+role,metrics=compare_bytes(ev.tensor(captures['F'][0],b),ev.tensor(captures['S'][0],a),a['dtype'])))
 for boundary in BOUNDARIES[:-1]:
  a=coverage['S']['internal_'+boundary];b=coverage['F']['internal_'+boundary]
  if isinstance(a,dict) and isinstance(b,dict) and a['status']==b['status']=='CAPTURED':
   require(a['dtype']==b['dtype'],'internal boundary dtype differs')
   observations.append(dict(boundary=boundary,metrics=compare_aligned((ev.tensor(captures['S'][0],a),a['shape']),(ev.tensor(captures['F'][0],b),b['shape']),a['dtype'],boundary)))
 if all('reference' in inputs[l] for l in ('S','F')):
  a,ai=inputs['S']['reference'];b,bi=inputs['F']['reference'];require(ai['dtype']==bi['dtype'],'module output dtype differs')
  observations.append(dict(boundary='module_output',metrics=compare_bytes(b,a,ai['dtype'])))
 if not fixture:
  require(all(x['metrics']['bitwise_equal'] for x in observations if x['boundary'].startswith('raw_')) or cfg['i5r2c2']['mode']=='OFF','corrected-mode prefix QKV is not bitwise equal')
 return dict(plan=p,config=cfg,inputs=inputs,identities=identities,backend=backends['S'],contexts=contexts,observations=observations,coverage=coverage,mask_checks=mask_checks,missing=missing,
  proof=dict(plan=ps,run=rs,audit=ats,historical_source=provenance),fixture=fixture)


def stage_a(experiment,run,audit,output,expected_commit,allow_incomplete=False,fixture=False):
 h=history(experiment,run,audit,expected_commit,fixture);out=fresh(output,(experiment,run,audit))
 result=verdict([],False,False,complete=not h['missing'],fixture=fixture)
 result.update(EVIDENCE_VALIDATED='CPU_FIXTURE_ONLY' if fixture else 'PASS',INTERNAL_BOUNDARIES='OBSERVED_PARTIAL_COMPARISONS_ONLY',FIRST_OBSERVED_HISTORICAL_INTERNAL_DIFFERENCE=next((x['boundary'] for x in h['observations'] if x['boundary'] in BOUNDARIES[:-1] and not x['metrics']['bitwise_equal']),None),GPU_REPLAY_READY=not h['missing'],missing=h['missing'])
 write(out/'i5r2c3_evidence_inventory.json',dict(proof=h['proof'],coverage=h['coverage'],mask_checks=h['mask_checks'],backend=h['backend'],identities=h['identities'],contexts=h['contexts']))
 write(out/'i5r2c3_tensor_comparisons.json',h['observations']);write(out/'i5r2c3_source_provenance.json',h['proof']['historical_source']);write(out/'i5r2c3_verdicts.json',result)
 (out/'i5r2c3_report.md').write_text('# C3 attention evidence review\n\nCPU structural/value checks only; no GPU FP16 arithmetic reproduction.\n\n'+'Replay readiness: '+str(result['GPU_REPLAY_READY'])+'\n\nMissing operands:\n'+''.join('- '+x+'\n' for x in h['missing'])+'\nRecover complete sealed original Layer0 QKV/mask if present. Otherwise a separately approved bounded capture of S64/F256 is required; never invent suffixes or rerun nine conditions. ROOT_CAUSE remains UNRESOLVED.\n')
 seal(out)
 require(allow_incomplete or not h['missing'],'INCOMPLETE_EVIDENCE: report sealed; use --allow-incomplete only for explicit coverage review')
 return result


def replay_operands(h,companion=None):
 if companion is not None:
  require(set(companion)=={'experiment','run','audit'} and all(companion.values()),'all three explicit B0 companion paths required')
  from .r5_i5r2c3b0 import verified_companion
  return verified_companion(companion['experiment'],companion['run'],companion['audit'],h)
 require(not h['missing'],'INCOMPLETE_EVIDENCE: replay forbidden: '+repr(h['missing']))
 return h['inputs'],None


def prepare(experiment,run,audit,base_prepared,output,expected_commit,companion=None):
 h=history(experiment,run,audit,expected_commit);inputs,companion_proof=replay_operands(h,companion)
 require(all(x['metrics']['bitwise_equal'] for x in h['observations'] if x['boundary'].startswith('raw_')),'replay requires equal prefix QKV')
 from .r3a_artifacts import validate_prepared,versions
 pm,_,_=validate_prepared(base_prepared);require(pm['identity']==h['plan']['i5r1']['model_identity'] and versions()==pm['identity']['versions'],'prepared model/environment differs')
 current=source_identity();out=fresh(output,(experiment,run,audit,base_prepared,*(companion.values() if companion else ())))
 from .r5_i5r2bs1_runtime import Store
 store=Store(out,256*2**20,192);blobs={}
 for label,entries in inputs.items():
  blobs[label]={name:dict(store.tensor('inputs/'+label+'_'+name+'.bin',data,item['shape'],item['dtype']),historical=item) for name,(data,item) in entries.items()}
 p=dict(version=VERSION,DIAGNOSTIC_ONLY=True,source=current,history=dict(experiment=str(Path(experiment).resolve()),run=str(Path(run).resolve()),audit=str(Path(audit).resolve()),expected_commit=expected_commit,proof=h['proof']),
  base_prepared=str(Path(base_prepared).resolve()),model_identity=pm['identity'],contract=h['identities']['F']['contract'],backend=h['backend'],blobs=blobs,warmups=2,repetitions=6,budget=256*2**20,max_files=192,deadline_seconds=600)
 if companion is not None:
  p['companion']=dict(paths={k:str(Path(v).resolve()) for k,v in companion.items()},proof=companion_proof)
  p['operand_provenance']='NEW_B0_RECAPTURE_VERIFIED_AGAINST_C2'
 p['experiment_id']=digest(p);store.put('i5r2c3_plan.json',p);store.writer.reconcile();seal(out);return p


def load(path,current=False):
 ev=Evidence(path,max_input_bytes=256*2**20);ev.verify();p=ev.json('i5r2c3_plan.json');q=dict(p);eid=q.pop('experiment_id')
 require(digest(q)==eid and p['version']==VERSION and p['DIAGNOSTIC_ONLY'] and p['warmups']==2 and p['repetitions']==6 and p['budget']==256*2**20 and p['max_files']==192 and p['deadline_seconds']==600,'C3 sealed plan differs')
 source_provenance(dict(source_commit=p['source']['commit'],sources=p['source']['files']),expected_commit=p['source']['commit'],fixture=False)
 if current:require(p['source']==source_identity(),'reviewed current source differs')
 h=history(**{k:p['history'][k] for k in ('experiment','run','audit','expected_commit')});require(h['proof']==p['history']['proof'],'historical evidence/provenance changed')
 inputs,proof=replay_operands(h,p.get('companion',{}).get('paths'))
 if 'companion' in p:
  require(proof==p['companion']['proof'] and p['operand_provenance']=='NEW_B0_RECAPTURE_VERIFIED_AGAINST_C2','B0 companion provenance changed')
 for label,entries in p['blobs'].items():
  require(set(entries)==set(inputs[label]),'replay operand coverage differs')
  for name,item in entries.items():require(ev.tensor('',item)==inputs[label][name][0] and item['historical']==inputs[label][name][1],'replay operands differ from verified recorded bytes')
 return p,ev
