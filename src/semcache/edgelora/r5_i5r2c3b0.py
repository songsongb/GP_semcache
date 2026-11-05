"""Bounded, independently gated recapture; original C2 evidence is immutable."""
from copy import deepcopy
from pathlib import Path
from .communication import require,digest,digest_bytes
from .r3a_artifacts import ROOT,write,versions
from .r5_i5_evidence import seal
from .r5_i5r1_plan import source_identity
from .r5_i5r2_analysis import source_provenance
from .r5_i5r2_tensors import Evidence,DTYPES,slice_axis
from .r5_i5r2c3 import history,causal_mask
from .r5_i5r2bs1_analysis import compare_bytes
from .r5_multi_artifacts import validate,runtime_view

VERSION='I5R2C3_B0_ACTUAL_OPERANDS_V1'
ORIGIN='NEW_B0_RECAPTURE_VERIFIED_AGAINST_C2'
NAMESPACE=ROOT/'results/r5/i5r/numerical_fidelity/attention_operand_capture'


def fresh(path,protected=()):
 p=Path(path).resolve();require(not Path(path).is_symlink() and not p.exists(),'fresh B0 output required')
 require(not p.is_relative_to(ROOT) or p.is_relative_to(NAMESPACE),'wrong B0 namespace')
 require(all(not p.is_relative_to(Path(x).resolve()) and not Path(x).resolve().is_relative_to(p) for x in protected),'B0 output overlaps evidence')
 p.mkdir(parents=True,mode=0o700);return p


def anchor(h):
 return dict(plan=h['proof']['plan'],run=h['proof']['run'],audit=h['proof']['audit'],
  historical_commit=h['config']['source_commit'],historical_sources=h['config']['sources'])


def prepared(h,multi):
 m,pm,specs,registry=validate(multi)
 require(pm['identity']==h['plan']['i5r1']['model_identity'] and specs['Adapter_A']==h['identities']['S']['adapter'] and
  m['tokenizer']==h['identities']['S']['tokenizer'],'B0 prepared model/adapter/tokenizer differs')
 return m,pm,specs,registry


def prepare(experiment,run,audit,multi,output,expected_commit):
 h=history(experiment,run,audit,expected_commit);require(h['config']['i5r2c2']['mode']=='SOURCE_ONLY','B0 requires historical SOURCE_ONLY')
 m,pm,specs,registry=prepared(h,multi);require(versions()==pm['identity']['versions'],'B0 prepared/runtime software differs');current=source_identity()
 from .r4c_workload import policy
 tasks=deepcopy(h['config']['block']['tasks'])
 for t in tasks:
  # One genuine greedy selection completes the frozen TCP lifecycle; no decode.
  t['policy']=policy({k:t['policy'][k] for k in ('bos_token_id','eos_token_id','pad_token_id')},t['policy']['mode'],1)
 p=dict(version=VERSION,DIAGNOSTIC_ONLY=True,TEST_FIXTURE_ONLY=False,source=current,
  history=dict(experiment=str(Path(experiment).resolve()),run=str(Path(run).resolve()),audit=str(Path(audit).resolve()),expected_commit=expected_commit,anchor=anchor(h)),
  multi_prepared=str(Path(multi).resolve()),base_prepared=m['base_prepared'],model_identity=pm['identity'],adapter=specs['Adapter_A'],registry=registry,
  tokenizer=m['tokenizer'],contract=h['identities']['S']['contract'],backend=h['backend'],tasks=tasks,
  historical_contexts=h['contexts'],budget=256*2**20,max_files=192,deadline_seconds=900,
  scope='Two actual full-model prefills, one model/UD, no decode/admission/reuse; only Layer0 tensor bytes saved')
 p['experiment_id']=digest(p);out=fresh(output,(experiment,run,audit,multi));write(out/'i5r2c3b0_plan.json',p);seal(out);return p


def load(experiment,current=False,fixture=False):
 pe=Evidence(experiment,max_input_bytes=128*2**20);pe.verify();p=pe.json('i5r2c3b0_plan.json');q=dict(p);eid=q.pop('experiment_id')
 require(digest(q)==eid and p['version']==VERSION and p['DIAGNOSTIC_ONLY'] and p['TEST_FIXTURE_ONLY']==fixture and p['budget']==256*2**20 and p['max_files']==192 and p['deadline_seconds']==900,'B0 plan/bounds/scope differs')
 source_provenance(dict(source_commit=p['source']['commit'],sources=p['source']['files']),expected_commit=p['source']['commit'],fixture=fixture)
 if current:require(p['source']==source_identity(fixture),'B0 reviewed execution source differs')
 h=history(**{k:p['history'][k] for k in ('experiment','run','audit','expected_commit')},fixture=fixture)
 require(anchor(h)==p['history']['anchor'] and h['config']['i5r2c2']['mode']=='SOURCE_ONLY','original C2 anchor changed')
 require(p['model_identity']==h['plan']['i5r1']['model_identity'] and p['adapter']==h['identities']['S']['adapter'] and p['backend']==h['backend'] and p['contract']==h['identities']['S']['contract'],'B0 historical identity/backend differs')
 old=h['config']['block']['tasks'];require(len(p['tasks'])==2,'one B0 pair required')
 from .r4c_workload import policy
 for task,original in zip(p['tasks'],old):
  changed=deepcopy(original);changed['policy']=policy({k:original['policy'][k] for k in ('bos_token_id','eos_token_id','pad_token_id')},original['policy']['mode'],1)
  require(task==changed,'B0 token/pair/routing/prefill policy differs')
 return p,pe,h


def compare_capture(h,ev,records,*,fixture=False):
 """Pure CPU identity/value gate; never infers absent operands."""
 require(set(records)=={'S','F'},'B0 operand coverage differs');checks=[];recovered={}
 from .r5_i5r2bs1_plan import CORE
 for label,n in (('S',64),('F',256)):
  r=records[label];ctx=r['context'];old=h['contexts'][label];ident=r['identity']
  require(ident['model_identity']==h['identities'][label]['model_identity'] and ident['adapter']==h['identities'][label]['adapter'] and ident['contract']==h['identities'][label]['contract'],'new B0 model/adapter/contract differs')
  physical=ident['physical_identity'];previous=h['identities'][label]['physical_identity']
  require(set(physical)==set(previous) and all(physical[k]==previous[k] for k in physical if k!='gpu_uuid'),'new B0 physical namespace differs')
  fields=('input_token_ids','target_token_ids','positions','cache_positions','attention_mask','past_length','pass_index','input_length','mode','phase')
  require(all(ctx[k]==old[k] for k in fields),'new B0 token/position/mask context differs')
  obs=r['position_observation'];require(obs['positions']==old['positions'] and obs['past_length']==0 and (fixture or obs['source']=='ACTUAL_OPT_POSITION_MODULE' and obs['lookup_output_equal'] and obs['attention_mask_sha256']==digest([1]*n)),'new B0 position observation differs')
  require(r['backend_before']==r['backend_after'],'new B0 backend changed')
  require(r['backend_before']==h['backend'] if fixture else all(r['backend_before'][k]==h['backend'][k] for k in (*CORE,'torch','transformers','cuda_compiled_version')),'new B0 backend differs')
  require(set(r['tensors'])==set('qkv')|{'mask','reference'},'B0 missing/unexpected tensors')
  recovered[label]={}
  for role,item in r['tensors'].items():
   expected=[1,1,n,n] if role=='mask' else [1,64,h['plan']['i5r1']['model_identity']['hidden_size']] if role=='reference' else [1,n,h['plan']['i5r1']['model_identity']['hidden_size']]
   require(item['shape']==expected and item['dtype']=='float16' and item['semantic_positions']==[0,64 if role=='reference' else n],'B0 tensor shape/dtype/position differs')
   if not fixture:
    observed=[1,1,n,n] if role=='mask' else [1,n,h['plan']['i5r1']['model_identity']['hidden_size']]
    require(item['observed_shape']==observed and len(item['observed_stride'])==len(observed) and all(type(v) is int and v>=0 for v in item['observed_stride']) and type(item['storage_offset']) is int and item['storage_offset']>=0 and type(item['original_contiguous']) is bool and item['layout']['device']=='cuda:0' and item['layout']['dtype']=='float16','actual B0 tensor layout/device differs')
   data=ev.tensor('',item);recovered[label][role]=(data,item)
   if role=='mask':
    mask_receipt=causal_mask(data,item['shape'],item['dtype'],n,n)
    olditem=h['coverage'][label].get('actual_attention_mask');require(isinstance(olditem,dict),'original C2 actual mask missing')
    current,_=slice_axis(data,item['shape'],2,0,64,2)
    oldev=h['historical_evidence']
    before=oldev.tensor(('SEMCACHE_INTEGRATED' if label=='S' else 'FULL_RECOMPUTE')+'/Adapter_A/es',olditem)
    if olditem['shape'][2]!=64:before,_=slice_axis(before,olditem['shape'],2,0,64,2)
    require(olditem['dtype']==item['dtype'],'historical/current mask dtype differs')
    checks.append(dict(label=label,boundary='mask',bitwise_equal=current==before,complete_causal_mask=mask_receipt))
   elif role=='reference':
    before=h['inputs'][label]['reference'][0];m=compare_bytes(before,data,item['dtype']);checks.append(dict(label=label,boundary=role,bitwise_equal=m['bitwise_equal'],metrics=m))
   else:
    current,_=slice_axis(data,item['shape'],1,0,64,2);olditem=h['coverage'][label]['raw_'+role]
    oldev=h['historical_evidence'];base=('SEMCACHE_INTEGRATED' if label=='S' else 'FULL_RECOMPUTE')+'/Adapter_A/es'
    require(olditem['dtype']==item['dtype'],'historical/current QKV dtype differs')
    m=compare_bytes(oldev.tensor(base,olditem),current,item['dtype']);checks.append(dict(label=label,boundary=role,bitwise_equal=m['bitwise_equal'],metrics=m))
 cross=[]
 for role in 'qkv':
  a,ai=recovered['S'][role];b,bi=recovered['F'][role];b,_=slice_axis(b,bi['shape'],1,0,64,2)
  cross.append(dict(role=role,metrics=compare_bytes(a,b,'float16')))
 prefix=all(c['bitwise_equal'] for c in checks if c['boundary'] in 'qkv') and all(c['metrics']['bitwise_equal'] for c in cross)
 attention=all(c['bitwise_equal'] for c in checks if c['boundary']=='reference')
 masks=all(c['bitwise_equal'] for c in checks if c['boundary']=='mask')
 ready=prefix and attention and masks
 return dict(NEW_CAPTURE_COMPLETE=True,B0_CAPTURE_COMPLETE=True,HISTORICAL_C2_PREFIX_REPRODUCED=prefix,
  HISTORICAL_C2_ATTENTION_REPRODUCED=attention,HISTORICAL_C2_MASK_REPRODUCED=masks,
  COMPANION_REPLAY_ELIGIBLE=ready and not fixture,C3_COMPANION_READY=ready and not fixture,
  HISTORICAL_REPLAY_NOT_ESTABLISHED=not ready or fixture,TEST_FIXTURE_ONLY=fixture,
  FIRST_ATTENTION_DIVERGENCE='NOT_EVALUATED',STRICT_NUMERICAL_FIDELITY_RESTORED='NOT_EVALUATED',ROOT_CAUSE_CONFIRMED=False,
  checks=checks,new_source_vs_full=cross,provenance=ORIGIN if ready and not fixture else 'NEW_CAPTURE_NOT_AUTHORIZED_FOR_HISTORICAL_LOCALIZATION'),recovered


def inspect_run(experiment,run,fixture=False):
 p,pe,h=load(experiment,fixture=fixture);ev=Evidence(run,max_input_bytes=p['budget']);rs=ev.verify();st=ev.json('i5r2c3b0_status.json')
 require(st['status']=='COMPLETE' and st['GPU_NOT_RUN']==fixture and st['source_commit']==p['source']['commit'] and st['experiment_id']==p['experiment_id'] and ev.json('i5r2c3b0_plan.json')==p,'B0 run completion/source identity differs')
 require(st['model_load_count']==1 and st['prefill_forwards']==2 and st['decode_forwards']==st['cache_reads']==st['cache_admissions']==0 and st['protocol_selected_tokens']==2,'B0 execution scope differs')
 require(len(ev.files)<=p['max_files']+2,'B0 file budget exceeded')
 h['historical_run']=p['history']['run'];h['historical_evidence']=Evidence(h['historical_run'],max_input_bytes=8*2**30)
 records=ev.json('i5r2c3b0_records.json')
 if not fixture:
  from .r5_i5r2c3b0_runtime import audit_receipts
  audit_receipts(p,ev,records,h)
 result,inputs=compare_capture(h,ev,records,fixture=fixture)
 return p,h,result,inputs,rs


def analyze(experiment,run,output):
 p,h,result,inputs,rs=inspect_run(experiment,run)
 require(p['source']==source_identity(),'B0 audit requires the reviewed capture source snapshot')
 out=fresh(output,(experiment,run,p['history']['run'],p['history']['audit']))
 write(out/'i5r2c3b0_comparisons.json',dict(checks=result['checks'],source_vs_full=result['new_source_vs_full']))
 write(out/'i5r2c3b0_decision.json',result)
 write(out/'i5r2c3b0_independent_audit.json',dict(status='PASS' if result['COMPANION_REPLAY_ELIGIBLE'] else 'HISTORICAL_REPLAY_NOT_ESTABLISHED',inventory=rs,experiment_id=p['experiment_id'],source=p['source'],historical_anchor=p['history']['anchor']))
 (out/'i5r2c3b0_report.md').write_text('# B0 operand recapture\n\nCompanion replay eligible: '+str(result['COMPANION_REPLAY_ELIGIBLE'])+'\n\nNew B0 operands remain separate from immutable C2 evidence. First attention divergence and strict fidelity restoration NOT_EVALUATED; no performance claim.\n')
 seal(out);require(result['COMPANION_REPLAY_ELIGIBLE'],'HISTORICAL_REPLAY_NOT_ESTABLISHED: audit sealed; C3 companion forbidden');return result


def verified_companion(experiment,run,audit,h):
 p,old,result,inputs,rs=inspect_run(experiment,run)
 require(result['COMPANION_REPLAY_ELIGIBLE'] and anchor(h)==p['history']['anchor'],'B0 companion does not reproduce this original C2 run')
 ae=Evidence(audit,max_input_bytes=128*2**20);ats=ae.verify();a=ae.json('i5r2c3b0_independent_audit.json')
 require(a['status']=='PASS' and a['inventory']==rs and a['experiment_id']==p['experiment_id'] and a['source']==p['source'] and a['historical_anchor']==anchor(h) and ae.json('i5r2c3b0_decision.json')==result,'B0 exported independent audit differs')
 output={}
 for label in ('S','F'):
  output[label]={role:(data,dict(item,capture_origin=ORIGIN)) for role,(data,item) in inputs[label].items() if role!='reference'}
  output[label]['reference']=h['inputs'][label]['reference']  # ORIGINAL C2 reference, never replaced.
 proof=dict(provenance=ORIGIN,plan=Evidence(experiment).verify(),run=rs,audit=ats,source=p['source'],historical_reference='ORIGINAL_C2_MODULE_OUTPUT')
 return output,proof
