"""Sealed C4 score controls and bounded read-only I5R-2 assessment."""
from pathlib import Path
from .communication import require,digest,digest_bytes
from .r3a_artifacts import ROOT,write,validate_prepared,versions
from .r5_i5_evidence import seal
from .r5_i5r1_plan import source_identity
from .r5_i5r2_analysis import source_provenance
from .r5_i5r2_tensors import Evidence,slice_axis
from .r5_i5r2bs1_analysis import compare_bytes,promote
from .r5_i5r2bs1_plan import CORE

VERSION='I5R2C4_SCALED_SCORE_V1'
NAMESPACE=ROOT/'results/r5/i5r/numerical_fidelity/scaled_score_isolation'
CASES={'A':(64,64),'B':(64,256),'C':(256,64),'D':(256,256)}
LAYOUTS=('NATIVE','CONTIGUOUS')
PRECISIONS=('FP16','FP32','FP32_CAST_FP16')


def fresh(path,protected=()):
 p=Path(path).resolve();require(not Path(path).is_symlink() and not p.exists(),'fresh C4 output required')
 require(not p.is_relative_to(ROOT) or p.is_relative_to(NAMESPACE),'wrong C4 namespace')
 require(all(not p.is_relative_to(Path(x).resolve()) and not Path(x).resolve().is_relative_to(p) for x in protected),'output overlaps evidence')
 p.mkdir(parents=True,mode=0o700);return p


def schedule():
 conditions=[(l,c,p) for l in LAYOUTS for c in CASES for p in ('FP16','FP32')]
 return [dict(warmup=w,repetition=i,layout=l,case=c,precision=p) for w,n in ((True,2),(False,6)) for i in range(n) for l,c,p in (conditions if i%2==0 else conditions[::-1])]


def expanded_schedule():
 return [dict(s,precision=p) for s in schedule() if not s['warmup'] for p in (('FP32','FP32_CAST_FP16') if s['precision']=='FP32' else ('FP16',))]


def history(inputs,expected):
 """Verify archived exports/Git blobs, not a current-source rerun of C3 audit."""
 from .r5_i5r2c3 import history as c2history,VERSION as C3_VERSION
 from .r5_i5r2c3b0 import verified_companion
 from .r5_i5r2c3_runtime import validate_repetitions
 require(set(inputs)=={stage+'_'+role for stage in ('c2','b0','c3') for role in ('experiment','run','audit')},'all nine explicit input paths required')
 es={k:Evidence(v,max_input_bytes=8*2**30 if k.endswith('_run') else 256*2**20) for k,v in inputs.items()};seals={k:e.verify() for k,e in es.items()}
 p=es['c3_experiment'].json('i5r2c3_plan.json');q=dict(p);eid=q.pop('experiment_id');require(digest(q)==eid,'C3 plan identity differs')
 prov=source_provenance(dict(source_commit=p['source']['commit'],sources=p['source']['files']),expected_commit=expected['c3'],fixture=False)
 require(p['version']==C3_VERSION and p['DIAGNOSTIC_ONLY'] is True and p['warmups']==2 and p['repetitions']==6 and p['budget']==256*2**20 and p['max_files']==192 and p['deadline_seconds']==600,'C3 sealed plan scope differs')
 require(p['history']['expected_commit']==expected['c2'] and p['operand_provenance']=='NEW_B0_RECAPTURE_VERIFIED_AGAINST_C2','C3 historical/companion scope differs')
 for stage in ('c2','b0'):
  declared={k:Path(v).resolve() for k,v in (p['history'] if stage=='c2' else p['companion']['paths']).items() if k in ('experiment','run','audit')}
  require(declared=={k:Path(inputs[stage+'_'+k]).resolve() for k in ('experiment','run','audit')},'explicit '+stage+' paths differ from sealed C3 chain')
 h=c2history(inputs['c2_experiment'],inputs['c2_run'],inputs['c2_audit'],expected['c2'])
 operands,proof=verified_companion(inputs['b0_experiment'],inputs['b0_run'],inputs['b0_audit'],h)
 require(proof==p['companion']['proof'] and proof['source']['commit']==expected['b0'],'B0 source/proof differs')
 require(p['model_identity']==h['plan']['i5r1']['model_identity'] and p['backend']==h['backend'] and p['contract']==h['identities']['F']['contract'],'C3/C2 identity/backend differs')
 require(set(p['blobs'])==set(operands)=={'S','F'},'C3 source/full coverage differs')
 for label in ('S','F'):
  require(set(p['blobs'][label])==set(operands[label]),'C3 operand coverage differs')
  for role,item in p['blobs'][label].items():require(es['c3_experiment'].tensor('',item)==operands[label][role][0] and item['historical']==operands[label][role][1],'C3 operand differs from B0/C2')
 run=es['c3_run'];st=run.json('i5r2c3_status.json');a=es['c3_audit'].json('i5r2c3_independent_audit.json');v=es['c3_audit'].json('i5r2c3_verdicts.json')
 require(st['status']=='COMPLETE' and not st['GPU_NOT_RUN'] and st['source_commit']==expected['c3']==p['source']['commit'] and st['experiment_id']==eid and run.json('i5r2c3_plan.json')==p,'C3 completion/source differs')
 require(a['status']=='PASS' and a['source']==p['source'] and a['inventory']==seals['c3_run'] and v['HISTORICAL_REPLAY_REPRODUCED'] is True and v['score_operand_prefixes_equal'] is True and v['FIRST_DIVERGENCE_LOCALIZED']=='scores','C3 authoritative audit differs')
 require(p['model_identity']['model']=='facebook/opt-2.7b' and p['model_identity']['hidden_size']==2560 and p['model_identity']['layers']==32 and p['model_identity']['dtype']=='float16' and p['contract']['heads']==32,'unsupported historical model dimensions')
 rows=run.json('i5r2c3_records.json');validate_repetitions(rows)
 for r in rows:
  n=64 if r['label']=='S' else 256;t=r['boundaries']['scores']
  require(t['dtype']=='float16' and t['shape']==[32,64,n] and [x['layout']['shape'] for x in t['input_receipts']]==[[1,32,n,80],[1,32,80,n]],'unsupported native C3 score/operand capture')
  for t in r['boundaries'].values():run.tensor('',t)
 env=run.json('i5r2c3_backend.json');w=run.json('i5r2c3_weights.json');fp=w['base_fingerprint']
 require(digest(fp['tensors'])==fp['sha256']==p['model_identity']['base_weight_sha256'] and w['before']==w['after'],'C3 model/weight proof differs')
 pm,_,_=validate_prepared(p['base_prepared'])
 from .gpu_identity import validate_gpu,SAME_COMPATIBLE_GPU_CLASS
 allocation=validate_gpu(pm['gpu'],env['gpu']['gpu'],'cuda:0',SAME_COMPATIBLE_GPU_CLASS)
 require(all(env['gpu'][k]==x for k,x in allocation.items()) and all(env['effective'][k]==env['after'][k]==p['backend'][k] for k in CORE),'C3 GPU/backend differs')
 require(pm['identity']==p['model_identity'],'base preparation differs')
 return dict(plan=p,rows=rows,evidence=es,seals=seals,backend=env['effective'],source=prov,c2=h)


def prepare(inputs,expected,base_prepared,output):
 h=history(inputs,expected);old=h['plan'];require(Path(base_prepared).resolve()==Path(old['base_prepared']).resolve(),'explicit base preparation differs')
 pm,_,_=validate_prepared(base_prepared);require(versions()==pm['identity']['versions'],'prepared software differs')
 p=dict(version=VERSION,DIAGNOSTIC_ONLY=True,source=source_identity(),inputs={k:str(Path(v).resolve()) for k,v in inputs.items()},expected_commits=expected,input_seals=h['seals'],base_prepared=str(Path(base_prepared).resolve()),model_identity=old['model_identity'],backend=h['backend'],schedule=schedule(),bounds=dict(bytes=256*2**20,files=256,seconds=900),scope='QK operator controls only; case C noncausal; no fidelity/performance fix')
 p['experiment_id']=digest(p);out=fresh(output,(*inputs.values(),base_prepared));write(out/'i5r2c4_plan.json',p);seal(out);return p


def load(path,current=False):
 e=Evidence(path,max_input_bytes=128*2**20);e.verify();p=e.json('i5r2c4_plan.json');q=dict(p);eid=q.pop('experiment_id')
 require(digest(q)==eid and p['version']==VERSION and p['schedule']==schedule() and p['bounds']==dict(bytes=256*2**20,files=256,seconds=900),'C4 sealed plan/bounds differs')
 source_provenance(dict(source_commit=p['source']['commit'],sources=p['source']['files']),expected_commit=p['source']['commit'],fixture=False)
 if current:require(p['source']==source_identity(),'reviewed C4 execution source differs')
 h=history(p['inputs'],p['expected_commits']);require(h['seals']==p['input_seals'] and h['plan']['model_identity']==p['model_identity'] and h['backend']==p['backend'],'C4 inputs changed')
 return p,h


def validate_records(rows):
 keys=('warmup','repetition','layout','case','precision')
 require([{k:r[k] for k in keys} for r in rows]==expanded_schedule(),'incomplete/duplicate/reordered C4 controls')
 for r in rows:
  require(r['prefix_equal'] is True and r['input_values_verified'] is True,'operand values/prefix unverified')
  require(r['output']['shape']==[32,64,64] and r['output']['dtype']==('float32' if r['precision']=='FP32' else 'float16'),'common output contract differs')


def summarize(rows,read_tensor,*,baseline,fixture=False):
 validate_records(rows);by={(r['layout'],r['case'],r['precision'],r['repetition']):r for r in rows}
 stable=[];same=[];shape=[];layout=[];precision=[]
 for l in LAYOUTS:
  for c in CASES:
   for d in PRECISIONS:
    for i in range(1,6):
     m=compare_bytes(read_tensor(by[l,c,d,0]['output']),read_tensor(by[l,c,d,i]['output']),'float32' if d=='FP32' else 'float16')
     stable.append(m['bitwise_equal']);same.append(dict(layout=l,case=c,precision=d,repetition=i,reference_repetition=0,metrics=m))
   for i in range(6):precision.append(dict(layout=l,case=c,repetition=i,metrics=compare_bytes(read_tensor(by[l,c,'FP16',i]['output']),read_tensor(by[l,c,'FP32_CAST_FP16',i]['output']),'float16')))
  for d in PRECISIONS:
   for a,b in (('A','B'),('A','C'),('B','D'),('C','D'),('A','D')):
    for i in range(6):shape.append(dict(layout=l,precision=d,pair=a+b,repetition=i,metrics=compare_bytes(read_tensor(by[l,a,d,i]['output']),read_tensor(by[l,b,d,i]['output']),'float32' if d=='FP32' else 'float16')))
 for c in CASES:
  for d in PRECISIONS:
   for i in range(6):layout.append(dict(case=c,precision=d,repetition=i,metrics=compare_bytes(read_tensor(by['NATIVE',c,d,i]['output']),read_tensor(by['CONTIGUOUS',c,d,i]['output']),'float32' if d=='FP32' else 'float16')))
 distinct=any(r.get('layout_matches_native') is False for r in rows if r['layout']=='CONTIGUOUS')
 qualified=baseline and all(stable) and not fixture;differs=lambda group:any(not x['metrics']['bitwise_equal'] for x in group)
 flags={d:differs([r for r in shape if r['layout']=='CONTIGUOUS' and r['precision']==d]) for d in PRECISIONS}
 effects=[]
 for l in LAYOUTS:
  for pair in ('AB','AC','BD','CD','AD'):
   for i in range(6):
    ms={d:next(r['metrics'] for r in shape if r['layout']==l and r['pair']==pair and r['repetition']==i and r['precision']==d) for d in PRECISIONS}
    base,cast=ms['FP16'],ms['FP32_CAST_FP16']
    effect=('NO_FP16_SHAPE_DISCREPANCY' if base['bitwise_equal'] else 'CAST_SHAPE_DISCREPANCY_ELIMINATED' if cast['bitwise_equal'] else 'CAST_SHAPE_DISCREPANCY_REDUCED' if cast['max_abs']<base['max_abs'] and cast['relative_l2']<base['relative_l2'] else 'CAST_SHAPE_DISCREPANCY_PERSISTS')
    effects.append(dict(layout=l,pair=pair,repetition=i,effect=effect if qualified else 'NOT_EVALUATED',max_abs={d:m['max_abs'] for d,m in ms.items()},relative_l2={d:m['relative_l2'] for d,m in ms.items()},bitwise_equal={d:m['bitwise_equal'] for d,m in ms.items()}))
 verdict=dict(NATIVE_C3_BASELINE_REPRODUCED=baseline,SAME_SHAPE_REPEATABILITY=all(stable),GPU_NOT_RUN=fixture,CONTROLLED_EVIDENCE_ELIGIBLE=qualified,
  shape_dependence=flags if qualified else 'NOT_EVALUATED',query_length_dependence=differs([r for r in shape if r['layout']=='CONTIGUOUS' and r['precision']=='FP16' and r['pair'] in ('AC','BD')]) if qualified else 'NOT_EVALUATED',key_length_dependence=differs([r for r in shape if r['layout']=='CONTIGUOUS' and r['precision']=='FP16' and r['pair'] in ('AB','CD')]) if qualified else 'NOT_EVALUATED',layout_control_status='COMPLETED_DIFFERENT_STRIDES' if distinct else 'UNSUPPORTED_NO_DISTINCT_LAYOUT',layout_difference=differs(layout) if qualified and distinct else 'NOT_EVALUATED',precision_difference=differs(precision) if qualified else 'NOT_EVALUATED',ROOT_CAUSE_CONFIRMED=False,KERNEL_MECHANISM='UNRESOLVED',STRICT_NUMERICAL_FIDELITY_RESTORED='NOT_EVALUATED',PERFORMANCE_BENEFIT='NOT_EVALUATED')
 return dict(verdict=verdict,same_shape=same,shape=shape,layout=layout,precision=precision,precision_shape_effects=effects)


def inspect_run(experiment,run,allow_failed=False):
 p,h=load(experiment);ev=Evidence(run,max_input_bytes=p['bounds']['bytes']);receipt=ev.verify();st=ev.json('i5r2c4_status.json')
 require(st['experiment_id']==p['experiment_id'] and st['source_commit']==p['source']['commit'] and ev.json('i5r2c4_plan.json')==p,'C4 source identity differs')
 if st['status']=='FAILED' and allow_failed:
  v=dict(CONTROLLED_EVIDENCE_ELIGIBLE=False,NATIVE_C3_BASELINE_REPRODUCED=False,reason=st.get('reason','incomplete capture'),**{k:'NOT_EVALUATED' for k in ('shape_dependence','query_length_dependence','key_length_dependence','layout_difference','precision_difference')})
  return p,h,dict(verdict=v),receipt,ev
 require(st['status']=='COMPLETE' and not st['GPU_NOT_RUN'],'C4 completion differs')
 require(len(ev.files)<=p['bounds']['files']+2 and st['model_load_count']==1 and st['model_forward_calls']==0 and st['generation_tokens']==0,'C4 execution scope differs')
 env=ev.json('i5r2c4_backend.json');fp=ev.json('i5r2c4_base_fingerprint.json')
 require(digest(fp['tensors'])==fp['sha256']==p['model_identity']['base_weight_sha256']==st['base_fingerprint_after'],'C4 base fingerprint differs')
 pm,_,_=validate_prepared(p['base_prepared'])
 from .gpu_identity import validate_gpu,SAME_COMPATIBLE_GPU_CLASS
 allocation=validate_gpu(pm['gpu'],env['gpu']['gpu'],'cuda:0',SAME_COMPATIBLE_GPU_CLASS)
 require(all(env['gpu'][k]==v for k,v in allocation.items()) and all(env['effective'][k]==env['after'][k]==p['backend'][k] for k in CORE),'C4 GPU/backend differs')
 native_contract=ev.json('native_contract/i5r2c3b0_contract_difference.json')
 require(native_contract['CONTRACT_VERIFIED'] in ('EXACT','VERIFIED_JSON_KEY_REPRESENTATION') and native_contract['physical_opt_contract_consistent'],'native OPT contract receipt unverified')
 baseline=ev.json('i5r2c4_reproduction.json');require(baseline['passed'] is True,'native baseline not reproduced')
 inputs=ev.json('i5r2c4_operands.json');c3=h['evidence']['c3_run']
 for label,n in (('S',64),('F',256)):
  old=[r for r in h['rows'] if r['label']==label]
  for j,role in enumerate(('q','kt')):
   item=inputs[label][role];data=ev.tensor('',item);axis=2 if role=='q' else 3;prefix,_=slice_axis(data,item['shape'],axis,0,64,2)
   require(item['shape']==([1,32,n,80] if role=='q' else [1,32,80,n]) and item['dtype']=='float16','recovered operand shape differs')
   require(digest_bytes(prefix)==inputs[label]['q_prefix_sha256' if role=='q' else 'k_prefix_sha256'],'recovered prefix hash metadata differs')
   require(all(digest_bytes(data)==r['boundaries']['scores']['input_receipts'][j]['full_sha256'] and digest_bytes(prefix)==r['boundaries']['scores']['input_receipts'][j]['prefix_sha256'] and item['layout']==r['boundaries']['scores']['input_receipts'][j]['layout'] for r in old),'C3 recovered value/layout receipt differs')
  native=ev.tensor('',baseline[label]['score'])
  for r in old:
   t=r['boundaries']['scores'];ref,_=slice_axis(c3.tensor('',t),t['shape'],2,0,64,2);require(native==ref and ev.tensor('',baseline[label]['native_score'])==ref and ev.tensor('',baseline[label]['module_output'])==c3.tensor('',r['boundaries']['module_output']),'C3 native score/module baseline differs')
 require(inputs['S']['q_prefix_sha256']==inputs['F']['q_prefix_sha256'] and inputs['S']['k_prefix_sha256']==inputs['F']['k_prefix_sha256'],'recovered S/F prefixes differ')
 rows=ev.json('i5r2c4_records.json');validate_records(rows);native_checks=[]
 for r in rows:
  nq,nk=CASES[r['case']];require(r['q']['shape']==[1,32,nq,80] and r['kt']['shape']==[1,32,80,nk],'matrix dimensions differ')
  require(r['q_prefix_sha256']==inputs['S']['q_prefix_sha256'] and r['k_prefix_sha256']==inputs['S']['k_prefix_sha256'],'control prefix hashes differ')
  for j,(label,role) in enumerate((('S' if nq==64 else 'F','q'),('S' if nk==64 else 'F','kt'))):
   t=inputs[label][role];raw=ev.tensor('',t);dt='float16' if r['precision']=='FP16' else 'float32'
   require(r['operand_hashes'][j]==t['sha256'] and r['execution_input_hashes'][j]==digest_bytes(raw if dt=='float16' else promote(raw)),'control values are not exact original/promoted operands')
   actual=r['q' if j==0 else 'kt'];require(actual['dtype']==dt and actual['device']=='cuda:0','control dtype/device differs')
   if r['layout']=='NATIVE':require(actual['stride']==t['layout']['stride'] and actual['storage_offset']==t['layout']['storage_offset'],'native layout differs')
   else:require(actual['contiguous'] and actual['storage_offset']==0,'materialized layout unverified')
  matches=all(r[key]['stride']==inputs[label][role]['layout']['stride'] and r[key]['storage_offset']==inputs[label][role]['layout']['storage_offset'] for key,label,role in (('q','S' if nq==64 else 'F','q'),('kt','S' if nk==64 else 'F','kt')))
  require(r['layout_matches_native']==matches,'layout distinction receipt differs')
  if r['precision']=='FP32_CAST_FP16':
   from .r5_i5r2c0_analysis import cast_half
   parent=next(x for x in rows if x['layout']==r['layout'] and x['case']==r['case'] and x['precision']=='FP32' and x['repetition']==r['repetition'])
   require(ev.tensor('',r['output'])==cast_half(ev.tensor('',parent['output'])),'output FP32-to-FP16 cast differs')
  if r['precision']=='FP16' and r['layout']=='NATIVE' and r['case'] in ('A','D'):
   native_checks.append(dict(case=r['case'],repetition=r['repetition'],metrics=compare_bytes(ev.tensor('',baseline['S' if r['case']=='A' else 'F']['score']),ev.tensor('',r['output']),'float16')))
 result=summarize(rows,lambda t:ev.tensor('',t),baseline=all(x['metrics']['bitwise_equal'] for x in native_checks));result['repeated_native_baseline_checks']=native_checks
 if not result['verdict']['CONTROLLED_EVIDENCE_ELIGIBLE']:result['verdict']['reason']='Native baseline drift or same-shape repeat variation; no controlled attribution'
 return p,h,result,receipt,ev


def analyze(experiment,run,output):
 p,h,result,receipt,ev=inspect_run(experiment,run,allow_failed=True);analysis_source=source_identity();out=fresh(output,(experiment,run,*p['inputs'].values()))
 write(out/'i5r2c4_comparisons.json',result);write(out/'i5r2c4_independent_audit.json',dict(status='PASS' if result['verdict']['CONTROLLED_EVIDENCE_ELIGIBLE'] else 'FAIL',inventory=receipt,source=p['source'],analysis_source=analysis_source,input_seals=p['input_seals']))
 seal(out);require(result['verdict']['CONTROLLED_EVIDENCE_ELIGIBLE'],'C4 controls incomplete/unstable; no attribution');return result['verdict']


def extract_values(tree,key):
 if isinstance(tree,dict):return ([tree[key]] if key in tree else [])+sum((extract_values(x,key) for x in tree.values()),[])
 if isinstance(tree,list):return sum((extract_values(x,key) for x in tree),[])
 return []


def assessment_data(h,result):
 v=result['verdict'];num=h['evidence']['c2_audit'].json('i5r2c2_numerical.json');old=h['evidence']['c2_audit'].json('i5r2c2_decision.json')
 frozen=extract_values(num,'frozen_tolerance_pass');tokens=extract_values(num,'generated_tokens_equal');texts=extract_values(num,'generated_text_equal')
 quality={c:h['evidence']['c2_run'].jsonl(c+'/quality.jsonl',optional=True) for c in ('FULL_RECOMPUTE','SEMANTIC_NO_REUSE','SEMCACHE_INTEGRATED')}
 return dict(schema='I5R2_FINAL_BOUNDED_ASSESSMENT_V1',independent_prompt_groups=1,
  observed_facts=dict(historical_reproduction={'status':'CONFIRMED','basis':'sealed C2/B0/C3 exports'},first_divergence={'status':'CONFIRMED','boundary':'scores'},strict_bitwise_fidelity={'status':'CONFIRMED','finding':'HISTORICAL_C3_SCORE_AND_ATTENTION_DIFFERENCES'},frozen_numerical_tolerance={'status':'CONFIRMED' if frozen else 'NOT_EVALUATED','all_recorded_checks_pass':all(frozen) if frozen else None},generation={'status':'CONFIRMED' if tokens and texts else 'NOT_EVALUATED','all_recorded_tokens_equal':all(tokens) if tokens else None,'all_recorded_text_equal':all(texts) if texts else None,'native_task_quality':quality},physical_reuse={'status':'CONFIRMED','scope':'HISTORICAL_C2_ONLY','verdict':old['PHYSICAL_REUSE_PRESERVED']}),
  c4_native_reproduction=dict(status='CONFIRMED' if v['NATIVE_C3_BASELINE_REPRODUCED'] else 'NOT_EVALUATED',passed=v['NATIVE_C3_BASELINE_REPRODUCED'],reason=v.get('reason')),
  controlled_findings={k:dict(status=('SUPPORTED_BUT_NOT_PROVEN' if k=='layout_difference' and v[k] is True else 'CONFIRMED') if v['CONTROLLED_EVIDENCE_ELIGIBLE'] and v[k]!='NOT_EVALUATED' else 'NOT_EVALUATED',finding=v[k]) for k in ('shape_dependence','query_length_dependence','key_length_dependence','layout_difference','precision_difference')},
  precision_shape_controls=dict(status='CONFIRMED' if v['CONTROLLED_EVIDENCE_ELIGIBLE'] else 'NOT_EVALUATED',comparisons=result.get('precision_shape_effects',[]),scope='Compare cross-shape discrepancy under original FP16, promoted FP32 and rounded FP32; no specific kernel inference'),
  hypotheses=dict(kernel_level_mechanism={'status':'UNRESOLVED'},specific_kernel_selection_or_accumulation_order={'status':'UNRESOLVED'}),
  recommendations=['No whole-model fidelity restoration or correctness-qualified speedup follows from operator controls.','Keep original frozen numerical failures visible.','Any correction needs a separate integrated generation/KV audit.','Terminate I5R-2 at bounded operator evidence; exact CUDA kernel/accumulation mechanism remains unresolved.'],STRICT_NUMERICAL_FIDELITY_RESTORED='NOT_EVALUATED',ROOT_CAUSE_CONFIRMED=False)


def final_assessment(experiment,run,audit,output):
 p,h,result,receipt,ev=inspect_run(experiment,run,allow_failed=True);ae=Evidence(audit,max_input_bytes=128*2**20);ae.verify();a=ae.json('i5r2c4_independent_audit.json')
 require(a['status']==('PASS' if result['verdict']['CONTROLLED_EVIDENCE_ELIGIBLE'] else 'FAIL') and a['inventory']==receipt and a['source']==p['source'] and a['input_seals']==p['input_seals'] and ae.json('i5r2c4_comparisons.json')==result,'C4 independent audit differs')
 source_provenance(dict(source_commit=a['analysis_source']['commit'],sources=a['analysis_source']['files']),expected_commit=a['analysis_source']['commit'],fixture=False)
 result=assessment_data(h,result);result.update(source=p['source'],analysis_source=source_identity(),input_seals=p['input_seals'],historical_sources=p['expected_commits'],input_paths=p['inputs'])
 out=fresh(output,(experiment,run,audit,*p['inputs'].values()));write(out/'i5r2_final_assessment.json',result)
 facts=result['observed_facts']
 lines=['# Bounded I5R-2 final assessment','','## Observed historical facts','',
  'Historical reproduction: CONFIRMED from sealed C2/B0/C3 exports. First divergence: scaled scores.',
  'Strict bitwise fidelity: historical score/attention differences remain.',
  'Frozen numerical checks all pass: '+str(facts['frozen_numerical_tolerance']['all_recorded_checks_pass'])+'.',
  'Generated tokens/text all equal: '+str(facts['generation']['all_recorded_tokens_equal'])+'/'+str(facts['generation']['all_recorded_text_equal'])+'.',
  'Physical reuse: historical C2 only; no fresh cache execution in C4.','','## Controlled C4 findings','',
  'Native baseline reproduction: '+str(result['c4_native_reproduction'])+'.']
 lines += [k+': '+str(v)+'.' for k,v in result['controlled_findings'].items()]
 lines += ['','Precision cross-shape effects, repeat metrics and operand identities are in the sealed JSON; general FP16/FP32 output differences are separate from cross-shape effects.','','## Hypotheses and bounded limitations','',
  'Specific CUDA kernel selection/accumulation mechanism: UNRESOLVED. ROOT_CAUSE_CONFIRMED=false.',
  'STRICT_NUMERICAL_FIDELITY_RESTORED=NOT_EVALUATED. One prompt group; no E2E speedup or generalization claim.','','## Operational recommendations','']+['- '+r for r in result['recommendations']]
 (out/'i5r2_final_assessment.md').write_text('\n'.join(lines)+'\n');seal(out);return result
