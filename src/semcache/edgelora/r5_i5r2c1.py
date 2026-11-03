"""DIAGNOSTIC_ONLY fixed-shape native FP16 control; historical references immutable."""
import os,time,math,statistics
from pathlib import Path
from itertools import combinations
from .communication import require,digest,digest_bytes
from .r3a_artifacts import ROOT,write,validate_prepared,full_base_fingerprint,tensor_bytes,tensor_from_bytes,versions
from .r5_i5_evidence import seal
from .r5_i5r1_plan import source_identity
from .r5_i5r2_analysis import source_provenance
from .r5_i5r2_tensors import Evidence,file_sha,slice_axis
from .r5_i5r2bs1_plan import validate_plan as s1plan,suffix,CORE
from .r5_i5r2bs1_runtime import Store,parameters,layout,apply_recorded_backend
from .r5_i5r2bs1_analysis import compare_bytes
from .r5_i5r2b_capture import backend

KINDS=('A','B','ZERO','NONZERO','SAME256')
VERSION='I5R2C1_FIXED_SHAPE_V1'

def fresh(path,protected=()):
 p=Path(path).resolve();require(not Path(path).is_symlink() and not p.exists(),'fresh output required')
 require(not p.is_relative_to(ROOT) or p.is_relative_to(ROOT/'results/r5/i5r/numerical_fidelity/fixed_shape_feasibility'),'wrong C1 namespace')
 require(all(not p.is_relative_to(Path(x).resolve()) and not Path(x).resolve().is_relative_to(p) for x in protected),'output overlaps evidence')
 p.mkdir(parents=True,mode=0o700);return p

def schedule():
 return [dict(warmup=w,repetition=i,kind=k) for w,n in ((True,2),(False,6)) for i in range(n) for k in (KINDS if i%2==0 else KINDS[::-1])]

def flops(rows,inputs,outputs,bias):
 # Conventional GEMM count: one multiply and one add per MAC; bias separate.
 return dict(gemm=2*rows*inputs*outputs,bias_add=rows*outputs if bias else 0,total=2*rows*inputs*outputs+(rows*outputs if bias else 0))

def operands(short,long,hidden):
 require(len(short)==64*hidden*2 and len(long)==256*hidden*2 and long[:len(short)]==short,'historical input prefix/shape differs')
 seed=43;other=suffix(short,hidden,seed)
 while other==long:seed+=1;other=suffix(short,hidden,seed)
 require(long!=short+b'\0'*(192*hidden*2),'zero padding would duplicate B: tautological primary control')
 return dict(A=short,B=long,ZERO=short+b'\0'*(192*hidden*2),NONZERO=other,SAME256=long),seed

def verified(experiment,run,audit,fixture=False):
 pe=Evidence(experiment,max_input_bytes=128*2**20);ps=pe.verify();p=s1plan(experiment,check_source=False)
 require(p['TEST_FIXTURE_ONLY']==fixture,'S1 fixture scope differs')
 prov=source_provenance(dict(source_commit=p['source']['commit'],sources=p['source']['files']),expected_commit=p['source']['commit'],fixture=fixture)
 ev=Evidence(run,max_input_bytes=512*2**20);rs=ev.verify();ae=Evidence(audit,max_input_bytes=128*2**20);ats=ae.verify()
 st=ev.json('i5r2bs1_status.json');au=ae.json('i5r2bs1_independent_audit.json')
 require(st['status']=='COMPLETE' and st['GPU_NOT_RUN']==fixture and ev.json('i5r2bs1_plan.json')==p and st['experiment_id']==p['experiment_id']==au['experiment_id'] and
  st['source_commit']==p['source']['commit'] and au['status']=='PASS' and au['TEST_FIXTURE_ONLY']==fixture and au['inventory']==rs and au['current_source']==p['source'],'S1 audit/source chain differs')
 rows=ev.json('i5r2bs1_records.json');expected=[(r,d,c,s['length'],s['round']) for r in 'qkv' for d in p['precision_conditions'] for c in p['roles'][r]['layouts'] for s in p['schedule'] if not s['warmup']]
 require([(r['role'],r['precision'],r['control'],r['length'],r['round']) for r in rows]==expected,'S1 repetitions incomplete/duplicate')
 weights=ev.json('i5r2bs1_weights.json')
 for r in 'qkv':
  w=weights[r];require(w['before']==w['after'],'historical weights mutated')
  for k,t in w['saved'].items():
   if t is not None:ev.tensor('',t);require(t['sha256']==w['before'][k]['sha256'],'historical parameter differs')
 if not fixture:
  fp=ev.json('i5r2bs1_base_fingerprint.json');require(digest(fp['tensors'])==fp['sha256']==p['model_identity']['base_weight_sha256']==st['base_fingerprint_before']==st['base_fingerprint_after'],'historical model fingerprint differs')
 return p,pe,ev,rows,weights,dict(plan=ps,run=rs,audit=ats,source=prov)

def prepare(experiment,run,audit,output,fixture=False):
 p,pe,ev,rows,w,proof=verified(experiment,run,audit,fixture)
 pm,_,_=validate_prepared(p['base_prepared']);require(pm['identity']==p['model_identity'],'prepared model differs')
 current=source_identity(fixture);out=fresh(output,(experiment,run,audit,p['base_prepared']));store=Store(out,256*2**20)
 inputs={};refs={};seeds={}
 oldinputs=ev.json('i5r2bs1_inputs.json')
 for r in 'qkv':
  h=p['roles'][r]['hidden'];group=r+'/NATIVE_FP16/CONTIGUOUS';short=ev.tensor('',oldinputs[group]['64']);long=ev.tensor('',oldinputs[group]['256'])
  require(oldinputs[group]['64']['dtype']==oldinputs[group]['256']['dtype']=='float16','historical input dtype differs')
  values,seeds[r]=operands(short,long,h)
  inputs[r]={k:store.tensor('inputs/'+r+'_'+k+'.bin',v,[1,64 if k=='A' else 256,h],'float16') for k,v in values.items()}
  refs[r]=[]
  for i in range(6):
   pair={}
   for n,label in ((64,'A'),(256,'B')):
    selected=[x for x in rows if (x['role'],x['precision'],x['control'],x['length'],x['round'])==(r,'NATIVE_FP16','CONTIGUOUS',n,i)]
    require(len(selected)==1,'six historical native repetitions required');t=selected[0]['output']
    require(t['dtype']=='float16' and t['shape']==[1,n,h],'historical output contract differs')
    data,_=slice_axis(ev.tensor('',t),t['shape'],1,0,64,2)
    pair[label]=store.tensor('references/'+r+'_'+label+'_'+str(i)+'.bin',data,[1,64,h],'float16')
   refs[r].append(pair)
  t=p['blobs'][r+'/target_base'];refs[r+'original']=store.tensor('references/'+r+'_original.bin',pe.tensor('',t),[1,64,h],'float16')
 plan=dict(version=VERSION,TEST_FIXTURE_ONLY=fixture,DIAGNOSTIC_ONLY=True,source=current,s1_experiment=str(Path(experiment).resolve()),s1_run=str(Path(run).resolve()),s1_audit=str(Path(audit).resolve()),proof=proof,
  base_prepared=p['base_prepared'],model_identity=p['model_identity'],backend=ev.json('i5r2bs1_backend.json')['effective'],weights={r:w[r]['before'] for r in 'qkv'},inputs=inputs,references=refs,
  seeds=seeds,schedule=schedule(),budget=256*2**20,deadline_seconds=900,scope='Contiguous isolated Linear hidden rows only; no tokens/masks/generation padded')
 plan['experiment_id']=digest(plan);store.put('i5r2c1_plan.json',plan);store.writer.reconcile();seal(out);return plan

def load(path):
 ev=Evidence(path,max_input_bytes=256*2**20);ev.verify();p=ev.json('i5r2c1_plan.json');q=dict(p);eid=q.pop('experiment_id')
 require(digest(q)==eid and p['version']==VERSION and p['schedule']==schedule() and p['budget']==256*2**20 and p['deadline_seconds']==900 and p['source']==source_identity(p['TEST_FIXTURE_ONLY']),'C1 plan/source differs')
 for r in 'qkv':
  h=p['model_identity']['hidden_size'];values,seed=operands(ev.tensor('',p['inputs'][r]['A']),ev.tensor('',p['inputs'][r]['B']),h)
  require(seed==p['seeds'][r] and all(ev.tensor('',p['inputs'][r][k])==v for k,v in values.items()),'padding/suffix values differ')
 return p,ev

def execute(experiment,output,commit):
 p,pe=load(experiment);require(not p['TEST_FIXTURE_ONLY'] and commit==p['source']['commit'] and os.environ.get('C1_ENABLE_GPU')=='1' and os.environ.get('SLURM_JOB_ID'),'explicit real GPU allocation/source required')
 old,_,_,_,_,proof=verified(p['s1_experiment'],p['s1_run'],p['s1_audit']);require(proof==p['proof'],'S1 evidence changed')
 pm,_,_=validate_prepared(p['base_prepared']);require(pm['identity']==p['model_identity'],'model changed')
 from .r3a_runtime import load_bare,memory
 from .r5_i5_gpu import diagnostics
 import torch,shutil
 require(os.environ.get('HF_HUB_OFFLINE')==os.environ.get('TRANSFORMERS_OFFLINE')=='1','offline assets required')
 require(int(os.environ.get('SLURM_CPUS_PER_TASK','0'))>=4 and int(os.environ.get('SLURM_MEM_PER_NODE','0'))>=65536,'four cores/64 GiB allocation required')
 available=next(int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:'))
 require(available>=32*2**30 and shutil.disk_usage(Path(experiment).parent).free>2**30,'CPU/disk reserve unavailable')
 gpu=diagnostics(pm);require(gpu['eligible'] and gpu['gpu']['free_bytes']>5400000000+5*2**30,'one compatible RTX 3090 with memory headroom required')
 torch.cuda.set_device(0);initial=backend();effective=apply_recorded_backend(p['backend']);require(versions()==pm['identity']['versions'],'runtime versions differ')
 out=fresh(output,(experiment,p['s1_run'],p['s1_audit'],p['base_prepared']));store=Store(out,p['budget']);status=dict(status='FAILED',GPU_NOT_RUN=True);rows=[];start=time.monotonic()
 try:
  model,tok,meta=load_bare(pm['identity'],'cuda:0')
  from semcache.models.model_adapter import OPTModelAdapter
  from semcache.system_cost.es_base_profile import assert_no_lora
  adapter=OPTModelAdapter(model);assert_no_lora(model,adapter);fp=full_base_fingerprint(model)
  require(fp['sha256']==pm['identity']['base_weight_sha256'],'loaded base differs')
  with torch.inference_mode():
   for r,m in adapter.projection_modules(0).items():
    require(type(m) is torch.nn.Linear and 'forward' not in m.__dict__ and not m._forward_hooks and not m._forward_pre_hooks and m.weight.dtype==torch.float16,'actual native FP16 Linear required')
    original=parameters(m);require(original==p['weights'][r],'historical native parameters differ')
    xs={k:tensor_from_bytes(pe.tensor('',v),'float16',v['shape']).to('cuda:0') for k,v in p['inputs'][r].items()}
    for step in p['schedule']:
     k=step['kind'];x=xs[k];require(time.monotonic()-start<900,'bounded runtime exceeded')
     before=parameters(m);inp=digest_bytes(tensor_bytes(x));prefix=digest_bytes(tensor_bytes(x[:,:64]))
     require(before==original and inp==p['inputs'][r][k]['sha256'] and prefix==p['inputs'][r]['A']['sha256'] and all(backend()[c]==effective[c] for c in CORE) and memory('cuda:0')['uuid']==gpu['execution_gpu_uuid'],'precall identity differs')
     torch.cuda.synchronize();a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);a.record();y=m.forward(x);b.record();b.synchronize();ms=a.elapsed_time(b)
     status['GPU_NOT_RUN']=False
     require(parameters(m)==original and digest_bytes(tensor_bytes(x))==inp,'operands changed')
     if not step['warmup']:
      t=store.tensor('outputs/'+r+'_'+k+'_'+str(step['repetition'])+'.bin',tensor_bytes(y[:,:64]),[1,64,m.out_features],'float16',layout(y))
      rows.append(dict(step,role=r,output=t,input_layout=layout(x),operator_cuda_ms=ms,input_sha256=inp,prefix_sha256=prefix,weight=original,gpu_uuid=gpu['execution_gpu_uuid'],backend={c:effective[c] for c in CORE}))
  require(full_base_fingerprint(model)==fp and load(experiment)[0]==p,'model/source changed')
  store.put('i5r2c1_plan.json',p);store.put('i5r2c1_records.json',rows);store.put('i5r2c1_environment.json',dict(initial=initial,effective=effective,after=backend(),gpu=gpu,loader=meta,model_load_count=1,full_fingerprint=fp))
  status.update(status='COMPLETE',experiment_id=p['experiment_id'])
 except BaseException as exc:status['reason']=str(exc);raise
 finally:store.put('i5r2c1_status.json',status);store.writer.reconcile();seal(out)


def analyze(experiment,run,output):
 p,pe=load(experiment);ev=Evidence(run,max_input_bytes=256*2**20);ev.verify();st=ev.json('i5r2c1_status.json')
 require(st['status']=='COMPLETE' and ev.json('i5r2c1_plan.json')==p and st['experiment_id']==p['experiment_id'] and st['GPU_NOT_RUN']==p['TEST_FIXTURE_ONLY'],'C1 run incomplete/mismatched')
 oldplan,oldpe,oldev,oldrows,_,proof=verified(p['s1_experiment'],p['s1_run'],p['s1_audit'],p['TEST_FIXTURE_ONLY']);require(proof==p['proof'],'historical evidence changed')
 rows=ev.json('i5r2c1_records.json');require([(x['role'],x['kind'],x['repetition']) for x in rows]==[(r,s['kind'],s['repetition']) for r in 'qkv' for s in p['schedule'] if not s['warmup']],'missing/duplicate/reordered repetitions')
 env=ev.json('i5r2c1_environment.json')
 if not p['TEST_FIXTURE_ONLY']:
  from .gpu_identity import validate_gpu,SAME_COMPATIBLE_GPU_CLASS
  pm,_,_=validate_prepared(p['base_prepared']);require(pm['identity']==p['model_identity'],'base preparation changed')
  receipt=validate_gpu(pm['gpu'],env['gpu']['gpu'],'cuda:0',SAME_COMPATIBLE_GPU_CLASS)
  require(all(env['gpu'][k]==v for k,v in receipt.items()),'GPU identity differs')
 fp=env['full_fingerprint'];require(env['model_load_count']==1 and digest(fp['tensors'])==fp['sha256']==p['model_identity']['base_weight_sha256'],'model fingerprint differs')
 require(all(env['effective'][c]==env['after'][c]==p['backend'][c] for c in CORE),'backend changed')
 results=[];cost={};repeatability={}
 for r in 'qkv':
  w=p['weights'][r];i,o=w['weight']['shape'][1],w['weight']['shape'][0]
  for name,item in w.items():
   if item is not None:require(item['sha256']==fp['tensors']['model.decoder.layers.0.self_attn.'+r+'_proj.'+name]['sha256'],'projection fingerprint differs')
  oldinputs=oldev.json('i5r2bs1_inputs.json')[r+'/NATIVE_FP16/CONTIGUOUS']
  require(all(pe.tensor('',p['inputs'][r][k])==oldev.tensor('',oldinputs['64' if k=='A' else '256']) for k in ('A','B')),'prepared S1 inputs differ')
  cost[r]=dict(native=flops(64,i,o,w['bias'] is not None),fixed=flops(256,i,o,w['bias'] is not None),extra=flops(192,i,o,w['bias'] is not None),ratio=4,
   timing_ms={k:[x['operator_cuda_ms'] for x in rows if x['role']==r and x['kind']==k] for k in KINDS})
  cost[r]['timing_summary_ms']={k:dict(mean=statistics.mean(v),median=statistics.median(v),samples=len(v)) for k,v in cost[r]['timing_ms'].items()}
  selected=[x for x in rows if x['role']==r]
  for x in selected:
   require(x['input_sha256']==p['inputs'][r][x['kind']]['sha256'] and x['prefix_sha256']==p['inputs'][r]['A']['sha256'] and x['weight']==w and x['gpu_uuid']==env['gpu']['execution_gpu_uuid'] and x['backend']=={c:env['effective'][c] for c in CORE},'operand receipt differs')
   require(x['output']['shape']==[1,64,o] and x['output']['dtype']=='float16' and math.isfinite(x['operator_cuda_ms']) and x['operator_cuda_ms']>=0,'output/timing contract differs')
  repeatability[r]={k:all(ev.tensor('',a['output'])==ev.tensor('',b['output']) for a,b in combinations([x for x in selected if x['kind']==k],2)) for k in KINDS}
  for n in range(6):
   current={x['kind']:ev.tensor('',x['output']) for x in selected if x['repetition']==n};ref=p['references'][r][n];a,b=pe.tensor('',ref['A']),pe.tensor('',ref['B'])
   for length,label in ((64,'A'),(256,'B')):
    original=next(x for x in oldrows if (x['role'],x['precision'],x['control'],x['length'],x['round'])==(r,'NATIVE_FP16','CONTIGUOUS',length,n))
    require(pe.tensor('',ref[label])==oldev.tensor('',original['output'])[:64*o*2],'historical reference changed')
   for k in ('ZERO','NONZERO'):
    m=compare_bytes(b,current[k],'float16');old=compare_bytes(a,b,'float16')
    results.append(dict(role=r,repetition=n,kind=k,historical_A_vs_B=old,fixed_vs_unchanged_S1_B=m,
     fixed_vs_original_base=compare_bytes(pe.tensor('',p['references'][r+'original']),current[k],'float16'),
     current_A_vs_S1_A=compare_bytes(a,current['A'],'float16'),current_B_vs_S1_B=compare_bytes(b,current['B'],'float16'),same_input_sanity=compare_bytes(current['B'],current['SAME256'],'float16'),
     FIXED_SHAPE_PREFIX_EXACT=m['bitwise_equal'],FIXED_SHAPE_PREFIX_DIFFERENT=not m['bitwise_equal'],
     FIXED_SHAPE_PREFIX_CLOSER=m['max_abs']<=old['max_abs'] and m['relative_l2']<=old['relative_l2'] and (m['max_abs']<old['max_abs'] or m['relative_l2']<old['relative_l2'])))
 out=fresh(output,(experiment,run,p['s1_run'],p['s1_audit']));write(out/'i5r2c1_comparisons.json',results)
 write(out/'i5r2c1_cost.json',dict(per_role=cost,QKV_extra_flops=sum(x['extra']['total'] for x in cost.values()),scope='2*MAC plus bias adds; CUDA events isolate forward, exclude copies/hash/I/O; no E2E claim; source amortization needs integrated measurement'))
 summary=dict(INCOMPLETE_EVIDENCE=False,TEST_FIXTURE_ONLY=p['TEST_FIXTURE_ONLY'],GPU_DIAGNOSTIC_COMPLETE=not p['TEST_FIXTURE_ONLY'],repeatability=repeatability,ROOT_CAUSE_CONFIRMED=False,NUMERICAL_FIDELITY_RESTORED='NOT_EVALUATED',scope='Contiguous isolated Linear only; historical S1 suffix may be synthetic')
 write(out/'i5r2c1_summary.json',summary);write(out/'i5r2c1_provenance.json',dict(current=p['source'],historical=p['proof'],independent_audit='PASS'));seal(out);return summary
