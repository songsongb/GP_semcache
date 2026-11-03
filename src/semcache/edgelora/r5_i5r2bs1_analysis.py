"""CPU read-only audit of isolated Linear operands, repeats and cross-shape errors."""
from itertools import combinations
import json
import struct

from .communication import digest,digest_bytes,require
from .r3a_artifacts import THRESHOLDS,pack,unpack
from .gpu_identity import validate_gpu,SAME_COMPATIBLE_GPU_CLASS
from .r5_i5_evidence import seal
from .r5_i5r2_tensors import Evidence,metrics,slice_axis
from .r5_i5r2bs1_plan import CORE,validate_plan,historical,fresh,layout_storage,verify_history_inputs


def promote(data):
    return b''.join(pack(unpack(data[i:i+131072],'float16'),'float32') for i in range(0,len(data),131072))


def compare_bytes(a,b,dtype):
    m=metrics(a,b,dtype)
    if dtype=='float16':
        def ordered(u):return 0x8000-(u&0x7fff) if u&0x8000 else 0x8000+u
        ulps=[abs(ordered(x[0])-ordered(y[0])) for x,y in zip(struct.iter_unpack('<H',a),struct.iter_unpack('<H',b))]
        m['FP16_ULP']=dict(max=max(ulps),mean=sum(ulps)/len(ulps),nonzero=sum(x>0 for x in ulps),
            convention='Monotone finite IEEE binary16 encoding; signed zero distance zero; NaN/Inf rejected')
    m['frozen_max_abs_pass']=m['max_abs']<=THRESHOLDS['max_abs'];m['frozen_relative_l2_pass']=m['relative_l2']<=THRESHOLDS['relative_l2']
    return m


def verdicts(same,cross,*,fixture=False,complete=True):
    stable=lambda dtype:all(x['metrics']['bitwise_equal'] for x in same if x['precision']==dtype)
    difference=lambda dtype:any(not x['metrics']['bitwise_equal'] for x in cross if x['precision']==dtype)
    complete=complete and all(any(x['precision']==d for x in same) and any(x['precision']==d for x in cross) for d in ('NATIVE_FP16','DERIVED_FP32'))
    s16,s32=stable('NATIVE_FP16'),stable('DERIVED_FP32');d16,d32=difference('NATIVE_FP16'),difference('DERIVED_FP32')
    case='CASE_F_INCOMPLETE' if not complete else 'CASE_B_REPEAT_VARIATION_CONFOUNDED' if not s16 else (
        'CASE_E_FP16_DIFFERENCE_NOT_REPRODUCED' if not d16 else 'CASE_B_FP32_REPEAT_VARIATION_CONFOUNDED' if not s32 else
        'CASE_C_PRECISION_SENSITIVE_SHAPE_EFFECT' if not d32 else 'CASE_D_BOTH_PRECISIONS_SHAPE_SENSITIVE')
    return dict(case=case,SAME_SHAPE_REPEATABILITY=dict(FP16='STABLE' if s16 and complete else 'UNSTABLE_OR_INCOMPLETE',
        FP32='STABLE' if s32 and complete else 'UNSTABLE_OR_INCOMPLETE'),CONTROLLED_FP16_SHAPE_DIFFERENCE=d16 if complete else 'UNKNOWN',
        CONTROLLED_FP32_SHAPE_DIFFERENCE=d32 if complete else 'UNKNOWN',
        SHAPE_DEPENDENCE_DEMONSTRATED='CPU_FIXTURE_ONLY' if fixture else bool(complete and s16 and d16),
        HISTORICAL_DIVERGENCE_REPRODUCED='NOT_ESTABLISHED: suffix/layout/allocator/scheduling/hardware context is not a full replay',
        CACHE_RELATED_CAUSALITY='NOT_TESTED: no cache operation in this control',ROOT_CAUSE_CONFIRMED=False,ROOT_CAUSE='UNRESOLVED',
        NUMERICAL_FIDELITY_RESTORED='NOT_EVALUATED',GPU_NOT_RUN=fixture,performance_claims='PROHIBITED_DIAGNOSTIC_ONLY',
        interpretation='Stable identical-input/weight cross-shape variation demonstrates behavior of this operation; no cuBLAS algorithm or accumulation mechanism is identified.')


def analyze(experiment,run,output):
    plan=validate_plan(experiment);ev=Evidence(run,max_input_bytes=512*2**20);seal_receipt=ev.verify()
    status=ev.json('i5r2bs1_status.json');require(status['status']=='COMPLETE' and ev.json('i5r2bs1_plan.json')==plan and
        status['experiment_id']==plan['experiment_id'] and status['source_commit']==plan['source']['commit'],'S1 incomplete/mixed-source run')
    require(ev.total_bytes<=plan['bounds']['evidence_bytes'] and len(ev.files)<=plan['bounds']['max_files']+2,'S1 evidence bound exceeded')
    fixture=plan['TEST_FIXTURE_ONLY'];require(status['GPU_NOT_RUN']==fixture,'S1 fixture/GPU scope differs')
    require(status['model_load_count']==1 and all(status[k]==0 for k in ('production_forward_calls','LoRA_calls','TCP_frames','cache_reads','generation_tokens')),
        'not an isolated single-model Linear experiment')
    history=historical(plan['historical_run_dir'],plan['historical_review_dir'],expected_commit=plan['historical_source_commit'],fixture=fixture)
    require(history['run_seal']==plan['historical_run_seal'] and history['review_seal']==plan['historical_review_seal'],'historical evidence changed')
    env=ev.json('i5r2bs1_backend.json');weights=ev.json('i5r2bs1_weights.json');inputs=ev.json('i5r2bs1_inputs.json');rows=ev.json('i5r2bs1_records.json')
    if not fixture:
        from .r3a_artifacts import validate_prepared
        pm,_,_=validate_prepared(plan['base_prepared']);fp=ev.json('i5r2bs1_base_fingerprint.json')
        require(pm['identity']==plan['model_identity'] and digest(fp['tensors'])==fp['sha256']==pm['identity']['base_weight_sha256']==
                status['base_fingerprint_before']==status['base_fingerprint_after'],'base model fingerprint/proof differs')
        gpu=env['gpu'];allocation=validate_gpu(pm['gpu'],gpu['gpu'],'cuda:0',SAME_COMPATIBLE_GPU_CLASS)
        require(all(gpu[k]==value for k,value in allocation.items()),'execution GPU receipt differs')
        require(all(env['effective'][k]==env['after'][k]==plan['historical_backend'][k] for k in CORE),'backend settings changed or differ from baseline')
    expected=[(r,p,c,s['length'],s['round']) for r in 'qkv' for p in plan['precision_conditions'] for c in plan['roles'][r]['layouts']
        for s in plan['schedule'] if not s['warmup']]
    require([(x['role'],x['precision'],x['control'],x['length'],x['round']) for x in rows]==expected,'missing/duplicate/reordered measured operations')
    pe=Evidence(experiment);pe.verify();verify_history_inputs(plan,pe,history)
    same=[];cross=[];historical_comparisons=[];layout_comparisons=[]
    for role in 'qkv':
        w=weights[role];require(w['before']==w['after'] and w['module_class']=='Linear','native projection identity/immutability differs')
        if not fixture:require(w['module']=='model.decoder.layers.0.self_attn.'+role+'_proj','not the actual Layer-0 projection')
        converted={}
        for name,item in w['saved'].items():
            if item is None:require(w['before'][name] is None,'missing native parameter');converted[name]=None;continue
            raw=ev.tensor('',item);unpack(raw,'float16')
            require(item['dtype']=='float16' and item['shape']==w['before'][name]['shape'] and item['sha256']==w['before'][name]['sha256'],
                'native saved parameter differs')
            if not fixture:require(item['sha256']==fp['tensors'][w['module']+'.'+name]['sha256'],'parameter/base fingerprint differs')
            converted[name]=dict(dtype='float32',shape=item['shape'],sha256=digest_bytes(promote(raw)))
        require(w['NATIVE_FP16_temporary_parameter_hashes']==w['before'] and w['DERIVED_FP32_temporary_parameter_hashes']==converted,
            'temporary FP32 operands are not exact promotions of native parameters')
        for precision in plan['precision_conditions']:
            dtype='float16' if precision=='NATIVE_FP16' else 'float32';width=2 if dtype=='float16' else 4
            active=w['before'] if dtype=='float16' else converted
            for control in plan['roles'][role]['layouts']:
                group=role+'/'+precision+'/'+control;xs=inputs[group];hidden=plan['roles'][role]['hidden']
                expected_long=pe.tensor('',plan['blobs'][role+'/input_long']);expected_long=expected_long if width==2 else promote(expected_long)
                require(ev.tensor('',xs['256'])==expected_long and ev.tensor('',xs['64'])==expected_long[:64*hidden*width], 'controlled input values differ')
                for n in (64,256):
                    item=xs[str(n)];require(item['shape']==[1,n,hidden] and item['dtype']==dtype,'input shape/dtype differs')
                    info=item['layout'];require(info['shape']==item['shape'] and info['dtype']==dtype,'layout metadata differs')
                    layout_storage(info['shape'],info['stride'],info['storage_offset'],width=width)
                    if control=='CONTIGUOUS':require(info['contiguous'] and info['stride']==[n*hidden,hidden,1] and info['storage_offset']==0,'not the declared contiguous control')
                    else:
                        old=plan['roles'][role]['captures'][('source' if n==64 else 'target')+'_projection_input']['item']
                        require(info['stride']==old['observed_stride'] and info['storage_offset']==old['storage_offset'],'original-layout reproduction differs')
                sets={n:[x for x in rows if (x['role'],x['precision'],x['control'],x['length'])==(role,precision,control,n)] for n in (64,256)}
                for n,subset in sets.items():
                    for row in subset:
                        item=row['output'];require(item['shape']==[1,n,hidden] and item['dtype']==dtype and not row['warmup'],'output shape/dtype differs')
                        values=ev.tensor('',item);unpack(values,dtype)
                        receipt=row['receipt'];require(receipt['input_sha256']==xs[str(n)]['sha256'] and
                            receipt['input_prefix_sha256']==xs['64']['sha256'] and receipt['weight']==w['before'] and receipt['active_parameters']==active and
                            receipt['gpu_uuid']==env['gpu']['execution_gpu_uuid'] and receipt['backend']=={k:env['effective'][k] for k in CORE},'pre/post-call operand/backend receipt differs')
                    for a,b in combinations(subset,2):
                        same.append(dict(role=role,precision=precision,control=control,length=n,rounds=[a['round'],b['round']],
                            metrics=compare_bytes(ev.tensor('',a['output']),ev.tensor('',b['output']),dtype)))
                for a,b in zip(sets[64],sets[256]):
                    y,_=slice_axis(ev.tensor('',b['output']),b['output']['shape'],1,0,64,width)
                    cross.append(dict(role=role,precision=precision,control=control,round=a['round'],semantic_positions=[0,64],
                        metrics=compare_bytes(ev.tensor('',a['output']),y,dtype)))
                if dtype=='float16':
                    for label,n in (('source',64),('target',256)):
                        current=ev.tensor('',sets[n][0]['output'])[:64*hidden*2]
                        recorded=pe.tensor('',plan['blobs'][role+'/'+label+'_base'])
                        historical_comparisons.append(dict(role=role,control=control,operand=label,metrics=compare_bytes(recorded,current,'float16'),
                            scope='Descriptive comparison to captured base outputs; original allocation alignment/runtime context not reproduced',
                            real_suffix_available=plan['roles'][role]['full_suffix_available']))
                if control=='HISTORICAL_STRIDED':
                    for n in (64,256):
                        normal=[x for x in rows if (x['role'],x['precision'],x['control'],x['length'])==(role,precision,'CONTIGUOUS',n)]
                        for a,b in zip(normal,sets[n]):
                            layout_comparisons.append(dict(role=role,precision=precision,length=n,round=a['round'],
                                metrics=compare_bytes(ev.tensor('',a['output']),ev.tensor('',b['output']),dtype)))
    primary_same=[x for x in same if x['control']=='CONTIGUOUS'];primary_cross=[x for x in cross if x['control']=='CONTIGUOUS']
    decisions=verdicts(primary_same,primary_cross,fixture=fixture)
    decisions['primary_control']='CONTIGUOUS'
    decisions['per_role']={r:verdicts([x for x in primary_same if x['role']==r],[x for x in primary_cross if x['role']==r],fixture=fixture) for r in 'qkv'}
    decisions['per_layout']={c:verdicts([x for x in same if x['control']==c],[x for x in cross if x['control']==c],fixture=fixture)
        for c in sorted({x['control'] for x in cross})}
    if 'HISTORICAL_STRIDED' in decisions['per_layout']:
        decisions['per_layout']['HISTORICAL_STRIDED']['SHAPE_DEPENDENCE_DEMONSTRATED']='COMPOUND_SHAPE_AND_RECORDED_LAYOUT_CONTROL; see contiguous primary verdict'
    decisions['STRIDE_LAYOUT_CONTROL']={r:dict(executed=plan['roles'][r]['layouts'],historical=plan['roles'][r]['captures'],
        scope='Recorded shape/stride/offset only; allocator address alignment and original suffix may be unavailable') for r in 'qkv'}
    out=fresh(output,(experiment,run,plan['historical_run_dir'],plan['historical_review_dir'],plan['base_prepared']))
    from .r3a_artifacts import write
    write(out/'i5r2bs1_comparisons.json',dict(same_shape_all_pairs=same,cross_shape_paired_rounds=cross,historical=historical_comparisons,
        layout_matched_rounds=layout_comparisons,thresholds=THRESHOLDS))
    write(out/'i5r2bs1_verdicts.json',decisions)
    write(out/'i5r2bs1_independent_audit.json',dict(status='PASS',TEST_FIXTURE_ONLY=fixture,inventory=seal_receipt,
        experiment_id=plan['experiment_id'],current_source=plan['source'],historical_source=history['provenance'],
        scope='Sealed tensors/operand and parameter receipts/backend identities plus independent numeric reductions; no new historical GPU audit'))
    write(out/'i5r2bs1_source_provenance.json',dict(current=plan['source'],historical=history['provenance'],original=history['verification_scope']))
    (out/'i5r2bs1_report.md').write_text('# Isolated native Linear shape diagnostic\n\n'+json.dumps(decisions,indent=2)+'\n\nNo cache, LoRA, TCP, attention, generation or performance conclusion is tested here. Numerical fidelity is not restored by this diagnostic.\n')
    seal(out);return decisions
