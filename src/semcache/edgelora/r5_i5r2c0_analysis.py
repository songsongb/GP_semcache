"""Read-only archived S1 cast control. No production arithmetic or GPU imports."""
import math
import struct
from pathlib import Path
from .communication import digest,digest_bytes,require
from .r3a_artifacts import ROOT,THRESHOLDS,write,unpack
from .r5_i5_evidence import seal
from .r5_i5r2_tensors import Evidence,slice_axis
from .r5_i5r2_analysis import source_provenance
from .r5_i5r2bs1_plan import validate_plan
from .r5_i5r2bs1_analysis import compare_bytes,promote
from .r5_i5r1_plan import source_identity


def cast_half(data):
    require(len(data)%4==0,'misaligned FP32 bytes')
    out=bytearray()
    for (x,) in struct.iter_unpack('<f',data):
        require(math.isfinite(x),'nonfinite FP32 cast input')
        try:b=struct.pack('<e',x)
        except OverflowError:raise ValueError('FP32 value overflows finite FP16')
        require(math.isfinite(struct.unpack('<e',b)[0]),'nonfinite cast output');out.extend(b)
    return bytes(out)


def rounding(a,b,c,d):
    count=ties=straddles=multiple=0;examples=[]
    for i,((x,),(y,),(u,),(v,)) in enumerate(zip(struct.iter_unpack('<f',a),struct.iter_unpack('<f',b),struct.iter_unpack('<e',c),struct.iter_unpack('<e',d))):
        if u==v:continue
        count+=1;lo,hi=sorted((u,v));mid=(lo+hi)/2
        bits=lambda z:struct.unpack('<H',struct.pack('<e',z))[0]
        ordered=lambda n:0x8000-(n&32767) if n&32768 else 0x8000+n
        adjacent=abs(ordered(bits(u))-ordered(bits(v)))==1
        ties+=adjacent and (x==mid or y==mid);straddles+=adjacent and min(x,y)<=mid<=max(x,y);multiple+=not adjacent
        if len(examples)<8:examples.append(dict(element=i,FP32_short=x,FP32_long=y,cast_short=u,cast_long=v,adjacent_bins=adjacent,
            midpoint=mid if adjacent else None,exact_midpoint_tie=adjacent and (x==mid or y==mid)))
    return dict(numeric_mismatches=count,adjacent_bin_midpoint_straddles=straddles,exact_midpoint_ties=ties,
        multiple_bin_mismatches=multiple,examples=examples,scope='C/D numeric mismatches only; signed-zero bit differences are in comparison metrics')


def compare_four(a,b,x,y):
    c,d=cast_half(x),cast_half(y);pairs={'A_vs_B':(a,b),'C_vs_D':(c,d),'C_vs_A':(c,a),'D_vs_B':(d,b),'C_vs_B':(c,b)}
    result={}
    for name,(reference,actual) in pairs.items():
        m=compare_bytes(reference,actual,'float16');m['differing_elements_percent']=100*m['differing_values']/m['elements']
        m['bitwise_differing_elements']=sum(reference[i:i+2]!=actual[i:i+2] for i in range(0,len(reference),2))
        m['bitwise_differing_elements_percent']=100*m['bitwise_differing_elements']/m['elements'];result[name]=m
    original=result['A_vs_B'];cast=result['C_vs_D']
    return dict(comparisons=result,rounding_boundaries=rounding(x,y,c,d),
        CAST_PREFIX_BITWISE_EQUAL=cast['bitwise_equal'],CAST_PREFIX_STILL_DIFFERENT=not cast['bitwise_equal'],
        CAST_PREFIX_NUMERICALLY_CLOSER=cast['max_abs']<=original['max_abs'] and cast['relative_l2']<=original['relative_l2'] and
            (cast['max_abs']<original['max_abs'] or cast['relative_l2']<original['relative_l2']),
        SOURCE_ONLY_BASELINE_INCOMPATIBLE=not result['C_vs_B']['bitwise_equal'])


def analyze(experiment,run,audit,output):
    pe=Evidence(experiment,max_input_bytes=128*2**20);ps=pe.verify()
    # Current-source equivalence is replaced by strict verification of every
    # recorded historical Git blob, not by accepting unverified old sources.
    plan=validate_plan(experiment,check_source=False)
    provenance=source_provenance(dict(source_commit=plan['source']['commit'],sources=plan['source']['files']),
        expected_commit=plan['source']['commit'],fixture=plan['TEST_FIXTURE_ONLY'])
    ev=Evidence(run,max_input_bytes=512*2**20);rs=ev.verify();ae=Evidence(audit,max_input_bytes=128*2**20);ats=ae.verify()
    status=ev.json('i5r2bs1_status.json');old=ae.json('i5r2bs1_independent_audit.json');fixture=plan['TEST_FIXTURE_ONLY']
    require(status['status']=='COMPLETE' and status['GPU_NOT_RUN']==fixture and ev.json('i5r2bs1_plan.json')==plan and
        status['experiment_id']==old['experiment_id']==plan['experiment_id'] and status['source_commit']==plan['source']['commit'] and
        old['status']=='PASS' and old['TEST_FIXTURE_ONLY']==fixture and old['current_source']==plan['source'] and old['inventory']==rs,
        'incomplete/mismatched archived S1 audit chain')
    require(ae.json('i5r2bs1_comparisons.json')['thresholds']==THRESHOLDS,'frozen thresholds differ')
    rows=ev.json('i5r2bs1_records.json');weights=ev.json('i5r2bs1_weights.json');env=ev.json('i5r2bs1_backend.json')
    expected=[(r,p,c,s['length'],s['round']) for r in 'qkv' for p in plan['precision_conditions'] for c in plan['roles'][r]['layouts']
        for s in plan['schedule'] if not s['warmup']]
    require([(r['role'],r['precision'],r['control'],r['length'],r['round']) for r in rows]==expected,'missing/duplicate/misaligned S1 repetitions')
    by={(r['role'],r['precision'],r['control'],r['length'],r['round']):r for r in rows};results=[]
    if not fixture:
        fp=ev.json('i5r2bs1_base_fingerprint.json')
        require(digest(fp['tensors'])==fp['sha256']==plan['model_identity']['base_weight_sha256']==status['base_fingerprint_before']==status['base_fingerprint_after'], 'model fingerprint differs')
    for role in 'qkv':
        w=weights[role];require(w['before']==w['after'],'native weights changed');converted={}
        for name,item in w['saved'].items():
            if item is None:converted[name]=None;require(w['before'][name] is None,'missing parameter');continue
            raw=ev.tensor('',item);require(item['sha256']==w['before'][name]['sha256'] and item['dtype']=='float16','weight identity differs')
            if not fixture:require(item['sha256']==fp['tensors'][w['module']+'.'+name]['sha256'],'projection/model fingerprint differs')
            converted[name]=dict(shape=item['shape'],dtype='float32',sha256=digest_bytes(promote(raw)))
        require(w['DERIVED_FP32_temporary_parameter_hashes']==converted,'FP32 weight promotion differs')
        for control in plan['roles'][role]['layouts']:
            for i in range(plan['repetitions']):
                data=[]
                for precision,n in (('NATIVE_FP16',64),('NATIVE_FP16',256),('DERIVED_FP32',64),('DERIVED_FP32',256)):
                    row=by[role,precision,control,n,i];t=row['output'];dtype='float16' if precision=='NATIVE_FP16' else 'float32'
                    require(t['dtype']==dtype and t['shape']==[1,n,plan['roles'][role]['hidden']],'S1 output dtype/shape differs')
                    require(row['receipt']['weight']==w['before'] and row['receipt']['active_parameters']==(w['before'] if dtype=='float16' else converted), 'projection receipt differs')
                    group=role+'/'+precision+'/'+control;inp=ev.json('i5r2bs1_inputs.json')[group]
                    require(row['receipt']['input_sha256']==inp[str(n)]['sha256'] and row['receipt']['input_prefix_sha256']==inp['64']['sha256'], 'input receipt differs')
                    for item in inp.values():ev.tensor('',item)
                    blob,_=slice_axis(ev.tensor('',t),t['shape'],1,0,64,2 if dtype=='float16' else 4);unpack(blob,dtype);data.append(blob)
                results.append(dict(role=role,layout=control,repetition=i,positions=[0,64],**compare_four(*data)))
    out=Path(output).resolve();namespace=ROOT/'results/r5/i5r/numerical_fidelity/fp32_cast_feasibility'
    require(not out.is_relative_to(ROOT) or out.is_relative_to(namespace),'use fp32_cast_feasibility namespace')
    require(not Path(output).is_symlink() and not out.exists() and all(not out.is_relative_to(Path(p).resolve()) and not Path(p).resolve().is_relative_to(out) for p in (experiment,run,audit)), 'fresh nonoverlapping output required')
    out.mkdir(parents=True,mode=0o700)
    summary=dict(INCOMPLETE_EVIDENCE=False,TEST_FIXTURE_ONLY=fixture,scope='Layer 0 isolated outputs; no whole-model or performance conclusion',
        ROOT_CAUSE_CONFIRMED=False,NUMERICAL_FIDELITY_RESTORED='NOT_EVALUATED',groups={})
    for role in 'qkv':
        for layout in plan['roles'][role]['layouts']:
            selected=[r for r in results if r['role']==role and r['layout']==layout]
            summary['groups'][role+'/'+layout]={k:sum(r[k] for r in selected) for k in
                ('CAST_PREFIX_BITWISE_EQUAL','CAST_PREFIX_NUMERICALLY_CLOSER','CAST_PREFIX_STILL_DIFFERENT','SOURCE_ONLY_BASELINE_INCOMPATIBLE')}
            summary['groups'][role+'/'+layout]['repetitions']=len(selected)
    write(out/'i5r2c0_comparisons.json',results);write(out/'i5r2c0_summary.json',summary)
    write(out/'i5r2c0_provenance.json',dict(plan=ps,run=rs,audit=ats,historical_execution=provenance,current_analysis=source_identity(True),
        scope='Verified exported S1 independent audit and sealed operands; no fresh execution of the historical S1 auditor'))
    (out/'i5r2c0_report.md').write_text('# Offline FP32→FP16 feasibility\n\n'+__import__('json').dumps(summary,indent=2)+'\n\nC == D does not imply C == B. Closer means non-increasing max absolute and relative L2 errors, with at least one strictly smaller. Incomplete required evidence is rejected before report creation.\n')
    seal(out);return summary
