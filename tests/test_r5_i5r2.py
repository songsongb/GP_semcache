"""Synthetic tensors/schema only. No GPU, model downloads or simulated GPU claims."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from semcache.edgelora.communication import digest
from semcache.edgelora.r3a_artifacts import ROOT,THRESHOLDS,pack,write
from semcache.edgelora.r4b2_analysis import raw_head_bytes
from semcache.edgelora.r4c_workload import policy
from semcache.edgelora.r5_i5_contracts import request,namespace
from semcache.edgelora.r5_i5_protocol import context
from semcache.edgelora.r5_multi_artifacts import registry_for
from semcache.edgelora.r5_i5r2_tensors import Evidence,UNKNOWN,metrics,compare,slice_axis,align,first,safe_path
from semcache.edgelora.r5_i5r2_analysis import run,read_run,analyze,source_provenance,locate,CONDITIONS


def seal(path):
    write(path/'i5_output_manifest.json',dict(files={p.relative_to(path).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(path.rglob('*')) if p.is_file() and p.name!='i5_output_manifest.json'}))


def tensor(root,name,values,shape,dtype='float32',device='cpu'):
    path=root/name;path.parent.mkdir(parents=True,exist_ok=True);blob=pack(values,dtype);path.write_bytes(blob)
    return dict(file=name,shape=shape,dtype=dtype,device=device,sha256=hashlib.sha256(blob).hexdigest())


def fixture(parent,*,prefix_perturbation=0.):
    """Known arrays, not execution receipts from a real model or network."""
    parent=Path(parent);root=parent/'runs/block_0_123';root.mkdir(parents=True);plan_path=parent/'plan';plan_path.mkdir()
    special=dict(bos_token_id=0,eos_token_id=6,pad_token_id=0);p=2;n=4;hidden=4;heads=2;layers=3
    registry=registry_for(['Adapter_A','Adapter_B','Adapter_C'],'expanded')
    tasks=[dict(request_id=rid,user_id='User_01',adapter_id='Adapter_A',dataset='qasper',sample_id='FIXTURE',group_id='FIXTURE',input_text='SYNTHETIC',
        token_ids=ids,policy=policy(special,'fixed' if rid=='S' else 'quality',1 if rid=='S' else 2),request_sequence_number=i,
        role='PRIMING_SOURCE' if rid=='S' else 'SCORED_TARGET',scored=rid!='S',strict=True) for i,(rid,ids) in enumerate((('S',[1,2]),('T',[1,2,3,4])))]
    block=dict(dataset='qasper',track='CONTROLLED',tasks=tasks,prefix_length=p);block['block_id']=digest(block)
    plan=dict(schema='TEST_FIXTURE_ONLY_SCHEMA_NO_MODEL_EXECUTION',blocks=[block],i5r1=dict(source=dict(commit=None,files={})))
    plan['experiment_id']=digest(plan);write(plan_path/'i5_workload_manifest.json',plan);seal(plan_path)
    cfg=dict(fixture=True,source_commit=None,sources={},block=block,experiment_id=plan['experiment_id'],experiment_prepared=str(plan_path),
        experiment_manifest_sha256=hashlib.sha256((plan_path/'i5_output_manifest.json').read_bytes()).hexdigest())
    write(root/'i5_execution_config.json',cfg);write(root/'i5_block_status.json',dict(status='COMPLETE',TEST_FIXTURE_ONLY=True))
    adapter_specs={a:dict(matrix_sha256=chr(ord('b')+i)*64) for i,a in enumerate(registry['users'][u]['adapter_id'] for u in ('User_01','User_11','User_21'))}
    write(root/'i5_adapter_registry.json',dict(registry=registry,adapters=adapter_specs))
    for condition in CONDITIONS:
        dest=root/condition;dest.mkdir();write(dest/'plan.json',dict(cfg,condition=condition))
        write(dest/'result.json',dict(status='COMPLETE',TEST_FIXTURE_ONLY=True,gpu_inference_performed=False));write(dest/'lifecycle.json',dict(returncode=0,pipe_errors=[],forced_termination=False))
        physical={}
        for aid in adapter_specs:
            (dest/aid).mkdir()
            ident=dict(model='TEST_FIXTURE_ONLY',base_revision='TEST_FIXTURE_ONLY',base_weight_sha256='a'*64,adapter=aid,
                adapter_matrix_sha256=adapter_specs[aid]['matrix_sha256'],hidden_dtype='float32',layers=layers,hidden_size=hidden,rank=8)
            physical[aid]=namespace(dict(identity=ident),{'fixture':'tokenizer'},'cpu','TEST_FIXTURE_ONLY',{'fixture':'contract'})
            write(dest/aid/'identity.json',dict(physical_identity=physical[aid],contract={'fixture':True},gpu={'fixture':True}))
            es=dest/aid/'es';es.mkdir();(es/'r5_tokens.jsonl').write_text('')
        arrivals=[];rows=[];all_steps=[];source_specs=None;source_kv=None
        for task in tasks:
            rid=task['request_id'];ids=task['token_ids'];length=len(ids);hit=condition==CONDITIONS[2] and rid=='T';es=dest/'Adapter_A/es'
            req=request(task,ids,task['policy'],physical['Adapter_A'],fixture=True);arrivals.append(dict(kind='ARRIVAL',request=req))
            grant=dict(entry_id='e'*64,source_request_id='S',prefix_length=p,prefix_token_ids=[1,2],proof_sha256='f'*64) if hit else None
            mode='PARTIAL_PREFIX_HIT' if hit else 'COLD_MISS' if condition==CONDITIONS[2] else 'NO_REUSE'
            generated=[5] if rid=='S' else [5,5];steps=[];qkv={};kv=[];components={};cached={}
            for layer in range(layers):
                raws={}
                for role in 'qkv':
                    start=p if hit else 0
                    values=[1.+layer*.1+token*.01+j*.001 for token in range(start,length) for j in range(hidden)]
                    if condition==CONDITIONS[2] and rid=='S' and role=='k' and layer==0:values[0]+=prefix_perturbation
                    raws[role]=values
                    qkv[f'{layer}.{role}']=tensor(es,f'tensors/{rid}_L{layer}_{role}.bin',values,[1,len(values)//hidden,hidden])
                    if hit:
                        s=source_specs[f'{layer}.{role}'];blob=(es/s['file']).read_bytes();name=f'tensors/{rid}_cached_L{layer}_{role}.bin'
                        (es/name).write_bytes(blob);cached[f'{layer}.{role}']=dict(s,file=name)
                pair={}
                for role in 'kv':
                    raw=pack(raws[role],'float32');head_bytes=raw_head_bytes(raw,[1,len(raws[role])//hidden,hidden],heads,4)
                    if hit:
                        prefix=(es/source_kv[layer][role]['file']).read_bytes();dim=hidden//heads
                        head_bytes=b''.join(prefix[h*p*dim*4:(h+1)*p*dim*4]+head_bytes[h*(n-p)*dim*4:(h+1)*(n-p)*dim*4] for h in range(heads))
                    values=[x[0] for x in struct.iter_unpack('<f',head_bytes)]
                    pair[role]=tensor(es,f'tensors/{rid}_L{layer}_past_{role}.bin',values,[1,heads,length,hidden//heads])
                kv.append(pair)
                for role in ('attention','ffn','layer'):
                    start=p if hit else 0
                    vals=[2.+layer*.2+token*.02+j*.002+(0.001 if hit and layer==0 and role=='attention' else 0.) for token in range(start,length) for j in range(hidden)]
                    components[f'{layer}.{role}']=tensor(es,f'tensors/{rid}_L{layer}_{role}.bin',vals,[1,len(vals)//hidden,hidden])
            if rid=='S':source_specs=qkv;source_kv=kv
            for step in range(len(generated)):
                inputs=ids[p if hit else 0:] if step==0 else [generated[step-1]];old=(p if hit else 0) if step==0 else length+step-1
                ctxt=context(rid,step,inputs,ids,old,task['policy'],mode,'d'*64,grant)
                item=dict(request_id=rid,step=step,forward_id=f'{rid}.s{step:02d}',context=ctxt,
                    position_observation=dict(positions=ctxt['positions'],past_length=old),qkv=qkv,kv=kv,components=components,
                    input_kv=[[{k:v for k,v in item[role].items() if k!='file'} for role in 'kv'] for item in source_kv] if hit and step==0 else [])
                if step:
                    item['qkv']={f'{l}.{role}':tensor(es,f'tensors/{rid}_s{step}_L{l}_{role}.bin',[.25]*hidden,[1,1,hidden]) for l in range(layers) for role in 'qkv'}
                    item['kv']=[{role:tensor(es,f'tensors/{rid}_s{step}_L{l}_past_{role}.bin',[.25]*(hidden*(old+1)),[1,heads,old+1,hidden//heads]) for role in 'kv'} for l in range(layers)]
                    item['components']={f'{l}.{role}':tensor(es,f'tensors/{rid}_s{step}_L{l}_{role}.bin',[.25]*hidden,[1,1,hidden]) for l in range(layers) for role in ('attention','ffn','layer')}
                steps.append(item)
            all_steps.extend(steps)
            logit_values=[]
            for step in range(len(generated)):
                values=[.1,.2,.3,.4,.5,1.,0.]
                if hit:values[0]+=.03125
                logit_values.extend(values)
            logit=tensor(es,f'logits/{rid}.bin',logit_values,[1,len(generated),7])
            row=dict(condition=condition,request_id=rid,status='COMPLETE',user_id=task['user_id'],adapter_id='Adapter_A',scored=task['scored'],
                request_sha256=req['request_sha256'],physical_identity=physical['Adapter_A'],matched_identity_sha256=digest(dict(task=task,adapter_sha256=adapter_specs['Adapter_A']['matrix_sha256'],tokenizer=physical['Adapter_A']['tokenizer'])),
                generated_tokens=len(generated),generated_token_ids=generated,generated_text='SYNTHETIC_NONEMPTY',logits_file=logit['file'],logits_sha256=logit['sha256'],logits_shape=logit['shape'],
                source=grant,reused_prefix_tokens=p if hit else 0,mode=mode,arrival_ns=10 if rid=='S' else 30,completion_ns=20 if rid=='S' else 40,
                strict_cached_qkv=cached)
            rows.append(row)
        for name,records in [('generation.jsonl',rows),('requests.jsonl',arrivals),('Adapter_A/es/r5_tokens.jsonl',all_steps)]:
            (dest/name).write_text(''.join(json.dumps(r)+'\n' for r in records))
        if condition!=CONDITIONS[0]:(dest/'events.jsonl').write_text('')
    seal(root);audit=root.with_name(root.name+'_audit');audit.mkdir()
    write(audit/'i5_execution_config.json',cfg);write(audit/'i5_independent_audit.json',dict(status='PASS',TEST_FIXTURE_ONLY=True,scope='SYNTHETIC_SCHEMA_NO_EXECUTION'))
    write(audit/'i5_numerical_fidelity.json',dict(thresholds=THRESHOLDS,comparisons=[],TEST_FIXTURE_ONLY=True));seal(audit)
    return root,plan_path,audit


class TensorMath(unittest.TestCase):
    def test_bitwise_equality_and_signed_zero(self):
        m=metrics(pack([1.,2.],'float32'),pack([1.,2.],'float32'),'float32');self.assertTrue(m['bitwise_equal']);self.assertEqual(m['mean_abs'],0)
        m=metrics(pack([0.],'float32'),pack([-0.],'float32'),'float32');self.assertFalse(m['bitwise_equal']);self.assertEqual(m['max_abs'],0)
    def test_injected_error_uses_frozen_thresholds(self):
        m=metrics(pack([1.,2.],'float16'),pack([1.03125,2.],'float16'),'float16')
        self.assertEqual(m['max_abs'],.03125);self.assertEqual(m['elementwise_tolerance_violations'],1);self.assertFalse(m['activation_tolerance_pass'])
    def test_head_axis_slicing_preserves_all_heads(self):
        blob=pack(range(12),'float32');data,shape=slice_axis(blob,[1,2,3,2],2,0,2,4)
        self.assertEqual(shape,[1,2,2,2]);self.assertEqual([x[0] for x in struct.iter_unpack('<f',data)],[0,1,2,3,6,7,8,9])
    def test_nonfinite_only_allowed_for_masked_negative_infinity(self):
        a=struct.pack('<ff',1.,-float('inf'))
        with self.assertRaises(ValueError):metrics(a,a,'float32')
        self.assertTrue(metrics(a,a,'float32',allow_negative_infinity=True)['bitwise_equal'])
        with self.assertRaises(ValueError):metrics(struct.pack('<f',float('nan')),struct.pack('<f',float('nan')),'float32',allow_negative_infinity=True)
    def test_earliest_difference_keeps_parallel_peers(self):
        records=[dict(family='prefix',layer=l,role=r,step=0,metrics={'bitwise_equal':False}) for l,r in ((2,'q'),(0,'k'),(0,'q'))]
        x=first(records);self.assertEqual(x['record']['layer'],0);self.assertEqual(len(x['causal_peers']),2)
    def test_shape_and_dtype_mismatch(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);a=tensor(p,'a.bin',[1,2],[1,2,1]);b=tensor(p,'b.bin',[1,2],[1,1,2]);e=Evidence(p)
            with self.assertRaisesRegex(ValueError,'shapes'):compare(e,a,'',b,'')
            b=tensor(p,'c.bin',[1,2],[1,2,1],dtype='float16');e=Evidence(p)
            with self.assertRaisesRegex(ValueError,'dtypes'):compare(e,a,'',b,'')
    def test_first_difference_token_position_respects_suffix_offset(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);a=tensor(p,'a.bin',[0,1,2,3],[1,4,1]);b=tensor(p,'b.bin',[2,3.1],[1,2,1]);e=Evidence(p)
            m=compare(e,a,'',b,'',range_a=(2,4),range_b=(0,2))
            self.assertEqual(m['first_differing_stored_token_indices'],dict(reference=3,actual=1))
    def test_bounds_and_unsafe_paths(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);a=tensor(p,'a.bin',[1,2],[1,2,1]);e=Evidence(p,max_tensor_bytes=4)
            with self.assertRaisesRegex(ValueError,'bound'):e.tensor('',a)
            for name in ('../a','/etc/passwd','x/../../a','x\\a','x//a'):
                with self.subTest(name=name),self.assertRaises(ValueError):safe_path(p,name)
            (p/'link').symlink_to(p/'a.bin')
            with self.assertRaises(ValueError):safe_path(p,'link')


class Historical(unittest.TestCase):
    def test_prefix_non_equivalence_is_not_cache_corruption(self):
        with tempfile.TemporaryDirectory() as d:
            root,plan,audit=fixture(d,prefix_perturbation=.02)
            data=read_run(root,experiment=plan,fixture=True);r=analyze(data)
            self.assertEqual(r['gates']['CACHE_SOURCE_INTEGRITY'],'PASS')
            self.assertEqual(r['gates']['CACHE_RECONSTRUCTION_INTEGRITY'],'PASS')
            self.assertEqual(r['gates']['FULL_PREFIX_KV_EQUIVALENCE'],'FAIL')
            self.assertEqual(r['hypotheses']['H1']['baseline_controls']['within_A_short_vs_full_prefix_QKV'],'PASS')
            self.assertEqual(r['gates']['FROZEN_NUMERICAL_TOLERANCE'],'FAIL');self.assertEqual(r['gates']['TOKEN_TEXT_FIDELITY'],'PASS')
            self.assertEqual(r['first_divergence']['ROOT_CAUSE'],'UNRESOLVED')
            self.assertEqual(r['first_divergence']['FIRST_OBSERVED_DIFFERENCE']['standalone_source_vs_full_prefix']['record']['layer'],0)
    def test_alignment_is_semantic_and_mask_position_checked(self):
        with tempfile.TemporaryDirectory() as d:
            root,plan,_=fixture(d);data=read_run(root,experiment=plan,fixture=True)
            a=data['steps'][CONDITIONS[0],'T'][0]['context'];s=data['steps'][CONDITIONS[2],'S'][0]['context'];h=data['steps'][CONDITIONS[2],'T'][0]['context']
            self.assertEqual(align(a,s,relation='FULL_PREFIX_VS_SOURCE',prefix=2),((0,2),(0,2)))
            self.assertEqual(align(a,h,relation='FULL_SUFFIX_VS_HIT',prefix=2),((2,4),(0,2)))
            for field in ('positions','attention_mask'):
                bad=deepcopy(h);bad[field][0]=8
                with self.subTest(field=field),self.assertRaises(ValueError):align(a,bad,relation='FULL_SUFFIX_VS_HIT',prefix=2)
            bad=deepcopy(s);bad['target_token_ids'][0]=9;bad['input_token_ids'][0]=9;bad['prefix_token_ids']=[]
            with self.assertRaises(ValueError):align(a,bad,relation='FULL_PREFIX_VS_SOURCE',prefix=2)
    def test_corrupt_tensor_and_inventory_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root,plan,_=fixture(d);p=next(root.rglob('tensors/*.bin'));blob=p.read_bytes();p.write_bytes(bytes([blob[0]^1])+blob[1:])
            with self.assertRaisesRegex(ValueError,'corrupt sealed'):read_run(root,experiment=plan,fixture=True)
            seal(root)
            with self.assertRaisesRegex(ValueError,'corrupt tensor'):read_run(root,experiment=plan,fixture=True)
    def test_missing_optional_capture_is_unknown_not_pass(self):
        with tempfile.TemporaryDirectory() as d:
            root,plan,_=fixture(d);r=analyze(read_run(root,experiment=plan,fixture=True))
            self.assertEqual(r['gates']['ATTENTION_INPUT_EQUIVALENCE'],UNKNOWN)
            self.assertTrue(any(m['role']=='projection_input' for m in r['missing']))
    def test_hash_only_capture_is_not_invented(self):
        with tempfile.TemporaryDirectory() as d:
            root,plan,_=fixture(d);path=root/'SEMCACHE_INTEGRATED/Adapter_A/es/r5_tokens.jsonl'
            records=[json.loads(line) for line in path.read_text().splitlines()]
            records[0]['qkv']['0.q'].pop('file');path.write_text(''.join(json.dumps(r)+'\n' for r in records));seal(root)
            report=analyze(read_run(root,experiment=plan,fixture=True))
            self.assertEqual(report['gates']['CACHE_SOURCE_INTEGRITY'],UNKNOWN)
            self.assertTrue(any(m['missing'][0]['reason']=='HASH_ONLY_NO_TENSOR_BYTES' for m in report['missing'] if isinstance(m.get('missing'),list)))
    def test_optional_r1_export_is_verified_and_tied_to_block(self):
        with tempfile.TemporaryDirectory() as d:
            root,plan,_=fixture(d);cfg=json.loads((root/'i5_execution_config.json').read_text());r1=root.with_name(root.name+'_i5r1_audit');r1.mkdir()
            write(r1/'i5r1_decision.json',dict(experiment_id=cfg['experiment_id']))
            write(r1/'i5r1_independent_audit.json',dict(status='PASS',TEST_FIXTURE_ONLY=True,blocks=[dict(block_id=cfg['block']['block_id'],source_commit=None)]))
            write(r1/'i5r1_layer_diagnostics.json',[dict(scope='SYNTHETIC_LAST_POSITION_ONLY')]);seal(r1)
            self.assertEqual(read_run(root,experiment=plan,fixture=True)['r1_export']['status'],'HISTORICAL_EXPORT_VERIFIED')
            write(r1/'i5r1_decision.json',dict(experiment_id='wrong'));seal(r1)
            with self.assertRaisesRegex(ValueError,'R1 audit identity'):read_run(root,experiment=plan,fixture=True)
    def test_decode_different_histories_are_not_numerically_paired(self):
        with tempfile.TemporaryDirectory() as d:
            root,plan,_=fixture(d);data=read_run(root,experiment=plan,fixture=True)
            a=data['steps'][CONDITIONS[0],'T'][1]['context'];c=data['steps'][CONDITIONS[2],'T'][1]['context']
            with self.assertRaisesRegex(ValueError,'history'):
                align(a,c,relation='FULL_SUFFIX_VS_HIT',prefix=2,history_reference=[5],history_actual=[4])
    def test_synthetic_evidence_cannot_be_claimed_real_gpu(self):
        with tempfile.TemporaryDirectory() as d:
            root,plan,_=fixture(d)
            with self.assertRaisesRegex(ValueError,'scope mismatch'):read_run(root,experiment=plan,expected_commit='a'*40)
    def test_missing_required_tensor_file_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root,plan,_=fixture(d);next(root.rglob('tensors/*.bin')).unlink()
            with self.assertRaisesRegex(ValueError,'inventory'):read_run(root,experiment=plan,fixture=True)
    def test_available_source_hash_mismatch_cannot_be_waived(self):
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        cfg=dict(source_commit=commit,sources={'src/semcache/edgelora/r4b2_kv.py':'0'*64})
        with self.assertRaisesRegex(ValueError,'source SHA-256 mismatch'):
            source_provenance(cfg,expected_commit=commit,fixture=False,allow_unavailable=True)
    def test_available_historical_blob_verified_without_current_source_match(self):
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();name='src/semcache/edgelora/r4b2_kv.py'
        blob=subprocess.check_output(['git','show',commit+':'+name],cwd=ROOT)
        cfg=dict(source_commit=commit,sources={name:hashlib.sha256(blob).hexdigest()})
        self.assertEqual(source_provenance(cfg,expected_commit=commit,fixture=False)['historical_source_verification'],'ALL_RECORDED_BLOBS_MATCH')
    def test_missing_source_object_requires_explicit_partial_provenance(self):
        cfg=dict(source_commit='0'*40,sources={'missing.py':'1'*64})
        with self.assertRaisesRegex(ValueError,'object unavailable'):source_provenance(cfg,expected_commit='0'*40,fixture=False)
        p=source_provenance(cfg,expected_commit='0'*40,fixture=False,allow_unavailable=True)
        self.assertEqual(p['historical_source_verification'],'HISTORICAL_OBJECT_UNAVAILABLE')
    def test_expected_commit_mismatch_rejected(self):
        with self.assertRaisesRegex(ValueError,'source commit mismatch'):
            source_provenance(dict(source_commit='a'*40,sources={}),expected_commit='b'*40,fixture=False)
    def test_ambiguous_original_blocks_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'block_0_1').mkdir();(p/'block_0_2').mkdir();(p/'block_0_1_audit').mkdir()
            with self.assertRaisesRegex(ValueError,'ambiguous'):locate(runs_root=p)
    def test_report_deterministic_and_inputs_read_only(self):
        with tempfile.TemporaryDirectory() as d:
            root,plan,_=fixture(d);before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(d).rglob('*') if p.is_file()}
            a=run(run_dir=root,experiment=plan,output_dir=Path(d)/'a',fixture=True)
            b=run(run_dir=root,experiment=plan,output_dir=Path(d)/'b',fixture=True)
            self.assertTrue(a['GPU_NOT_RUN']);self.assertEqual(a['status'],'CPU_FIXTURE_DIAGNOSIS_COMPLETE')
            self.assertEqual({p.name:p.read_bytes() for p in (Path(d)/'a').iterdir()},{p.name:p.read_bytes() for p in (Path(d)/'b').iterdir()})
            self.assertTrue(all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in before.items()))
            with self.assertRaisesRegex(ValueError,'fresh'):run(run_dir=root,experiment=plan,output_dir=Path(d)/'a',fixture=True)


if __name__=='__main__':unittest.main()
