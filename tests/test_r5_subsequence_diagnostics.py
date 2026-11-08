"""CPU synthetic forensic/profiling fixtures; never GPU numerical evidence."""
import copy,importlib.util,json,subprocess,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from semcache.edgelora.communication import digest,digest_bytes
from semcache.edgelora.r3a_artifacts import ROOT,write,append,pack,THRESHOLDS,check_parity
from semcache.edgelora.r3a_tiny import f32
from semcache.edgelora.r3b_fixture import ArrayTensor
from semcache.edgelora.r4c_evidence import Evidence as Writer
from semcache.edgelora.r5_i5_evidence import seal
from semcache.edgelora.r5_i5r2_tensors import Evidence
from semcache.edgelora.r5_sparse_qkv import envelope,gather
from semcache.edgelora.r5_subsequence_profile import Profiler,CURRENT,call,validate_receipt
from semcache.edgelora.r5_subsequence_physical import PhysicalOwner
from semcache.edgelora import r5_subsequence_diagnostics as api
import test_r5_sparse_physical as fixture

def fixture_records(root):
    f=fixture.Physical();f.setUp();records={}
    source=f.req('S',[1,2,3,4,5,6]);target=f.req('T',[1,2,3,4,5,6,7,8],user='User_02')
    target['request_sequence_number']=1;target.pop('request_sha256');target['request_sha256']=digest(target)
    for condition in api.CONDITIONS:
        out=root/condition;(out/'es').mkdir(parents=True);(out/'ud').mkdir();writer=Writer(out/'es');results=[]
        for index,req in enumerate((source,target)):
            n=len(req['token_ids']);recs=[]
            for layer in range(2):
                drift=.5 if condition==api.CONDITIONS[2] and index==1 and layer==1 else 0.
                x=ArrayTensor([float(t)+drift for t in req['token_ids'] for j in range(4)],[1,n,4])
                reused=list(range(6)) if condition==api.CONDITIONS[2] and index==1 and layer==0 else []
                fresh=[p for p in range(n) if p not in reused];inp=gather(x,fresh)
                bias=.25 if index==0 else .125 if condition==api.CONDITIONS[2] else 0.
                base=ArrayTensor([2*v+bias for v in inp.values],inp.shape);delta=ArrayTensor([f32(.1)]*len(inp.values),inp.shape)
                hidden=pack(inp.values,'float32');db=pack(delta.values*3,'float32')
                hn=f'{req["request_id"]}_{layer}_hidden.bin';dn=f'{req["request_id"]}_{layer}_delta.bin'
                (out/'ud'/hn).write_bytes(hidden);(out/'ud'/dn).write_bytes(db)
                append(out/'ud/layers.jsonl',dict(context=envelope(req,layer,fresh,req['request_id']+'_s000'),hidden_file=hn,delta_file=dn,hidden_sha256=digest_bytes(hidden),delta_sha256=digest_bytes(db)))
                for role in 'qkv':
                    tag=f'{req["request_id"]}_{layer}_{role}';vals=[f32(a+b) for a,b in zip(base.values,delta.values)];consumed=[]
                    if reused:
                        output=[f32(2*float(t)+.25+f32(.1)) for t in req['token_ids'][:6] for j in range(4)]+vals
                        for start in (0,3):
                            cached=ArrayTensor(output[start*4:(start+3)*4],[1,3,4])
                            consumed.append(dict(key=[0,req['token_ids'][start:start+3]],source_request_id='S',source_positions=[start,start+3],target_positions=[start,start+3],tensor=writer.save(cached,tag+f'_cache{start}')))
                    else:output=vals
                    recs.append(dict(layer=layer,role=role,input=writer.save(x,tag+'_input'),output=writer.save(ArrayTensor(output,[1,n,4]),tag+'_output'),fresh_base=writer.save(base,tag+'_base'),fresh_delta=writer.save(delta,tag+'_delta'),fresh_positions=fresh,reused_positions=reused,consumed=consumed,mode='EXACT_CONTEXT_VERIFIED'))
            physical=None if condition==api.CONDITIONS[0] else dict(records=recs)
            candidates=[dict(source_request_id='S',source_user='User_01',source_adapter='Adapter_A',source_positions=[s,s+3],target_positions=[s,s+3],rejections=[]) for s in (0,3)]
            result=dict(request_id=req['request_id'],condition=condition,context_mode='EXACT_CONTEXT_VERIFIED',prefill_records=recs if physical is None else [],completion={'physical':physical},matching=dict(selected_spans=[[0,3],[3,6]],candidates=candidates),arrival_ns=1 if not index else 11,completion_ns=10 if not index else 20,
                        generated_token_ids=[3],generated_text='TEST_FIXTURE_ONLY',logits=writer.save(ArrayTensor([0.,0.,0.,1.],[1,1,4]),tag+'_logits'),logits_shape=[1,1,4],steps=[dict(selected_token_id=3,logits_sha256=digest_bytes(pack([0.,0.,0.,1.],'float32')))])
            results.append(result)
        seal(out);ev=Evidence(out);ev.verify();ev.registry=f.registry
        records[condition]=dict(ev=ev,requests=[source,target],results=results,config=dict(physical=f.ident,registry=f.registry),network={})
    return records

def sealed_history(root):
    plan=root/'plan';plan.mkdir();run=root/'run';audit=root/'audit';audit.mkdir();cpu=root/'cpu';cpu.mkdir()
    records=fixture_records(run)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();name='src/semcache/edgelora/r5_i5_contracts.py'
    source=dict(commit=commit,branch='exp/raw-semcache-r5-integration',files={name:digest_bytes(subprocess.check_output(['git','show',commit+':'+name],cwd=ROOT))})
    tasks=[dict(req,policy=req['generation_policy']) for req in records[api.CONDITIONS[0]]['requests']]
    p=dict(schema='R5_SUBSEQUENCE_GENERATION_PILOT_V1',mode='EXACT_CONTEXT_VERIFIED',allow_approximate=False,source=source,source_tokens=6,tasks=tasks,thresholds=THRESHOLDS,multi=str(root/'prepared_multi'),dataset=str(root/'prepared_dataset'))
    write(plan/'subsequence_plan.json',p);seal(plan);pr=Evidence(plan).verify()
    write(cpu/'cpu_validation.json',dict(status='PASS',failed=0,skipped=0,source=source,plan_inventory=pr));seal(cpu)
    write(run/'status.json',dict(status='COMPLETE',source=source,plan_inventory=pr,cpu_validation_path=str(cpu),cpu_validation=Evidence(cpu).verify()))
    exported={}
    for condition,r in records.items():
        path=run/condition;config=dict(r['config'],condition=condition,source=source,plan_inventory=pr)
        write(path/'identity.json',config);write(path/'requests.json',r['requests']);write(path/'results.json',r['results']);write(path/'status.json',dict(status='COMPLETE'))
        send=dict(operation='send',complete=True,frame_sha256='a'*64,wire_bytes=100,header={'sequence':0});reply=dict(send,operation='recv',frame_sha256='b'*64)
        append(path/'es/wire.jsonl',send);append(path/'es/wire.jsonl',reply)
        append(path/'ud/wire.jsonl',dict(send,operation='recv'));append(path/'ud/wire.jsonl',dict(reply,operation='send'))
        seal(path);exported[condition]=dict(inventory=Evidence(path).verify(),config=config,requests=r['requests'],results=r['results'])
    seal(run);gate=check_parity([0.,0.,0.,1.],[0.,0.,0.,1.],[1,1,4])
    write(audit/'subsequence_audit.json',dict(status='INDEPENDENT_EVIDENCE_AUDIT_PASS',thresholds=THRESHOLDS,reports=exported,comparisons=[dict(condition=c,request_id=rid,numerical={'per_step':[gate]}) for c in api.CONDITIONS for rid in ('S','T')]))
    seal(audit);return plan,run,audit,commit

class Diagnostics(unittest.TestCase):
    def test_full_historical_reader_and_deterministic_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);plan,run,audit,commit=sealed_history(root)
            before=Evidence(run).verify()
            with patch.object(api,'prepared_identity'):
                api.diagnose(plan,run,audit,root/'review1',commit);api.diagnose(plan,run,audit,root/'review2',commit)
            a=Evidence(root/'review1');b=Evidence(root/'review2');a.verify();b.verify();self.assertEqual(a.inventory,b.inventory)
            self.assertEqual(before,Evidence(run).verify());self.assertEqual(a.json('subsequence_diagnostics.json')['ROOT_CAUSE'],'UNRESOLVED')
    def test_historical_inventory_and_export_tamper_rejected(self):
        for target in ('raw','audit','source'):
            with self.subTest(target=target),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);plan,run,audit,commit=sealed_history(root)
                path=run/api.CONDITIONS[2]/'results.json' if target=='raw' else audit/'subsequence_audit.json' if target=='audit' else plan/'subsequence_plan.json'
                path.write_bytes(path.read_bytes()+b' ')
                with patch.object(api,'prepared_identity'),self.assertRaises(ValueError):api.history(plan,run,audit,commit)
    def test_revalidation_preserves_tasks_and_uses_new_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);plan,run,audit,commit=sealed_history(root);new=dict(commit='c'*40,files={'new':'a'*64},branch='exp/raw-semcache-r5-integration')
            with patch.object(api,'prepared_identity'),patch('semcache.edgelora.r5_i5r1_plan.source_identity',return_value=new),patch.object(api,'diagnostic_source_files',return_value={'TEST_FIXTURE_ONLY':'a'*64}):
                api.prepare_revalidation(plan,run,audit,root/'new_plan',new['commit'],commit)
            old=Evidence(plan).json('subsequence_plan.json');out=Evidence(root/'new_plan');out.verify();p=out.json('subsequence_plan.json')
            self.assertEqual(p['tasks'],old['tasks']);self.assertEqual(p['source'],new);self.assertNotEqual(p['source'],old['source'])
            self.assertEqual(p['diagnostics']['classification'],'DIAGNOSTIC_ONLY_UNCHANGED_ARITHMETIC');self.assertFalse(p['diagnostics']['cuda_events'])
    def test_cached_fresh_and_scatter_errors_are_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            data=api.forensics(fixture_records(Path(tmp)))
        select=lambda name:next(x for x in data if x['comparison']==name and x['layer']==0 and x['role']=='q')
        self.assertFalse(select('cached_rows_A_vs_C')['bitwise_equal']);self.assertFalse(select('fresh_rows_A_vs_C_fresh_base')['bitwise_equal'])
        self.assertTrue(select('fresh_rows_A_vs_C_fresh_delta')['bitwise_equal']);self.assertTrue(select('source_cache_vs_C_scatter')['bitwise_equal'])
        self.assertTrue(select('fresh_add_cast_vs_C_scatter')['bitwise_equal']);self.assertEqual(select('A_vs_C_output')['first_absolute_token_position'],0)
    def test_layer_one_rejection_reconstructed_from_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:layers=api.eligibility(fixture_records(Path(tmp)))
        self.assertEqual(layers[0]['reused_qkv_rows'],18);self.assertEqual(layers[0]['observed_input_mismatch_count'],0)
        self.assertEqual(layers[1]['observed_input_mismatch_count'],2);self.assertEqual(layers[1]['reused_qkv_rows'],0)
        self.assertEqual(layers[1]['historical_rejection_event_count'],'NOT_OBSERVABLE')
        self.assertEqual(len(layers[1]['reuse_specific_rejection_windows']),2)
    def test_baseline_source_mismatch_not_attributed_only_to_reuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            records=fixture_records(Path(tmp));a=records[api.CONDITIONS[0]];spec=api.projections(a['results'][1])[1,'q']
            spec['input']=Writer(a['ev'].root/'es').save(ArrayTensor([float(t)+.75 for t in a['requests'][1]['token_ids'] for j in range(4)],[1,8,4]),'alternative_baseline_input')
            seal(a['ev'].root);a['ev']=Evidence(a['ev'].root);a['ev'].verify()
            layers=api.eligibility(records)
            self.assertEqual(len(layers[1]['source_vs_A_input_mismatch_windows']),2)
            self.assertEqual(len(layers[1]['C_vs_A_input_mismatch_windows']),2)
            self.assertEqual(layers[1]['reuse_specific_rejection_windows'],[])
    def test_corrupted_or_missing_raw_tensor_rejected(self):
        for missing in (False,True):
            with self.subTest(missing=missing),tempfile.TemporaryDirectory() as tmp:
                records=fixture_records(Path(tmp));r=records[api.CONDITIONS[2]];spec=api.projections(r['results'][1])[0,'q']['output'];path=Path(r['ev'].root)/'es'/spec['file']
                if missing:path.unlink()
                else:path.write_bytes(b'bad')
                with self.assertRaises((ValueError,OSError)):api.forensics(records)
    def test_actual_prefix_positions_not_last_position(self):
        with tempfile.TemporaryDirectory() as tmp:
            records=fixture_records(Path(tmp));spec=api.projections(records[api.CONDITIONS[0]]['results'][1])[0,'q']['input']
            self.assertEqual(api.aligned(records[api.CONDITIONS[0]]['ev'],spec,[0,3]),pack([1.]*4+[4.]*4,'float32'))
            with self.assertRaises(ValueError):api.aligned(records[api.CONDITIONS[0]]['ev'],spec,[8])
    def test_tensor_dtype_shape_and_context_mismatch_rejected(self):
        for kind in ('dtype','shape','tokens'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as tmp:
                records=fixture_records(Path(tmp));data=records[api.CONDITIONS[2]];record=api.projections(data['results'][1])[0,'q']
                if kind=='dtype':record['input']['dtype']='float16'
                elif kind=='shape':record['input']['shape']=[1,8,8]
                else:data['requests'][1]['token_ids'][0]=99
                with self.assertRaises(ValueError):api.forensics(records)
    def test_frozen_per_step_logit_replay_and_tamper(self):
        with tempfile.TemporaryDirectory() as tmp:
            records=fixture_records(Path(tmp));gate=check_parity([0.,0.,0.,1.],[0.,0.,0.,1.],[1,1,4])
            export={'comparisons':[dict(condition=c,request_id=rid,numerical={'per_step':[gate]}) for c in api.CONDITIONS for rid in ('S','T')]}
            self.assertTrue(all(r['per_step'][0]['passed'] for r in api.logits(records,export)))
            export['comparisons'][0]['numerical']['per_step']=[dict(gate,max_abs=999)]
            with self.assertRaises(ValueError):api.logits(records,export)
    def owner_run(self,profile=None):
        f=fixture.Physical();f.setUp();owner=PhysicalOwner(f.ident,f.registry)
        class Session:
            def correction(self,l,r,x,positions):return ArrayTensor([0.]*(len(positions)*4),[1,len(positions),4]) if positions else None
        source=f.req('S',[1,2,3,4,5,6]);owner.begin(source,{'cluster_id':0})
        for layer in range(2):
            for role in 'qkv':owner.project(layer,role,ArrayTensor([1.]*24,[1,6,4]),lambda x:x.clone(),Session())
        with patch('semcache.edgelora.r5_subsequence_physical.actual_attention_impact',return_value=1.):owner.complete(completed=True,attentions=['CPU_FIXTURE_ONLY']*2)
        target=f.req('T',[1,2,3,4,5,6,7,8]);owner.begin(target,{'cluster_id':0})
        for layer in range(2):
            for role in 'qkv':owner.project(layer,role,ArrayTensor([float(layer+1)]*32,[1,8,4]),lambda x:x.clone(),Session())
        return owner
    def test_profile_off_preserves_values_and_no_cuda_dependency(self):
        plain=self.owner_run();profiler=Profiler()
        with patch.dict(sys.modules,{'torch':None}),profiler.activate('T'):observed=self.owner_run(profiler)
        self.assertEqual([r['output'] for r in plain.active['records']],[r['output'] for r in observed.active['records']])
        self.assertIsNone(CURRENT.get());self.assertFalse(profiler.finish()['cuda_events'])
    def test_profile_counts_full_entry_rehashes_and_safe_rejections(self):
        profiler=Profiler()
        with profiler.activate('T'):owner=self.owner_run(profiler)
        receipt=profiler.finish();stats=receipt['components']
        # Source admission adds proofs; the target PREFILL counters isolate valid().
        self.assertEqual(sum(r['count'] for r in stats if r['component']=='owner_valid' and r['phase']=='PREFILL'),10)
        rejects=[r for r in receipt['eligibility'] if r.get('reason')=='PER_LAYER_INPUT_BYTES_MISMATCH']
        self.assertEqual(len(rejects),2);self.assertTrue(all(r['layer']==1 for r in rejects))
        self.assertGreater(sum(r['bytes'] for r in stats if r['component']=='tensor_serialization_host_transfer_inclusive'),0)
        validate_receipt(receipt,'T',dict(records=owner.active['records']))
        broken=copy.deepcopy(receipt);broken['calls']+=1
        with self.assertRaises(ValueError):validate_receipt(broken,'T',None)
    def test_integrity_failure_remains_hard_failure_with_receipt(self):
        owner=self.owner_run();entry=next(iter(owner.cache.entries.values()));entry.tensors[0][0].values[0]+=1
        profile=Profiler()
        with profile.activate('bad'),self.assertRaisesRegex(ValueError,'integrity failure'):owner.valid(entry)
        self.assertTrue(any(x['component']=='owner_valid' for x in profile.finish()['components']))
    def test_failed_profile_export_cannot_mask_primary_exception(self):
        from semcache.edgelora.r5_subsequence_generation import generate
        from unittest.mock import Mock
        for writer_fails in (False,True):
            with self.subTest(writer_fails=writer_fails),tempfile.TemporaryDirectory() as tmp:
                primary=ValueError('original integrity failure');writer=Mock() if writer_fails else Writer(tmp)
                if writer_fails:writer.blob.side_effect=OSError('disk failure')
                with patch('semcache.edgelora.r5_subsequence_generation._generate',side_effect=primary),self.assertRaises(ValueError) as raised:
                    generate(None,None,None,{'request_id':'F'},None,None,None,condition='SEMCACHE_SUBSEQUENCE',profiler=Profiler(),evidence=writer)
                self.assertIs(raised.exception,primary);self.assertIsNone(CURRENT.get())
                if not writer_fails:self.assertEqual(json.loads((Path(tmp)/'F_FAILED_profile.json').read_text())['status'],'FAILED')
    def test_profile_bounds_and_nested_exclusive_time(self):
        p=Profiler(max_calls=2,max_details=1)
        with p.activate('R'):
            with p.scope('outer'):
                call('inner',lambda:None)
            with self.assertRaises(ValueError):call('over',lambda:None)
            p.detail({'boundary':'fixture'})
            with self.assertRaises(ValueError):p.detail({'boundary':'extra'})
        result=p.finish();self.assertTrue(all(r['exclusive_wall_ms']<=r['inclusive_wall_ms'] for r in result['components']))
    def test_source_blob_identity_and_mismatch(self):
        from semcache.edgelora.r5_i5r2_analysis import source_provenance
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();name='src/semcache/edgelora/r5_i5_contracts.py'
        blob=subprocess.check_output(['git','show',commit+':'+name],cwd=ROOT);config=dict(source_commit=commit,sources={name:digest_bytes(blob)})
        self.assertEqual(source_provenance(config,expected_commit=commit,fixture=False)['historical_source_verification'],'ALL_RECORDED_BLOBS_MATCH')
        config['sources'][name]='0'*64
        with self.assertRaises(ValueError):source_provenance(config,expected_commit=commit,fixture=False)
    def test_no_overwrite_or_historical_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'original';path.mkdir()
            with self.assertRaises(ValueError):api.fresh(path,(path,))
            with self.assertRaises(ValueError):api.fresh(path/'nested',(path,))
        with self.assertRaises(ValueError):api.fresh(ROOT/'results/r5/i5/diagnostic_write',())
    def test_cli_dispatch_is_cpu_only(self):
        import contextlib,io
        spec=importlib.util.spec_from_file_location('diagnostic_cli',ROOT/'scripts/152_r5_subsequence_diagnostics.py');cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)
        args=['--experiment','p','--run','r','--audit','a','--output-dir','o']
        with patch.object(api,'diagnose',return_value={'TEST_FIXTURE_ONLY':True}) as diagnose,contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(cli.main(args),0);self.assertEqual(diagnose.call_args.kwargs['expected_source_commit'],api.REFERENCE_COMMIT)
        with self.assertRaises(SystemExit),contextlib.redirect_stderr(io.StringIO()):cli.main(args+['--prepare-revalidation'])
    @unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('transformers'),'native OPT profile observation NOT_RUN: dependencies absent')
    def test_native_cpu_opt_profile_does_not_change_logits(self):
        import torch
        from transformers import OPTConfig,OPTForCausalLM
        from semcache.models.model_adapter import OPTModelAdapter
        from semcache.edgelora.r5_subsequence_generation import StreamingImpact
        from semcache.semantic.subsequence import SubsequenceExtractor
        f=fixture.Physical();f.setUp();req=f.req('N',[1,2,3,4,5,6])
        cfg=OPTConfig(vocab_size=32,hidden_size=8,word_embed_proj_dim=8,ffn_dim=16,num_hidden_layers=2,num_attention_heads=2,dropout=0.,attention_dropout=0.);cfg._attn_implementation='eager'
        model=OPTForCausalLM(cfg).eval();ids=torch.tensor([req['token_ids']]);weights=[p.detach().clone() for p in model.parameters()]
        with torch.inference_mode():original=model(ids,output_attentions=True).logits
        impact=StreamingImpact(req,2,SubsequenceExtractor(3).extract(req['token_ids']));profile=Profiler()
        with torch.inference_mode(),profile.activate('N'),impact.hooks(OPTModelAdapter(model)):out=model(ids,output_attentions=True)
        self.assertTrue(torch.equal(original,out.logits));self.assertTrue(all(torch.equal(a,b) for a,b in zip(weights,model.parameters())))
        self.assertTrue(any(r['component']=='actual_attention_impact_reduction' for r in profile.finish()['components']))

if __name__=='__main__':unittest.main()
