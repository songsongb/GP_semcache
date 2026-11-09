"""Explicit TEST_FIXTURE_ONLY checks; no real NATURAL/GPU outcome is inferred."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import contextlib,importlib.util,io,json,multiprocessing,os,tempfile,unittest
from unittest.mock import patch
from semcache.edgelora.communication import digest,Channel,connect,header
from semcache.edgelora.r3b_fixture import ArrayTensor
from semcache.edgelora.r5_subsequence_physical import PhysicalOwner,EXACT,APPROX
from semcache.edgelora.r5_subsequence_integrity import SCOPED
from semcache.edgelora.r5_subsequence_audit import replay_validation
from semcache.edgelora.r4c_evidence import Evidence as Writer
from semcache.edgelora.r5_i5r2_tensors import Evidence
from semcache.edgelora.r5_i5_evidence import seal as old_seal
from semcache.edgelora.r5_subsequence_natural_io import output,put,seal,verify
from semcache.edgelora import r5_natural_approx as api
from semcache.edgelora.r5_natural_approx_audit import numerical,row_categories
from test_r5_continuous_natural_phase2 import fixture_inputs
import test_r5_sparse_physical as physical_fixture

ROOT=Path(__file__).resolve().parents[1]


class Approx(unittest.TestCase):
    def setUp(self):
        self.w,_,_,_,self.tok,self.users=fixture_inputs()
        self.f=physical_fixture.Physical();self.f.setUp()

    def test_original_first_five_preserved_and_no_priming(self):
        before=deepcopy(self.w);b,provenance=api.track_tasks(self.w,'NATURAL',self.tok)
        self.assertEqual(b,self.w['blocks'][0]);self.assertEqual(len(b['tasks']),5)
        self.assertEqual(provenance['priming_requests'],0);self.assertEqual(self.w,before)
        b['tasks'][0]['token_ids'][0]+=1;self.assertEqual(self.w,before)

    def test_controlled_source_is_genuine_fixed_nonprefix_slice(self):
        b,p=api.track_tasks(self.w,'CONTROLLED',self.tok);source,target=b['tasks'];original=self.w['blocks'][0]['tasks'][0]
        self.assertEqual(source['token_ids'],original['token_ids'][32:35]);self.assertEqual(len(source['token_ids']),3)
        self.assertFalse(source['scored']);self.assertTrue(target['scored']);self.assertEqual(source['policy']['max_new_tokens'],1)
        self.assertEqual(target['token_ids'],original['token_ids']);self.assertEqual(target['policy'],original['policy'])
        self.assertEqual(source['adapter_id'],target['adapter_id']);self.assertEqual(p['source_slice'],[32,35])
        self.assertEqual([t['request_sequence_number'] for t in b['tasks']],[0,1])
        repeated=deepcopy(self.w);repeated['blocks'][0]['tasks'][0]['token_ids'][32:35]=repeated['blocks'][0]['tasks'][0]['token_ids'][:3]
        body=dict(repeated['blocks'][0]);body.pop('block_id');repeated['blocks'][0]['block_id']=digest(body)
        with self.assertRaisesRegex(ValueError,'also a prefix'):api.track_tasks(repeated,'CONTROLLED',self.tok)

    def test_explicit_condition_modes_and_no_other_tracks(self):
        self.assertEqual([api.execution(c) for c in api.CONDITIONS],[(EXACT,False),(EXACT,False),(APPROX,True)])
        with self.assertRaises(ValueError):api.execution('SEMCACHE_OTHER')
        with self.assertRaises(ValueError):api.track_tasks(self.w,'EXPANDED_150',self.tok)

    def test_three_condition_resource_guard_and_original_limits(self):
        tasks=deepcopy(self.w['blocks'][0]['tasks'])
        for t in tasks:t['token_ids']=list(range(512));t['policy']['max_new_tokens']=32
        r=api.preflight(tasks)
        self.assertLess(r['run_evidence_forecast_bytes'],64*2**30);self.assertLess(r['estimated_run_file_count'],500000)
        self.assertEqual(r['run_limit_bytes'],64*2**30);self.assertEqual(r['per_adapter_es_budget_bytes'],16*2**30)
        self.assertEqual(r['per_adapter_ud_budget_bytes'],4*2**30);self.assertEqual(r['total_run_timeout_seconds'],14460)
        for t in tasks:t['policy']['max_new_tokens']=96
        with self.assertRaises(ValueError):api.preflight(tasks)

    def test_optins_and_allocation_fail_before_any_gpu_setup(self):
        with patch.dict(os.environ,CUDA_VISIBLE_DEVICES=''):
            with self.assertRaises(ValueError):api.prepare('old','multi','sha','NATURAL','unused')
        with self.assertRaises(ValueError):api.run('p','o','sha','cpu',enable_gpu_experiment=True)
        with patch.dict(os.environ,{},clear=True),patch('semcache.edgelora.r4a_runtime.setup',side_effect=AssertionError('GPU setup forbidden')):
            with self.assertRaises(ValueError):api.run('p','o','sha','cpu',enable_gpu_experiment=True,enable_approximate_experiment=True)

    def test_complete_plan_reconstruction_and_pinned_mode(self):
        with tempfile.TemporaryDirectory() as tmp,patch.dict(os.environ,CUDA_VISIBLE_DEVICES=''):
            root=Path(tmp);original=root/'original';multi=root/'multi';original.mkdir();multi.mkdir()
            from semcache.edgelora.r5_evidence import seal as multi_seal
            multi_seal(multi)
            impl=dict(source={'commit':'TEST_FIXTURE_ONLY'},files={})
            args=(self.w,{},dict(TEST_FIXTURE_ONLY=True),{},dict(base_prepared='TEST_FIXTURE_ONLY'),{}, {},self.users,SimpleNamespace(tokenizer=self.tok,identity={'TEST_FIXTURE_ONLY':True}))
            def fixture_transport(tasks,*args):return dict(TEST_FIXTURE_ONLY=True,rows=[dict(request_id=t['request_id'],maximum_header_bytes=12000) for t in tasks])
            with patch.object(api,'implementation',return_value=impl),patch.object(api,'inputs',return_value=args),patch.object(api,'transport',side_effect=fixture_transport):
                for track in ('CONTROLLED','NATURAL'):
                    plan=root/track;api.prepare(original,multi,'TEST_FIXTURE_ONLY',track,plan,True)
                    p,receipt,rest=api.load(plan);self.assertEqual(len(rest[0]['tasks']),2 if track=='CONTROLLED' else 5)
                    self.assertEqual(p['execution'][api.CONDITIONS[2]],dict(mode=APPROX,allow_approximate=True))
                    audit=output(root/(track+'_cpu'));put(audit/'cpu_validation.json',dict(status='PASS',failed=0,skipped=0,implementation=impl,plan_inventory=receipt));seal(audit)
                    self.assertIn('READY_FOR_MANUAL',api.inspect(plan,audit)['status'])
                    (plan/'approx_plan.json').write_text('{}')
                    with self.assertRaises(ValueError):api.load(plan)

    def test_runtime_adapter_passes_existing_mode_without_operator_replacement(self):
        with patch('semcache.edgelora.r5_subsequence_natural_runtime.worker',return_value='TEST_FIXTURE_ONLY') as worker:
            self.assertEqual(api.worker('p','o',api.CONDITIONS[2],'cpu'),'TEST_FIXTURE_ONLY')
            profile=worker.call_args.kwargs['_profile']
            self.assertEqual(profile.execution(api.CONDITIONS[2]),(APPROX,True));self.assertEqual(profile.script,api.SCRIPT)
        with patch('semcache.edgelora.r5_subsequence_natural_runtime.ud') as ud:
            api.ud('c','o');self.assertIs(ud.call_args.kwargs['_implementation'],api.implementation)

    def test_three_workers_share_deadline_without_gpu_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);resources=api.preflight(self.w['blocks'][0]['tasks'])
            p=dict(implementation={'source':{'commit':'TEST_FIXTURE_ONLY'}},resources=resources,
                multi=str(root/'multi'),original_experiment=str(root/'original'),dataset_paths={})
            calls=[]
            def fake_worker(argv,timeout):
                calls.append((argv[argv.index('--condition')+1],timeout))
                child=output(argv[argv.index('--output-dir')+1]);put(child/'identity.json',{'gpu':{'uuid':'TEST_FIXTURE_ONLY_NOT_RUN'}});seal(child)
                return 0
            with patch.dict(os.environ,dict(SLURM_JOB_ID='TEST_FIXTURE_ONLY',R5_ENABLE_GPU='1',R5_I5_ENABLE_GPU='1'),clear=True), \
                 patch.object(api,'approval',return_value=(p,{},None,{})),patch.object(api.shutil,'disk_usage',return_value=SimpleNamespace(free=100*2**30)), \
                 patch.object(api.time,'monotonic',side_effect=[0,0,7201,14400]),patch('semcache.edgelora.r5_subsequence_pilot.wait_worker',side_effect=fake_worker):
                r=api.run(root/'plan',root/'run','TEST_FIXTURE_ONLY',root/'cpu',True,True)
            self.assertEqual(r['status'],'COMPLETE');self.assertEqual([c for c,_ in calls],list(api.CONDITIONS))
            self.assertEqual([t for _,t in calls],[7200,7200,60]);verify(root/'run')

    def test_numerical_fail_is_separate_from_generation_and_quality(self):
        a=dict(generated_token_ids=[0],generated_text='same');c=dict(a)
        r=numerical([9.,0.],[0.,9.],a,c,2)
        self.assertEqual(r['numerical_tolerance'],'FAIL');self.assertTrue(r['generated_token_ids_equal'])
        self.assertFalse(r['bitwise_logits_equal']);self.assertIn('last_token_kl',r['full_vocabulary_per_step'][0])
        c=dict(generated_token_ids=[1,0],generated_text='different')
        r=numerical([9.,0.],[0.,9.,9.,0.],a,c,2)
        self.assertFalse(r['generation_length_equal']);self.assertFalse(r['final_output_histories_matched'])

    def test_actual_consumption_categories_and_zero_hit(self):
        tasks=[dict(request_id='S',token_ids=[1,2,3]),dict(request_id='T',token_ids=[9,1,2,3,4])]
        row=dict(by_layer_role=[dict(layer=0,role='q',consumed_windows=[dict(source_request_id='S',source_positions=[0,3],target_positions=[1,4])])],reused_projection_rows=3)
        self.assertEqual(row_categories({'T':row},tasks),dict(A=0,B=0,C=3,D=0))
        self.assertEqual(row_categories({'T':dict(by_layer_role=[],reused_projection_rows=0)},tasks),dict(A=0,B=0,C=0,D=0))

    def test_scoped_approx_authorization_replays_nonprefix_and_rejects_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'es').mkdir();writer=Writer(root/'es');owner=PhysicalOwner(self.f.ident,self.f.registry,evidence=writer,validation_mode=SCOPED)
            class Session:
                def correction(s,l,r,x,positions):return ArrayTensor([0.]*(len(positions)*4),[1,len(positions),4]) if positions else None
            results=[];requests=[];matching=[]
            for rid,ids in (('S',[1,2,3]),('T',[9,1,2,3,4])):
                req=self.f.req(rid,ids);requests.append(req);matching.append(owner.begin(req,{'cluster_id':0},mode=APPROX,allow_approximate=True))
                for l in range(2):
                    x=ArrayTensor([float(i) for i in range(len(ids)*4)],[1,len(ids),4])
                    for r in 'qkv':owner.project(l,r,x,lambda y:y.clone(),Session())
                # Existing scalar integrity fixture only, not a native Impact measurement.
                with patch('semcache.edgelora.r5_subsequence_physical.actual_attention_impact',return_value=1.):results.append(owner.complete(completed=True,attentions=['TEST_FIXTURE_ONLY']*2))
            old_seal(root);ev=Evidence(root);ev.verify();ev.registry=self.f.registry
            sources={'S':(requests[0],{(r['layer'],r['role']):r for r in results[0]['records']})}
            result=dict(condition='SEMCACHE_SUBSEQUENCE',context_mode=APPROX,matching=matching[1],assignment={'cluster_id':0},completion={'physical':results[1]})
            self.assertGreater(results[1]['physical_projection_rows_reused'],0)
            replay_validation(ev,requests[1],result,sources,allow_approximate=True)
            missing=deepcopy(result);missing['completion']['physical'].pop('integrity_validation')
            with self.assertRaises(ValueError):replay_validation(ev,requests[1],missing,sources,allow_approximate=True)
            with self.assertRaises(ValueError):replay_validation(ev,requests[1],result,sources)
            result['completion']['physical']['integrity_validation']['authorizations'][0]['admission_proof']='0'*64
            with self.assertRaises(ValueError):replay_validation(ev,requests[1],result,sources,allow_approximate=True)

    def test_cli_requires_approximate_optin_and_job_is_only_parsed(self):
        spec=importlib.util.spec_from_file_location('approx_cli',api.SCRIPT);cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)
        args=['prepare','--original-experiment','o','--multi','m','--source-commit','s','--track','NATURAL','--output-dir','new']
        with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):cli.main(args)
        with patch.object(api,'prepare',return_value={}) as prepare,contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(cli.main(args+['--enable-approximate-experiment']),0)
            self.assertTrue(prepare.call_args.kwargs['enable_approximate_experiment'])
        text=(ROOT/'jobs/r5_natural_approx.sh').read_text()
        self.assertIn('--enable-approximate-experiment',text);self.assertIn('241m',text)


class Native(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('transformers'),'native APPROX CPU smoke NOT_RUN: Torch/Transformers absent')
    def test_nonprefix_physical_skipping_real_opt_cpu_ud(self):
        import torch
        from transformers import OPTConfig,OPTForCausalLM
        from semcache.models.model_adapter import OPTModelAdapter
        from semcache.semantic.encoder import ControlledEncoder
        from semcache.edgelora.r5_semantic_index import SemanticIndex
        from semcache.edgelora.r5_subsequence_generation import generate
        from test_r5_subsequence_generation import torch_ud_child
        f=physical_fixture.Physical();f.setUp();ctx=multiprocessing.get_context('spawn');outcomes={}
        class Tok:
            def decode(self,ids,**kwargs):return ' '.join(map(str,ids))
        for condition in api.CONDITIONS:
            torch.manual_seed(42);cfg=OPTConfig(vocab_size=32,hidden_size=4,word_embed_proj_dim=4,ffn_dim=16,num_hidden_layers=2,num_attention_heads=2,max_position_embeddings=64,dropout=0.,attention_dropout=0.);cfg._attn_implementation='eager'
            model=OPTForCausalLM(cfg).eval();adapter=OPTModelAdapter(model);owner=None if condition==api.CONDITIONS[0] else PhysicalOwner(f.ident,f.registry,validation_mode=SCOPED)
            index=None if owner is None else SemanticIndex(ControlledEncoder({'fixture':[1.,0.]}))
            parent,kid=ctx.Pipe();proc=ctx.Process(target=torch_ud_child,args=(kid,f.ident,f.registry,f.matrices));proc.start();channel=None
            try:
                self.assertTrue(parent.poll(20));channel=Channel(connect(parent.recv()),'fixture_session',f.ident['adapter_matrix_fingerprint'])
                mode,allowed=api.execution(condition)
                for rid,ids in (('S',[1,2,3]),('T',[9,8,1,2,3,7])):
                    req=f.req(rid,ids)
                    r=generate(model,adapter,Tok(),req,channel,owner,index,condition='FULL_RECOMPUTE' if owner is None else 'SEMCACHE_SUBSEQUENCE',mode=mode,allow_approximate=allowed)
                    if rid=='T':outcomes[condition]=r
                channel.send(header('STOP','shutdown',-1));channel.recv(kind='BYE');self.assertTrue(parent.poll(20));parent.recv()
            finally:
                if owner:owner.dispose()
                if channel:channel.sock.close()
                proc.join(10)
                if proc.is_alive():proc.terminate();proc.join()
                parent.close()
            self.assertEqual(proc.exitcode,0)
        self.assertEqual(outcomes[api.CONDITIONS[1]]['completion']['physical']['physical_projection_rows_reused'],0)
        approx=outcomes[api.CONDITIONS[2]]
        self.assertGreater(approx['completion']['physical']['physical_projection_rows_reused'],0)
        self.assertGreater(sum(s['skipped_rows_per_role'] for s in approx['sparse_receipts']),0)


if __name__=='__main__':unittest.main()
