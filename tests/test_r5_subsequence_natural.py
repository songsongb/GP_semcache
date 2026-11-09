"""Synthetic CPU fixtures only. No inference/GPU/real workload conclusions."""
import copy,contextlib,importlib.util,io,json,os,shlex,subprocess,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from semcache.edgelora.communication import digest
from semcache.edgelora.r5_multi_artifacts import registry_for
from semcache.edgelora.r5_subsequence_natural import check_block,feasibility,resources,NaturalProfiler
from semcache.edgelora.r5_subsequence_natural_io import put,seal,verify,output,RequestWriter,save_result,load_result,JSON_LIMIT
from semcache.edgelora.r5_subsequence_physical import PhysicalOwner
from semcache.edgelora.r5_subsequence_integrity import SCOPED
from semcache.edgelora.r3b_fixture import ArrayTensor
from test_r5_subsequence_validation import Session
import test_r5_sparse_physical as fixture
ROOT=Path(__file__).resolve().parents[1]


def task(rid,ids,user='User_01',adapter='Adapter_A',budget=32):
    return dict(request_id=rid,sample_id=rid,request_sequence_number=0,dataset='qasper',group_id=rid,role='SCORED_TARGET',scored=True,user_id=user,adapter_id=adapter,token_ids=list(ids),policy=dict(max_new_tokens=budget),prepared=dict(sample=dict(context='REAL_DATA_UNAVAILABLE_CPU_FIXTURE_'+rid)))
def block(tasks):
    ts=copy.deepcopy(tasks)
    for i,t in enumerate(ts):t['request_sequence_number']=i
    b=dict(dataset='qasper',track='NATURAL',priming_requests=0,tasks=ts);b['block_id']=digest(b);return b

class Natural(unittest.TestCase):
    def setUp(self):self.users=registry_for(['Adapter_A','Adapter_B','Adapter_C'],'expanded');self.f=fixture.Physical();self.f.setUp()
    def test_equal_length_natural_requests_need_no_shorter_source(self):
        b=block([task('R1',[1,2,3,4,5,6]),task('R2',[1,2,3,4,5,7],user='User_02')]);r=feasibility(b,self.users)
        self.assertEqual(r['requests'][0]['raw_matching_target_windows'],0)
        self.assertEqual(r['requests'][1]['static_exact_context_windows'],3)
        self.assertEqual(r['requests'][1]['optimistic_nonoverlap_windows'],[[0,3]])
        self.assertEqual(r['requests'][1]['physical_hit'],'NOT_EVALUATED')
    def test_context_shift_and_cross_adapter_rejections_are_distinct(self):
        for ids,aid,user,reason in [([9,1,2,3,4,5,6],'Adapter_A','User_02','EXACT_POSITION_MISMATCH'),([1,2,3,4,5,6],'Adapter_B','User_11','CROSS_ADAPTER_REUSE_FORBIDDEN'),([9,2,3,4,5,6],'Adapter_A','User_02','EXACT_CAUSAL_CONTEXT_MISMATCH')]:
            r=feasibility(block([task('R1',[1,2,3,4,5,6]),task('R2',ids,user,aid)]),self.users)
            self.assertGreater(r['pairs'][0]['diagnostic_rejections'][reason],0);self.assertEqual(r['requests'][1]['static_exact_context_windows'],0)
    def test_no_match_and_independent_context_counts(self):
        r=feasibility(block([task('R1',[1,2,3]),task('R2',[7,8,9])]),self.users)
        self.assertEqual(r['potential_w3_occurrence_pairs'],0);self.assertEqual(r['independent_contexts'],2)
    def test_order_duplicate_priming_and_altered_block_rejected(self):
        for field,value in [('request_sequence_number',5),('sample_id','R1'),('scored',False),('role','PRIMING_SOURCE')]:
            b=block([task('R1',[1,2,3]),task('R2',[1,2,3])]);b['tasks'][1][field]=value;b.pop('block_id');b['block_id']=digest(b)
            with self.assertRaises(ValueError):check_block(b)
        b=block([task('R1',[1,2,3])]);b['tasks'][0]['token_ids'][0]=9
        with self.assertRaises(ValueError):check_block(b)
    def test_actual_sparse_transport_envelopes_all_generation_steps(self):
        from semcache.edgelora.r5_subsequence_natural import transport
        from semcache.edgelora.r4c_workload import policy
        pm={'identity':dict(model='facebook/opt-2.7b',base_revision='a'*40,base_weight_sha256='b'*64,dtype='float16',hidden_dtype='float16',layers=32,hidden_size=2560)}
        spec=dict(adapter_id='Adapter_A',matrix_sha256='c'*64,rank=8,matrix_dtype='float32')
        t=task('NATURAL',range(512),budget=96);t['input_text']='genuine prompt CPU envelope fixture';t['policy']=policy(dict(bos_token_id=2,eos_token_id=2,pad_token_id=1),'quality',96)
        report=transport([t],pm,{'Adapter_A':spec},{'tokenizer_revision':'d'*40},self.users)
        self.assertLess(report['rows'][0]['maximum_header_bytes']+8192,65536)
        self.assertGreater(report['rows'][0]['maximum_header_bytes'],0)
    def test_resource_512_by96_is_bounded_and_unchanged(self):
        tasks=[task('R'+str(i),list(range(512)),budget=96) for i in range(5)]
        r=resources(tasks);self.assertEqual(r['max_forwards'],480);self.assertEqual(r['max_context_tokens'],608)
        self.assertLess(r['probability_layer_bytes'],128*2**20);self.assertLess(r['gpu_estimate_bytes'],22*2**30)
        self.assertEqual(r['required_free_disk_bytes'],96*2**30)
    def test_resource_does_not_silently_drop_overbudget_tasks(self):
        with self.assertRaises(ValueError):resources([task(str(i),list(range(512)),budget=96) for i in range(20)])
    def test_hierarchical_seals_tamper_missing_extra_and_determinism(self):
        with tempfile.TemporaryDirectory() as tmp:
            a=output(Path(tmp)/'a');put(a/'es/r0/step.json',{'data':[1,2,3]});seal(a);original=verify(a)
            b=output(Path(tmp)/'b');put(b/'es/r0/step.json',{'data':[1,2,3]});seal(b);self.assertEqual(verify(b),original)
            put(a/'es/r0/step.json',{'data':[2,3,4]})
            with self.assertRaises(ValueError):verify(a)
            with self.assertRaises(ValueError):seal(b)
            (b/'extra').write_bytes(b'x')
            with self.assertRaises(ValueError):verify(b)
    def test_symlink_and_existing_historical_output_forbidden(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=output(Path(tmp)/'new');(p/'link').symlink_to('/tmp')
            with self.assertRaises(ValueError):seal(p)
            with self.assertRaises(ValueError):output(p)
            with self.assertRaises(ValueError):output(p/'nested',(p,))
    def test_json_artifact_bound_and_step_shards(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);r=dict(request_id='R1',steps=[{'record':{'sparse':[{'i':i}]}} for i in range(96)],sparse_receipts=[{'i':i} for i in range(96)])
            expected=copy.deepcopy(r);save_result(p,r);self.assertEqual(load_result(p,'R1'),expected)
            (p/'R1/step_000.json').unlink()
            with self.assertRaises(FileNotFoundError):load_result(p,'R1')
            with self.assertRaises(ValueError):put(p/'large.json',{'data':'x'*JSON_LIMIT})
    def test_tensor_writer_request_binding_and_replay_paths(self):
        from semcache.edgelora.r5_i5r2_tensors import Evidence
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);(p/'es').mkdir();w=RequestWriter(p/'es',request_ids=['R1','R2']);x=ArrayTensor([1.]*12,[1,3,4])
            with self.assertRaises(ValueError):w.blob(b'x','unbound')
            w.current='R1';ref=w.save(x,'R1_x');self.assertTrue(ref['file'].startswith('R1/'));self.assertEqual(Evidence(p).tensor('es',ref),x.fixture_bytes())
            w.current='R2';w.save(x,'R2_x');seal(p);verify(p)
    def run_request(self,owner,rid,ids,value=1.):
        req=self.f.req(rid,ids);owner.begin(req,{'cluster_id':0})
        for layer in range(2):
            for role in 'qkv':owner.project(layer,role,ArrayTensor([value]*(len(ids)*4),[1,len(ids),4]),lambda t:t.clone(),Session())
        with patch('semcache.edgelora.r5_subsequence_physical.actual_attention_impact',return_value=1.):return owner.complete(completed=True,attentions=['EXPLICIT_CPU_FIXTURE']*2)
    def test_empty_completion_residency_chu_and_natural_equal_length(self):
        events=[];o=PhysicalOwner(self.f.ident,self.users,validation_mode=SCOPED,trace=events.append)
        self.assertEqual(len(o.cache.entries),0);self.assertEqual(self.run_request(o,'R1',[1,2,3,4,5,6])['physical_projection_rows_reused'],0)
        self.assertGreater(o.cache.physical_tensor_bytes,0);r=self.run_request(o,'R2',[1,2,3,4,5,7]);self.assertGreater(r['physical_projection_rows_reused'],0)
        self.assertTrue(any(e['kind']=='CHU' for e in events));self.assertFalse(any(e['kind']=='PBR' for e in events))
        o.dispose();self.assertFalse(o.cache.entries);self.assertFalse(o.proofs)
    def test_no_admission_before_completion_and_failed_cleanup(self):
        o=PhysicalOwner(self.f.ident,self.users,validation_mode=SCOPED);o.begin(self.f.req('R1',[1,2,3]),{'cluster_id':0})
        self.assertFalse(o.cache.entries)
        with self.assertRaises(ValueError):o.complete(completed=False,attentions=[])
        o.abort();self.assertIsNone(o.active);self.assertFalse(o.validation.open)
    def test_layer_mismatch_native_computation_and_safe_rejection(self):
        o=PhysicalOwner(self.f.ident,self.users,validation_mode=SCOPED);self.run_request(o,'R1',[1,2,3,4,5,6])
        r=self.run_request(o,'R2',[1,2,3,4,5,6],value=2.)
        self.assertEqual(r['physical_projection_rows_reused'],0)
        self.assertTrue(all(record['native_projection_rows']==6 for record in r['records']))
    def test_interleaved_adapter_owner_isolation(self):
        a=PhysicalOwner(self.f.ident,self.users,validation_mode=SCOPED);b=PhysicalOwner(self.f.ident,self.users,validation_mode=SCOPED)
        self.run_request(a,'A1',[1,2,3,4,5,6]);self.assertFalse(b.cache.entries)
        self.run_request(b,'B1',[7,8,9,10,11,12]);self.assertGreater(self.run_request(a,'A2',[1,2,3,4,5,6])['physical_projection_rows_reused'],0)
        a.dispose();self.assertTrue(b.cache.entries);b.dispose()
    def test_pbr_uses_actual_configured_trigger(self):
        events=[];o=PhysicalOwner(self.f.ident,self.users,validation_mode=SCOPED,trace=events.append)
        o.bridge.pbr_interval=2 # Explicit fixture only, actual NATURAL remains100.
        self.run_request(o,'R1',[1,2,3]);self.assertFalse(any(e['kind']=='PBR' for e in events))
        self.run_request(o,'R2',[1,2,3]);self.assertEqual(sum(e['kind']=='PBR' for e in events),1)
    def test_profile_bounded_96_decode_passes_preserves_eligibility(self):
        p=NaturalProfiler();record=dict(boundary='native_projection_layout',phase='DECODE',layer=0,role='q',shape=[1,1,2560])
        for _ in range(95):p.detail(record)
        p.detail(dict(boundary='layer_eligibility',layer=0,reason='PER_LAYER_INPUT_BYTES_MISMATCH'))
        self.assertEqual(len(p.details),2);self.assertEqual(p.finish()['decode_layout_counts']['0:q'],95)
        with self.assertRaises(ValueError):p.detail(dict(record,shape=[1,2,2560]))
    def cli(self):
        s=importlib.util.spec_from_file_location('natural_cli',ROOT/'scripts/153_r5_subsequence_natural.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
    def test_actual_job_cli_contract_and_missing_cpu_approval(self):
        script=(ROOT/'jobs/r5_subsequence_natural.sh').read_text();argv=shlex.split(script.split('python -u scripts/153_r5_subsequence_natural.py ',1)[1].replace('\\\n',' '))
        with patch('semcache.edgelora.r5_subsequence_natural_runtime.run',return_value={'TEST_FIXTURE_ONLY':True}) as run,contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(self.cli().main(argv),0);self.assertEqual(run.call_args.kwargs['cpu_audit'],'$D5N_CPU_AUDIT')
        pos=argv.index('--cpu-audit');argv=argv[:pos]+argv[pos+2:]
        with self.assertRaises(SystemExit),contextlib.redirect_stderr(io.StringIO()):self.cli().main(argv)
        self.assertIn('04:30:00',script);self.assertIn('[[ -d logs',script)
    def test_prepare_dispatch_no_prompt_rebuilding(self):
        with patch('semcache.edgelora.r5_subsequence_natural.prepare',return_value={}) as prepare,contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(self.cli().main(['prepare','--original-experiment','old','--multi','m','--source-commit','s','--block-index','0','--output-dir','new']),0)
            self.assertEqual(prepare.call_args.kwargs['original_experiment'],'old');self.assertEqual(prepare.call_args.kwargs['block_index'],0)
    def test_complete_sealed_plan_reader_cpu_audit_and_inspect_contract(self):
        from semcache.edgelora import r5_subsequence_natural as api
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);original=output(root/'original');multi=output(root/'multi');put(original/'immutable.json',{'original':True});seal(original)
            # Explicit fixture plumbing; production inputs() is not relaxed.
            from semcache.edgelora.r5_i5_evidence import seal as old_seal
            old_seal(multi)
            from semcache.edgelora.r5_i5r2_tensors import Evidence
            inv=Evidence(multi).verify();tasks=[task('R'+str(i),range(512),budget=96) for i in range(5)]
            b=block(tasks);b['repetition']=0;b.pop('block_id');b['block_id']=digest(b)
            w={'blocks':[b]};src={'commit':'CPU_FIXTURE_NOT_A_GPU_COMMIT'};impl={'source':src,'files':{}}
            args=(w,{},verify(original),{},dict(base_prepared='CPU_FIXTURE'),{}, {},self.users,SimpleNamespace(identity={'TEST_FIXTURE_ONLY':True}))
            plan=root/'plan'
            with patch.object(api,'implementation',return_value=impl),patch.object(api,'inputs',return_value=args),patch.object(api,'transport',return_value={'TEST_FIXTURE_ONLY':True}),patch.object(api.Evidence,'verify',return_value=inv):
                api.prepare(str(original),str(multi),src['commit'],0,plan)
                p,pr,*_=api.load(plan);self.assertEqual(p['resources']['max_forwards'],480)
                audit=output(root/'cpu');put(audit/'cpu_validation.json',dict(status='PASS',failed=0,skipped=0,implementation=impl,plan_inventory=pr));seal(audit)
                self.assertEqual(api.inspect(plan,audit)['status'],'READY_FOR_MANUAL_ONE_BLOCK_ONLY_GPU_NOT_RUN')
                (plan/'natural_feasibility.json').write_text('{}')
                with self.assertRaises(ValueError):api.load(plan)
    @unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('transformers'),'installed native NATURAL CPU NOT_RUN: torch/transformers absent')
    def test_native_cpu_three_natural_requests_exact_mode_real_ud(self):
        import multiprocessing,torch
        from transformers import OPTConfig,OPTForCausalLM
        from semcache.models.model_adapter import OPTModelAdapter
        from semcache.edgelora.r5_subsequence_generation import generate
        from semcache.edgelora.r5_semantic_index import SemanticIndex
        from semcache.semantic.encoder import ControlledEncoder
        from semcache.edgelora.communication import Channel,connect,header
        from test_r5_subsequence_generation import torch_ud_child
        ctx=multiprocessing.get_context('spawn');parent,kid=ctx.Pipe();proc=ctx.Process(target=torch_ud_child,args=(kid,self.f.ident,self.f.registry,self.f.matrices));proc.start();channel=None
        try:
            self.assertTrue(parent.poll(20));channel=Channel(connect(parent.recv()),'fixture_session',self.f.ident['adapter_matrix_fingerprint'])
            cfg=OPTConfig(vocab_size=32,hidden_size=4,word_embed_proj_dim=4,ffn_dim=16,num_hidden_layers=2,num_attention_heads=2,max_position_embeddings=64,dropout=0.,attention_dropout=0.)
            cfg._attn_implementation='eager';model=OPTForCausalLM(cfg).eval();adapter=OPTModelAdapter(model)
            index=SemanticIndex(ControlledEncoder({'fixture':[1.,0.]}));owner=PhysicalOwner(self.f.ident,self.f.registry,validation_mode=SCOPED)
            class Tok:
                def decode(self,ids,**kwargs):return ' '.join(map(str,ids))
            rows=[]
            for i in range(3):
                req=self.f.req('N'+str(i),[1,2,3,4,5,6],user='User_01' if not i else 'User_02');req['request_sequence_number']=i;req.pop('request_sha256');req['request_sha256']=digest(req)
                r=generate(model,adapter,Tok(),req,channel,owner,index,condition='SEMCACHE_SUBSEQUENCE',profiler=NaturalProfiler())
                rows.append(r)
            self.assertEqual(rows[0]['completion']['physical']['physical_projection_rows_reused'],0)
            self.assertGreater(rows[1]['completion']['physical']['physical_projection_rows_reused'],0)
            self.assertEqual(rows[1]['logits_blob'],rows[2]['logits_blob'])
            self.assertFalse(owner.validation.open);owner.dispose();self.assertFalse(owner.cache.entries)
            channel.send(header('STOP','shutdown',-1));channel.recv(kind='BYE');self.assertTrue(parent.poll(20));parent.recv()
        finally:
            if channel:channel.sock.close()
            proc.join(10)
            if proc.is_alive():proc.terminate();proc.join()
            parent.close()
        self.assertEqual(proc.exitcode,0)
    def test_shell_requires_manual_allocation_before_execution(self):
        r=subprocess.run(['bash',str(ROOT/'jobs/r5_subsequence_natural.sh')],env={'PATH':os.environ['PATH']},capture_output=True,text=True)
        self.assertNotEqual(r.returncode,0);self.assertIn('manual allocation required',r.stderr)

if __name__=='__main__':unittest.main()
