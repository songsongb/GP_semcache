"""CPU-only policy/campaign fixtures; no GPU performance or HIT claims."""
import copy,importlib.util,json,os,subprocess,sys,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
from semcache.edgelora.communication import digest
from semcache.edgelora.r3a_artifacts import THRESHOLDS,write
from semcache.edgelora.r5_i5r4_policy import Engine,parameters,PolicyOwner,PolicyController,CAPACITY,factories
from semcache.edgelora.r5_i5r4 import resource_plan,simulate,cpu_receipt,reviewed,output,schedule,choose,build_trace,check_trace,task_block
from semcache.edgelora.r5_i5r4_analysis import policy_replay,seal_bundle,bundle,aggregate
from semcache.edgelora.r5_i5r2_tensors import Evidence
from semcache.edgelora.r5_i5_evidence import seal
from semcache.edgelora.r5_i5_contracts import request
from semcache.edgelora.r5_i5_cache import I5Owner
from semcache.edgelora.r5_i5_controller import I5Controller
from semcache.edgelora.r5_multi_artifacts import runtime_view,physical
from semcache.edgelora.r5_semantic_index import SemanticIndex
from semcache.edgelora.r4c_prepare import TinyTokenizer
from semcache.edgelora.r5_workload import generation_policy
from semcache.semantic.encoder import ControlledEncoder
import test_r5_multi as legacy

def params(**kwargs):return parameters(100.,20.,retention_ms_per_mib_request=.001,**kwargs)

class PolicyTests(unittest.TestCase):
    def test_two_real_240MiB_entries_compete(self):
        e=Engine('P3',params());e.begin('s1');e.admitted('1','a',512*491520);e.begin('s2');e.admitted('2','b',512*491520)
        self.assertEqual(e.victims(512*491520),['1']);self.assertEqual(512*491520,240*2**20)
    def test_ttl_request_idle_and_late_reentry(self):
        e=Engine('P2',params(ttl_idle_requests=2));e.begin('s');e.admitted('a','p',16)
        self.assertEqual(e.begin('idle1'),[]);self.assertEqual(e.begin('idle2'),[]);self.assertEqual(e.begin('idle3'),['a']);e.remove('a');e.admitted('a','p',16);self.assertEqual(e.entries['a']['born'],3)
    def test_no_ttl_baseline_and_admission_only(self):
        for policy in ('P0','P1','P3'):
            e=Engine(policy,params());e.begin('s');e.admitted('k','x',10)
            for i in range(18):self.assertEqual(e.begin(str(i)),[])
    def test_cold_exploration_is_explicit_and_causal(self):
        e=Engine('P1',parameters(0.,9999.));a=e.admission('a',16);b=e.admission('b',16)
        self.assertTrue(a['allowed']);self.assertTrue(a['exploration']);self.assertFalse(b['allowed']);self.assertEqual(a['expected_reuses'],2.)
        self.assertFalse(e.admission('oversize',CAPACITY+1)['allowed'])
    def test_source_forward_not_charged_twice(self):
        v=Engine('P4',params()).value('new',2**20);self.assertAlmostEqual(v['admission_net_ms'],200-20-.004);self.assertNotIn('source_forward',v)
    def test_deterministic_ties_and_no_eviction_case(self):
        e=Engine('P3',params(),32);e.begin('s');e.admitted('z','z',16);e.admitted('a','a',16)
        self.assertEqual(e.victims(1),['a']);e.remove('a');self.assertEqual(e.victims(1),[])
    def test_hit_history_uses_observed_requests_only(self):
        e=Engine('P4',params());e.begin('s');e.admitted('k','prefix',16);before=e.expected('prefix');e.begin('t');e.hit('k');self.assertGreater(e.expected('prefix'),before)
        with self.assertRaises(ValueError):e.hit('missing')
    def test_duplicate_request_fail_closed(self):
        e=Engine('P0',params());e.begin('one')
        with self.assertRaises(ValueError):e.begin('one')
    def test_bad_parameters_rejected(self):
        for kwargs in ({'future_uses':9},{'ttl_max':99},{'saving_ms':float('nan')},{'max_entries':3}):
            with self.assertRaises((ValueError,TypeError)):parameters(1,1,**kwargs)
    def test_entry_bound_64(self):
        e=Engine('P3',params(),1024);e.begin('s')
        for i in range(64):e.admitted(str(i),str(i),1)
        self.assertEqual(len(e.victims(1)),1)

class PolicyOwnerTests(unittest.TestCase):
    setUpClass=classmethod(legacy.MultiTests.setUpClass.__func__)
    tearDownClass=classmethod(legacy.MultiTests.tearDownClass.__func__)
    computed=legacy.MultiTests.computed
    def manager(self,policy='P0',capacity=12288,parameter=None):
        spec=self.adapters['Adapter_A'];view=runtime_view(self.pm,spec)
        identity,contract=physical(view,self.manifest['tokenizer'],'cpu','TEST_FIXTURE_ONLY',{'scope':'TEST_FIXTURE_ONLY','output_attentions':False},spec,self.registry)
        events=[];receipts=[];controls=[];index=SemanticIndex(ControlledEncoder({t:[1.,0.] for t in ('source','target','other')}),trace=lambda e:events.append(dict(sequence=len(events),adapter_id='Adapter_A',stream='semantic',event=e,observed_ns=time.monotonic_ns())))
        cls=I5Owner if policy in ('P0',None) else PolicyOwner
        owner=cls(view,digest(dict(preparation=self.manifest,adapter=spec)),'TEST_FIXTURE_ONLY','cpu',capacity,index=index,tokenizer=self.manifest['tokenizer'],sources={'fixture':'unit'},fixture=True,contract=contract,observed_cache_type=tuple,spec=spec,registry=self.registry,control=controls.append,trace=lambda e:events.append(dict(sequence=len(events),adapter_id='Adapter_A',stream='cache',event=e,observed_ns=time.monotonic_ns())))
        if policy is None:controller=I5Controller(owner,index,identity,condition='SEMCACHE_INTEGRATED',registry=self.registry)
        else:controller=PolicyController(owner,index,identity,condition='SEMCACHE_INTEGRATED',registry=self.registry,policy=policy,params=parameter or params(),receipt=receipts.append)
        return owner,controller,identity,events,receipts,controls
    def req(self,rid,ids,identity,user='User_01',text='source'):
        return request(dict(request_id=rid,user_id=user,adapter_id=identity['adapter_id'],input_text=text,dataset='coqa',sample_id=rid,request_sequence_number=0),ids,generation_policy(TinyTokenizer(),budget=2),identity,fixture=True)
    def perform(self,m,rid,ids,user='User_01'):
        o,c,i,events,logs,controls=m;r=self.req(rid,ids,i,user);before=o.snapshot();event,found=c.decide(r)
        blocks,proof=(None,None) if found else self.computed(ids);g=c.admit(r,event['assignment'],blocks,proof,completed=True)
        if found:o.reused(o.entries[found[1]['entry_id']].context,found[0])
        return dict(request_id=rid,source=None if found is None else found[1],admission=g,cache_before=before,cache_after=o.snapshot()),r
    def test_p0_exact_original_replay_equivalence(self):
        old=self.manager(None);new=self.manager('P0')
        trace=[('s0',[1,2,3,4]),('t0',[1,2,3,4,5]),('s1',[2,3,4,5]),('s2',[3,4,5,6]),('s0again',[1,2,3,4])]
        for rid,ids in trace:
            a,_=self.perform(old,rid,ids);b,_=self.perform(new,rid,ids);self.assertEqual(a,b)
        fields=('kind','context_sha256','reason','occupancy_bytes','lru_to_mru')
        self.assertEqual([{k:e[k] for k in fields if k in e} for e in old[0].events],[{k:e[k] for k in fields if k in e} for e in new[0].events]);self.assertEqual(old[-1],new[-1])
    def test_cost_eviction_invalidates_index_proof_and_UD(self):
        m=self.manager('P3');self.perform(m,'s0',[1,2,3,4]);self.perform(m,'s1',[2,3,4,5]);key=next(iter(m[0].entries));self.perform(m,'s2',[3,4,5,6])
        for mapping in (m[0].entries,m[0].index.entries,m[0].proofs,m[0].proof_seals):self.assertNotIn(key,mapping)
        self.assertTrue(any(x['action']=='EVICT_PREFIX' and x['entry_id']==key for x in m[-1]));self.assertEqual(len(m[0].entries),2)
    def test_bad_proof_cannot_evict_existing_entries(self):
        m=self.manager('P4');self.perform(m,'s0',[1,2,3,4]);self.perform(m,'s1',[2,3,4,5]);o,c,i,*_=m;before=list(o.entries)
        req=self.req('bad',[3,4,5,6],i);d,_=c.decide(req);blocks,proof=self.computed([3,4,5,6]);proof['layers'][0]['bitwise_equal']=False
        with self.assertRaises(ValueError):c.admit(req,d['assignment'],blocks,proof,completed=True)
        self.assertEqual(list(o.entries),before)
    def test_hidden_lower_level_eviction_is_fatal(self):
        m=self.manager('P3');self.perform(m,'s0',[1,2,3,4]);self.perform(m,'s1',[2,3,4,5])
        with patch.object(m[1].engine,'victims',return_value=[]):
            with self.assertRaisesRegex(ValueError,'hidden lower-level'):self.perform(m,'s2',[3,4,5,6])
    def test_authorized_cross_user_and_unauthorized_rejected(self):
        m=self.manager('P4');self.perform(m,'source',[1,2,3,4]);row,_=self.perform(m,'target',[1,2,3,4,5],user='User_02');self.assertIsNotNone(row['source'])
        before=m[1].engine.snapshot()
        with self.assertRaises(ValueError):m[1].decide(self.req('wrong',[1,2,3,4,5],m[2],user='User_03'))
        self.assertEqual(before,m[1].engine.snapshot())
    def test_ttl_removes_full_actual_owner_chain(self):
        m=self.manager('P2',parameter=params(ttl_idle_requests=2));self.perform(m,'s',[1,2,3,4]);key=next(iter(m[0].entries))
        # Oversize completed real misses cannot enter this small fixture cache.
        for rid in ('gap1','gap2','gap3'):self.perform(m,rid,[6,5,4,3,2,1,6,5,4])
        self.assertNotIn(key,m[0].entries);self.assertNotIn(key,m[0].proofs);self.assertNotIn(key,m[0].index.entries)
        self.assertTrue(any(e.get('reason')=='I4_IDLE_TTL_EXPIRED' for e in m[0].events))
    def test_no_eviction_and_explicit_owner_disposal(self):
        m=self.manager('P4');self.perform(m,'s',[1,2,3,4]);self.perform(m,'t',[1,2,3,4,5]);self.assertFalse(any(e['kind']=='EVICT' for e in m[0].events))
        for key in list(m[0].entries):m[0].remove(key,'INVALIDATE','fixture session disposal')
        self.assertFalse(m[0].entries or m[0].index.entries or m[0].proofs or m[0].proof_seals);self.assertEqual(m[0].bytes,0)
    def test_namespace_mutation_fails_closed(self):
        m=self.manager('P3');self.perform(m,'s',[1,2,3,4]);m[0].gpu_uuid='CHANGED'
        with self.assertRaises(ValueError):self.perform(m,'t',[1,2,3,4,5])
    def test_policy_receipt_independent_replay_and_tamper(self):
        m=self.manager('P3');rows={};reqs={}
        for rid,ids in [('s0',[1,2,3,4]),('s1',[2,3,4,5]),('s2',[3,4,5,6])]:rows[rid],reqs[rid]=self.perform(m,rid,ids)
        # Pure fixture capacity is smaller than real campaign constant.
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            for name,rs in [('i4_policy',m[4]),('events',m[3])]:
                (p/(name+'.jsonl')).write_text(''.join(json.dumps(x)+'\n' for x in rs))
            with patch('semcache.edgelora.r5_i5r4_analysis.CAPACITY',12288):
                r=policy_replay(p,{'i4':{'policy':'P3','parameters':params()}},reqs,rows);self.assertEqual(r['status'],'PASS')
                m[4][0]['state']['index']=99;(p/'i4_policy.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in m[4]))
                with self.assertRaises(ValueError):policy_replay(p,{'i4':{'policy':'P3','parameters':params()}},reqs,rows)
    def test_incomplete_source_rejected(self):
        m=self.manager('P1');req=self.req('s',[1,2,3,4],m[2]);d,_=m[1].decide(req)
        with self.assertRaises(ValueError):m[1].admit(req,d['assignment'],None,None,completed=False)
    def test_semantic_overlap_and_equal_length_never_authorize(self):
        m=self.manager('P4');self.perform(m,'source',[1,2,3,4])
        for rid,ids,expected in [('overlap',[2,3,4,5,6],'EXACT_PREFIX_TOKEN_MISMATCH'),('equal',[1,2,3,4],'TARGET_NOT_LONGER_THAN_SOURCE')]:
            d,found=m[1].decide(self.req(rid,ids,m[2]));self.assertIsNone(found);self.assertTrue(d['candidates']);self.assertEqual(d['candidate_reasons'][0]['rejection'],expected)
    def test_physical_semantic_and_policy_independent_replay(self):
        for policy in ('P0','P1','P2','P3','P4'):
            with self.subTest(policy=policy):self.physical_case(policy)
    def physical_case(self,policy):
        from semcache.edgelora.r5_i5r4_analysis import physical_replay
        m=self.manager(policy);o,c,ident,events,receipts,controls=m;rows={};reqs={};steps={};ownership=[];isolation=[]
        for rid,ids in [('s0',[1,2,3,4]),('t0',[1,2,3,4,5]),('s1',[2,3,4,5]),('s2',[3,4,5,6])]:
            r=self.req(rid,ids,ident);reqs[rid]=r;before=o.snapshot();d,found=c.decide(r);blocks,proof=(None,None) if found else self.computed(ids);end=time.monotonic_ns();g=c.admit(r,d['assignment'],blocks,proof,completed=True)
            if found:o.reused(o.entries[found[1]['entry_id']].context,found[0])
            rows[rid]=dict(source=None if not found else found[1],admission=g,cache_before=before,cache_after=o.snapshot(),model_end_ns=end,completion_ns=time.monotonic_ns(),role='SCORED_TARGET',candidate_reasons=d['candidate_reasons'],virtual_selected_entry_id=d['virtual_selected_entry_id'],candidate_count=len(d['candidates']),exact_prefix_eligible=any(x['rejection'] is None for x in d['candidate_reasons']),rejected_cross_adapter_candidates=0,evidence_profile='SCALABLE_EVALUATION')
            isolation.append(dict(request_id=rid,rejected=[],foreign_tensor_reads=0,foreign_wire_frames=0))
            if g:
                owned=o.entries[g['entry_id']];entry=next(iter(owned.pool.entries.values()));specs=owned.validate(entry,owned.context)
                ownership.append(dict(entry_id=g['entry_id'],source_request_id=rid,source_user=r['user_id'],adapter_id='Adapter_A',registry_sha256=self.registry['sha256'],sharing_authorization=self.registry['users'][r['user_id']],context=owned.context,proof=copy.deepcopy(o.proofs[g['entry_id']]),seal=owned.seals[entry.key],tensor_specs=specs,physical_bytes=owned.bytes))
                steps[rid]=[dict(qkv=specs,raw_kv_proof=copy.deepcopy(o.proofs[g['entry_id']]),kv=[{role:layer['actual_'+role] for role in 'kv'} for layer in proof['layers']])]
        plan=dict(i4={'policy':policy,'parameters':params()},device='cpu',fixture=True,sources={'fixture':'unit'},condition='SEMCACHE_INTEGRATED',block={'resources':{'per_adapter_capacity_bytes':12288}})
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)
            for name,rs in [('events',events),('i4_policy',receipts),('ownership',ownership),('isolation',isolation)]:
                (p/(name+'.jsonl')).write_text(''.join(json.dumps(x)+'\n' for x in rs))
            for aid in self.adapters:
                (p/aid/'es').mkdir(parents=True);write(p/aid/'identity.json',{'physical_identity':{'gpu_uuid':'TEST_FIXTURE_ONLY'},'contract':o.contract})
                write(p/aid/'final.json',{'cache':o.snapshot() if aid=='Adapter_A' else {'occupancy_bytes':0}})
                (p/aid/'es/controls.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in controls if aid=='Adapter_A'))
            with patch('semcache.edgelora.r5_i5r4_analysis.CAPACITY',12288):
                rule=policy_replay(p,plan,reqs,rows);r=physical_replay(p,plan,self.registry,self.adapters,self.pm,self.manifest,reqs,rows,steps,rule['expected_removals']);self.assertEqual(r['actual_hits'],1);self.assertEqual(r['actual_evictions'],1)
                bad=copy.deepcopy(rows);bad['s2']['source']={'entry_id':'FAKE'}
                with self.assertRaises((ValueError,KeyError)):physical_replay(p,plan,self.registry,self.adapters,self.pm,self.manifest,reqs,bad,steps,rule['expected_removals'])

class CampaignTests(unittest.TestCase):
    def test_production_factories_restored(self):
        from semcache.edgelora.r5_i5r4_runtime import injection
        from semcache.edgelora import r5_i5_runtime
        original=(r5_i5_runtime.MultiOwner,r5_i5_runtime.I5Controller)
        with self.assertRaises(RuntimeError):
            with injection('P4',params(),lambda r:None):
                self.assertIs(r5_i5_runtime.MultiOwner,PolicyOwner);raise RuntimeError('fixture failure')
        self.assertEqual((r5_i5_runtime.MultiOwner,r5_i5_runtime.I5Controller),original)
        self.assertIs(factories('P0',params(),lambda r:None)[0],I5Owner)
    def test_historical_SHA_cannot_be_new_source(self):
        old='f7157c9b8b97fdce56f87dcbc3566cb677404bb8'
        with patch('semcache.edgelora.r5_i5r4.source_identity',return_value={'commit':old,'files':{}}):
            with self.assertRaises(ValueError):reviewed(old)
    def test_no_input_verification_bypass(self):
        from semcache.edgelora.r5_i5r4 import load
        with self.assertRaisesRegex(ValueError,'cannot be disabled'):load('/unavailable',recheck_inputs=False)
    def test_child_seals_bundle_and_corruption(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'children/x').mkdir(parents=True);write(root/'children/x/file.json',{'ok':1});seal(root/'children/x');write(root/'status.json',{'status':'FAILED'});seal_bundle(root)
            r=bundle(root);self.assertEqual(len(r['children']),1)
            (root/'children/x/file.json').write_text('bad')
            with self.assertRaises(ValueError):bundle(root)
    def test_duplicate_missing_child_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'children/x').mkdir(parents=True);seal(p/'children/x');seal_bundle(p);(p/'children/extra').mkdir()
            with self.assertRaises(ValueError):bundle(p)
    def test_fresh_output_and_sealed_ancestor_protection(self):
        with tempfile.TemporaryDirectory() as d:
            p=output(Path(d)/'one');seal(p)
            with self.assertRaises(ValueError):output(p)
            with self.assertRaises(ValueError):output(p/'inside')
    def test_failure_finalization_preserves_primary(self):
        from semcache.edgelora.r5_i5r4_runtime import finalize
        with tempfile.TemporaryDirectory() as d:
            (Path(d)/'children').mkdir();primary=RuntimeError('primary')
            with patch('semcache.edgelora.r5_i5r4_analysis.seal_bundle',side_effect=ValueError('secondary')),patch('sys.stderr') as err:finalize(d,{'status':'FAILED'},primary)
            if hasattr(primary,'__notes__'):self.assertIn('secondary',' '.join(primary.__notes__))
            else:self.assertTrue(err.write.called)
            self.assertEqual(str(primary),'primary');self.assertEqual(json.loads((Path(d)/'i4_run_status.json').read_text())['status'],'FAILED')
    def test_cli_real_dispatch_all_public_actions(self):
        spec=importlib.util.spec_from_file_location('i4_cli',Path(__file__).resolve().parents[1]/'scripts/148_r5_i5r4.py');cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)
        args=['prepare']
        for key in ('output-dir','source-commit','plan-3a','audit-3a','verification-3a','expected-3a-commit','be-experiment','be-run','be-audit','be-verification','nf1-audit'):args+=['--'+key,key]
        with patch('semcache.edgelora.r5_i5r4.prepare',return_value={'status':'TEST_FIXTURE_ONLY'}) as fn:
            self.assertEqual(cli.main(args),0);self.assertEqual(fn.call_args.kwargs['source_commit'],'source-commit');self.assertEqual(fn.call_args.kwargs['trace_units'],4)
        for action,keys,module in [('cpu-validate',('experiment','output-dir'),'r5_i5r4'),('inspect',('experiment','cpu-audit','source-commit'),'r5_i5r4'),('run',('experiment','cpu-audit','source-commit','output-dir'),'r5_i5r4_runtime'),('analyze',('experiment','run','cpu-audit','output-dir'),'r5_i5r4_analysis'),('verify',('experiment','run','audit','cpu-audit','output-dir'),'r5_i5r4_analysis'),('report',('experiment','run','audit','verification','output-dir'),'r5_i5r4_analysis')]:
            args=[action]
            for k in keys:args+=['--'+k,k]
            if action=='inspect':args+=['--require-cpu-validation']
            if action=='run':args+=['--enable-gpu-experiment']
            with patch('semcache.edgelora.'+module+'.'+action.replace('-','_'),return_value={}) as fn:self.assertEqual(cli.main(args),0);fn.assert_called_once()
    def test_thresholds_unchanged(self):
        self.assertEqual(THRESHOLDS,{'atol':0.001,'rtol':0.001,'max_abs':0.01,'relative_l2':0.001,'last_token_kl':1e-5,'last_token_argmax_agreement':True})
    def test_module_import_no_torch(self):
        code="""import sys,importlib.abc
class Reject(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,*a):
  if fullname.split('.')[0] in ('torch','transformers'):raise RuntimeError(fullname)
sys.meta_path.insert(0,Reject())
import semcache.edgelora.r5_i5r4,semcache.edgelora.r5_i5r4_policy,semcache.edgelora.r5_i5r4_analysis
"""
        r=subprocess.run([sys.executable,'-c',code],env=dict(os.environ,PYTHONPATH='src'),capture_output=True,text=True);self.assertEqual(r.returncode,0,r.stderr)
    def test_analysis_verify_chain_recomputes_and_preserves_failure(self):
        from semcache.edgelora import r5_i5r4_analysis as api
        result=dict(source={'commit':'CPU_FIXTURE_ONLY'},plan_id='CPU_FIXTURE',children=[],summary={'NUMERICAL_FIDELITY':'FAIL'},policy_totals={},numerical={'thresholds':copy.deepcopy(THRESHOLDS),'comparisons':[]},independent_audit={'status':'PASS','GPU_NOT_RUN':True},run_inventory={},cpu_audit={})
        with tempfile.TemporaryDirectory() as d,patch.object(api,'compute',return_value=result) as independent:
            root=Path(d);api.analyze(root/'plan',root/'run',root/'cpu',root/'audit');api.verify(root/'plan',root/'run',root/'audit',root/'cpu',root/'verification')
            self.assertEqual(independent.call_count,2);self.assertEqual(json.loads((root/'audit/i4_summary.json').read_text())['NUMERICAL_FIDELITY'],'FAIL')
            write(root/'audit/i4_summary.json',{'NUMERICAL_FIDELITY':'PASS'});seal(root/'audit')
            with self.assertRaises(ValueError):api.verify(root/'plan',root/'run',root/'audit',root/'cpu',root/'bad_verification')
    def test_single_slurm_script_static_contract(self):
        path=Path(__file__).resolve().parents[1]/'jobs/r5_i5r4_integrated.sh';text=path.read_text()
        self.assertTrue(text.startswith('#!/usr/bin/env bash\n'))
        for line in ('#SBATCH --time=23:50:00','#SBATCH --cpus-per-task=4','#SBATCH --mem=64G','#SBATCH --partition=batch_eebme_ugrad'):
            self.assertIn(line,text)
        self.assertNotIn('#SBATCH --array',text);self.assertNotIn('#SBATCH --nodelist',text);self.assertNotIn('sbatch ',text)
        for key in ('SLURM_JOB_ID','I4_EXPERIMENT','I4_CPU_AUDIT','I4_SOURCE_COMMIT','I4_RUN_ROOT','I4_NODE','I5_CONDA_ROOT','I4_ENABLE_GPU','R5_I5_ENABLE_GPU','R5_ENABLE_GPU'):self.assertIn(key,text)
        self.assertIn('[[ -d logs',text);self.assertIn('set -Eeuo pipefail',text)
        self.assertEqual(subprocess.run(['bash','-n',str(path)],capture_output=True).returncode,0)

class WorkloadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from test_r5_i5r3b import prompt,USERS
        cls.users=USERS;cls.prompts=[]
        for group in range(8):
            chosen=[];prefix=[1+group%5,1+group//5]+[1]*510
            for i in range(300):
                p=prompt(f'I4Fixture_{group}_{i}',f'FIXTURE_context_{group}',prefix+[2]*767+[1+i%5])
                if p['route']['adapter_id']=='Adapter_A' and (not chosen or p['route']['user_id']!=chosen[0]['route']['user_id']):chosen.append(p)
                if len(chosen)==2:break
            assert len(chosen)==2
            if chosen[0]['token_ids']==chosen[1]['token_ids']:chosen[1]['token_ids'][-1]=6;chosen[1]['token_ids_sha256']=digest(chosen[1]['token_ids'])
            cls.prompts+=chosen
        cls.special={'bos_token_id':0,'eos_token_id':6,'pad_token_id':0}
    def trace(self):
        from test_r5_i5r3b import Tokenizer
        groups,_=choose(copy.deepcopy(self.prompts),self.users,2,set())
        ts=build_trace(groups[0],Tokenizer(),self.special,42,32)
        return ts,groups
    def test_three_competing_sources_and_original_routing(self):
        ts,groups=self.trace();r=check_trace(ts,self.users,42)
        self.assertEqual(r['distinct_source_prefixes'],3);self.assertEqual(r['priming_requests'],4);self.assertEqual(r['unique_scored_samples'],7);self.assertEqual(r['repeated_scored_observations'],8)
        self.assertNotEqual(ts[1]['user_id'],ts[2]['user_id']);self.assertEqual(ts[0]['token_ids'],ts[1]['token_ids'][:512]);self.assertEqual(ts[0]['token_ids'],ts[2]['token_ids'][:512])
        self.assertNotEqual(ts[0]['token_ids'],ts[7]['token_ids'][:512]);self.assertTrue(all(t['policy']['max_new_tokens']==32 for t in ts if t['scored']))
    def test_simulated_competition_late_reentry_negatives_and_no_future(self):
        ts,_=self.trace();a=simulate(ts,'P0',params(),self.users);b=simulate(ts,'P4',params(),self.users)
        self.assertTrue(any(r['victims'] for r in a));self.assertTrue(any(r['expired'] for r in b));self.assertIsNotNone(a[1]['simulated_prefix_opportunity']);self.assertIsNotNone(a[16]['simulated_prefix_opportunity'])
        self.assertTrue(all(a[i]['simulated_prefix_opportunity'] is None for i in range(7,13)))
        self.assertEqual(a[:7],simulate(ts[:7],'P0',params(),self.users))
    def test_disjoint_contexts_and_calibration_exclusion(self):
        groups,_=choose(copy.deepcopy(self.prompts),self.users,2,set());keys=[x['context_sha256'] for g in groups for x in g];self.assertEqual(len(set(keys)),6)
        negative=[g[0]['negative']['sample']['context'] for g in groups];self.assertEqual(len(set(negative)),2)
        with self.assertRaises(ValueError):choose(copy.deepcopy(self.prompts),self.users,3,set())
        with self.assertRaises(ValueError):choose(copy.deepcopy(self.prompts),self.users,2,set(keys[:4]))
    def test_no_forced_hits_and_bad_order_rejected(self):
        ts,_=self.trace();ts[0]['cache_grant']={}
        with self.assertRaises(ValueError):check_trace(ts,self.users,42)
        ts,_=self.trace();ts[0],ts[1]=ts[1],ts[0]
        for i,t in enumerate(ts):t['request_sequence_number']=i
        with self.assertRaises(ValueError):check_trace(ts,self.users,42)
    def test_balanced_fixed_order_p5_scope_sentinel_reset(self):
        ts,_=self.trace();sent=copy.deepcopy([ts[0],ts[2],ts[7]])
        for i,t in enumerate(sent):t.update(strict=True,request_sequence_number=i)
        jobs=schedule([ts,copy.deepcopy(ts)],sent,2,42);self.assertEqual(jobs,schedule([ts,copy.deepcopy(ts)],sent,2,42));self.assertEqual(len(jobs),30)
        for j in jobs:
            b=j['block'];self.assertEqual(b['cache_resets_per_condition'],1);self.assertTrue(all(t['strict']==(b['segment']=='STRICT_SENTINEL') for t in b['tasks']))
            if j['policy']=='P5':self.assertTrue(all(t['scored'] for t in b['tasks']))
        self.assertNotEqual([j['policy'] for j in jobs[:6]],[j['policy'] for j in jobs[12:18]])
        first=[j['policy'] for j in jobs[:6]];second=[j['policy'] for j in jobs[12:18]]
        self.assertTrue(all(first.index(policy)+second.index(policy)==5 for policy in first))
    def test_resource_rejects_excess_and_reports_bounds(self):
        ts,_=self.trace();b=task_block(ts,0,0,'MEASUREMENT');j=[dict(child_id='unit',block=b)]
        rates=dict(prefill_seconds_per_token=.001,decode_seconds_per_token=.01,admission_seconds=2,other_request_seconds=.1)
        r=resource_plan(j,rates);self.assertLess(r['GPU_peak_estimate_bytes'],22*2**30);self.assertEqual(r['CPU_UD_process_launches'],3);self.assertEqual(r['allocation_headroom_seconds'],750)
        self.assertEqual(r['required_free_disk_bytes'],16*2**30+32*2**30)
        with self.assertRaises(ValueError):resource_plan(j,dict(rates,prefill_seconds_per_token=99))
    def test_transport_each_real_contract_fixture_not_a_grant(self):
        from test_r5_i5r3b import Transport
        from semcache.edgelora.r5_i5r3b_transport import preflight
        helper=Transport();helper.setUp();ts,_=self.trace()
        plans={policy:dict(blocks=[{'tasks':ts if policy!='P5' else [t for t in ts if t['scored']]}],conditions=['FULL_RECOMPUTE' if policy=='P5' else 'SEMCACHE_INTEGRATED']) for policy in ('P0','P1','P2','P3','P4','P5')}
        r=preflight(plans,helper.pm,helper.specs,helper.tokenizer,self.users)
        self.assertEqual(len(r['rows']),110);self.assertTrue(all(x['maximum_header_bytes']+8192<=65536 for x in r['rows']))
        self.assertTrue(any('prefix_512_step_31' in x['scenario_header_bytes'] for x in r['rows']))
    def test_cpu_audit_cannot_bind_other_plan(self):
        ts,_=self.trace();p=dict(source={'commit':'TEST_FIXTURE'},plan_id='a',jobs=[dict(child_id='j',policy='P0',block={'tasks':ts})],policy_parameters=params(),registry=self.users)
        with tempfile.TemporaryDirectory() as d:
            write(Path(d)/'i4_cpu_audit.json',dict(status='PASS',source=p['source'],plan_id='OTHER',plan_inventory={},GPU_NOT_RUN=True));seal(d)
            with self.assertRaises(ValueError):cpu_receipt(p,{},d)
    def test_small_quality_scope_does_not_claim_noninferiority(self):
        rows=[dict(trace_unit=i,reference='P0',policy='P4',D_ms=-10,quality_delta_f1_points=0,warm_target_mean_delta_ms=-1) for i in range(3)]
        r=aggregate({'policies':['P0','P4'],'quality':{'minimum_independent_traces_for_confirmatory':8,'margin_f1_points':2.}},rows,[dict(reference='P0',policy='P4',frozen_tolerance_pass=False,generated_tokens_equal=True,generated_text_equal=True)])['P0_vs_P4']
        self.assertEqual(r['quality_noninferiority'],'DESCRIPTIVE_ONLY');self.assertEqual(r['NUMERICAL_FIDELITY'],'FAIL');self.assertEqual(r['CORRECTNESS_QUALIFIED_SPEEDUP'],'NOT_ESTABLISHED');self.assertEqual(r['independent_traces'],3)

class NativeCPU(unittest.TestCase):
    @unittest.skipUnless(all(importlib.util.find_spec(n) for n in ('torch','transformers','numpy')),'installed native OPT CPU integration NOT_RUN: dependencies unavailable')
    def test_installed_native_contract(self):
        from semcache.edgelora.r5_i5_preflight import installed_opt_check
        installed_opt_check()

if __name__=='__main__':unittest.main()
