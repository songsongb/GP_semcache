"""CPU-only policy/campaign fixtures; no GPU performance or HIT claims."""
import copy,importlib.util,json,os,re,subprocess,sys,tempfile,time,unittest,contextlib,io
from pathlib import Path
from unittest.mock import patch
from semcache.edgelora.communication import digest
from semcache.edgelora.r3a_artifacts import THRESHOLDS,write
from semcache.edgelora.r5_i5r4_policy import Engine,parameters,PolicyOwner,PolicyController,CAPACITY,factories
from semcache.edgelora.r5_i5r4 import resource_plan,simulate,cpu_receipt,reviewed,output,schedule,choose,build_trace,check_trace,task_block,discrimination
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

class SlurmInvocationTests(unittest.TestCase):
    """Execute the actual SH with parser-only command recorders, never Slurm/GPU.

    The fake python loads the real argparse parser but does not call main/run.
    No real git, Conda, timeout or inference command is invoked by this fixture.
    """
    def run_shell(self,root,*,omit_cpu=False,logs=True):
        root=Path(root);binpath=root/'bin';binpath.mkdir();conda=root/'conda/etc/profile.d';conda.mkdir(parents=True)
        (conda/'conda.sh').write_text('conda() { :; }\n')
        if logs:(root/'logs').mkdir()
        for name in ('plan','cpu','runs'):(root/name).mkdir()
        repo=Path(__file__).resolve().parents[1];script=(repo/'jobs/r5_i5r4_integrated.sh').read_text()
        if omit_cpu:
            before,sep,after=script.partition('stage=GPU_CAMPAIGN');gpu,sep2,tail=after.partition('stage=INDEPENDENT_ANALYSIS')
            self.assertIn('    --cpu-audit "$I4_CPU_AUDIT" \\\n',gpu)
            gpu=gpu.replace('    --cpu-audit "$I4_CPU_AUDIT" \\\n','',1);script=before+sep+gpu+sep2+tail
        (root/'job.sh').write_text(script)
        command={
            'git':'''#!/bin/bash
case "$*" in
 'branch --show-current') echo exp/raw-semcache-r5-integration ;;
 'rev-parse HEAD') echo "$I4_SOURCE_COMMIT" ;;
 'status --porcelain --untracked-files=no') : ;;
 *) exit 90 ;;
esac
''',
            'timeout':'''#!/bin/bash
while [[ "$1" == --* ]]; do shift; done
printf '%s\\n' "$1" >> "$I4_TEST_TIMEOUTS"
shift
exec "$@"
''',
            'python':f'''#!{sys.executable}
import importlib.util,json,os,sys
args=sys.argv[1:]
if args[0]=='-u':args.pop(0)
assert args.pop(0)=='scripts/148_r5_i5r4.py'
spec=importlib.util.spec_from_file_location('real_i4_parser',os.environ['I4_TEST_CLI'])
cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)
with open(os.environ['I4_TEST_ARGV'],'a') as f:f.write(json.dumps(args)+'\\n')
cli.parser().parse_args(args)
'''}
        for name,text in command.items():p=binpath/name;p.write_text(text);p.chmod(0o700)
        env=dict(os.environ,PATH=str(binpath)+os.pathsep+os.environ['PATH'],SLURM_JOB_ID='CPU_FIXTURE',SLURMD_NODENAME='moana-y3',I4_NODE='moana-y3',
                 I4_EXPERIMENT=str(root/'plan'),I4_CPU_AUDIT=str(root/'cpu'),I4_RUN_ROOT=str(root/'runs'),I5_CONDA_ROOT=str(root/'conda'),
                 I4_SOURCE_COMMIT='e'*40,I4_ENABLE_GPU='1',R5_I5_ENABLE_GPU='1',R5_ENABLE_GPU='1',
                 I4_TEST_CLI=str(repo/'scripts/148_r5_i5r4.py'),I4_TEST_ARGV=str(root/'argv.jsonl'),I4_TEST_TIMEOUTS=str(root/'timeouts.txt'))
        env.pop('SLURM_ARRAY_TASK_ID',None);env.pop('SLURM_ARRAY_JOB_ID',None)
        result=subprocess.run(['bash',str(root/'job.sh')],cwd=root,env=env,capture_output=True,text=True,timeout=20)
        argv=[json.loads(line) for line in (root/'argv.jsonl').read_text().splitlines()] if (root/'argv.jsonl').exists() else []
        return result,argv
    def test_actual_shell_commands_satisfy_real_parser(self):
        with tempfile.TemporaryDirectory() as d:
            result,calls=self.run_shell(d);self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual([c[0] for c in calls],['inspect','run','analyze','verify','report'])
            for c in calls:
                if c[0]!='report':self.assertEqual(c[c.index('--cpu-audit')+1],str(Path(d)/'cpu'))
            run=calls[1];self.assertEqual(run[run.index('--runtime-budget-seconds')+1],'68400');self.assertIn('--enable-gpu-experiment',run)
            self.assertEqual((Path(d)/'timeouts.txt').read_text().splitlines(),['15m','19h','3h','1h','20m'])
    def test_missing_cpu_audit_in_actual_run_command_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            result,calls=self.run_shell(d,omit_cpu=True);self.assertEqual(result.returncode,2)
            self.assertEqual([c[0] for c in calls],['inspect','run']);self.assertIn('--cpu-audit',result.stderr);self.assertIn('stage=GPU_CAMPAIGN',result.stderr)
    def test_missing_logs_rejected_before_any_cli_execution(self):
        with tempfile.TemporaryDirectory() as d:
            result,calls=self.run_shell(d,logs=False);self.assertNotEqual(result.returncode,0);self.assertFalse(calls);self.assertIn('logs must exist before job submission',result.stderr)

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

class DiscriminationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):WorkloadTests.setUpClass()
    def plan(self,parameter=None):
        from test_r5_i5r3b import Tokenizer
        groups,_=choose(copy.deepcopy(WorkloadTests.prompts),WorkloadTests.users,2,set())
        traces=[build_trace(g,Tokenizer(),WorkloadTests.special,42,32) for g in groups]
        sentinel=copy.deepcopy([traces[0][i] for i in (0,2,7)])
        for i,t in enumerate(sentinel):
            t.update(strict=True,request_sequence_number=i);t['policy']['max_new_tokens']=1 if not t['scored'] else 8
        return dict(jobs=schedule(traces,sentinel,2,42),registry=WorkloadTests.users,policy_parameters=parameter or params())
    def report(self,p):
        rs={j['child_id']:simulate(j['block']['tasks'],j['policy'],p['policy_parameters'],p['registry']) for j in p['jobs']}
        return discrimination(p,rs),rs
    def test_identical_admission_is_not_discriminated(self):
        r,_=self.report(self.plan());p0,p1=r['policy_summary']['P0'],r['policy_summary']['P1']
        self.assertEqual(p0['eligible_512_admission_decisions'],p1['eligible_512_admission_decisions']);self.assertEqual(p1['eligible_512_admission_decisions']['denials'],0)
        self.assertGreater(p1['admission_decisions']['denials'],0) # Oversized full prompts alone prove nothing about admission policy.
        self.assertEqual(r['ablations']['P1']['status'],'NOT_DISCRIMINATED_BY_WORKLOAD');self.assertFalse(r['ablations']['P1']['mechanism_exercised'])
    def test_ttl_capacity_victims_and_lost_opportunities(self):
        r,_=self.report(self.plan());p0,p2=r['policy_summary']['P0'],r['policy_summary']['P2']
        self.assertGreater(p2['TTL_expirations'],0);self.assertGreater(p0['capacity_evictions'],0)
        self.assertGreater(p2['exact_prefix_opportunities']['lost'],0);self.assertEqual(p0['exact_prefix_opportunities']['lost'],0)
        self.assertTrue(r['ablations']['P2']['mechanism_exercised']);self.assertTrue(r['ablations']['P3']['mechanism_exercised'])
        child=r['children']['m_t0_r0_P3'];self.assertTrue(all(x['source_request_id'] and x['bytes']==240*2**20 for x in child['capacity_eviction_victims']))
    def test_pairwise_and_target_only_traffic_separate(self):
        r,_=self.report(self.plan());self.assertEqual(len(r['pairwise']),5*15)
        self.assertTrue(all(x['status']=='NOT_COMPARABLE_DIFFERENT_SOURCE_TRAFFIC' for x in r['pairwise'] if x['policy']=='P5'))
        self.assertEqual(r['physical_HIT_status'],'NOT_EVALUATED');self.assertEqual(r['scope'],'STATIC_ONLY')
    def test_exploration_cost_denial_reported_without_retuning(self):
        p=self.plan(parameters(0.,9999.));before=copy.deepcopy(p['policy_parameters']);r,_=self.report(p)
        a=r['policy_summary']['P1']['eligible_512_admission_decisions'];self.assertGreater(a['denials'],0);self.assertGreater(a['exploration_admissions'],0)
        self.assertTrue(r['ablations']['P1']['mechanism_exercised']);self.assertEqual(p['policy_parameters'],before);self.assertEqual(r['parameters'],before)
    def test_scores_or_unused_exploration_flags_do_not_imply_difference(self):
        p=self.plan();r,rs=self.report(p)
        for j in p['jobs']:
            if j['policy']=='P1':
                for row in rs[j['child_id']]:
                    if row['admission_decision'] is not None:row['admission_decision']['admission_net_ms']+=1
        changed=discrimination(p,rs);self.assertEqual(changed['ablations']['P1']['status'],'NOT_DISCRIMINATED_BY_WORKLOAD')
    def test_missing_policy_or_bad_coverage_rejected(self):
        p=self.plan();_,rs=self.report(p);bad=copy.deepcopy(p);bad['jobs']=bad['jobs'][1:]
        with self.assertRaises(ValueError):discrimination(bad,rs)
        rs[p['jobs'][0]['child_id']].pop()
        with self.assertRaises(ValueError):discrimination(p,rs)
    def test_sentinels_do_not_discriminate_measurement_ablations(self):
        p=self.plan();r,rs=self.report(p)
        for j in p['jobs']:
            if j['policy']=='P1' and j['block']['segment']=='STRICT_SENTINEL':rs[j['child_id']][0]['admitted']=None
        changed=discrimination(p,rs);self.assertEqual(changed['ablations'],r['ablations']);self.assertEqual(changed['policy_summary'],r['policy_summary'])
    def test_deterministic_report_and_complete_source_denominator(self):
        p=self.plan();r,rs=self.report(p);self.assertEqual(r,discrimination(p,rs))
        self.assertEqual(r['policy_summary']['P0']['exact_prefix_opportunities'],{'requests':32,'retained':32,'lost':0})
        self.assertEqual(r['policy_summary']['P1']['eligible_512_admission_decisions']['candidates'],16)

class CPUReportTests(unittest.TestCase):
    def plan(self):
        DiscriminationTests.setUpClass();p=DiscriminationTests().plan()
        p.update(plan_id='CPU_FIXTURE',source={'commit':'CPU_FIXTURE'},multi_prepared='/fixture_adapters',datasets={})
        p['resources']=resource_plan(p['jobs'],dict(prefill_seconds_per_token=.001,decode_seconds_per_token=.01,admission_seconds=2,other_request_seconds=.1))
        p.update(trace_checks=[],transport={'maximum_header_bytes':12345})
        return p
    def test_cpu_audit_inspect_and_tamper_recomputation(self):
        from semcache.edgelora import r5_i5r4 as api
        p=self.plan();tests={'run':1,'passed':1,'failed':0,'skipped':0}
        with tempfile.TemporaryDirectory() as d,patch.object(api,'load',return_value=(p,{'TEST_FIXTURE_ONLY':True})),patch.object(api,'invariant_tests',return_value=tests):
            root=Path(d);r=api.cpu_validate(root/'plan',root/'cpu');self.assertEqual(r['policy_discrimination']['scope'],'STATIC_ONLY')
            result=api.inspect(root/'plan',root/'cpu','CPU_FIXTURE',require_cpu_validation=True)
            self.assertEqual(result['policy_discrimination'],r['policy_discrimination']);self.assertEqual(result['resources']['required_free_disk_bytes'],512*2**30)
            report=json.loads((root/'cpu/i4_policy_discrimination.json').read_text());report['ablations']['P1']['status']='DISCRIMINATED_BY_WORKLOAD'
            write(root/'cpu/i4_policy_discrimination.json',report);seal(root/'cpu')
            with self.assertRaisesRegex(ValueError,'discrimination changed'):api.inspect(root/'plan',root/'cpu','CPU_FIXTURE',require_cpu_validation=True)
    def test_two_by_two_budget_and_timeout_totals(self):
        p=self.plan();r=p['resources'];self.assertEqual(len(p['jobs']),30);self.assertEqual(r['evidence_budget_bytes'],480*2**30);self.assertEqual(r['required_free_disk_bytes'],512*2**30)
        self.assertAlmostEqual(r['estimated_CPU_verification_seconds'],2878.99,places=1);self.assertLess(r['estimated_CPU_verification_seconds'],3600)
        self.assertEqual(r['GPU_phase_limit_seconds'],68400);self.assertEqual(r['allocation_seconds'],85800);self.assertEqual(r['allocation_headroom_seconds'],750)
        self.assertEqual(sum(r[k] for k in ('GPU_phase_limit_seconds','analysis_limit_seconds','verification_limit_seconds','report_limit_seconds','inspect_limit_seconds','timeout_kill_after_reserve_seconds','allocation_headroom_seconds')),r['allocation_seconds'])
        script=(Path(__file__).resolve().parents[1]/'jobs/r5_i5r4_integrated.sh').read_text()
        limits=re.findall(r'timeout --signal=TERM --kill-after=(\d+)s (\d+)([mh])',script)
        self.assertEqual(len(limits),5);self.assertEqual(sum(int(k) for k,n,u in limits),r['timeout_kill_after_reserve_seconds'])
        self.assertEqual(sum(int(n)*{'m':60,'h':3600}[u]+int(k) for k,n,u in limits)+r['allocation_headroom_seconds'],r['allocation_seconds'])

class NativeCPU(unittest.TestCase):
    @unittest.skipUnless(all(importlib.util.find_spec(n) for n in ('torch','transformers','numpy')),'installed native OPT CPU integration NOT_RUN: dependencies unavailable')
    def test_installed_native_contract(self):
        from semcache.edgelora.r5_i5_preflight import installed_opt_check
        installed_opt_check()

# Numbers supplied by the researcher from archived BE, not locally SHA-verified
# GPU measurements. Used only to reproduce the declared budget arithmetic.
BE_RATE_HANDOFF=dict(prefill_seconds_per_token=.021003049537109377,decode_seconds_per_token=3.932975382,
                     admission_seconds=1.932976289,other_request_seconds=3.073750925)

def budget_templates(units,rotations,foreign=False):
    """Length/count-only CPU templates; NOT genuine dataset or experiment inputs."""
    source=dict(token_ids=[1]*512,policy={'max_new_tokens':1},strict=False,scored=False)
    target=dict(token_ids=[1]*1280,policy={'max_new_tokens':32},strict=False,scored=True)
    traces=[[copy.deepcopy(source) for _ in range(4)]+[copy.deepcopy(target) for _ in range(15+int(foreign))] for _ in range(units)]
    sentinel=[dict(copy.deepcopy(source),strict=True),*[dict(copy.deepcopy(target),strict=True,policy={'max_new_tokens':8}) for _ in range(2)]]
    return schedule(traces,sentinel,rotations,42)

class BudgetDiagnosticsTests(unittest.TestCase):
    def test_archived_rates_reproduce_rejection_and_fallbacks(self):
        expected={(2,2):121228,(2,1):62404,(3,1):91816}
        for (units,rotations),seconds in expected.items():
            r=resource_plan(budget_templates(units,rotations),BE_RATE_HANDOFF,diagnostic_only=True)
            self.assertEqual(r['GPU_phase_seconds'],seconds);self.assertEqual(r['GPU_excess_seconds'],max(0,seconds-68400));self.assertEqual(r['accepted'],(units,rotations)==(2,1))
        r=resource_plan(budget_templates(2,2),BE_RATE_HANDOFF,diagnostic_only=True)
        measured_decode=sum(x['conservative_components_seconds']['decoding'] for x in r['children'] if x['segment']=='MEASUREMENT')
        self.assertGreater(measured_decode,86400);self.assertEqual(r['dominant_contributing_factors'][0]['component'],'decoding')
    def test_optional_foreign_request_bound_still_fits_only_two_by_one(self):
        for profile,seconds in (((2,2),128620),((2,1),66100),((3,1),97360)):
            r=resource_plan(budget_templates(*profile,foreign=True),BE_RATE_HANDOFF,diagnostic_only=True)
            self.assertEqual(r['GPU_phase_seconds'],seconds);self.assertEqual(r['accepted'],profile==(2,1))
        r=resource_plan(budget_templates(2,1,True),BE_RATE_HANDOFF)
        self.assertEqual(r['GPU_headroom_seconds'],2300);self.assertEqual(r['required_free_disk_bytes'],320*2**30)
    def test_all_components_and_policy_totals_reconcile(self):
        r=resource_plan(budget_templates(2,1),BE_RATE_HANDOFF)
        self.assertAlmostEqual(sum(r['components_seconds'].values()),r['GPU_phase_seconds'])
        self.assertEqual(sum(x['duration_seconds'] for x in r['by_policy'].values())+900,r['GPU_phase_seconds']);self.assertEqual(set(r['by_policy']),set(('P0','P1','P2','P3','P4','P5')))
        for child in r['children']:
            self.assertAlmostEqual(sum(child['conservative_components_seconds'].values()),child['duration_seconds'])
            self.assertAlmostEqual(child['safety_factor_contribution_seconds'],sum(child['raw_components_seconds'].values()))
        self.assertEqual(r['safety_factor'],2.);self.assertEqual(r['startup_seconds'],180)
    def test_resource_rejection_has_complete_numerical_receipt(self):
        from semcache.edgelora.r5_i5r4 import ResourceBudgetError
        with self.assertRaises(ResourceBudgetError) as caught:resource_plan(budget_templates(2,2),BE_RATE_HANDOFF)
        r=caught.exception.report;self.assertEqual(r['GPU_phase_seconds'],121228);self.assertEqual(r['GPU_excess_seconds'],52828);self.assertEqual(len(r['children']),30)
        self.assertIn('estimate=121228',str(caught.exception));self.assertIn('bound=68400',str(caught.exception));self.assertFalse(r['accepted'])
    def test_bad_rates_and_safety_rejected(self):
        for rates in (dict(BE_RATE_HANDOFF,decode_seconds_per_token=float('nan')),dict(BE_RATE_HANDOFF,admission_seconds=-1)):
            with self.assertRaises(ValueError):resource_plan(budget_templates(2,1),rates)
        with self.assertRaises(ValueError):resource_plan(budget_templates(2,1),BE_RATE_HANDOFF,safety=1.)
    def test_recommendation_never_resizes_requested_plan(self):
        from semcache.edgelora.r5_i5r4 import compare_budgets
        r=compare_budgets(budget_templates,BE_RATE_HANDOFF,2.,180,2**50,(2,2))
        self.assertFalse(r['accepted']);self.assertEqual(r['recommended_profile'],{'trace_units':2,'rotations':1});self.assertFalse(r['automatic_resize']);self.assertFalse(r['plan_sealed'])
        self.assertEqual(r['requested'],{'trace_units':2,'rotations':2})
        fast=compare_budgets(budget_templates,{k:v*.01 for k,v in BE_RATE_HANDOFF.items()},2.,180,2**50,(2,2))
        self.assertEqual(fast['recommended_profile'],{'trace_units':2,'rotations':2}) # Synthetic unit control, never a new performance claim.
    def test_no_feasible_configuration_is_not_positive(self):
        from semcache.edgelora.r5_i5r4 import compare_budgets
        r=compare_budgets(budget_templates,BE_RATE_HANDOFF,2.,180,1,(2,2))
        self.assertIsNone(r['recommended_profile']);self.assertFalse(r['accepted'])
        for c in r['configurations'].values():self.assertTrue(any(x['metric']=='required_free_disk_bytes' for x in c['failures']))
    def test_inadequate_genuine_groups_not_replaced_by_templates(self):
        from semcache.edgelora.r5_i5r4 import compare_budgets
        def make(units,rotations):
            if units==3:raise ValueError('INSUFFICIENT_DISJOINT_REAL_GROUPS')
            return budget_templates(units,rotations)
        r=compare_budgets(make,BE_RATE_HANDOFF,2.,180,2**50,(2,2));self.assertFalse(r['configurations']['3x1']['accepted']);self.assertNotIn('resources',r['configurations']['3x1'])
    def test_cli_budget_dispatch_and_structured_rejection(self):
        from semcache.edgelora.r5_i5r4 import ResourceBudgetError
        spec=importlib.util.spec_from_file_location('budget_cli',Path(__file__).resolve().parents[1]/'scripts/148_r5_i5r4.py');cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)
        args=['budget']
        for k in ('source-commit','plan-3a','audit-3a','verification-3a','expected-3a-commit','be-experiment','be-run','be-audit','be-verification','nf1-audit'):args+=['--'+k,k]
        with patch('semcache.edgelora.r5_i5r4.budget',return_value={'CPU_FIXTURE_ONLY':True}) as run,contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(cli.main(args),0);self.assertEqual(run.call_args.kwargs['trace_units'],2);self.assertNotIn('output_dir',run.call_args.kwargs)
        report=resource_plan(budget_templates(2,2),BE_RATE_HANDOFF,diagnostic_only=True);stderr=io.StringIO()
        with patch('semcache.edgelora.r5_i5r4.budget',side_effect=ResourceBudgetError(report)),contextlib.redirect_stderr(stderr):self.assertEqual(cli.main(args),2)
        receipt=json.loads(stderr.getvalue());self.assertEqual(receipt['status'],'REJECTED');self.assertEqual(receipt['resource_diagnostics']['GPU_phase_seconds'],121228)

class PrepareBudgetTests(unittest.TestCase):
    def setup_inputs(self,stack):
        from types import SimpleNamespace
        from test_r5_i5r3b import Tokenizer,Transport
        from semcache.edgelora import r5_i5r4 as api
        WorkloadTests.setUpClass();ps=copy.deepcopy(WorkloadTests.prompts);helper=Transport();helper.setUp()
        old=dict(seed=42,registry_path='/CPU_FIXTURE/registry',multi_prepared='/CPU_FIXTURE/adapters')
        result={'qasper':{'samples':[dict(id=p['sample']['id'],route=p['route']) for p in ps]}}
        be=dict(rates=BE_RATE_HANDOFF,calibration_contexts=[],inventories={'CPU_FIXTURE_ONLY':True},historical_source={'recorded_source_commit':'CPU_FIXTURE'},paths={})
        data={'qasper':[p['sample'] for p in ps]};by={p['sample']['id']:p for p in ps}
        patches=[patch.object(api,'reviewed',return_value={'CPU_FIXTURE_ONLY':True}),patch.object(api,'baseline',return_value=be),
                 patch.object(api,'history',return_value=(old,result,{})),patch.object(api,'inputs',return_value=(SimpleNamespace(tokenizer=Tokenizer()),WorkloadTests.users,{'qasper':{'path':'/CPU_FIXTURE/qasper'}},data,helper.pm,{'base_prepared':'/CPU_FIXTURE/base','tokenizer':helper.tokenizer})),
                 patch('semcache.edgelora.r5_i5_preflight.readiness',return_value={'status':'READY_FOR_MANUAL_GPU_SMOKE','runtime':{}}),
                 patch.object(api,'prepare_prompt',side_effect=lambda s,*a,**k:copy.deepcopy(by[s['id']])),patch.object(api,'original_link',return_value={'CPU_FIXTURE_ONLY':True}),
                 patch.object(api,'validate',return_value=({},helper.pm,helper.specs,WorkloadTests.users)),patch.object(api,'transport',return_value={}),
                 patch.object(api.shutil,'disk_usage',return_value=SimpleNamespace(free=2**50))]
        for item in patches:stack.enter_context(item)
        return api
    def args(self):
        return dict(source_commit='CPU_FIXTURE',plan_3a='/p',audit_3a='/a',verification_3a='/v',expected_3a_commit='CPU_FIXTURE',be_experiment='/be',be_run='/br',be_audit='/ba',be_verification='/bv',nf1_audit='/nf',trace_units=2,rotations=2)
    def test_failed_prepare_never_creates_or_seals_plan(self):
        with contextlib.ExitStack() as stack,tempfile.TemporaryDirectory() as d:
            api=self.setup_inputs(stack)
            with patch.object(api,'output') as out,patch.object(api,'seal') as sealed,self.assertRaises(api.ResourceBudgetError) as caught:api.prepare(output_dir=str(Path(d)/'plan'),**self.args())
            out.assert_not_called();sealed.assert_not_called();self.assertFalse((Path(d)/'plan').exists());self.assertEqual(caught.exception.report['recommended_profile'],{'trace_units':2,'rotations':1})
    def test_budget_only_reports_all_profiles_without_plan(self):
        with contextlib.ExitStack() as stack:
            api=self.setup_inputs(stack)
            with patch.object(api,'output') as out,patch.object(api,'seal') as sealed:result=api.budget(**self.args())
            out.assert_not_called();sealed.assert_not_called();self.assertEqual(result['status'],'CPU_BUDGET_DIAGNOSTIC_COMPLETE');self.assertFalse(result['accepted']);self.assertFalse(result['plan_sealed'])
            self.assertEqual(set(result['configurations']),{'2x2','2x1','3x1'})

if __name__=='__main__':unittest.main()
