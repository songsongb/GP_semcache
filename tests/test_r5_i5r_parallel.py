"""CPU fixtures only: no GPU/semantic acceptance or natural-data claims."""
import copy,importlib.util,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from semcache.edgelora.r5_i5r_parallel import audit_prompts,longest_contiguous,token_origins,output,cpu,cpu_verify
from semcache.edgelora.r5_i5r_parallel_gpu import repeated_tasks
from semcache.edgelora.r5_i5r3bbe import economics

class Natural(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from test_r5_i5r4 import WorkloadTests
        WorkloadTests.setUpClass();cls.users=WorkloadTests.users;cls.base=copy.deepcopy(WorkloadTests.prompts[0])
    def prompt(self,ids,n,origin='CONTEXT',aid=None):
        p=copy.deepcopy(self.base);p['sample'].update(id=str(n),group_id=str(n));p['token_ids']=ids;p['origins']=[origin]*len(ids)
        if aid:p['route']['adapter_id']=aid
        return p
    def audit(self,a,b,**kw):return audit_prompts([a,b],self.users,**kw)
    def test_strict_prefix_all_diagnostic_conditions(self):
        r=self.audit(self.prompt([1,2,3],0),self.prompt([1,2,3,4],1));p=r['rows'][0]
        self.assertTrue(p['strict_prefix']);self.assertEqual(p['strict_reuse_fraction'],.75);self.assertEqual(r['strict_prefix_pairs'],1)
    def test_equal_prompts_not_strict(self):
        p=self.audit(self.prompt([1,2,3],0),self.prompt([1,2,3],1))['rows'][0]
        self.assertTrue(p['full_sequence_equal']);self.assertFalse(p['strict_prefix']);self.assertTrue(p['raw_w3_candidate'])
    def test_shifted_middle_and_partial_prefix(self):
        p=self.audit(self.prompt([1,2,3,4],0),self.prompt([9,1,2,3,8],1))['rows'][0]
        self.assertFalse(p['strict_prefix']);self.assertEqual(p['shifted_position_occurrences'],1);self.assertEqual(p['longest_contiguous_tokens'],3)
    def test_repeated_gram_occurrences_without_cartesian_records(self):
        r=self.audit(self.prompt([1]*30,0),self.prompt([1]*40,1));p=r['rows'][0]
        self.assertEqual(p['matching_occurrence_pairs'],28*38);self.assertEqual(p['same_position_occurrences'],28);self.assertEqual(r['requests_with_raw_w3_candidates'],1)
    def test_template_only_excluded(self):
        r=self.audit(self.prompt([1,2,3],0,'TEMPLATE'),self.prompt([1,2,3,4],1,'TEMPLATE'));self.assertEqual(r['non_template_candidate_pairs'],0)
    def test_unknown_is_not_proven_content(self):
        p=self.audit(self.prompt([1,2,3],0,'UNKNOWN'),self.prompt([1,2,3,4],1,'UNKNOWN'))['rows'][0]
        self.assertEqual(p['proven_content_occurrences'],0);self.assertGreater(p['non_template_only_occurrences'],0)
    def test_adapter_isolation(self):
        p=self.audit(self.prompt([1,2,3],0),self.prompt([1,2,3,4],1,aid='Adapter_C'))['rows'][0]
        self.assertFalse(p['authorized']);self.assertIn('authorized',p['rejections'])
    def test_authorized_cross_user(self):
        a,b=self.prompt([1,2,3],0),self.prompt([1,2,3,4],1)
        a['route']=dict(user_id='User_01',adapter_id='Adapter_A');b['route']=dict(user_id='User_02',adapter_id='Adapter_A')
        self.assertTrue(self.audit(a,b)['rows'][0]['authorized'])
    def test_offsets_keep_question_and_omission_marker_separate(self):
        from semcache.edgelora.r5_i5_workload import parts
        class Char:
            def __call__(self,text,**kw):return dict(input_ids=list(map(ord,text)),offset_mapping=[(i,i+1) for i in range(len(text))])
        sample=dict(dataset='qasper',question='UniqueQuestion',context='ActualContext')
        before,after=parts(sample);context='Actual\n[...context omitted...]\nContext';text=before+context+after
        labels=token_origins(dict(sample=sample,input_text=text,token_ids=list(map(ord,text)),truncated=True),Char())
        self.assertEqual(labels[text.index('Actual')],'CONTEXT');self.assertEqual(labels[text.index('omitted')],'TEMPLATE');self.assertEqual(labels[text.index('UniqueQuestion')],'QUESTION')
    def test_cache_reset_vs_sequential(self):
        p=self.audit(self.prompt([1,2,3],0),self.prompt([1,2,3,4],1),block_size=1)['rows'][0]
        self.assertFalse(p['hypothetical_short_resident']);self.assertTrue(p['hypothetical_sequential_resident'])
    def test_capacity_eviction_and_oversized_source(self):
        ps=[self.prompt([n]*512,n) for n in range(3)];ps.append(self.prompt([0]*513,3))
        r=audit_prompts(ps,self.users);self.assertGreaterEqual(r['hypothetical_LRU']['sequential_evictions'],1)
        p=self.audit(self.prompt([1]*1100,0),self.prompt([1]*1101,1))['rows'][0];self.assertFalse(p['source_capacity'])
    def test_truncation_changes_prefix_and_independent_groups(self):
        a=self.prompt([1,2,3],0);b=self.prompt([1,2,3,4],1);r=self.audit(a,b);b['token_ids']=[1,2,9];b['origins']=['CONTEXT']*3
        self.assertEqual(r['independent_original_contexts'],1);self.assertFalse(self.audit(a,b)['rows'][0]['strict_prefix'])
    def test_no_match_and_duplicate_rejection(self):
        self.assertEqual(self.audit(self.prompt([1,2,3],0),self.prompt([4,5,6],1))['raw_w3_candidate_pairs'],0)
        with self.assertRaises(ValueError):self.audit(self.prompt([1],0),self.prompt([1,2],0))
    def test_contiguous_automaton_and_determinism(self):
        self.assertEqual(longest_contiguous([1,2,1,2,3],[0,1,2,3,5]),3)
        a,b=self.prompt([1,2,3],0),self.prompt([1,2,3,4],1);self.assertEqual(self.audit(a,b),self.audit(a,b))
    def test_cpu_seal_recompute_and_tamper(self):
        from semcache.edgelora.r5_i5r_parallel import cpu_compute
        fixture=dict(results={'snips':{'status':'DATASET_NOT_READY'}},GPU_NOT_RUN=True)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);reg=root/'registry.json';reg.write_text('{}')
            with patch('semcache.edgelora.r5_i5r_parallel.cpu_compute',return_value=copy.deepcopy(fixture)),patch('semcache.edgelora.r5_i5r_parallel.source_identity',return_value={'CPU_FIXTURE_ONLY':True}):
                cpu(reg,root/'multi',root/'audit');cpu_verify(root/'audit',root/'verification')
                with self.assertRaises(ValueError):cpu_verify(root/'audit',root/'audit/nested_verification')
                (root/'audit/experiment2_summary.csv').write_text('corrupt')
                with self.assertRaises(ValueError):cpu_verify(root/'audit',root/'bad')
    def test_five_unavailable_datasets_are_never_fixtures(self):
        from types import SimpleNamespace
        from semcache.edgelora.r5_i5r_parallel import cpu_compute
        with patch('semcache.edgelora.r5_i5r_parallel.assets',return_value=({},None,{'identity':'fixture'},None,None,SimpleNamespace(identity='fixture'))),patch('semcache.edgelora.r5_i5r_parallel.Evidence'):
            r=cpu_compute('unused','unused');self.assertEqual(len(r['results']),5)
            self.assertTrue(all(x['status']=='DATASET_NOT_READY' for x in r['results'].values()))

class Controlled(unittest.TestCase):
    def tasks(self):
        base=dict(sample_id='original',group_id='original_group',user_id='User_01',adapter_id='Adapter_A',input_text='real fixture',policy={'max_new_tokens':8},prepared={'reference':'unchanged'})
        s=dict(copy.deepcopy(base),request_id='S',scored=False,token_ids=list(range(512)),role='PRIMING_SOURCE')
        t=dict(copy.deepcopy(base),request_id='T',scored=True,token_ids=list(range(600)),role='SCORED_TARGET')
        n=dict(copy.deepcopy(base),request_id='N',scored=True,token_ids=[900]*600,role='NEGATIVE_TARGET');return s,t,n
    def test_unique_request_repeated_sample_identity_and_immutable_inputs(self):
        s,t,n=self.tasks();before=copy.deepcopy((s,t,n));ts=repeated_tasks(s,t,n,3)
        self.assertEqual(len({x['request_id'] for x in ts}),5);self.assertEqual([x['request_sequence_number'] for x in ts],list(range(5)))
        self.assertTrue(all(x['sample_id']=='original' and x['token_ids']==t['token_ids'] for x in ts[1:-1]));self.assertEqual(before,(s,t,n));self.assertFalse(any('forced_hit' in x for x in ts))
    def test_bad_prefix_negative_adapter_and_repeat_rejected(self):
        for kind in ('prefix','negative','adapter','positive_adapter','repeat'):
            s,t,n=self.tasks()
            if kind=='prefix':t['token_ids'][0]=999
            if kind=='negative':n['token_ids']=t['token_ids']
            if kind=='adapter':n['adapter_id']='Adapter_B'
            if kind=='positive_adapter':t['adapter_id']='Adapter_B'
            with self.assertRaises(ValueError):repeated_tasks(s,t,n,5 if kind=='repeat' else 3)
    def test_measured_economics_includes_source_once(self):
        def rows(source):
            ts=[];start=0
            ids=([('S',20,False,'PRIMING_SOURCE')]+[(f'T{i}',5,True,'SCORED_TARGET') for i in range(3)]+[('N',11,True,'NEGATIVE_TARGET')]) if source else ([(f'T{i}',10,True,'SCORED_TARGET') for i in range(3)]+[('N',10,True,'NEGATIVE_TARGET')])
            for rid,ms,scored,role in ids:
                ts.append(dict(request_id=rid,sample_id='same_positive' if rid.startswith('T') else rid,user_id='User_01',adapter_id='Adapter_A',status='COMPLETE',arrival_ns=start,completion_ns=start+ms*1000000,application_e2e_ms=ms,model_execution_e2e_ms=8,cache_admission_eviction_ms=2,scored=scored,role=role,source={'CPU_FIXTURE_ONLY':True} if source and rid.startswith('T') else None));start+=ms*1000000
            return ts
        r=economics(rows(True),rows(False));self.assertEqual(r['D_ms'],6);self.assertEqual(r['source_cost_ms'],20);self.assertEqual(r['scored_request_denominator'],4);self.assertEqual(r['actual_physical_hits'],3)
    def test_cli_prepare_and_missing_required_opt_in(self):
        spec=importlib.util.spec_from_file_location('parallel_cli',Path(__file__).resolve().parents[1]/'scripts/149_r5_i5r_parallel.py');cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)
        with patch('semcache.edgelora.r5_i5r_parallel_gpu.prepare',return_value={'CPU_FIXTURE_ONLY':True}) as fn:
            self.assertEqual(cli.main(['prepare','--registry','r','--multi','m','--output-dir','o','--source-commit','s','--dataset','multi_news']),0);self.assertEqual(fn.call_args.kwargs['reuses'],3)
        with self.assertRaises(SystemExit):cli.main(['run','--experiment','e','--source-commit','s','--output-dir','o'])
    def test_real_shell_fails_before_dispatch_without_allocation(self):
        import os,subprocess
        job=Path(__file__).resolve().parents[1]/'jobs/r5_i5r_parallel_pilot.sh'
        result=subprocess.run(['bash',str(job)],env={'PATH':os.environ['PATH']},capture_output=True,text=True)
        self.assertNotEqual(result.returncode,0);self.assertIn('manual allocation required',result.stderr)

if __name__=='__main__':unittest.main()
