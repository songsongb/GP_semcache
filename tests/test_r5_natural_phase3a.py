"""TEST_FIXTURE_ONLY: no fixture establishes dataset availability or GPU behavior."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import importlib.util,json,os,tempfile,unittest
from unittest.mock import patch
from semcache.edgelora.communication import digest
from semcache.edgelora.r4c_prompts import FixtureTokenizer
from semcache.edgelora.r5_i5_data import normalize
from semcache.edgelora.r5_i5_workload import prepare_prompt,requests,BUDGETS,TEMPLATE
from semcache.edgelora.r5_i5_quality import score
from semcache.edgelora.r5_subsequence_natural_io import output,put,seal,verify
from semcache.edgelora.r5_subsequence_natural_audit import policy_counts_by_request
from semcache.edgelora import r5_natural_phase3a as api
from test_r5_i5 import samples,dataset


def fixture(name,budget=None):
    """Original-schema test fixtures, explicitly isolated from real preparation."""
    tok=FixtureTokenizer()
    if name=='coqa':
        rows=normalize(name,[dict(TEST_FIXTURE_ONLY=True,id='TEST_FIXTURE_ONLY',story='fixture story',
            questions=[dict(turn_id=i,input_text='Q?') for i in range(1,6)],
            answers=[dict(turn_id=i,input_text='A') for i in range(1,6)])],fixture=True)
    elif name=='multiwoz':
        rows=normalize(name,[dict(TEST_FIXTURE_ONLY=True,dialogue_id='TEST_FIXTURE_ONLY',
            turns=[dict(turn_id=str(2*i),speaker='USER',utterance='hotel',
                frames=[dict(service='hotel',state=dict(slot_values={'hotel-pricerange':['cheap']}))]) for i in range(5)])],
            fixture=True,ontology=['hotel-pricerange'])
    else:rows=samples(name,5)
    prompts=[prepare_prompt(s,tok,2048,BUDGETS[name] if budget is None else budget,allow_truncation=True,input_limit=512) for s in rows]
    for p in prompts:p['strict']=True
    tasks=requests(prompts,track='NATURAL',prefix=None,special=dict(bos_token_id=2,eos_token_id=2,pad_token_id=1))
    b=dict(dataset=name,track='NATURAL',repetition=0,priming_requests=0,routing_seed=42,continuity_breaks=[],tasks=tasks)
    b['block_id']=digest(b)
    return dict(blocks=[b],template_version=TEMPLATE,seed=42,special_tokens=dict(bos_token_id=2,eos_token_id=2,pad_token_id=1)),rows


def reseal(w):
    for b in w['blocks']:b.pop('block_id',None);b['block_id']=digest(b)
    return w


class Phase3A(unittest.TestCase):
    def test_five_dataset_original_payload_routing_and_block_ids_preserved(self):
        for name in api.DATASETS:
            w,rows=fixture(name);before=deepcopy(w);b=api.select_block(w,name,0,rows)
            self.assertEqual(b,w['blocks'][0]);self.assertEqual(len(b['tasks']),5)
            b['tasks'][0]['token_ids'][0]+=1;self.assertEqual(w,before)
        with self.assertRaises(ValueError):api.select_block(*[fixture('qasper')[0],'qasper',0,fixture('qasper')[1]])

    def test_first_occurrence_only_and_no_hit_optimized_block_choice(self):
        w,rows=fixture('snips');w['blocks'].append(deepcopy(w['blocks'][0]))
        with self.assertRaisesRegex(ValueError,'first original'):api.select_block(w,'snips',1,rows)
        w['blocks'][0]['repetition']=1;w['blocks'][1]['repetition']=1;reseal(w)
        with self.assertRaisesRegex(ValueError,'MISSING_ORIGINAL'):api.select_block(w,'snips',0,rows)

    def test_no_slicing_combining_priming_or_duplicate_sources(self):
        for mutation in ('short','priming','duplicate','unsealed'):
            w,rows=fixture('snips');tasks=w['blocks'][0]['tasks']
            if mutation=='short':tasks.pop()
            if mutation=='priming':tasks[0]['scored']=False
            if mutation=='duplicate':tasks[1]['request_id']=tasks[0]['request_id']
            if mutation!='unsealed':reseal(w)
            else:tasks[0]['token_ids'][0]+=1
            with self.assertRaises(ValueError):api.select_block(w,'snips',0,rows)

    def test_full_dialogue_groups_and_numeric_turn_order(self):
        for name in ('coqa','multiwoz'):
            w,rows=fixture(name);self.assertEqual(len(api.select_block(w,name,0,rows)['tasks']),5)
            extra=deepcopy(rows[-1]);extra['id']='TEST_FIXTURE_ONLY_EXTRA';extra['turn_id']=20
            with self.assertRaisesRegex(ValueError,'INCOMPLETE'):api.select_block(w,name,0,rows+[extra])
            w['blocks'][0]['tasks'][1:3]=reversed(w['blocks'][0]['tasks'][1:3])
            for n,t in enumerate(w['blocks'][0]['tasks']):t['request_sequence_number']=n
            reseal(w)
            with self.assertRaisesRegex(ValueError,'TURN_ORDER'):api.select_block(w,name,0,rows)

    def test_dialogue_boundary_split_and_budget_change_rejected(self):
        w,rows=fixture('coqa');w['blocks'][0]['continuity_breaks']=[dict(group_id=rows[0]['group_id'])];reseal(w)
        with self.assertRaisesRegex(ValueError,'GROUP_SPLIT'):api.select_block(w,'coqa',0,rows)
        w,rows=fixture('multi_news');w['blocks'][0]['tasks'][0]['policy']['max_new_tokens']=32;reseal(w)
        with self.assertRaisesRegex(ValueError,'generation budget differs'):api.select_block(w,'multi_news',0,rows)

    def test_longbench_smoke_recorded_budgets_preserved_in_selection_and_material(self):
        # Config supplies fixture budgets only. Production uses the sealed tasks.
        smoke=json.loads((api.ROOT/'configs/r5_i5_smoke_budgets.json').read_text())
        self.assertEqual((smoke['2wikimqa'],smoke['multi_news']),(8,32))
        for name in ('2wikimqa','multi_news'):
            with self.subTest(dataset=name),tempfile.TemporaryDirectory() as tmp:
                w,rows=fixture(name,smoke[name]);before=deepcopy(w)
                self.assertEqual(api.select_block(w,name,0,rows),w['blocks'][0])
                tok=FixtureTokenizer()
                for k,v in w['special_tokens'].items():setattr(tok,k,v)
                multi=Path(tmp)/'multi';multi.mkdir()
                from semcache.edgelora.r5_evidence import seal as multi_seal
                multi_seal(multi)
                args=(w,{name:{'prepared_path':'TEST_FIXTURE_ONLY'}},{},{},dict(stage='expanded',base_prepared='TEST_FIXTURE_ONLY'),
                    {},{},{},SimpleNamespace(tokenizer=tok,identity={'TEST_FIXTURE_ONLY':True}))
                network=dict(rows=[dict(request_id=t['request_id'],maximum_header_bytes=12000) for t in w['blocks'][0]['tasks']])
                with patch.object(api,'inputs',return_value=args),patch.object(api,'provenance',return_value=(rows,{})), \
                     patch.object(api,'validate_scorers'),patch.object(api,'transport',return_value=network):
                    fields,rest=api.material('TEST_FIXTURE_ONLY',multi,name,0)
                self.assertEqual(fields['tasks'],before['blocks'][0]['tasks']);self.assertEqual(w,before)
                self.assertEqual(rest[0]['block_id'],before['blocks'][0]['block_id'])
                self.assertLess(fields['resources']['estimated_run_file_count'],500000)
                self.assertEqual(fields['resources']['max_forwards'],5*smoke[name])

    def test_smoke_policy_mismatch_tampering_and_resource_bounds_rejected(self):
        for name,budget in (('2wikimqa',8),('multi_news',32)):
            for mutation in ('policy_budget','prepared_budget','do_sample','mode','special_token','extra_field','zero','over_limit','bool','float','token_tamper'):
                with self.subTest(dataset=name,mutation=mutation):
                    w,rows=fixture(name,budget);t=w['blocks'][0]['tasks'][0]
                    if mutation=='policy_budget':t['policy']['max_new_tokens']+=1
                    elif mutation=='prepared_budget':t['prepared']['max_new_tokens']+=1
                    elif mutation=='do_sample':t['policy']['do_sample']=True
                    elif mutation=='mode':t['policy']['mode']='fixed'
                    elif mutation=='special_token':t['policy']['eos_token_id']+=1
                    elif mutation=='extra_field':t['policy']['temperature']=1
                    elif mutation=='token_tamper':t['token_ids'][0]+=1
                    else:
                        value={'zero':0,'over_limit':97,'bool':True,'float':float(budget)}[mutation]
                        t['prepared']['max_new_tokens']=t['policy']['max_new_tokens']=value
                    # Resealing a fixture isolates consistency checks. A token
                    # mutation without resealing must fail the original hash.
                    if mutation!='token_tamper':reseal(w)
                    with self.assertRaises(ValueError):api.select_block(w,name,0,rows)

    def test_multinews_default96_fixture_file_guard_remains_unchanged(self):
        w,_=fixture('multi_news');before=deepcopy(w)
        with self.assertRaisesRegex(ValueError,'file guard'):api.base.preflight(w['blocks'][0]['tasks'])
        self.assertEqual(w,before)
        w,_=fixture('multiwoz');r=api.base.preflight(w['blocks'][0]['tasks'])
        self.assertEqual(r['estimated_run_file_count'],499392);self.assertEqual(r['maximum_run_files'],500000)

    def test_read_only_discovery_missing_registry_fixture_and_real_scope_rejection(self):
        with tempfile.TemporaryDirectory() as tmp,patch.dict(os.environ,CUDA_VISIBLE_DEVICES=''):
            root=Path(tmp);p=dataset(root,'coqa',1);before={str(f):f.read_bytes() for f in root.rglob('*') if f.is_file()}
            r=api.discover([root],[root/'missing_registry.json'],[root/'missing_plan'],['snips='+str(root/'missing_snips')])
            self.assertEqual(r['GPU_status'],'GPU_NOT_RUN');self.assertEqual(r['registries'][0]['status'],'MISSING')
            self.assertTrue(all(v['status']=='BLOCKED' for v in r['datasets'].values()))
            self.assertEqual(r['datasets']['snips']['declared_prepared_paths'],[str(root/'missing_snips')])
            self.assertTrue(any(str(p)==v['path'] for v in r['rejected_candidates']))
            self.assertEqual(before,{str(f):f.read_bytes() for f in root.rglob('*') if f.is_file()})
            registry=root/'relative_registry.json';put(registry,{'coqa':os.path.relpath(p,api.ROOT)})
            r=api.discover(registries=[registry])
            self.assertEqual(r['datasets']['coqa']['declared_prepared_paths'],[str(p)])

    def test_source_sealed_plan_rejects_fixture_and_tamper(self):
        from semcache.edgelora.r5_i5_evidence import seal as i5seal
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);w,_=fixture('snips');w.update(schema='I5_EXPERIMENT_V1',TEST_FIXTURE_ONLY=True)
            w['experiment_id']=digest(w);put(root/'i5_workload_manifest.json',w);put(root/'i5_dataset_manifest.json',{});i5seal(root)
            with self.assertRaisesRegex(ValueError,'real original'):api.original_plan(root)
            (root/'i5_workload_manifest.json').write_text('{}')
            with self.assertRaises(ValueError):api.original_plan(root)

    def test_phase3a_plan_reconstruction_and_resealed_payload_mutation(self):
        with tempfile.TemporaryDirectory() as tmp,patch.dict(os.environ,CUDA_VISIBLE_DEVICES=''):
            root=Path(tmp);w,_=fixture('snips');fields=dict(original_experiment=str(root/'original'),multi=str(root/'multi'),
                dataset_paths={},dataset='snips',block_index=0,tasks=w['blocks'][0]['tasks'],resources=api.base.preflight(w['blocks'][0]['tasks']))
            impl=dict(source={'commit':'TEST_FIXTURE_ONLY'},files={})
            with patch.object(api,'implementation',return_value=impl),patch.object(api,'material',return_value=(fields,[])):
                api.prepare(root/'original',root/'multi','TEST_FIXTURE_ONLY','snips',0,root/'plan',True)
                p,_,_=api.load(root/'plan');self.assertEqual(p['dataset'],'snips');self.assertEqual(p['conditions'],list(api.base.CONDITIONS))
                for mutation in ('payload','policy'):
                    bad=deepcopy(p)
                    if mutation=='payload':bad['tasks'][0]['input_text']='TEST_FIXTURE_ONLY_MUTATION'
                    else:
                        # Internally consistent policy editing still violates
                        # the binding to the original sealed request.
                        bad['tasks'][0]['policy']['max_new_tokens']+=1
                        bad['tasks'][0]['prepared']['max_new_tokens']+=1
                    bad.pop('plan_id');bad['plan_id']=digest(bad)
                    d=output(root/('bad_'+mutation));put(d/'approx_plan.json',bad);seal(d)
                    with self.assertRaisesRegex(ValueError,'original selected'):api.load(d)

    def test_zero_skip_cpu_approval_and_gpu_optin_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=dict(implementation={'TEST_FIXTURE_ONLY':True});receipt={'TEST_FIXTURE_ONLY':True}
            root=output(Path(tmp)/'audit');put(root/'cpu_validation.json',dict(status='PASS',failed=0,skipped=1,implementation=p['implementation'],plan_inventory=receipt));seal(root)
            with patch.object(api,'load',return_value=(p,receipt,[])):
                with self.assertRaisesRegex(ValueError,'zero-skip'):api.approval('fixture_plan',root)
        with patch.dict(os.environ,{},clear=True),patch.object(api,'approval',side_effect=AssertionError('GPU/approval forbidden')):
            with self.assertRaises(ValueError):api.run(experiment='x',output_dir='y',source_commit='z',cpu_audit='c',enable_gpu_experiment=True,enable_approximate_experiment=True)

    def test_delegates_existing_execution_cpu_and_auditor_with_explicit_profile(self):
        with patch.object(api.base,'run',return_value='TEST_FIXTURE_ONLY') as run:
            self.assertEqual(api.run(experiment='x'),'TEST_FIXTURE_ONLY');profile=run.call_args.kwargs['_profile']
            self.assertEqual(profile.script,api.SCRIPT);self.assertIs(profile.approval,api.approval)
            self.assertEqual(profile.execution(api.base.CONDITIONS[2]),api.base.execution(api.base.CONDITIONS[2]))
        with patch.object(api.base,'cpu_validate') as cpu:
            api.cpu_validate('p','o');self.assertIn('test_r5_natural_phase3a',cpu.call_args.kwargs['_profile'].tests)
        with patch('semcache.edgelora.r5_natural_approx_audit.analyze') as audit:
            api.analyze(experiment='p');self.assertIs(audit.call_args.kwargs['_approval'],api.approval)
        with patch('semcache.edgelora.r5_subsequence_natural_runtime.worker') as worker:
            api.worker('p','o',api.base.CONDITIONS[2],'c');self.assertIs(worker.call_args.kwargs['_profile'].approval,api.approval)

    def test_summary_keeps_numerical_fail_separate_and_means_and_quality(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);w,_=fixture('snips');plan=dict(tasks=w['blocks'][0]['tasks']);totals={}
            for c in api.base.CONDITIONS:
                quality=[]
                for task in plan['tasks']:
                    q=score(task['prepared']['sample'],task['prepared']['sample']['references'][0],TEMPLATE);quality.append(q)
                    put(root/c/(task['request_id']+'.json'),dict(cache_bytes=0,cache_policy_counts=dict(INSERT=0,EVICT=0,CHU=0,PBR=0),model_timings=dict(prefill_ms=2,decode_ms=3)))
                    if c!=api.base.CONDITIONS[0]:put(root/c/(task['request_id']+'_numerical.json'),dict(numerical_tolerance='FAIL',generated_token_ids_equal=True,generated_text_equal=False))
                totals[c]=dict(physical_hit_requests=0,projection_rows_skipped=0,physical_qkv_rows_consumed_by_category=dict(A=0,B=0,C=0,D=0),policy=[],
                    application_e2e_ms=100,application_ttft_sum_ms=25,wire_bytes=123,cache_and_inference_component_inclusive_ms={},
                    component_scope='inclusive, overlapping',quality=quality,task_quality_delta_vs_full=0,delta_e2e_ms_vs_full=0)
            rows=api.summary_rows('snips',plan,dict(conditions=totals),root)
            self.assertEqual(rows[1]['numerical_fidelity'],'FAIL');self.assertEqual(rows[1]['evidence_integrity'],'PASS')
            self.assertEqual(rows[1]['mean_e2e_ms'],20);self.assertEqual(rows[1]['mean_ttft_ms'],5)
            self.assertEqual(rows[1]['quality']['macro_f1'],5/7);self.assertEqual(rows[1]['generated_text_agreements'],0)

    def test_per_request_admission_eviction_chu_counts_have_real_boundaries(self):
        events=[dict(kind='MSTAGE_MATCH',request_id='S'),dict(kind='INSERT'),dict(kind='PHYSICAL_SUBSEQUENCE_COMPLETE'),
            dict(kind='MSTAGE_MATCH',request_id='T'),dict(kind='CHU'),dict(kind='EVICT'),dict(kind='INSERT'),dict(kind='PHYSICAL_SUBSEQUENCE_COMPLETE')]
        counts=policy_counts_by_request(events,['S','T'])
        self.assertEqual(counts['S'],dict(INSERT=1,EVICT=0,CHU=0,PBR=0));self.assertEqual(counts['T']['CHU'],1)
        with self.assertRaises(ValueError):policy_counts_by_request(events,['T','S'])
        with self.assertRaises(ValueError):policy_counts_by_request([dict(kind='CHU')],['S'])
        with self.assertRaises(ValueError):policy_counts_by_request(events[:-1],['S','T'])

    def test_aggregate_requires_all_five_and_independent_verification_binding(self):
        with tempfile.TemporaryDirectory() as tmp,patch.dict(os.environ,CUDA_VISIBLE_DEVICES=''):
            root=Path(tmp);reg=root/'registry.json';put(reg,{'qasper':{}})
            with self.assertRaisesRegex(ValueError,'exactly five'):api.aggregate(reg,root/'out')
            mapping={};plans={}
            for name in api.DATASETS:
                record={k:str(root/name/k) for k in ('experiment','run','cpu_audit','audit','verification')};mapping[name]=record
                run=output(record['run']);put(run/'TEST_FIXTURE_ONLY.json',{});seal(run)
                w,_=fixture(name);p=dict(dataset=name,plan_id=name,tasks=w['blocks'][0]['tasks'],multi='TEST_FIXTURE_ONLY',original_experiment='TEST_FIXTURE_ONLY',dataset_paths={});plans[record['experiment']]=p
                audit=output(record['audit']);totals={}
                for c in api.base.CONDITIONS:
                    quality=[]
                    for task in p['tasks']:
                        sample=task['prepared']['sample'];text='{}' if name=='multiwoz' else sample['references'][0]
                        quality.append(score(sample,text,TEMPLATE))
                        put(audit/c/(task['request_id']+'.json'),dict(cache_bytes=0,cache_policy_counts=dict(INSERT=0,EVICT=0,CHU=0,PBR=0),model_timings=dict(prefill_ms=2,decode_ms=3)))
                        if c!=api.base.CONDITIONS[0]:put(audit/c/(task['request_id']+'_numerical.json'),dict(numerical_tolerance='FAIL',generated_token_ids_equal=False,generated_text_equal=False))
                    totals[c]=dict(physical_hit_requests=0,projection_rows_skipped=0,physical_qkv_rows_consumed_by_category=dict(A=0,B=0,C=0,D=0),policy=[],
                        application_e2e_ms=100,application_ttft_sum_ms=25,wire_bytes=123,cache_and_inference_component_inclusive_ms={},
                        component_scope='inclusive, overlapping',quality=quality,task_quality_delta_vs_full=0,delta_e2e_ms_vs_full=0)
                put(audit/'approx_audit.json',dict(status='EVIDENCE_INTEGRITY_PASS',schema=api.SCHEMA,track='NATURAL',plan_id=name,
                    plan_inventory={},cpu_audit_inventory={},run_inventory=verify(run),conditions=totals));seal(audit)
                v=output(record['verification']);put(v/'approx_verification.json',dict(status='PASS',audit_inventory=verify(audit)));seal(v)
            put(reg,mapping)
            with patch.object(api,'approval',side_effect=lambda e,c:(plans[e],{},[],{})):
                result=api.aggregate(reg,root/'out');self.assertEqual(result['rows'],15);verify(root/'out')
                table=json.loads((root/'out/phase3a_results.json').read_text())['rows']
                self.assertEqual({r['dataset'] for r in table},set(api.DATASETS))
                self.assertTrue(all(r['numerical_fidelity']=='FAIL' for r in table if r['condition']!=api.base.CONDITIONS[0]))
                bad=Path(mapping['coqa']['verification'])/'approx_verification.json';bad.write_text('{}')
                with self.assertRaises(ValueError):api.aggregate(reg,root/'bad_output')


if __name__=='__main__':unittest.main()
