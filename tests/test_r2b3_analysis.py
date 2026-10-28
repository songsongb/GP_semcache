import unittest
from semcache.diagnostics.r2b3_analysis import aggregate,classify,experimental_config


def session(run,delta=2.,strict=False,lengths=(128,),count=20):
    conditions={}; rows=[]
    for n in lengths:
        key=f'L{n}_P{n-64}_off'
        conditions[key]=dict(L=n,P=n-64,S=64,attention_arm='off',status='VALID',condition_identity=str(n),
            strict_correctness='STRICT_PASS' if strict else 'STRICT_FAIL')
        for i in range(count):
            rows.append(dict(independent_run_id=run,case_id=f'L{n}_P{n-64}',attention_arm='off',endpoint_policy='b1_all_logits',
                L=n,P=n-64,S=64,model='facebook/opt-2.7b',dtype='float16',backend_effective='eager',
                benchmark_phase='latency',phase='measured',instrumentation='minimal',pair_index=i,
                condition_identity=str(n),structural_reuse_valid=True,cache_integrity_valid=True,
                output_matched=False,materialization_confounded=True,full_wall_ms=10.,normal_warm_wall_ms=10.-delta,
                full_minus_normal_ms=delta,prepared_wall_ms=5.,normal_minus_prepared_ms=5.-delta,
                cache_hit=True,correctness_status='STRICT_PASS' if strict else 'STRICT_FAIL'))
    return dict(experiment='R2-B3',execution_status='COMPLETE',benchmark_config=dict(phase='latency',
        independent_run_id=run,model='facebook/opt-2.7b',dtype='float16',measured_runs=count),process_instance=run,
        hardware_identity='GPU-same',environment_identity='same',environment_verified=True,conditions=conditions),rows


class AnalysisTests(unittest.TestCase):
    def test_labels_strict_fail_endpoint_and_region(self):
        self.assertEqual(classify([-1,-2,-1],[-2,-1],False,True),'NO_OBSERVED_SPEEDUP')
        self.assertEqual(classify([2],None,False,True),'POSITIVE_MEAN_ONLY')
        self.assertEqual(classify([1,-1,2],[-1,2],False,True),'POSITIVE_MEAN_ONLY')
        self.assertEqual(classify([1,2,3],[1,3],False,True),'REPRODUCED_POSITIVE_LATENCY_DIFFERENCE')
        self.assertEqual(classify([1,2,3],[1,3],False,False),'BREAK_EVEN_UNRESOLVED')
        result=aggregate([session(str(i),lengths=(128,256)) for i in range(3)],draws=100)
        self.assertTrue(all(r['conclusion']=='BREAK_EVEN_REGION_CANDIDATE' for r in result))
        self.assertTrue(all(not r['correctness_qualified_speedup'] and r['materialization_confounded'] for r in result))
        result=aggregate([session(str(i),strict=True) for i in range(3)],draws=100)
        self.assertFalse(result[0]['service_speedup_claimed']); self.assertFalse(result[0]['correctness_qualified_speedup'])

    def test_equal_session_weighting_and_outlier_preserved(self):
        sessions=[session('a',1,count=20),session('b',2,count=20),session('c',-1,count=200)]
        r=aggregate(sessions,seed=3,draws=100)[0]
        self.assertAlmostEqual(r['mean_paired_difference_ms'],2/3)
        self.assertEqual(r['conclusion'],'POSITIVE_MEAN_ONLY')
        self.assertEqual(r,aggregate(sessions,seed=3,draws=100)[0])
        m,rows=session('outlier'); rows[-1].update(normal_warm_wall_ms=100.,full_minus_normal_ms=-90.)
        r=aggregate([(m,rows)],draws=100)[0]
        self.assertEqual(r['session_results'][0]['count'],20)
        self.assertLess(r['mean_paired_difference_ms'],0)

    def test_hardware_manifest_gate_and_incomplete_rejection(self):
        a=session('a'); b=session('b'); b[0]['hardware_identity']='other'
        with self.assertRaisesRegex(ValueError,'HARDWARE_MISMATCH'): aggregate([a,b])
        b=session('b'); b[0]['environment_identity']='other'
        with self.assertRaisesRegex(ValueError,'MISMATCH'): aggregate([a,b])
        with self.assertRaises(ValueError): aggregate([a,a])
        with self.assertRaises(ValueError): aggregate([(a[0],a[1][:-1])])
        a[1][0]['cache_integrity_valid']=False
        with self.assertRaises(ValueError): aggregate([a])

    def test_partial_case_is_unresolved_and_analysis_config_from_inputs(self):
        s=session('a'); key=next(iter(s[0]['conditions']))
        s[0]['conditions'][key]['status']='OOM'; s[0]['execution_status']='COMPLETE_WITH_FAILURES'
        result=aggregate([s])[0]
        self.assertEqual(result['conclusion'],'BREAK_EVEN_UNRESOLVED'); self.assertIsNone(result['mean_paired_difference_ms'])
        config=experimental_config([s[0]])
        self.assertEqual(config['model'],['facebook/opt-2.7b']); self.assertEqual(config['dtype'],['float16'])
        self.assertEqual(config['cases'],[(128,64,64)])

    def test_pilot_profile_or_model_case_mismatch_rejected(self):
        for phase in ('pilot','profile','smoke'):
            m,rows=session('one'); m['benchmark_config']['phase']=phase
            with self.assertRaises(ValueError): aggregate([(m,rows)])
        for field,value in [('model','other'),('L',512),('backend_effective','sdpa')]:
            m,rows=session('one'); rows[0][field]=value
            with self.assertRaises(ValueError): aggregate([(m,rows)])

    def test_no_hit_control_never_qualifies(self):
        sessions=[]
        for i in range(3):
            m,rows=session(str(i)); c=m['conditions'].pop('L128_P64_off')
            c.update(P=0,S=128); m['conditions']['L128_P0_off']=c
            for r in rows: r.update(case_id='L128_P0',P=0,S=128,cache_hit=False)
            sessions.append((m,rows))
        r=aggregate(sessions,draws=100)[0]
        self.assertEqual(r['conclusion'],'BREAK_EVEN_UNRESOLVED'); self.assertTrue(r['no_hit_control'])


if __name__=='__main__': unittest.main()
