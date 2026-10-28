import unittest
from semcache.diagnostics.r2b35_executor import MODES
from semcache.diagnostics.r2b35_benchmark import comparisons
from semcache.diagnostics.r2b35_analysis import summarize,aggregate,performance_label


def session(run,strict=False,equivalent=True,matched_delta=3.,count=20):
    condition=dict(L=128,P=127,S=1,attention_arm='off',condition_identity='same',status='VALID',
        namespace={'adapter':'user_a'},source_snapshot_id='source',
        native_endpoint_equivalent=equivalent,full_warm_strict=strict)
    m=dict(experiment='R2-B3.5',execution_status='COMPLETE',benchmark_config=dict(phase='latency',independent_run_id=run,
        measured_runs=count,model='facebook/opt-2.7b',dtype='float16',attention_arm='off'),process_instance=run,environment_identity='same',
        hardware_identity='GPU-same',environment_verified=True,conditions={'L128_P127_off':condition})
    walls=dict(zip(MODES,[20.,10.,12.,12.-matched_delta]))
    rows=[dict(independent_run_id=run,case_id='L128_P127',attention_arm='off',pair_index=i,phase='measured',
        instrumentation='minimal',benchmark_phase='latency',wall_ms=walls,**comparisons({k:dict(wall_ms=v) for k,v in walls.items()},127),
        L=128,P=127,S=1,model='facebook/opt-2.7b',dtype='float16',condition_identity='same',structural_valid=True,
        namespace={'adapter':'user_a'},source_snapshot_id='source',backend='eager',endpoints=['legacy_all_logits','matched_last_hidden_head'],
        cache_integrity_valid=True,output_matching='OUTPUT_MATCHED_VALID',full_warm_strict=strict,native_endpoint_equivalent=equivalent) for i in range(count)]
    return m,rows


class AnalysisTests(unittest.TestCase):
    def test_four_path_deltas_and_fail_not_qualified(self):
        m,rows=session('a'); s=summarize(rows)[0]
        self.assertEqual(s['differences']['delta_matched_ms']['mean'],3.)
        self.assertEqual(s['differences']['delta_difference_ms']['mean'],7.)
        r=aggregate([session(str(i)) for i in range(3)],draws=100)[0]
        self.assertEqual(r['performance'],'MATCHED_REPRODUCED_POSITIVE_DIFFERENCE')
        self.assertFalse(r['correctness_qualified_speedup']); self.assertEqual(r['numerical_correctness'],'FULL_WARM_STRICT_FAIL')
        r=aggregate([session(str(i),strict=True) for i in range(3)],draws=100)[0]
        self.assertTrue(r['correctness_qualified_speedup'])
        r=aggregate([session(str(i),strict=True,equivalent=False) for i in range(3)],draws=100)[0]
        self.assertEqual(r['performance'],'MATCHED_RESULT_UNRESOLVED'); self.assertFalse(r['correctness_qualified_speedup'])

    def test_session_equal_weight_outliers_and_mixed_signs(self):
        sessions=[session('a',matched_delta=1,count=20),session('b',matched_delta=2,count=20),session('c',matched_delta=-1,count=200)]
        r=aggregate(sessions,seed=4,draws=100)[0]
        self.assertAlmostEqual(r['equal_session_differences']['delta_matched_ms']['mean'],2/3)
        self.assertEqual(r['performance'],'MATCHED_POSITIVE_MEAN_ONLY')
        self.assertEqual(r,aggregate(sessions,seed=4,draws=100)[0])
        self.assertEqual(performance_label([-1,-2,-3],[-3,-1],True,True,True),'MATCHED_NO_OBSERVED_SPEEDUP')

    def test_reject_bad_session_gpu_endpoint_and_incomplete_samples(self):
        a=session('a'); b=session('b')
        with self.assertRaises(ValueError): aggregate([a,a])
        b[0]['hardware_identity']='other'
        with self.assertRaises(ValueError): aggregate([a,b])
        with self.assertRaises(ValueError): aggregate([(a[0],a[1][:-1])])
        for field,value in [('condition_identity','other'),('output_matching','OUTPUT_MATCHED_INVALID'),('dtype','float32'),
            ('namespace',{'adapter':'other'}),('endpoints',['posthoc_slice']),('backend','sdpa')]:
            m,rows=session('one'); rows[0][field]=value
            with self.assertRaises(ValueError): aggregate([(m,rows)])
        m,rows=session('one'); rows[0]['instrumentation']='profile'
        with self.assertRaises(ValueError): summarize(rows)


if __name__=='__main__': unittest.main()
