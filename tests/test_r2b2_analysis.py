"""No GPU numbers: deliberately artificial samples test statistics/gates only."""
import unittest
from semcache.diagnostics.r2b2_analysis import summarize,aggregate


def session(run,delta=2,count=20,strict=True):
    m=dict(execution_status='COMPLETE',config=dict(phase='latency',independent_run_id=run,measured_runs=count),
        process_instance=run,comparison_identity='same')
    rows=[dict(independent_run_id=run,case_id='L128_P64',attention_arm='off',endpoint_policy='b1_all_logits',
        phase='measured',pair_index=i,instrumentation='minimal',full_wall_ms=10.,normal_warm_wall_ms=10.-delta,
        full_minus_normal_ms=delta,prepared_wall_ms=7.,normal_minus_prepared_ms=3.-delta,
        correctness_status='STRICT_PASS' if strict else 'STRICT_FAIL') for i in range(count)]
    return m,rows


class StatisticsTests(unittest.TestCase):
    def test_sign_duplicate_instrumentation_and_strict_failure(self):
        m,rows=session('one',strict=False)
        s=summarize(rows)[0]; self.assertEqual(s['paired_difference']['mean'],2)
        self.assertFalse(s['all_strict_pass']); self.assertFalse(s['correctness_qualified_speedup'])
        with self.assertRaises(ValueError): summarize(rows+[rows[0]])
        with self.assertRaises(ValueError): summarize([dict(rows[0],instrumentation='profile')])
        with self.assertRaises(ValueError): summarize([dict(rows[0],full_minus_normal_ms=-2)])
        result=aggregate([session(str(i),strict=False) for i in range(3)],draws=100)[0]
        self.assertTrue(result['LATENCY_REPEATED']); self.assertFalse(result['CORRECTNESS_QUALIFIED_SPEEDUP'])

    def test_independent_session_not_pooled_pairs(self):
        inputs=[session('a',1,20),session('b',2,20),session('c',-1,200)]
        r=aggregate(inputs,seed=13,draws=100)[0]
        self.assertAlmostEqual(r['run_mean_difference']['mean'],2/3)
        self.assertEqual(r['sessions'],3); self.assertFalse(r['all_session_differences_positive'])
        self.assertEqual(r,aggregate(inputs,seed=13,draws=100)[0])
        self.assertFalse(r['CORRECTNESS_QUALIFIED_SPEEDUP'])
        self.assertIsNone(aggregate(inputs[:2],draws=100)[0]['run_bootstrap_95_ci'])

    def test_incomplete_incompatible_duplicate_run_rejected(self):
        a=session('a'); b=session('b')
        with self.assertRaises(ValueError): aggregate([a,a])
        b[0]['comparison_identity']='different'
        with self.assertRaises(ValueError): aggregate([a,b])
        a[0]['execution_status']='FAILED'
        with self.assertRaises(ValueError): aggregate([a])
        with self.assertRaises(ValueError): aggregate([session('c',count=3)])

    def test_no_hit_and_missing_gpu_identity_never_qualify(self):
        inputs=[session(str(i)) for i in range(3)]
        self.assertFalse(aggregate(inputs,draws=100)[0]['CORRECTNESS_QUALIFIED_SPEEDUP'])
        for m,rows in inputs: m['qualification_eligible']=True
        self.assertFalse(aggregate(inputs,draws=100)[0]['CORRECTNESS_QUALIFIED_SPEEDUP'])
        for m,rows in inputs:
            for r in rows: r['cache_hit']=True
        self.assertTrue(aggregate(inputs,draws=100)[0]['CORRECTNESS_QUALIFIED_SPEEDUP'])
        inputs[0][1][0]['correctness_status']='STRICT_FAIL'
        self.assertFalse(aggregate(inputs,draws=100)[0]['CORRECTNESS_QUALIFIED_SPEEDUP'])


if __name__=='__main__': unittest.main()
