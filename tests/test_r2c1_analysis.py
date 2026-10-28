import unittest
from semcache.diagnostics.r2c1_analysis import summarize,aggregate,hypotheses
from semcache.diagnostics.r2c1_workloads import MODES


def session(run,delta=2,strict=False,count=20):
    m=dict(experiment='R2-C1',execution_status='COMPLETE',benchmark_config=dict(phase='latency',independent_run_id=run,
        measured_runs=count,model='facebook/opt-2.7b',dtype='float16'),process_instance=run,environment_identity='same',hardware_identity='samegpu',
        environment_verified=True,GPU_VALIDATION='EXECUTED',settings={'head':'native_all_token_logits'},
        conditions={'L512_covmax':dict(status='VALID',length=512,requested_coverage='max',condition_identity='same',strict_pass=strict)})
    rows=[dict(independent_run_id=run,case_id='L512_covmax',phase='measured',benchmark_phase='latency',instrumentation='minimal',pair_index=i,
        length=512,requested_coverage='max',wall_ms=dict(zip(MODES,(20.,30.,20.-delta))),delta_full_raw_ms=delta,delta_miss_raw_ms=10.+delta,delta_full_miss_ms=-10.,
        condition_identity='same',structural_valid=True,cache_integrity_valid=True,model='facebook/opt-2.7b',dtype='float16',
        endpoint_policy='all_token_logits_use_cache_false_attention_true',coverage={'reused_tokens':510},strict_pass=strict) for i in range(count)]
    return m,rows


class Analysis(unittest.TestCase):
    def test_sign_and_strict_fail_not_qualified(self):
        _,rows=session('one'); s=summarize(rows)[0]
        self.assertEqual(s['differences']['delta_full_miss_ms']['mean'],-10.)
        r=aggregate([session(str(i)) for i in range(3)],draws=100)
        self.assertEqual(r[0]['conclusion'],'REPRODUCED_POSITIVE_DIFFERENCE'); self.assertFalse(r[0]['correctness_qualified_speedup'])
        self.assertFalse(r[0]['production_safe_reuse_claimed'])
        self.assertEqual(hypotheses(r)['H3']['status'],'SUPPORTED')
        negative=aggregate([session('one',delta=-2)],draws=100)
        self.assertEqual(hypotheses(negative)['H3']['result'],'NO_OBSERVED_BREAK_EVEN')

    def test_sessions_not_pooled_and_outliers_preserved(self):
        r=aggregate([session('a',1,count=20),session('b',2,count=20),session('c',-1,count=200)],seed=5,draws=100)[0]
        self.assertAlmostEqual(r['paired_difference']['mean'],2/3); self.assertEqual(r['conclusion'],'POSITIVE_MEAN_ONLY')
        m,rows=session('o'); rows[-1]['wall_ms']['RAW_SEMCACHE_QKV']=100.
        rows[-1].update(delta_full_raw_ms=-80.,delta_miss_raw_ms=-70.)
        s=summarize(rows)[0]; self.assertEqual(s['count'],20); self.assertFalse(s['outliers_removed'])
        self.assertLess(s['differences']['delta_full_raw_ms']['mean'],0.)

    def test_reject_cross_harness_profile_duplicates_and_gpu_mismatch(self):
        a=session('a'); b=session('b')
        with self.assertRaises(ValueError): aggregate([a,a])
        b[0]['hardware_identity']='other'
        with self.assertRaises(ValueError): aggregate([a,b])
        a=session('a'); a[0]['experiment']='R2-B3.5'
        with self.assertRaises(ValueError): aggregate([a])
        a=session('a'); a[1][0]['instrumentation']='profile'
        with self.assertRaises(ValueError): aggregate([a])
        with self.assertRaises(ValueError): aggregate([(session('a')[0],session('a')[1][:-1])])


if __name__=='__main__': unittest.main()
