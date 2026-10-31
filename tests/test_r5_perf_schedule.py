from copy import deepcopy
import unittest

from semcache.edgelora.r5_perf_schedule import schedule,validate_schedule,TINYBERT_REVISION
from semcache.edgelora.r5_perf_analysis import bootstrap,trend,flags


class ScheduleTests(unittest.TestCase):
    def test_abba_baab_reproducibility_and_internal_request_order(self):
        a=schedule(2,42);self.assertEqual(a,schedule(2,42));validate_schedule(a)
        self.assertEqual(''.join(x['condition'] for x in a['blocks']),'ABBABAAB')
        self.assertEqual(''.join(x['condition'] for x in schedule(1,42,'BAAB')['blocks']),'BAAB')
        self.assertTrue(all(x['request_order']==list(range(10)) for x in a['blocks']))

    def test_manifest_corruption_and_invalid_repetitions(self):
        a=deepcopy(schedule());a['blocks'][0]['condition']='B'
        with self.assertRaises(ValueError):validate_schedule(a)
        for n in (0,17,-1):
            with self.assertRaises(ValueError):schedule(n)

    def test_bootstrap_uses_repetitions_and_is_deterministic(self):
        self.assertIsNone(bootstrap([1.])['ci95'])
        a=bootstrap([-1.,-2.,-3.],42);self.assertEqual(a,bootstrap([-1.,-2.,-3.],42))
        self.assertEqual(a['n'],3);self.assertEqual(a['mean'],-2.);self.assertEqual(a['standard_deviation'],1.)
        self.assertLess(a['ci95'][1],0);self.assertEqual(a['status'],'EXPLORATORY_CLUSTER_BOOTSTRAP')

    def test_descriptive_trend_and_unqualified_defaults(self):
        self.assertEqual(trend([(1,2),(2,4),(3,6)])['slope'],2.)
        self.assertTrue(all(v is False for v in flags().values()));self.assertEqual(len(TINYBERT_REVISION),40)
