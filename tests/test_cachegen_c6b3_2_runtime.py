import unittest
from semcache.experiments.cachegen.c6b3_2_runtime import MultiwozBackend,traffic_summary
from semcache.experiments.cachegen.c6b2_runtime import Backend

class Tests(unittest.TestCase):
    def test_shared_physical_implementation(self):
        self.assertIs(MultiwozBackend.event_identity,Backend.event_identity)
        self.assertIs(MultiwozBackend.greedy,Backend.greedy)
        self.assertTrue(issubclass(MultiwozBackend,Backend))
    def test_transport_cdf_included_and_raw_separate(self):
        c={f'{prefix}_{r}_delta_bytes':n for r in 'qkv' for prefix,n in [('raw',100),('compressed',25)]}
        c.update({f'{r}_cdf_bytes':5 for r in 'qkv'})
        s=traffic_summary(c,True)
        self.assertEqual(s['transmitted_total_including_cdf_bytes'],90)
        self.assertEqual(s['transport_compression_ratio'],300/90)
        raw=traffic_summary({f'raw_{r}_delta_bytes':100 for r in 'qkv'},False)
        self.assertEqual(raw['transport_compression_ratio'],1)
        self.assertEqual(raw['compressed_total_delta_bytes'],0)

if __name__=='__main__':unittest.main()
