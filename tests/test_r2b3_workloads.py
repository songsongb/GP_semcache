import unittest
from semcache.diagnostics.r2b3_workloads import cases,validate_case,ordered_conditions


class WorkloadTests(unittest.TestCase):
    def test_groups_and_ratios(self):
        self.assertEqual(cases(group='A'),[(128,64),(256,192),(512,448),(1024,960)])
        self.assertEqual(cases(group='B'),[(128,127),(256,255),(512,511),(1024,1023)])
        self.assertEqual(len(cases(group='ABC')),13)
        for n,p in cases(group='ABC'):
            c=validate_case(n,p); self.assertEqual(c['P']+c['S'],c['L'])
            self.assertEqual(c['cache_hit'],p>0)
            self.assertAlmostEqual(c['prefix_ratio']+c['suffix_ratio'],1)
        self.assertEqual(cases(phase='smoke'),[(32,16),(32,31)])

    def test_bounds_and_no_hit(self):
        for n,p in [(128,128),(128,-1),(0,0),(2048,1)]:
            with self.assertRaises(ValueError): validate_case(n,p)
        with self.assertRaises(ValueError): validate_case(512,448,256)
        with self.assertRaises(ValueError): cases('128:64,128:64')
        self.assertFalse(validate_case(128,0)['cache_hit'])
        self.assertEqual(validate_case(128,0)['S'],128)

    def test_case_order_is_seeded_and_pilot_off_first(self):
        workloads=cases(group='AB')
        a=ordered_conditions(workloads,'off',19,'latency')
        self.assertEqual(a,ordered_conditions(workloads,'off',19,'latency'))
        self.assertNotEqual(a,ordered_conditions(workloads,'off',20,'latency'))
        p=ordered_conditions(workloads,'both',9,'pilot')
        self.assertTrue(all(x[2]=='off' for x in p[:8]))
        self.assertEqual(p[:8],sorted(p[:8]))


if __name__=='__main__': unittest.main()
