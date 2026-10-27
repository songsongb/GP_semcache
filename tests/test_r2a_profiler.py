import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch
from semcache.diagnostics.r2a_profiler import require_profiler,ProfilerUnavailable,operator_rows,trace_rows
from semcache.diagnostics.r2a_analysis import union_us

class ProfilingContracts(unittest.TestCase):
    def test_unavailable(self):
        with patch.dict('sys.modules',{'torch':None}):
            with self.assertRaises(ProfilerUnavailable): require_profiler('cuda:0')
    def test_union(self):
        self.assertEqual(union_us([(0,10),(2,6),(8,12),(20,24)]),16)
        with self.assertRaises(ValueError): union_us([(4,1)])
    def test_self_not_inclusive(self):
        parent=NS(name='r2a::q_base',cpu_parent=None)
        e=NS(name='aten::linear',cpu_parent=parent,device_type='CPU',input_shapes=[[1,128,8]],self_cpu_time_total=100,
            cpu_time_total=900,self_device_time_total=50,self_cpu_memory_usage=2,self_device_memory_usage=4)
        r=operator_rows([e])[0]
        self.assertEqual(r['cpu_self_ms'],.1); self.assertEqual(r['device_self_attributed_ms'],.05)
        self.assertEqual(r['component'],'q_base')
    def test_explicit_correlation_not_kernel_name_guess(self):
        events=[dict(ph='X',cat='user_annotation',name='r2a::q_base',ts=0,dur=20,pid=1,tid=1),
            dict(ph='X',cat='cpu_op',name='aten::mm',ts=1,dur=10,pid=1,tid=1,args={'External id':9}),
            dict(ph='X',cat='kernel',name='opaque_kernel',ts=30,dur=10,args={'External id':9,'stream':7}),
            dict(ph='X',cat='kernel',name='gemm_name_is_not_proof',ts=35,dur=10,args={'stream':8}),
            dict(ph='X',cat='gpu_memcpy',name='copy',ts=50,dur=2,args={})]
        rows,m=trace_rows(dict(traceEvents=events))
        self.assertEqual(rows[0]['component'],'q_base'); self.assertEqual(rows[1]['component'],'unknown_or_unattributed')
        self.assertEqual(m['gpu_activity_union_ms'],.017); self.assertEqual(m['gpu_activity_sum_ms'],.022)
        self.assertEqual(m['kernel_attributed_fraction'],.5); self.assertEqual(m['qkv_kernel_fraction'],.5)
        self.assertIsNone(rows[-1]['memcpy_bytes'])
    def test_empty_not_zero_gpu_evidence(self):
        _,m=trace_rows({}); self.assertIsNone(m['gpu_activity_union_ms'])

try:
    import torch
except ImportError: torch=None

@unittest.skipIf(torch is None,'Torch profiler unavailable locally')
class ActualProfiler(unittest.TestCase):
    def test_cpu_profiler(self):
        p,activities=require_profiler('cpu')
        with p.profile(activities=activities,record_shapes=True) as profile:
            with p.record_function('r2a::q_base'): torch.ones(4,4)@torch.ones(4,4)
        self.assertTrue(operator_rows(profile.events()))
    @unittest.skipUnless(torch is not None and torch.cuda.is_available(),'CUDA unavailable')
    def test_cuda_profiler(self):
        p,activities=require_profiler('cuda:0')
        with p.profile(activities=activities) as profile:
            x=torch.ones(4,4,device='cuda'); x@x; torch.cuda.synchronize()
        self.assertTrue(profile.events())
if __name__=='__main__': unittest.main()
