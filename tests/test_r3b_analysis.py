import copy
import unittest
from semcache.edgelora.communication import ProtocolError
from semcache.edgelora.r3b_analysis import MODES,flags,summarize,validate_row


def row(mode,repeat=0,wall=10.):
    hit=mode=='WARM_HIT'
    return dict(request_id=f'{mode}_{repeat}',stage='B3',phase='measured',repeat=repeat,mode=mode,
        observed_qkv_outputs=96,reused_qkv_outputs=96 if hit else 0,
        cache_status='HIT' if hit else 'MISS' if mode=='COLD_MISS' else 'BYPASS',
        hidden_frames=0 if hit else 32,delta_frames=0 if hit else 32,
        hidden_payload_bytes=0 if hit else 100,delta_payload_bytes=0 if hit else 600,
        application_wire_bytes=0 if hit else 900,ud_cpu_lora_ms=0 if hit else 1.,cache_bitwise_passed=True,
        request_wall_ms=wall,cache_lookup_ms=1.,cache_insert_ms=1.,cache_bytes_before=0,cache_admitted=mode=='COLD_MISS',
        native_parity=dict(passed=False),**flags(True))


class AnalysisTests(unittest.TestCase):
    def test_native_failure_is_separate_from_cache_pass_and_statistics_keep_negative_results(self):
        rows=[row(mode,i,wall=20. if mode=='WARM_HIT' else 10.+i) for mode in MODES for i in range(3)]
        result=summarize(rows,3,32)
        self.assertEqual(result['by_mode']['WARM_HIT']['median_request_wall_ms'],20.)
        self.assertEqual(result['by_mode']['NO_REUSE']['p90_request_wall_ms'],11.8)
        self.assertEqual(result['by_mode']['WARM_HIT']['samples'],3)
        self.assertEqual(result['by_mode']['NO_REUSE']['native_strict_passes'],0)
        self.assertFalse(flags()['correctness_qualified_reuse_speedup'])
        self.assertFalse(flags()['semantic_generalization_validated'])

    def test_hit_frames_ud_work_partial_reuse_and_bitwise_mismatch_rejected(self):
        for field,value in (('hidden_frames',1),('ud_cpu_lora_ms',.1),('reused_qkv_outputs',95),
                            ('cache_bitwise_passed',False),('observed_qkv_outputs',95),('application_wire_bytes',1)):
            changed=row('WARM_HIT');changed[field]=value
            with self.subTest(field=field),self.assertRaises(ProtocolError):validate_row(changed,32)

    def test_cold_reset_admission_no_reuse_messages_and_nonfinite_timing_rejected(self):
        for mode,field,value in (('COLD_MISS','cache_bytes_before',1),('COLD_MISS','cache_admitted',False),
            ('NO_REUSE','delta_frames',31),('NO_REUSE','request_wall_ms',float('nan'))):
            changed=row(mode);changed[field]=value
            with self.subTest(field=field),self.assertRaises(ProtocolError):validate_row(changed,32)

    def test_missing_duplicate_and_warmup_imputation_rejected(self):
        rows=[row(mode,i) for mode in MODES for i in range(2)]
        with self.assertRaises(ProtocolError):summarize(rows[:-1],2,32)
        rows[1]['request_id']=rows[0]['request_id']
        with self.assertRaises(ProtocolError):summarize(rows,2,32)
        rows[1]['request_id']='fixed';rows[1]['phase']='warmup'
        with self.assertRaises(ProtocolError):summarize(rows,2,32)

    def test_b1_or_b2_failure_never_executes_b3(self):
        from tempfile import TemporaryDirectory
        from pathlib import Path
        from types import SimpleNamespace
        from unittest.mock import patch
        from semcache.edgelora.r3b_runtime import Experiment
        for failure_call in (2,3):
            with self.subTest(failure_call=failure_call),TemporaryDirectory() as directory:
                cache=SimpleNamespace(context={},bytes=0)
                experiment=Experiment({'identity':{'layers':32}}, {},cache,None,Path(directory),fixture_base={})
                calls=[]
                def request(mode,stage,phase,repeat):
                    calls.append((stage,mode))
                    if len(calls)==failure_call:raise ValueError('TEST_FIXTURE_ONLY gate failure')
                    return dict(reused_qkv_outputs=96)
                with patch.object(experiment,'request',side_effect=request),self.assertRaises(ValueError):experiment.run(1,2)
                self.assertFalse(any(stage=='B3' for stage,mode in calls))
                self.assertEqual(experiment.stages['B3']['status'],'NOT_RUN')


if __name__=='__main__':unittest.main()
