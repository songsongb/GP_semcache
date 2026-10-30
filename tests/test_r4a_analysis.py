"""Pure CPU inference-gate arithmetic; scalar fixtures never imply GPU benefit."""
from copy import deepcopy
import unittest

from semcache.edgelora.communication import ProtocolError
from semcache.edgelora.r4a_analysis import blocked_contrast
from semcache.edgelora.r4a_evidence import make_suite,planned_counts,schedule


def pairs(value=10.,mode='fixed'):
    return [dict(comparison_id=f'{block}:{i}',block=block,policy=dict(mode=mode),generated_tokens=8,ttft_ms=value,e2e_wall_ms=value)
        for block in (101,202) for i in range(4)]


class AnalysisTests(unittest.TestCase):
    def test_two_blocks_signed_contrast_and_no_synthetic_gpu_pass(self):
        r=blocked_contrast(pairs(8.),pairs(10.),'ttft_ms',fixture=True)
        self.assertEqual(r['delta_mean_ms'],-2.);self.assertLess(r['ci95_high_ms'],0.)
        self.assertFalse(r['benefit_observed']);self.assertEqual(r['pairs'],8);self.assertEqual(r['repeat_blocks'],2)
        r=blocked_contrast(pairs(12.),pairs(10.),'e2e_wall_ms',fixture=True);self.assertGreater(r['delta_mean_ms'],0.)
    def test_natural_eos_or_mismatched_lengths_cannot_create_timing_benefit(self):
        r=blocked_contrast(pairs(1.,'natural-eos'),pairs(10.,'natural-eos'),'ttft_ms')
        self.assertFalse(r['timing_comparable']);self.assertFalse(r['benefit_observed'])
        bad=pairs();bad[0]['generated_tokens']=7
        with self.assertRaises(ProtocolError):blocked_contrast(bad,pairs(),'ttft_ms')
    def test_block_uncertainty_spans_zero_without_cherry_picking(self):
        actual=pairs(8.)
        for r in actual:
            if r['block']==202:r['ttft_ms']=12.
        r=blocked_contrast(actual,pairs(),'ttft_ms');self.assertLess(r['ci95_low_ms'],0.);self.assertGreater(r['ci95_high_ms'],0.)
        self.assertFalse(r['benefit_observed'])
    def test_fixture_and_real_planned_schedules_are_bounded_and_cold_precedes_hit(self):
        suite=make_suite(fixture=True);self.assertEqual(planned_counts(suite)['generation_requests'],15)
        for c in suite['cells']:
            tasks=schedule(suite,c)
            for i in range(0,len(tasks),3):
                modes=[t['mode'] for t in tasks[i:i+3]];self.assertLess(modes.index('COLD_MISS'),modes.index('WARM_HIT'))
        class Tokenizer:
            bos_token_id=2;eos_token_id=2;pad_token_id=1;padding_side='left'
            def __call__(self,text,**kwargs):
                n=kwargs['max_length'];salt=len(text);return dict(input_ids=[salt]*n,attention_mask=[1]*n)
        real=make_suite(Tokenizer(),scope='matrix');self.assertEqual(planned_counts(real)['generation_requests'],108)
        self.assertEqual(planned_counts(real)['max_generation_forward_passes'],1296)


if __name__=='__main__':unittest.main()
