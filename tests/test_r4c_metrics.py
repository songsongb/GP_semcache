import importlib.util
import math
import unittest

from semcache.edgelora.r4c_metrics import qa,gold_score,diagnostic_rouge_l,numerical


class MetricTests(unittest.TestCase):
    def test_official_qa_normalization_multigold_and_empty_f1(self):
        self.assertEqual(qa('The Cat, an animal!','cat animal'),dict(f1=1.,exact_match=1.))
        self.assertAlmostEqual(qa('red blue','red green')['f1'],.5)
        self.assertEqual(qa('','')['f1'],0.)
        score=gold_score('qasper','THE Answer.',['wrong','answer']);self.assertEqual(score['score'],1.)
        self.assertEqual(score['exact_match'],1.)

    def test_gold_quality_is_independent_of_output_fidelity(self):
        same='irrelevant';self.assertEqual(same,same)
        self.assertEqual(gold_score('2wikimqa',same,['correct'])['score'],0.)
        with self.assertRaisesRegex(ValueError,'MISSING_GOLD'):gold_score('qasper','x',[])

    def test_summary_fixture_diagnostic_is_explicit(self):
        self.assertAlmostEqual(diagnostic_rouge_l('red blue','red green'),.5)
        self.assertEqual(diagnostic_rouge_l('','x'),0.)
        result=gold_score('multi_news','red blue',['red blue'],fixture=True)
        self.assertGreater(result['score'],.99)
        if not importlib.util.find_spec('rouge'):
            self.assertFalse(result['official_compatible'])
            with self.assertRaisesRegex(ValueError,'MISSING_DEPENDENCY'):gold_score('multi_news','x',['x'])

    @unittest.skipUnless(importlib.util.find_spec('rouge'),'official rouge==1.0.1 not installed; no official summary-score claim')
    def test_official_rouge_package(self):
        r=gold_score('multi_news','A cat sat.',['A cat sat.']);self.assertGreater(r['score'],.999)
        self.assertEqual(r['package_version'],'1.0.1');self.assertTrue(r['official_compatible'])

    def test_numerical_equal_and_divergent_histories(self):
        ref=[[0.,1.],[2.,0.],[0.,1.]];hit=[[0.,1.0001],[0.,2.],[1.,0.]]
        d=numerical(ref,hit,[1,0,1],[1,1,0]);self.assertGreater(d['max_abs'],0.)
        self.assertEqual(len(d['aligned_history_steps']),2);self.assertEqual(d['downstream_divergent_steps'],1)
        self.assertEqual(d['first_divergent_generated_token'],1);self.assertFalse(d['generation_histories_match'])
        e=numerical(ref,ref,[1,0,1],[1,0,1]);self.assertEqual(e['max_abs'],0.);self.assertEqual(e['kl'],0.);self.assertTrue(e['argmax_agreement'])
        with self.assertRaisesRegex(ValueError,'nonfinite'):numerical([[math.nan]],[[0.]],[0],[0])

    def test_independent_decode_reconstruction_preserves_head_major_layout(self):
        from semcache.edgelora.r3a_artifacts import pack,unpack
        from semcache.edgelora.r4c_analysis import append_head_bytes
        previous=pack([1.,2.,3.,4.,11.,12.,13.,14.],'float32');new=pack([5.,6.,15.,16.],'float32')
        result=append_head_bytes(previous,new,heads=2,old_length=2,head_dim=2,element_bytes=4)
        self.assertEqual(unpack(result,'float32'),[1.,2.,3.,4.,5.,6.,11.,12.,13.,14.,15.,16.])
