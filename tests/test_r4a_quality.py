"""Corpus alignment and no-gold/missing-dependency semantics; real BLEU optional."""
from copy import deepcopy
import importlib.util
import unittest
from unittest.mock import patch

from semcache.edgelora.communication import ProtocolError
from semcache.edgelora.r4a_quality import BLEU,aligned_fidelity,corpus,evaluate


def records(text='the quick brown fox jumped over the quiet fence'):
    return [dict(comparison_id='sample_1',sample_id='sample_1',prompt_sha256='p',policy=dict(mode='fixed'),
        generated_text=text,generated_token_ids=list(range(9)))]


class QualityTests(unittest.TestCase):
    def test_missing_package_is_not_zero_or_perfect_score(self):
        with patch('semcache.edgelora.r4a_quality.compute_bleu',side_effect=RuntimeError('missing sacrebleu')):
            result=aligned_fidelity(records(),records());self.assertEqual(result['corpus_bleu']['status'],'MISSING_DEPENDENCY')
            self.assertIsNone(result['corpus_bleu']['score']);self.assertEqual(result['exact_generated_token_match_fraction'],1.)
    def test_duplicate_wrong_prompt_policy_or_ids_fail_closed(self):
        for field,value in (('comparison_id','unknown'),('sample_id','other'),('prompt_sha256','different'),('policy',dict(mode='natural-eos'))):
            altered=records();altered[0][field]=value
            with self.subTest(field=field),self.assertRaises(ProtocolError):aligned_fidelity(altered,records())
        with self.assertRaises(ProtocolError):aligned_fidelity(records()*2,records()*2)
    def test_no_gold_is_not_model_pseudolabel_and_exact_tokens_not_bleu_alone(self):
        full=dict(records()[0],phase='measured',cell_id='C',mode='NO_REUSE');warm=dict(full,mode='WARM_HIT');cold=dict(full,mode='COLD_MISS')
        with patch('semcache.edgelora.r4a_quality.compute_bleu',side_effect=RuntimeError('not installed')):
            quality,sig=evaluate([full,cold,warm])
        self.assertEqual(quality['bleu_vs_ground_truth']['status'],'NOT_ESTIMABLE_NO_GOLD_REFERENCES')
        self.assertEqual(quality['by_cell']['C']['WARM_HIT']['bleu_vs_native_peft']['status'],'NOT_RUN_NO_MATCHED_NATIVE_GENERATION')
        bad=deepcopy(warm);bad['generated_token_ids'][0]=100
        with patch('semcache.edgelora.r4a_quality.compute_bleu',side_effect=RuntimeError('not installed')):
            result=aligned_fidelity([bad],[full])
        self.assertEqual(result['changed_output_count'],1);self.assertLess(result['token_positional_agreement'],1.)
        bad=deepcopy(warm);bad['generated_text']='different decoded text'
        with patch('semcache.edgelora.r4a_quality.compute_bleu',side_effect=RuntimeError('not installed')):
            result=aligned_fidelity([bad],[full])
        self.assertEqual(result['changed_output_count'],1);self.assertEqual(result['exact_generated_token_match_fraction'],1.)
    def test_invalid_external_gold_is_rejected(self):
        full=dict(records()[0],phase='measured',cell_id='C',mode='NO_REUSE')
        gold=[dict(sample_id='sample_1',prompt_sha256='wrong',reference_text='real externally supplied label',dataset='test fixture',revision='v1')]
        with patch('semcache.edgelora.r4a_quality.compute_bleu',side_effect=RuntimeError('not installed')):
            with self.assertRaises(ProtocolError):evaluate([full,dict(full,mode='COLD_MISS'),dict(full,mode='WARM_HIT')],gold=gold)


@unittest.skipUnless(importlib.util.find_spec('sacrebleu') is not None,'SacreBLEU not installed: real corpus score tests NOT_RUN')
class RealSacreBLEU(unittest.TestCase):
    def test_perfect_aligned_corpus_real_signature(self):
        result=aligned_fidelity(records(),records())['corpus_bleu']
        self.assertAlmostEqual(result['score'],100.);self.assertIn('tok:13a',result['signature']);self.assertEqual(result['n_references'],1)
        self.assertFalse(result['lowercase']);self.assertFalse(result['effective_order']);self.assertEqual(result['smoothing'],'exp')
    def test_intentionally_changed_outputs_lower_real_corpus_bleu(self):
        actual=records('a completely different generated sentence about blue oceans and islands')
        result=aligned_fidelity(actual,records())['corpus_bleu'];self.assertLess(result['score'],100.)


if __name__=='__main__':unittest.main()
