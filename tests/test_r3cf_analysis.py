"""Synthetic scalar statistics only; never a 151893 replay or GPU finding."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from semcache.edgelora.communication import ProtocolError
from semcache.edgelora.r3a_artifacts import ROOT,sha,write
from semcache.edgelora.r3c_analysis import contrast
from semcache.edgelora.r3cf_schedule import events,flags,make_schedule
from semcache.edgelora.r3cf_analysis import auxiliary_sensitivity,bracket,fixed_count,focused_analysis,mixture_sensitivity
from semcache.edgelora.r3cf_evidence import legacy_reanalysis,verify_sources,load_focused


def synthetic_rows():
    manifest=make_schedule();rows=[]
    for i,e in enumerate(events(manifest)):
        wall=10. if e['mode']=='NO_REUSE' else 12. if e['mode']=='COLD_MISS' else 1.
        rows.append(dict(e,request_id=f'TEST_FIXTURE_ONLY_{i}',analysis_role=e['role'],execution_order=i,
            schedule_sha256=manifest['schedule_sha256'],request_wall_ms=wall,
            cache_status='HIT' if e['mode']=='WARM_HIT' else 'MISS' if e['mode']=='COLD_MISS' else 'BYPASS',
            hidden_frames=0 if e['mode']=='WARM_HIT' else 32,delta_frames=0 if e['mode']=='WARM_HIT' else 32,
            application_wire_bytes=0 if e['mode']=='WARM_HIT' else 1000,hidden_payload_bytes=0 if e['mode']=='WARM_HIT' else 100,
            delta_payload_bytes=0 if e['mode']=='WARM_HIT' else 600,calibration_status='VALID',impairment_status='VALID',
            cache_bitwise_passed=True,native_parity=dict(passed=False),cache_reset_ms_outside_request=.1,
            arm_reset_ms_outside_request=0.,**flags(True)))
    return manifest,rows


class AnalysisTests(unittest.TestCase):
    def test_fixed_counts_signs_percentile_tie_and_determinism(self):
        a=fixed_count([1.,2.],[10.]*14,[10.]*16)
        self.assertEqual(a,fixed_count([1.,2.],[10.]*14,[10.]*16))
        self.assertEqual((a['resampled_hit_count'],a['resampled_miss_count']),(2,14))
        self.assertLess(a['delta_mean_ms'],0);self.assertEqual(a['status'],'CACHE_FASTER')
        self.assertEqual(fixed_count([], [11.]*16,[10.]*16)['status'],'CACHE_SLOWER')
        self.assertEqual(fixed_count([], [10.]*16,[10.]*16)['status'],'UNRESOLVED')
        for bad in ([float('nan')],[-1.],[float('inf')]):
            with self.assertRaises(ProtocolError):fixed_count(bad,[],[1.]*16)
        with self.assertRaises(ProtocolError):fixed_count([1.],[],[1.]*16,resamples=9999)

    def test_fixed_count_removes_random_hit_count_variance_not_hit_latency_uncertainty(self):
        fixed=fixed_count([1.],[12.]*7,[10.]*8)
        unconditional=contrast([1.]+[12.]*7,[10.]*8)
        self.assertEqual(fixed['ci95_low_ms'],fixed['ci95_high_ms'])
        self.assertGreater(unconditional['ci95_high_ms']-unconditional['ci95_low_ms'],0)
        self.assertTrue(fixed['single_hit_degenerate']);self.assertFalse(fixed['uncertainty_adequate_for_diagnostic_bracket'])
        self.assertIn('SINGLE_HIT_SAMPLE_DEGENERACY',fixed['warnings'][0])

    def test_brackets_require_opposite_sign_and_non_degenerate_primary_variance(self):
        slow=fixed_count([],[12.]*16,[10.]*16)
        fast=fixed_count([1.]*4,[12.]*12,[10.]*16)
        self.assertEqual(bracket([slow,fast])['status'],'CONFIRMED_DIAGNOSTIC_BRACKET')
        self.assertEqual(bracket([slow,fast],True)['status'],'RETROSPECTIVE_EXPLORATORY_ONLY')
        one=fixed_count([0.],[10.1]*15,[10.]*16)
        self.assertEqual(one['status'],'CACHE_FASTER');self.assertEqual(bracket([slow,one])['status'],'UNRESOLVED')
        self.assertEqual(bracket([slow])['status'],'UNRESOLVED')

    def test_auxiliary_variability_is_separate_and_never_enlarges_primary(self):
        primary=fixed_count([1.],[12.]*15,[10.]*16)
        support=auxiliary_sensitivity([1.],[12.]*15,[10.]*16,[.5,1,1.5,2,.75,1.25,1.75,2.25])
        self.assertEqual(primary['n_hit'],1);self.assertEqual(primary['delta_mean_ms'],support['primary_delta_mean_ms'])
        self.assertFalse(support['permits_primary_bracket_confirmation']);self.assertTrue(support['primary_point_unchanged'])
        self.assertNotEqual(support['model_assisted_delta_mean_ms'],primary['delta_mean_ms'])

    def test_auxiliary_rows_excluded_from_all_primary_counts_and_means(self):
        manifest,rows=synthetic_rows();points,decision=focused_analysis(rows,manifest,True)
        self.assertEqual(len(points),4);self.assertEqual(decision['primary_measurement_rows'],96)
        self.assertEqual(decision['auxiliary_measurement_rows'],16);self.assertFalse(decision['PRIMARY_PERFORMANCE_ELIGIBLE'])
        altered=deepcopy(rows)
        for row in altered:
            if row['phase']=='aux_measured':row['request_wall_ms']=10000.
        alternate,other=focused_analysis(altered,manifest,True)
        for a,b in zip(points,alternate):
            for field in ('delta_mean_ms','ci95_low_ms','ci95_high_ms','n_hit','n_miss'):self.assertEqual(a[field],b[field])
            self.assertEqual(a['cache']['samples'],16);self.assertGreater(a['controlled_warmup_fill_reset_amortized_delta_ms'],a['delta_mean_ms'])
        bad=deepcopy(rows);bad[-1]['cache_bitwise_passed']=False
        with self.assertRaises(ProtocolError):focused_analysis(bad,manifest,True)
        with self.assertRaises(ProtocolError):focused_analysis(rows[:-1],manifest,True)

    def test_model_assisted_mixture_keeps_out_of_domain_roots_and_missing_data(self):
        wrap=lambda v:[dict(request_wall_ms=x,request_id=f'TEST_FIXTURE_ONLY_{i}') for i,x in enumerate(v)]
        result=mixture_sensitivity(wrap([12.]*8),wrap([1.]*8),wrap([10.]*8))
        self.assertAlmostEqual(result['hit_fraction_point'],2/11)
        self.assertTrue(result['different_arms_combined']);self.assertTrue(result['reused_control_observations'])
        outside=mixture_sensitivity(wrap([12.]*8),wrap([1.]*8),wrap([20.]*8))
        self.assertLess(outside['hit_fraction_point'],0);self.assertFalse(outside['in_unit_interval'])
        self.assertEqual(mixture_sensitivity([],[],[])['status'],'NOT_ESTIMABLE')
        self.assertEqual(mixture_sensitivity(wrap([1.]),wrap([1.]),wrap([2.]))['status'],'NOT_ESTIMABLE')

    def test_missing_legacy_is_not_reconstructed_and_source_tamper_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            result=legacy_reanalysis(Path(tmp)/'missing')
            self.assertEqual(result['execution_status'],'NOT_RUN_MISSING_INPUT');self.assertNotIn('arms',result)
        path=ROOT/'scripts/100_run_r3c_network_one_shot.py';before=sha(path)
        verify_sources({str(path.relative_to(ROOT)):before})
        with self.assertRaises(ProtocolError):verify_sources({str(path.relative_to(ROOT)):'0'*64})
        self.assertEqual(before,sha(path))

    def test_read_only_loader_rejects_claim_promotion_and_missing_required_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp);report=dict(execution_status='COMPLETE',audit=dict(status='PASS'),output_hashes={},**flags(True))
            report['PRIMARY_PERFORMANCE_ELIGIBLE']=True;write(path/'r3cf_results.json',report)
            with self.assertRaisesRegex(ProtocolError,'qualification'):load_focused(path)
            report['PRIMARY_PERFORMANCE_ELIGIBLE']=False;write(path/'r3cf_results.json',report)
            with self.assertRaisesRegex(ProtocolError,'required focused artifact'):load_focused(path)


if __name__=='__main__':unittest.main()
