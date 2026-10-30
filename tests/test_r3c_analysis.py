"""CPU-only declared schedules and uncertainty gates; synthetic arithmetic only."""
import importlib.util
import json
import tempfile
import types
import sys
from pathlib import Path
import unittest
from unittest.mock import patch

from semcache.edgelora.communication import ProtocolError
from semcache.edgelora.r3a_artifacts import ROOT,sha,write
from semcache.edgelora.r3c_analysis import (analyze,budget,contrast,crossover,profiles,r2d_comparison,schedule,validate_profiles)
from semcache.edgelora.r3c_calibration import validate_calibration

SCRIPT=ROOT/'scripts/100_run_r3c_network_one_shot.py'


class AnalysisTests(unittest.TestCase):
    def test_default_factorial_counts_exact_controlled_hits_and_deterministic_order(self):
        grid=profiles([0,10],[None,200]);tasks=schedule(grid)
        self.assertEqual(len(grid),4);self.assertEqual(len(tasks),24)
        self.assertEqual(tasks,schedule(grid));self.assertNotEqual(tasks,schedule(grid,seed=43))
        self.assertEqual(budget(dict(profiles=grid,schedule=tasks))['model_request_bound'],228)
        for task in tasks:
            if task['hit_fraction'] is not None:
                self.assertEqual(task['measured_modes'].count('WARM_HIT'),int(task['hit_fraction']*8))
                self.assertEqual(task['warmup_modes'],['COLD_MISS'])
        self.assertTrue(all(task['condition']['rtt_ms']==0 and task['condition']['es_to_ud_mbps'] is None for task in tasks[:6]))

    def test_bad_or_unbounded_grids_and_nonintegral_hits_rejected(self):
        for args in (([0,0],[None]),([10],[200]),([0],[None,0]),([0],[None,-1]),([0,float('nan')],[None]),([0],[None,float('inf')])):
            with self.subTest(args=args),self.assertRaises(ProtocolError):profiles(*args)
        for fractions in ([.1],[0,0],[float('nan')]):
            with self.assertRaises(ProtocolError):schedule(profiles([0],[None]),fractions)
        grid=profiles([0,10,20,30],[None,200])
        with self.assertRaises(ProtocolError):budget(dict(profiles=grid,schedule=schedule(grid,n=16,warmup=2)))

    def test_both_signs_tie_and_small_n_uncertainty_no_forced_winner(self):
        faster=contrast([5.]*8,[10.]*8);slower=contrast([15.]*8,[10.]*8)
        self.assertEqual(faster['status'],'CACHE_FASTER');self.assertEqual(slower['status'],'CACHE_SLOWER')
        self.assertEqual(contrast([10.]*8,[10.]*8)['status'],'UNRESOLVED')
        self.assertEqual(contrast([1.,19.]*4,[10.]*8)['status'],'UNRESOLVED')
        self.assertLess(faster['delta_mean_ms'],0)
        with self.assertRaises(ProtocolError):contrast([1.]*7,[2.]*7)

    def test_measured_brackets_only_no_root_and_exploratory_interpolation(self):
        slow=dict(hit_fraction=0.,**contrast([12.]*8,[10.]*8))
        fast=dict(hit_fraction=.5,**contrast([8.]*8,[10.]*8))
        result=crossover([slow,fast],'hit_fraction')
        self.assertEqual(result['status'],'R3C_DIAGNOSTIC_CROSSOVER')
        self.assertEqual(result['brackets'][0]['exploratory_linear_estimate'],.25)
        self.assertEqual(crossover([slow],'hit_fraction')['status'],'ALWAYS_SLOWER_IN_TESTED_DOMAIN')
        self.assertEqual(crossover([fast],'hit_fraction')['status'],'ALWAYS_FASTER_IN_TESTED_DOMAIN')
        self.assertEqual(crossover([],'hit_fraction')['status'],'NOT_ESTIMABLE')
        tied=dict(hit_fraction=1.,**contrast([10.]*8,[10.]*8))
        self.assertEqual(crossover([slow,tied],'hit_fraction')['status'],'UNRESOLVED')

    def test_analysis_keeps_zero_hit_insertion_fill_and_missing_rows_fail(self):
        grid=profiles([0],[None]);tasks=schedule(grid,[0.,1.]);plan=dict(profiles=grid,schedule=tasks,fixture=True,seed=42)
        rows=[]
        for task in tasks:
            for phase,modes in (('warmup',task['warmup_modes']),('measured',task['measured_modes'])):
                for i,mode in enumerate(modes):
                    rows.append(dict(request_id=f'{task["arm"]}_{phase}_{i}',phase=phase,condition_id=grid[0]['id'],
                        arm=task['arm'],ordinal=i,mode=mode,cache_bitwise_passed=True,calibration_status='VALID',
                        request_wall_ms=10. if mode=='NO_REUSE' else 11. if mode=='COLD_MISS' else 1.,
                        cache_reset_ms_outside_request=1. if mode=='COLD_MISS' else 0.,arm_reset_ms_outside_request=1. if i==0 and phase=='warmup' else 0.,
                        application_wire_bytes=0 if mode=='WARM_HIT' else 1000,cache_status='HIT' if mode=='WARM_HIT' else 'MISS'))
        points,decision=analyze(rows,plan)
        self.assertEqual(len(points),2);self.assertEqual({p['realized_hit_fraction'] for p in points},{0.,1.})
        self.assertTrue(all(p['controlled_warmup_fill_and_reset_amortized_delta_ms']>p['delta_mean_ms'] for p in points))
        self.assertFalse(decision['PRIMARY_PERFORMANCE_ELIGIBLE']);self.assertFalse(decision['r2d_break_even_validated'])
        self.assertTrue(all(v['status']=='NOT_ESTIMABLE' for v in decision['bandwidth_crossovers'].values()))
        with self.assertRaises(ProtocolError):analyze(rows[:-1],plan)

    def test_optional_r2d_sha_is_read_only_and_never_numeric_validation(self):
        self.assertEqual(r2d_comparison()['status'],'NOT_DIRECTLY_COMPARABLE')
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)
            write(path/'r2d_evidence_audit.json',dict(execution_status='COMPLETE'))
            write(path/'r2d_break_even_summary.json',dict(schema='r2d_break_even_v1',historical_m9a=[]))
            write(path/'r2d_decision.json',dict(execution_status='COMPLETE',classification='ANALYTICAL_DIAGNOSTIC'))
            (path/'r2d_bandwidth_rtt_sweep.csv').write_text('TEST_FIXTURE_ONLY\n')
            hashes={p.name:sha(p) for p in path.iterdir()}
            write(path/'r2d_manifest.json',dict(execution_status='COMPLETE',classification='ANALYTICAL_DIAGNOSTIC',output_hashes=hashes))
            before={p.name:sha(p) for p in path.iterdir()};result=r2d_comparison(path)
            self.assertEqual(result['status'],'NOT_DIRECTLY_COMPARABLE');self.assertFalse(result['r2d_break_even_validated'])
            self.assertEqual(before,{p.name:sha(p) for p in path.iterdir()})
            (path/'r2d_bandwidth_rtt_sweep.csv').write_text('CORRUPTED')
            with self.assertRaises(ProtocolError):r2d_comparison(path)

    def test_missing_input_artifact_rejected_without_model_dependencies(self):
        spec=importlib.util.spec_from_file_location('r3c_cli',SCRIPT);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(FileNotFoundError):module.load_run(Path(temporary))

    def test_calibration_rejects_rate_residual_and_missing_samples_without_changing_tolerance(self):
        grid=profiles([0,10],[None,200]);rows=[]
        for p in grid:
            for probe in ('round_trip','upload','download'):
                for i in range(3):
                    frames=[dict(direction=d,complete=True,wire_bytes_forwarded=1024,
                        effective_mbps=1000 if p[d+'_mbps'] is None else p[d+'_mbps']*.99)
                        for d in ('es_to_ud','ud_to_es')]
                    serialization=sum(f['wire_bytes_forwarded']*.008/p[f['direction']+'_mbps']
                        if p[f['direction']+'_mbps'] else 0 for f in frames)
                    rows.append(dict(condition_id=p['id'],probe=probe,wall_ms=1+p['rtt_ms']+serialization,proxy_frames=frames))
        self.assertTrue(all(r['status']=='VALID' for r in validate_calibration(rows,grid)))
        bad=json.loads(json.dumps(rows))
        for row in bad:
            if row['condition_id']=='C01' and row['probe']=='upload':row['proxy_frames'][0]['effective_mbps']=100
        self.assertEqual(validate_calibration(bad,grid)[1]['status'],'INVALID_IMPAIRMENT')
        bad=json.loads(json.dumps(rows))
        for row in bad:
            if row['condition_id']=='C02' and row['probe']=='round_trip':row['wall_ms']+=20
        self.assertEqual(validate_calibration(bad,grid)[2]['status'],'INVALID_IMPAIRMENT')
        with self.assertRaises(ProtocolError):validate_calibration(rows[:-1],grid)

    def test_offline_model_preflight_does_not_load_models_and_fails_on_missing_shard(self):
        from semcache.edgelora.r3c_runtime import offline_model_preflight
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/('a'*40);path.mkdir()
            write(path/'config.json',dict(model_type='opt',num_hidden_layers=32,hidden_size=2560))
            write(path/'pytorch_model.bin.index.json',dict(weight_map={'TEST_FIXTURE_ONLY':'part-00001.bin'}))
            (path/'part-00001.bin').write_bytes(b'TEST_FIXTURE_ONLY_NOT_MODEL_WEIGHTS')
            (path/'tokenizer.json').write_bytes(b'TEST_FIXTURE_ONLY')
            def cached(repo,name,revision):return str(path/name) if (path/name).is_file() else None
            module=types.SimpleNamespace(try_to_load_from_cache=cached)
            with patch.dict(sys.modules,{'huggingface_hub':module}):
                self.assertEqual(offline_model_preflight()['status'],'PASS')
                (path/'part-00001.bin').unlink()
                with self.assertRaises(ProtocolError):offline_model_preflight()


if __name__=='__main__':unittest.main()
