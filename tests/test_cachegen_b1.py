"""B1 tests with tiny packed tensors. No models, downloads, CUDA or coder runs."""
from contextlib import ExitStack
import csv
import importlib.util
import json
import math
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from semcache.experiments.cachegen import b1
from semcache.experiments.cachegen.shared.device_contract import make_contract

CONTRACT = make_contract('cuda', cuda_available=True, cuda_device_count=1)


def blocks():
    return [dict(block_id=f'{s}_{d}', partition=s, dataset=d, token_group_size=10,
                 query_id=f'{s}_{d}') for s in b1.SPLITS for d in b1.DATASETS]


def toy_diagnostics(root, rec, block, contract):
    data, shape = bytes([0, 127, 254]*10), (1, 10, 3)
    hist = {c: b1.representation_counts(data, shape) for c in ('K', 'V')}
    local = {c: [dict(token_distance=i, element_count=3, abs_delta_sum=3., delta_sq_sum=3.,
        target_sq_sum=12., anchor_sq_sum=3., dot_sum=6.) for i in range(1, 10)] for c in ('K', 'V')}
    exact = dict(block_id=block['block_id'], K_symbol_roundtrip_exact=True, V_symbol_roundtrip_exact=True,
                 scales_preserved_exact=True, unchanged_quantizer_reconstruction_exact=True,
                 saved_c1_reconstruction_exact=block['partition'] == 'evaluation')
    return hist, local, exact


class TransformTests(unittest.TestCase):
    def test_bijection_every_symbol_and_invalid_values(self):
        self.assertEqual([b1.symbol_to_u8_domain(x) for x in range(-127, 128)], list(range(255)))
        for s in range(-127, 128):
            self.assertEqual(b1.u8_domain_to_symbol(b1.symbol_to_u8_domain(s)), s)
        for s in (-128, 128, 1.0, True):
            with self.assertRaises(ValueError): b1.symbol_to_u8_domain(s)
        for u in (-1, 255, 1.0, True):
            with self.assertRaises(ValueError): b1.u8_domain_to_symbol(u)

    def test_all_anchor_value_pairs_extremes_and_every_alphabet_value(self):
        # Every one of the 255**2 anchor/token pairs, packed as channels.
        shape = (1, 10, 255)
        for anchor in range(255):
            original = bytes([anchor]*255)+bytes(range(255))*9
            transformed = b1.transform(original, shape)
            self.assertEqual(transformed[:255], original[:255])
            self.assertEqual(b1.transform(transformed, shape, inverse=True), original)
        original = bytes([0]+[254]*9)
        transformed = b1.transform(original, (1, 10, 1))
        self.assertEqual(transformed, bytes([0]+[126]*9))  # residual 254 -> -1 -> histogram index 126
        self.assertEqual(b1.transform(bytes([254]+[0]*9), (1, 10, 1)), bytes([254]+[128]*9))

    def test_center_mapping_zero_negative_and_positive(self):
        # Anchor u=0, residual_u = 0,1,127,128,254.
        original = bytes([0, 0, 1, 127, 128, 254, 0, 1, 127, 128])
        transformed = b1.transform(original, (1, 10, 1))
        self.assertEqual(list(transformed[:6]), [0, 127, 128, 254, 0, 126])

    def test_random_K_V_tensor_shapes_and_anchor_unchanged(self):
        rng = random.Random(51)
        for component in ('K', 'V'):
            for shape in ((32, 10, 8), (2, 10, 257)):
                original = bytes(rng.randrange(255) for _ in range(math.prod(shape)))
                result = b1.transform(original, shape)
                self.assertEqual(b1.transform(result, shape, inverse=True), original)
                self.assertEqual(b1.role_domains(result, shape)[0], b1.role_domains(original, shape)[0])

    def test_every_token_references_same_anchor_not_previous(self):
        original = bytes(range(20, 30))
        transformed = b1.transform(original, (1, 10, 1))
        self.assertEqual(transformed, bytes([20]+list(range(128, 137))))
        self.assertNotEqual(transformed, bytes([20]+[128]*9))

    def test_raw_role_split_never_transforms_values(self):
        data = bytes(range(40))
        anchor, nonanchor = b1.role_domains(data, (2, 10, 2))
        self.assertEqual(anchor, bytes([0, 1, 20, 21]))
        self.assertEqual(nonanchor, data[2:20]+data[22:])
        hist = b1.representation_counts(data, (2, 10, 2))
        self.assertEqual(hist['RAW_ROLE_SPLIT', 'anchor'], b1.counts(anchor))
        self.assertEqual(hist['RAW_ROLE_SPLIT', 'nonanchor'], b1.counts(nonanchor))

    def test_invalid_shapes_T3_and_domain_are_rejected(self):
        for data, shape in ((b'\0'*3, (1, 3, 1)), (b'\0'*9, (1, 10, 1)),
                            (b'\xff'*10, (1, 10, 1)), (b'', (0, 10, 1))):
            with self.assertRaises(ValueError): b1.transform(data, shape)

    def test_inverse_mismatch_fails_closed(self):
        original = b1.transform
        def corrupt(data, shape, *, inverse=False):
            return b'\x00'*len(data) if inverse else original(data, shape)
        with patch.object(b1, 'transform', corrupt), self.assertRaisesRegex(ValueError, 'inverse'):
            b1.representation_counts(bytes([127]*10), (1, 10, 1))


class EntropyTests(unittest.TestCase):
    def populate(self, counts_fn):
        acc = b1.EntropyAccumulator()
        for block in blocks():
            for component in ('K', 'V'):
                acc.add(block, component, counts_fn(block, component))
        return acc

    def test_actual_count_weighting_and_separate_K_V_models(self):
        # Deliberately use 2/3, not 1/10, anchor weight to test actual-count formula.
        def hist(block, component):
            a = [0]*255; a[0 if component == 'K' else 254] = 2
            n = [0]*255; n[1] = 1
            r = [0]*255; r[127] = 1
            return {('RAW_GLOBAL', 'all'): [x+y for x,y in zip(a,n)],
                ('RAW_ROLE_SPLIT', 'anchor'): a, ('RAW_ROLE_SPLIT', 'nonanchor'): n,
                ('ANCHOR_MOD_RESIDUAL', 'anchor'): a, ('ANCHOR_MOD_RESIDUAL', 'residual'): r}
        summary = self.populate(hist).summary()
        self.assertEqual(len(summary), 54)
        for row in summary:
            if row['representation'] == 'RAW_GLOBAL':
                self.assertAlmostEqual(row['weighted_bits_per_symbol'], -(2/3)*math.log2(2/3)-(1/3)*math.log2(1/3))
            else:
                self.assertEqual(row['weighted_bits_per_symbol'], 0)
                self.assertEqual(row['anchor_symbol_count']/row['symbol_count'], 2/3)
            if row['component'] == 'K+V':
                self.assertIsNone(row['unique_symbol_count'])
        # Nonzero role entropies weighted by unequal counts.
        a, n, r = [0]*255, [0]*255, [0]*255
        a[0]=1; a[1]=1; n[1]=1; r[127]=1
        acc=self.populate(lambda *_: {('RAW_GLOBAL','all'):[x+y for x,y in zip(a,n)],
            ('RAW_ROLE_SPLIT','anchor'):a, ('RAW_ROLE_SPLIT','nonanchor'):n,
            ('ANCHOR_MOD_RESIDUAL','anchor'):a, ('ANCHOR_MOD_RESIDUAL','residual'):r})
        for row in acc.summary():
            if row['representation']!='RAW_GLOBAL': self.assertAlmostEqual(row['weighted_bits_per_symbol'], 2/3)

    def test_calibration_evaluation_histograms_do_not_mix(self):
        def hist(block, component):
            data = bytes([127]*10) if block['partition']=='calibration' else bytes(range(10))
            return b1.representation_counts(data,(1,10,1))
        acc=self.populate(hist)
        for row in acc.summary():
            if row['representation']=='RAW_GLOBAL':
                self.assertAlmostEqual(row['weighted_bits_per_symbol'], 0 if row['split']=='calibration' else math.log2(10))

    def test_duplicate_and_invalid_count_conservation_rejected(self):
        block=blocks()[0]; hist=b1.representation_counts(bytes(range(10)),(1,10,1))
        acc=b1.EntropyAccumulator(); acc.add(block,'K',hist)
        with self.assertRaisesRegex(ValueError,'Duplicate'): acc.add(block,'K',hist)
        hist['ANCHOR_MOD_RESIDUAL','residual'][0]+=1
        with self.assertRaisesRegex(ValueError,'conservation'): b1.EntropyAccumulator().add(block,'K',hist)

    def test_decision_threshold_and_both_datasets_predeclared(self):
        def rows(overall, snips, multiwoz):
            return [dict(split='evaluation',component='K+V',representation='ANCHOR_MOD_RESIDUAL',
                dataset=d,reduction_vs_RAW_ROLE_SPLIT_percent=v) for d,v in zip(('ALL','snips','multiwoz'),(overall,snips,multiwoz))]
        self.assertEqual(b1.decision(rows(3,1,1))['recommendation'],'GO_TO_B2')
        for values in ((2.99,5,5),(5,0,10),(5,-1,10),(None,1,1),(0,0,0)):
            self.assertEqual(b1.decision(rows(*values))['recommendation'],'STOP')
        self.assertEqual(b1.RULE['provenance'],'REPRODUCTION_CHOICE')
        self.assertEqual(b1.reduction(0,0),0)
        self.assertIsNone(b1.reduction(1,0))

    def test_locality_definitions_zero_norm_and_pooled_counts(self):
        acc={('evaluation','ALL','K',1):dict(element_count=4,abs_delta_sum=8.,delta_sq_sum=16.,target_sq_sum=4.,anchor_sq_sum=4.,dot_sum=0.),
             ('evaluation','ALL','V',1):dict(element_count=4,abs_delta_sum=0.,delta_sq_sum=0.,target_sq_sum=0.,anchor_sq_sum=0.,dot_sum=0.)}
        k,v=b1.locality_summary(acc)
        self.assertEqual((k['mean_absolute_delta'],k['rms_delta'],k['relative_L2']),(2,2,2))
        self.assertIsNone(v['relative_L2']); self.assertIsNone(v['cosine_similarity'])


class HarnessTests(unittest.TestCase):
    def test_binding_checks_completed_A_C1_device_baseline_and_profile_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'c1'; a=Path(tmp)/'a'; root.mkdir(); (a/'full_storage').mkdir(parents=True)
            capture={'scope':'frozen'}
            b1.atomic_json(root/'capture_manifest.json',capture)
            b1.atomic_json(root/'reconstruction_manifest.json',{'capture_manifest_sha256':b1.digest(capture)})
            for name in ('c1_block_raw.csv','c1_summary.csv','c1_quality.csv','environment.json'):
                (root/name).write_text('frozen')
            for name in ('manifest.json','profile_manifest.json','shared_cdf_global.bin','shared_cdf_layergroup.bin',
                         'full_storage/storage_manifest.json','full_storage/c15_full_block_raw.csv',
                         'full_storage/c15_full_summary.csv','full_storage/environment.json'):
                (a/name).write_text('frozen')
            state=dict(status='COMPLETED',primary_result_eligible=True,completed_block_mode_pairs=2616,failed_pairs=0,
                run_contract=dict(device_contract=CONTRACT,profile_manifest_sha256=b1.file_hash(a/'profile_manifest.json')),
                baseline_source_sha256=b1.file_hash(Path(b1.__file__).parent/'codecs.py'),
                c1_capture_manifest_sha256=b1.file_hash(root/'capture_manifest.json'),
                c1_reconstruction_manifest_sha256=b1.file_hash(root/'reconstruction_manifest.json'),
                profile_hashes={'SHARED_CDF_GLOBAL':b1.file_hash(a/'shared_cdf_global.bin'),
                                'SHARED_CDF_LAYERGROUP':b1.file_hash(a/'shared_cdf_layergroup.bin')})
            b1.atomic_json(a/'full_storage/manifest.json',state)
            args=SimpleNamespace(capture_manifest=root/'capture_manifest.json',c15a_dir=a)
            selected=[dict(b,file=b['block_id']+'.pt') for b in blocks()]
            with patch.object(b1,'select_blocks',return_value=(selected,{})), patch.object(b1,'resolve_contract',return_value=CONTRACT):
                result=b1.bind_inputs(args)
                self.assertEqual(result[1],selected)
                self.assertIn(str(a/'shared_cdf_global.bin'),result[-1])
                (a/'shared_cdf_global.bin').write_text('corrupt')
                with self.assertRaisesRegex(ValueError,'profile hash'): b1.bind_inputs(args)
                (a/'shared_cdf_global.bin').write_text('frozen')
                state['status']='INCOMPLETE'; b1.atomic_json(a/'full_storage/manifest.json',state)
                with self.assertRaisesRegex(ValueError,'completed primary'): b1.bind_inputs(args)
                state['status']='COMPLETED'; state['baseline_source_sha256']='wrong'
                b1.atomic_json(a/'full_storage/manifest.json',state)
                with self.assertRaisesRegex(ValueError,'Baseline source'): b1.bind_inputs(args)
            with patch.object(b1,'select_blocks',return_value=(selected,{})), patch.object(b1,'resolve_contract',return_value=make_contract('cpu')):
                with self.assertRaisesRegex(ValueError,'CUDA'): b1.bind_inputs(args)

    def test_selection_excludes_T3_uses_all_T10_both_splits(self):
        selected=[dict(block_id=f'{s}{i}',partition=s,dataset='snips' if i%2 else 'multiwoz',token_group_size=10)
                  for s,n in [('calibration',18),('evaluation',236)] for i in range(n)]
        capture=dict(scope='base_raw_unscaled_linear_projection; no LoRA adapter',blocks=selected+[
            dict(block_id='not_loaded',partition='evaluation',dataset='snips',token_group_size=3)])
        with patch.object(b1,'verify_capture'):
            actual, population=b1.select_blocks(capture)
            self.assertEqual(actual,selected)
            self.assertEqual(population['calibration'],dict(snips=9,multiwoz=9,ALL=18))
            self.assertEqual(population['evaluation']['ALL'],236)
            capture['blocks']=selected[:-1]
            with self.assertRaisesRegex(ValueError,'236'): b1.select_blocks(capture)

    def job(self, tmp, stack, *, failure=False):
        parent, c1, a=Path(tmp)/'b', Path(tmp)/'c1',Path(tmp)/'a'
        for p in (parent,c1,a): p.mkdir()
        b2,b3={'status':'DEFERRED_B2'}, {'status':'DEFERRED_B3'}
        b1.atomic_json(parent/'design_manifest.json',dict(stages=dict(B1={},B2=b2,B3=b3)))
        for p in (c1/'capture_manifest.json',a/'profile_manifest.json',a/'smoke.json'):
            p.write_text('immutable')
        hashes={str(p):b1.file_hash(p) for p in (c1/'capture_manifest.json',a/'profile_manifest.json',a/'smoke.json')}
        args=SimpleNamespace(output_dir=parent,capture_manifest=c1/'capture_manifest.json',c15a_dir=a)
        chosen=blocks()
        stack.enter_context(patch.object(b1,'bind_inputs',return_value=(c1,chosen,
            {s:dict(snips=1,multiwoz=1,ALL=2) for s in b1.SPLITS},{},CONTRACT,hashes)))
        stack.enter_context(patch.object(b1,'resolve_contract',return_value=CONTRACT))
        stack.enter_context(patch.object(b1,'environment',return_value={'test':'no torch'}))
        def diagnostics(*a):
            state=b1.read_json(parent/'b1/manifest.json')
            design=b1.read_json(parent/'design_manifest.json')
            self.assertEqual(state['config']['decision_rule'],b1.RULE)
            self.assertEqual(design['stages']['B1']['decision_rule'],b1.RULE)
            self.assertFalse(state['diagnostic_result_eligible'])
            if failure: raise ValueError('roundtrip differs')
            return toy_diagnostics(*a)
        spy=stack.enter_context(patch.object(b1,'tensor_diagnostics',side_effect=diagnostics))
        for function in ('encode','decode','arithmetic_encode','arithmetic_decode','fit_profiles','cdf_from_counts'):
            stack.enter_context(patch('semcache.experiments.cachegen.shared.core.'+function,
                side_effect=AssertionError('B1 cannot invoke coding/profile fitting')))
        stack.enter_context(patch('builtins.print'))
        return args,hashes,spy

    def test_run_outputs_decision_frozen_inputs_no_coder_no_profile_fitting(self):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            args,hashes,spy=self.job(tmp,stack)
            b1.run(args)
            out=args.output_dir/'b1'
            state=b1.read_json(out/'manifest.json')
            self.assertEqual(state['status'],'COMPLETED')
            self.assertEqual(state['decision']['recommendation'],'GO_TO_B2')
            self.assertFalse(state['decision']['B2_executed'])
            self.assertEqual(spy.call_count,4)
            self.assertEqual([call.args[2]['partition'] for call in spy.call_args_list],['calibration']*2+['evaluation']*2)
            for name,expected in [('b1_entropy_raw.csv',40),('b1_entropy_summary.csv',54),('b1_locality_summary.csv',120)]:
                with (out/name).open() as f: self.assertEqual(len(list(csv.DictReader(f))),expected)
            for path,value in hashes.items(): self.assertEqual(b1.file_hash(path),value)
            design=b1.read_json(args.output_dir/'design_manifest.json')
            self.assertEqual(design['stages']['B2'],{'status':'DEFERRED_B2'})
            self.assertEqual(design['stages']['B3'],{'status':'DEFERRED_B3'})
            self.assertFalse(list(out.glob('*.bin')))
            with self.assertRaises(FileExistsError): b1.run(args)

    def test_failed_correctness_noneligible_no_decision(self):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            args,hashes,spy=self.job(tmp,stack,failure=True)
            with self.assertRaisesRegex(ValueError,'roundtrip'): b1.run(args)
            state=b1.read_json(args.output_dir/'b1/manifest.json')
            self.assertEqual(state['status'],'FAILED')
            self.assertFalse(state['diagnostic_result_eligible'])
            self.assertIsNone(state['decision'])
            self.assertFalse((args.output_dir/'b1/b1_entropy_summary.csv').exists())

    def test_interrupted_noneligible(self):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            args,hashes,spy=self.job(tmp,stack)
            spy.side_effect=KeyboardInterrupt()
            with self.assertRaises(KeyboardInterrupt): b1.run(args)
            state=b1.read_json(args.output_dir/'b1/manifest.json')
            self.assertEqual(state['status'],'INCOMPLETE')
            self.assertFalse(state['diagnostic_result_eligible'])

    def test_output_guard_preserves_C1_A(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); c1=root/'c1'; a=root/'a'; c1.mkdir(); a.mkdir()
            for out in (c1,a,a/'b1',root):
                args=SimpleNamespace(output_dir=out,capture_manifest=c1/'capture_manifest.json',c15a_dir=a)
                with self.assertRaises(ValueError): b1.run(args)


@unittest.skipUnless(importlib.util.find_spec('torch'), 'PyTorch unavailable; no installation or download')
class TensorBoundaryTests(unittest.TestCase):
    def test_real_tensor_quantization_inverse_saved_reference_and_Q(self):
        import torch
        from semcache.experiments.cachegen.codecs import Baseline
        rng=torch.Generator().manual_seed(73)
        qkv={c:torch.randn(32,10,8,generator=rng).half() for c in 'qkv'}
        q=qkv['q'].clone()
        encoded=Baseline('UNIFORM_INT8').encode(qkv['k'],qkv['v'])
        reference=Baseline('UNIFORM_INT8').decode(encoded)
        block=dict(blocks()[2])
        def saved(root, rec, b, actual):
            self.assertTrue(all(torch.equal(x,y) for x,y in zip(actual,reference)))
        with patch.object(b1,'load_fixture',return_value=qkv), patch.object(b1,'existing_uniform',side_effect=saved) as check:
            hist, locality, exact=b1.tensor_diagnostics(None,None,block,
                dict(quantization_device_resolved='cpu',reconstruction_device='cpu'))
            self.assertTrue(exact['saved_c1_reconstruction_exact'])
            check.assert_called_once()
            self.assertTrue(torch.equal(q,qkv['q']))
            self.assertEqual(qkv['q'].dtype,torch.float16)
            self.assertEqual(sum(hist['K']['RAW_GLOBAL','all']),32*10*8)
            self.assertEqual(len(locality['V']),9)


if __name__=='__main__': unittest.main()
