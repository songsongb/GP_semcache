"""Tiny CPU-only holdout checks; no SERAPH fixture, model or real evaluation."""
from dataclasses import replace
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from semcache.experiments.cachegen.b2 import format as fmt
from semcache.experiments.cachegen.c15c import holdout as h
from semcache.experiments.cachegen.c15c import rate_storage as rs
from semcache.experiments.cachegen.c15c import harness
from semcache.experiments.cachegen.common import DIMENSIONS, digest, file_hash, write_json
from semcache.experiments.cachegen.shared import core
from semcache.experiments.cachegen.shared.source_audit import REVISION

torch_available = pytest.mark.skipif(not importlib.util.find_spec('torch'), reason='Existing CPU torch required')


def capture(blocks):
    plans = {}
    for dataset in sorted({b.get('dataset') for b in blocks if b['partition'] == 'evaluation'}):
        samples = [dict(source_id=f'eval_{dataset}')]
        plans[dataset] = dict(calibration=[], evaluation=samples,
                              hashes=dict(calibration=digest([]), evaluation=digest(samples)))
    return dict(status='CAPTURED', model_config=dict(name='facebook/opt-2.7b', dtype='float16'),
        model_metadata=dict(resolved_model_revision='model', resolved_tokenizer_revision='tokenizer'),
        scope='base_raw_unscaled_linear_projection; no LoRA adapter', blocks=blocks,
        sampling=plans, sampling_sha256=digest(plans))


def eval_block(dataset='snips', tokens=3, block_id='eval'):
    ids = list(range(2, 2+tokens))
    return dict(block_id=block_id, partition='evaluation', dataset=dataset, token_group_size=tokens,
        start_position=0, query_token_ids=ids, query_id=f'eval_{dataset}',
        source_group_id=f'eval_{dataset}', absolute_positions=list(range(tokens)), token_ids=ids,
        model_revision='model', tokenizer_revision='tokenizer', file=f'{block_id}.pt', sha256='placeholder', **DIMENSIONS)


def frozen_calibration(tmp_path, capture_sha):
    profile = fmt.Profile(rs.MODE, (core.cdf_from_counts([0]*255),)*4)
    folder = tmp_path/'rate_calibration'
    (folder/'profiles').mkdir(parents=True)
    selected = {}
    for key, name in (('released_ql2', h.QL2), ('matched_uniform', h.UNIFORM)):
        rel = h.PROFILE_FILES[name]
        (folder/rel).write_bytes(profile.to_bytes())
        policy = h.CACHEGEN_RELEASED_QL2 if name == h.QL2 else h.UniformKVPolicy(20, 16)
        selected[key] = dict(file=rel, sha256=profile.sha256, serialized_bytes=len(profile.to_bytes()),
            policy=name, layer_profile=policy.profile(), mode=rs.MODE, calibration_scope_sha256='a'*64)
    selection = dict(primary_policy=h.UNIFORM, primary=dict(K_bins=20, V_bins=16), profiles=selected)
    write_json(folder/'selection.json', selection)
    write_json(folder/'calibration_manifest.json', dict(status='COMPLETED', profiles_status='FROZEN_FOR_C15C3',
        capture_manifest_sha256=capture_sha, w=3, dtype='float16', calibration_block_count=1094,
        cachegen_reference=dict(commit=REVISION), selected_profiles=selected, calibration_scope_sha256='a'*64,
        output_sha256={'selection.json': file_hash(folder/'selection.json')}, device_contract={'reference': 'synthetic'}))
    return folder, profile


def test_evaluation_partition_only_and_metadata_counts_without_calibration_details():
    class PoisonCalibration(dict):
        def __getitem__(self, key):
            if key != 'partition': raise AssertionError('Calibration details accessed')
            return 'calibration'
    a, b = eval_block('snips', 3, 'snips_3'), eval_block('multiwoz', 10, 'multiwoz_10')
    selected, counts = h.evaluation_blocks(capture([PoisonCalibration(), b, a]))
    assert [x['block_id'] for x in selected] == ['snips_3']
    assert all(x['token_group_size'] == h.REQUIRED_WINDOW_SIZE for x in selected)
    assert counts == dict(total_capture_blocks=3, total_calibration_blocks=1,
        total_evaluation_blocks=2, evaluation_w3_count=1, evaluation_non_w3_count=1,
        selected_holdout_count=1, required_window_size=3)
    with pytest.raises(ValueError, match='No evaluation w=3'):
        h.evaluation_blocks(dict(capture([a]), blocks=[PoisonCalibration()]))
    with pytest.raises(ValueError, match='No evaluation w=3'):
        h.evaluation_blocks(capture([b]))


def test_frozen_profile_loader_checks_selection_hash_names_scope_and_commit(tmp_path):
    directory, p = frozen_calibration(tmp_path, 'capture-hash')
    profiles, refs, contract = h.frozen_profiles(directory, 'capture-hash')
    assert set(profiles) == set(h.POLICIES)
    assert profiles[h.QL2] == profiles[h.UNIFORM] == p
    assert contract == {'reference': 'synthetic'}
    assert refs['profiles'][h.UNIFORM]['bytes'] == 4108
    with pytest.raises(ValueError): h.frozen_profiles(directory, 'different-capture')
    (directory/h.PROFILE_FILES[h.UNIFORM]).write_bytes(p.to_bytes()+b'x')
    with pytest.raises(ValueError, match='SHA256'): h.frozen_profiles(directory, 'capture-hash')
    (directory/h.PROFILE_FILES[h.UNIFORM]).write_bytes(p.to_bytes())
    selection_path = directory/'selection.json'
    selection = h.read_json(selection_path)
    selection['primary_policy'] = 'UNIFORM_K18_V16'
    write_json(selection_path, selection)
    with pytest.raises(ValueError, match='selection'):
        h.frozen_profiles(directory, 'capture-hash')


def test_direct_quality_delta_sign_convention():
    q = dict(K=dict(relative_l2=.2, cosine_similarity=.9, MSE=4., max_abs_error=3.),
             V=dict(relative_l2=.1, cosine_similarity=.95, MSE=2., max_abs_error=1.))
    u = dict(K=dict(relative_l2=.1, cosine_similarity=.95, MSE=3., max_abs_error=2.),
             V=dict(relative_l2=.2, cosine_similarity=.9, MSE=3., max_abs_error=2.))
    delta = h.compare_quality(q, u)
    assert delta['K']['relative_l2_difference'] == pytest.approx(-.1)
    assert delta['K']['relative_l2_absolute_difference'] == pytest.approx(.1)
    assert delta['K']['relative_l2_percent_difference'] == pytest.approx(-50)
    assert delta['K']['cosine_difference'] == pytest.approx(-.05)
    assert delta['K']['MSE_percent_difference'] == pytest.approx(-25)
    assert delta['K']['max_abs_difference'] == -1
    assert delta['V']['relative_l2_difference'] == pytest.approx(.1)
    assert delta['V']['cosine_difference'] == pytest.approx(.05)
    q['K']['relative_l2'] = 0
    assert h.compare_quality(q, u)['K']['relative_l2_percent_difference'] is None


def test_comparison_flags_and_exact_rate_denominators():
    def item(k, v, local, profile, m):
        total = k+v+local+profile
        return dict(physical_storage=dict(k_payload_bytes=k, v_payload_bytes=v,
            local_metadata_bytes=local, global_profile_bytes=profile, total_physical_bytes=total),
            reconstruction={r: dict(relative_l2=x, MSE=x*x, cosine_similarity=1-x,
                max_abs_error=x) for r, x in m.items()})
    q = item(100, 200, 867, 4108, {'K': .2, 'V': .1})
    u = item(90, 220, 867, 4108, {'K': .1, 'V': .2})
    comparison = h.compare_results(dict(overall={h.QL2: q, h.UNIFORM: u}))
    assert comparison['K_payload_gap_percent'] == -10
    assert comparison['V_payload_gap_percent'] == 10
    assert comparison['overall_physical_storage_gap_percent'] == pytest.approx(100*10/q['physical_storage']['total_physical_bytes'])
    assert not comparison['uniform_overall_storage_smaller']
    assert comparison['uniform_K_rel_l2_lower'] and comparison['uniform_K_MSE_lower']
    assert not comparison['uniform_V_rel_l2_lower'] and not comparison['uniform_V_MSE_lower']
    assert comparison['selected_policy_changed'] is False


@torch_available
def test_one_loader_scan_eight_encodes_no_decode_no_fit_and_w3_accounting():
    import torch
    generator = torch.Generator().manual_seed(19)
    fixtures = {name: {r: torch.randn(32, 3, 2, generator=generator).half() for r in ('k', 'v')}
                for name in ('a', 'b')}
    blocks, counts = h.evaluation_blocks(capture([
        eval_block('snips', 3, 'a'), eval_block('multiwoz', 3, 'b'),
        eval_block('snips', 10, 'excluded_w10')]))
    assert counts['selected_holdout_count'] == 2
    assert counts['evaluation_non_w3_count'] == 1
    cdf = core.cdf_from_counts([0]*255)
    profile = fmt.Profile(rs.MODE, (cdf,)*4)
    loaded, progress = [], []
    def loader(block):
        loaded.append(block['block_id'])
        return fixtures[block['block_id']]
    with patch.object(core, 'arithmetic_encode', wraps=core.arithmetic_encode) as encode, \
         patch.object(core, 'arithmetic_decode', side_effect=AssertionError('No holdout decode')), \
         patch.object(core, 'cdf_from_counts', side_effect=AssertionError('No CDF fitting')), \
         patch.object(rs, 'fit_role', side_effect=AssertionError('No role fitting')):
        summary, comparison = h.evaluate(blocks, {h.QL2: profile, h.UNIFORM: profile}, loader,
            device='cpu', progress=lambda done, total: progress.append((done, total)))
    assert encode.call_count == 16
    assert loaded == ['a', 'b'] and progress == [(1, 2), (2, 2)]
    assert summary['arithmetic_encode_calls'] == 16
    assert summary['required_window_size'] == 3
    assert summary['arithmetic_decode_calls'] == summary['cdf_fit_calls'] == 0
    assert set(summary['datasets']) == {'snips', 'multiwoz'}
    for policy in h.POLICIES:
        overall = summary['overall'][policy]
        a, b = summary['datasets']['snips'][policy], summary['datasets']['multiwoz'][policy]
        storage = overall['physical_storage']
        assert overall['block_count'] == 2
        assert storage['local_transform_metadata_bytes'] == 2*99
        assert storage['scale_maxabs_metadata_bytes'] == 2*32*(3+3)*4
        assert storage['global_profile_bytes'] == 4108
        assert storage['total_physical_bytes'] == storage['bitstream_pool_bytes']+4108
        assert storage['k_payload_bytes'] == a['physical_storage']['k_payload_bytes']+b['physical_storage']['k_payload_bytes']
        assert storage['v_payload_bytes'] == a['physical_storage']['v_payload_bytes']+b['physical_storage']['v_payload_bytes']
        assert storage['original_fp16_kv_bytes'] == 2*32*(3+3)*2*2
        for role in ('K', 'V'):
            pooled = overall['reconstruction'][role]
            assert pooled['element_count'] == a['reconstruction'][role]['element_count']+b['reconstruction'][role]['element_count']
            assert pooled['MSE'] >= 0 and pooled['relative_l2'] >= 0
    assert comparison['calibration_selected_policy'] == h.UNIFORM
    rows = h.per_dataset_csv_rows(summary)
    assert len(rows) == 8 and {r['policy'] for r in rows} == set(h.POLICIES)
    assert all(r['compression_ratio'] > 0 for r in rows)
    with pytest.raises(ValueError, match='exactly the two'):
        h.evaluate(blocks, {h.QL2: profile}, lambda b: pytest.fail('Loader reached'), device='cpu')
    with pytest.raises(ValueError, match='evaluation'):
        h.evaluate([dict(blocks[0], partition='calibration')], {h.QL2: profile, h.UNIFORM: profile},
                   lambda b: pytest.fail('Calibration loaded'), device='cpu')
    with pytest.raises(ValueError, match='w=3'):
        h.evaluate([dict(blocks[0], token_group_size=10)], {h.QL2: profile, h.UNIFORM: profile},
                   lambda b: pytest.fail('T=10 fixture loaded'), device='cpu')


def test_dry_run_counts_profiles_and_encodes_without_fixture_cuda_or_output(tmp_path, capsys):
    blocks = [eval_block('snips', 3, 'w3_a'), eval_block('multiwoz', 3, 'w3_b'),
              eval_block('snips', 10, 'excluded_w10')]
    c = capture([dict(partition='calibration'), *blocks])
    capture_path = tmp_path/'capture_manifest.json'
    write_json(capture_path, c)
    frozen, _ = frozen_calibration(tmp_path, file_hash(capture_path))
    out = tmp_path/'holdout'
    args = SimpleNamespace(capture_manifest=capture_path, rate_calibration_dir=frozen,
        output_dir=out, seed=42, dry_run=True)
    with patch.object(h, 'load_fixture', side_effect=AssertionError('No fixture reads')), \
         patch.object(h, 'resolve_contract', side_effect=AssertionError('No CUDA preflight')), \
         patch.object(core, 'arithmetic_encode', side_effect=AssertionError('No encoding')):
        result = h.run_holdout(args)
    assert result['expected_arithmetic_encode_calls'] == 2*2*4
    assert result['counts']['required_window_size'] == 3
    assert result['counts']['selected_holdout_count'] == 2
    assert not out.exists()
    printed = capsys.readouterr().out
    assert 'capture=4\ncalibration=1\nevaluation_total=3\nevaluation_w3=2\n' in printed
    assert 'evaluation_excluded_non_w3=1\nselected_holdout=2\n' in printed
    assert 'expected arithmetic encode calls=16' in printed
    assert 'no fixtures loaded' in printed


def test_cli_registration_and_no_calibration_candidate_flags(capsys):
    with pytest.raises(SystemExit) as error:
        harness.main(['holdout-compare', '--help'])
    assert error.value.code == 0
    assert '--dry-run' in capsys.readouterr().out
    with pytest.raises(SystemExit):
        harness.main(['holdout-compare', '--candidate-bins', '8'])
