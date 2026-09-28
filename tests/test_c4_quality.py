"""C4 synthetic controls; no OPT download or real inference."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from semcache.experiments.cachegen import c4_quality as c4
from semcache.experiments.cachegen.c2.physical_storage import PROFILE_SHA256
from semcache.experiments.m9b_semantic_workload import MODEL_ID, MODEL_REVISION


def row(dataset='snips', source_id='q0', ids=None, cluster=2):
    ids = ids or [1, 2, 3, 4, 5, 6]
    group = source_id.rpartition(':')[0]
    return dict(dataset=dataset, source_id=source_id,
                domain_or_intent=group if dataset == 'snips' else None,
                conversation_id=group if dataset == 'multiwoz' else None,
                query_text=f'prepared {source_id}',
                token_ids=ids, cluster_id=cluster, model_id=MODEL_ID,
                model_revision=MODEL_REVISION, tokenizer_id=f'{MODEL_ID}@{MODEL_REVISION}',
                semantic_assignment_source='prepared-semantic-fixture')


def test_deterministic_exact_repeat_selection_and_one_block():
    rows = [row(source_id='short', ids=[1, 2, 3, 4]), row(source_id='a'),
            row(source_id='a'), row(source_id='b', ids=[8, 9, 10, 11, 12])]
    selected, skipped = c4.select_cases(rows, 'snips', 2)
    assert [c['source_id'] for c in selected] == ['a', 'b']
    assert selected == c4.select_cases(rows, 'snips', 2)[0]
    assert skipped['prompt_length_outside_5_to_32'] == 1
    for case in selected:
        assert c4.validate_case(case)
        assert case['target_id'] != case['source_id']
        assert case['reused_token_ids'] == case['target_token_ids'][:3]
        assert case['reuse_start'] == 0 and case['reuse_end'] == 3
    selected[0]['reused_token_ids'] = [9, 9, 9]
    with pytest.raises(ValueError, match='exact-repeat'): c4.validate_case(selected[0])


def test_diverse_snips_round_robin_source_id_intents():
    rows = [row(source_id=f'AddToPlaylist:{i}') for i in range(4)] + [
        row(source_id=f'BookRestaurant:{i}') for i in range(3)] + [
        row(source_id=f'GetWeather:{i}') for i in range(3)]
    selected, skipped = c4.select_cases(rows, 'snips', 7, mode='diverse')
    assert [c['source_id'] for c in selected] == [
        'AddToPlaylist:0', 'BookRestaurant:0', 'GetWeather:0',
        'AddToPlaylist:1', 'BookRestaurant:1', 'GetWeather:1', 'AddToPlaylist:2']
    assert [c['selection_group'] for c in selected[:3]] == [
        'AddToPlaylist', 'BookRestaurant', 'GetWeather']
    assert selected == c4.select_cases(rows, 'snips', 7, mode='diverse')[0]
    assert skipped == {}
    bad = [dict(row(source_id='AddToPlaylist:0'), domain_or_intent='BookRestaurant')]
    with pytest.raises(ValueError, match='disagrees'): c4.select_cases(bad, 'snips', 1, mode='diverse')


def test_diverse_multiwoz_one_per_dialogue_before_second_turn():
    rows = [row('multiwoz', source_id) for source_id in (
        'dlgA:0', 'dlgA:2', 'dlgB:0', 'dlgB:2', 'dlg:part:1', 'dlg:part:3')]
    selected, _ = c4.select_cases(rows, 'multiwoz', 5, mode='diverse')
    assert [c['source_id'] for c in selected] == [
        'dlgA:0', 'dlgB:0', 'dlg:part:1', 'dlgA:2', 'dlgB:2']
    assert [c['selection_group'] for c in selected[:3]] == ['dlgA', 'dlgB', 'dlg:part']
    assert selected == c4.select_cases(rows, 'multiwoz', 5, mode='diverse')[0]
    colon_turn = dict(row('multiwoz', 'dlgA:turn:2'), conversation_id='dlgA')
    assert c4.select_cases([colon_turn], 'multiwoz', 1, mode='diverse')[0][0]['selection_group'] == 'dlgA'
    with pytest.raises(ValueError, match='dialogue ID'):
        c4.select_cases([dict(row('multiwoz', 'dlgA:0'), conversation_id='wrong')],
                        'multiwoz', 1, mode='diverse')


def test_exact_token_safety_rejects_wrong_prompt_position_and_key():
    case = c4.select_cases([row()], 'snips', 1)[0][0]
    for altered in (dict(case, target_token_ids=[1, 2, 3, 4, 5, 7]),
                    dict(case, reuse_start=1),
                    dict(case, cache_key=(99, tuple(case['reused_token_ids'])))):
        with pytest.raises(ValueError): c4.validate_case(altered)


def test_frozen_profile_hash_guard(tmp_path, monkeypatch):
    path = tmp_path/'profile.bin'
    with pytest.raises(ValueError, match='missing'): c4.verify_profile(path)
    path.write_bytes(b'wrong')
    with pytest.raises(ValueError, match='SHA256'): c4.verify_profile(path)
    monkeypatch.setattr(c4, 'file_hash', lambda p: PROFILE_SHA256)
    assert c4.verify_profile(path)['sha256'] == PROFILE_SHA256


def test_generation_agreement_metrics():
    same = c4.generation_comparison([1, 2, 3], [1, 2, 3])
    assert same == dict(exact_sequence_match=True, prefix_agreement_length=3,
                        token_position_agreement=1., normalized_token_edit_distance=0.)
    changed = c4.generation_comparison([1, 2, 3], [1, 4, 3])
    assert changed['exact_sequence_match'] is False
    assert changed['prefix_agreement_length'] == 1
    assert changed['token_position_agreement'] == pytest.approx(2/3)
    assert changed['normalized_token_edit_distance'] == pytest.approx(1/3)


def test_dry_run_does_not_load_model_or_fit_cdf(tmp_path, monkeypatch, capsys):
    from semcache.experiments.cachegen.shared import core
    monkeypatch.setattr(core, 'cdf_from_counts', lambda *a, **k: pytest.fail('CDF fit'))
    monkeypatch.setattr(c4, 'verify_profile', lambda path: dict(path=str(path),
                        sha256=PROFILE_SHA256, bytes=4108))
    import semcache.models.loader as loader
    monkeypatch.setattr(loader, 'load_model', lambda *a, **k: pytest.fail('model loaded in dry-run'))
    paths = {}
    for dataset in ('snips', 'multiwoz'):
        path = tmp_path/f'{dataset}.jsonl'
        source_id = 'AddToPlaylist:0' if dataset == 'snips' else 'dlg0:0'
        path.write_text(json.dumps(row(dataset, source_id))+'\n')
        paths[dataset] = path
    args = SimpleNamespace(snips=paths['snips'], multiwoz=paths['multiwoz'], profile_path=tmp_path/'profile.bin',
        per_dataset=1, coder_backend='FAST_PY_BITEXACT', max_new_tokens=16, dry_run=True)
    plan = c4.run_quality(args)
    assert len(plan['cases']) == 2
    report = capsys.readouterr().out
    assert 'expected controlled teacher-forced inference evaluations=6' in report
    assert 'one_exact_w3_hit=true' in report
    assert 'total_selected_cases=2; selection_mode=diverse' in report
    assert 'snips cases_per_intent={"AddToPlaylist": 1}' in report
    assert 'multiwoz cases_per_dialogue={"dlg0": 1}' in report
    assert not (tmp_path/'results').exists()


def test_logit_metrics_synthetic_if_torch_available():
    torch = pytest.importorskip('torch')
    full = torch.tensor([[[0., 1., 2.], [2., 1., 0.], [1., 3., 2.],
                          [4., 1., 0.], [0., 4., 1.]]])
    same = c4.logit_comparison(full, full.clone(), 3)
    assert same['position_count'] == 2
    assert same['mean_kl'] == pytest.approx(0, abs=1e-7)
    assert same['top1_agreement'] == same['top5_set_overlap'] == 1.
    assert same['max_absolute_logit_difference'] == 0.
    changed = full.clone()
    changed[0, 4] = torch.tensor([5., 0., 1.])
    metric = c4.logit_comparison(full, changed, 3)
    assert metric['mean_kl'] > 0
    assert metric['top1_agreement'] == .5
    assert metric['max_absolute_logit_difference'] == 5.
    assert c4.observed_suffix_nll(full, [0, 1, 2, 1, 0], 3) > 0


def test_real_cache_hit_codec_and_q_invariants_if_torch_available():
    torch = pytest.importorskip('torch')
    from semcache.cache.cache_entry import CacheEntry
    from semcache.experiments.cachegen.c2.physical_storage import CompressedKVPayload, DecodedCacheEntryView
    case = c4.select_cases([row()], 'snips', 1)[0][0]
    blocks = {layer: tuple(torch.full((1, 3, 4), float(layer+index+1), dtype=torch.float16)
                           for index in range(3)) for layer in range(32)}
    class FakeCodec:
        quantization_device = 'cpu'
        made = decoded = 0
        def make_entry(self, entry_type, cluster_id, token_ids, positions, tensors, storage_device):
            self.made += 1
            size = sum(t.numel()*t.element_size() for triple in tensors.values() for t in triple)
            payload = CompressedKVPayload(b'frame', (32, 3, 4), 'torch.float16', PROFILE_SHA256,
                                          (1, 1, 1, 1), 1)
            return entry_type(cluster_id, token_ids, positions, size,
                              q_tensors={i: tensors[i][0].clone() for i in tensors}, compressed_kv=payload,
                              storage_timings_ms={'quantize_ms': 0., 'encode_ms': 0.})
        def decode_entry(self, resident):
            self.decoded += 1
            tensors = {i: (resident.q_tensors[i], blocks[i][1], blocks[i][2]) for i in blocks}
            return DecodedCacheEntryView(resident, tensors, (), dict(decode_ms=0., dequantize_ms=0.))
    codec = FakeCodec()
    caches = c4._make_caches(case, blocks, codec)
    assert codec.made == 1
    raw = c4._lookup_one(case, caches['RAW_REUSE'][0])
    compressed = c4._lookup_one(case, caches['COMPRESSED_REUSE'][0])
    assert codec.decoded == 1
    assert raw.window == compressed.window and raw.entry.key == compressed.entry.key
    assert len(caches['RAW_REUSE'][0].entries) == len(caches['COMPRESSED_REUSE'][0].entries) == 1
    assert caches['COMPRESSED_REUSE'][1].tensors is None
    assert all(torch.equal(raw.entry.tensors[i][0], compressed.entry.tensors[i][0]) for i in blocks)


def test_direct_quantization_reconstruction_and_symbol_check_if_torch_available():
    torch = pytest.importorskip('torch')
    from semcache.semantic.hit_selection import CacheHit
    from semcache.semantic.subsequence import Subsequence
    case = c4.select_cases([row()], 'snips', 1)[0][0]
    blocks = {layer: tuple(torch.tensor([[[1., -2., 3., -4.]]*3], dtype=torch.float16)
                           for _ in range(3)) for layer in range(32)}
    direct = {role: c4.POLICY.quantize(torch.cat([blocks[i]['qkv'.index(role.lower())]
                                                 for i in range(32)], 0), role)
              for role in ('K', 'V')}
    raw_tensors = blocks
    comp_tensors = {layer: (blocks[layer][0], direct['K'].reconstructed[layer:layer+1],
                             direct['V'].reconstructed[layer:layer+1]) for layer in range(32)}
    window = Subsequence(tuple(case['reused_token_ids']), 0, 3)
    raw = CacheHit(window, SimpleNamespace(key=case['cache_key'], tensors=raw_tensors))
    comp = CacheHit(window, SimpleNamespace(key=case['cache_key'], tensors=comp_tensors,
                                            kv_symbols=(direct['K'].symbols, direct['V'].symbols)))
    resident = SimpleNamespace(tensors=None,
                               compressed_kv=SimpleNamespace(profile_sha256=PROFILE_SHA256))
    c4._verify_codec(case, blocks, raw, comp, resident, 'cpu')
    comp_tensors[0] = (torch.zeros_like(blocks[0][0]), *comp_tensors[0][1:])
    with pytest.raises(ValueError, match='Q differs'): c4._verify_codec(case, blocks, raw, comp, resident, 'cpu')


def test_full_mode_has_no_reuse_if_torch_available(monkeypatch):
    torch = pytest.importorskip('torch')
    import semcache.system_cost.base_projection as projection
    monkeypatch.setattr(projection, 'base_projection_path', lambda *a, **k: pytest.fail('FULL reused cache'))
    class FakeModel:
        config = SimpleNamespace(eos_token_id=None)
        def __call__(self, **kwargs):
            return SimpleNamespace(logits=torch.tensor([[[0., 1.], [1., 0.], [0., 1.],
                                                         [0., 1.], [1., 0.]]]), past_key_values=None)
    logits, generated, timing, captured = c4._forward_and_generate(
        FakeModel(), None, [1, 2, 3, 4, 5], None, 1, 'cpu')
    assert logits.shape == (1, 5, 2)
    assert len(generated) == 1 and captured is None
