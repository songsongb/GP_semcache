"""Synthetic, hash-bound evidence in the actual C7 writer schemas; no model/codec."""
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from semcache.experiments.cachegen import c7b3_q_freeze as freeze
from semcache.experiments.cachegen import c8a_capacity as capacity


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n')


def csv_rows(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in row.items()})


def bind(root, manifest):
    manifest['output_hashes'] = {str(p.relative_to(root)): freeze.sha(p) for p in root.rglob('*')
                               if p.is_file() and p != root/'manifest.json'}
    put(root/'manifest.json', manifest)


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    q, b2, b2a, plan = [tmp_path/name for name in ('q', 'b2', 'b2a', 'plan')]
    for root in (q, b2, b2a, plan): root.mkdir()
    (q/'profiles').mkdir(); (q/'profiles/q24.bin').write_bytes(b'literal synthetic frozen Q24 profile')
    kv = tmp_path/'kv.bin'; kv.write_bytes(b'literal synthetic frozen K20 V16 profile')
    monkeypatch.setattr(freeze, 'Q_SHA', freeze.sha(q/'profiles/q24.bin'))
    monkeypatch.setattr(freeze, 'KV_SHA', freeze.sha(kv))
    episodes = [dict(episode_id=f'episode-{i}', history_depth=i//8+1, cluster=i%3,
        token_ids=[10+i, 50+i, 90+i], selected_hit_count=1, source_id=f'source-{i}', target_id=f'target-{i}',
        source_start=3, target_start=4, source_user='user_a', target_user='user_b') for i in range(32)]
    for e in episodes: e['cache_key'] = [e['cluster'], e['token_ids']]
    put(plan/'evaluation_selection.json', dict(episodes=episodes, selection_sha256=freeze.digest(episodes)))
    monkeypatch.setattr(freeze, 'SELECTION_SHA', freeze.sha(plan/'evaluation_selection.json'))
    plan_m = dict(selection_version='current_user_content_v2', history_k=4, physical_safety_contract=freeze.CONTRACT)
    bind(plan, plan_m)
    b1_m = dict(stage='C7-B1', status='COMPLETE', selected_q_candidate=None, q_profile_frozen=False,
        runtime_profile_fit_count_on_candidate_select=0, q_candidates=[16, 20, 24, 32], fit_blocks=96,
        candidate_select_blocks=32, fit_select_conversation_overlap_count=0, frozen32_overlap_count=0,
        capability64_overlap_count=0, existing_kv_profile_modified=False,
        transform_provenance=dict(transform=freeze.TRANSFORM, role_generic_verified=True),
        profile_hashes={'Q24': freeze.Q_SHA, 'Q32': 'c'*64}, existing_kv_profile=dict(sha256=freeze.KV_SHA))
    namespace = dict(model='synthetic-model', model_revision='synthetic-revision', tokenizer_revision='synthetic-revision',
                     adapter_hashes={'user_a': 'a'*64, 'user_b': 'b'*64})
    b1_m.update(namespace)
    put(q/'candidate_summary.json', dict(candidates=[dict(candidate='Q'+str(b), q_resident_compression_ratio=r)
        for b, r in ((16, 8.), (20, 7.), (24, 6.), (32, 5.))]))
    modes = (freeze.BASELINE, freeze.Q24, freeze.Q32)
    summaries, pairs, rows, accounts = [], [], [], {}
    for mode in modes:
        stored = []
        for i, e in enumerate(episodes):
            kv_frame = 90_000+193*i
            q_frame = 0 if mode == freeze.BASELINE else (60_000 if mode == freeze.Q24 else 75_000)+127*i
            raw_q = capacity.RAW_ROLE_BYTES if mode == freeze.BASELINE else 0
            account = dict(raw_q_bytes=capacity.RAW_ROLE_BYTES, raw_k_bytes=capacity.RAW_ROLE_BYTES,
                raw_v_bytes=capacity.RAW_ROLE_BYTES, raw_qkv_bytes=capacity.RAW_ENTRY_BYTES,
                resident_raw_q_bytes=raw_q, compressed_q_bitstream_bytes=max(0, q_frame-512),
                local_q_metadata_bytes=512 if q_frame else 0, compressed_q_frame_bytes=q_frame,
                compressed_kv_frame_bytes=kv_frame, local_kv_metadata_bytes=1024,
                total_resident_qkv_bytes=raw_q+q_frame+kv_frame,
                incremental_resident_byte_reduction_vs_kv_baseline=capacity.RAW_ROLE_BYTES-raw_q-q_frame,
                raw_q_resident_after_insert=mode == freeze.BASELINE, shared_profile_bytes_charged_per_entry=0)
            rows.append(dict(e, episode_index=i, mode=mode, logical_event_hash=freeze.digest(e),
                generated_text='response '+str(i), generated_token_ids=[100+i, 2], storage_accounting=account))
            stored.append(account)
            if mode != freeze.BASELINE:
                pairs.append(dict(episode_id=e['episode_id'], mode=mode, baseline_mode=freeze.BASELINE,
                                  generation_fidelity=dict(exact_generation=True)))
        total = sum(a['total_resident_qkv_bytes'] for a in stored)
        accounts[mode] = dict(totals={k: sum(a[k] for a in stored) for k in stored[0] if k != 'raw_q_resident_after_insert'},
            total_resident_bytes=total, mean_resident_bytes=total/32,
            whole_qkv_compression_ratio=32*capacity.RAW_ENTRY_BYTES/total, shared_profile_bytes_charged_per_entry=0)
        summaries.append(dict(mode=mode, cases=32, corpus_bleu=dict(value=3.1478254770301533)))
    csv_rows(b2/'per_case.csv', rows)
    put(b2/'summary.json', dict(modes=summaries))
    put(b2/'paired_quality.json', dict(per_case=pairs, aggregates=[dict(mode=mode, baseline_mode=freeze.BASELINE,
        cases=32, exact_generation_match_count=32) for mode in modes[1:]]))
    put(b2/'storage_accounting.json', dict(modes=accounts,
        shared_profile_bytes={'Q24': (q/'profiles/q24.bin').stat().st_size, 'Q32': 50, 'KV': kv.stat().st_size}))
    b2_m = dict(stage='C7-B2', status='COMPLETE', baseline_matches_c6b3_2=True, runtime_q_profile_fit_count=0,
        runtime_storage_kv_cdf_fit_count=0, transport_compression_enabled=False, kv_profile_sha256=freeze.KV_SHA,
        q_profile_hashes={'Q24': freeze.Q_SHA, 'Q32': 'c'*64}, frozen32_selection_sha256=freeze.SELECTION_SHA,
        physical_safety_contract=freeze.CONTRACT, cached_payload='TOTAL_QKV',
        teacher_forced_continuation_provenance=dict(plan_manifest_sha256=freeze.sha(plan/'manifest.json')))
    b2_m.update(namespace)
    chosen = [dict(episodes[i], audit_case_index=j, canonical_episode_index=i) for j, i in enumerate((0, 8, 16, 24))]
    put(b2a/'selected_cases.json', dict(status='COMPLETE', frozen32_selection_sha256=freeze.SELECTION_SHA, cases=chosen))
    reports = {k: dict(status='COMPLETE', cases=[]) for k in
               ('q_distortion', 'injection_audit', 'internal_causal_effect', 'continuation_causal_effect')}
    for case in chosen:
        episode_id = case['episode_id']
        reports['q_distortion']['cases'].append(dict(episode_id=episode_id, candidates={candidate:
            [dict(layer=l, exact_equal=False, mse=0.001, raw_sha256='a'*64, comparison_sha256='b'*64) for l in range(32)]
            for candidate in ('Q24', 'Q32')}))
        injected = {mode: dict(hit_mask_equal=True, q_injection_verified=True, fresh_q_rows_equal=True,
            k_rows_equal=True, v_rows_equal=True, raw_q_resident_after_insert=mode == 'KV_BASELINE', hit_positions=[4, 5, 6],
            per_layer=[dict(layer=l, q_injection_verified=True, k_injection_verified=True, v_injection_verified=True,
                           fresh_q_rows_equal=True, k_rows_equal=True, v_rows_equal=True) for l in range(32)])
            for mode in ('KV_BASELINE', 'Q24', 'Q32', 'Q_ZERO_COUNTERFACTUAL')}
        reports['injection_audit']['cases'].append(dict(episode_id=episode_id, modes=injected,
            q24_injection_verified=True, q32_injection_verified=True, qzero_injection_verified=True,
            hit_mask_equal=True, fresh_q_rows_equal=True, k_rows_equal=True, v_rows_equal=True))
        reports['internal_causal_effect']['cases'].append(dict(episode_id=episode_id, hit_row_causal_effect_verified=True,
            fresh_attention_numerically_negligible=True, per_layer=[dict(layer=l, hit_attention=dict(max_absolute_error=0.1),
            hit_post_layer_hidden=dict(max_absolute_error=0.2), fresh_attention_max_abs_diff=0.) for l in range(32)]))
        reports['continuation_causal_effect']['cases'].append(dict(episode_id=episode_id, causal_isolation_supported=True,
            comparisons={mode: dict(numerically_negligible=True, max_abs_logit_difference=0., mean_abs_logit_difference=0.,
                top1_agreement=1., top5_agreement=1.) for mode in ('Q24', 'Q32', 'Q_ZERO_COUNTERFACTUAL')}))
    for name, data in reports.items(): put(b2a/(name+'.json'), data)
    questions = ('Decoded Q24/Q32 different from raw TOTAL Q in every case?',
        'Decoded tensors actually injected into the selected HIT rows?', 'Zeroing cached Q changed hit-row attention/internal states?',
        'Fresh-row attention remained numerically unchanged?', 'Canonical teacher-forced continuation remained numerically unchanged?',
        'Compressed Q is genuinely consumed but causally isolated here?')
    (b2a/'summary.md').write_text('\n'.join(f'{i}. {question} True' for i, question in enumerate(questions, 1)))
    b2a_m = dict(stage='C7-B2A', status='COMPLETE', audit_cases=4, causal_isolation_supported=True,
        runtime_q_profile_fit_count=0, runtime_storage_kv_cdf_fit_count=0, transport_encode_calls=0, transport_decode_calls=0,
        kv_profile_sha256=freeze.KV_SHA, q_profile_hashes={'Q24': freeze.Q_SHA, 'Q32': 'c'*64},
        frozen32_selection_sha256=freeze.SELECTION_SHA, physical_safety_contract=freeze.CONTRACT, cached_payload='TOTAL_QKV',
        tolerances=dict(logit_max_abs=1e-6, logit_mean_abs=1e-7, fresh_attention_max_abs=1e-6))
    b2a_m.update(namespace)
    def refresh():
        bind(q, b1_m); monkeypatch.setattr(freeze, 'B1_SHA', freeze.sha(q/'manifest.json'))
        b2_m['c7b1_manifest_sha256'] = freeze.B1_SHA
        bind(b2, b2_m); monkeypatch.setattr(freeze, 'B2_SHA', freeze.sha(b2/'manifest.json'))
        b2a_m['c7b2_manifest_sha256'] = freeze.B2_SHA
        bind(b2a, b2a_m)
    refresh()
    args = SimpleNamespace(q_calibration_root=q, c7b2_root=b2, c7b2a_root=b2a, kv_profile=kv, output_root=tmp_path/'freeze')
    return SimpleNamespace(args=args, q=q, b2=b2, b2a=b2a, plan=plan, b1_m=b1_m, b2_m=b2_m, b2a_m=b2a_m,
        refresh=refresh, rows=rows, episodes=episodes, tmp=tmp_path)


def test_manual_freeze_schema_and_exact_hashes(evidence):
    e = evidence
    before = {str(p): freeze.sha(p) for p in e.tmp.rglob('*') if p.is_file()}
    result = freeze.freeze(e.args)
    assert result['stage'] == 'C7-B3' and result['status'] == 'FROZEN'
    assert result['manual_decision'] == 'SELECT_Q24' and result['selected_q_candidate'] == 'Q24'
    assert (result['q_bins'], result['k_bins'], result['v_bins']) == (24, 20, 16)
    assert result['frozen32_used_for_selection'] and result['q_profile_frozen']
    assert not result['additional_q_fitting_after_freeze'] and not result['additional_kv_fitting_after_freeze']
    assert result['q_profile_sha256'] == freeze.sha(e.q/'profiles/q24.bin')
    assert result['b2a_manifest_sha256'] == freeze.sha(e.b2a/'manifest.json')
    assert result['q24_quality_evidence'] == dict(delta_bleu_vs_kv_baseline=0., exact_generation_match_count=32, cases=32)
    assert all(result['causal_audit'].values())
    assert freeze.validate_freeze(e.args.output_root/'freeze_decision.json')['manual_decision'] == 'SELECT_Q24'
    assert before == {p: freeze.sha(p) for p in before}
    with pytest.raises(ValueError, match='overwrite'): freeze.freeze(e.args)
    assert result == freeze.decision(e.q, e.b2, e.b2a, e.args.kv_profile)


@pytest.mark.parametrize('artifact', ('q24', 'b1', 'b2', 'b2a_output'))
def test_hash_mismatch_fails_closed(evidence, artifact):
    e = evidence
    path = {'q24': e.q/'profiles/q24.bin', 'b1': e.q/'manifest.json', 'b2': e.b2/'manifest.json',
            'b2a_output': e.b2a/'injection_audit.json'}[artifact]
    path.write_bytes(path.read_bytes()+b' ')
    with pytest.raises(ValueError, match='Hash mismatch'): freeze.freeze(e.args)
    assert not e.args.output_root.exists()


@pytest.mark.parametrize('stage,key,value', [('b1_m', 'runtime_profile_fit_count_on_candidate_select', 1),
    ('b2_m', 'baseline_matches_c6b3_2', False), ('b2a_m', 'causal_isolation_supported', False),
    ('b2a_m', 'transport_encode_calls', 1), ('b2a_m', 'status', 'INVALID')])
def test_manifest_claims_verified_not_assumed(evidence, stage, key, value):
    e = evidence; getattr(e, stage)[key] = value; e.refresh()
    with pytest.raises(ValueError): freeze.freeze(e.args)


@pytest.mark.parametrize('damage', ('delta', 'generation', 'injection', 'zero_effect', 'summary', 'decoded_identical'))
def test_actual_quality_and_causal_evidence_required(evidence, damage):
    e = evidence
    if damage == 'delta':
        path = e.b2/'summary.json'; value = freeze.read(path); value['modes'][1]['corpus_bleu']['value'] += 0.1; put(path, value)
    elif damage == 'generation':
        e.rows[32]['generated_text'] = 'changed'; csv_rows(e.b2/'per_case.csv', e.rows)
    elif damage == 'summary':
        (e.b2a/'summary.md').write_text('NOT_COMPLETED')
    else:
        name = {'injection': 'injection_audit', 'zero_effect': 'internal_causal_effect', 'decoded_identical': 'q_distortion'}[damage]
        path = e.b2a/(name+'.json'); value = freeze.read(path); first = value['cases'][0]
        if damage == 'injection': first['modes']['Q24']['q_injection_verified'] = False
        elif damage == 'zero_effect':
            for r in first['per_layer']:
                r['hit_attention']['max_absolute_error'] = r['hit_post_layer_hidden']['max_absolute_error'] = 0.
        else:
            for r in first['candidates']['Q24']: r['exact_equal'] = True
        put(path, value)
    e.refresh()
    with pytest.raises(ValueError): freeze.freeze(e.args)


def test_freeze_cannot_authorize_fitting_or_profile_change(evidence):
    e = evidence; freeze.freeze(e.args); path = e.args.output_root/'freeze_decision.json'
    saved = freeze.read(path); saved['additional_q_fitting_after_freeze'] = True; put(path, saved)
    with pytest.raises(ValueError): freeze.validate_freeze(path)
    saved['additional_q_fitting_after_freeze'] = False; saved['q_profile_sha256'] = 'wrong'; put(path, saved)
    with pytest.raises(ValueError): freeze.validate_freeze(path)
