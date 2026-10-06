"""Synthetic C9-B manifests/adapter headers only; no tensors, models or codecs."""
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from semcache.experiments.cachegen import c9b_network as network
from semcache.experiments.cachegen import c9b_transport as transport
from semcache.experiments.cachegen import c9a_latency as local
from semcache.experiments.cachegen import c8a_capacity as capacity
from semcache.experiments.cachegen import c7b3_q_freeze as provenance
from test_cachegen_c9a_latency import upstream, replay, evidence
from test_cachegen_c7b3_q_freeze import put, csv_rows


def header_file(path, dtype='F32'):
    path.parent.mkdir(parents=True)
    header, offset = {'__metadata__': dict(user=path.parent.name)}, 0
    for l in range(32):
        for role in 'qkv':
            for matrix in 'AB':
                size = 8*2560*4
                name = f'base_model.model.model.decoder.layers.{l}.self_attn.{role}_proj.lora_{matrix}.weight'
                header[name] = dict(dtype=dtype, shape=[8, 2560] if matrix == 'A' else [2560, 8], data_offsets=[offset, offset+size])
                offset += size
    encoded = json.dumps(header).encode()
    with path.open('wb') as stream:
        stream.write(len(encoded).to_bytes(8, 'little')); stream.write(encoded)
        stream.seek(offset-1, 1); stream.write(b'\0')  # Sparse dummy weights; never loaded.
    return header


@pytest.fixture
def measured(upstream):
    e, old_args, _, old_m = upstream
    adapters = {u: e.tmp/'adapters'/u/'adapter_model.safetensors' for u in ('user_a', 'user_b')}
    for p in adapters.values(): header_file(p)
    hashes = {u: provenance.sha(p) for u, p in adapters.items()}
    # Bind new synthetic model namespace through the full freeze/C8 chain.
    for m in (e.b1_m, e.b2_m, e.b2a_m): m['adapter_hashes'] = hashes
    e.refresh()
    fargs = SimpleNamespace(**vars(e.args)); fargs.output_root = e.tmp/'frozen-network'
    provenance.freeze(fargs)
    args = SimpleNamespace(freeze_decision=fargs.output_root/'freeze_decision.json', plan_dir=e.plan,
        c8a_root=e.tmp/'capacity-network', c8b_root=e.tmp/'quality-network',
        c9a_root=e.tmp/'measured', output_root=e.tmp/'network-output')
    aargs = SimpleNamespace(**vars(args)); aargs.output_root = args.c8a_root; aargs.c7b2_root = e.b2
    capacity.run(aargs)
    prepared = network.capacity.verify_capacity(args)
    semantic = e.tmp/'semantic.jsonl'
    workloads = []
    for i, ep in enumerate(prepared.episodes):
        ids = list(range(20+i%7)); ids[ep['target_start']:ep['target_start']+3] = ep['token_ids']
        workloads.append(dict(source_id=ep['target_id'], prompt_version='c6b3_multiwoz_history_v1', token_ids=ids))
    semantic.write_text(''.join(json.dumps(r)+'\n' for r in workloads))
    official = e.tmp/'full'/'capability_per_case.csv'; official.parent.mkdir()
    csv_rows(official, [dict(source_id=ep['target_id'], user=ep['target_user'], history_depth=ep['history_depth'],
        generated_token_ids=[100, 2]) for ep in prepared.episodes])
    c6 = e.tmp/'c6'; c6.mkdir()
    c6_rows = []
    for mode in ('RAW_SEMCACHE', 'STORAGE_KV_COMP'):
        for ep, workload in zip(prepared.episodes, workloads):
            role_bytes = (len(workload['token_ids'])+1-3)*32*2560*4
            c6_rows.append(dict(episode_id=ep['episode_id'], mode=mode, logical_event_hash=provenance.digest(ep),
                diagnostic_transport_accounting=dict(**{'raw_'+r+'_delta_bytes': role_bytes for r in 'qkv'},
                    transmitted_total_including_cdf_bytes=role_bytes*3)))
    csv_rows(c6/'per_case.csv', c6_rows)
    c6_m = dict(stage='C6-B3-2', status='COMPLETE', revision=e.b1_m['model_revision'], adapter_hashes=hashes,
        evaluation_selection_sha256=provenance.SELECTION_SHA, plan_manifest_sha256=provenance.sha(e.plan/'manifest.json'),
        semantic_workload_sha256=provenance.sha(semantic), output_hashes={'per_case.csv': provenance.sha(c6/'per_case.csv')})
    put(c6/'manifest.json', c6_m)
    inputs = dict(prepared.input_hashes)
    for p in [*adapters.values(), semantic, official, c6/'manifest.json', c6/'per_case.csv']: inputs[str(p.resolve())] = provenance.sha(p)
    m = json.loads(json.dumps(old_m))
    m.update(adapter_hashes=hashes, c7b3_freeze_decision_sha256=provenance.sha(args.freeze_decision), input_hashes=inputs,
        canonical_full_provenance=dict(semantic_workload_sha256=provenance.sha(semantic), adapter_hashes=hashes),
        teacher_forced_canonical_source=dict(per_case_path=str(official.resolve()), per_case_sha256=provenance.sha(official)))
    for name in ('manifest.json', 'summary.json', 'residency_trace.json', 'per_event.csv'):
        m['c8a_'+name.rsplit('.', 1)[0]+'_sha256'] = provenance.sha(args.c8a_root/name)
    args.c8b_root.mkdir()
    data = provenance.read(old_args.c8b_root/'hit_audit.json'); put(args.c8b_root/'hit_audit.json', data)
    m['output_hashes'] = {'hit_audit.json': provenance.sha(args.c8b_root/'hit_audit.json')}
    put(args.c8b_root/'manifest.json', m)
    local.verify_c8b(args, prepared)
    raw = []
    for name in local.CONDITIONS:
        for index, ep in enumerate(prepared.episodes):
            event = None if name == local.REFERENCE else prepared.expected[name]['lookups'][index]
            hit = bool(event and event['hit'])
            for repeat in range(3):
                decode = float((20000 if name.endswith('_Q24_KV_COMP') else 10000)+repeat) if hit and not name.endswith('_RAW_QKV') else 0.
                lookup = 0. if event is None else .1; model = float(10+repeat)
                raw.append(dict(condition=name, episode_id=ep['episode_id'], episode_index=index, repeat=repeat,
                    hit=hit, retained_source_episode_id=event['resident_source_episode_id'] if event else None,
                    prompt_tokens=len(workloads[index]['token_ids']), native_projection_rows_skipped_per_role_per_layer=3 if hit else 0,
                    lookup_ms=lookup, storage_decode_ms=decode, model_forward_ms=model, projection_or_mixed_projection_ms=1.,
                    control_overhead_ms=.01, target_total_ms=lookup+decode+model+.01))
    source = [dict(condition=name, source_forward_ms=1., source_capture_ms=.1, storage_encode_ms=float(1 if name.endswith('_RAW_QKV') else 10 if name.endswith('_KV_COMP') and not name.endswith('_Q24_KV_COMP') else 20),
        cache_admission_ms=.01) for name in local.CONDITIONS[1:] for _ in range(32)]
    a = dict(stage='C9-A', status='COMPLETE', recommendation='C9_A_READY_FOR_NETWORK_ACCOUNTING',
        c8_hit_vectors_reproduced=True, latency_decomposition_consistent=True, conditions=list(local.CONDITIONS),
        budgets_raw_entry_equivalent=[2, 8], policies=list(local.POLICIES), reference_mode=local.REFERENCE,
        repeats_per_target=3, warmup_forwards=5, dtype='float16',
        physical_safety_contract=provenance.CONTRACT, cached_payload='TOTAL_QKV',
        model_inference_performed=True, training_performed=False, quality_evaluated_in_c9=False, network_latency_evaluated=False,
        transport_compression_enabled=False, source_build_cost_in_primary_target_latency=False, timing_method=local.TIMING_METHOD,
        cuda_synchronization_policy='synthetic synchronized evidence', c7b3_freeze_sha256=provenance.sha(args.freeze_decision),
        c8b_manifest_sha256=provenance.sha(args.c8b_root/'manifest.json'), q24_profile_sha256=provenance.Q_SHA,
        kv_profile_sha256=provenance.KV_SHA, frozen32_selection_sha256=provenance.SELECTION_SHA,
        **prepared.frozen['model_namespace'], **{k: 0 for k in network.capacity.SAFETY},
        c8a_artifact_hashes={name: provenance.sha(args.c8a_root/name) for name in ('manifest.json', 'summary.json', 'residency_trace.json', 'per_event.csv')},
        input_hashes=dict(prepared.input_hashes), counters=local.expected_execution_counts(prepared.expected), tie_tolerance_ms=1e-5)
    args.c9a_root.mkdir(); local.write_reports(args.c9a_root, raw, source, prepared, a)
    return e, args


def test_actual_c6_accounting_and_raw_delta_not_resident_bytes():
    c6 = transport.C6Accounting()
    projections = [dict(layer=l, role=r, out_features=2560, element_size=4) for l in range(32) for r in 'qkv']
    roles, total, fresh = c6.bytes(9, 3, projections, 'user_a')
    assert fresh == 6 and all(v == 32*6*2560*4 for v in roles.values())
    assert total == 32*6*2560*4*3
    assert c6.bytes(9, 0, projections, 'user_a')[1]-total == 2*capacity.RAW_ENTRY_BYTES
    assert c6.bytes(3, 3, projections, 'user_a')[1] == 0
    assert 'module.lora_B[user].weight.element_size()' in c6.provenance['raw_bytes_expression']
    assert c6.provenance['network_transaction_count'] is None
    half = [dict(p, element_size=2) for p in projections]
    assert c6.bytes(9, 3, half, 'user_a')[1] == total//2  #Math is role/dtype generic, frozen headers independently checked.


def test_shape_headers_read_only_and_fp32_dtype_required(tmp_path):
    path = tmp_path/'user_a'/'adapter_model.safetensors'; header_file(path)
    before = provenance.sha(path); shapes = transport.adapter_header(path)
    assert len(shapes) == 96 and all(p['out_features'] == 2560 and p['element_size'] == 4 for p in shapes)
    assert provenance.sha(path) == before
    other = tmp_path/'user_b'/'adapter_model.safetensors'; header_file(other, 'F16')
    with pytest.raises(ValueError, match='Unproven runtime LoRA delta dtype'): transport.adapter_header(other)


def test_hash_bound_c9_medians_and_c6_transport_crosscheck(measured, monkeypatch):
    _, args = measured
    monkeypatch.setattr(local, 'prepare', lambda *a: pytest.fail('No GPU/model preparation'))
    monkeypatch.setattr(network.capacity, 'prepare', lambda *a: pytest.fail('No quality preparation'))
    p = network.prepare(args)
    assert len(p.local_cases) == len(p.transport) == 224
    assert p.transport_summary['cross_checks']['status'] == 'VERIFIED'
    assert len(p.transport_summary['cross_checks']['checks']) == 64
    for name, count in zip(local.CONDITIONS, (0, 2, 4, 13, 8, 18, 32)):
        assert sum(r['hit'] for r in p.transport if r['condition'] == name) == count
    full = {r['episode_id']: r for r in p.transport if r['condition'] == local.REFERENCE}
    assert len({r['total_transport_bytes'] for r in full.values()}) > 1
    for row in p.transport:
        assert row['bytes_saved_vs_FULL'] == (2*capacity.RAW_ENTRY_BYTES if row['hit'] else 0)
        assert row['total_transport_bytes'] == sum(row[r+'_transport_bytes'] for r in 'qkv')


@pytest.mark.parametrize('damage', ('hash', 'manifest', 'fit', 'vector', 'median', 'c6_bytes', 'dtype', 'source_hash'))
def test_provenance_or_accounting_mismatch_fails_closed(measured, damage):
    _, args = measured
    m = provenance.read(args.c9a_root/'manifest.json')
    if damage in ('manifest', 'fit'):
        m['status' if damage == 'manifest' else 'runtime_q_profile_fit_count'] = 'INVALID' if damage == 'manifest' else 1
    elif damage == 'hash': (args.c9a_root/'target_latency_raw.csv').write_text('tampered')
    elif damage in ('vector', 'median'):
        path = args.c9a_root/('target_latency_raw.csv' if damage == 'vector' else 'target_latency_per_case.csv')
        rows = provenance.read_csv(path)
        rows[0]['hit' if damage == 'vector' else 'target_total_ms'] = 'True' if damage == 'vector' else '100000'
        csv_rows(path, rows); m['output_hashes'][path.name] = provenance.sha(path)
    else:
        suffix = 'c6/per_case.csv' if damage == 'c6_bytes' else 'semantic.jsonl' if damage == 'source_hash' else 'user_a/adapter_model.safetensors'
        path = next(Path(path) for path in m['input_hashes'] if path.endswith(suffix))
        with path.open('ab') as stream: stream.write(b'changed')
    put(args.c9a_root/'manifest.json', m)
    with pytest.raises(ValueError): network.prepare(args)


def test_payload_formula_and_exact_bounded_points():
    assert network.BANDWIDTHS == (10, 50, 100, 500, 1000)
    assert network.SPEEDUPS == (1, 10, 50, 100, 500, 1000)
    assert network.REUSE_COUNTS == (1, 8, 32)
    assert network.transfer_ms(125000, 1000) == 1
    assert network.transfer_ms(125000, 10) == 100
    for b, bw in ((-1, 10), (1, 0), (float('nan'), 10)):
        with pytest.raises(ValueError): network.transfer_ms(b, bw)


@pytest.mark.parametrize('dl,db,root,region', [(10., -125000, 100., 'BELOW_BREAK_EVEN'),
    (-10., 125000, 100., 'ABOVE_BREAK_EVEN'), (0., -1, None, 'ALL_POSITIVE_BANDWIDTHS'),
    (1., 0, None, 'NONE'), (-1., -1, None, 'ALL_POSITIVE_BANDWIDTHS'), (0., 0, None, None)])
def test_exact_network_root_and_undefined_reasons(dl, db, root, region):
    r = network.network_root(dl, db)
    assert r['break_even_bandwidth_mbps'] == root and r['winning_bandwidth_region'] == region
    assert r['reason']
    if root: assert dl+.008*db/root == pytest.approx(0)


def test_codec_acceleration_and_exact_solve_including_floor():
    assert network.accelerated_local(110, 100, 10) == 20
    assert network.codec_root(-10, 100)['required_speedup'] == 10
    assert network.codec_root(-100, 100)['required_speedup'] == 1
    assert network.codec_root(0, 100)['status'] == 'NO_FINITE_CODEC_SPEEDUP_CAN_BREAK_EVEN'
    assert network.codec_root(1, 100)['required_speedup'] is None
    assert network.codec_root(-1, 0)['required_speedup'] == 1
    #Equal S on both paths uses DECODE DIFFERENCE, not Q24's entire decode.
    assert network.codec_root(-10, 200-100)['required_speedup'] == 10
    assert network.codec_root(-10, 200)['required_speedup'] == 20  #Asymmetric secondary scenario.
    inverse = network.codec_root(5, -10)
    assert inverse['required_speedup'] == 1 and inverse['maximum_winning_speedup'] == 2
    assert network.codec_root(11, -10)['required_speedup'] is None
    for total, decode, factor in ((10, 11, 1), (10, 1, 0), (float('inf'), 1, 2)):
        with pytest.raises(ValueError): network.accelerated_local(total, decode, factor)


@pytest.mark.parametrize('variant', ('contradictory', 'absent'))
def test_bound_c6_transport_semantics_not_just_file_hashes(measured, variant):
    _, args = measured
    p = network.prepare(args)
    manifest_path = next(Path(path) for path in p.input_hashes if path.endswith('c6/manifest.json'))
    manifest = provenance.read(manifest_path)
    if variant == 'absent':
        del p.input_hashes[str(manifest_path)]
    else:
        # Even a consistently rehashed artifact cannot change raw transport
        # shapes. This checks semantic contradiction, not only file damage.
        path = manifest_path.parent/'per_case.csv'
        rows = provenance.read_csv(path)
        recorded = json.loads(rows[0]['diagnostic_transport_accounting'])
        recorded['raw_q_delta_bytes'] += 4
        rows[0]['diagnostic_transport_accounting'] = recorded
        csv_rows(path, rows)
        manifest['output_hashes'][path.name] = provenance.sha(path)
        put(manifest_path, manifest)
        p.input_hashes[str(path)] = provenance.sha(path)
        p.input_hashes[str(manifest_path)] = provenance.sha(manifest_path)
    accounting = transport.C6Accounting()
    lengths = {r['episode_id']: r['prompt_rows'] for r in p.transport if r['condition'] == local.REFERENCE}
    headers = p.transport_summary['per_layer_shapes']
    if variant == 'contradictory':
        with pytest.raises(ValueError, match='C6 raw transport accounting disagrees'):
            transport.cross_check_c6(p, lengths, headers, accounting)
    else:
        result = transport.cross_check_c6(p, lengths, headers, accounting)
        assert result['status'] == 'NOT_AVAILABLE' and result['reason']


@pytest.mark.parametrize('outcome', ('SYSTEM_COMPETITIVE', 'NETWORK_CONDITIONAL', 'NOT_LATENCY_COMPETITIVE'))
def test_predeclared_classification_and_no_candidate_freeze(outcome):
    pairs = []
    for _, before, after in network.comparisons():
        for bandwidth in network.BANDWIDTHS:
            delta = -1 if outcome == 'SYSTEM_COMPETITIVE' or (outcome == 'NETWORK_CONDITIONAL' and bandwidth == 10) else 1
            pairs.append(dict(before=before, after=after, bandwidth_mbps=bandwidth, mean_delta_e2e_ms=delta))
    codecs = dict(comparator_unchanged=[], same_speedup_q24_vs_kv=[])
    result = network.classify(pairs, codecs, 1e-5)
    assert result['recommendation'] == 'CURRENT_PY_CODEC_'+outcome
    assert 'selected_policy' not in result and 'freeze_decision' not in result


def test_per_target_before_aggregation_bandwidth_codec_and_amortization(measured):
    _, args = measured; p = network.prepare(args); cases = network.analyzed_cases(p)
    sweep, summaries, pairs, roots = network.bandwidth_analysis(cases, p.tie_tolerance_ms)
    assert len(sweep) == 224*5 and len(summaries) == 7*5 and len(pairs) == 12*5
    assert all(r['analytical_e2e_ms'] == r['measured_local_ms']+r['analytical_transport_ms'] for r in sweep)
    assert len(roots) == 12 and all(len(r['per_target']) == 32 for r in roots)
    codec_sweep, solved = network.codec_analysis(cases, p.tie_tolerance_ms)
    assert len(codec_sweep) == 4*5*6 and len(solved['same_speedup_q24_vs_kv']) == 2*5
    assert all(r['model'] == 'SAME_UNIFORM_SPEEDUP_BOTH_DECODE_PATHS' for r in solved['same_speedup_q24_vs_kv'])
    amortized = network.source_amortization(p.source)
    inc = next(r for r in amortized['comparisons'] if r['before'] == 'B2_KV_COMP')
    assert inc['incremental_mean_storage_encode_ms'] == 10
    assert inc['amortized_per_request_ms'] == {'1': 10., '8': 1.25, '32': .3125}
    assert amortized['source_cost_in_primary_steady_state'] is False
    classification = network.classify(pairs, solved, p.tie_tolerance_ms)
    assert classification['recommendation'] == 'CURRENT_PY_CODEC_NOT_LATENCY_COMPETITIVE'
    assert classification['native_backend_required'] is True


def test_complete_cpu_outputs_labels_no_freeze_or_historical_changes(measured):
    e, args = measured
    before = {str(path): provenance.sha(path) for path in e.tmp.rglob('*') if path.is_file()}
    p = network.prepare(args); m = network.run(args, p)
    assert m['stage'] == 'C9-B' and m['status'] == 'COMPLETE'
    for flag in ('distributed_network_measured', 'transport_compression_enabled', 'model_inference_performed',
        'quality_evaluated', 'training_performed', 'system_policy_frozen'): assert m[flag] is False
    assert m['runtime_profile_fit_count'] == 0
    assert len(m['output_hashes']) == 10
    assert all(provenance.sha(path) == sha for path, sha in before.items())
    assert 'HYPOTHETICAL_ACCELERATION' in (args.output_root/'summary.md').read_text()
    assert not (args.output_root/'freeze_decision.json').exists()
    with pytest.raises(ValueError, match='Refusing nonempty'): network.run(args, p)


def test_cli_and_import_require_no_gpu_model_or_codec_dependencies():
    repo = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONPATH=str(repo/'src'), CUDA_VISIBLE_DEVICES='')
    code = 'import sys; from semcache.experiments.cachegen.c9b_transport import C6Accounting; C6Accounting(); assert not any(n in sys.modules for n in ("torch","peft","transformers","sacrebleu","torchac_cuda"))'
    subprocess.run([sys.executable, '-c', code], env=env, check=True)
    result = subprocess.run([sys.executable, str(repo/'scripts/68_run_cachegen_c9b_network_break_even.py'), '--help'], env=env, capture_output=True, text=True)
    assert result.returncode == 0 and '--c9a-root' in result.stdout
