"""Measured decomposition and two-user diagnostics, without cache reuse."""
import torch
from semcache.models.model_adapter import OPTModelAdapter
from semcache.models.capture import qkv_capture
from semcache.models.lora_decomposition import decompose_projection, assert_decomposition
from semcache.models.lora_fixtures import base_weight_fingerprint, assert_frozen_base
from semcache.edgelora.lora_projection import edge_qkv_projection, reconstructed_projection_path
from semcache.evaluation.qkv_metrics import similarity
from semcache.evaluation.logit_metrics import compare_logits
from semcache.semantic.probes import construct_probes


def capture_user(model, ids, user, layers):
    model.set_adapter(user)
    model.eval()
    inputs = dict(input_ids=torch.tensor([ids], device=next(model.parameters()).device))
    with qkv_capture(model, layers, storage_device=None) as capture:
        with torch.inference_mode():
            logits = model(**inputs, use_cache=False).logits.detach()
    assert_frozen_base(model)
    return capture.records, logits


def provenance(metadata, fixture):
    return dict(model_id=metadata['model'], resolved_model_revision=metadata.get('resolved_model_revision'),
        dtype=metadata['dtype'], **fixture, metric_source='measured', safe_reuse_claimed=False)


def decomposition_rows(model, records, metadata, fixture, tolerance=1e-6, probe_mode='decomposition'):
    adapter = OPTModelAdapter(model)
    rows = []
    for layer, record in records.items():
        parts, comm = edge_qkv_projection(record['hidden_states'], adapter, layer, fixture['adapter_name'], tolerance)
        for name, p in parts.items():
            metrics = assert_decomposition(p, tolerance)
            bn, dn, tn = [t.double().norm().item() for t in (p.base_output, p.lora_delta, p.combined_output)]
            rows.append(dict(**provenance(metadata, fixture), layer=layer, tensor_type=name.upper(),
                component='total', probe_mode=probe_mode, base_norm=bn, lora_delta_norm=dn, total_norm=tn,
                lora_to_base_norm_ratio=dn/bn if bn else None,
                **{'decomposition_'+k: v for k, v in metrics.items()}, **comm,
                communication_scope='one full layer exchange; repeated across Q/K/V rows, do not sum'))
    return rows


def assert_return_identity(first, returned, first_logits, returned_logits):
    if not torch.equal(first_logits, returned_logits):
        raise AssertionError('user_a logits changed after user_a -> user_b -> user_a')
    for layer in first:
        for name in ('hidden_states', 'q', 'k', 'v'):
            if not torch.equal(first[layer][name], returned[layer][name]):
                raise AssertionError('user_a projection/input changed after adapter switching')


def compare_users(model, records_a, records_b, metadata, fixtures, mode, tolerance=1e-6):
    adapter = OPTModelAdapter(model)
    rows = []
    fixed = mode == 'fixed_hidden_input'
    for layer in records_a:
        by_user = {}
        for user, record in [('user_a', records_a[layer]), ('user_b', records_b[layer])]:
            model.set_adapter(user)
            model.eval()
            hidden = records_a[layer]['hidden_states'] if fixed else record['hidden_states']
            by_user[user] = {}
            for name, module in adapter.projection_modules(layer).items():
                p = decompose_projection(module, hidden, user)
                assert_decomposition(p, tolerance)
                by_user[user][name] = p
        for name in 'qkv':
            a, b = (by_user[u][name] for u in ('user_a', 'user_b'))
            if fixed:
                if not torch.equal(a.base_output, b.base_output):
                    raise AssertionError('Base differs for same hidden input')
                if torch.equal(a.lora_delta, b.lora_delta):
                    raise AssertionError('User deltas are not distinguishable')
            for component, attr in [('base', 'base_output'), ('lora_delta', 'lora_delta'), ('total', 'combined_output')]:
                metric = similarity(getattr(a, attr), getattr(b, attr))
                for user, p in [('user_a', a), ('user_b', b)]:
                    bn, dn, tn = [t.double().norm().item() for t in (p.base_output, p.lora_delta, p.combined_output)]
                    rows.append(dict(**provenance(metadata, fixtures[user]), layer=layer, tensor_type=name.upper(),
                        probe_mode=mode, user_a='user_a', user_b='user_b', component=component, same_hidden_input=fixed,
                        base_norm=bn, lora_delta_norm=dn, total_norm=tn, lora_to_base_norm_ratio=dn/bn if bn else None,
                        **{'decomposition_'+k: v for k, v in assert_decomposition(p, tolerance).items()},
                        **{'cross_user_'+k: v for k, v in metric.items()}))
    return rows


def run_multiuser_probe(model, tokenizer, metadata, fixtures, layers, window_size=3, tolerance=1e-6):
    adapter = OPTModelAdapter(model)
    before = base_weight_fingerprint(adapter)
    probes = construct_probes(tokenizer, window_size)
    ids = probes[0]['input_ids_a']  # same exact complete controlled query for both users
    a, la = capture_user(model, ids, 'user_a', layers)
    b, _ = capture_user(model, ids, 'user_b', layers)
    returned, lr = capture_user(model, ids, 'user_a', layers)
    assert_return_identity(a, returned, la, lr)
    rows = []
    for mode in ('fixed_hidden_input', 'full_user_forward'):
        measured = compare_users(model, a, b, metadata, fixtures, mode, tolerance)
        for row in measured:
            row.update(probe_case='same_query', input_ids_a=ids, input_ids_b=ids,
                       source_start=0, source_end=len(ids), target_start=0, target_end=len(ids))
        rows.extend(measured)
    # Cross-context is observational only: align existing A/B matched windows.
    for probe in probes[:2]:
        ca, _ = capture_user(model, probe['input_ids_a'], 'user_a', layers)
        cb, _ = capture_user(model, probe['input_ids_b'], 'user_b', layers)
        for records, window in [(ca, probe['window_a']), (cb, probe['window_b'])]:
            for record in records.values():
                for name in ('hidden_states', 'q', 'k', 'v'):
                    record[name] = record[name][:, window.start:window.end]
        measured = compare_users(model, ca, cb, metadata, fixtures, 'cross_context_full_user_forward', tolerance)
        for row in measured:
            row.update(probe_case=probe['probe_case'], input_ids_a=probe['input_ids_a'], input_ids_b=probe['input_ids_b'],
                source_start=probe['source_start'], source_end=probe['source_end'],
                target_start=probe['target_start'], target_end=probe['target_end'])
        rows.extend(measured)
    model.set_adapter('user_a')
    model.eval()
    assert_frozen_base(model)
    if before != base_weight_fingerprint(adapter):
        raise AssertionError('Base changed after user switching')
    return rows


def validate_forward(model, ids, user='user_a', layers=None, tolerance=1e-6):
    adapter = OPTModelAdapter(model)
    before = base_weight_fingerprint(adapter)
    model.set_adapter(user)
    model.eval()
    inputs = dict(input_ids=torch.tensor([ids], device=next(model.parameters()).device))
    with torch.inference_mode():
        baseline = model(**inputs, use_cache=False).logits
        with reconstructed_projection_path(adapter, user, layers, tolerance) as seen:
            emulated = model(**inputs, use_cache=False).logits
    metric = compare_logits(baseline, emulated, 0)
    if (metric['max_abs_logit_diff'] > tolerance or metric['relative_l2_logit_diff'] is None
            or metric['relative_l2_logit_diff'] > tolerance
            or abs(metric['affected_suffix_mean_kl']) > tolerance
            or abs(metric['last_position_kl_baseline_to_injected']) > tolerance
            or not metric['last_argmax_agreement']):
        raise AssertionError(f'EdgeLoRA full-forward mismatch: {metric}')
    if before != base_weight_fingerprint(adapter):
        raise AssertionError('Base changed during reconstructed forward')
    return dict(**metric, reconstructed_projections=len(seen), component='total',
                probe_mode='edgelora_reconstructed_forward', safe_reuse_claimed=False)
