"""Exact identity control is mandatory before any approximate trace."""
from semcache.models.capture import qkv_capture
from semcache.cache.cache_entry import CacheEntry
from semcache.cache.global_cache import GlobalCache
from semcache.semantic.subsequence import Subsequence
from semcache.semantic.hit_selection import CacheHit
from semcache.edgelora.mixed_projection import mixed_projection_path
from .logit_metrics import compare_logits
from .qkv_metrics import similarity


CONTROL_TEXT = 'I need clear health advice about sleep and exercise today please.'


def validate_mixed_control(model, tokenizer, adapter, text=CONTROL_TEXT):
    import torch
    model.set_adapter('user_a')
    model.eval()
    ids = tokenizer(text)['input_ids']
    if len(ids) < 8 or tuple(ids[:3]) == tuple(ids[3:6]):
        raise ValueError('Exact fixture needs two distinct windows plus unmatched rows')
    inputs = dict(input_ids=torch.tensor([ids], device=next(model.parameters()).device), use_cache=False)
    with torch.inference_mode(), qkv_capture(model, validate=False) as capture:
        baseline = model(**inputs).logits.detach().cpu()
    cache = GlobalCache(20_000_000_000)
    hits = []
    for start in (0, 3):
        window = Subsequence(tuple(ids[start:start+3]), start, start+3)
        assert cache.lookup((0, window.token_ids)) is None
        entry = CacheEntry.from_tensors(0, window.token_ids, (start, start+3),
            {layer: tuple(record[n][:, start:start+3] for n in 'qkv') for layer, record in capture.records.items()})
        entry.qkv_metadata = dict(component_scope='total_qkv', source_user='user_a')
        assert cache.insert(entry)
        assert cache.lookup(entry.key) is entry and entry.physical_tensor_bytes > 0
        hits.append(CacheHit(window, entry))
    with torch.inference_mode(), mixed_projection_path(adapter, 'user_a', hits, len(ids)) as audit:
        mixed = model(**inputs).logits.detach().cpu()
    metrics = compare_logits(baseline, mixed, 0)
    errors = [similarity(capture.records[layer][n], record[n])
              for layer, record in audit.projections.items() for n in 'qkv']
    # FP32 implementation-parity controls, not cache-reuse safety thresholds.
    absolute_tolerance = 1e-5
    relative_tolerance = 1e-6
    kl_tolerance = 1e-8  # Dimensionless KL has its own implementation tolerance.
    projection_max_abs_error = max(m['max_abs_error'] for m in errors)
    projection_max_relative_l2 = (
        None if any(m['relative_l2'] is None for m in errors)
        else max(m['relative_l2'] for m in errors))
    if (any(m['max_abs_error'] > absolute_tolerance
            or m['relative_l2'] is None or m['relative_l2'] > relative_tolerance for m in errors)
            or metrics['max_abs_logit_diff'] > absolute_tolerance or metrics['relative_l2_logit_diff'] is None
            or metrics['relative_l2_logit_diff'] > relative_tolerance
            or abs(metrics['affected_suffix_mean_kl']) > kl_tolerance
            or abs(metrics['last_position_kl_baseline_to_injected']) > kl_tolerance
            or not metrics['last_argmax_agreement']):
        raise AssertionError(
            f'STOP: exact-control mixed parity failed (implementation-parity controls, '
            f'not cache-reuse safety thresholds): device={inputs["input_ids"].device}, '
            f'dtype={baseline.dtype}, absolute_tolerance={absolute_tolerance}, '
            f'relative_tolerance={relative_tolerance}, kl_tolerance={kl_tolerance}, '
            f'projection_max_abs_error={projection_max_abs_error}, '
            f'projection_max_relative_l2={projection_max_relative_l2}, '
            f'logit_abs_error={metrics["max_abs_logit_diff"]}, '
            f'logit_relative_l2={metrics["relative_l2_logit_diff"]}, '
            f'KL_suffix={metrics["affected_suffix_mean_kl"]}, '
            f'KL_last={metrics["last_position_kl_baseline_to_injected"]}; '
            f'{metrics}; projection errors={errors}')
    for record in audit.records.values():
        if record['native_positions'] != list(range(6, len(ids))) or record['reused_projection_rows'] != 6:
            raise AssertionError('STOP: native projection received cached positions')
    return dict(**metrics, projection_max_abs_error=projection_max_abs_error,
        projection_max_relative_l2=projection_max_relative_l2,
        full_fresh_projected_rows=len(ids), mixed_native_projected_rows=len(ids)-6, reused_rows=6,
        matched_windows=2, projection_calls=list(audit.records.values()), exact_control_passed=True,
        physical_cache_bytes=cache.physical_tensor_bytes, component_scope='total_qkv',
        metric_source='measured', safe_reuse_claimed=False)
