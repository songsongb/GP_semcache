"""Controlled cache-backed development experiment; no reuse safety policy."""
import json
import math
from semcache.models.model_adapter import OPTModelAdapter
from semcache.models.qkv_injection import qkv_injection
from semcache.models.capture import qkv_capture
from semcache.probe import observe, slices
from semcache.semantic.probes import construct_probes
from semcache.semantic.matcher import ExactTokenMatcher
from semcache.cache.cache_entry import CacheEntry
from semcache.cache.global_cache import GlobalCache
from .qkv_metrics import similarity
from .logit_metrics import compare_logits


def run_output_impact(model, tokenizer, metadata, window_size=3, layers=(0, 1, 5, 11),
                      modes=('q', 'k', 'v', 'qkv'), cases=('A', 'B', 'C', 'D'),
                      seed=42, tolerance=1e-6, on_row=None):
    import torch
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError('Tolerance must be finite and nonnegative')
    adapter = OPTModelAdapter(model)
    layers = list(range(len(adapter.layers))) if layers is None else list(layers)
    if not layers or len(set(layers)) != len(layers):
        raise ValueError('Need unique nonempty layers')
    for layer in layers:
        adapter.projection_modules(layer)
    if not modes or any(m not in ('q', 'k', 'v', 'qkv') for m in modes):
        raise ValueError('Invalid modes')
    if not cases or any(c not in ('A', 'B', 'C', 'D') for c in cases):
        raise ValueError('Invalid cases')
    model.eval()
    device = next(model.parameters()).device
    rows = []
    for pair in construct_probes(tokenizer, window_size):
        if pair['probe_case'] not in cases:
            continue
        wa, wb = pair['window_a'], pair['window_b']
        source = slices(observe(model, pair['input_ids_a'], layers), wa)
        cache = GlobalCache(20_000_000_000)
        matcher = ExactTokenMatcher()
        source_key = matcher.key(0, wa)
        if cache.lookup(source_key) is not None:
            raise AssertionError('Expected MISS')
        entry = CacheEntry.from_tensors(0, wa.token_ids, (wa.start, wa.end), source)
        if not cache.insert(entry):
            raise AssertionError('Expected INSERT')
        inputs = dict(input_ids=torch.tensor([pair['input_ids_b']], device=device), use_cache=False)
        with qkv_capture(model, layers=layers) as capture:
            with torch.inference_mode():
                baseline = model(**inputs).logits.detach().cpu()
        target = slices(capture.records, wb)
        del capture
        similarities = {i: {f'{n}_{k}': v for n, a, b in zip('qkv', source[i], target[i])
                            for k, v in similarity(a, b).items() if k in ('cosine_similarity', 'relative_l2')}
                        for i in layers}
        for layer in layers:
            for mode in modes:
                # D intentionally fetches A's key, NOT a valid match for B.
                key = matcher.key(0, wb) if pair['valid_reuse_candidate'] else source_key
                cache.advance(1)
                hit = cache.lookup(key)
                if hit is None or not hit.tensors or not hit.physical_tensor_bytes:
                    raise AssertionError('Expected physical HIT/FETCH')
                with qkv_injection(adapter, layer, hit.tensors[layer], (0, len(wa.token_ids)),
                                   (wb.start, wb.end), mode) as audit:
                    with torch.inference_mode():
                        injected = model(**inputs).logits.detach().cpu()
                metrics = compare_logits(baseline, injected, wb.start)
                row = dict(model_id=metadata['model'], resolved_model_revision=metadata['resolved_model_revision'],
                           dtype=metadata['dtype'], seed=seed, probe_case=pair['probe_case'],
                           valid_reuse_candidate=pair['valid_reuse_candidate'], description=pair['description'],
                           query_a=pair['query_a'], query_b=pair['query_b'],
                           shared_token_ids=json.dumps(wa.token_ids) if pair['valid_reuse_candidate'] else '[]',
                           source_token_ids=json.dumps(wa.token_ids), target_token_ids=json.dumps(wb.token_ids),
                           source_start=wa.start, source_end=wa.end, target_start=wb.start, target_end=wb.end,
                           layer=layer, injection_mode=mode, cache_hit=True, cache_storage_device=str(hit.tensors[layer][0].device),
                           cache_lookup_kind='exact_token_match' if pair['valid_reuse_candidate'] else 'forced_source_key_stress_control',
                           cache_events='MISS->INSERT->HIT->FETCH->INJECT',
                           injected_tensor_bytes=audit.injected_tensor_bytes,
                           projection_integrity_passed=set(audit.records) == set('qkv'),
                           metric_source='measured', safe_reuse_claimed=False, **similarities[layer], **metrics)
                if on_row:
                    on_row(row)
                if metrics['prefix_max_abs_logit_diff'] > tolerance:
                    raise AssertionError(f'Causal prefix changed: {row}')
                if pair['probe_case'] == 'A' or (pair['probe_case'] == 'B' and layer == 0):
                    if metrics['max_abs_logit_diff'] > tolerance:
                        raise AssertionError(f'Identity implementation control failed: {row}')
                rows.append(row)
    return rows
