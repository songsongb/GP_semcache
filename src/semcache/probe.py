"""Base OPT observations only: cached tensors are never passed to attention."""
import json
from .models.capture import qkv_capture
from .models.qkv_projection import extract_qkv_positions
from .evaluation.qkv_metrics import similarity
from .semantic.probes import construct_probes
from .semantic.intent_clusterer import IntentClusterer
from .semantic.encoder import ControlledEncoder
from .semantic.matcher import ExactTokenMatcher
from .cache.cache_entry import CacheEntry
from .cache.global_cache import GlobalCache


def observe(model, ids, layers):
    import torch
    device = next(model.parameters()).device
    with qkv_capture(model, layers=layers, storage_device='cpu', validate=True) as capture:
        with torch.inference_mode():
            model(input_ids=torch.tensor([ids], device=device), use_cache=False)
    return capture.records


def slices(records, window):
    return {i: extract_qkv_positions(*(r[n] for n in ('q','k','v')), range(window.start, window.end))
            for i,r in records.items()}


def rows_for(pair, a, b, metadata):
    wa, wb = pair['window_a'], pair['window_b']
    common = dict(model_id=metadata['model'], model_revision=metadata['resolved_model_revision'] or metadata['model_revision'],
                  dtype=metadata['dtype'], query_pair=pair['probe_case'], probe_case=pair['probe_case'],
                  query_a=pair['query_a'], query_b=pair['query_b'], window_size=len(wa.token_ids),
                  shared_token_ids=json.dumps(wa.token_ids) if wa.token_ids == wb.token_ids else '[]',
                  token_ids_a=json.dumps(wa.token_ids), token_ids_b=json.dumps(wb.token_ids),
                  full_input_ids_a=json.dumps(pair['input_ids_a']), full_input_ids_b=json.dumps(pair['input_ids_b']),
                  position_a_start=wa.start, position_a_end=wa.end, position_b_start=wb.start, position_b_end=wb.end)
    return [dict(common, layer=i, tensor_type=n.upper(), **similarity(x,y))
            for i in sorted(a) for n,x,y in zip(('q','k','v'), a[i], b[i])]


def run_probe(model, tokenizer, metadata, window_size=3, layers=None, physical=False):
    pairs = construct_probes(tokenizer, window_size)
    if physical:
        pairs = [p for p in pairs if p['probe_case'] == 'B']
    rows, details = [], []
    for pair in pairs:
        wa, wb = pair['window_a'], pair['window_b']
        cache = GlobalCache(20_000_000_000)
        clusterer = IntentClusterer(1)
        clusterer.initialize([[0.0]])
        encoder = ControlledEncoder({pair['query_a']: [0.0], pair['query_b']: [0.0]})
        ca = clusterer.observe(encoder.encode([pair['query_a']])[0])
        matcher = ExactTokenMatcher()
        if physical and cache.lookup(matcher.key(ca, wa)) is not None:
            raise AssertionError('Expected MISS')
        ra = observe(model, pair['input_ids_a'], layers)
        a = slices(ra, wa)
        parity_a = {i: r['validation'] for i,r in ra.items()}
        shape_metadata = {i: r['metadata'] for i,r in ra.items()}
        del ra
        if physical:
            entry = CacheEntry.from_tensors(ca, wa.token_ids, (wa.start, wa.end), a)
            if not cache.insert(entry):
                raise RuntimeError('Physical admission failed')
            cache.advance(1)
            cb = clusterer.observe(encoder.encode([pair['query_b']])[0])
            hit = cache.lookup(matcher.key(cb, wb))
            if hit is None or not hit.tensors or not hit.physical_tensor_bytes:
                raise AssertionError('Expected physical HIT')
            a = hit.tensors
        rb = observe(model, pair['input_ids_b'], layers)
        b = slices(rb, wb)
        rows.extend(rows_for(pair, a, b, metadata))
        details.append(dict(probe_case=pair['probe_case'], query_a=pair['query_a'], query_b=pair['query_b'],
                            full_input_ids_a=pair['input_ids_a'], full_input_ids_b=pair['input_ids_b'],
                            window_a=vars(wa), window_b=vars(wb),
                            decoded_a=tokenizer.decode(list(wa.token_ids)), decoded_b=tokenizer.decode(list(wb.token_ids)),
                            projection_validation_a=parity_a,
                            projection_validation_b={i:r['validation'] for i,r in rb.items()},
                            layer_metadata=shape_metadata,
                            physical_tensor_bytes=cache.physical_tensor_bytes,
                            cache_events=['MISS','INSERT','HIT'] if physical else [],
                            cluster_backend='controlled_vectors_v1' if physical else None))
        del rb
    return rows, dict(metadata=metadata, cases=details, physical_cache_hit=physical,
                      safe_inference_reuse=False, scope='base_OPT_observation_only')
