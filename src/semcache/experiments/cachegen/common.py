"""Explicit tensor/JSON interchange and storage accounting (no model imports)."""
import csv
import hashlib
import json
import platform
import random
import statistics
from pathlib import Path

MODES = ('FP16_RAW', 'UNIFORM_INT8', 'CACHEGEN_QUANT', 'CACHEGEN_FULL')
DIMENSIONS = dict(layers=32, hidden_dim=2560, heads=32, head_dim=80, dtype='float16')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def read_json(path):
    return json.loads(Path(path).read_text())


def write_csv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(fields)), extrasaction='raise')
        writer.writeheader()
        writer.writerows(rows)


def raw_bytes(tokens, layers=32, hidden_dim=2560, element_size=2):
    if any(type(v) is not int or v <= 0 for v in (tokens, layers, hidden_dim, element_size)):
        raise ValueError('Positive integer dimensions required')
    q = layers * tokens * hidden_dim * element_size
    return dict(raw_q_bytes=q, raw_kv_bytes=2*q, raw_qkv_bytes=3*q)


def storage(tokens, payload, metadata, **dimensions):
    if payload < 0 or metadata < 0 or payload + metadata <= 0:
        raise ValueError('Invalid encoded byte counts')
    raw = raw_bytes(tokens, **dimensions)
    encoded = payload + metadata
    total = raw['raw_q_bytes'] + encoded
    return dict(**raw, encoded_kv_payload_bytes=payload, encoded_metadata_bytes=metadata,
                encoded_kv_total_bytes=encoded, kv_compression_ratio=raw['raw_kv_bytes']/encoded,
                semcache_total_stored_bytes=total,
                semcache_total_compression_ratio=raw['raw_qkv_bytes']/total)


def to_heads(x, heads=32):
    if x.ndim != 3 or x.shape[-1] % heads:
        raise ValueError('Expected [L,T,d] with d divisible by heads')
    l, t, d = x.shape
    return x.reshape(l, t, heads, d//heads).permute(0, 2, 1, 3).contiguous()


def from_heads(x):
    if x.ndim != 4:
        raise ValueError('Expected [L,H,T,Dh]')
    l, h, t, dh = x.shape
    return x.permute(0, 2, 1, 3).reshape(l, t, h*dh).contiguous()


def validate_qkv(qkv, tokens):
    import torch
    if set(qkv) != set('qkv'):
        raise ValueError('Fixture must contain exactly q, k, v tensors')
    for x in qkv.values():
        if x.shape != (32, tokens, 2560) or x.dtype != torch.float16 or not torch.isfinite(x).all():
            raise ValueError('Expected finite OPT FP16 [32,T,2560] projections')


def sample_partitions(records, count=128, seed=42):
    """Group MultiWOZ by dialogue, preventing cross-partition source leakage."""
    if count < 1:
        raise ValueError('Positive query count required')
    groups, seen = {}, set()
    for record in records:
        sid = str(record['source_id'])
        if sid in seen:
            raise ValueError('Duplicate source/query identity')
        seen.add(sid)
        group = str(record.get('conversation_id') or sid)
        groups.setdefault(group, []).append(record)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    result = {}
    cursor = 0
    for partition in ('calibration', 'evaluation'):
        chosen = []
        while len(chosen) < count and cursor < len(keys):
            group = keys[cursor]
            cursor += 1
            chosen.extend(sorted(groups[group], key=lambda r: str(r['source_id']))[:count-len(chosen)])
        if len(chosen) != count:
            raise ValueError('Insufficient source-disjoint queries; supply more data')
        result[partition] = chosen
    result['hashes'] = {p: digest(result[p]) for p in ('calibration', 'evaluation')}
    return result


def errors(k, v, dk, dv):
    from semcache.evaluation.qkv_metrics import similarity
    out = {}
    for name, a, b in (('k', k, dk), ('v', v, dv)):
        m = similarity(a, b)
        out.update({name+'_max_abs': m['max_abs_error'], name+'_rel_l2': m['relative_l2'],
                    name+'_cosine': m['cosine_similarity']})
    return out


def environment():
    result = dict(python=platform.python_version(), platform=platform.platform(),
                  torch=None, cuda=None, gpu=None, measurement_status='NOT_RUN')
    try:
        import torch
        result.update(torch=torch.__version__, cuda=torch.version.cuda,
                      gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None)
    except ImportError:
        pass
    return result


def load_fixture(root, block):
    import torch
    path = Path(root) / block['file']
    if file_hash(path) != block['sha256']:
        raise ValueError('Fixture SHA256 mismatch')
    qkv = torch.load(path, map_location='cpu', weights_only=True)
    validate_qkv(qkv, block['token_group_size'])
    return qkv


def percentile(values, p):
    values = sorted(values)
    i = (len(values)-1)*p
    lo = int(i)
    return values[lo] + (values[min(lo+1, len(values)-1)]-values[lo])*(i-lo)


SUMMARY_METRICS = ('encoded_kv_total_bytes', 'kv_compression_ratio',
    'semcache_total_compression_ratio', 'encode_ms', 'decode_ms', 'k_rel_l2', 'v_rel_l2')
SUMMARY_FIELDS = ['dataset', 'token_group_size', 'compression_mode', 'block_count'] + [
    f'{m}_{s}' for m in SUMMARY_METRICS for s in ('mean', 'p50', 'p95')]


def summarize(rows):
    groups = {}
    for r in rows:
        if r['status'] == 'MEASURED':
            groups.setdefault(tuple(r[k] for k in SUMMARY_FIELDS[:3]), []).append(r)
    result = []
    for key, group in sorted(groups.items()):
        row = dict(zip(SUMMARY_FIELDS[:3], key), block_count=len(group))
        for metric in SUMMARY_METRICS:
            values = [r[metric] for r in group if r[metric] is not None]
            for suffix, value in (('mean', statistics.mean(values) if values else None),
                                  ('p50', percentile(values, .5) if values else None),
                                  ('p95', percentile(values, .95) if values else None)):
                row[f'{metric}_{suffix}'] = value
        result.append(row)
    return result
