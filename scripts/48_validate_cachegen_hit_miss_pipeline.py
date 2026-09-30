"""Tiny real-codec integration; no model, dataset, or engine modifications.

Transport is the teammate's per-projection LoRA-delta codec, exercised with
synthetic projection tensors (zero base). Storage is the existing C2 frozen
K20/V16 research codec, not the transport packet or official LMCache serializer.
"""
import argparse
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


class TransportCacheGenCodec:
    def __init__(self):
        from semcache.edgelora.cachegen_codec import encode_lora_delta, decode_lora_delta
        self.encode = encode_lora_delta
        self.decode = decode_lora_delta
        self.calls = 0

    def transmit(self, tensors):
        # Single-process simulated transmission, as in cachegen_projection.py.
        self.calls += 1
        packet = {layer: tuple(self.encode(t) for t in qkv)
                  for layer, qkv in tensors.items()}
        return {layer: tuple(self.decode(p) for p in qkv)
                for layer, qkv in packet.items()}


class StorageCacheGenCodec:
    def __init__(self, profile, device):
        from semcache.experiments.cachegen.c2.physical_storage import FrozenK20V16Codec
        self.codec = FrozenK20V16Codec(
            profile, quantization_device=device, decode_device=device, expected_hidden=8)
        self.encodes = self.decodes = 0

    def encode(self, tensors):
        self.encodes += 1
        def kv_only_entry(*args, compressed_kv, **kwargs):
            # make_entry temporarily copies Q; discard it at the entry boundary.
            # The resident object has only the existing compressed K/V payload.
            return compressed_kv
        return self.codec.make_entry(kv_only_entry, 0, (11, 12, 13), (0, 3),
                                     tensors, 'cpu')

    def decode(self, payload, current_q):
        self.decodes += 1
        # Q belongs only to this temporary request view, never to the cache.
        view = SimpleNamespace(compressed_kv=payload, q_tensors=current_q)
        return self.codec.decode_entry(view).tensors


class Pipeline:
    def __init__(self, transport, storage, project, downstream):
        self.transport, self.storage = transport, storage
        self.project, self.downstream = project, downstream
        self.cache = {}

    def request(self, key):
        network_before = self.transport.calls
        if key in self.cache:
            current_q = self.project(q_only=True)
            tensors = self.storage.decode(self.cache[key], current_q)
            record = dict(cache_lookup='HIT', network_path_used=False,
                          storage_decode_used=True, q_recomputed_or_current=True)
            assert all(tensors[layer][0] is q for layer, q in current_q.items())
        else:
            tensors = self.transport.transmit(self.project(q_only=False))
            self.cache[key] = self.storage.encode(tensors)
            record = dict(cache_lookup='MISS', network_path_used=True,
                          cache_inserted=True, stored_q=False, stored_k=True, stored_v=True)
        assert self.transport.calls - network_before == int(record['network_path_used'])
        self.downstream(tensors)
        return record, tensors


def run(args):
    import torch
    from semcache.evaluation.qkv_metrics import similarity
    if args.storage_src:
        import semcache.experiments
        semcache.experiments.__path__.append(str(
            args.storage_src.resolve() / 'semcache' / 'experiments'))
    from semcache.experiments.cachegen.c2.physical_storage import DEFAULT_PROFILE
    device = torch.device(args.device)
    if device.type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('The existing transport codec requires CUDA and torchac_cuda; no CPU substitute is used')
    # _prepare uses .cuda(): scope its default GPU explicitly without global edits.
    with torch.cuda.device(device), torch.inference_mode():
        generator = torch.Generator().manual_seed(2026)
        hidden = torch.randn(1, 3, 8, generator=generator)
        weights = torch.randn(32, 3, 8, 8, generator=generator)

        def project(q_only=False):
            if q_only:
                return {layer: (hidden @ weights[layer, 0]).to(device, torch.float16)
                        for layer in range(32)}
            return {layer: tuple((hidden @ weights[layer, role]).to(device, torch.float16)
                                 for role in range(3)) for layer in range(32)}

        def downstream(tensors):
            assert set(tensors) == set(range(32))
            for q, k, v in tensors.values():
                assert all(t.shape == (1, 3, 8) and t.device == q.device
                           and torch.isfinite(t).all() for t in (q, k, v))
                output = torch.softmax(q.float() @ k.float().transpose(-1, -2) / 8**0.5, -1) @ v.float()
                assert output.shape == q.shape and torch.isfinite(output).all()

        transport = TransportCacheGenCodec()
        storage = StorageCacheGenCodec(args.profile or DEFAULT_PROFILE, str(device))
        pipeline = Pipeline(transport, storage, project, downstream)
        reference = project()
        key = (0, (11, 12, 13))
        miss, miss_tensors = pipeline.request(key)
        payload = pipeline.cache[key]
        # Inspect the actual resident schema, rather than relying on the log flags.
        assert set(vars(payload)) == {'bitstream', 'shape', 'dtype', 'profile_sha256',
                                     'role_payload_bytes', 'local_metadata_bytes'}
        assert isinstance(payload.bitstream, bytes) and payload.bitstream
        hit, hit_tensors = pipeline.request(key)
        assert miss == dict(cache_lookup='MISS', network_path_used=True,
                            cache_inserted=True, stored_q=False, stored_k=True, stored_v=True)
        assert hit == dict(cache_lookup='HIT', network_path_used=False,
                           storage_decode_used=True, q_recomputed_or_current=True)
        assert transport.calls == storage.encodes == storage.decodes == len(pipeline.cache) == 1
        assert pipeline.cache[key] is payload
        for number, record in enumerate((miss, hit), 1):
            print(f'\n[Request {number}]')
            for name, value in record.items():
                print(f'{name:25}: {value}')
        print('\n[Validation]')
        print('transport_codec          : CacheGen Q/K/V, synthetic zero-base projections')
        print('storage_codec            : FrozenK20V16Codec (existing C2 research codec)')
        print(f'resident_KV_bytes        : {len(payload.bitstream)} (Q absent)')
        for stage, tensors in [('MISS', miss_tensors), ('HIT', hit_tensors)]:
            for role, index in [('Q', 0), ('K', 1), ('V', 2)]:
                a = torch.cat([reference[i][index] for i in range(32)])
                b = torch.cat([tensors[i][index] for i in range(32)])
                metrics = similarity(a, b)
                print(f'{stage} {role}: cosine={metrics["cosine_similarity"]:.6f} '
                      f'relative_l2={metrics["relative_l2"]:.6f}')
        print('routing                  : PASS\n\nRESULT: PASS')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--storage-src', type=Path,
                        help='Existing storage checkout src directory when absent from this branch')
    parser.add_argument('--profile', type=Path, help='Existing frozen K20/V16 profile')
    args = parser.parse_args()
    print('=' * 60 + '\nCacheGen Integrated HIT/MISS Pipeline Smoke Test\n' + '=' * 60)
    try:
        run(args)
    except (ImportError, FileNotFoundError, RuntimeError) as exc:
        print(f'\nRESULT: BLOCKED/ERROR: {exc}', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
