"""C2.5 one-fixture, four-role reference-versus-bitexact-fast coder benchmark."""
from pathlib import Path
import statistics
import struct
import subprocess
import time

from .b2 import format as fmt
from .c15c.harness import git_state
from .c15c.holdout import evaluation_blocks
from .c15c.policy import UniformKVPolicy
from .c15c.rate_calibration import observe
from .c2.physical_storage import DEFAULT_PROFILE, PROFILE_SHA256
from .common import file_hash, load_fixture, percentile, read_json, write_json
from .shared.device_contract import resolve_contract
from .shared.source_audit import REVISION

ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT/'results/cachegen/c2_5/coder_benchmark'
POLICY = UniformKVPolicy(20, 16)
BACKENDS = (fmt.REFERENCE_CODER, fmt.FAST_CODER)


def _stats(values):
    return dict(mean=statistics.mean(values), p50=percentile(values, .5))


def benchmark_streams(streams, profile, *, repeats=3):
    """Pure CPU benchmark; no fixture access, model, CDF fitting or GPU calls."""
    if type(repeats) is not int or not 1 <= repeats <= 5 or len(streams) != 4:
        raise ValueError('One to five repeats and four B2 streams required')
    results, samples = {}, {}
    for backend in BACKENDS:
        encode, decode = fmt.coder_functions(backend)
        encode_times, decode_times = [[] for _ in range(4)], [[] for _ in range(4)]
        total_encode, total_decode = [], []
        for _ in range(repeats):
            payloads = []
            start_total = time.perf_counter()
            for i, (stream, cdf) in enumerate(zip(streams, profile.cdfs)):
                start = time.perf_counter()
                payloads.append(encode(stream, cdf))
                encode_times[i].append(1000*(time.perf_counter()-start))
            total_encode.append(1000*(time.perf_counter()-start_total))
            decoded = []
            start_total = time.perf_counter()
            for i, (payload, stream, cdf) in enumerate(zip(payloads, streams, profile.cdfs)):
                start = time.perf_counter()
                decoded.append(decode(payload, len(stream), cdf))
                decode_times[i].append(1000*(time.perf_counter()-start))
            total_decode.append(1000*(time.perf_counter()-start_total))
            if tuple(decoded) != tuple(streams):
                raise ValueError(f'{backend} decoded symbol mismatch')
        samples[backend] = (tuple(payloads), tuple(decoded))
        roles = {}
        for i, name in enumerate(('k_anchor', 'k_residual', 'v_anchor', 'v_residual')):
            encode_mean = statistics.mean(encode_times[i])/1000
            decode_mean = statistics.mean(decode_times[i])/1000
            roles[name] = dict(symbol_count=len(streams[i]), encoded_bytes=len(payloads[i]),
                encode_ms=_stats(encode_times[i]), decode_ms=_stats(decode_times[i]),
                encode_MB_per_s=len(streams[i])/1e6/encode_mean,
                decode_MB_per_s=len(streams[i])/1e6/decode_mean,
                encode_symbols_per_s=len(streams[i])/encode_mean,
                decode_symbols_per_s=len(streams[i])/decode_mean)
        results[backend] = dict(backend=backend, total_encode_ms=_stats(total_encode),
            total_decode_ms=_stats(total_decode), encoded_payload_bytes=sum(map(len, payloads)),
            symbol_mismatches=0, roles=roles)
    if samples[fmt.REFERENCE_CODER][0] != samples[fmt.FAST_CODER][0]:
        raise ValueError('Fast coder is not byte-for-byte reference compatible')
    return results, samples


def run(args):
    import torch
    repo = args.cachegen_repo.resolve()
    reference_commit = subprocess.run(['git', '-C', str(repo), 'rev-parse', 'HEAD'],
        check=True, text=True, capture_output=True).stdout.strip()
    if reference_commit != REVISION:
        raise ValueError('Pinned released CacheGen commit required')
    capture_path = args.capture_manifest.resolve()
    capture = read_json(capture_path)
    blocks, counts = evaluation_blocks(capture)
    block = blocks[0]  # One deterministic w=3 evaluation fixture only.
    root = capture_path.parent
    contract = resolve_contract(root)
    device = contract['quantization_device_resolved']
    if not device.startswith('cuda:'):
        raise ValueError('SERAPH benchmark requires recorded C1 CUDA quantization device')
    profile_path = args.profile_path.resolve()
    profile = fmt.Profile.from_bytes(profile_path.read_bytes(), PROFILE_SHA256)
    if profile.mode != fmt.MODES[1]:
        raise ValueError('Frozen B2 anchor/mod-residual profile required')
    out = args.output_dir.resolve()
    if not out.is_relative_to(OUTPUT.resolve()) or out.is_relative_to(root) or out.is_relative_to(profile_path.parent):
        raise ValueError('C2.5 output must be a fresh directory under results/cachegen/c2_5/coder_benchmark')
    out.mkdir(parents=True, exist_ok=False)
    manifest = dict(stage='C2.5-CODER-BENCHMARK', status='RUNNING', gp_semcache_git=git_state(),
        reference_cachegen_commit=reference_commit,
        capture_manifest_path=str(capture_path), capture_manifest_sha256=file_hash(capture_path),
        block_id=block['block_id'], dataset=block['dataset'], window_size=3, capture_counts=counts,
        fixture_path=str((root/block['file']).resolve()), fixture_sha256=block['sha256'],
        profile_path=str(profile_path), profile_sha256=PROFILE_SHA256,
        policy=POLICY.name, B2_mode=fmt.MODES[1], backends=list(BACKENDS),
        repeats=args.repeats, device_contract=contract, no_cdf_fit=True, no_model_inference=True)
    write_json(out/'manifest.json', manifest)
    try:
        fixture = load_fixture(root, block)
        encoded = {role: POLICY.quantize(fixture[role.lower()].to(device), role)
                   for role in ('K', 'V')}
        for role, value in encoded.items():
            observe(value, block['block_id'], 'C2.5_'+role)
        start = time.perf_counter()
        domains = tuple(bytes((encoded[role].signed_symbols.cpu().to(torch.int16)+127)
                              .to(torch.uint8).flatten().tolist()) for role in ('K', 'V'))
        shape = tuple(encoded['K'].symbols.shape)
        streams = fmt.streams_from_domains(profile.mode, domains, shape, expected_tokens=3)
        transform_ms = 1000*(time.perf_counter()-start)
        maxima = struct.pack('<192f', *[x for role in ('K', 'V')
            for x in encoded[role].storage_metadata.cpu().flatten().tolist()])
        results, samples = benchmark_streams(streams, profile, repeats=args.repeats)
        frames = {backend: fmt.frame_payloads(profile, samples[backend][0], maxima, shape, expected_tokens=3)
                  for backend in BACKENDS}
        if frames[fmt.REFERENCE_CODER] != frames[fmt.FAST_CODER]:
            raise ValueError('B2 framed bitstreams differ')
        raw_q = fixture['q'].numel()*fixture['q'].element_size()
        raw_kv = sum(fixture[r].numel()*fixture[r].element_size() for r in ('k', 'v'))
        framed_bytes = len(frames[fmt.REFERENCE_CODER])
        for backend in BACKENDS:
            actual_shape, stored_maxima, _, _ = fmt.inspect(frames[backend], profile, expected_tokens=3)
            if actual_shape != shape or stored_maxima != maxima:
                raise ValueError('B2 frame changed shape/maxabs metadata')
            recovered_maxima = torch.tensor(struct.unpack('<192f', stored_maxima),
                dtype=torch.float16).reshape(2, 32, 3, 1)
            recovered = fmt.domains_from_streams(profile.mode, samples[backend][1], shape, expected_tokens=3)
            mismatches = sum(sum(a != b for a, b in zip(actual, original))
                             for actual, original in zip(recovered, domains))
            reconstruction_failures = 0
            for index, (role, domain) in enumerate(zip(('K', 'V'), recovered)):
                signed = (torch.frombuffer(bytearray(domain), dtype=torch.uint8).to(torch.int16)-127).reshape(shape)
                direct = encoded[role]
                restored = direct.restore_storage(signed.to(device), recovered_maxima[index].to(device))
                reconstruction_failures += int(not torch.equal(restored.reconstructed, direct.reconstructed))
            results[backend].update(encoded_frame_bytes=framed_bytes,
                raw_kv_bytes=raw_kv, raw_q_bytes=raw_q,
                kv_compression_ratio=raw_kv/framed_bytes,
                whole_entry_compression_ratio=(raw_q+raw_kv)/(raw_q+framed_bytes),
                symbol_mismatches=mismatches, reconstruction_failures=reconstruction_failures)
            if mismatches or reconstruction_failures:
                raise ValueError('C2.5 symbol/reconstruction invariant failed')
        report = dict(transform_preparation_ms=transform_ms,
            symbol_counts=dict(zip(('k_anchor', 'k_residual', 'v_anchor', 'v_residual'), map(len, streams))),
            bitstream_identical=True, backend_comparison=results,
            frozen_profile_bytes_counted_once=len(profile.to_bytes()),
            fast_selected_automatically=False, physical_size_accounting_changed=False)
        write_json(out/'coder_benchmark.json', report)
        manifest.update(status='COMPLETED', output_sha256={'coder_benchmark.json': file_hash(out/'coder_benchmark.json')})
        for name in BACKENDS:
            item = results[name]
            print(name, 'encode p50 ms', item['total_encode_ms']['p50'],
                'decode p50 ms', item['total_decode_ms']['p50'],
                'frame bytes', item['encoded_frame_bytes'])
        print('bitstream identical:', report['bitstream_identical'])
    except BaseException as exc:
        manifest.update(status='FAILED', reason=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        write_json(out/'manifest.json', manifest)
    return report
