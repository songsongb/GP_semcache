#!/usr/bin/env python3
"""CPU-only L2E compute benchmark, operation census and tensor-memory profile.

No model download, approval generation, GPU launcher, or production PLAN write.
"""
import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack
import gzip
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import statistics
import struct
import subprocess
import sys
import tempfile
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from semcache.edgelora.communication import require
from semcache.edgelora import r5_d5_impact as l2c
from semcache.edgelora import r5_d5_impact_compute as l2e
from semcache.edgelora.r5_subsequence_generation import StreamingImpact
from semcache.cache.attention_impact import PaperRowL2SumReducer
from semcache.semantic.subsequence import SubsequenceExtractor

MODES = (l2c.REFERENCE, l2c.HOST_BATCH, l2e.MASK_INDEX_REUSE)


def describe(values):
    quartiles = statistics.quantiles(values, n=4)
    return dict(count=len(values), median=statistics.median(values), mean=statistics.mean(values),
                p25=quartiles[0], p75=quartiles[2], minimum=min(values), maximum=max(values))


def bit_hash(values):
    return hashlib.sha256(b''.join(struct.pack('<d', v) for v in values)).hexdigest()


def operation_observer(fn, padding):
    """Python API calls, not ATen kernel counts; nested times are inclusive."""
    import torch
    counts, times = Counter(), defaultdict(float)
    def wrapper(original, name):
        def run(*args, **kwargs):
            category = 'padding_mask_to' if name == 'to' and args[0] is padding else name
            start = time.perf_counter_ns()
            try:
                return original(*args, **kwargs)
            finally:
                counts[category] += 1
                times[category] += (time.perf_counter_ns() - start) / 1e6
        return run
    started = time.perf_counter_ns()
    with ExitStack() as stack:
        for target, names in ((torch, ('arange', 'isfinite', 'stack')),
                              (torch.Tensor, ('double', 'detach', 'to', '__le__', '__and__', '__iand__',
                                              '__invert__', 'masked_fill', 'all', 'norm', 'mean', 'sum',
                                              'cpu', 'tolist', 'item', '__bool__'))):
            for name in names:
                stack.enter_context(patch.object(target, name, wrapper(getattr(target, name), name)))
        result = fn()
    return result, dict(counts), dict(times), (time.perf_counter_ns() - started) / 1e6


def memory_profile(fn):
    """CPU tensor allocator timeline; excludes Python/allocator overhead/workspaces."""
    import torch
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],
                                profile_memory=True, record_shapes=True, with_stack=True) as profiler:
        fn()
    operators = [dict(name=e.key, count=e.count, cpu_time_us=e.cpu_time_total,
                      self_cpu_time_us=e.self_cpu_time_total,
                      net_cpu_bytes=e.cpu_memory_usage, self_net_cpu_bytes=e.self_cpu_memory_usage)
                 for e in profiler.key_averages()]
    from torch.profiler._memory_profiler import Action
    with tempfile.TemporaryDirectory(prefix='d5_l2e_memory_') as tmp:
        path = Path(tmp) / 'memory.raw.json.gz'
        profiler.export_memory_timeline(str(path), device='cpu')
        with gzip.open(path, 'rt') as stream:
            events = json.load(stream)
    current = peak = preexisting = allocated = 0
    for timestamp, action, numbytes, category in events:
        current += numbytes  # DESTROY and old-version events already have a negative sign.
        if action == Action.PREEXISTING.value:
            preexisting += numbytes
        if action == Action.CREATE.value:
            allocated += numbytes
        peak = max(peak, current)
    return dict(peak_live_tensor_bytes=peak, preexisting_tensor_bytes=preexisting,
                peak_above_preexisting_bytes=peak - preexisting,
                cumulative_created_tensor_bytes=allocated, memory_events=len(events), operators=operators,
                scope='one instrumented CPU layer; torch tensor allocations only; '
                      'cumulative allocation traffic is not peak; not process RSS or GPU VRAM')


def benchmark(args):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and os.environ.get('HIP_VISIBLE_DEVICES') == '',
            'explicit CPU-only environment required')
    require(args.repeats >= 30 and 1 <= args.layers <= 32 and 3 <= args.tokens <= 512 and 1 <= args.heads <= 32,
            'minimum 30 repeats and bounded CPU inputs required')
    import torch
    require(torch.version.cuda is None or not torch.cuda.is_initialized(), 'CUDA must remain uninitialized')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    n, h, layers = args.tokens, args.heads, args.layers
    prob = torch.rand(1, h, n, n, dtype=torch.float32, device='cpu')
    windows = SubsequenceExtractor(3).extract(list(range(n)))
    padding = torch.ones(1, n, device='cpu')
    padding[:, 1::7] = 0
    req = dict(request_id='CPU_BENCH_ONLY', token_ids=list(range(n)))
    def layer(mode, mask):
        double = prob.detach().double()  # Actual streaming path casts once, not per window.
        if mode == l2c.REFERENCE:
            reducer = PaperRowL2SumReducer()
            return [reducer.reduce([double], w.start, w.end, mask) for w in windows]
        fn = l2c.deferred_window_values if mode == l2c.HOST_BATCH else l2e.reused_window_values
        return fn(double, windows, mask)
    def total(mode):
        cls = {l2c.REFERENCE: StreamingImpact, l2c.HOST_BATCH: l2c.DeferredHostImpact,
               l2e.MASK_INDEX_REUSE: l2e.IndexReuseImpact}[mode]
        stream = cls(req, layers, windows)
        for i in range(layers):
            stream.observe(i, prob)
        return list(stream.values.values())
    cases = []
    for label, mask in (('NO_PADDING', None), ('PADDING_2D', padding), ('STREAM_32_LAYERS' if layers == 32 else 'STREAM_LAYERS', None)):
        run = (lambda mode: total(mode)) if label.startswith('STREAM') else (lambda mode: layer(mode, mask))
        expected = bit_hash(run(l2c.REFERENCE))
        for _ in range(3):
            for mode in MODES:
                require(bit_hash(run(mode)) == expected, 'warmup FP64 mismatch')
        timings = {mode: [] for mode in MODES}
        raw_samples = []
        for repeat in range(args.repeats):
            order = MODES[repeat % 3:] + MODES[:repeat % 3]
            sample = dict(repeat=repeat, order=list(order), ms={})
            for mode in order:
                tick = time.perf_counter_ns()
                values = run(mode)
                elapsed = (time.perf_counter_ns() - tick) / 1e6
                require(bit_hash(values) == expected, 'timed FP64 mismatch')
                timings[mode].append(elapsed)
                sample['ms'][mode] = elapsed
            raw_samples.append(sample)
            if (repeat + 1) % 5 == 0:
                print(f'{label}: {repeat + 1}/{args.repeats}', flush=True)
        cases.append(dict(case=label, timing_ms={k: describe(v) for k, v in timings.items()},
                          raw_samples=raw_samples, fp64_sha256=expected,
                          window_count=len(windows), layers=layers if label.startswith('STREAM') else 1,
                          amortized_window_ms={k: statistics.median(v) / len(windows) /
                              (layers if label.startswith('STREAM') else 1) for k, v in timings.items()}))
    # Keep observers outside all primary performance samples.
    observers = []
    for label, mask in (('NO_PADDING', None), ('PADDING_2D', padding)):
        for mode in MODES:
            times, walls, counts = [], [], None
            for _ in range(args.repeats):
                values, observed_counts, component_times, wall = operation_observer(lambda: layer(mode, mask), mask)
                require(counts is None or counts == observed_counts, 'nondeterministic API census')
                counts = observed_counts
                times.append(component_times)
                walls.append(wall)
            mem = memory_profile(lambda: layer(mode, mask))
            observers.append(dict(case=label, implementation=mode, python_api_counts=counts,
                observed_wall_ms=describe(walls), component_inclusive_ms={k: describe([r.get(k, 0.) for r in times])
                    for k in sorted({k for r in times for k in r})}, memory=mem,
                observer_scope='30 separate patched observations; wall includes patch install/restore; '
                               'component times include timer/wrapper cost and may nest; profiler is separate'))
            print(f'CPU observer complete: {label} {mode}', flush=True)
    require(not torch.cuda.is_initialized(), 'CPU benchmark initialized CUDA')
    files = [Path(__file__), ROOT / 'src/semcache/edgelora/r5_d5_impact.py',
             ROOT / 'src/semcache/edgelora/r5_d5_impact_compute.py',
             ROOT / 'src/semcache/cache/attention_impact.py',
             ROOT / 'src/semcache/edgelora/r5_subsequence_generation.py']
    return dict(schema='D5_L2E_NATIVE_CPU_COMPUTE_BENCHMARK_V1', CPU_ONLY=True,
                source_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                source_files_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
                versions={m: importlib.metadata.version(m) for m in ('torch', 'transformers', 'numpy')},
                threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads(),
                affinity=sorted(os.sched_getaffinity(0)), input=dict(shape=list(prob.shape), dtype=str(prob.dtype),
                    contiguous=prob.is_contiguous(), probability_bytes=prob.numel()*prob.element_size(), seed=42),
                layers=layers, repeats=args.repeats, warmup=3, cases=cases, observers=observers,
                gpu='NOT_RUN', cpu_approval='NOT_CREATED',
                limitations='synthetic real PyTorch CPU attention; same tensor reused across 32 layers; '
                'compute times do not include a model/communication/owner/E2E; window mean is amortized, not isolated norm time; '
                'CPU clocks/affinity not pinned; no GPU gain inference; memory profiler excludes Python/allocator overhead')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--repeats', type=int, default=30)
    parser.add_argument('--tokens', type=int, default=512)
    parser.add_argument('--heads', type=int, default=32)
    parser.add_argument('--layers', type=int, default=32)
    args = parser.parse_args()
    require(not args.output.exists(), 'fresh CPU diagnostic output required')
    result = benchmark(args)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print(json.dumps(dict(status='NATIVE_CPU_BENCHMARK_COMPLETE', output=str(args.output), gpu='NOT_RUN')))


if __name__ == '__main__':
    main()
