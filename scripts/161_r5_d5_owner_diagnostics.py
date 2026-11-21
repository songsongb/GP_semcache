#!/usr/bin/env python3
"""Repository-only stdlib CPU harness; uses existing explicit ArrayTensor fixtures.

No GPU launcher, model download, native CPU approval, or optimization candidate.
"""
import argparse
from contextlib import nullcontext
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tests')]
import test_r5_subsequence_validation as fixtures
from semcache.cache.admission import AdmissionPolicy
from semcache.edgelora.communication import digest, require
from semcache.edgelora.r5_subsequence_physical import PhysicalOwner
from semcache.edgelora.r5_subsequence_integrity import SCOPED
from semcache.edgelora.r5_subsequence_profile import Profiler
from semcache.edgelora.r5_d5_owner_diagnostics import CPUOwnerDiagnostics

SCENARIOS = ('ADMIT', 'CAPACITY_REJECT', 'POLICY_REJECT', 'REUSE', 'EVICTION',
             'SELF_EVICTION', 'CHU_PBR', 'ALL_PRESENT')


def stable(value):
    """Remove only observational timing; retain all semantic/validation fields."""
    if isinstance(value, dict):
        return {(digest(k) if isinstance(k, tuple) else k): stable(v) for k, v in value.items()
                if k != 'costs' and not (isinstance(k, str) and k.endswith(('_ms', '_ns')))}
    if isinstance(value, (list, tuple)):
        return [stable(v) for v in value]
    return value


def run_fixture(scenario, observer=None):
    require(scenario in SCENARIOS, 'unknown fixture')
    f = fixtures.Validation()
    f.setUp()
    events = []
    owner = PhysicalOwner(f.f.ident, f.f.registry, validation_mode=SCOPED,
                          capacity=1 if scenario == 'CAPACITY_REJECT' else 288 if scenario in ('EVICTION', 'SELF_EVICTION') else 2**20,
                          trace=events.append)
    if scenario == 'POLICY_REJECT':
        owner.cache.admission = AdmissionPolicy(threshold=1.)
    if scenario in ('REUSE', 'CHU_PBR', 'ALL_PRESENT'):
        owner.begin(f.f.req('S', [1, 2, 3, 4, 5, 6]), {'cluster_id': 0})
        f.project_all(owner, 6)
        f.complete(owner)
        events.clear()
    if scenario == 'CHU_PBR':
        owner.bridge.pbr_interval = 1
    n = 8 if scenario in ('REUSE', 'CHU_PBR') else 6
    context = (observer.observe('CPU_OWNER_FIXTURE') if isinstance(observer, CPUOwnerDiagnostics)
               else observer.activate('CPU_OWNER_FIXTURE') if observer is not None else nullcontext())
    started = time.perf_counter_ns()
    with context:
        ids = list(range(1, n + 1))
        owner.begin(f.f.req('T', ids[::-1] if scenario == 'SELF_EVICTION' else ids), {'cluster_id': 0})
        f.project_all(owner, n)
        complete_start = time.perf_counter_ns()
        completion = f.complete(owner)
        completion_ms = (time.perf_counter_ns() - complete_start) / 1e6
    elapsed_ms = (time.perf_counter_ns() - started) / 1e6
    state = dict(completion=stable(completion), events=stable(events),
                 history=list(owner.bridge.metrics.history.history[0]),
                 hits=owner.cache.hits, misses=owner.cache.misses,
                 occupancy=owner.cache.physical_tensor_bytes,
                 logical_bytes=owner.cache.logical_cache_bytes,
                 entries=[dict(key=e.key, metadata=e.qkv_metadata, positions=e.positions,
                               frequency=e.frequency, impact=e.impact, created_at=e.created_at,
                               updated_at=e.updated_at, last_access=e.last_access,
                               proof=owner.proofs[e.key]) for e in owner.cache.entries.values()])
    report = observer.report() if isinstance(observer, CPUOwnerDiagnostics) else (
        observer.finish() if observer is not None else None)
    owner.dispose()
    return dict(semantic_sha256=digest(stable(state)), completion_ms=completion_ms, elapsed_ms=elapsed_ms,
                reused_rows=completion['physical_projection_rows_reused'],
                fresh_rows=sum(len(r['fresh_positions']) for r in completion['records']),
                cache_entries=len(state['entries']), occupancy_bytes=state['occupancy'],
                integrity=stable({k: v for k, v in completion['integrity_validation'].items()
                                  if k != 'authorizations'}), diagnostics=report)


def summary(values):
    return dict(median=statistics.median(values), minimum=min(values), maximum=max(values),
                p25=statistics.quantiles(values, n=4)[0], p75=statistics.quantiles(values, n=4)[2])


def benchmark(repeats):
    require(4 <= repeats <= 100, 'bounded repeats in [4,100] required')
    rows = []
    for scenario in SCENARIOS:
        timings = {mode: [] for mode in ('OFF', 'EXISTING_PROFILER', 'OWNER_DIAGNOSTICS')}
        completions = {mode: [] for mode in timings}
        expected = run_fixture(scenario)['semantic_sha256']  # warmup
        sample = None
        diagnostic_runs = []
        for repeat in range(repeats):
            modes = list(timings)
            # Rotate order to reduce fixed-order warming/drift bias.
            modes = modes[repeat % 3:] + modes[:repeat % 3]
            for mode in modes:
                observer = CPUOwnerDiagnostics() if mode == 'OWNER_DIAGNOSTICS' else Profiler() if mode == 'EXISTING_PROFILER' else None
                result = run_fixture(scenario, observer)
                require(result['semantic_sha256'] == expected, 'observer changed fixture semantics')
                timings[mode].append(result['elapsed_ms'])
                completions[mode].append(result['completion_ms'])
                if mode == 'OWNER_DIAGNOSTICS':
                    sample = result
                    diagnostic_runs.append(result['diagnostics'])
        rows.append(dict(scenario=scenario, semantic_parity='PASS_ARRAY_FIXTURE_ONLY',
                         whole_observation_ms={k: summary(v) for k, v in timings.items()},
                         completion_wall_ms={k: summary(v) for k, v in completions.items()},
                         completion_bins_ms={name: summary([r['completion_ms'][name] for r in diagnostic_runs])
                                             for name in diagnostic_runs[0]['completion_ms']},
                         sample=sample))
    source_files = [Path(__file__), ROOT / 'src/semcache/edgelora/r5_d5_owner_diagnostics.py',
                    ROOT / 'tests/test_r5_d5_owner_diagnostics.py',
                    ROOT / 'src/semcache/edgelora/r5_subsequence_physical.py',
                    ROOT / 'src/semcache/edgelora/r5_subsequence_integrity.py',
                    ROOT / 'src/semcache/edgelora/r5_subsequence_profile.py']
    return dict(schema='D5_L2D_CPU_FIXTURE_BENCHMARK_V1', TEST_FIXTURE_ONLY=True,
                source_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                source_files_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
                python=sys.version, dependencies={m: importlib.util.find_spec(m) is not None
                    for m in ('torch', 'transformers', 'numpy', 'pytest', 'yaml')}, repeats=repeats,
                candidate=None, native_cpu='BLOCKED' if importlib.util.find_spec('torch') is None else 'NOT_RUN',
                gpu='NOT_RUN', cpu_approval='NOT_CREATED', rows=rows,
                limitations='fixed tiny CPU ArrayTensor fixture; synthetic impact=1 (no attention arithmetic validation); '
                'no generated token/logit parity in this benchmark (separate scalar tests); '
                'OFF vs profiler vs diagnostics measures observer cost, not a candidate speedup; '
                'whole observation includes wrapper installation/restoration, completion excludes those; '
                'proof timing includes observer and Python timer overhead; no native CPU/GPU inference')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=20)
    args = parser.parse_args()
    require(not args.output.exists(), 'diagnostic output must be fresh')
    result = benchmark(args.repeats)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print(json.dumps(dict(status='CPU_FIXTURE_OBSERVATION_COMPLETE', candidate=None,
                          native_cpu=result['native_cpu'], gpu='NOT_RUN', output=str(args.output))))


if __name__ == '__main__':
    main()
