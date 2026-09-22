"""CPU-only synthetic Q/K/V LoRA microbenchmark; imports torch only on invocation."""
from contextlib import contextmanager
from datetime import datetime, timezone
import os
from pathlib import Path
import platform
from statistics import mean, median
import time
from .common import integer, nonnegative


def percentile95(values):
    ordered = sorted(values)
    pos = .95 * (len(ordered)-1)
    lower = int(pos)
    return ordered[lower] + (ordered[min(lower+1, len(ordered)-1)]-ordered[lower])*(pos-lower)


def timing_summary(raw):
    if not raw:
        raise ValueError('Empty calibration')
    for value in raw:
        nonnegative(value, 'raw timing')
    return dict(mean_ms=mean(raw), p50_ms=median(raw), p95_ms=percentile95(raw))


def cpu_info():
    model, frequencies = platform.processor() or None, []
    try:
        for line in Path('/proc/cpuinfo').read_text().splitlines():
            if line.startswith('model name'):
                model = line.split(':', 1)[1].strip()
            elif line.startswith('cpu MHz'):
                frequencies.append(float(line.split(':', 1)[1]))
    except (OSError, ValueError):
        pass
    return dict(cpu_model=model, observed_cpu_frequency_mhz=(
        dict(min=min(frequencies), max=max(frequencies), mean=mean(frequencies),
             scope='all /proc/cpuinfo logical CPUs; instantaneous reported frequency, not fixed frequency')
        if frequencies else None))


@contextmanager
def cpu_resources(torch):
    original_threads = torch.get_num_threads()
    original_affinity = None
    affinity = dict(requested_logical_cpus=4, applied=False, selected_logical_cpus=None, error=None)
    try:
        if hasattr(os, 'sched_getaffinity') and hasattr(os, 'sched_setaffinity'):
            try:
                original_affinity = os.sched_getaffinity(0)
                chosen = sorted(original_affinity)[:4]
                os.sched_setaffinity(0, chosen)
                affinity.update(applied=True, selected_logical_cpus=sorted(os.sched_getaffinity(0)),
                                four_cpus_available=len(chosen) == 4)
            except OSError as exc:
                affinity['error'] = str(exc)
        else:
            affinity['error'] = 'CPU affinity API unavailable'
        # Restrict affinity before constructing/resizing the intra-op worker pool.
        torch.set_num_threads(4)
        yield affinity
    finally:
        try:
            if original_affinity is not None and affinity['applied']:
                os.sched_setaffinity(0, original_affinity)
        finally:
            torch.set_num_threads(original_threads)


def calibrate(dims, sequence_lengths, *, warmup=5, measured=20, seed=42, host_label='SERAPH'):
    integer(warmup, 'warmup', 1)
    integer(measured, 'measured', 1)
    lengths = sorted(set(sequence_lengths))
    if not lengths:
        raise ValueError('Supply at least one sequence length')
    for n in lengths:
        integer(n, 'sequence length', 1)
    import torch
    # No model, CUDA, Transformers, adapter loading, networking or training.
    with cpu_resources(torch) as affinity:
        generator = torch.Generator(device='cpu').manual_seed(seed)
        weights = [(torch.randn(dims.hidden_size, dims.rank, generator=generator, dtype=torch.float32, device='cpu'),
                    torch.randn(dims.rank, dims.hidden_size, generator=generator, dtype=torch.float32, device='cpu'))
                   for _ in ('q_proj', 'k_proj', 'v_proj')]
        before = cpu_info()
        samples = []
        with torch.inference_mode():
            for n in lengths:
                x = torch.randn(n, dims.hidden_size, generator=generator, dtype=torch.float32, device='cpu')
                def projection():
                    return [(x @ a) @ b for a, b in weights]
                for _ in range(warmup):
                    projection()
                timings = []
                for _ in range(measured):
                    start = time.perf_counter_ns()
                    output = projection()
                    timings.append((time.perf_counter_ns()-start)/1_000_000)
                    del output  # output destruction excluded from the recorded interval
                samples.append(dict(sequence_length=n, raw_timings_ms=timings,
                    **timing_summary(timings), provenance='CALIBRATED',
                    scope='one_layer_three_LoRA_branches_six_matmuls'))
        after = cpu_info()
    return dict(schema='m9a_ud_lora_calibration_v1', **dims.record(), dtype='float32',
        parameter_element_bytes=4, temporary_element_bytes=4, num_threads=4,
        warmup_count=warmup, measured_count=measured, seed=seed,
        cpu_model=before['cpu_model'], observed_cpu_frequency_mhz=before['observed_cpu_frequency_mhz'],
        cpu_observation_after=after, affinity=affinity, hostname=platform.node(),
        host_label=host_label, host_label_source='operator_declared_not_hardware_equivalence',
        timestamp=datetime.now(timezone.utc).isoformat(), torch_version=torch.__version__,
        provenance='CALIBRATED', calibration_provenance=f'CALIBRATED_ON_{host_label}_CPU',
        paper_ud_equivalence_claimed=False, target_modules=['q_proj', 'k_proj', 'v_proj'],
        measurement_scope='CPU FP32 matmuls and output allocation; no full model',
        excluded='input/output embedding/logit computation, wire dtype conversion, network, trained adapters',
        layer_extrapolation='per-layer timing multiplied by layer count; REPRODUCTION_CHOICE',
        samples=samples)


def calibration_latency(artifact, dims, tokens):
    """Require exact length calibration; no hidden linear interpolation."""
    integer(tokens, 'fresh tokens')
    if artifact.get('provenance') != 'CALIBRATED':
        raise ValueError('UD artifact must have CALIBRATED provenance')
    for key, value in dims.record().items():
        if artifact.get(key) != value:
            raise ValueError(f'Calibration dimension mismatch: {key}')
    if artifact.get('num_threads') != 4 or artifact.get('dtype') != 'float32':
        raise ValueError('M9-A requires four-thread FP32 CPU calibration')
    if tokens == 0:
        return dict(value_ms=0., provenance='SIMULATED', source='zero fresh tokens; no LoRA calls',
                    dependency_provenance=[])
    matches = [s for s in artifact.get('samples', []) if s.get('sequence_length') == tokens]
    if len(matches) != 1:
        raise ValueError(f'Need exactly one CPU calibration for sequence_length={tokens}; no interpolation')
    sample = matches[0]
    if sample.get('provenance') != 'CALIBRATED':
        raise ValueError('Calibration sample has invalid provenance')
    raw = sample.get('raw_timings_ms', [])
    if len(raw) != artifact.get('measured_count'):
        raise ValueError('Calibration raw count mismatch')
    summary = timing_summary(raw)
    for field, expected in summary.items():
        if abs(nonnegative(sample.get(field), field)-expected) > 1e-8 * max(1, expected):
            raise ValueError(f'Calibration summary does not match raw samples: {field}')
    return dict(value_ms=summary['mean_ms']*dims.layers, provenance='CALIBRATED',
                source='exact-length CPU mean times layer count; REPRODUCTION_CHOICE extrapolation',
                dependency_provenance=['CALIBRATED'])
