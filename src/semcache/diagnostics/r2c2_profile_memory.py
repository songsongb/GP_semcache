"""Profile-only empirical capacity guard; never a workload or executor override."""
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

GIB = 1024**3
PILOT_MEMORY_SCHEMA = 'r2c2-peak-across-resets-v1'


class ProfileMemoryRejected(MemoryError):
    def __init__(self, audit):
        self.audit = audit
        super().__init__(audit['reason'])


def byte_count(value, name, positive=False):
    if isinstance(value, bool) or not isinstance(value, int) or value < (1 if positive else 0):
        raise ValueError(f'missing/invalid {name}: integer bytes required')
    return value


def uuid(value):
    value = str(value or '').strip().lower().removeprefix('gpu-')
    if value in ('', 'none', 'null', 'unavailable'):
        raise ValueError('actual GPU UUID unavailable')
    return value


def pilot_evidence(directory, current, length, coverage):
    """Require sealed fresh C1 evidence, not a hand-edited approval/old peak."""
    directory = Path(directory)
    paths = [directory / n for n in ('manifest.json', 'capacity_pilot.json')]
    manifest, pilot = [json.loads(p.read_text()) for p in paths]
    key = f'L{length}_cov{coverage}'
    for name, value in (('manifest', manifest), ('capacity pilot', pilot)):
        if value.get('experiment') != 'R2-C1' or value.get('execution_status') != 'COMPLETE':
            raise ValueError(f'{name} is not a completed R2-C1 Pilot')
        if value.get('environment_identity') != current.get('environment_identity'):
            raise ValueError(f'{name} environment/code/model/adapter identity mismatch')
    if manifest.get('benchmark_config', {}).get('phase') != 'pilot':
        raise ValueError('input is not a Pilot phase')
    if manifest.get('GPU_VALIDATION') != 'EXECUTED':
        raise ValueError('Pilot GPU execution not verified')
    if not manifest.get('final_weights_verified') or not pilot.get('model_weights_verified'):
        raise ValueError('Pilot weights not verified')
    if uuid(manifest.get('hardware_identity', {}).get('uuid')) != uuid(current.get('hardware_identity', {}).get('uuid')):
        raise ValueError('Pilot actual GPU UUID mismatch')
    if (not current.get('implementation_sha256') or
        manifest.get('implementation_sha256') != current['implementation_sha256'] or
        not current.get('code', {}).get('sha256') or
        manifest.get('code', {}).get('sha256') != current['code']['sha256']):
        raise ValueError('Pilot implementation/code hashes mismatch; run a fresh Pilot')
    if not current.get('model_weight_identity') or manifest.get('model_weight_identity') != current['model_weight_identity']:
        raise ValueError('Pilot model/adapter weights mismatch')
    for field in ('model', 'dtype', 'capacity_mib', 'cache_device', 'seed'):
        expected = current.get('benchmark_config', {}).get(field)
        if expected is None or manifest.get('benchmark_config', {}).get(field) != expected:
            raise ValueError(f'Pilot {field} mismatch')
    if not current.get('settings') or manifest.get('settings') != current['settings']:
        raise ValueError('Pilot adapter/backend/output/tolerance settings mismatch')
    condition = pilot.get('conditions', {}).get(key)
    if not condition or condition != manifest.get('conditions', {}).get(key):
        raise ValueError('Pilot condition missing or differs between manifest and capacity evidence')
    if (condition.get('status') != 'CAPACITY_VALIDATED' or not condition.get('structural_valid') or
        condition.get('length') != length or condition.get('requested_coverage') != coverage or
        not condition.get('condition_identity')):
        raise ValueError('Pilot condition is not structurally/capacity approved')
    measured = condition.get('pilot_memory', {})
    if measured.get('schema') != PILOT_MEMORY_SCHEMA or measured.get('status') != 'MEASURED':
        raise ValueError('Pilot lacks whole-case allocated peak/baseline evidence; run a fresh Pilot')
    baseline = byte_count(measured.get('baseline_allocated_bytes'), 'Pilot baseline', True)
    peak = byte_count(measured.get('peak_allocated_bytes'), 'Pilot peak', True)
    total = byte_count(measured.get('total_bytes'), 'Pilot total memory', True)
    if peak <= baseline or peak > total or measured.get('allocation_increase_bytes') != peak - baseline:
        raise ValueError('inconsistent Pilot peak/baseline/increase')
    if byte_count(measured.get('samples'), 'Pilot memory samples', True) < 2:
        raise ValueError('insufficient Pilot memory observations')
    return dict(directory=str(directory.resolve()), case_id=key,
        files_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        gpu_uuid=manifest['hardware_identity']['uuid'], model_weight_identity=manifest['model_weight_identity'],
        condition_identity=condition['condition_identity'], memory=measured)


def preflight(policy, conservative, gpu, evidence=None, profiler_allowance_bytes=GIB,
              safety_reserve_bytes=2*GIB, evidence_error=None):
    """Return both budgets, without CUDA allocation or altering physical checks.

    Headroom = driver free + reusable allocator reserve. The empirical budget is
    the entire Pilot increase above its source-free model/encoder baseline, plus
    allowances. Charging the entire increase even with an already resident source
    is deliberately conservative; it never subtracts assumed source residency.
    """
    audit = dict(policy=policy, conservative=conservative, current_gpu=gpu, pilot=evidence,
        guarded=None, approved=False, reason=None, evidence_error=evidence_error)
    try:
        if policy not in ('conservative', 'pilot_guarded'):
            raise ValueError('unsupported profile memory policy')
        allocated = byte_count(gpu.get('allocated_bytes'), 'current allocation')
        reserved = byte_count(gpu.get('reserved_bytes'), 'current reserve')
        free = byte_count(gpu.get('free_bytes'), 'current driver free')
        total = byte_count(gpu.get('total_bytes'), 'current total', True)
        if allocated > reserved or free > total or reserved > total or free + reserved > total:
            raise ValueError('inconsistent current GPU memory evidence')
        available = free + reserved - allocated
        audit['available_with_allocator_bytes'] = available
        required = byte_count(conservative.get('estimated_additional_bytes'), 'conservative estimate', True)
        if policy == 'pilot_guarded':
            if evidence_error or not evidence:
                raise ValueError(evidence_error or 'missing compatible Pilot memory evidence')
            measured = evidence['memory']
            baseline = byte_count(measured.get('baseline_allocated_bytes'), 'Pilot baseline', True)
            peak = byte_count(measured.get('peak_allocated_bytes'), 'Pilot peak', True)
            increase = byte_count(measured.get('allocation_increase_bytes'), 'Pilot increase', True)
            if peak - baseline != increase or peak > total or measured.get('total_bytes') != total:
                raise ValueError('inconsistent Pilot/current memory measurements')
            allowance = byte_count(profiler_allowance_bytes, 'profiler allowance', True)
            reserve = byte_count(safety_reserve_bytes, 'safety reserve', True)
            required = increase + allowance + reserve
            audit['guarded'] = dict(pilot_baseline_allocated_bytes=baseline, pilot_peak_allocated_bytes=peak,
                pilot_allocation_increase_bytes=increase, profiler_allowance_bytes=allowance,
                safety_reserve_bytes=reserve, estimated_additional_bytes=required,
                margins_provisional=True, formula='(pilot_peak - pilot_baseline) + profiler_allowance + safety_reserve')
        audit['estimated_additional_required_bytes'] = required
        audit['approved'] = required <= available
        audit['reason'] = ('APPROVED' if audit['approved'] else 'INSUFFICIENT_GPU_HEADROOM')
    except (KeyError, TypeError, ValueError) as exc:
        audit['reason'] = f'INVALID_MEMORY_EVIDENCE: {exc}'
    return audit


class PilotPeakRecorder:
    """Pilot-only, reset-aware allocator observations; no new synchronization.

    Frozen target requests reset peak counters. Sample *before* every such reset
    to preserve source preparation and earlier request peaks. Wrapping is absent
    from latency/profile phases. Carry the length baseline/maximum across coverage
    cases sharing a source, rather than calling their source-resident state a new
    baseline. This is an upper bound across those completed same-length cases.
    """
    def __init__(self, device):
        self.device = device
        self.active = False

    def begin(self, baseline, previous=None):
        import torch
        self.torch = torch
        self.baseline = dict(baseline['gpu'])
        self.peak = (previous or {}).get('peak_allocated_bytes', self.baseline['allocated_bytes'])
        self.samples = 0
        original = torch.cuda.reset_peak_memory_stats
        def reset(device=None):
            selected = torch.device(f'cuda:{device}' if isinstance(device,int) else device if device is not None else f'cuda:{torch.cuda.current_device()}')
            target = torch.device(self.device)
            if selected.index is None: selected = torch.device(f'cuda:{torch.cuda.current_device()}')
            if target.index is None: target = torch.device(f'cuda:{torch.cuda.current_device()}')
            if selected == target: self.sample()
            return original(device)
        self.patcher = patch.object(torch.cuda, 'reset_peak_memory_stats', reset)
        self.patcher.start()
        self.active = True
        self.sample()

    def sample(self):
        self.peak = max(self.peak, self.torch.cuda.max_memory_allocated(self.device),
                        self.torch.cuda.memory_allocated(self.device))
        self.samples += 1

    def finish(self):
        try:
            self.sample()
            baseline = self.baseline['allocated_bytes']
            return dict(schema=PILOT_MEMORY_SCHEMA, status='MEASURED',
                baseline_allocated_bytes=baseline, peak_allocated_bytes=self.peak,
                allocation_increase_bytes=self.peak-baseline, total_bytes=self.baseline['total_bytes'],
                samples=self.samples, scope='source-free length baseline; source build + gates + requests; peak preserved across request resets and same-length coverage cases')
        finally:
            self.close()

    def close(self):
        if self.active:
            self.patcher.stop()
            self.active = False


def failure_status(exc):
    """Actual Torch OOM has priority over screens; trace failures stay distinct."""
    try:
        import torch
        if isinstance(exc, torch.cuda.OutOfMemoryError): return 'OOM'
    except (ImportError, AttributeError):
        pass
    if 'out of memory' in str(exc).lower(): return 'OOM'
    if isinstance(exc, MemoryError): return 'MEMORY_PREFLIGHT_FAILED'
    if 'TRACE_' in str(exc): return 'TRACE_FAILED'
    return 'INVALID_EXECUTION'


def validation_report(directory, expected_code_sha256=None, expected_uuid=None):
    """Used by Slurm validation: report original case errors, never bare assert."""
    directory = Path(directory)
    manifest = json.loads((directory/'manifest.json').read_text())
    issues = []
    if manifest.get('execution_status') != 'COMPLETE':
        issues.append(f"execution_status={manifest.get('execution_status')}")
    if expected_code_sha256 and manifest.get('code', {}).get('sha256') != expected_code_sha256:
        issues.append('execution code hash differs from launcher identity')
    try:
        if expected_uuid and uuid(manifest.get('hardware_identity', {}).get('uuid')) != uuid(expected_uuid):
            issues.append('physical GPU UUID changed across processes')
    except ValueError as exc:
        issues.append(str(exc))
    failures = {key: {k: value.get(k) for k in ('status','context','failure_type','failure','traceback',
        'memory_preflight_audit','preflight','memory_before','memory_at_failure','pilot_memory')}
        for key,value in manifest.get('conditions', {}).items() if value.get('status') not in ('VALID','CAPACITY_VALIDATED')}
    if failures: issues.append('one or more failed/unexecuted cases')
    if not manifest.get('final_weights_verified'):
        issues.append('final model/adapter weights unverified')
    if any(not value.get('structural_valid') for value in manifest.get('conditions', {}).values()):
        issues.append('structural gate missing/failed')
    if manifest.get('failure'): issues.append(manifest['failure'])
    if manifest.get('benchmark_config', {}).get('phase') == 'profile':
        path = directory/'profiling_status.json'
        profiles = json.loads(path.read_text()) if path.exists() else {}
        from semcache.diagnostics.r2c1_workloads import MODES
        expected = {key+'_'+mode for key in manifest.get('conditions', {}) for mode in MODES}
        if set(profiles) != expected or any(p.get('status') != 'PROFILE_COLLECTED' for p in profiles.values()):
            issues.append('profiler evidence missing/unavailable/incomplete')
    return dict(valid=not issues, directory=str(directory), execution_status=manifest.get('execution_status'),
        issues=issues, failed_cases=failures, global_failure=manifest.get('failure'),
        code_sha256=manifest.get('code', {}).get('sha256'), gpu_uuid=manifest.get('hardware_identity', {}).get('uuid'))
