"""CPU-first D5 Light preparation and explicitly gated native ES/UD execution."""
from pathlib import Path
import argparse
import csv
import importlib.util
import io
import json
import os
import sys
import time
import unittest

from .communication import digest, require
from .r3a_artifacts import ROOT, THRESHOLDS
from .r5_d5_lightperf_io import (SCHEMA, MODE, ENGINE, BRANCH, SummaryWriter, fresh, hidden_cuda,
                               source_identity, read_json, verify)
from .r5_d5_lightperf_workload import DATASETS, COUNTS, CONDITIONS, knobs, stream, preflight, matrix

TEST_MODULES = ('test_r5_d5_lightperf', 'test_r5_d5_lightperf_worker', 'test_r5_d5_lightperf_native',
                'test_r5_natural_shutdown', 'test_r5_natural_phase3a',
                'test_r5_subsequence_generation', 'test_r5_subsequence_validation',
                'test_r5_sparse_physical', 'test_r5_subsequence_audit', 'test_r5_natural_approx')


def dependencies():
    return {n: importlib.util.find_spec(n) is not None for n in ('torch', 'transformers', 'numpy')}


def material(original_experiment, multi, dataset, count, performance_knobs, *,
             continuous_contract=None, strict_pilot=None, ud_operation_seconds=None, cpu_only=True):
    if cpu_only:
        hidden_cuda()
    require(all(dependencies().values()), 'BLOCKED_MISSING_NATIVE_DEPENDENCIES: ' + str(dependencies()))
    require(os.environ.get('HF_HUB_OFFLINE') == os.environ.get('TRANSFORMERS_OFFLINE') == '1',
            'offline-only input/tokenizer validation required')
    from .r5_subsequence_natural import inputs, transport
    from .r5_i5_data import DATASETS as I5_DATASETS
    from .r5_subsequence_natural_io import verify as verify_strict
    from .r5_i5r2_tensors import Evidence
    w, d, receipt, inv, m, pm, specs, users, tok = inputs(original_experiment, multi, _datasets=I5_DATASETS)
    from .r3a_artifacts import versions
    require(versions() == pm['identity']['versions'], 'prepared/runtime native versions differ')
    pilot_tasks, pilot_inventory = None, None
    if strict_pilot is not None:
        pilot_inventory = verify_strict(strict_pilot)
        pilot = json.loads((Path(strict_pilot) / 'approx_plan.json').read_text())
        require(pilot['track'] == 'NATURAL' and len(pilot['tasks']) == 5 and
                all(t['dataset'] == dataset for t in pilot['tasks']), 'incompatible frozen Strict pilot')
        pilot_tasks = pilot['tasks']
    selected = stream(w, dataset, count, continuous_contract=continuous_contract, strict_pilot_tasks=pilot_tasks)
    network = transport(selected['tasks'], pm, specs, tok.identity, users)
    from .r5_i5_quality import validate_scorers
    validate_scorers({dataset})
    # Verify the exact local semantic checkpoint/weights during CPU preflight too.
    # This is a real CPU encoder, never a controlled fixture substitution.
    from .r5_runtime import encoder_for
    from .r5_perf_schedule import TINYBERT_REVISION
    encoder = encoder_for(dict(fixture=False, semantic_revision=TINYBERT_REVISION), selected['tasks'])
    semantic_identity = encoder.metadata
    del encoder
    return dict(stream=selected, transport=network, original_inventory=receipt, dataset_inventories=inv,
                multi_inventory=Evidence(multi).verify('r5_output_manifest.json'),
                dataset_paths={name: v['prepared_path'] for name, v in d.items()},
                dataset_split=d[dataset]['split'], dataset_revision=d[dataset]['revision'],
                seed=w['seed'], base_prepared=m['base_prepared'], tokenizer_identity=tok.identity,
                semantic_identity=semantic_identity,
                strict_pilot_inventory=pilot_inventory,
                resources=preflight(selected['tasks'], performance_knobs, ud_operation_seconds=ud_operation_seconds))


def prepare(original_experiment, multi, dataset, count, output_dir, *, bandwidth_mbps=None,
            ud_throughput_factor=1., development=False, reviewed_commit=None,
            reviewed_fingerprint=None, continuous_contract=None, strict_pilot=None,
            ud_operation_seconds=None):
    hidden_cuda()
    source = source_identity(development=development, reviewed_commit=reviewed_commit,
                             reviewed_fingerprint=reviewed_fingerprint)
    performance_knobs = knobs(bandwidth_mbps, ud_throughput_factor)
    contract = None if continuous_contract is None else read_json(continuous_contract)
    fields = material(original_experiment, multi, dataset, count, performance_knobs,
                      continuous_contract=contract, strict_pilot=strict_pilot, ud_operation_seconds=ud_operation_seconds)
    p = dict(schema=SCHEMA, mode=MODE, engine=ENGINE, source=source, dataset=dataset, count=count,
             original_experiment=str(Path(original_experiment).resolve()), multi=str(Path(multi).resolve()),
             continuous_contract=contract, strict_pilot=None if strict_pilot is None else str(Path(strict_pilot).resolve()),
             ud_operation_seconds=ud_operation_seconds, performance_knobs=performance_knobs,
             conditions=list(CONDITIONS), approx_requires_explicit_run_opt_in=True,
             validation_mode='REQUEST_SCOPED_VERIFIED', thresholds=THRESHOLDS, warmup_requests=0,
             gpu_launch_status='REQUIRES_FRESH_NATIVE_CPU_APPROVAL_AND_MANUAL_GPU_OPT_IN',
             correctness_qualified_speedup='NOT_ESTABLISHED', **fields)
    p['plan_id'] = digest(p)
    protected = [original_experiment, multi, *p['dataset_paths'].values()]
    if continuous_contract:
        protected.append(continuous_contract)
    if strict_pilot:
        protected.append(strict_pilot)
    writer = SummaryWriter(fresh(output_dir, protected))
    writer.write('plan.json', p)
    inventory = writer.seal()
    return dict(status='CPU_PLAN_ONLY_GPU_BLOCKED', plan_id=p['plan_id'], inventory=inventory,
                dataset=dataset, count=count, strict_parity=p['stream']['strict_five_parity'], resources=p['resources'])


def load_plan(experiment, *, cpu_only=True):
    inventory = verify(experiment)
    p = read_json(Path(experiment) / 'plan.json')
    body = dict(p)
    pid = body.pop('plan_id')
    require(p['schema'] == SCHEMA and p['mode'] == MODE and p['engine'] == ENGINE and digest(body) == pid,
            'Light plan identity/engine differs; regenerate old plans')
    src = p['source']
    current = source_identity(development=src['approval_scope'] == 'DEVELOPMENT_NOT_GPU_APPROVED',
                              reviewed_commit=src['commit'], reviewed_fingerprint=src['content_fingerprint'])
    require(current == src and p['conditions'] == list(CONDITIONS) and p['validation_mode'] == 'REQUEST_SCOPED_VERIFIED' and
            p['thresholds'] == THRESHOLDS and p['warmup_requests'] == 0 and
            p['approx_requires_explicit_run_opt_in'] is True and
            p['correctness_qualified_speedup'] == 'NOT_ESTABLISHED', 'source/safety settings differ')
    fields = material(p['original_experiment'], p['multi'], p['dataset'], p['count'], p['performance_knobs'],
                      continuous_contract=p['continuous_contract'], strict_pilot=p['strict_pilot'],
                      ud_operation_seconds=p['ud_operation_seconds'], cpu_only=cpu_only)
    require(all(p[k] == value for k, value in fields.items()), 'changed prepared input/workload/resources')
    return p, inventory


def cpu_validate(experiment, output_dir):
    hidden_cuda()
    p, inventory = load_plan(experiment)
    require(p['resources']['status'] == 'STATIC_BOUNDS_PASS_RUNTIME_CHECKS_PENDING', 'BLOCKED_INFEASIBLE; no CPU launch approval')
    require(all(dependencies().values()), 'native CPU dependencies unavailable; no approval')
    import torch
    require(not torch.cuda.is_initialized(), 'CPU approval initialized CUDA')
    sys.path.insert(0, str(ROOT / 'tests'))
    suite = unittest.defaultTestLoader.loadTestsFromNames(TEST_MODULES)
    result = unittest.TestResult()
    start = time.monotonic()
    suite.run(result)
    require(not result.errors and not result.failures and not result.skipped and not torch.cuda.is_initialized(),
            'CPU approval failed/skipped: ' + str(result.errors + result.failures + result.skipped))
    # Reconstruct after tests as well: concurrent source/input edits invalidate approval.
    require(load_plan(experiment) == (p, inventory), 'source/input changed during CPU validation')
    clean = p['source']['approval_scope'] == 'CLEAN_REVIEWED_SOURCE'
    from .r5_d5_lightperf_runtime import native_versions
    record = dict(schema=SCHEMA, mode=MODE, engine=ENGINE, status='PASS' if clean else 'DEVELOPMENT_TESTS_PASS',
                  scope='CLEAN_NATIVE_CPU_APPROVAL' if clean else 'DEVELOPMENT_NOT_GPU_APPROVED',
                  source=p['source'], plan_inventory=inventory, tests=result.testsRun,
                  failed=0, errors=0, skipped=0, wall_seconds=time.monotonic() - start, modules=list(TEST_MODULES),
                  versions=native_versions(), native_parity='TINY_OPT_STRICT_LIGHT_CPU_PASS', gpu_parity='NOT_GPU_VERIFIED')
    writer = SummaryWriter(fresh(output_dir, [experiment, p['original_experiment'], p['multi'], *p['dataset_paths'].values()]))
    writer.write('cpu_validation.json', record)
    writer.seal()
    return record


def inspect_plan(experiment, cpu_audit=None):
    hidden_cuda()
    p, inventory = load_plan(experiment)
    approval_label = 'NOT_PROVIDED'
    if cpu_audit is not None:
        approval(experiment, cpu_audit)
        approval_label = 'MATCHED_CLEAN_NATIVE_CPU_APPROVAL'
    return dict(status='CPU_APPROVED_MANUAL_GPU_NOT_RUN' if cpu_audit is not None else 'PARTIAL_BLOCKED',
                GPU_status='GPU_NOT_RUN', source_commit=p['source']['commit'],
                plan_id=p['plan_id'], cpu_approval=approval_label, resources=p['resources'],
                remaining='manual allocation, GPU UUID/VRAM/runtime guards and explicit user GPU approval')


def approval(experiment, cpu_audit, *, cpu_only=True):
    p, pi = load_plan(experiment, cpu_only=cpu_only)
    from .r5_d5_lightperf_runtime import bound_receipts
    checked, receipt, ai = bound_receipts(experiment, cpu_audit)
    require(p == checked and pi == receipt, 'CPU approval/plan reconstruction differs')
    return p, pi, ai


def run(**kwargs):
    from .r5_d5_lightperf_runtime import run as native_run
    return native_run(**kwargs)


def analyze(registry, output_dir):
    hidden_cuda()
    from .r5_d5_lightperf_analysis import verify_run, summarize, compare
    records = read_json(registry)
    require(type(records) is list and 0 < len(records) <= 324, 'bounded explicit receipt registry required')
    runs, bindings, protected = [], [], [registry]
    for paths in records:
        require(set(paths) == {'run', 'plan', 'cpu_audit'}, 'explicit run/plan/CPU receipt triple required')
        r, binding = verify_run(paths['run'], paths['plan'], paths['cpu_audit'])
        runs.append(r)
        bindings.append(binding)
        protected.extend(paths.values())
    require(len({digest(b['run']) for b in bindings}) == len(bindings), 'duplicate run registry')
    groups = {}
    for r in runs:
        key = digest({k: r[k] for k in ('dataset', 'scale', 'tasks_sha256', 'source', 'performance_knobs')})
        group = groups.setdefault(key, {})
        require(r['condition'] not in group, 'duplicate condition; do not pool trials')
        group[r['condition']] = r
    comparisons = []
    for group in groups.values():
        if CONDITIONS[0] in group:
            comparisons.extend(dict(dataset=group[c]['dataset'], scale=group[c]['scale'],
                                    tasks_sha256=group[c]['tasks_sha256'], performance_knobs=group[c]['performance_knobs'],
                                    **compare(group[CONDITIONS[0]], group[c])) for c in CONDITIONS[1:] if c in group)
    rows = [summarize(r) for r in runs]
    rows.sort(key=lambda r: (r['dataset'], r['scale'], r['condition'], str(r['performance_knobs'])))
    report = dict(schema=SCHEMA, mode=MODE, status='LIGHT_RECEIPTS_VERIFIED_DESCRIPTIVE_ONLY',
                  rows=rows, comparisons=comparisons, bindings=bindings,
                  break_even='NOT_EVALUATED_UNTIL_MATCHED_SINGLE_AXIS_SERIES',
                  correctness_qualified_speedup='NOT_ESTABLISHED')
    columns = ('dataset', 'scale', 'condition', 'mean_ttft_ms', 'mean_e2e_ms', 'p95_e2e_ms', 'requests_per_second', 'output_tokens_per_second', 'quality_metric', 'quality_mean')
    buf = io.StringIO()
    csv_writer = csv.writer(buf)
    csv_writer.writerow(columns)
    lines = ['| ' + ' | '.join(columns) + ' |', '| ' + ' | '.join(['---'] * len(columns)) + ' |']
    for r in rows:
        values = [r['dataset'], r['scale'], r['condition'], r['ttft_ms']['mean'], r['e2e_ms']['mean'], r['e2e_ms']['p95'],
                  r['requests_per_second'], r['output_tokens_per_second'], r['quality']['metric'], r['quality']['mean']]
        csv_writer.writerow(values)
        lines.append('| ' + ' | '.join(map(str, values)) + ' |')
    writer = SummaryWriter(fresh(output_dir, protected))
    writer.write('analysis.json', report)
    writer.write_text('tables.csv', buf.getvalue())
    writer.write_text('tables.md', '\n'.join(lines) + '\n')
    writer.seal()
    return dict(status=report['status'], rows=len(rows), comparisons=len(comparisons))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('fingerprint', help='development source fingerprint; never GPU approval')
    sub.add_parser('matrix', help='CPU candidate matrix; no submissions')
    sub.add_parser('environment', help='read-only local dependencies/source availability')
    p = sub.add_parser('prepare')
    for name in ('original-experiment', 'multi', 'output-dir'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--dataset', choices=DATASETS, required=True)
    p.add_argument('--count', type=int, choices=COUNTS, required=True)
    p.add_argument('--bandwidth-mbps', type=float, choices=(200, 100, 50), default=None)
    p.add_argument('--ud-throughput-factor', type=float, choices=(1., .5, .25), default=1.)
    p.add_argument('--ud-operation-seconds', type=float)
    p.add_argument('--development', action='store_true')
    p.add_argument('--reviewed-commit')
    p.add_argument('--reviewed-fingerprint')
    p.add_argument('--continuous-contract')
    p.add_argument('--strict-pilot')
    for command in ('cpu-validate', 'inspect'):
        p = sub.add_parser(command)
        p.add_argument('--experiment', required=True)
        p.add_argument('--' + ('output-dir' if command == 'cpu-validate' else 'cpu-audit'), required=command == 'cpu-validate')
    for command in ('run', '_worker'):
        p = sub.add_parser(command, help='native execution; clean CPU approval + manual allocation required')
        p.add_argument('--experiment', required=True)
        p.add_argument('--cpu-audit', required=True)
        p.add_argument('--condition', choices=CONDITIONS, required=True)
        p.add_argument('--output-dir', required=True)
        p.add_argument('--source-commit', required=True)
        p.add_argument('--expected-gpu-uuid', required=True)
        p.add_argument('--enable-approximate-experiment', action='store_true')
        p.add_argument('--enable-extended-experiment', action='store_true')
        if command == 'run':
            p.add_argument('--enable-gpu-experiment', action='store_true')
    p = sub.add_parser('_ud')
    p.add_argument('--config', required=True)
    p.add_argument('--output-dir', required=True)
    for command in ('analyze', 'aggregate'):
        p = sub.add_parser(command)
        p.add_argument('--registry', required=True)
        p.add_argument('--output-dir', required=True)
    p = sub.add_parser('verify')
    p.add_argument('--run', required=True)
    p.add_argument('--experiment', required=True)
    p.add_argument('--cpu-audit', required=True)
    args = vars(parser.parse_args(argv))
    command = args.pop('command')
    try:
        if command not in ('run', '_worker'):
            hidden_cuda()
        if command == 'fingerprint':
            src = source_identity(development=True)
            result = {k: v for k, v in src.items() if k != 'files'}
        elif command == 'matrix':
            result = matrix()
        elif command == 'environment':
            result = dict(status='CPU_DISCOVERY_ONLY', root=str(ROOT), branch=BRANCH, native_dependencies=dependencies(),
                          GPU_status='GPU_NOT_RUN', native_worker=ENGINE)
        elif command == 'prepare':
            result = prepare(**args)
        elif command == 'cpu-validate':
            result = cpu_validate(**args)
        elif command == 'inspect':
            result = inspect_plan(**args)
        elif command == 'run':
            result = run(**args)
        elif command == '_worker':
            from .r5_d5_lightperf_runtime import es_worker
            es_worker(**args)
            result = dict(status='NATIVE_ES_COMPLETE')
        elif command == '_ud':
            from .r5_d5_lightperf_runtime import ud_worker
            ud_worker(**args)
            result = dict(status='NATIVE_UD_COMPLETE')
        elif command in ('analyze', 'aggregate'):
            result = analyze(**args)
        else:
            from .r5_d5_lightperf_analysis import verify_run
            _, binding = verify_run(args['run'], args['experiment'], args['cpu_audit'])
            result = dict(status='LIGHT_RECEIPT_INTEGRITY_PASS', binding=binding, numerical_audit='NOT_EVALUATED')
        print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
        return 2 if result.get('status') == 'PARTIAL_BLOCKED' else 0
    except (ValueError, KeyError, OSError, ImportError) as exc:
        print(json.dumps(dict(status='PARTIAL_BLOCKED', reason=str(exc),
                              GPU_status='CHECK_FAILED_RECEIPTS' if command in ('run', '_worker') else 'GPU_NOT_RUN')), file=sys.stderr)
        return 2
