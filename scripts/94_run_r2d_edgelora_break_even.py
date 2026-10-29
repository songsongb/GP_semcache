#!/usr/bin/env python3
"""R2-D CPU-only evidence-gated EdgeLoRA analysis; no inference or networking."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from semcache.diagnostics.r2d_evidence import EvidenceAudit, load_m9a, load_c4, sha256, require, resolve
from semcache.diagnostics.r2d_network import finite
from semcache.diagnostics.r2d_analysis import analyze

VERSION = '1.0.0'
OUTPUTS = ('r2d_evidence_audit.json', 'r2d_bandwidth_rtt_sweep.csv',
           'r2d_break_even_summary.json', 'r2d_decision.json')


def grid(text, *, positive):
    values = [float(v.strip()) for v in text.split(',')]
    for value in values:
        finite(value, 'grid value')
        if value < 0 or (positive and value == 0):
            raise ValueError('bandwidth must be positive and RTT nonnegative')
    if not values or len(values) != len(set(values)):
        raise ValueError('grid must be nonempty and unique')
    return values


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', required=True, choices=('analyze',))
    p.add_argument('--m9a-summary', required=True, type=Path)
    p.add_argument('--c4-aggregate-dir', required=True, type=Path)
    p.add_argument('--bandwidth-mbps', default='10,50,100,200,500,1000', help='comma-separated decimal Mbps')
    p.add_argument('--rtt-ms', default='0,1,10,30,50', help='comma-separated hypothetical RTT milliseconds')
    p.add_argument('--output-dir', required=True, type=Path, help='new or empty directory; inputs are read-only')
    args = p.parse_args(argv)
    try:
        args.bandwidths = grid(args.bandwidth_mbps, positive=True)
        args.rtts = grid(args.rtt_ms, positive=False)
    except ValueError as exc:
        p.error(str(exc))
    return args


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False)+'\n')


def write_csv(path, rows):
    fields = sorted({key for row in rows for key in row})
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def code_identity():
    def git(*args):
        run = subprocess.run(['git', *args], cwd=ROOT, capture_output=True, text=True, check=False)
        return run.stdout.strip() if run.returncode == 0 else None
    paths = [ROOT/'scripts/94_run_r2d_edgelora_break_even.py',
             *sorted((ROOT/'src/semcache/diagnostics').glob('r2d_*.py'))]
    return dict(git_head=git('rev-parse', 'HEAD'), git_branch=git('branch', '--show-current'),
        git_status=git('status', '--short', '--untracked-files=normal'),
        files={str(p.relative_to(ROOT)): sha256(p) for p in paths})


def main(argv=None):
    a = parse_args(argv)
    out = a.output_dir.resolve()
    # Refuse overwrite before creating even the initial manifest.
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        print('R2-D requires a new or empty output directory: '+str(out), file=sys.stderr)
        return 1
    if out in (resolve(a.m9a_summary, ROOT).parent, resolve(a.c4_aggregate_dir, ROOT), ROOT):
        print('R2-D output directory cannot be an input/repository directory', file=sys.stderr)
        return 1
    out.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    manifest = dict(experiment='R2-D', program_version=VERSION, execution_status='RUNNING',
        classification='ANALYTICAL_DIAGNOSTIC', started_at_utc=started,
        cli=[sys.executable, str(Path(__file__).resolve()), *(sys.argv[1:] if argv is None else argv)],
        code=code_identity(), python=platform.python_version(),
        gpu_inference_performed=False, network_executed=False, model_execution_performed=False,
        cachegen_enabled=False, bandwidth_mbps=a.bandwidths, rtt_ms=a.rtts,
        input_hashes={}, output_hashes={}, output_hash_scope='four exported artifacts; manifest excluded to avoid circular self-hash')
    write_json(out/'r2d_manifest.json', manifest)
    audit = EvidenceAudit(ROOT)
    try:
        m9a = load_m9a(a.m9a_summary, audit)
        for entry in audit.files.values():
            entry['cohort_validation'] = 'VALIDATED_M9A_RECONSTRUCTION'
        c4 = load_c4(a.c4_aggregate_dir, audit)
        for entry in audit.files.values():
            if entry['cohort_validation'] == 'NOT_ESTIMABLE_UNTIL_COHORT_RECONSTRUCTION':
                entry['cohort_validation'] = 'VALIDATED_C4_RECONSTRUCTION'
        sweep, summary, decision, compat = analyze(m9a, c4, a.bandwidths, a.rtts)
        manifest['TEST_FIXTURE_ONLY'] = decision['TEST_FIXTURE_ONLY']
        record = dict(audit.record(), execution_status='COMPLETE', validity_status='VALID_SEPARATE_COHORTS',
            TEST_FIXTURE_ONLY=decision['TEST_FIXTURE_ONLY'],
            evidence_classes=['HISTORICAL_M9A_ANALYTICAL', 'R2C4_MEASURED_LOCAL_CONDITIONAL', 'HYPOTHETICAL_NETWORK_RTT'],
            compatibility=compat, provenance_gates=dict(m9a_reconstructed=True, c4_reconstructed=True,
                historical_correctness_verified=True, r2c4_strict_failures_preserved=True,
                oracle_equivalence_is_not_full_relative_correctness=True, cross_cohort_addition_allowed=False))
        # Recheck bytes at export, detecting changes during reconstruction.
        for filename, entry in list(audit.files.items()):
            require(sha256(filename) == entry['sha256'], 'input changed before export: '+filename)
        status = 'COMPLETE'
    except (OSError, ValueError, KeyError, TypeError, AttributeError, OverflowError, ImportError) as exc:
        reason = f'{type(exc).__name__}: {exc}'
        record = dict(audit.record(), execution_status='FAILED', validity_status='INVALID_REQUIRED_EVIDENCE', error=reason)
        summary = dict(status='NOT_ESTIMABLE', reason=reason, no_invalid_rows_omitted=True)
        decision = dict(execution_status='FAILED', classification='ANALYTICAL_DIAGNOSTIC',
            validity_status='NOT_ESTIMABLE', primary_claim_eligible=False,
            real_network_measured=False, full_generation_e2e_measured=False, correctness_qualified_speedup=False,
            conclusion='Required evidence failed validation; no performance estimate is qualified. '+reason)
        sweep = [dict(validity_status='NOT_ESTIMABLE', error=reason, source_scope='FAILED_EVIDENCE_VALIDATION')]
        manifest['failure'] = reason
        status = 'FAILED'
        print(reason, file=sys.stderr)
    write_json(out/OUTPUTS[0], record)
    write_csv(out/OUTPUTS[1], sweep)
    write_json(out/OUTPUTS[2], summary)
    write_json(out/OUTPUTS[3], decision)
    manifest.update(execution_status=status, completed_at_utc=datetime.now(timezone.utc).isoformat(),
        input_hashes={name: item.get('sha256') for name, item in audit.files.items()},
        output_hashes={name: sha256(out/name) for name in OUTPUTS})
    write_json(out/'r2d_manifest.json', manifest)
    print(f'R2-D {status}: {out}; CPU analytical diagnostics only')
    return 0 if status == 'COMPLETE' else 1


if __name__ == '__main__':
    sys.exit(main())
