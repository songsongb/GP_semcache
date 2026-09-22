#!/usr/bin/env python3
"""M9-A.2 local-only attention audit; existing strict artifacts stay read-only."""
import argparse
import csv
import json
from pathlib import Path
import sys
from statistics import mean
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from semcache.system_cost.common import write_json, file_sha256
from semcache.system_cost.profiles import load_rows, pair_rows
from semcache.system_cost.es_base_profile import validate_base_profile, validate_fresh_m85, control_plane
from semcache.diagnostics.impact import MODES
from semcache.diagnostics.impact_cost import recompose, break_even


def write_csv(path, rows):
    if not rows:
        return
    keys = sorted({key for row in rows for key in row})
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows({k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in row.items()} for row in rows)


def load_compositions(path, es_input, pairs):
    """Bind original modeled totals to strict measured inputs before any model load."""
    summary = json.loads(path.read_text())
    env = json.loads((path.parent/'m9a_environment.json').read_text())
    if file_sha256(es_input) not in env.get('input_files', {}).values():
        raise ValueError('Original cost environment does not identify the strict input hash')
    if summary.get('result_label') != 'STRICT_BASE_ONLY_SYSTEM_MODEL':
        raise ValueError('Only existing strict-base-only compositions can be recomposed')
    by_repeat = {p[0]['repeat_index']: p for p in pairs}
    found = {}
    for result in summary['comparisons']:
        row = result['row']
        index = row['repeat_index']
        if index not in by_repeat:
            raise ValueError('Cost comparison references an unknown strict repeat')
        native, _, physical = by_repeat[index]
        if (row['model'] != native['model_id'] or row['experiment_id'] != native['experiment_id']
                or row['source_prompt_hash'] != native['prompt_token_ids_sha256']
                or row['query_id'] != 'same_user_exact'
                or row['reused_tokens'] != physical['reused_tokens']
                or row['es_compute_ms'] != native['es_base_profile']['prefill_wall_ms']
                or row['semcache_es_ms'] != physical['es_base_profile']['prefill_wall_ms']
                or row['es_compute_ms_provenance'] != 'MEASURED'
                or row['semcache_es_ms_provenance'] != 'MEASURED'
                or row['ud_lora_ms_provenance'] != 'CALIBRATED'
                or row['semcache_ud_ms_provenance'] != 'CALIBRATED'
                or result['base_profiles'] != [native['es_base_profile'], physical['es_base_profile']]
                or result['semcache_control_profile'] != control_plane(physical)
                or row['attention_impact_ms'] != physical['attention_impact_ms']
                or row['semcache_control_ms'] != control_plane(physical)['total_control_ms']):
            raise ValueError('Cost comparison differs from strict measured fixture')
        # Duplicate bandwidth rows are accepted only if nonnetwork decomposition
        # and tensor byte accounting are identical. Keep one to rescale analytically.
        signature = tuple(row[k] for k in ('es_compute_ms', 'ud_lora_ms', 'semcache_es_ms',
            'semcache_ud_ms', 'semcache_control_ms', 'edge_total_network_bytes',
            'semcache_total_network_bytes', 'communication_saved_bytes'))
        if index in found and found[index][0] != signature:
            raise ValueError('Original bandwidth rows use different cost/communication assumptions')
        recompose(result, row['attention_impact_ms'], MODES[0])  # Enforce gate before running models.
        found[index] = (signature, result)
    if set(found) != set(by_repeat):
        raise ValueError('Missing original strict system comparisons')
    return [found[index][1] for index in sorted(found)]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--es-input', type=Path, required=True, help='M9-A.1 strict_es_input.jsonl')
    p.add_argument('--source-environment', type=Path, help='Default: ES_INPUT parent/fresh_m85/inference_environment.json')
    p.add_argument('--system-summary', type=Path, help='Existing strict m9a_summary.json; companion environment must exist')
    p.add_argument('--impact-only', action='store_true', help='Smoke only: explicitly omit system recomposition')
    p.add_argument('--device', default='cuda')
    p.add_argument('--warmup-runs', type=int, default=3)
    p.add_argument('--measured-runs', type=int, default=10)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args(argv)
    if args.warmup_runs < 1 or args.measured_runs < 1:
        p.error('Use positive warmup and measured counts')
    if bool(args.system_summary) == args.impact_only:
        p.error('Supply --system-summary, or explicitly choose --impact-only')
    if args.output_dir.exists():
        p.error('Choose a new output directory; no audit/source artifacts are overwritten')
    source_env = args.source_environment or args.es_input.parent/'fresh_m85/inference_environment.json'
    rows = load_rows(args.es_input)
    model_ids = {r['model_id'] for r in rows}
    if len(model_ids) != 1:
        p.error('One model per audit input')
    model_id = model_ids.pop()
    pairs = sorted(pair_rows(rows, model_id), key=lambda pair: pair[0]['repeat_index'])
    for native, lookup, physical in pairs:
        for row in (native, lookup, physical):
            validate_fresh_m85(row)
        validate_base_profile(native, 'ES_BASE_NATIVE')
        validate_base_profile(physical, 'ES_BASE_SEMCACHE_REUSE')
    first = pairs[0][2]
    for _, _, physical in pairs:
        for field in ('prompt_token_ids_sha256', 'model_revision', 'tokenizer_revision', 'tokenizer_source_id',
                      'dtype', 'hostname', 'gpu_name', 'reused_tokens', 'reuse_block_provenance'):
            if physical[field] != first[field]:
                p.error(f'Fixed-fixture audit requires invariant {field} across source repetitions')
    environment = json.loads(source_env.read_text())
    if any(p[2]['es_base_profile']['source_environment_sha256'] != file_sha256(source_env) for p in pairs):
        p.error('Source environment hash differs from strict profile')
    compositions = load_compositions(args.system_summary, args.es_input, pairs) if args.system_summary else []
    hashes = {str(path): file_sha256(path) for path in (args.es_input, source_env)}
    if args.system_summary:
        hashes[str(args.system_summary)] = file_sha256(args.system_summary)
        hashes[str(args.system_summary.parent/'m9a_environment.json')] = file_sha256(args.system_summary.parent/'m9a_environment.json')
    args.output_dir.mkdir(parents=True)
    manifest = dict(schema='m9a2_impact_audit_v1', state='STARTED', input_hashes=hashes,
        model_id=model_id, configuration={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        formula_provenance='PAPER_DEFINED', interpretation_provenance='REPRODUCTION_CHOICE',
        normal_inference_modified=False, system_recomposition='SIMULATED_RESEARCH_EXTENSION',
        timing_design='fixed captured actual attention; rotated mode order; separate attribution/uninstrumented passes',
        uninstrumented_total_used_for_recomposition=True,
        no_downloads=True, full_decode=False, distributed_execution=False)
    write_json(args.output_dir/'audit_manifest.json', manifest)
    from semcache.diagnostics.impact_runtime import capture_fixture, benchmark, summarize
    captures, queries, capture_record, resident_objects = capture_fixture(first, environment, args.device)
    write_json(args.output_dir/'fixture_capture.json', capture_record)
    raw = benchmark(captures, queries, capture_record['capacity_bytes'], args.warmup_runs, args.measured_runs)
    summary = summarize(raw)
    with (args.output_dir/'impact_raw.jsonl').open('w') as f:
        for row in raw:
            f.write(json.dumps(row, allow_nan=False)+'\n')
    write_json(args.output_dir/'impact_summary.json', summary)
    write_csv(args.output_dir/'impact_summary.csv', summary)
    eligible = all(r['diagnostic_composition_eligible'] for r in raw)
    system = []
    if eligible:
        for item in summary:
            replacement = item['uninstrumented_impact_total_ms']['mean']
            for comparison in compositions:
                for bandwidth in (200, 500, 1000):
                    system.append(recompose(comparison, replacement, item['implementation'], bandwidth_mbps=bandwidth))
    system_summary = []
    for mode in MODES:
        for bandwidth in (200, 500, 1000):
            group = [r for r in system if r['implementation'] == mode and r['bandwidth_mbps'] == bandwidth]
            if group:
                averaged = {key: mean(r[key] for r in group) for key in (
                    'edge_lora_total_ms', 'semcache_total_ms', 'system_delta_ms', 'compute_only_delta_ms', 'saved_network_bytes')}
                system_summary.append(dict(implementation=mode, bandwidth_mbps=bandwidth, **averaged,
                    **break_even(averaged['saved_network_bytes'], averaged['compute_only_delta_ms']),
                    result_label='RESEARCH_EXTENSION_DIAGNOSTIC', provenance='SIMULATED_RESEARCH_EXTENSION',
                    aggregation='mean source decomposition with mean uninstrumented audit impact'))
    write_json(args.output_dir/'system_diagnostic.json', dict(rows=system, summary=system_summary,
        original_results_modified=False, recomposition_eligible=eligible,
        omitted_reason='impact_only_requested' if args.impact_only else 'parity_failed' if not eligible else None))
    write_csv(args.output_dir/'system_diagnostic.csv', system)
    write_csv(args.output_dir/'system_summary.csv', system_summary)
    manifest.update(state='COMPLETE' if eligible else 'PARITY_FAILED', parity_passed=eligible,
        source_artifacts_unchanged=all(file_sha256(path) == value for path, value in hashes.items()))
    write_json(args.output_dir/'audit_manifest.json', manifest)
    if not eligible:
        raise SystemExit('Parity failed; raw diagnostics saved; no system recomposition permitted')
    print(f'Saved semantic-impact audit to {args.output_dir}; research diagnostics only')


if __name__ == '__main__':
    main()
