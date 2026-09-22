#!/usr/bin/env python3
"""Compose M8 ES profiles, CPU calibration and analytical network costs; no model runs."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from semcache.system_cost.common import MODELS, PAPER_REFERENCE, read_dimensions, write_json, file_sha256
from semcache.system_cost.profiles import load_rows, pair_rows
from semcache.system_cost.model import compare_request


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--es-input', type=Path, required=True)
    p.add_argument('--ud-calibration', type=Path, required=True)
    p.add_argument('--model', choices=MODELS, default='facebook/opt-2.7b')
    p.add_argument('--model-config', type=Path)
    p.add_argument('--query-id', default='same_user_exact')
    p.add_argument('--reuse-mode', choices=('SEMCACHE_PHYSICAL_REUSE', 'SEMCACHE_POSITION_ALIGNED_DIAGNOSTIC'),
                   default='SEMCACHE_PHYSICAL_REUSE')
    p.add_argument('--bandwidth-mbps', type=float, nargs='+', default=[200], help='e.g. 200 500 1000')
    p.add_argument('--hidden-element-bytes', type=int, help='Default: M8 activation dtype width')
    p.add_argument('--delta-element-bytes', type=int, help='Default: hidden width; explicit wire precision choice')
    p.add_argument('--projection-exchange-only', action='store_true', help='Omit h0/hL boundary transfers; labelled choice')
    p.add_argument('--es-compute-policy', choices=('require-base-only', 'peft-prefill-proxy'), default='require-base-only')
    p.add_argument('--allow-invalid-reuse', action='store_true', help='Unsafe exploration only; never labelled safe reuse')
    p.add_argument('--output-dir', type=Path, default=ROOT / 'results/m9a')
    args = p.parse_args(argv)
    try:
        dims, source = read_dimensions(args.model, args.model_config)
        calibration = json.loads(args.ud_calibration.read_text())
        pairs = pair_rows(load_rows(args.es_input), args.model, args.query_id, args.reuse_mode)
        # All inputs are validated before writing any outputs; invalid rows are not silently dropped.
        results = [compare_request(native, lookup, physical, calibration, dims,
            bandwidth_mbps=bandwidth, hidden_element_bytes=args.hidden_element_bytes,
            delta_element_bytes=args.delta_element_bytes, boundary_transfers=not args.projection_exchange_only,
            es_compute_policy=args.es_compute_policy, allow_invalid_reuse=args.allow_invalid_reuse)
            for native, lookup, physical in pairs for bandwidth in args.bandwidth_mbps]
        es_hardware = [dict(gpu=pair[0].get('gpu_name'), hostname=pair[0].get('hostname'),
                            provenance='MEASURED', source='M8 artifact') for pair in pairs]
        env = dict(timestamp=datetime.now(timezone.utc).isoformat(), composition_host=platform.node(),
            composition_python=platform.python_version(), dimension_source=source,
            paper_reference=PAPER_REFERENCE, measured_es=es_hardware,
            current_es_user_report=dict(gpu='RTX 3090', cpu='SERAPH host CPU',
                source='USER_REPORTED_TARGET_NOT_SUBSTITUTED_FOR_ARTIFACT_HARDWARE'),
            calibrated_ud=dict(hostname=calibration.get('hostname'), cpu_model=calibration.get('cpu_model'),
                provenance='CALIBRATED', paper_ud_equivalence_claimed=False),
            input_files={str(path): file_sha256(path) for path in (args.es_input, args.ud_calibration)},
            model_execution_performed=False, distributed_execution=False, user_count=1,
            configuration={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()})
        args.output_dir.mkdir(parents=True, exist_ok=True)
        with (args.output_dir / 'single_user_cost_breakdown.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0]['row']))
            writer.writeheader()
            writer.writerows(result['row'] for result in results)
        write_json(args.output_dir / 'm9a_environment.json', env)
        write_json(args.output_dir / 'm9a_summary.json', dict(schema='m9a_single_user_v1',
            total_provenance='SIMULATED', safe_reuse_claimed=False,
            decision_provenance='RESEARCH_EXTENSION', decisions_applied=False,
            comparison_count=len(results), comparisons=results,
            missing_components=['full autoregressive decode', 'real UD-ES networking',
                '50-user concurrency', 'trained user-specific LoRA inference', 'paper-scale experiments'],
            excluded_scope=['OPT-6.7B', 'Cloud placement', 'performance optimization']))
    except (ValueError, OSError, KeyError, TypeError) as exc:
        p.error(str(exc))
    print(f'Saved {len(results)} single-request comparisons to {args.output_dir}; totals are SIMULATED')


if __name__ == '__main__':
    main()
