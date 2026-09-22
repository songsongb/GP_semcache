#!/usr/bin/env python3
"""Model-free paired logical multi-user SemCache simulation."""
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from semcache.simulation.multi_user import read_workload, fixture_costs, run_matrix


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snips', type=Path, help='Existing pre-tokenized semantic workload JSONL')
    p.add_argument('--multiwoz', type=Path, help='Existing pre-tokenized semantic workload JSONL')
    p.add_argument('--system-summary', type=Path, help='M9-A strict m9a_summary.json')
    p.add_argument('--impact-summary', type=Path, help='M9-A.2 impact_summary.json')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--smoke', action='store_true', help='Synthetic model-free fixtures, not dataset results')
    p.add_argument('--output-dir', type=Path, default=ROOT/'results/m9b/run')
    args = p.parse_args()
    try:
        if args.smoke and (args.snips or args.multiwoz):
            raise ValueError('Smoke and real datasets are mutually exclusive')
        if bool(args.system_summary) != bool(args.impact_summary):
            raise ValueError('Supply both cost artifacts or neither')
        workloads = {name: read_workload(path, name) for name, path in
                     (('snips', args.snips), ('multiwoz', args.multiwoz)) if path}
        if args.smoke:
            workloads = {name: [dict(dataset=name, source_id=str(i), token_ids=[1,2,3,4,i%4], cluster_id=i%2)
                                for i in range(120)] for name in ('snips', 'multiwoz')}
        if not workloads:
            raise ValueError('Supply local --snips/--multiwoz semantic artifacts or --smoke')
        costs = fixture_costs(args.system_summary, args.impact_summary) if args.system_summary else None
        paths = [x for x in (args.snips,args.multiwoz,args.system_summary,args.impact_summary) if x]
        rows = run_matrix(workloads, args.output_dir, args.seed, costs, paths, args.smoke)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        p.error(str(exc))
    print(f'Saved {len(rows)} SIMULATED configurations to {args.output_dir}; safe_reuse_claimed=false')


if __name__ == '__main__':
    main()
