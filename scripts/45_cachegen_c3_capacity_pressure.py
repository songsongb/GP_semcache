#!/usr/bin/env python3
"""C3 model-free SemCache byte-capacity replay; no quantizer or coder execution."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

from semcache.experiments.cachegen.c3_capacity import DEFAULT_BUDGET_MIB, DEFAULT_OUTPUT, run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture-manifest', type=Path,
                        default=ROOT/'results/cachegen/c1/capture_manifest.json')
    parser.add_argument('--holdout-dir', type=Path,
                        default=ROOT/'results/cachegen/c1_5c/holdout')
    parser.add_argument('--rate-calibration-dir', type=Path,
                        default=ROOT/'results/cachegen/c1_5c/rate_calibration')
    parser.add_argument('--snips', type=Path,
                        default=ROOT/'results/workloads/m9b_snips_semantic.jsonl')
    parser.add_argument('--multiwoz', type=Path,
                        default=ROOT/'results/workloads/m9b_multiwoz_semantic.jsonl')
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--budgets-mib', type=int, nargs='+', default=DEFAULT_BUDGET_MIB)
    parser.add_argument('--users', type=int, default=25,
                        help='Frozen M9-B seeded user assignment count (default: 25)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    try:
        if args.users < 1:
            raise ValueError('Positive M9-B user count required')
        run(args)
    except (OSError, ValueError, KeyError, TypeError, AssertionError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
