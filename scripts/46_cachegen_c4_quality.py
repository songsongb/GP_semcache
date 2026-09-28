#!/usr/bin/env python3
"""C4 controlled OPT-2.7B quality comparison: FULL, RAW, frozen compressed reuse."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

from semcache.experiments.cachegen.b2 import format as fmt
from semcache.experiments.cachegen.c2.physical_storage import DEFAULT_PROFILE
from semcache.experiments.cachegen.c4_quality import OUTPUT, run_quality


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snips', type=Path,
                        default=ROOT/'results/workloads/m9b_snips_semantic.jsonl')
    parser.add_argument('--multiwoz', type=Path,
                        default=ROOT/'results/workloads/m9b_multiwoz_semantic.jsonl')
    parser.add_argument('--profile-path', type=Path, default=DEFAULT_PROFILE)
    parser.add_argument('--per-dataset', type=int, default=16)
    parser.add_argument('--coder-backend', choices=(fmt.FAST_CODER,), default=fmt.FAST_CODER)
    parser.add_argument('--max-new-tokens', type=int, default=16)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.max_new_tokens <= 16:
        parser.error('C4 greedy generation is limited to 1..16 new tokens')
    try:
        run_quality(args)
    except (OSError, ValueError, KeyError, TypeError, AssertionError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
