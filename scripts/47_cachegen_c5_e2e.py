#!/usr/bin/env python3
"""C5 natural cross-query SemCache RAW vs frozen compressed integration smoke."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

from semcache.experiments.cachegen.b2 import format as fmt
from semcache.experiments.cachegen.c2.physical_storage import DEFAULT_PROFILE
from semcache.experiments.cachegen.c5_e2e import OUTPUT, run


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snips', type=Path,
                   default=ROOT/'results/workloads/m9b_snips_semantic.jsonl')
    p.add_argument('--multiwoz', type=Path,
                   default=ROOT/'results/workloads/m9b_multiwoz_semantic.jsonl')
    p.add_argument('--profile-path', type=Path, default=DEFAULT_PROFILE)
    p.add_argument('--per-dataset', type=int, choices=(1, 2, 4), default=2)
    p.add_argument('--max-prompt-tokens', type=int, default=16)
    p.add_argument('--max-semantic-prefix-rows', type=int, default=2048)
    p.add_argument('--max-encodes', type=int, default=32)
    p.add_argument('--max-new-tokens', type=int, default=16)
    p.add_argument('--coder-backend', choices=(fmt.FAST_CODER,), default=fmt.FAST_CODER)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--semantic-device', default='cpu')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output-dir', type=Path, default=OUTPUT)
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    try:
        run(args)
    except (OSError, ValueError, KeyError, TypeError, AssertionError, RuntimeError) as exc:
        p.error(str(exc))


if __name__ == '__main__':
    main()
