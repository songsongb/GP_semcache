#!/usr/bin/env python3
"""Phase 1 CPU-only D5 continuous NATURAL validation. No GPU/run action."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('cpu-validate',))
    for name in ('original-experiment', 'multi', 'output-dir'):
        parser.add_argument('--'+name, required=True)
    args = vars(parser.parse_args(argv)); args.pop('action')
    from semcache.edgelora.r5_continuous_natural import cpu_validate
    try:
        result = cpu_validate(**args)
    except (ValueError, FileNotFoundError) as exc:
        print(json.dumps(dict(phase1_status='FAIL', blocker=str(exc))))
        return 2
    print(json.dumps({k: v for k, v in result.items() if k not in ('sessions', 'lifecycle')}, indent=2))
    return 0 if result['phase1_status'] == 'PASS' else 2


if __name__ == '__main__':
    sys.exit(main())
