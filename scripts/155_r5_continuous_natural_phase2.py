#!/usr/bin/env python3
"""D5 Phase 2 CPU-only preparation and independently reconstructed validation."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    prep = sub.add_parser('prepare')
    for name in ('original-experiment', 'multi', 'phase1-audit', 'output-dir'):
        prep.add_argument('--'+name, required=True)
    prep.add_argument('--input-limit', type=int, required=True)
    prep.add_argument('--context-truncation', choices=('allow', 'forbid'), required=True)
    validate = sub.add_parser('cpu-validate')
    validate.add_argument('--experiment', required=True); validate.add_argument('--output-dir', required=True)
    args = vars(parser.parse_args(argv)); action = args.pop('action')
    from semcache.edgelora import r5_continuous_natural_phase2 as api
    try:
        result = getattr(api, action.replace('-', '_'))(**args)
    except (ValueError, KeyError, TypeError, FileNotFoundError, ImportError) as exc:
        print(json.dumps(dict(status='FAIL', reason=str(exc)))); return 2
    print(json.dumps(result, indent=2))
    return 0 if result['status'].startswith('PASS') else 2


if __name__ == '__main__': sys.exit(main())
