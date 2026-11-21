#!/usr/bin/env python3
"""Read-only CPU review of newly approved L2C/L2E Light runs; never GPU launch."""
import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for prefix in ('reference', 'candidate', 'full'):
        for kind in ('run', 'plan', 'cpu-audit'):
            parser.add_argument('--' + prefix + '-' + kind, required=prefix != 'full')
    parser.add_argument('--output-dir', required=True)
    args = vars(parser.parse_args(argv))
    triples = {}
    for prefix in ('reference', 'candidate', 'full'):
        triple = {k: args[prefix + '_' + k] for k in ('run', 'plan', 'cpu_audit')}
        if prefix == 'full' and any(triple.values()) and not all(triple.values()):
            parser.error('supply all three FULL paths or none')
        triples[prefix] = triple if all(triple.values()) else None
    try:
        from semcache.edgelora.r5_d5_impact_review import review
        result = review(triples['reference'], triples['candidate'], args['output_dir'], triples['full'])
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0 if result['status'] == 'IMPACT_TOKEN_AND_LIGHT_SUMMARY_PARITY_PASS' else 2
    except (ValueError, OSError, KeyError, ImportError) as exc:
        print(json.dumps(dict(status='BLOCKED', reason=str(exc), GPU_status='GPU_NOT_RUN')), file=sys.stderr)
        return 2


if __name__ == '__main__': sys.exit(main())
