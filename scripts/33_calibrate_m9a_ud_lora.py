#!/usr/bin/env python3
"""CPU-only rank-8 QKV LoRA calibration. Local config only; no model loading."""
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from semcache.system_cost.common import MODELS, read_dimensions, write_json
from semcache.system_cost.profiles import load_rows
from semcache.system_cost.calibration import calibrate


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=MODELS, default='facebook/opt-2.7b')
    p.add_argument('--model-config', type=Path, help='Local config.json; otherwise resolve existing HF cache only')
    p.add_argument('--es-input', type=Path, help='Derive exact full/fresh token lengths from M8 JSON/JSONL')
    p.add_argument('--sequence-lengths', help='Additional comma-separated positive lengths; default 32,64,128,256 without ES input')
    p.add_argument('--warmup-count', type=int, default=5)
    p.add_argument('--measured-count', type=int, default=20)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--host-label', default='SERAPH', help='Operator-declared host label; actual hostname/CPU are recorded')
    p.add_argument('--output', type=Path, default=ROOT / 'results/m9a/ud_lora_calibration.json')
    args = p.parse_args(argv)
    try:
        dims, source = read_dimensions(args.model, args.model_config)
        lengths = set()
        if args.sequence_lengths:
            lengths.update(int(x) for x in args.sequence_lengths.split(','))
        if args.es_input:
            for row in load_rows(args.es_input):
                if row.get('model_id') != args.model:
                    continue
                n, r = row['prompt_tokens'], row['reused_tokens']
                if type(n) is not int or type(r) is not int or not 0 <= r <= n or n < 1:
                    raise ValueError('Invalid token counts in M8 input')
                lengths.add(n)
                if n > r:
                    lengths.add(n-r)
        if not args.es_input and not args.sequence_lengths:
            lengths.update([32, 64, 128, 256])
        result = calibrate(dims, lengths, warmup=args.warmup_count, measured=args.measured_count,
                           seed=args.seed, host_label=args.host_label)
        result['dimension_source'] = source
        write_json(args.output, result)
    except (ValueError, OSError, ImportError) as exc:
        p.error(str(exc))
    print(f'Saved CPU calibration to {args.output}; no full model was loaded')


if __name__ == '__main__':
    main()
