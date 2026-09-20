#!/usr/bin/env python3
"""Regenerate M8 aggregate CSV from raw JSONL."""
import argparse
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from semcache.metrics.m8 import aggregate_raw, reuse_delta_summary
from semcache.utils.io import read_jsonl, write_csv

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--input", type=Path, default=ROOT / "results/m8/inference_raw.jsonl")
p.add_argument("--output", type=Path, default=ROOT / "results/m8/inference_summary.csv")
p.add_argument("--delta-output", type=Path,
               help="Defaults to inference_reuse_deltas.csv beside --output")
args = p.parse_args()
rows = list(read_jsonl(args.input))
write_csv(args.output, aggregate_raw(rows))
delta_output = args.delta_output or args.output.with_name("inference_reuse_deltas.csv")
write_csv(delta_output, reuse_delta_summary(rows))
print(f"Summarized {len(rows)} rows to {args.output} and {delta_output}")
