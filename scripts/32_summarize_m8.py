#!/usr/bin/env python3
"""Regenerate M8 aggregate CSV from raw JSONL."""
import argparse
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from semcache.metrics.m8 import aggregate_raw
from semcache.utils.io import read_jsonl, write_csv

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--input", type=Path, default=ROOT / "results/m8/inference_raw.jsonl")
p.add_argument("--output", type=Path, default=ROOT / "results/m8/inference_summary.csv")
args = p.parse_args()
rows = list(read_jsonl(args.input))
write_csv(args.output, aggregate_raw(rows))
print(f"Summarized {len(rows)} rows to {args.output}")
