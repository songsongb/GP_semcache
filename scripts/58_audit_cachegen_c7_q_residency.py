#!/usr/bin/env python3
"""CPU-only C7-A audit; no Slurm, model downloads, or production mutations."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from semcache.experiments.cachegen.c7_q_residency import main

if __name__ == '__main__':
    raise SystemExit(main())
