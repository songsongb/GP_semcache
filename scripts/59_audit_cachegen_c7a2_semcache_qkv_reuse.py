#!/usr/bin/env python3
"""CPU-only C7-A2 corrected SemCache subsequence payload audit."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from semcache.experiments.cachegen.c7a2_qkv_reuse_audit import main

if __name__ == '__main__':
    raise SystemExit(main())
