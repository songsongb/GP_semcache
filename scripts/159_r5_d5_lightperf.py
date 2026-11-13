#!/usr/bin/env python3
"""Local-only, CPU-first D5 Light CLI. Native GPU run remains fail-closed."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from semcache.edgelora.r5_d5_lightperf import main

if __name__ == '__main__':
    raise SystemExit(main())
