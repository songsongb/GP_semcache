#!/usr/bin/env python3
"""Bounded SERAPH C8-B quality replay; no latency or profile fitting."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from semcache.experiments.cachegen.c8b_quality import main

if __name__ == '__main__':
    main()
