#!/usr/bin/env python3
"""Bounded C9-A local/GPU prompt latency; no network or quality evaluation."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from semcache.experiments.cachegen.c9a_latency import main

if __name__ == '__main__':
    main()
