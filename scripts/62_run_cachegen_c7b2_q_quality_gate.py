#!/usr/bin/env python3
"""SERAPH-only real quality execution; imports/CPU tests load no model."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from semcache.experiments.cachegen.c7b2_q_quality import main

if __name__ == '__main__':
    main()
