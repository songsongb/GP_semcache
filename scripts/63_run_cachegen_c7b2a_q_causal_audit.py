#!/usr/bin/env python3
"""SERAPH-only four-case causal audit; --help and synthetic tests load no model."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from semcache.experiments.cachegen.c7b2a_q_causal_audit import main

if __name__ == '__main__':
    main()
