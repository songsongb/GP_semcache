#!/usr/bin/env python3
"""CPU-only manual Q24 freeze; no model or codec execution."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from semcache.experiments.cachegen.c7b3_q_freeze import main

if __name__ == '__main__':
    main()
