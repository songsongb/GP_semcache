#!/usr/bin/env python3
"""CPU-only analytical network/codec break-even; no inference or codec execution."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from semcache.experiments.cachegen.c9b_network import main

if __name__ == '__main__':
    main()
