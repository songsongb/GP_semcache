#!/usr/bin/env python3
"""Frozen MultiWOZ five-mode causal comparison; offline only."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from semcache.experiments.cachegen.c6b3_2_multiwoz import main
if __name__=='__main__':main()
