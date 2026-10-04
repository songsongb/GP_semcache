#!/usr/bin/env python3
"""Offline B3 history analysis and frozen preparation."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from semcache.experiments.cachegen.c6b3_multiwoz import main
if __name__=='__main__':
    main()
