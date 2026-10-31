#!/usr/bin/env python3
"""Read-only independent tensor/context/TCP audit; writes a fresh report directory."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from importlib import import_module

if __name__=='__main__':
    sys.exit(import_module('114_run_r5_fidelity_forensics').main(['--phase','analyze',*sys.argv[1:]]))
