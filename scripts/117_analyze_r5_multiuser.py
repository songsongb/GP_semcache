#!/usr/bin/env python3
"""Independent read-only multi-adapter audit into a new immutable output."""
from importlib import import_module
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
if __name__=='__main__':sys.exit(import_module('116_run_r5_multiuser').main(['--phase','analyze',*sys.argv[1:]]))
