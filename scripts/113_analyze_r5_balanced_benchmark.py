#!/usr/bin/env python3
"""Read-only independent R5 balanced benchmark analysis."""
import sys
from pathlib import Path
from importlib.util import spec_from_file_location,module_from_spec

if __name__=='__main__':
    s=spec_from_file_location('r5_balanced',Path(__file__).with_name('112_run_r5_balanced_benchmark.py'))
    m=module_from_spec(s);s.loader.exec_module(m);sys.exit(m.main(['--phase','analyze',*sys.argv[1:]]))
