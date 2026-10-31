#!/usr/bin/env python3
"""Independent read-only R5 artifact audit and deterministic reports."""
import sys
from importlib.util import spec_from_file_location,module_from_spec
from pathlib import Path

if __name__=='__main__':
    spec=spec_from_file_location('r5_launcher',Path(__file__).with_name('110_run_r5_semcache_e2e.py'))
    module=module_from_spec(spec);spec.loader.exec_module(module)
    sys.exit(module.main(['--phase','analyze',*sys.argv[1:]]))
