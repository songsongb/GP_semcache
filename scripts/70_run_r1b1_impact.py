#!/usr/bin/env python3
"""R1-B1 paired impact integration, using the R1-A model/timing harness."""
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from semcache.metrics.r1b1 import R1B1Extension


def main(argv=None):
    spec = importlib.util.spec_from_file_location('r1_raw_runner', ROOT/'scripts/69_run_r1_raw_semcache.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main(argv, extension=R1B1Extension())


if __name__ == '__main__':
    raise SystemExit(main())
