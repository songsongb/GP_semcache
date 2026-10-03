"""Offline FULL_RECOMPUTE-only C6-B1 SNIPS task-capability validation."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from semcache.experiments.cachegen.c6b_snips import main

if __name__ == '__main__':
    try:
        main(training=False)
    except (ImportError,OSError,ValueError,RuntimeError) as exc:
        print(f'C6-B1 BLOCKED/ERROR: {exc}',file=sys.stderr)
        sys.exit(2)
