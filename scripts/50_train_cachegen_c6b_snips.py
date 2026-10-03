"""Offline C6-B1 SNIPS task-adapter training; --dry-run does data planning only."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from semcache.experiments.cachegen.c6b_snips import main

if __name__ == '__main__':
    try:
        main(training=True)
    except (ImportError,OSError,ValueError,RuntimeError) as exc:
        print(f'C6-B1 BLOCKED/ERROR: {exc}',file=sys.stderr)
        sys.exit(2)
