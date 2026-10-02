"""Run deterministic C6 task quality; --dry-run loads no model or codec."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from semcache.experiments.cachegen.c6_quality import main

if __name__ == '__main__':
    try:
        main()
    except (ImportError, FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f'C6 BLOCKED/ERROR: {exc}', file=sys.stderr)
        sys.exit(2)
