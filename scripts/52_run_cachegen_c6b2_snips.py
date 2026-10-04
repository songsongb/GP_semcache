"""Five-mode task-trained SNIPS evaluation; --dry-run imports no model or codec."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from semcache.experiments.cachegen.c6b2_snips import main

if __name__=='__main__':
    try:
        main()
    except (ImportError,OSError,ValueError,RuntimeError,KeyError,TypeError) as exc:
        print(f'C6-B2 BLOCKED/ERROR: {exc}',file=sys.stderr)
        sys.exit(2)
