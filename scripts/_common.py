import argparse
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from semcache.utils.io import load_config


def arguments(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/base.yaml")
    parser.add_argument("--output", type=Path)
    parser.add_argument('--model-id')
    parser.add_argument('--dtype', choices=['float16', 'float32', 'bfloat16'])
    parser.add_argument('--device')
    parser.add_argument('--revision')
    parser.add_argument('--window-size', type=int)
    parser.add_argument('--layers', type=int, nargs='+')
    parser.add_argument('--allow-download', action='store_true')
    args = parser.parse_args()
    config = load_config(args.config)
    for arg, key in [('model_id', 'name'), ('dtype', 'dtype'), ('device', 'device'), ('revision', 'revision')]:
        if getattr(args, arg):
            config['model'][key] = getattr(args, arg)
    if args.model_id:
        config['model']['tokenizer'] = args.model_id
    if args.revision:
        config['model']['tokenizer_revision'] = args.revision
    if args.allow_download:
        config['model']['local_files_only'] = False
    if args.window_size is not None:
        config['subsequence_window'] = args.window_size
    if args.layers is not None:
        config.setdefault('probe', {})['layers'] = args.layers
    return args, config
