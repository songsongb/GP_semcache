"""M6-only CLI helpers; legacy command contracts are preserved."""
import argparse
from pathlib import Path
from _common import ROOT
from semcache.experiments.config import load_paper_config


def parser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument('--config',type=Path,default=ROOT/'configs/paper/common.yaml')
    return p


def tokenizer_options(p):
    p.add_argument('--tokenizer-id')
    p.add_argument('--tokenizer-revision')
    p.add_argument('--allow-download',action='store_true')


def tokenizer_for(args):
    if not args.tokenizer_id:
        return None
    if not args.tokenizer_revision:
        raise ValueError('--tokenizer-revision required with --tokenizer-id')
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(args.tokenizer_id,revision=args.tokenizer_revision,local_files_only=not args.allow_download)
