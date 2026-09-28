#!/usr/bin/env python3
"""Small SERAPH C2 physical-cache storage smoke; no model inference."""
import argparse
from pathlib import Path
from _common import ROOT
from semcache.experiments.cachegen.c2.physical_storage import DEFAULT_PROFILE
from semcache.experiments.cachegen.c2_storage_smoke import smoke


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture-manifest', type=Path,
                        default=ROOT/'results/cachegen/c1/capture_manifest.json')
    parser.add_argument('--profile-path', type=Path, default=DEFAULT_PROFILE)
    parser.add_argument('--output-dir', type=Path, default=ROOT/'results/cachegen/c2/storage_smoke')
    parser.add_argument('--per-dataset', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    return smoke(parser.parse_args(argv))


if __name__ == '__main__':
    main()
