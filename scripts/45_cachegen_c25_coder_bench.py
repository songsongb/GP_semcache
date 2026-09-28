#!/usr/bin/env python3
"""One real C1 w=3 fixture; reference versus bit-exact Python fast coder."""
import argparse
from pathlib import Path
from _common import ROOT
from semcache.experiments.cachegen.c2.physical_storage import DEFAULT_PROFILE
from semcache.experiments.cachegen.c25_coder_bench import run


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture-manifest', type=Path,
                        default=ROOT/'results/cachegen/c1/capture_manifest.json')
    parser.add_argument('--profile-path', type=Path, default=DEFAULT_PROFILE)
    parser.add_argument('--cachegen-repo', type=Path, default=Path('/data/khuss/repos/CacheGen'))
    parser.add_argument('--output-dir', type=Path, default=ROOT/'results/cachegen/c2_5/coder_benchmark')
    parser.add_argument('--repeats', type=int, default=3)
    return run(parser.parse_args(argv))


if __name__ == '__main__':
    main()
