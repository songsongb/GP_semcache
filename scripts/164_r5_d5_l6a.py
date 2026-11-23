#!/usr/bin/env python3
"""CPU-only long replay preparation and offline analysis."""
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
os.environ.update(CUDA_VISIBLE_DEVICES='', HIP_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1',
                  TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from semcache.edgelora.r5_d5_l6a import main

if __name__ == '__main__':
    sys.exit(main())
