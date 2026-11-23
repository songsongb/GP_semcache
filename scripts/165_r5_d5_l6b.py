#!/usr/bin/env python3
"""L6-B CPU controls and registered allocation worker."""
import os
import sys
from pathlib import Path

sys.dont_write_bytecode=True
os.environ.update(PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
if sys.argv[1:2]!=['_job']:
    os.environ.update(CUDA_VISIBLE_DEVICES='',HIP_VISIBLE_DEVICES='')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from semcache.edgelora.r5_d5_l6b_runner import main

if __name__=='__main__':
    sys.exit(main())
