#!/usr/bin/env python3
"""Read-only allocated-GPU diagnostics. Does not load a model or run inference."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from semcache.edgelora.r3a_artifacts import read
from semcache.edgelora.r5_i5_gpu import diagnostics
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base-prepared',type=Path,required=True)
p.add_argument('--gpu-identity-policy',choices=('STRICT_PREPARATION_GPU','SAME_COMPATIBLE_GPU_CLASS'),default='SAME_COMPATIBLE_GPU_CLASS')
a=p.parse_args();r=diagnostics(read(a.base_prepared/'r3a_prepare_manifest.json'),policy=a.gpu_identity_policy)
print(json.dumps(r,indent=2));sys.exit(0 if r['eligible'] else 1)
