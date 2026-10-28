#!/usr/bin/env python3
"""R2-C2 launcher identity and explicit case-level validation diagnostics."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from semcache.diagnostics.r1b3_break_even import code_identity
from semcache.diagnostics.r2c2_profile_memory import validation_report,uuid
from semcache.metrics.r1 import atomic_json


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    identity=sub.add_parser('identity',help='seal launcher/code/physical GPU before separate Python executions')
    identity.add_argument('--output',type=Path,required=True)
    identity.add_argument('--launcher',type=Path,required=True)
    identity.add_argument('--device',default='cuda:0')
    check=sub.add_parser('check-code',help='reject code/launcher edits during a Slurm allocation')
    check.add_argument('--identity',type=Path,required=True)
    validate=sub.add_parser('validate',help='print all failed case errors/memory before returning nonzero')
    validate.add_argument('--identity',type=Path,required=True)
    validate.add_argument('--directory',type=Path,required=True)
    a=parser.parse_args(argv)
    try:
        if a.command=='identity':
            if a.output.exists(): raise FileExistsError(a.output)
            import torch
            physical=str(torch.cuda.get_device_properties(a.device).uuid)
            uuid(physical)
            record=dict(code=code_identity(ROOT),gpu_uuid=physical,device=a.device,
                launcher=str(a.launcher.resolve()),launcher_sha256=hashlib.sha256(a.launcher.read_bytes()).hexdigest())
            atomic_json(a.output,record); print(json.dumps(record,indent=2)); return 0
        record=json.loads(a.identity.read_text())
        if code_identity(ROOT)['sha256']!=record['code']['sha256']:
            raise ValueError('code changed within allocation; fresh compatible Pilot required')
        if hashlib.sha256(Path(record['launcher']).read_bytes()).hexdigest()!=record['launcher_sha256']:
            raise ValueError('launcher changed within allocation')
        if a.command=='check-code': return 0
        report=validation_report(a.directory,record['code']['sha256'],record['gpu_uuid'])
        print(json.dumps(report,indent=2))
        return 0 if report['valid'] else 1
    except Exception as exc:
        print(f'R2-C2 VALIDATION FAILED: {type(exc).__name__}: {exc}',file=sys.stderr)
        # A code mismatch must not conceal an already recorded runtime failure.
        if a.command=='validate' and (a.directory/'manifest.json').exists():
            print(json.dumps(validation_report(a.directory),indent=2))
        return 1


if __name__=='__main__': raise SystemExit(main())
