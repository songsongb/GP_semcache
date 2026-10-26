#!/usr/bin/env python3
"""Aggregate disjoint R1-B2 checkpoints; reject duplicates and incomplete runs by default."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from semcache.metrics.r1 import atomic_json
from semcache.metrics.r1b2 import load_checkpoints, reports, summarize


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',type=Path,nargs='+',required=True,help='Run roots or tokens_<length> directories')
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--allow-incomplete',action='store_true',help='Diagnostic partial aggregation; never silently ignore failures')
    a=p.parse_args(argv)
    manifests=[]
    for root in a.inputs:
        paths=[root/'manifest.json'] if (root/'manifest.json').exists() else sorted(root.glob('tokens_*/manifest.json'))
        if not paths: raise ValueError('No manifests: '+str(root))
        manifests.extend(paths)
    if len(set(x.resolve() for x in manifests))!=len(manifests): raise ValueError('Overlapping input directories')
    rows=[]; sources=[]
    for path in manifests:
        m=json.loads(path.read_text())
        if m.get('experiment')!='R1-B2': raise ValueError('Not R1-B2: '+str(path))
        if m['status']!='COMPLETE' and not a.allow_incomplete: raise ValueError('Incomplete/failed run: '+str(path)+' '+m['status'])
        part=load_checkpoints(path.parent,m['identity'])
        if m['status']=='COMPLETE' and len(part)!=m['rows']: raise ValueError('Completed manifest row count mismatch')
        rows.extend(part)
        sources.append(dict(path=str(path.resolve()),identity=m['identity'],status=m['status'],rows=len(part),
            comparison_identity=m.get('comparison_identity')))
    summarize(rows)  # Validate before creating any output.
    a.output_dir.mkdir(parents=True,exist_ok=False)
    reports(a.output_dir,rows,aggregate=True)
    atomic_json(a.output_dir/'manifest.json',dict(experiment='R1-B2 aggregate',sources=sources,
        status='COMPLETE' if all(s['status']=='COMPLETE' for s in sources) else 'PARTIAL_INPUTS',
        comparison_identities=sorted({r['comparison_identity'] for r in rows}),
        incompatible_identities_pooled=False,reference_reproduced=False,
        missing_measurements_extrapolated=False))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
