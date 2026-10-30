#!/usr/bin/env python3
"""Offline C0 LongBench preparation. No download or physical inference."""
import argparse
import json
from pathlib import Path
import sys
import traceback

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from semcache.edgelora.communication import require
from semcache.edgelora.r3a_artifacts import read,validate_prepared,write
from semcache.edgelora.r4c_dataset import DATASETS,load_dataset
from semcache.edgelora.r4c_evidence import fresh,seal_directory
from semcache.edgelora.r4c_prepare import opt_tokenizer,prepare_collection


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--registry',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--base-prepared',type=Path);p.add_argument('--validate-datasets-only',action='store_true');p.add_argument('--datasets',nargs='+',choices=DATASETS,default=list(DATASETS))
    p.add_argument('--count',type=int,default=2);p.add_argument('--seed',type=int,default=42);p.add_argument('--cohort',choices=('development','independent'),default='development')
    p.add_argument('--min-input-tokens',type=int,help='actual original formatted input eligibility floor; default prefix+1')
    p.add_argument('--exclude-cohort',type=Path);p.add_argument('--input-tokens',type=int,default=256);p.add_argument('--prefix-tokens',type=int,default=128)
    p.add_argument('--qa-budget',type=int,default=32);p.add_argument('--summary-budget',type=int,default=64);p.add_argument('--policy',choices=('quality','fixed'),default='quality')
    p.add_argument('--strict-count',type=int,default=1);p.add_argument('--enable-512',action='store_true');a=p.parse_args(argv)
    if not a.validate_datasets_only and a.base_prepared is None:p.error('--base-prepared required for actual model/tokenizer configuration')
    try:
        protected=[a.registry,a.base_prepared,a.exclude_cohort]
        if a.registry.is_file():
            configured=read(a.registry)
            protected.extend((a.registry.parent/spec['path']).resolve() for spec in configured.values())
        out=fresh(a.output_dir,protected)
    except (ValueError,OSError,KeyError) as exc:print(str(exc),file=sys.stderr);return 1
    result=dict(execution_status='RUNNING',stage='DATASET_VALIDATION',gpu_inference_performed=False,TEST_FIXTURE_ONLY=False)
    try:
        require(a.registry.is_file(),'MISSING_DATASET: local registry absent: '+str(a.registry))
        registry=read(a.registry)
        for spec in registry.values():spec['path']=str((a.registry.parent/spec['path']).resolve())
        for name in a.datasets:
            require(name in registry,'MISSING_DATASET: expected registry entry '+name)
            load_dataset(name,registry[name])
        if not a.validate_datasets_only:
            result['stage']='TOKENIZER_PROMPT_PREPARATION';pm,_,_=validate_prepared(a.base_prepared)
            excluded=[]
            if a.exclude_cohort:
                import json
                excluded=[r['sample_id'] for r in map(json.loads,a.exclude_cohort.read_text().splitlines())]
            manifest,_,_=prepare_collection(registry,opt_tokenizer(pm),out,datasets=a.datasets,count=a.count,seed=a.seed,cohort=a.cohort,excluded_ids=excluded,
                input_budget=a.input_tokens,qa_budget=a.qa_budget,summary_budget=a.summary_budget,prefix_length=a.prefix_tokens,policy_mode=a.policy,
                strict_count=a.strict_count,enable_512=a.enable_512,min_input_tokens=a.min_input_tokens)
            result['dataset_manifest']=manifest
        result.update(execution_status='COMPLETE',stage='COMPLETE')
    except BaseException as exc:result.update(execution_status='FAILED',reason=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
    write(out/'r4c_prepare_result.json',result);seal_directory(out);print(json.dumps(result));return 0 if result['execution_status']=='COMPLETE' else 1


if __name__=='__main__':sys.exit(main())
