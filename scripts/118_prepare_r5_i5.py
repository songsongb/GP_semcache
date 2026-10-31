#!/usr/bin/env python3
"""Offline I5 native dataset normalization and tokenizer-exact block preparation."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from semcache.edgelora.r3a_artifacts import read
from semcache.edgelora.r5_i5_data import DATASETS,prepare_dataset
from semcache.edgelora.r5_i5_workload import prepare_experiment


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='phase',required=True)
    ad=sub.add_parser('adapters');ad.add_argument('--base-prepared',type=Path,required=True);ad.add_argument('--output-dir',type=Path,required=True)
    d=sub.add_parser('dataset');d.add_argument('--dataset',choices=DATASETS,required=True)
    d.add_argument('--source',type=Path,required=True);d.add_argument('--revision',required=True);d.add_argument('--split',required=True)
    d.add_argument('--format',choices=('json','jsonl','hf_saved_dataset'),default='json');d.add_argument('--ontology',type=Path)
    d.add_argument('--output-dir',type=Path,required=True)
    e=sub.add_parser('experiment');e.add_argument('--registry',type=Path,required=True);e.add_argument('--base-prepared',type=Path,required=True)
    e.add_argument('--output-dir',type=Path,required=True);e.add_argument('--scale',choices=('SMOKE','SMALL','MEDIUM','LARGE'),default='SMOKE')
    e.add_argument('--tracks',nargs='+',choices=('NATURAL','CONTROLLED'),default=['NATURAL'])
    e.add_argument('--prefixes',nargs='+',type=int,default=[64,128,256,512]);e.add_argument('--counts',type=Path);e.add_argument('--budgets',type=Path)
    e.add_argument('--repetitions',type=int,default=2);e.add_argument('--block-size',type=int,default=5);e.add_argument('--seed',type=int,default=42)
    e.add_argument('--strict-per-block',type=int,default=1);e.add_argument('--capacity-bytes',type=int,default=512*2**20)
    e.add_argument('--evidence-budget-bytes',type=int,default=8*2**30);e.add_argument('--input-limit',type=int)
    e.add_argument('--seconds-per-forward',type=float,default=5.);e.add_argument('--allow-context-truncation',action='store_true')
    a=p.parse_args(argv)
    if a.phase=='adapters':
        from semcache.edgelora.r5_i5_artifacts import prepare
        print(prepare(a.base_prepared,a.output_dir,'expanded'));return 0
    if a.phase=='dataset':
        m=prepare_dataset(a.dataset,a.source,a.output_dir,revision=a.revision,split=a.split,kind=a.format,ontology_path=a.ontology)
        print(json.dumps(dict(status='PREPARED',samples=m['samples'])));return 0
    from semcache.edgelora.r3a_artifacts import validate_prepared
    from semcache.edgelora.r4c_prepare import opt_tokenizer
    from semcache.edgelora.r5_i5_quality import validate_scorers
    validate_scorers(read(a.registry))
    pm,_,_=validate_prepared(a.base_prepared);tok=opt_tokenizer(pm)
    special={k:getattr(tok.tokenizer,k) for k in ('bos_token_id','eos_token_id','pad_token_id')}
    m=prepare_experiment(read(a.registry),tok.tokenizer,tok.identity,tok.supported_context_length,special,a.output_dir,
        scale=a.scale,overrides=read(a.counts) if a.counts else None,budgets=read(a.budgets) if a.budgets else None,
        tracks=a.tracks,prefixes=a.prefixes,repetitions=a.repetitions,block_size=a.block_size,seed=a.seed,
        strict_per_block=a.strict_per_block,capacity=a.capacity_bytes,evidence_budget=a.evidence_budget_bytes,
        allow_truncation=a.allow_context_truncation,input_limit=a.input_limit,seconds_per_forward=a.seconds_per_forward)
    print(json.dumps(dict(status='PREPARED',blocks=len(m['blocks']),GPU_NOT_RUN=True)));return 0


if __name__=='__main__':sys.exit(main())
