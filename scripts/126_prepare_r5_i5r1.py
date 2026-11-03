#!/usr/bin/env python3
"""CPU-only real-data controlled I5R-1 preparation and immutable-plan inspection."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from semcache.edgelora.r3a_artifacts import read
from semcache.edgelora.r5_i5r1_plan import prepare,inspect,PROFILES,DATASETS,CASES


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='phase',required=True)
    q=sub.add_parser('prepare');q.add_argument('--registry',type=Path,required=True);q.add_argument('--multi-prepared',type=Path,required=True)
    q.add_argument('--output-dir',type=Path,required=True);q.add_argument('--profile',choices=PROFILES,default='SMOKE')
    q.add_argument('--datasets',nargs='+',choices=DATASETS);q.add_argument('--cases',nargs='+',choices=CASES)
    q.add_argument('--groups',type=int);q.add_argument('--adapters',nargs='+',choices=('Adapter_A','Adapter_B','Adapter_C'))
    q.add_argument('--target-repeats',nargs='+',type=int);q.add_argument('--generation-tokens',type=int,default=8)
    q.add_argument('--sharing',choices=('SAME_USER','AUTHORIZED_CROSS_USER'),default='SAME_USER')
    q.add_argument('--seed',type=int,default=42);q.add_argument('--capacity-bytes',type=int,default=512*2**20)
    q.add_argument('--evidence-budget-bytes',type=int,default=8*2**30);q.add_argument('--seconds-per-forward',type=float,default=5.)
    q.add_argument('--r0-diagnostic',type=Path)
    q=sub.add_parser('inspect');q.add_argument('--experiment',type=Path,required=True)
    a=p.parse_args()
    if a.phase=='inspect':result=inspect(a.experiment)
    else:
        from semcache.edgelora.r5_multi_artifacts import validate
        from semcache.edgelora.r4c_prepare import opt_tokenizer
        from semcache.edgelora.r5_i5_quality import validate_scorers
        m,pm,_,_=validate(a.multi_prepared);selected=a.datasets or PROFILES[a.profile]['datasets']
        registry=read(a.registry);validate_scorers({d:registry[d] for d in selected})
        tok=opt_tokenizer(pm);special={k:getattr(tok.tokenizer,k) for k in ('bos_token_id','eos_token_id','pad_token_id')}
        prepare(registry,tok.tokenizer,tok.identity,tok.supported_context_length,special,a.output_dir,multi_prepared=a.multi_prepared,
            profile=a.profile,datasets=a.datasets,cases=a.cases,groups=a.groups,adapters=a.adapters,repeats=a.target_repeats,
            generation_tokens=a.generation_tokens,sharing_mode=a.sharing,seed=a.seed,capacity=a.capacity_bytes,
            evidence_budget=a.evidence_budget_bytes,seconds_per_forward=a.seconds_per_forward,r0_diagnostic=a.r0_diagnostic)
        result=inspect(a.output_dir)
    print(json.dumps(result,sort_keys=True,indent=2));return 0


if __name__=='__main__':sys.exit(main())
