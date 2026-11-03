#!/usr/bin/env python3
"""Fresh bounded arithmetic diagnostic from the immutable historical token plan."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from semcache.edgelora.r5_i5r2b_plan import prepare,validate_plan

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--historical-plan',type=Path,required=True)
    p.add_argument('--r2a-review',type=Path,required=True);p.add_argument('--multi-prepared',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args()
    from semcache.edgelora.r5_i5r2b_capture import installed_observer_check
    check=installed_observer_check()
    from semcache.edgelora.r5_multi_artifacts import validate
    from semcache.edgelora.r4c_prepare import opt_tokenizer
    from semcache.edgelora.r3a_artifacts import read
    _,pm,_,_=validate(a.multi_prepared);tok=opt_tokenizer(pm)
    target=read(a.historical_plan/'i5_workload_manifest.json')['blocks'][0]['tasks'][1]
    from semcache.edgelora.communication import require
    require(list(tok.tokenizer.encode(target['input_text'],add_special_tokens=False))==target['token_ids'],'historical tokenizer IDs differ')
    plan=prepare(a.historical_plan,a.multi_prepared,a.output_dir,r2a_review=a.r2a_review)
    validate_plan(a.output_dir,a.multi_prepared)
    print(json.dumps(dict(status='PREPARED_GPU_NOT_RUN',blocks=1,array='0-0%1',source_commit=plan['i5r2b']['source']['commit'],
        observer_check=check,resources=plan['blocks'][0]['resources']),indent=2))

if __name__=='__main__':main()
