#!/usr/bin/env python3
"""CPU integration audit and fail-closed physical-subsequence readiness check."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from semcache.edgelora.r5_mstage_bridge import BLOCKERS,MODE
from semcache.edgelora.r5_i5r2_tensors import Evidence
from semcache.edgelora.r5_i5r1_plan import source_identity
from semcache.edgelora.r5_i5_evidence import seal,sources
from semcache.edgelora.r3a_artifacts import write
from semcache.edgelora.communication import require

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='action',required=True)
    a=s.add_parser('prepare');a.add_argument('--output-dir',required=True)
    a=s.add_parser('gpu-preflight');a.add_argument('--experiment',required=True);a.add_argument('--source-commit',required=True)
    args=p.parse_args(argv)
    try:
        if args.action=='prepare':
            from semcache.edgelora.r5_i5r_parallel import output
            plan=dict(schema='R5_MSTAGE_GAP_V1',source=source_identity(),mode=MODE,conditions=['FULL_RECOMPUTE','SEMANTIC_NO_REUSE','SEMCACHE_SUBSEQUENCE'],
                      physical_execution_supported=False,blockers=list(BLOCKERS),GPU_NOT_RUN=True,physical_hits='NOT_EVALUATED',correctness_qualified_speedup='NOT_ESTABLISHED')
            out=output(args.output_dir);write(out/'mstage_integration_plan.json',plan);seal(out);print(json.dumps(plan,indent=2));return 0
        ev=Evidence(args.experiment);ev.verify();plan=ev.json('mstage_integration_plan.json')
        require(plan['source']==source_identity() and plan['source']['commit']==args.source_commit,'pinned source differs')
        require(plan['physical_execution_supported'] is False and plan['blockers']==list(BLOCKERS),'readiness evidence differs')
        print(json.dumps(dict(status='BLOCKED_NOT_GPU_READY',blockers=list(BLOCKERS),GPU_NOT_RUN=True)),file=sys.stderr);return 2
    except (ValueError,KeyError,FileNotFoundError,ImportError) as exc:
        print(json.dumps(dict(status='REJECTED',reason=str(exc))),file=sys.stderr);return 2

if __name__=='__main__':sys.exit(main())
