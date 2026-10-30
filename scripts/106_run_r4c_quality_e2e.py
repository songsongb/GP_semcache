#!/usr/bin/env python3
"""C1 bounded OPT quality pilot and independent read-only artifact analysis."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import traceback

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from semcache.edgelora.communication import require
from semcache.edgelora.r3a_artifacts import now,read,sha,source_identity,write
from semcache.edgelora.r4a_evidence import verify_sources
from semcache.edgelora.r4c_evidence import fresh,seal_directory,verify_directory
from semcache.edgelora.r4c_identity import sources
from semcache.edgelora.r4c_runtime import flags


def gpu_preflight():
    require(os.environ.get('R4C_ENABLE_GPU')=='1','explicit R4C_ENABLE_GPU=1 required')
    require(source_identity()['git_branch']=='exp/raw-semcache-r4c','C1 work branch exp/raw-semcache-r4c required')
    require(os.environ.get('SLURM_JOB_ID') and os.environ.get('CUDA_VISIBLE_DEVICES') and len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))==1,'one explicit Slurm RTX3090 allocation required')
    require(os.environ.get('HF_HUB_OFFLINE')=='1' and os.environ.get('TRANSFORMERS_OFFLINE')=='1','offline HF mode required')
    missing=[n for n in ('torch','numpy','transformers','peft','rouge','sacrebleu') if importlib.util.find_spec(n) is None]
    require(not missing,'MISSING_DEPENDENCY: '+', '.join(missing))
    from importlib.metadata import version
    require(version('rouge')=='1.0.1','MISSING_DEPENDENCY: pinned rouge==1.0.1 required')
    require(os.environ.get('CUDA_LAUNCH_BLOCKING')!='1','CUDA_LAUNCH_BLOCKING invalidates timing')


def parse(argv):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--phase',choices=('dry-run','fixture','run','analyze'));p.add_argument('--output-dir',type=Path)
    p.add_argument('--prepared',type=Path);p.add_argument('--base-prepared',type=Path);p.add_argument('--inputs',type=Path)
    p.add_argument('--test-fixture-only',action='store_true');p.add_argument('--enable-gpu-experiment',action='store_true')
    p.add_argument('--capacity-bytes',type=int,default=256*2**20);p.add_argument('--evidence-budget-bytes',type=int,default=2*2**30)
    p.add_argument('--fixture-qa-budget',type=int,default=5);p.add_argument('--fixture-summary-budget',type=int,default=7)
    p.add_argument('--fixture-policy',choices=('quality','fixed'),default='quality')
    p.add_argument('--worker',choices=('es','ud'),help=argparse.SUPPRESS);p.add_argument('--plan',type=Path,help=argparse.SUPPRESS);a=p.parse_args(argv)
    if a.worker:
        if not a.plan:p.error('worker plan required')
        return a
    if not a.phase or not a.output_dir:p.error('--phase and --output-dir required')
    if a.phase=='fixture' and not a.test_fixture_only:p.error('fixture requires explicit --test-fixture-only')
    if a.test_fixture_only!=(a.phase=='fixture'):p.error('--test-fixture-only belongs to fixture only')
    if a.enable_gpu_experiment!=(a.phase=='run'):p.error('run requires explicit --enable-gpu-experiment')
    if a.phase=='run' and (a.base_prepared is None or a.prepared is None):p.error('run requires --base-prepared and --prepared')
    if a.phase!='run' and (a.base_prepared or a.prepared):p.error('prepared paths belong to run only')
    if (a.inputs is not None)!=(a.phase=='analyze'):p.error('--inputs required only for analyze')
    if not 0<a.capacity_bytes<=2**30 or not 0<a.evidence_budget_bytes<=16*2**30:p.error('cache/evidence budget out of range')
    return a


def worker(a):
    plan=read(a.plan);out=Path(plan['output']);result={}
    try:
        verify_sources(plan['sources'])
        if not plan['fixture']:require(os.environ.get('R4C_ENABLE_GPU')=='1','GPU worker opt-in missing')
        if a.worker=='ud':
            from semcache.edgelora.r4c_dynamic_ud import serve
            serve(plan,out);result=read(out/'ud_result.json')
        else:
            from semcache.edgelora.r4c_runtime import execute
            result=execute(plan,out)
    except BaseException as exc:
        result=dict(execution_status='FAILED',stage=read(out/'r4c_failure.json').get('stage') if (out/'r4c_failure.json').exists() else 'WORKER_START',
                    reason=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc());traceback.print_exc()
    result.update(pid=os.getpid(),finished_at=now(),**flags(plan['fixture']));write(out/(a.worker+'_result.json'),result)
    return 0 if result['execution_status']=='COMPLETE' else 1


def main(argv=None):
    a=parse(argv)
    if a.worker:return worker(a)
    try:out=fresh(a.output_dir,(a.prepared,a.base_prepared,a.inputs))
    except (OSError,ValueError) as exc:print(str(exc),file=sys.stderr);return 1
    fixture=a.phase=='fixture';result=dict(execution_status='RUNNING',stage='PREFLIGHT',created_at=now(),sources=sources(),source_identity=source_identity(),gpu_inference_performed=False,**flags(fixture))
    try:
        if a.phase=='dry-run':
            result['preflight']=dict(missing_dependencies=[n for n in ('torch','numpy','transformers','peft','datasets','rouge','sacrebleu') if importlib.util.find_spec(n) is None],
                datasets='MISSING_UNTIL_EXPLICIT_LOCAL_REGISTRY_VALIDATED',gpu_run=False,network='localhost only',downloads=False)
        elif a.phase=='analyze':
            from semcache.edgelora.r4c_analysis import analyze,export
            result.update(analysis_only=True,inputs=str(a.inputs.resolve()),stage='INDEPENDENT_READ_ONLY_ANALYSIS')
            analysis=analyze(a.inputs,sealed=True);export(out,analysis);fixture=analysis['decision']['TEST_FIXTURE_ONLY']
            result.update(flags(fixture),audit=analysis['audit'],decision=analysis['decision'],input_manifest_sha256=sha(a.inputs/'r4c_output_manifest.json'))
        else:
            from semcache.edgelora.r4c_prepare import fixture_inputs,prepare_collection,load_prepared
            if fixture:
                from semcache.edgelora.r3a_tiny import prepare_tiny
                base=prepare_tiny(out/'base_prepare');prepared=out/'prepared';prepared.mkdir()
                registry,tok=fixture_inputs(out/'synthetic_data')
                prepare_collection(registry,tok,prepared,count=1,qa_budget=a.fixture_qa_budget,summary_budget=a.fixture_summary_budget,
                    policy_mode=a.fixture_policy,strict_count=1,fixture=True)
            else:
                gpu_preflight();base=a.base_prepared.resolve();prepared=a.prepared.resolve()
            manifest,_,_=load_prepared(prepared)
            require(manifest['TEST_FIXTURE_ONLY']==fixture,'prepared scope differs')
            for name in ('r4c_dataset_manifest.json','r4c_prepared_prompts.jsonl','r4c_prefix_coverage.json','r4c_schedule.json'):
                (out/name).write_bytes((prepared/name).read_bytes())
            result['stage']='PHYSICAL_EXECUTION';plan=dict(fixture=fixture,base_prepared=str(base),prepared=str(prepared),device='cpu' if fixture else 'cuda:0',
                base_manifest_sha256=sha(base/'r3a_prepare_manifest.json'),sources=result['sources'],capacity_bytes=a.capacity_bytes,evidence_budget_bytes=a.evidence_budget_bytes)
            es=out/'es';es.mkdir();write(es/'es_plan.json',dict(plan,output=str(es)))
            from semcache.edgelora.r3cf_evidence import legacy_runner
            runner=legacy_runner().process_runner()
            env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
            runner([sys.executable,'-u',str(Path(__file__).resolve()),'--worker','es','--plan',str(es/'es_plan.json')],es,timeout=7500.,env=env)
            result['model_result']=read(es/'es_result.json');result['stage']='INDEPENDENT_AUDIT_AND_ANALYSIS'
            from semcache.edgelora.r4c_analysis import analyze,export
            analysis=analyze(out);export(out,analysis);result.update(audit=analysis['audit'],decision=analysis['decision'],gpu_inference_performed=not fixture)
            from semcache.edgelora.r4c_evidence import directory_bytes
            require(directory_bytes(out)<=a.evidence_budget_bytes,'EVIDENCE_BUDGET_EXCEEDED: complete run and analysis reports exceed budget')
        verify_sources(result['sources']);result.update(execution_status='COMPLETE',stage='COMPLETE')
    except BaseException as exc:
        result.update(execution_status='FAILED',reason=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc());traceback.print_exc()
        write(out/'r4c_failure.json',result)
        if not (out/'r4c_decision.json').exists():write(out/'r4c_decision.json',dict(**flags(fixture),R4C_PHYSICAL_E2E_PASS=False,R4C_EXACT_PREFIX_REUSE_PASS=False,R4C_INDEPENDENT_AUDIT_PASS=False))
    finally:
        if any(out.rglob('gpu_execution.json')):result['gpu_inference_performed']=True
        write(out/'r4c_limitations.json',dict(status='CPU_TESTED_GPU_NOT_RUN' if fixture else 'DIAGNOSTIC_ONLY',
            native_peft='FAILED_HISTORICAL_DEFERRED; no new Native parity claim',adapter='user_a untrained controlled adapter',
            evaluation='LONGBENCH_DERIVED_TRUNCATED_EVALUATION; not official leaderboard',network='LOCALHOST_TCP',
            workload='CONTROLLED_PROACTIVE_PREFIX_WARMING; no natural traffic hit rate',qualification='never correctness-qualified while deferred gates fail',
            numerical='B2 FP16 shape-dependent numerical differences remain possible; strict bitwise measured separately',
            evidence='STRICT_AUDIT stores QKV, prefill KV, components, logits; decode KV append verified live with digest chain; SCALABLE stores compact digests and full next-token logits',
            evidence_budget_bytes=a.evidence_budget_bytes,cache_capacity_bytes=a.capacity_bytes,qwen='C2 not implemented',repeated_experiments='C3 not implemented'))
        result['finished_at']=now();write(out/'r4c_results.json',result);seal_directory(out)
    print(json.dumps(dict(results=str(out/'r4c_results.json'),status=result['execution_status'])));return 0 if result['execution_status']=='COMPLETE' else 1


if __name__=='__main__':
    signal.signal(signal.SIGTERM,lambda signum,frame:sys.exit(128+signum));sys.exit(main())
