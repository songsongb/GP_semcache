#!/usr/bin/env python3
"""Auditable Raw-QKV-backed exact shared-prefix KV reuse: bounded B2 pilot."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import traceback

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from semcache.edgelora.communication import digest,require
from semcache.edgelora.r3a_artifacts import now,read,sha,source_identity,write
from semcache.edgelora.r3a_evidence import jsonl
from semcache.edgelora.parity_forensics_audit import Evidence
from semcache.edgelora.r4a_evidence import load_prepared,verify_sources
from semcache.edgelora.r4b2_analysis import analyze,export
from semcache.edgelora.r4b2_evidence import audit_run,load_run
from semcache.edgelora.r4b2_prefix_runtime import flags,load_inputs
from semcache.edgelora.r4b2_workload import make_suite


def frozen_runner():
    spec=importlib.util.spec_from_file_location('b2_frozen_b1',ROOT/'scripts/103_run_r4b1_multi_entry.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def sources():
    result=frozen_runner().sources()
    paths=[Path(__file__).resolve(),ROOT/'jobs/r4b2_opt_shared_prefix.sh',ROOT/'docs/R4B2_EXACT_SHARED_PREFIX.md',
        *sorted((ROOT/'src/semcache/edgelora').glob('r4b2_*.py'))]
    result.update({str(p.relative_to(ROOT)):sha(p) for p in paths});return result


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=('dry-run','fixture','prepare','run','analyze'));p.add_argument('--output-dir',type=Path)
    p.add_argument('--base-prepared',type=Path);p.add_argument('--prepared',type=Path);p.add_argument('--inputs',type=Path)
    p.add_argument('--target-length',type=int,choices=(64,128));p.add_argument('--max-new-tokens',type=int,choices=(3,8,16))
    p.add_argument('--prefix-lengths',type=int,nargs='+');p.add_argument('--capacity-bytes',type=int,default=128*2**20)
    p.add_argument('--device',default='cuda:0');p.add_argument('--enable-gpu-experiment',action='store_true');p.add_argument('--test-fixture-only',action='store_true')
    p.add_argument('--worker',choices=('es','ud'),help=argparse.SUPPRESS);p.add_argument('--plan',type=Path,help=argparse.SUPPRESS)
    a=p.parse_args(argv)
    if a.worker:
        if not a.plan or a.enable_gpu_experiment==a.test_fixture_only:p.error('worker requires plan and explicit scope')
        return a
    if not a.phase or not a.output_dir:p.error('--phase and --output-dir required')
    if not 0<a.capacity_bytes<=2**30:p.error('capacity must be in (0, 1 GiB]')
    if a.phase in ('prepare','run') and (not a.enable_gpu_experiment or a.test_fixture_only or a.device!='cuda:0'):p.error('GPU requires explicit allocated cuda:0 opt-in')
    if a.phase=='fixture' and (not a.test_fixture_only or a.enable_gpu_experiment):p.error('fixture requires --test-fixture-only')
    if a.phase in ('dry-run','analyze') and (a.test_fixture_only or a.enable_gpu_experiment):p.error('read-only phase executes no model')
    for attr,phase in (('base_prepared','prepare'),('prepared','run'),('inputs','analyze')):
        if (getattr(a,attr) is not None)!=(a.phase==phase):p.error('--'+attr.replace('_','-')+' required only for '+phase)
    if (a.target_length or a.max_new_tokens or a.prefix_lengths) and a.phase not in ('prepare','fixture','dry-run'):p.error('workload frozen at preparation')
    if a.phase=='prepare' and a.max_new_tokens==3:p.error('GPU policy is G8 or G16')
    return a


def output_dir(path,protected):
    path=path.resolve();base=ROOT/'results/r4/b2'
    require(path.is_relative_to(base) and path!=base,'new results/r4/b2 output required')
    require(all(p is None or not path.is_relative_to(p.resolve()) for p in protected),'output would alter inputs')
    require(not path.exists() or path.is_dir() and not any(path.iterdir()),'refusing nonempty output')
    path.mkdir(parents=True,exist_ok=True);return path


def gpu_preflight():
    require(os.environ.get('R4B2_ENABLE_GPU')=='1','explicit R4B2_ENABLE_GPU=1 required')
    require(source_identity()['git_branch']=='exp/raw-semcache-r4b2','R4-B2 requires exp/raw-semcache-r4b2')
    require(os.environ.get('SLURM_JOB_ID') and os.environ.get('CUDA_VISIBLE_DEVICES') and len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))==1,
        'one explicit Slurm RTX3090 allocation required')
    missing=[n for n in ('torch','numpy','transformers','peft','sacrebleu') if importlib.util.find_spec(n) is None]
    require(not missing,'MISSING_DEPENDENCY: '+', '.join(missing)+'; no installation/fallback')
    require(os.environ.get('CUDA_LAUNCH_BLOCKING')!='1','CUDA_LAUNCH_BLOCKING invalidates timing')


def worker_main(a):
    plan=read(a.plan);out=Path(plan['output']);result={}
    try:
        verify_sources(plan['sources']);require(plan['fixture']==a.test_fixture_only,'worker scope differs')
        if not plan['fixture']:require(os.environ.get('R4B2_ENABLE_GPU')=='1','GPU worker opt-in missing')
        if a.worker=='ud':
            from semcache.edgelora.r4b2_dynamic_ud import serve
            serve(plan,out);result=read(out/'ud_result.json')
        else:
            from semcache.edgelora.r4b2_prefix_runtime import execute
            result=execute(plan,out)
    except BaseException as exc:
        result=dict(execution_status='FAILED',reason=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc());traceback.print_exc()
    result.update(pid=os.getpid(),finished_at=now());result.update(flags(plan['fixture']));write(out/(a.worker+'_result.json'),result)
    return 0 if result['execution_status']=='COMPLETE' else 1


def negative_decision(fixture,status='NOT_RUN',reason=None):
    return dict(execution_status=status,reason=reason,**{k:False for k in ('R4B2_PHYSICAL_E2E_PASS','R4B2_EXACT_PREFIX_MATCH_PASS',
        'R4B2_PREFIX_KV_INTEGRITY_PASS','R4B2_SUFFIX_ONLY_EXECUTION_PASS','R4B2_MULTI_TARGET_REUSE_PASS','R4B2_STRICT_BITWISE_FIDELITY_PASS',
        'R4B2_TTFT_BENEFIT_OBSERVED','R4B2_E2E_BENEFIT_OBSERVED')},**flags(fixture))


def main(argv=None):
    a=parse_args(argv)
    if a.worker:return worker_main(a)
    try:out=output_dir(a.output_dir,(a.base_prepared,a.prepared,a.inputs))
    except (ValueError,OSError) as exc:print(str(exc),file=sys.stderr);return 1
    fixture=a.phase=='fixture';report=dict(execution_status='RUNNING',phase=a.phase,created_at=now(),sources=sources(),
        production_source_identity=source_identity(),gpu_inference_performed=False,**flags(fixture))
    decision=negative_decision(fixture);checks=[];pairs=[];quality=dict(status='NOT_RUN');evidence=Evidence()
    try:
        for name,h in report['sources'].items():evidence.file(ROOT/name,h)
        write(out/'r4b2_preflight.json',dict(branch=report['production_source_identity']['git_branch'],capacity_bytes=a.capacity_bytes,
            missing_dependencies=[n for n in ('torch','numpy','transformers','peft','sacrebleu') if importlib.util.find_spec(n) is None],
            installed_OPT_inspection='performed in actual model worker before admission',downloads_performed=False))
        if a.phase=='dry-run':pass
        elif a.phase=='analyze':
            old,plan,worker=load_run(a.inputs);fixture=plan['fixture'];report.update(flags(fixture));evidence.file(a.inputs/'r4b2_results.json')
            _,suite,*_=load_inputs(plan['prepared']);rows=jsonl(a.inputs/'es/r4b2_requests.jsonl');tokens=jsonl(a.inputs/'es/r4b2_tokens.jsonl')
            decision,checks,pairs,quality=analyze(a.inputs/'es',suite,rows,tokens,fixture=fixture,audit=old['audit'])
            report.update(analysis_only=True,input_directory=str(a.inputs.resolve()),audit=old['audit'])
        else:
            frozen=frozen_runner().frozen_runner();runner=frozen.legacy_runner().process_runner()
            if not fixture:gpu_preflight()
            if a.phase in ('prepare','fixture'):
                if fixture:
                    from semcache.edgelora.r3a_tiny import prepare_tiny
                    base=prepare_tiny(out/'base_prepare')
                else:base=a.base_prepared.resolve()
                prep=out/'prepared';prep.mkdir()
                original_plan=dict(base_prepared=str(base),fixture=fixture,device='cpu' if fixture else a.device,scope='pilot',policy='fixed',sources=report['sources'])
                # Frozen Native original-input controls and their original metrics
                # are preserved. B2 Native generation equivalence stays deferred.
                frozen.prepare_artifacts(original_plan,prep/'native',runner)
                native,_,pm,ref,am=load_prepared(prep/'native');tokenizer=None
                if not fixture:
                    from transformers import AutoTokenizer
                    tokenizer=AutoTokenizer.from_pretrained(pm['identity']['tokenizer_source'],revision=pm['identity']['tokenizer_revision'],local_files_only=True)
                    require(digest(tokenizer.get_vocab())==native['native_result']['tokenizer_vocab_sha256'],'suite tokenizer differs from actual Native tokenizer')
                length=a.target_length or 128;prefixes=tuple(a.prefix_lengths or [p for p in (32,64,96) if p<length])
                suite=make_suite(tokenizer,fixture=fixture,target_length=length,new_tokens=a.max_new_tokens or (3 if fixture else 8),prefix_lengths=prefixes)
                write(prep/'r4b2_suite.json',suite)
                write(prep/'r4b2_prepared.json',dict(schema='r4b2_prepared_v2',base_prepared=str(base),native_prepared=str(prep/'native'),
                    sources=report['sources'],TEST_FIXTURE_ONLY=fixture,hashes={'r4b2_suite.json':sha(prep/'r4b2_suite.json')},
                    native_B2_generation='NOT_RUN_DEFERRED'))
                report['prepared_directory']=str(prep)
            if a.phase in ('run','fixture'):
                prepared=out/'prepared' if fixture else a.prepared.resolve();m,suite,native,pm,ref,am=load_inputs(prepared)
                require(m['sources']==report['sources'] and m['TEST_FIXTURE_ONLY']==fixture,'B2 prepared scope/source mismatch')
                plan=dict(prepared=str(prepared),base_prepared=m['base_prepared'],fixture=fixture,device='cpu' if fixture else a.device,
                    sources=report['sources'],capacity_bytes=a.capacity_bytes)
                es=out/'es';es.mkdir();write(es/'es_plan.json',dict(plan,output=str(es)))
                runner([sys.executable,'-u',str(Path(__file__).resolve()),'--worker','es','--plan',str(es/'es_plan.json'),
                    '--test-fixture-only' if fixture else '--enable-gpu-experiment'],es,timeout=7500.,env=frozen.offline_env())
                worker=read(es/'es_result.json');report['audit']=audit_run(out,plan,worker);report['model_result']=worker
                rows=jsonl(es/'r4b2_requests.jsonl');tokens=jsonl(es/'r4b2_tokens.jsonl')
                decision,checks,pairs,quality=analyze(es,suite,rows,tokens,fixture=fixture,audit=report['audit'])
                report['gpu_inference_performed']=not fixture
        evidence.unchanged();report['execution_status']='COMPLETE'
    except BaseException as exc:
        report.update(execution_status='FAILED',reason=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc());traceback.print_exc()
        if (out/'es/es_result.json').exists():report['failed_worker']=read(out/'es/es_result.json')
        decision=negative_decision(fixture,'FAILED',report['reason'])
    finally:
        export(out,decision,checks,pairs,quality)
        if any(read(p).get('gpu_inference_performed') is True for p in out.rglob('gpu_execution.json')):report['gpu_inference_performed']=True
        report.update(finished_at=now(),decision=decision,input_hashes=list(evidence.files.values()))
        report['output_hashes']={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and p.name not in ('r4b2_results.json','r4b2_summary.txt')}
        write(out/'r4b2_results.json',report)
        (out/'r4b2_summary.txt').write_text('\n'.join(['R4B2_RAW_QKV_BACKED_PREFIX_KV','Execution: '+report['execution_status'],
            'Reason: '+report.get('reason','none'),'Actual GPU inference: '+str(report['gpu_inference_performed']),
            'Strict bitwise: '+str(decision.get('R4B2_NUMERICAL_FIDELITY_DIAGNOSTIC',{}).get('strict_bitwise')),
            'Native deferred; PRIMARY_PERFORMANCE_ELIGIBLE=False; correctness_qualified_speedup=False.'])+'\n')
    print(json.dumps(dict(report=str(out/'r4b2_results.json'),execution_status=report['execution_status'])))
    return 0 if report['execution_status']=='COMPLETE' else 1


if __name__=='__main__':
    signal.signal(signal.SIGTERM,lambda signum,frame:sys.exit(128+signum));sys.exit(main())
