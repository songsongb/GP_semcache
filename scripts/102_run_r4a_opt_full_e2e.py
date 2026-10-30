#!/usr/bin/env python3
"""Bounded OPT generation with physical CPU LoRA/TCP and exact total-QKV replay."""
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
from semcache.edgelora.r3a_evidence import jsonl
from semcache.edgelora.parity_forensics_audit import Evidence
from semcache.edgelora.r3cf_evidence import legacy_runner
from semcache.edgelora.r4a_evidence import audit_run,flags,load_prepared,load_run,make_suite,planned_counts,verify_sources
from semcache.edgelora.r4a_runtime import native_prepare,execute
from semcache.edgelora.r4a_analysis import analyze,export
from semcache.edgelora.r4a_quality import evaluate


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=('dry-run','fixture','prepare','run','analyze'))
    p.add_argument('--output-dir',type=Path);p.add_argument('--base-prepared',type=Path);p.add_argument('--prepared',type=Path);p.add_argument('--inputs',type=Path)
    p.add_argument('--device',default='cuda:0');p.add_argument('--scope',choices=('pilot','matrix'));p.add_argument('--generation-policy',choices=('fixed','natural-eos'))
    p.add_argument('--gold-jsonl',type=Path);p.add_argument('--enable-gpu-experiment',action='store_true');p.add_argument('--test-fixture-only',action='store_true')
    p.add_argument('--worker',choices=('native','es','ud'),help=argparse.SUPPRESS);p.add_argument('--plan',type=Path,help=argparse.SUPPRESS)
    a=p.parse_args(argv)
    if a.worker:
        if not a.plan or a.enable_gpu_experiment==a.test_fixture_only:p.error('worker requires plan and one explicit scope')
        return a
    if not a.phase or not a.output_dir:p.error('--phase and --output-dir required')
    if a.phase in ('prepare','run') and (not a.enable_gpu_experiment or a.test_fixture_only or a.device!='cuda:0'):p.error('GPU phases require explicit opt-in and allocated cuda:0')
    if a.phase=='fixture' and (not a.test_fixture_only or a.enable_gpu_experiment):p.error('fixture requires --test-fixture-only')
    if a.phase in ('dry-run','analyze') and (a.enable_gpu_experiment or a.test_fixture_only):p.error('read-only phases execute no model')
    if (a.base_prepared is not None)!=(a.phase=='prepare'):p.error('--base-prepared required only for R4 prepare (original 95 artifacts)')
    if (a.prepared is not None)!=(a.phase=='run'):p.error('--prepared required only for run (R4 artifacts)')
    if (a.inputs is not None)!=(a.phase=='analyze'):p.error('--inputs required only for analyze')
    if (a.scope or a.generation_policy) and a.phase not in ('prepare','fixture','dry-run'):p.error('scope/generation policy frozen during prepare')
    if a.gold_jsonl and a.phase!='run':p.error('--gold-jsonl supplied only on run; analyze consumes recorded gold')
    a.scope=a.scope or 'pilot';a.generation_policy=a.generation_policy or 'fixed';return a


def sources():
    result=legacy_runner().sources()
    paths=[Path(__file__).resolve(),ROOT/'scripts/101_run_r3c_break_even_focused.py',ROOT/'src/semcache/evaluation/bleu.py',
        ROOT/'src/semcache/models/model_adapter.py',ROOT/'jobs/r4a_opt_full_e2e.sh',ROOT/'docs/R4A_OPT_FULL_E2E.md',
        *sorted((ROOT/'src/semcache/edgelora').glob('r4a_*.py'))]
    result.update({str(p.relative_to(ROOT)):sha(p) for p in paths});return result


def output_dir(path,protected):
    path=path.resolve();base=ROOT/'results/r4/a'
    require(path.is_relative_to(base) and path!=base,'new results/r4/a/ output required')
    require(all(p is None or not path.is_relative_to(p.resolve()) for p in protected),'output cannot alter input artifacts')
    require(not path.exists() or path.is_dir() and not any(path.iterdir()),'refusing nonempty output')
    path.mkdir(parents=True,exist_ok=True);return path


def gpu_preflight():
    require(os.environ.get('SLURM_JOB_ID') and os.environ.get('CUDA_VISIBLE_DEVICES') and
        len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))==1,'one explicit Slurm GPU allocation required')
    require(source_identity()['git_branch']=='exp/raw-semcache-r4','wrong branch: R4-A requires exp/raw-semcache-r4')
    missing=[n for n in ('torch','numpy','transformers','peft','sacrebleu') if importlib.util.find_spec(n) is None]
    require(not missing,'MISSING_DEPENDENCY: '+', '.join(missing)+'; no automatic installation or GPU fallback')
    try:__import__('sacrebleu.metrics')
    except ImportError as exc:raise ValueError('MISSING_DEPENDENCY: broken SacreBLEU import; no installation') from exc
    require(os.environ.get('CUDA_LAUNCH_BLOCKING')!='1','CUDA_LAUNCH_BLOCKING invalidates timings')


def worker_main(a):
    plan=read(a.plan);out=Path(plan['output']);result={}
    try:
        verify_sources(plan['sources']);require(plan['fixture']==a.test_fixture_only,'worker scope mismatch')
        if a.worker=='native':result=native_prepare(plan,out)
        elif a.worker=='es':result=execute(plan,out)
        else:
            from semcache.edgelora.r4a_dynamic_ud import serve
            serve(plan,out);result=read(out/'ud_result.json')
    except BaseException as exc:
        partial=read(out/'es_partial.json') if (out/'es_partial.json').exists() else {}
        result=dict(partial,execution_status='FAILED',reason=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc());traceback.print_exc()
    result.update(flags(plan['fixture']));result.update(pid=os.getpid(),finished_at=now());write(out/(a.worker+'_result.json'),result)
    return 0 if result['execution_status']=='COMPLETE' else 1


def prepare_artifacts(plan,directory,runner):
    directory.mkdir(parents=True,exist_ok=True);pm=read(Path(plan['base_prepared'])/'r3a_prepare_manifest.json')
    write(directory/'native_plan.json',dict(plan,output=str(directory)))
    runner([sys.executable,'-u',str(Path(__file__).resolve()),'--worker','native','--plan',str(directory/'native_plan.json'),
        '--test-fixture-only' if plan['fixture'] else '--enable-gpu-experiment'],directory,timeout=1800.,env=offline_env())
    result=read(directory/'native_result.json');require(result['execution_status']=='COMPLETE','native preparation failed')
    write(directory/'r4a_prepared_manifest.json',dict(execution_status='COMPLETE',base_prepared=plan['base_prepared'],
        base_manifest_sha256=sha(Path(plan['base_prepared'])/'r3a_prepare_manifest.json'),native_result=result,sources=plan['sources'],
        native_process_exited=True,native_lifecycle=read(directory/'lifecycle.json'),**flags(plan['fixture']),
        output_hashes={str(p.relative_to(directory)):sha(p) for p in directory.rglob('*') if p.is_file() and p.name!='r4a_prepared_manifest.json'}))


def offline_env():return dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
    CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')


def finish_analysis(out,prepared,rows,tokens,worker,fixture,gold):
    contrasts,decision=analyze(out/'es' if (out/'es').exists() else Path(worker['evidence_directory']),prepared,rows,tokens,worker,fixture=fixture)
    native=read(prepared/'native_index.json')['generations'] if not fixture else None
    quality,signatures=evaluate(rows,native,gold)
    quality.update(native_peft_strict_parity_passed=decision['NATIVE_PEFT_STRICT_PARITY_PASS'],
        native_execution_scope='NOT_RUN_TEST_FIXTURE_ONLY' if fixture else 'matched independently executed Native PEFT reference; BLEU is not strict parity')
    available=quality['bleu_vs_ground_truth']
    decision['R4A_GOLD_BLEU_AVAILABLE']=available['status']=='EXTERNAL_GOLD_SUPPLIED' and all(
        s['corpus_bleu']['status']=='MEASURED' for s in available['modes'].values())
    write(out/'r4a_quality.json',quality);write(out/'r4a_sacrebleu_signatures.json',signatures);export(out,contrasts,decision)
    return decision


def main(argv=None):
    a=parse_args(argv)
    if a.worker:return worker_main(a)
    try:out=output_dir(a.output_dir,(a.base_prepared,a.prepared,a.inputs))
    except (ValueError,OSError) as exc:print(str(exc),file=sys.stderr);return 1
    fixture=a.phase=='fixture';evidence=Evidence()
    negative_gates={k:False for k in ('R4A_PHYSICAL_E2E_PASS','R4A_RAW_QKV_E2E_FIDELITY_PASS','R4A_TTFT_BENEFIT_OBSERVED',
        'R4A_E2E_BENEFIT_OBSERVED','NATIVE_PEFT_STRICT_PARITY_PASS','R4A_GOLD_BLEU_AVAILABLE','native_equivalent')}
    decision=dict(execution_status='NOT_RUN',**negative_gates,**flags(fixture))
    report=dict(execution_status='RUNNING',phase=a.phase,created_at=now(),program_version='r4a-v1',**flags(fixture),
        sources=sources(),production_source_identity=source_identity(),gpu_findings_status='NOT_RUN',gpu_inference_performed=False,
        actual_network_socket_used=False,cli=[sys.executable,str(Path(__file__).resolve()),*(sys.argv[1:] if argv is None else argv)])
    try:
        for name,h in report['sources'].items():evidence.file(ROOT/name,h)
        preflight=dict(execution_status='COMPLETE',missing_dependencies=[n for n in ('torch','numpy','transformers','peft','sacrebleu') if importlib.util.find_spec(n) is None],
            offline_required=True,no_models_or_datasets_downloaded=True,requested_scope=a.scope,policy=a.generation_policy)
        write(out/'r4a_preflight.json',preflight)
        if a.phase=='dry-run':report['readiness']=preflight
        elif a.phase=='analyze':
            old,plan,worker=load_run(a.inputs);fixture=plan['fixture'];report.update(flags(fixture))
            rows=jsonl(a.inputs/'r4a_requests.jsonl');tokens=jsonl(a.inputs/'r4a_tokens.jsonl')
            for name in ('r4a_requests.jsonl','r4a_tokens.jsonl','r4a_generation_outputs.jsonl','r4a_parity_diagnostics.json'):
                (out/name).write_bytes((a.inputs/name).read_bytes())
            gold=read(a.inputs/'r4a_gold.json') if (a.inputs/'r4a_gold.json').exists() else None
            worker=dict(worker,evidence_directory=str(a.inputs/'es'))
            decision=finish_analysis(out,Path(plan['prepared']),rows,tokens,worker,fixture,gold)
            report.update(analysis_only=True,input_result_sha256=sha(a.inputs/'r4a_results.json'),audit=old['audit'])
        else:
            if not fixture:gpu_preflight()
            runner=legacy_runner().process_runner()
            if a.phase=='prepare':
                base=a.base_prepared.resolve();plan=dict(base_prepared=str(base),fixture=False,device=a.device,scope=a.scope,
                    policy=a.generation_policy,sources=report['sources'])
                prepare_artifacts(plan,out/'native',runner);report['prepared_directory']=str(out/'native')
            else:
                if fixture:
                    from semcache.edgelora.r3a_tiny import prepare_tiny
                    base=prepare_tiny(out/'base_prepare');prepared=out/'prepared'
                    plan=dict(base_prepared=str(base),fixture=True,device='cpu',scope='pilot',policy=a.generation_policy,sources=report['sources'])
                    prepare_artifacts(plan,prepared,runner)
                else:prepared=a.prepared.resolve()
                m,suite,pm,ref,am=load_prepared(prepared)
                if not fixture:require(m['TEST_FIXTURE_ONLY'] is False,'fixture cannot enter GPU experiment')
                plan=dict(prepared=str(prepared),base_prepared=m['base_prepared'],fixture=fixture,device='cpu' if fixture else a.device,
                    sources=report['sources'],scope=suite['scope'])
                es=out/'es';es.mkdir();write(es/'es_plan.json',dict(plan,output=str(es)))
                runner([sys.executable,'-u',str(Path(__file__).resolve()),'--worker','es','--plan',str(es/'es_plan.json'),
                    '--test-fixture-only' if fixture else '--enable-gpu-experiment'],es,timeout=7500.,env=offline_env())
                result=read(es/'es_result.json');require(result['execution_status']=='COMPLETE','ES generation failed')
                report['audit']=audit_run(out,plan,result);report['model_result']=result
                rows=jsonl(es/'r4a_requests.jsonl');tokens=jsonl(es/'r4a_tokens.jsonl')
                for name in ('r4a_requests.jsonl','r4a_tokens.jsonl','r4a_generation_outputs.jsonl','r4a_parity_diagnostics.json'):(out/name).write_bytes((es/name).read_bytes())
                gold=None
                if a.gold_jsonl:
                    evidence.file(a.gold_jsonl);gold=jsonl(a.gold_jsonl);write(out/'r4a_gold.json',gold)
                decision=finish_analysis(out,prepared,rows,tokens,result,fixture,gold)
                report.update(gpu_findings_status='TEST_FIXTURE_ONLY_NO_GPU' if fixture else 'COMPLETE',gpu_inference_performed=not fixture,
                    actual_network_socket_used=True,budget=planned_counts(suite))
        evidence.unchanged();report['execution_status']='COMPLETE'
    except BaseException as exc:
        report.update(execution_status='FAILED',reason=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc());traceback.print_exc()
        if (out/'es/es_result.json').exists():report['failed_worker']=read(out/'es/es_result.json')
        if 'failed_worker' in report:report['reason']+='; ES: '+report['failed_worker'].get('reason','inspect worker trace')
        decision=dict(execution_status='FAILED',reason=report['reason'],**negative_gates,**flags(fixture));export(out,[],decision)
    finally:
        for name in ('r4a_requests.jsonl','r4a_tokens.jsonl','r4a_generation_outputs.jsonl'):
            if (out/'es'/name).exists():(out/name).write_bytes((out/'es'/name).read_bytes())
            elif not (out/name).exists():(out/name).touch()
        for name in ('r4a_quality.json','r4a_sacrebleu_signatures.json','r4a_parity_diagnostics.json','r4a_audit.json'):
            if not (out/name).exists():write(out/name,dict(execution_status='NOT_RUN',reason=report.get('reason')))
        if 'audit' in report:write(out/'r4a_audit.json',report['audit'])
        if (out/'es/r4a_parity_diagnostics.json').exists():(out/'r4a_parity_diagnostics.json').write_bytes((out/'es/r4a_parity_diagnostics.json').read_bytes())
        if not (out/'r4a_decision.json').exists():export(out,[],decision)
        if (out/'es/es_wire.jsonl').exists():
            report['actual_network_socket_used']=any(r.get('complete') for r in jsonl(out/'es/es_wire.jsonl'))
        gpu_marks=list(out.rglob('gpu_execution.json'))
        attempted=any(read(p).get('gpu_inference_attempted') is True for p in out.rglob('gpu_attempt.json'))
        report['gpu_inference_attempted']=attempted
        if any(read(p).get('gpu_inference_performed') is True for p in gpu_marks):
            report['gpu_inference_performed']=True
            if report['execution_status']=='FAILED':report['gpu_findings_status']='INCOMPLETE_FAILED'
            elif a.phase=='prepare':report['gpu_findings_status']='NATIVE_REFERENCES_COMPLETE'
        elif attempted and report['execution_status']=='FAILED':
            report.update(gpu_inference_performed=None,gpu_findings_status='FAILED_NO_COMPLETED_FORWARD')
        report.update(decision=decision,finished_at=now(),input_hashes=list(evidence.files.values()))
        report.update(native_equivalent=decision.get('native_equivalent',False),
            correctness_qualified_speedup=decision.get('correctness_qualified_speedup',False))
        report['output_hashes']={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and p.name not in ('r4a_results.json','r4a_summary.txt')}
        write(out/'r4a_results.json',report)
        (out/'r4a_summary.txt').write_text('\n'.join([report['classification'],'Execution: '+report['execution_status'],'GPU findings: '+report['gpu_findings_status'],
            'Reason: '+report.get('reason','none'),'Native equivalent: '+str(decision.get('native_equivalent')),
            'All-token generation; controlled exact prompt, untrained adapter, localhost TCP; no natural hits or remote link claim.'])+'\n')
    print(json.dumps(dict(report=str(out/'r4a_results.json'),execution_status=report['execution_status'],gpu_findings_status=report['gpu_findings_status'])))
    return 0 if report['execution_status']=='COMPLETE' else 1


if __name__=='__main__':
    def interrupted(signum,frame):raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,interrupted);sys.exit(main())
