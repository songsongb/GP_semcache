"""CPU-only readiness and sealed-plan inspection. Never downloads or submits jobs."""
from collections import Counter
from pathlib import Path
import importlib.metadata
import importlib.util
import subprocess
import shlex
import re
import unittest
from .communication import digest,require
from .r3a_artifacts import ROOT,read,sha
from .r5_i5_data import DATASETS,VERSIONS,SOURCES,load_dataset
from .r5_i5_quality import score,aggregate,validate_scorers
from .r5_i5_workload import prepare_prompt,BUDGETS,SCALE_SEMANTICS
from .r5_i5_evidence import verify
from .r5_perf_schedule import TINYBERT_REVISION


def attempt(fn):
    try:return dict(status='PASS',details=fn())
    except Exception as exc:return dict(status='NOT_RUN' if isinstance(exc,(ImportError,ModuleNotFoundError)) else 'FAIL',reason=str(exc))


def installed_opt_check():
    # Execute the existing frozen CPU OPT contract without its optional skip decorator.
    import torch,transformers,numpy
    import importlib.util
    spec=importlib.util.spec_from_file_location('i5p_opt_contract',ROOT/'tests/test_r4c_opt_contract.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    case=module.OPTContract('test_eager_attention_return_flag_and_raw_kv_layout')
    case.test_eager_attention_return_flag_and_raw_kv_layout()
    return {n:importlib.metadata.version(n) for n in ('torch','transformers','numpy')}


def scorer_smokes():
    results={}
    for name in DATASETS:
        def check(name=name):
            if name=='multi_news':validate_scorers([name])
            refs={'hotel-pricerange':['cheap']} if name=='multiwoz' else ['PlayMusic'] if name=='snips' else ['blue sky']
            original=dict(test='KNOWN_REFERENCE_SCORER_TEST_ONLY')
            sample=dict(dataset=name,version=VERSIONS[name],id='SCORER_TEST_ONLY',group_id='SCORER_TEST_ONLY',
                context='',question='',history='',references=refs,original=original,source_record_sha256=digest(original),
                TEST_FIXTURE_ONLY=True,ontology=['hotel-pricerange'])
            text='{"hotel-pricerange":"cheap"}' if name=='multiwoz' else refs[0]
            r=score(sample,text,'I5P_SCORER_TEST_ONLY');require(abs(r['score']-1)<1e-6,'known exact-reference score differs')
            if name=='snips':
                rows=[]
                from .r5_i5_data import LABELS
                for label in LABELS:rows.append(score(dict(sample,references=[label]),label,'I5P_SCORER_TEST_ONLY'))
                require(aggregate(rows)['macro_f1']==1,'intent macro-F1 differs')
            if name=='multiwoz':require(r['slot_tp']==1 and r['slot_fp']==r['slot_fn']==0,'slot counts differ')
            if 'exact_match' in r:require(r['exact_match']==1,'EM differs')
            return dict(metric=r['metric'],score=r['score'],scope='KNOWN_REFERENCE_SCORER_TEST_ONLY')
        results[name]=attempt(check)
        if results[name]['status']!='PASS':results[name]['status']='NOT_RUN'
    return results


def readiness(registry, *, base_prepared=None,multi_prepared=None,count=10,budgets=None,input_limit=512,allow_truncation=False):
    results={};runtime={};tok=None;limit=None
    def model_check():
        nonlocal tok,limit
        require(base_prepared is not None,'set --base-prepared to verified real OPT preparation')
        from .r3a_artifacts import validate_prepared
        from .r4c_prepare import opt_tokenizer
        pm,_,_=validate_prepared(base_prepared);require(not pm.get('TEST_FIXTURE_ONLY'), 'real preflight rejects fixture model')
        adapter=opt_tokenizer(pm);tok=adapter.tokenizer;limit=adapter.supported_context_length
        from transformers import AutoModelForCausalLM
        from huggingface_hub import snapshot_download
        path=Path(snapshot_download(pm['identity']['model'],revision=pm['identity']['base_revision'],local_files_only=True))
        require(any(path.glob('*.safetensors')) or any(path.glob('pytorch_model*.bin')),'local base weights missing')
        return dict(identity=pm['identity'],context_limit=limit,local_snapshot=str(path))
    runtime['model_tokenizer']=attempt(model_check)
    def adapter_check():
        require(multi_prepared is not None,'set --multi-prepared to real expanded A/B/C artifacts')
        from .r5_multi_artifacts import validate
        m,pm,a,r=validate(multi_prepared)
        require(not m['TEST_FIXTURE_ONLY'] and m['stage']=='expanded' and len(r['users'])==30,'real expanded adapters required')
        if base_prepared:require(Path(m['base_prepared']).resolve()==Path(base_prepared).resolve(),'adapter/base path mismatch')
        return dict(fingerprints={k:v['matrix_sha256'] for k,v in a.items()},registry_sha256=r['sha256'])
    runtime['adapters']=attempt(adapter_check)
    def semantic_check():
        from .r5_runtime import encoder_for
        e=encoder_for(dict(fixture=False,semantic_revision=TINYBERT_REVISION),[])
        return e.metadata
    runtime['tinybert']=attempt(semantic_check)
    runtime['installed_opt']=attempt(installed_opt_check)
    runtime['repair']='Activate the pinned semcache environment; install repository-compatible torch, numpy, transformers and peft matching base preparation versions. Install rouge==1.0.1 outside jobs. Prepare pinned model/tokenizer/TinyBERT snapshots outside jobs; no downloads here.'
    smokes=scorer_smokes()
    for name in DATASETS:
        spec=registry.get(name);path=spec.get('prepared_path') if isinstance(spec,dict) else spec
        entry=dict(status='DATASET_NOT_READY',dataset=name,version=VERSIONS[name],source=SOURCES[name],prepared_path=str(path) if path else None,
            scorer=smokes[name],eligibility='NOT_RUN',generation_budget=(budgets or {}).get(name,BUDGETS[name]))
        try:
            require(path and Path(path).exists(),'explicit prepared snapshot missing')
            rows,m=load_dataset(path,fixture=False)
            require(m['dataset']==name and m['source']==SOURCES[name] and m['version']==VERSIONS[name] and re.fullmatch('[0-9a-f]{40,64}',m.get('revision','')) and m.get('split'),'source/version/revision/split missing')
            source=Path(m['source_path']);require(source.exists(),'original source path missing')
            from .r4c_dataset import file_hashes
            require(file_hashes(source)==m['source_files'],'original source inventory differs')
            if m.get('ontology_path'):require(sha(m['ontology_path'])==m['ontology_sha256'],'ontology differs')
            entry.update(normalized_samples=len(rows),provenance=m,original_ids_unique=len({r['id'] for r in rows})==len(rows),references_complete=True)
            require(tok is not None,'model/tokenizer unavailable; context eligibility NOT_RUN')
            eligible=[];excluded=[]
            for row in rows:
                try:eligible.append(prepare_prompt(row,tok,limit,entry['generation_budget'],input_limit=input_limit,allow_truncation=allow_truncation))
                except ValueError as exc:excluded.append(dict(id=row['id'],reason=str(exc)))
            entry.update(eligibility=dict(eligible=len(eligible),requested=count,excluded=excluded),truncation_count=sum(r['truncated'] for r in eligible),
                         truncation_rate=sum(r['truncated'] for r in eligible)/len(eligible) if eligible else 0)
            require(len(eligible)>=count,'insufficient eligible unique samples')
            require(smokes[name]['status']=='PASS','scorer unavailable')
            entry['status']='DATASET_READY'
        except Exception as exc:entry['reason']=str(exc)
        entry['preparation_instruction']=f'Acquire {SOURCES[name]} version {VERSIONS[name]} outside GPU jobs; preserve immutable upstream commit/archive SHA256 and split. Run: python scripts/118_prepare_r5_i5.py dataset --dataset {name} --source /ABS/NATIVE_SOURCE --revision PIN --split SPLIT '+('--ontology /ABS/MultiWOZ_2.2/schema.json ' if name=='multiwoz' else '')+f'--output-dir results/r5/i5/data/{name}_FRESH; register its absolute directory. See docs/R5_I5_INTEGRATED_EVALUATION.md.'
        results[name]=entry
    selected=list(registry)
    ready=bool(selected) and all(results.get(n,{}).get('status')=='DATASET_READY' for n in selected) and all(runtime[k]['status']=='PASS' for k in ('model_tokenizer','adapters','tinybert','installed_opt'))
    return dict(status='READY_FOR_MANUAL_GPU_SMOKE' if ready else 'NOT_READY',GPU_NOT_RUN=True,datasets=results,runtime=runtime,
                recommended_first_dataset='coqa',coqa_smoke_blocker=None if results['coqa']['status']=='DATASET_READY' else 'Verified CoQA dev snapshot/readiness unavailable; do not silently switch datasets',
                separate_gates=['reference_task_quality','generated_token_text_equality','frozen_numerical_tolerance'],historical_numerical_status='ROOT_CAUSE_UNRESOLVED')


def inspect_plan(path):
    path=Path(path).resolve();verify(path);m=read(path/'i5_workload_manifest.json');copy=dict(m);identity=copy.pop('experiment_id')
    require(digest(copy)==identity and m['blocks'],'experiment seal/identity invalid')
    blocks=m['blocks']
    for b in blocks:
        value=dict(b);bid=value.pop('block_id');require(digest(value)==bid,'block identity invalid')
        require(0<b['scored_unique_samples']<=20 and len(b['tasks'])<=40,'unsupported block request limit')
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    env=['I5_EXPERIMENT','I5_MULTI_PREPARED','I5_RUN_ROOT','I5_SOURCE_COMMIT','R5_I5_ENABLE_GPU=1','R5_ENABLE_GPU=1']
    index_range=f'0-{len(blocks)-1}%1'
    return dict(status='CPU_VERIFIED_PLAN_STRUCTURE',TEST_FIXTURE_ONLY=m['TEST_FIXTURE_ONLY'],scale_semantics=m.get('scale_semantics','LEGACY_POSITIONAL_PLAN_REQUIRES_REPREPARATION'),
        total_blocks=len(blocks),blocks_per_dataset=dict(Counter(b['dataset'] for b in blocks)),blocks_per_track=dict(Counter(b['track'] for b in blocks)),
        blocks_per_repetition=dict(Counter(str(b['repetition']) for b in blocks)),minimum_block_index=0,maximum_block_index=len(blocks)-1,slurm_array=index_range,
        required_source_commit=head,tracked_worktree_dirty=bool(subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True).strip()),source_commit_requirement='Commit reviewed changes first; printed HEAD must be clean and include execution sources',required_environment=env,
        scale_summary=m.get('scale_summary'),cache_lifetime_assessment=m.get('cache_lifetime_assessment'),
        blocks=[dict(index=i,dataset=b['dataset'],repetition=b['repetition'],scored_targets=b['scored_unique_samples'],physical_requests=len(b['tasks']),
                     cache_resets=3,maximum_cache_lifetime_requests=len(b['tasks']),continuity_breaks=b.get('continuity_breaks',[])) for i,b in enumerate(blocks)],
        resource_totals=dict(maximum_forwards=3*sum(b['resources']['expected_maximum_forwards'] for b in blocks),
            duration_estimate_seconds=3*sum(b['resources']['duration_estimate_seconds'] for b in blocks),
            evidence_estimate_bytes=3*sum(b['resources']['estimated_evidence_bytes'] for b in blocks),
            evidence_budget_bytes=3*sum(b['resources']['evidence_budget_bytes'] for b in blocks),
            maximum_gpu_estimate_bytes=max(b['resources']['gpu_estimate_bytes'] for b in blocks)),
        unsupported=['continuous_1000_request_cache_session','cross_job_live_GPU_cache_restore'],
        operator_commands=dict(first_block='sbatch --export=ALL --array=0-0%1 jobs/r5_i5_plan.sh',
            remaining_blocks=f'sbatch --export=ALL --array=1-{len(blocks)-1}%1 jobs/r5_i5_plan.sh' if len(blocks)>1 else None,
            all_blocks=f'sbatch --export=ALL --array={index_range} jobs/r5_i5_plan.sh'),
        interpretation='Independent blocks reset each condition; summed target latency is not continuously serving throughput. Startup/model load and warmup are separate. GPU_NOT_RUN.')
