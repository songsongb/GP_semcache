"""SNIPS-only B2 planning, strict artifact verification and paired task metrics."""
import argparse
import csv
import importlib.metadata
import json
import math
from pathlib import Path
import platform
from statistics import mean, median
import subprocess
from types import SimpleNamespace

from . import c6b_snips as b1
from .c6_quality import MODES, PROFILE, CONTRACT, verify_profile
from .c6_runtime import RAW_ENTRY_BYTES
from semcache.models.task_adapters import USERS, validate_task_config

SCOPE = dict(schema='c6b2_snips_v1', task_capability_stage='C6-B2', dataset='snips',
    trained_adapter=True, adapter_source='task_finetuned_snips', paper_bleu_claimed=False,
    primary_task_metric=b1.SCOPE['task_metric'],
    physical_safety_contract='PHYSICAL_EXACT_W3_WITHIN_SEMANTIC_CLUSTER')
PAIRS = (('FULL_RECOMPUTE','RAW_SEMCACHE','semantic_reuse'),
         ('RAW_SEMCACHE','STORAGE_KV_COMP','storage'),
         ('RAW_SEMCACHE','TRANSPORT_QKV_COMP','transport'),
         ('TRANSPORT_QKV_COMP','FULL_PIPELINE','storage_after_transport'),
         ('RAW_SEMCACHE','FULL_PIPELINE','combined_compression'))
BRANCHING = ('one source prefill and one cache insertion/lookup per reuse episode/mode; '
             'seven independent target full-sequence forwards share the same decoded hit; '
             'greedy prompt prefill shares that hit, incremental decode uses fresh rows and its own past KV')
COUNTERS = ('source_forward_count','target_candidate_forward_count','target_greedy_forward_count',
            'storage_decode_count','transport_encode_calls','transport_decode_calls',
            'runtime_transport_cdf_fit_count','runtime_storage_cdf_fit_count')


def check_modes():
    if MODES != dict(FULL_RECOMPUTE=(False,False,False),RAW_SEMCACHE=(True,False,False),
        STORAGE_KV_COMP=(True,False,True),TRANSPORT_QKV_COMP=(True,True,False),FULL_PIPELINE=(True,True,True)):
        raise ValueError('C6 five-mode factor map changed')


def read_json(path):
    return json.loads(Path(path).read_text())


def close_tree(actual, expected):
    """CSV float roundtrips may differ in the last bit; structures must match."""
    if isinstance(expected,dict):
        return isinstance(actual,dict) and actual.keys()==expected.keys() and all(close_tree(actual[k],v) for k,v in expected.items())
    if isinstance(expected,list):
        return isinstance(actual,list) and len(actual)==len(expected) and all(close_tree(a,b) for a,b in zip(actual,expected))
    if type(expected) is float:
        return type(actual) in (int,float) and math.isfinite(actual) and math.isclose(actual,expected,rel_tol=1e-9,abs_tol=1e-10)
    return actual == expected


def verify_capability(args, training, plan):
    root=args.capability_root
    cap=read_json(root/'capability_manifest.json')
    summary=read_json(root/'capability_summary.json')
    if cap.get('status')!='COMPLETE' or cap.get('trained_adapter') is not True or cap.get('adapter_source')!='task_finetuned_snips':
        raise ValueError('Completed task-trained B1 capability provenance required')
    for key,value in b1.SCOPE.items():
        if cap.get(key)!=value or summary.get(key)!=value:
            raise ValueError(f'B1 capability scope mismatch: {key}')
    for key in ('workload_sha256','selection_sha256','train_selection_sha256','adapter_hashes',
                'holdout_ids','holdout_ids_sha256','train_row_ids','train_ids_sha256','counts',
                'model_revision','tokenizer_revision','resolved_model_revision','resolved_tokenizer_revision',
                'base_model','label_serialization','adapter_config'):
        if cap.get(key)!=training.get(key) or key not in cap:
            raise ValueError(f'B1 capability/training provenance mismatch: {key}')
    if cap.get('training_manifest_sha256')!=b1.file_hash(args.adapter_root/'training_manifest.json'):
        raise ValueError('B1 capability references different training manifest')
    for filename in ('train_selection.json','capability_per_case.csv','capability_summary.json'):
        if cap.get('output_hashes',{}).get(filename)!=b1.file_hash(root/filename):
            raise ValueError(f'B1 capability output hash mismatch: {filename}')
    if b1.file_hash(root/'train_selection.json')!=training['train_selection_sha256']:
        raise ValueError('B1 capability training plan mismatch')
    with (root/'capability_per_case.csv').open(newline='') as stream:
        reported=list(csv.DictReader(stream))
    if len(reported)!=32:
        raise ValueError('B1 capability must cover the full frozen 32-target cohort')
    recomputed=[]
    for reported_case,target in zip(reported,plan['targets']):
        for key in ('episode_id','source_id','user','reference_text','reference_sha256'):
            if reported_case.get(key)!=target[key]:
                raise ValueError('B1 capability case identity/order mismatch')
        if reported_case.get('mode')!='FULL_RECOMPUTE':
            raise ValueError('B1 capability must be FULL_RECOMPUTE')
        scores=json.loads(reported_case['candidate_scores'])
        result=b1.classify(scores,target['reference_text'])
        for key in ('predicted_label','correct'):
            if reported_case.get(key)!=str(result[key]):
                raise ValueError('B1 capability prediction disagrees with candidate scores')
        for key in ('correct_label_mean_log_probability','best_wrong_label_mean_log_probability','classification_margin'):
            if not close_tree(float(reported_case[key]),result[key]):
                raise ValueError('B1 capability score/margin mismatch')
        greedy=b1.normalized_label(reported_case['generated_text'])==b1.normalized_label(target['reference_text'])
        if reported_case.get('greedy_exact_label_match')!=str(greedy):
            raise ValueError('B1 greedy diagnostic mismatch')
        recomputed.append(dict(**target,**result,greedy_exact_label_match=greedy))
    expected=b1.aggregate(recomputed,summary.get('min_accuracy'))
    expected.update(trained_adapter=True,adapter_source='task_finetuned_snips')
    if not close_tree(summary,expected):
        raise ValueError('B1 capability summary disagrees with aligned per-case scores')
    return dict(manifest_sha256=b1.file_hash(root/'capability_manifest.json'),
        summary_sha256=b1.file_hash(root/'capability_summary.json'),
        per_case_sha256=b1.file_hash(root/'capability_per_case.csv'), measured_summary=summary,
        use='provenance only; accuracy is never an eligibility or subset criterion')


def prepare(args):
    check_modes()
    selection=read_json(args.selection)
    episodes=[e for e in selection.get('episodes',[]) if e.get('dataset')=='snips']
    if len(episodes)!=32:
        raise ValueError('Frozen selection must contain exactly 32 SNIPS episodes')
    if not 1<=args.max_episodes<=32:
        raise ValueError('max-episodes must be between 1 and 32')
    rows=[json.loads(line) for line in args.snips.read_text().splitlines() if line.strip()]
    saved=read_json(args.adapter_root/'train_selection.json')
    # Reuse the full B1 identity/span/reference/holdout validation, never select anew.
    plan=b1.build_plan(rows,selection,saved['per_intent_per_user'],saved['seed'])
    from .c6b_runtime import verify_training_artifacts
    training,plan=verify_training_artifacts(args,rows,plan)
    for user in USERS:
        config=read_json(args.adapter_root/user/'adapter_config.json')
        try:
            validate_task_config(SimpleNamespace(**config))
        except (AttributeError,TypeError) as exc:
            raise ValueError(f'Incomplete local adapter config: {user}') from exc
        if not any((args.adapter_root/user/name).is_file() and (args.adapter_root/user/name).stat().st_size
                   for name in ('adapter_model.safetensors','adapter_model.bin')):
            raise ValueError(f'Missing nonempty local adapter weights: {user}')
    capability=verify_capability(args,training,plan)
    verify_profile(args.profile_path)  # bytes/hash only, no codec import or fitting
    return rows,episodes[:args.max_episodes],dict(
        adapter_hashes=training['adapter_hashes'],training_manifest_sha256=b1.file_hash(args.adapter_root/'training_manifest.json'),
        train_selection_sha256=training['train_selection_sha256'],holdout_ids=plan['holdout_ids'],
        holdout_ids_sha256=plan['holdout_ids_sha256'],train_ids_sha256=plan['train_ids_sha256'],
        train_row_ids=plan['train_row_ids'],training_counts=plan['counts'],
        label_serialization=b1.LABEL_SERIALIZATION,capability=capability,
        workload_sha256=b1.file_hash(args.snips),selection_sha256=b1.file_hash(args.selection),
        frozen_snips_episode_ids=[e['episode_id'] for e in episodes],frozen_snips_plan_sha256=b1.digest(episodes),
        executed_episode_ids=[e['episode_id'] for e in episodes[:args.max_episodes]],
        executed_plan_sha256=b1.digest(episodes[:args.max_episodes]),subset=args.max_episodes<32,
        subset_rule='first N frozen SNIPS episodes, original order, quality-blind',
        storage_profile_sha256=b1.file_hash(args.profile_path),shared_profile_bytes=args.profile_path.stat().st_size)


def paired_rows(cases):
    grouped={}
    for case in cases:
        group=grouped.setdefault(case['episode_id'],{})
        if case['mode'] in group:
            raise ValueError('Duplicate episode/mode result')
        group[case['mode']]=case
    pairs=[]
    for eid,group in grouped.items():
        if set(group)!=set(MODES):
            raise ValueError('Every episode requires all five modes')
        for baseline,comparison,effect in PAIRS:
            a,b=group[baseline],group[comparison]
            if a['true_label']!=b['true_label'] or a['user']!=b['user']:
                raise ValueError('Incomparable task case identities')
            pairs.append(dict(episode_id=eid,baseline_mode=baseline,comparison_mode=comparison,effect=effect,
                baseline_correct=a['correct'],comparison_correct=b['correct'],baseline_prediction=a['predicted_label'],
                comparison_prediction=b['predicted_label'],prediction_changed=a['predicted_label']!=b['predicted_label'],
                correct_to_wrong=a['correct'] and not b['correct'],wrong_to_correct=not a['correct'] and b['correct'],
                delta_correct_label_log_probability=b['correct_label_mean_log_probability']-a['correct_label_mean_log_probability'],
                delta_best_wrong_log_probability=b['best_wrong_label_mean_log_probability']-a['best_wrong_label_mean_log_probability'],
                delta_classification_margin=b['classification_margin']-a['classification_margin']))
    return pairs


def summarize(cases,pairs):
    modes=[]
    for mode in MODES:
        subset=[c for c in cases if c['mode']==mode]
        summary=b1.aggregate(subset)
        summary.update(SCOPE)
        for metric in ('kv_compression_ratio','resident_qkv_compression_ratio','byte_reduction_percentage'):
            values=[c[metric] for c in subset if c[metric] is not None]
            summary[metric]=mean(values) if values else None
        summary.update(mode=mode,correct_count=sum(c['correct'] for c in subset),case_count=len(subset),
            subset=len(subset)!=32,frozen_cohort_size=32,semantic_reuse_enabled=MODES[mode][0],
            compression_enabled=any(MODES[mode][1:]),
            **{k:mean(c[k] for c in subset) for k in ('resident_payload_bytes','resident_q_bytes','resident_kv_frame_bytes','resident_local_metadata_bytes')})
        modes.append(summary)
    effects=[]
    for baseline,comparison,effect in PAIRS:
        rows=[p for p in pairs if p['effect']==effect]
        a=mean(p['baseline_correct'] for p in rows); b=mean(p['comparison_correct'] for p in rows)
        effects.append(dict(effect=effect,baseline_mode=baseline,comparison_mode=comparison,count=len(rows),
            baseline_accuracy=a,comparison_accuracy=b,delta_accuracy_percentage_points=100*(b-a),
            prediction_flips=sum(p['prediction_changed'] for p in rows),correct_to_wrong=sum(p['correct_to_wrong'] for p in rows),
            wrong_to_correct=sum(p['wrong_to_correct'] for p in rows),
            mean_delta_classification_margin=mean(p['delta_classification_margin'] for p in rows),
            median_delta_classification_margin=median(p['delta_classification_margin'] for p in rows),
            mean_delta_correct_label_log_probability=mean(p['delta_correct_label_log_probability'] for p in rows)))
    return dict(**SCOPE,modes=modes,causal_comparisons=effects,
                interpretation='wrong-to-correct changes are observed perturbations, not evidence of general compression benefit')


def byte_accounting(mode, measured):
    reuse,_,storage=MODES[mode]
    if not reuse:
        return dict(resident_payload_bytes=0,resident_q_bytes=0,resident_kv_frame_bytes=0,
            resident_local_metadata_bytes=0,raw_qkv_bytes=0,raw_kv_bytes=0,
            kv_compression_ratio=None,resident_qkv_compression_ratio=None,byte_reduction_percentage=None)
    raw=RAW_ENTRY_BYTES; q=measured['resident_q_bytes']; payload=measured['resident_payload_bytes']
    frame=measured['resident_kv_frame_bytes'] if storage else raw-q
    metadata=measured.get('resident_local_metadata_bytes',0)
    if q!=raw//3 or payload!=q+frame or frame<=0 or not 0<=metadata<=frame or (not storage and payload!=raw):
        raise ValueError('Inconsistent resident FP16-Q / KV byte accounting')
    return dict(resident_payload_bytes=payload,resident_q_bytes=q,resident_kv_frame_bytes=frame,
        resident_local_metadata_bytes=metadata,raw_qkv_bytes=raw,raw_kv_bytes=raw-q,
        kv_compression_ratio=(raw-q)/frame,resident_qkv_compression_ratio=raw/payload,
        byte_reduction_percentage=100*(1-payload/raw))


def evaluate_case(backend,episode,mode):
    """Model-free orchestration boundary; expensive construction is outside candidates."""
    before=dict(backend.counts)
    context=backend.prepare(episode,mode)
    expected=b1.digest(episode) if MODES[mode][0] else None
    scores={}
    for label in b1.LABELS:
        if backend.event_identity(context,episode)!=expected:
            raise ValueError('Logical event differs across candidates/modes')
        scores[label]=backend.score(context,episode,label)
        if backend.event_identity(context,episode)!=expected:
            raise ValueError('Candidate forward changed frozen event')
    generated,text=backend.greedy(context,episode)
    if backend.event_identity(context,episode)!=expected:
        raise ValueError('Greedy forward changed frozen event')
    counts={k:backend.counts[k]-before[k] for k in COUNTERS}
    reuse,transport,storage=MODES[mode]
    if (counts['source_forward_count']!=int(reuse) or counts['target_candidate_forward_count']!=7
            or counts['storage_decode_count']!=int(storage) or counts['runtime_storage_cdf_fit_count']!=0
            or counts['transport_encode_calls']!=counts['transport_decode_calls']
            or counts['transport_encode_calls']!=counts['runtime_transport_cdf_fit_count']
            or counts['target_greedy_forward_count']!=len(generated)
            or (transport and counts['transport_encode_calls'] != 3*32*(counts['source_forward_count']
                +counts['target_candidate_forward_count']+counts['target_greedy_forward_count']))
            or (transport and not counts['transport_encode_calls'])
            or (not transport and counts['transport_encode_calls'])):
        raise ValueError('C6-B2 execution counts violate mode/one-build/one-decode contract')
    result=b1.classify(scores,episode['reference_text'])
    wrong=max((l for l in b1.LABELS if l!=episode['reference_text']),key=lambda l:scores[l])
    return dict(**SCOPE,episode_id=episode['episode_id'],mode=mode,user=episode['target_user'],
        source_id=episode['source_id'],target_id=episode['target_id'],source_user=episode['source_user'],
        source_start=episode['source_start'],target_start=episode['target_start'],token_ids=episode['token_ids'],
        cluster=episode['cluster'],cache_key=episode['cache_key'],selected_hit_count=int(reuse),logical_event_hash=expected,
        reference_text=episode['reference_text'],reference_sha256=episode['reference_sha256'],true_label=episode['reference_text'],
        **result,best_wrong_label=wrong,generated_token_ids=generated,generated_text=text,generated_length=len(generated),
        greedy_exact_label_match=b1.normalized_label(text)==b1.normalized_label(episode['reference_text']),
        transport_scope='LoRA QKV delta' if transport else 'native',storage_scope='TOTAL KV compressed; Q FP16' if storage else 'raw TOTAL QKV' if reuse else 'none',
        **byte_accounting(mode,context.accounting),**counts)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snips',type=Path,default=Path('results/workloads/m9b_snips_semantic.jsonl'))
    for name in ('selection','adapter-root','capability-root'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--profile-path',type=Path,default=Path(PROFILE))
    parser.add_argument('--storage-src',type=Path)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--max-episodes',type=int,default=32)
    parser.add_argument('--max-new-tokens',type=int,default=12)
    parser.add_argument('--max-sequence-length',type=int,default=256)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args(argv)
    if args.max_new_tokens<1 or not 2<=args.max_sequence_length<=2048:
        raise ValueError('Invalid generation/context length bound')
    rows,episodes,verified=prepare(args)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    if any(args.output_dir.iterdir()):
        raise ValueError('Output directory must be empty')
    versions={}
    for package in ('torch','transformers','peft','safetensors','numpy','lmcache','torchac_cuda'):
        try: versions[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: versions[package]=None
    def git(*cmd): return subprocess.check_output(['git',*cmd],text=True).strip()
    manifest=dict(**SCOPE,**verified,status='DRY_RUN' if args.dry_run else 'STARTING',
        modes=MODES,label_order=list(b1.LABELS),candidate_branching_policy=BRANCHING,
        model=b1.MODEL_ID,tokenizer=b1.MODEL_ID,requested_revision=b1.MODEL_REVISION,resolved_revision=None,
        dtype='float16',device=args.device,actual_gpu=None,
        arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        generation=dict(do_sample=False,num_beams=1,max_new_tokens=args.max_new_tokens,respect_eos=True,seed=args.seed),
        software=dict(python=platform.python_version(),**versions),
        git=dict(branch=git('branch','--show-current'),commit=git('rev-parse','HEAD'),status=git('status','--porcelain')),
        counters={k:0 for k in COUNTERS},output_hashes={},codec_source_hashes={},
        capacity_policy='one cold raw-logical-size entry; capacity_charge=None; shared overhead=0',
        storage_policy=CONTRACT['storage_policy'],storage_transform=CONTRACT['storage_transform'],
        storage_profile_verified=True,transport_num_bins=32,transport_payload='LoRA Q/K/V deltas only',
        greedy_diagnostic_scope='secondary raw-logit argmax decoding, as in C6-A',
        resident_q='TOTAL Q FP16 uncompressed; Q compression deferred to C7')
    b1.write_json(args.output_dir/'manifest.json',manifest)
    if args.dry_run:
        print(json.dumps(dict(status='DRY_RUN',frozen_snips_episodes=32,executed_episodes=len(episodes),
            episode_ids=verified['executed_episode_ids'],subset=verified['subset'],modes=list(MODES),
            artifacts_verified=True,model_codec_loaded=False,candidate_branching_policy=BRANCHING),indent=2))
        return
    from .c6b2_runtime import run
    try:
        run(args,rows,episodes,manifest)
    except Exception as exc:
        manifest.update(status='FAILED',error=f'{type(exc).__name__}: {exc}')
        b1.write_json(args.output_dir/'manifest.json',manifest)
        raise
