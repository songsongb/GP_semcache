"""C6-B1 offline SNIPS training plan and FULL_RECOMPUTE capability validation."""
import argparse
from collections import Counter
import importlib.metadata
import json
import math
from pathlib import Path
import platform
from statistics import mean
import subprocess

from semcache.experiments.dataset_adapters import SNIPS_INTENTS
from semcache.experiments.m9b_semantic_workload import MODEL_ID, MODEL_REVISION, file_hash, assignment_source
from semcache.simulation.multi_user import read_workload, logical_user_assignment, digest
from semcache.models.task_adapters import USERS, ADAPTER_CONFIG
from .c6_quality import write_json, write_csv, normalized_label

LABELS = tuple(SNIPS_INTENTS)
USER_MAP = dict(user_000='user_a', user_001='user_b')
SCOPE = dict(task_capability_stage='C6-B1', paper_bleu_claimed=False,
    semantic_reuse_enabled=False, compression_enabled=False,
    task_metric='7-way length-normalized conditional label likelihood accuracy')


def build_plan(rows, selection, per_intent_per_user=200, seed=42):
    if type(per_intent_per_user) is not int or per_intent_per_user < 1:
        raise ValueError('per-intent-per-user must be positive')
    by_id = {}
    for i, row in enumerate(rows):
        sid = row.get('source_id')
        if not isinstance(sid,str) or not sid or sid in by_id:
            raise ValueError(f'Missing/duplicated source_id: {sid}')
        if row.get('reference_text') not in LABELS:
            raise ValueError(f'Missing or noncanonical reference_text: {sid}')
        if (row.get('dataset') != 'snips' or row.get('model_id') != MODEL_ID
                or row.get('model_revision') != MODEL_REVISION
                or row.get('tokenizer_id') != f'{MODEL_ID}@{MODEL_REVISION}'
                or row.get('semantic_execution_provenance') != 'MEASURED'
                or row.get('semantic_assignment_source') != assignment_source('snips')):
            raise ValueError(f'Incompatible canonical workload provenance: {sid}')
        if not isinstance(row.get('query_text'), str) or not row['query_text'].strip():
            raise ValueError(f'Missing prompt: {sid}')
        if row.get('domain_or_intent') not in (None, row['reference_text']):
            raise ValueError(f'Intent/reference disagreement: {sid}')
        ids = row.get('token_ids')
        if not isinstance(ids,list) or not ids or any(type(t) is not int or t < 0 for t in ids):
            raise ValueError(f'Invalid token IDs: {sid}')
        by_id[sid] = (i, row)
    episodes = selection.get('episodes')
    if not isinstance(episodes,list):
        raise ValueError('Expected frozen C6 selection episodes')
    episodes = [e for e in episodes if e.get('dataset') == 'snips']
    if not episodes:
        raise ValueError('Selection contains no SNIPS episodes')
    holdout, seen, targets = set(), set(), []
    for e in episodes:
        if not e.get('episode_id') or e['episode_id'] in seen:
            raise ValueError('Missing/duplicated selected episode identity')
        seen.add(e['episode_id'])
        source_id, target_id = e.get('source_id'), e.get('target_id')
        if source_id not in by_id or target_id not in by_id:
            raise ValueError('Evaluation source/target ID missing from workload')
        si, source = by_id[source_id]; ti, target = by_id[target_id]
        if (si >= ti or source_id == target_id or e.get('source_index') != si or e.get('target_index') != ti
                or e.get('cluster') != source['cluster_id'] or e['cluster'] != target['cluster_id']
                or e.get('reference_text') != target['reference_text']
                or e.get('reference_sha256') != digest(target['reference_text'])
                or e.get('intent') not in (None,target['reference_text'])
                or source['query_text'] == target['query_text'] or source['token_ids'] == target['token_ids']
                or e.get('source_user') not in USERS or e.get('target_user') not in USERS
                or e.get('cross_user') != (e['source_user'] != e['target_user'])):
            raise ValueError('Frozen selection inconsistent with workload/assignment')
        block = e.get('token_ids')
        if not isinstance(block,list) or len(block) != 3 or set(block)&{0,1,2}:
            raise ValueError('Invalid selected exact w3 block')
        for role, row in (('source',source),('target',target)):
            start = e.get(role+'_start')
            if type(start) is not int or start < 3 or row['token_ids'][start:start+3] != block:
                raise ValueError('Selection span inconsistent with workload')
        if e.get('cache_key') != [e['cluster'], block] or e.get('selected_hit_count') != 1:
            raise ValueError('Selection cache event inconsistent')
        holdout.update((source_id,target_id))
        targets.append(dict(episode_id=e['episode_id'], source_id=target_id, user=e['target_user'],
                            reference_text=target['reference_text'], reference_sha256=e['reference_sha256']))
    eligible = [r for r in rows if r['source_id'] not in holdout]
    assigned = logical_user_assignment(eligible, users=2, seed=seed, dataset='snips')
    counts = {u:{label:0 for label in LABELS} for u in USERS}
    available = {u:Counter() for u in USERS}
    chosen = []
    for row, logical_user in zip(eligible, assigned):
        user, label = USER_MAP[logical_user], row['reference_text']
        available[user][label] += 1
        if counts[user][label] < per_intent_per_user:
            chosen.append(dict(source_id=row['source_id'], user=user, intent=label))
            counts[user][label] += 1
    ids = [r['source_id'] for r in chosen]
    if len(set(ids)) != len(ids) or holdout.intersection(ids):
        raise ValueError('Training leakage or duplicated training rows')
    return dict(schema='c6b_train_selection_v1', **SCOPE, seed=seed,
        assignment_rule='logical_user_assignment on holdout-excluded rows, two users, seeded_round_robin',
        selection_rule='first N per user/intent in stable workload order; no oversampling',
        per_intent_per_user=per_intent_per_user, train_rows=chosen, train_row_ids=ids,
        holdout_ids=sorted(holdout), holdout_ids_sha256=digest(sorted(holdout)),
        train_ids_sha256=digest(ids), targets=targets, counts=counts,
        available_counts={u:{l:available[u][l] for l in LABELS} for u in USERS},
        shortfalls={u:{l:max(0,per_intent_per_user-counts[u][l]) for l in LABELS} for u in USERS})


def encode_example(tokenizer, row, max_sequence_length):
    """Explicit concatenation: no whitespace insertion, text slicing or truncation."""
    prompt = list(tokenizer(row['query_text'], add_special_tokens=True, truncation=False)['input_ids'])
    if prompt != row['token_ids']:
        raise ValueError('Tokenizer prompt IDs differ from canonical workload')
    completion = label_ids(tokenizer,row['reference_text'])
    eos = tokenizer.eos_token_id
    if type(eos) is not int or eos in completion:
        raise ValueError('Expected ordinary label tokens and an explicit EOS ID')
    ids = prompt + completion + [eos]
    if not prompt or len(ids) > max_sequence_length:
        raise ValueError(f'Example exceeds max sequence length; no silent truncation: {row["source_id"]}')
    return dict(input_ids=ids, labels=[-100]*len(prompt)+completion+[eos], prompt_length=len(prompt))


def label_ids(tokenizer, label):
    if label not in LABELS:
        raise ValueError('Noncanonical candidate label')
    ids = list(tokenizer(label,add_special_tokens=False,truncation=False)['input_ids'])
    if not ids or any(t in tokenizer.all_special_ids for t in ids):
        raise ValueError('Candidate label must have nonempty nonspecial tokens')
    return ids


def mean_label_log_probability(values):
    if not values or any(not math.isfinite(v) for v in values):
        raise ValueError('Finite nonempty candidate token scores required')
    return mean(values)


def score_candidate_logits(logits, prompt_length, candidate):
    """Causal predictors for label tokens only; exclude prompt and EOS."""
    import torch
    if prompt_length < 1 or not candidate or logits.ndim != 2 or logits.shape[0] < prompt_length+len(candidate)-1:
        raise ValueError('Invalid candidate scoring alignment')
    scores = logits[prompt_length-1:prompt_length+len(candidate)-1].float().log_softmax(-1)
    values = scores.gather(1,torch.tensor(candidate,device=scores.device)[:,None]).squeeze(1).tolist()
    return mean_label_log_probability(values)


def classify(scores, correct):
    if set(scores) != set(LABELS) or correct not in LABELS or any(not math.isfinite(s) for s in scores.values()):
        raise ValueError('Seven finite canonical scores required')
    prediction = max(LABELS,key=lambda label:scores[label])  # stable first-label tie
    wrong = max(scores[l] for l in LABELS if l != correct)
    return dict(predicted_label=prediction, correct=prediction==correct,
        correct_label_mean_log_probability=scores[correct], best_wrong_label_mean_log_probability=wrong,
        classification_margin=scores[correct]-wrong, candidate_scores=scores)


def aggregate(cases, min_accuracy=None):
    if not cases:
        raise ValueError('No capability cases')
    def metric(subset):
        return dict(count=len(subset),accuracy=mean(r['correct'] for r in subset) if subset else None)
    confusion = {truth:{pred:0 for pred in LABELS} for truth in LABELS}
    for row in cases:
        confusion[row['reference_text']][row['predicted_label']] += 1
    result = dict(**SCOPE, **metric(cases), label_order=list(LABELS),
        per_user={u:metric([r for r in cases if r['user']==u]) for u in USERS},
        per_intent={l:metric([r for r in cases if r['reference_text']==l]) for l in LABELS},
        confusion_matrix=confusion, confusion_axes='rows=true, columns=predicted',
        greedy_exact_label_accuracy=mean(r['greedy_exact_label_match'] for r in cases),
        **{name:mean(r[name] for r in cases) for name in
           ('correct_label_mean_log_probability','best_wrong_label_mean_log_probability','classification_margin')},
        min_accuracy=min_accuracy, threshold_provenance='REPRODUCTION_CHOICE' if min_accuracy is not None else None,
        threshold_passed=None if min_accuracy is None else mean(r['correct'] for r in cases)>=min_accuracy)
    return result


def file_hashes(directory):
    return {str(p.relative_to(directory)):file_hash(p) for p in sorted(directory.rglob('*')) if p.is_file()}


def provenance(args, plan):
    def git(*values): return subprocess.check_output(['git',*values],text=True).strip()
    versions = {}
    for name in ('torch','transformers','peft'):
        try: versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: versions[name] = None
    return dict(**SCOPE, base_model=MODEL_ID, model_revision=MODEL_REVISION, tokenizer_revision=MODEL_REVISION,
        resolved_model_revision=None, resolved_tokenizer_revision=None, local_files_only=True,
        workload_sha256=file_hash(args.snips), selection_sha256=file_hash(args.selection),
        holdout_ids=plan['holdout_ids'], holdout_ids_sha256=plan['holdout_ids_sha256'],
        train_row_ids=plan['train_row_ids'], train_ids_sha256=plan['train_ids_sha256'],
        counts=plan['counts'], seed=args.seed, adapter_config=ADAPTER_CONFIG,
        trained_adapter=False, adapter_source='task_finetuned_snips',
        adapter_validation='pending until real execution',
        git=dict(branch=git('branch','--show-current'),commit=git('rev-parse','HEAD'),status=git('status','--porcelain')),
        software=dict(python=platform.python_version(),**versions), arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        tokenization_protocol='prompt with special tokens; label separately without specials; concatenate; no added separator',
        training_label_policy='prompt -100; label tokens plus one EOS supervised',
        candidate_score_policy='mean label-token log probability; EOS excluded; canonical-order ties')


def parser(training):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snips',type=Path,default=Path('results/workloads/m9b_snips_semantic.jsonl'))
    p.add_argument('--selection',type=Path,required=True)
    p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--per-intent-per-user',type=int,default=200)
    p.add_argument('--max-sequence-length',type=int,default=256)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--dry-run',action='store_true')
    if training:
        p.add_argument('--epochs',type=int,default=3)
        p.add_argument('--learning-rate',type=float,default=1e-4)
        p.add_argument('--gradient-accumulation',type=int,default=8)
        p.add_argument('--gradient-checkpointing',action=argparse.BooleanOptionalAction,default=True)
    else:
        p.add_argument('--adapter-root',type=Path,required=True)
        p.add_argument('--max-new-tokens',type=int,default=12)
        p.add_argument('--min-accuracy',type=float)
    return p


def main(training, argv=None):
    args=parser(training).parse_args(argv)
    if args.max_sequence_length < 2 or args.max_sequence_length > 2048:
        raise ValueError('max-sequence-length must be in [2,2048]')
    if training and (args.epochs < 1 or args.gradient_accumulation < 1 or not math.isfinite(args.learning_rate) or args.learning_rate <= 0):
        raise ValueError('Positive finite training arguments required')
    if not training and (args.max_new_tokens < 1 or args.min_accuracy is not None and not 0 <= args.min_accuracy <= 1):
        raise ValueError('Invalid generation bound or accuracy threshold')
    raw_rows=[json.loads(line) for line in args.snips.read_text().splitlines() if line.strip()]
    if any(not isinstance(r,dict) or not isinstance(r.get('source_id'),str) or not r['source_id'] for r in raw_rows):
        raise ValueError('Canonical workload must supply explicit nonempty string source_id identities')
    rows=read_workload(args.snips,'snips')
    selection=json.loads(args.selection.read_text())
    plan=build_plan(rows,selection,args.per_intent_per_user,args.seed)
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise ValueError('Use an empty output root; never overwrite trained adapters or results')
    args.output_root.mkdir(parents=True,exist_ok=True)
    write_json(args.output_root/'train_selection.json',plan)
    manifest=provenance(args,plan)
    manifest['train_selection_sha256']=file_hash(args.output_root/'train_selection.json')
    manifest['status']='DRY_RUN' if args.dry_run else 'STARTING'
    name='training_manifest.json' if training else 'capability_manifest.json'
    write_json(args.output_root/name,manifest)
    if args.dry_run:
        print(json.dumps(dict(holdout_ids=plan['holdout_ids'],target_count=len(plan['targets']),
            train_count=len(plan['train_rows']),counts=plan['counts'],shortfalls=plan['shortfalls'],
            model_loaded=False,tokenization_validated=False,**SCOPE),indent=2))
        return
    from .c6b_runtime import train, evaluate
    try:
        (train if training else evaluate)(args,rows,plan,manifest)
    except Exception as exc:
        manifest.update(status='BELOW_MIN_ACCURACY' if manifest.get('status')=='BELOW_MIN_ACCURACY' else 'FAILED',
                        error=f'{type(exc).__name__}: {exc}')
        write_json(args.output_root/name,manifest)
        raise
