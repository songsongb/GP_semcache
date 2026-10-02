"""C6 matched task quality. Selection is model-free and independent of outputs."""
import argparse
import csv
import json
from collections import Counter, defaultdict, deque
from pathlib import Path
from statistics import mean

from semcache.simulation.multi_user import read_workload, logical_user_assignment, digest
from semcache.experiments.m9b_semantic_workload import MODEL_ID, MODEL_REVISION, file_hash, assignment_source
from semcache.evaluation.bleu import compute_bleu

PROFILE_SHA = '8defa27a83e8ad16e9c7efa0f40e68c302aef90363e14a1451288a3bf433cb2c'
PROFILE = 'results/cachegen/c1_5c/rate_calibration/profiles/matched_uniform_k20_v16.bin'
MODES = dict(FULL_RECOMPUTE=(False, False, False), RAW_SEMCACHE=(True, False, False),
             STORAGE_KV_COMP=(True, False, True), TRANSPORT_QKV_COMP=(True, True, False),
             FULL_PIPELINE=(True, True, True))
BLEU = dict(implementation='sacrebleu', scope='corpus', tokenizer='13a', smoothing='exp',
            effective_order=False, lowercase=False)
CONTRACT = dict(trained_adapter=False, paper_bleu_claimed=False,
    quality_scope='controlled untrained fixtures; matched-mode delta quality; no paper reproduction',
    physical_safety_contract='PHYSICAL_EXACT_W3_WITHIN_SEMANTIC_CLUSTER',
    strict_m9b_safety='diagnostic only', resident_q='source TOTAL Q FP16 uncompressed; Q compression deferred to C7',
    storage_policy='UNIFORM_K20_V16', storage_transform='B2_ANCHOR_MOD_RESIDUAL_KV',
    protocol_provenance='REPRODUCTION_CHOICE')
RULE = ('Earlier source in workload order, same measured cluster and exact w3, distinct source ID and query text; '
        'distinct full token sequences; starts >=3, no special tokens. One candidate per target: earliest source then source span then target span. '
        'Round-robin groups in first-appearance order: SNIPS intent / MultiWOZ conversation; '
        'stable target order within group. Cold single-entry admission must succeed; one selected hit; fixed raw logical capacity.')


def select(rows, dataset, count, seed=42):
    if count < 1:
        raise ValueError('per-dataset must be positive')
    users = logical_user_assignment(rows, 2, seed, dataset)
    index, groups = defaultdict(list), {}
    for ti, target in enumerate(rows):
        if (target.get('semantic_execution_provenance') != 'MEASURED' or target.get('model_revision') != MODEL_REVISION
                or target.get('semantic_assignment_source') != assignment_source(dataset)
                or target.get('tokenizer_id') != f'{MODEL_ID}@{MODEL_REVISION}'):
            raise ValueError('C6 requires measured semantic workloads and pinned OPT revision')
        if not isinstance(target.get('reference_text'), str):
            raise ValueError(f'Missing reference_text: {dataset}:{target["source_id"]}')
        specials = {0, 1, 2} | set(target.get('special_token_ids', []))
        windows = [(p, tuple(target['token_ids'][p:p+3])) for p in range(3, len(target['token_ids'])-2)
                   if not specials.intersection(target['token_ids'][p:p+3])]
        candidates = []
        for tp, ids in windows:
            for si, sp in index[(target['cluster_id'], ids)]:
                source = rows[si]
                if (source['source_id'] != target['source_id'] and source['query_text'] != target['query_text']
                        and source['token_ids'] != target['token_ids']):
                    candidates.append((si, sp, tp, ids))
                    break
        if candidates:
            si, sp, tp, ids = min(candidates)
            source = rows[si]
            user = lambda i: {'user_000': 'user_a', 'user_001': 'user_b'}[users[i]]
            episode = dict(episode_id=f'{dataset}:{si}:{ti}:{sp}:{tp}', dataset=dataset,
                source_index=si, target_index=ti, source_id=source['source_id'], target_id=target['source_id'],
                source_user=user(si), target_user=user(ti), cross_user=users[si] != users[ti],
                m9b_strict_eligible=False, m9b_strict_diagnostic_only=True,
                m9b_strict_reason='no external fixture evidence; natural cross-query reuse',
                cluster=target['cluster_id'], token_ids=list(ids), cache_key=[target['cluster_id'], list(ids)],
                source_start=sp, target_start=tp, selected_hit_count=1, admission='cold single-entry; normal admission must succeed',
                reference_text=target['reference_text'], reference_sha256=digest(target['reference_text']),
                intent=target.get('domain_or_intent'), conversation_id=target.get('conversation_id'))
            group = target.get('domain_or_intent') if dataset == 'snips' else target.get('conversation_id')
            if group is None:
                group = target['source_id']
            groups.setdefault(group, deque()).append(episode)
        for sp, ids in windows:
            index[(target['cluster_id'], ids)].append((ti, sp))
    selected = []
    while len(selected) < count and any(groups.values()):
        for queue in groups.values():
            if queue and len(selected) < count:
                selected.append(queue.popleft())
    if len(selected) != count:
        raise ValueError(f'{dataset}: requested {count}, only {len(selected)} eligible episodes')
    return selected



def validate_hit(episode, hit):
    """Validate observed retrieval identity, not just a copy of the planned hash."""
    entry, window = hit.entry, hit.window
    if (entry.key != (episode['cluster'], tuple(episode['token_ids']))
            or entry.positions != (episode['source_start'], episode['source_start']+3)
            or window.start != episode['target_start'] or window.end != episode['target_start']+3
            or tuple(window.token_ids) != tuple(episode['token_ids'])
            or entry.qkv_metadata.get('source_user') != episode['source_user']
            or entry.qkv_metadata.get('source_id') != episode['source_id']):
        raise ValueError('Observed hit differs from frozen logical episode')
    return digest(episode)



def case_result(episode, mode, tokens, text, accounting):
    reuse, transport, storage = MODES[mode]
    return dict(**{**episode, 'selected_hit_count': int(reuse)}, **CONTRACT, mode=mode,
        generated_token_ids=tokens, generated_text=text, generated_length=len(tokens),
        cache_hit=reuse, executed_hit_count=int(reuse), logical_event_hash=digest(episode) if reuse else None,
        transport_mode='LoRA delta CacheGen' if transport else 'raw' if reuse else 'none',
        storage_mode='TOTAL KV frozen; Q FP16' if storage else 'TOTAL QKV FP16' if reuse else 'none',
        **accounting)


def generation_metrics(raw, other):
    prefix = 0
    for a, b in zip(raw, other):
        if a != b:
            break
        prefix += 1
    previous = list(range(len(other)+1))
    for i, a in enumerate(raw, 1):
        current = [i]
        for j, b in enumerate(other, 1):
            current.append(min(current[-1]+1, previous[j]+1, previous[j-1]+(a != b)))
        previous = current
    aligned = min(len(raw), len(other))
    return dict(exact_generation=raw == other, common_prefix_length=prefix,
        position_agreement=sum(a == b for a, b in zip(raw, other))/aligned if aligned else float(raw == other),
        normalized_edit_distance=previous[-1]/max(len(raw), len(other), 1),
        first_divergent_position=None if raw == other else prefix, raw_length=len(raw), generated_length=len(other))


def logit_metrics(raw, other):
    import torch
    if raw.shape != other.shape or raw.ndim != 2 or raw.shape[-1] < 2 or not len(raw) or not torch.isfinite(raw).all() or not torch.isfinite(other).all():
        raise ValueError('Need aligned finite [positions,vocabulary] logits')
    a, b = raw.float(), other.float()
    la, lb = a.log_softmax(-1), b.log_softmax(-1)
    kl = (la.exp()*(la-lb)).sum(-1).clamp_min(0)
    k = min(5, a.shape[-1])
    ai, bi = a.topk(k).indices, b.topk(k).indices
    at, bt = a.argmax(-1), b.argmax(-1)
    am = a.topk(2).values.diff(dim=-1).neg().mean().item()
    bm = b.topk(2).values.diff(dim=-1).neg().mean().item()
    return dict(mean_kl=kl.mean().item(), median_kl=kl.quantile(.5).item(),
        max_abs_logit_difference=(a-b).abs().max().item(), mean_abs_logit_difference=(a-b).abs().mean().item(),
        mean_logit_cosine=torch.nn.functional.cosine_similarity(a,b,dim=-1).mean().item(),
        top1_agreement=(at == bt).float().mean().item(),
        top5_overlap=(ai[:,:,None] == bi[:,None,:]).any(-1).float().mean().item(),
        raw_top1_rank=(1+(b > b.gather(1,at[:,None])).sum(-1)).float().mean().item(),
        raw_margin=am, comparison_margin=bm, margin_delta=bm-am)


def normalized_label(text):
    return text.strip().casefold()


def verify_profile(path):
    if file_hash(path) != PROFILE_SHA:
        raise ValueError('Frozen K20/V16 profile SHA256 mismatch')


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False)+'\n')


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows({k: json.dumps(v) if isinstance(v, (dict,list)) else v for k,v in r.items()} for r in rows)


def summarize(cases, pairs):
    result = []
    for dataset in ('snips', 'multiwoz'):
        scores = {}
        for mode in MODES:
            subset = [r for r in cases if r['dataset'] == dataset and r['mode'] == mode]
            if not subset:
                continue
            score = compute_bleu([r['generated_text'] for r in subset], [r['reference_text'] for r in subset], **BLEU)
            scores[mode] = score['value']
            paired = [p for p in pairs if p['dataset'] == dataset and p['mode'] == mode and p['baseline'] == 'RAW_SEMCACHE']
            result.append(dict(dataset=dataset, mode=mode, case_count=len(subset), corpus_bleu=score['value'],
                bleu_protocol=score, **CONTRACT,
                intent_accuracy=mean(normalized_label(r['generated_text']) == normalized_label(r['reference_text']) for r in subset) if dataset == 'snips' else None,
                **{k: mean(p[k] for p in paired) if paired else None for k in
                   ('exact_generation','position_agreement','normalized_edit_distance','mean_kl','mean_logit_cosine','top1_agreement','top5_overlap')}))
        for row in result:
            if row['dataset'] == dataset:
                row['delta_bleu_vs_raw'] = row['corpus_bleu'] - scores['RAW_SEMCACHE']
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for dataset in ('snips', 'multiwoz'):
        parser.add_argument('--'+dataset, type=Path, default=Path(f'results/workloads/m9b_{dataset}_semantic.jsonl'))
    parser.add_argument('--profile-path', type=Path, default=Path(PROFILE))
    parser.add_argument('--storage-src', type=Path)
    parser.add_argument('--per-dataset', type=int, default=32)
    parser.add_argument('--max-new-tokens', type=int, default=32)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--semantic-device', default='cpu', help='Recorded only: use frozen measured cluster assignments')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--modes', nargs='+', choices=list(MODES), default=list(MODES))
    parser.add_argument('--output-dir', type=Path, default=Path('results/cachegen/c6/quality'))
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    if args.max_new_tokens < 1 or len(set(args.modes)) != len(args.modes) or 'RAW_SEMCACHE' not in args.modes:
        parser.error('positive max-new-tokens, unique modes and RAW_SEMCACHE required')
    workloads = {d: read_workload(getattr(args,d),d) for d in ('snips','multiwoz')}
    episodes = sum((select(rows,d,args.per_dataset,args.seed) for d,rows in workloads.items()), [])
    selection = dict(episodes=episodes, rule=RULE, modes={m: MODES[m] for m in args.modes}, **CONTRACT)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any(args.output_dir.iterdir()):
        raise ValueError('Output directory must be empty to preserve frozen artifacts')
    write_json(args.output_dir/'selection.json', selection)
    from .c6_runtime import manifest, execute
    provenance = manifest(args, selection)
    if args.dry_run:
        provenance.update(status='DRY_RUN', output_hashes={'selection.json': file_hash(args.output_dir/'selection.json')}, storage_profile_verified=False,
                          storage_runtime_cdf_fits=0, transport_runtime_cdf_fits=0)
        write_json(args.output_dir/'manifest.json', provenance)
        print(json.dumps(dict(selected_counts=dict(Counter(e['dataset'] for e in episodes)),
            diversity={d: dict(Counter(str(e['intent'] if d == 'snips' else e['conversation_id']) for e in episodes if e['dataset']==d)) for d in workloads},
            cross_user_count=sum(e['cross_user'] for e in episodes), references_available=True, **selection), indent=2))
        return
    provenance['status'] = 'STARTING'
    write_json(args.output_dir/'manifest.json', provenance)
    try:
        execute(args, workloads, selection, provenance)
    except Exception as exc:
        provenance.update(status='FAILED', error=f'{type(exc).__name__}: {exc}')
        write_json(args.output_dir/'manifest.json', provenance)
        raise
