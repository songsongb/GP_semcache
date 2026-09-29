"""Small natural cross-query SemCache physical-storage integration probe.

The planner uses measured M9-B TinyBERT assignments. The real runner replays
the same semantic encoder/clusterer stream, then calls SemCacheEngine.query for
each cold-source/later-target episode. Planning executes no model or codec.
"""
from __future__ import annotations

from collections import defaultdict, deque
import copy
import json
from pathlib import Path
import platform
import statistics
import time

from semcache.cache.cache_entry import CacheEntry
from semcache.cache.global_cache import GlobalCache
from semcache.cache.metric_manager import CacheMetricManager
from semcache.experiments.cachegen.b2 import format as fmt
from semcache.experiments.cachegen.c2.physical_storage import (
    FrozenK20V16Codec, MODE_COMPRESSED, MODE_RAW, POLICY, PROFILE_SHA256,
)
from semcache.experiments.cachegen.c4_quality import generation_comparison, verify_profile
from semcache.experiments.cachegen.common import file_hash, write_csv, write_json
from semcache.experiments.cachegen.c15c.harness import ROOT, git_state
from semcache.experiments.dataset_adapters import SNIPS_INTENTS
from semcache.experiments.m9b_semantic_workload import (
    CLUSTERS, ENCODER_ID, ENCODER_REVISION, MODEL_ID, MODEL_REVISION, assignment_source,
)
from semcache.semantic.hit_selection import CacheHit, select_nonoverlapping
from semcache.semantic.matcher import ExactTokenMatcher
from semcache.semantic.subsequence import SubsequenceExtractor
from semcache.simulation.multi_user import (
    digest, logical_user_assignment, read_workload, safety_eligible,
)

OUTPUT = ROOT/'results/cachegen/c5/e2e_smoke'
WINDOW = 3
MIN_REUSE_START = 3
RAW_ENTRY_BYTES = 3*3*2560*32*2
LOGICAL_CAPACITY_BYTES = 64*1024**2
MODES = ('RAW_SEMCACHE', 'COMPRESSED_SEMCACHE')
MAX_ENCODES = 32
MAX_ENCODES_LIMIT = 64
SAFETY_CONTRACT = 'PHYSICAL_EXACT_W3_WITHIN_SEMANTIC_CLUSTER'


def m9b_strict_diagnostic(source, target, source_user, target_user, spans):
    """Report M9-B's separate gate; never use it to select/execute C5 reuse."""
    source_hash, target_hash = digest(source['token_ids']), digest(target['token_ids'])
    source_adapter, target_adapter = source.get('adapter_id'), target.get('adapter_id')
    details = []
    for span in spans:
        original = dict(user_id=source_user, adapter_id=source_adapter,
                        prompt_hash=source_hash, start=span['source_start'])
        destination = dict(user_id=target_user, adapter_id=target_adapter,
                           prompt_hash=target_hash)
        accepted = safety_eligible(original, destination, span['target_start'], set())
        details.append(dict(accepted=accepted, same_user=source_user == target_user,
            adapter_present_and_equal=bool(source_adapter and source_adapter == target_adapter),
            full_prompt_identity=source_hash == target_hash,
            same_start=span['source_start'] == span['target_start'],
            external_fixture_evidence_present=False))
    return dict(decision='ACCEPTED' if details and all(x['accepted'] for x in details) else 'REJECTED',
                eligible=bool(details) and all(x['accepted'] for x in details),
                per_span=details, diagnostic_only=True,
                evidence_policy='M9-B simulate() supplies an empty external-evidence set')


def _group(row, dataset):
    if dataset == 'snips':
        group = str(row['source_id']).rpartition(':')[0]
        if group not in SNIPS_INTENTS:
            raise ValueError('Prepared SNIPS source ID lacks a known intent')
        return group
    group = row.get('conversation_id')
    if not group or not str(row['source_id']).startswith(str(group)+':'):
        raise ValueError('Prepared MultiWOZ row lacks a consistent dialogue ID')
    return str(group)


def _validate_row(row, dataset):
    if (row['dataset'] != dataset or row['model_id'] != MODEL_ID or
            row.get('model_revision') not in (None, MODEL_REVISION) or
            row.get('semantic_assignment_source') != assignment_source(dataset) or
            row.get('window_size', WINDOW) != WINDOW or
            not isinstance(row.get('query_text'), str) or not row['query_text'].strip()):
        raise ValueError('Incompatible prepared M9-B semantic row')
    return row


def _logical_pair(source, target):
    """Reproduce engine key extraction, source admission, then target selection.

    Impact is unavailable without OPT; None is the production policy's explicit
    no-evidence value. Runtime must confirm the actual attention-based decision.
    """
    extractor, matcher = SubsequenceExtractor(WINDOW), ExactTokenMatcher()
    cache = GlobalCache(LOGICAL_CAPACITY_BYTES)
    manager = CacheMetricManager(cache, rho=.8, history_lambda=100, frequency_window=100)
    src_windows = extractor.extract(source['token_ids'])
    dst_windows = extractor.extract(target['token_ids'])
    manager.arrive([matcher.key(source['cluster_id'], w) for w in src_windows])
    admitted = set()
    for w in src_windows:
        key = matcher.key(source['cluster_id'], w)
        if key in cache.entries:
            continue
        entry = CacheEntry(source['cluster_id'], w.token_ids, (w.start, w.end),
                           RAW_ENTRY_BYTES, impact=None,
                           qkv_metadata=dict(source_query_id=source['source_id']))
        if cache.insert(entry, manager.frequencies[key], manager.frequencies):
            admitted.add(key)
    manager.arrive([matcher.key(target['cluster_id'], w) for w in dst_windows])
    hits, misses = [], []
    for w in dst_windows:
        key = matcher.key(target['cluster_id'], w)
        entry = cache.lookup(key, record_reuse=False)
        if entry is None:
            misses.append(key)
        else:
            hits.append(CacheHit(w, entry, entry.impact or 0.))
    selected, _ = select_nonoverlapping(hits, len(target['token_ids']))
    target_admission_upper_bound = len(set(misses)-admitted)
    return dict(admitted_keys=admitted, selected=selected,
                candidate_hits=len(hits), expected_compressed_admissions_upper_bound=(
                    len(admitted)+target_admission_upper_bound))


def discover(rows, dataset, users, *, max_prompt_tokens=16):
    """Quality-blind ordered source→target candidates from measured semantics."""
    if type(max_prompt_tokens) is not int or max_prompt_tokens < WINDOW+1:
        raise ValueError('Invalid prompt-token bound')
    assignment = logical_user_assignment(rows, 2, 42, dataset) if users is None else users
    if len(assignment) != len(rows):
        raise ValueError('User assignment length mismatch')
    index = defaultdict(lambda: deque(maxlen=16))
    episodes = []
    extractor = SubsequenceExtractor(WINDOW)
    for target_index, target in enumerate(rows):
        _validate_row(target, dataset)
        ids = target['token_ids']
        eligible_target = WINDOW+1 <= len(ids) <= max_prompt_tokens
        windows = extractor.extract(ids) if eligible_target else []
        source_indices = set()
        for w in windows:
            if w.start >= MIN_REUSE_START:
                source_indices.update(index[(target['cluster_id'], w.token_ids)])
        ranked = sorted(source_indices, key=lambda i: (
            assignment[i] == assignment[target_index],
            len(rows[i]['token_ids']), -i))
        for source_index in ranked:
            source = rows[source_index]
            if (source_index >= target_index or source['source_id'] == target['source_id'] or
                    source['token_ids'] == ids):
                continue
            logical = _logical_pair(source, target)
            if not logical['selected'] or not logical['admitted_keys']:
                continue
            # The physical engine executes every selected nonoverlapping hit.
            # Reject the whole episode if any executed span is a prompt prefix.
            if any(hit.entry.positions[0] < MIN_REUSE_START or
                   hit.window.start < MIN_REUSE_START for hit in logical['selected']):
                continue
            spans = [dict(cache_key=[h.entry.cluster_id, list(h.window.token_ids)],
                          token_ids=list(h.window.token_ids),
                          source_start=h.entry.positions[0], source_end=h.entry.positions[1],
                          target_start=h.window.start,
                          target_end=h.window.end) for h in logical['selected']]
            if any(tuple(source['token_ids'][s['source_start']:s['source_start']+WINDOW]) != tuple(s['token_ids'])
                   or tuple(ids[s['target_start']:s['target_end']]) != tuple(s['token_ids']) for s in spans):
                raise AssertionError('Exact w=3 token safety failed')
            # M9-B rows use the pinned OPT tokenizer with special tokens. Using
            # each row's initial token ID avoids guessing a BOS ID from memory.
            avoids_initial_token_id = all(
                source['token_ids'][0] not in span['token_ids'] and
                ids[0] not in span['token_ids'] for span in spans)
            strict = m9b_strict_diagnostic(source, target,
                assignment[source_index], assignment[target_index], spans)
            episodes.append(dict(episode_id=f'{dataset}:{source_index}->{target_index}',
                dataset=dataset, source_id=source['source_id'], target_id=target['source_id'],
                source_query_ne_target_query=True,
                source_index=source_index, target_index=target_index,
                source_user=assignment[source_index], target_user=assignment[target_index],
                cross_user=assignment[source_index] != assignment[target_index],
                cluster_id=target['cluster_id'], source_cluster_id=source['cluster_id'],
                target_cluster_id=target['cluster_id'], semantic_candidate_match=True,
                physical_safety_contract=SAFETY_CONTRACT,
                m9b_strict_safety=strict,
                semantic_match_evidence=dict(prepared_cluster_id=target['cluster_id'],
                    assignment_source=target['semantic_assignment_source'],
                    source_embedding_sha256=source.get('semantic_embedding_sha256'),
                    target_embedding_sha256=target.get('semantic_embedding_sha256')),
                group=_group(target, dataset), spans=spans,
                nonprefix_reuse=True,
                initial_special_token_id_avoided=avoids_initial_token_id,
                expected_natural_hits=len(spans), candidate_hits=logical['candidate_hits'],
                expected_compressed_admissions=logical['expected_compressed_admissions_upper_bound'],
                source_admissions=len(logical['admitted_keys']),
                source_token_count=len(source['token_ids']), target_token_count=len(ids),
                exact_w3_safe=True, cold_episode=True))
            break
        # Only short sources are indexed; no target text, quality, or model output
        # enters the selection score. The first C anchors are not cold sources.
        if target_index >= CLUSTERS[dataset] and WINDOW+1 <= len(ids) <= max_prompt_tokens:
            first_occurrences = set()
            for w in windows:
                if w.token_ids in first_occurrences:
                    continue
                first_occurrences.add(w.token_ids)
                if w.start < MIN_REUSE_START:
                    continue
                key = (target['cluster_id'], w.token_ids)
                if not index[key] or index[key][-1] != target_index:
                    index[key].append(target_index)
    return episodes


def select_diverse(episodes, per_dataset, *, max_encodes=MAX_ENCODES,
                   max_semantic_prefix_rows=2048, diagnostics=None):
    if type(per_dataset) is not int or per_dataset not in (1, 2, 4):
        raise ValueError('C5 permits 1, 2, or 4 episodes per dataset')
    if type(max_encodes) is not int or max_encodes < 1:
        raise ValueError('Invalid encode guard')
    if type(max_semantic_prefix_rows) is not int or max_semantic_prefix_rows < 1:
        raise ValueError('Invalid semantic-prefix guard')
    chosen, groups = [], set()
    selected_indices, encode_rejected_indices = set(), set()
    # Prefer avoiding the initial OPT token ID, then smallest encode work;
    # cross-user and source order break ties.
    # A distinct intent/dialogue is taken before a second from one group.
    order = sorted(((i, e) for i, e in enumerate(episodes)
                    if e['target_index'] < max_semantic_prefix_rows), key=lambda item: (
        not item[1].get('initial_special_token_id_avoided', True),
        item[1]['expected_compressed_admissions'], not item[1]['cross_user'], item[0]))
    for distinct in (True, False):
        for candidate_index, episode in order:
            if len(chosen) >= per_dataset:
                break
            if episode in chosen or (distinct and episode['group'] in groups):
                continue
            if sum(e['expected_compressed_admissions'] for e in chosen)+episode['expected_compressed_admissions'] > max_encodes:
                encode_rejected_indices.add(candidate_index)
                continue
            chosen.append(episode)
            groups.add(episode['group'])
            selected_indices.add(candidate_index)
    if diagnostics is not None:
        rejected = []
        for candidate_index, episode in enumerate(episodes):
            if candidate_index in selected_indices:
                continue
            if episode['target_index'] >= max_semantic_prefix_rows:
                reason = 'rejected_by_max_semantic_prefix_rows'
            elif (episode['expected_compressed_admissions'] > max_encodes or
                  candidate_index in encode_rejected_indices):
                reason = 'rejected_by_max_encodes'
            else:
                reason = 'rejected_by_other_guard'
            rejected.append(dict(episode=episode, reason=reason))
        diagnostics.update(natural_episodes_after_nonprefix_filter=len(episodes),
            episodes_passing_prefix_guard=sum(e['target_index'] < max_semantic_prefix_rows
                                              for e in episodes),
            episodes_passing_encode_guard=sum(
                e['target_index'] < max_semantic_prefix_rows and
                e['expected_compressed_admissions'] <= max_encodes for e in episodes),
            final_selectable_episodes=len(chosen), rejected=rejected,
            rejection_counts={reason: sum(item['reason'] == reason for item in rejected)
                for reason in ('rejected_by_max_encodes',
                               'rejected_by_max_semantic_prefix_rows',
                               'rejected_by_other_guard')})
    return chosen


def minimum_balanced_plan(discovered, per_dataset, *, max_encodes=MAX_ENCODES,
                          max_semantic_prefix_rows=2048):
    """Exact cheapest valid N+N plan, independent of the global encode cap.

    Diversity is a selection preference with a repeat-group fallback, not an
    eligibility constraint. Ties retain the existing quality-blind priority.
    Discovery supplies unique episodes (one per target); cold episodes have
    no cross-episode conflict rule, so dataset costs compose independently.
    """
    if type(per_dataset) is not int or per_dataset not in (1, 2, 4):
        raise ValueError('C5 permits 1, 2, or 4 episodes per dataset')
    eligible = {}
    dataset_minima = {}
    for name in ('snips', 'multiwoz'):
        eligible[name] = [episode for _, episode in sorted(
            ((i, episode) for i, episode in enumerate(discovered[name])
             if episode['target_index'] < max_semantic_prefix_rows),
            key=lambda item: (
                item[1]['expected_compressed_admissions'],
                not item[1].get('initial_special_token_id_avoided', True),
                not item[1]['cross_user'], item[0]))]
        cheapest = eligible[name][:per_dataset]
        dataset_minima[name] = dict(
            minimum_single_episode_encode_cost=(
                cheapest[0]['expected_compressed_admissions'] if cheapest else None),
            minimum_single_episode_id=(cheapest[0]['episode_id'] if cheapest else None),
            minimum_valid_n_episode_combined_encode_cost=(
                sum(e['expected_compressed_admissions'] for e in cheapest)
                if len(cheapest) == per_dataset else None),
            minimum_valid_n_episode_ids=(
                [e['episode_id'] for e in cheapest] if len(cheapest) == per_dataset else []))
    minimum_episodes = (eligible['snips'][:per_dataset] + eligible['multiwoz'][:per_dataset]
                        if all(len(eligible[name]) >= per_dataset for name in eligible) else [])
    minimum_cost = (sum(e['expected_compressed_admissions'] for e in minimum_episodes)
                    if len(minimum_episodes) == 2*per_dataset else None)
    participation = {}
    for name in ('snips', 'multiwoz'):
        other = 'multiwoz' if name == 'snips' else 'snips'
        for episode in eligible[name]:
            # Only the first N candidates are needed to find N-1 companions.
            companions = [e for e in eligible[name][:per_dataset] if e is not episode][:per_dataset-1]
            required = [episode, *companions, *eligible[other][:per_dataset]]
            participation[id(episode)] = bool(
                len(companions) == per_dataset-1 and len(required) == 2*per_dataset and
                sum(e['expected_compressed_admissions'] for e in required) <= max_encodes)
    return dict(balanced_per_dataset=per_dataset,
                minimum_balanced_encode_upper_bound=minimum_cost,
                minimum_encode_cap_excess=(max(0, minimum_cost-max_encodes)
                                         if minimum_cost is not None else None),
                minimum_balanced_episode_ids=[e['episode_id'] for e in minimum_episodes],
                minimum_episodes=minimum_episodes, dataset_minima=dataset_minima,
                participation=participation)


def select_balanced_global(discovered, per_dataset, *, max_encodes=MAX_ENCODES,
                           max_semantic_prefix_rows=2048):
    """Find the first diverse, quality-blind balanced set under one encode cap.

    Candidate priority matches select_diverse. Distinct groups are preferred,
    then source order breaks ties. The cap is shared by both datasets.
    """
    if type(per_dataset) is not int or per_dataset not in (1, 2, 4):
        raise ValueError('C5 permits 1, 2, or 4 episodes per dataset')
    if type(max_encodes) is not int or not 1 <= max_encodes <= MAX_ENCODES_LIMIT:
        raise ValueError('C5 maximum compressed encodes must remain within 1..64')
    if type(max_semantic_prefix_rows) is not int or max_semantic_prefix_rows < 1:
        raise ValueError('Invalid semantic-prefix guard')
    names = ('snips', 'multiwoz')
    ordered = {}
    for name in names:
        ordered[name] = [e for _, e in sorted(
            ((i, e) for i, e in enumerate(discovered[name])
             if e['target_index'] < max_semantic_prefix_rows),
            key=lambda item: (
                not item[1].get('initial_special_token_id_avoided', True),
                item[1]['expected_compressed_admissions'],
                not item[1]['cross_user'], item[0]))]

    def completion_cost(candidates, count, distinct, groups=frozenset()):
        """Exact suffix cost; groups are the existing diversity preference."""
        if count == 0:
            return 0
        if distinct:
            group_costs = {}
            for episode in candidates:
                group = episode['group']
                if group in groups:
                    continue
                cost = episode['expected_compressed_admissions']
                group_costs[group] = min(cost, group_costs.get(group, float('inf')))
            costs = sorted(group_costs.values())
        else:
            costs = sorted(e['expected_compressed_admissions'] for e in candidates)
        return sum(costs[:count]) if len(costs) >= count else float('inf')

    def first_selection(name, budget, distinct):
        # Choose the earliest candidate that has a feasible suffix. This gives
        # exactly the old depth-first search's first combination without walking
        # combinations. At most M suffix checks, each O(M log M); no C(M,N).
        candidates = ordered[name]
        chosen, groups = [], set()
        for i, episode in enumerate(candidates):
            if distinct and episode['group'] in groups:
                continue
            cost = episode['expected_compressed_admissions']
            remaining = per_dataset-len(chosen)-1
            tail_cost = completion_cost(candidates[i+1:], remaining, distinct,
                                        groups | {episode['group']})
            if cost+tail_cost > budget:
                continue
            chosen.append(episode)
            groups.add(episode['group'])
            budget -= cost
            if len(chosen) == per_dataset:
                return chosen
        raise AssertionError('Exact completion bound did not produce a feasible set')

    selected = None
    # Preserve the old phase order, candidate priority, uniqueness (one use of
    # each discovered episode), and distinct-group preference/fallback exactly.
    for snips_distinct, multiwoz_distinct in ((True, True), (True, False),
                                               (False, True), (False, False)):
        min_snips = completion_cost(ordered['snips'], per_dataset, snips_distinct)
        min_multiwoz = completion_cost(ordered['multiwoz'], per_dataset, multiwoz_distinct)
        if min_snips + min_multiwoz > max_encodes:
            continue
        snips = first_selection('snips', max_encodes-min_multiwoz, snips_distinct)
        remaining = max_encodes-sum(e['expected_compressed_admissions'] for e in snips)
        multiwoz = first_selection('multiwoz', remaining, multiwoz_distinct)
        selected = dict(snips=snips, multiwoz=multiwoz)
        break

    minimum_plan = minimum_balanced_plan(
        discovered, per_dataset, max_encodes=max_encodes,
        max_semantic_prefix_rows=max_semantic_prefix_rows)
    diagnostics = {}
    for name in names:
        chosen = selected[name] if selected else []
        selected_ids = {id(e) for e in chosen}
        rejected = []
        for episode in discovered[name]:
            if id(episode) in selected_ids:
                continue
            if episode['target_index'] >= max_semantic_prefix_rows:
                reason = 'rejected_by_max_semantic_prefix_rows'
            elif episode['expected_compressed_admissions'] > max_encodes:
                reason = 'rejected_by_max_encodes'
            elif not minimum_plan['participation'][id(episode)]:
                reason = 'cannot_participate_in_balanced_selection_within_encode_cap'
            else:
                reason = 'not_selected_by_deterministic_priority'
            rejected.append(dict(episode=episode, reason=reason))
        diagnostics[name] = dict(
            natural_episodes_after_nonprefix_filter=len(discovered[name]),
            episodes_passing_prefix_guard=len(ordered[name]),
            episodes_passing_encode_guard=sum(
                e['expected_compressed_admissions'] <= max_encodes for e in ordered[name]),
            individual_candidates_passing_max_encodes=sum(
                e['expected_compressed_admissions'] <= max_encodes for e in ordered[name]),
            candidates_participating_in_feasible_balanced_selection=sum(
                minimum_plan['participation'][id(e)] for e in ordered[name]),
            candidates_considered=len(ordered[name]),
            final_selectable_episodes=len(chosen), rejected=rejected,
            rejection_counts={reason: sum(item['reason'] == reason for item in rejected)
                for reason in ('rejected_by_max_encodes',
                               'rejected_by_max_semantic_prefix_rows',
                               'cannot_participate_in_balanced_selection_within_encode_cap',
                               'not_selected_by_deterministic_priority')})
    return selected, diagnostics


def plan(args):
    if type(args.max_encodes) is not int or not 1 <= args.max_encodes <= MAX_ENCODES_LIMIT:
        raise ValueError('C5 maximum compressed encodes must remain within 1..64')
    profile = verify_profile(args.profile_path)
    if args.coder_backend != fmt.FAST_CODER:
        raise ValueError('C5 requires FAST_PY_BITEXACT')
    print(f'physical_safety_contract={SAFETY_CONTRACT}; m9b_strict_safety=diagnostic_only')
    paths = {'snips': Path(args.snips), 'multiwoz': Path(args.multiwoz)}
    rows = {name: read_workload(path, name) for name, path in paths.items()}
    preparation = {}
    for name, path in paths.items():
        sidecar = Path(str(path)+'.manifest.json')
        if sidecar.is_file():
            preparation[name] = json.loads(sidecar.read_text())
            if (preparation[name].get('output_semantic_sha256') != file_hash(path) or
                    preparation[name].get('dataset') != name or
                    preparation[name].get('metadata', {}).get('execution_provenance') != 'MEASURED' or
                    type(preparation[name].get('batch_size')) is not int or
                    preparation[name]['batch_size'] < 1):
                raise ValueError(f'Invalid M9-B prepared semantic sidecar: {sidecar}')
        else:
            # Synthetic unit fixtures can exercise planning; the actual C5 run
            # requires the producing manifest for exact TinyBERT batch replay.
            preparation[name] = None
    discovered = {}
    cumulative_encode_upper_bound = 0
    for name in ('snips', 'multiwoz'):
        assignment = logical_user_assignment(rows[name], 2, args.seed, name)
        discovered[name] = discover(rows[name], name, assignment,
                                    max_prompt_tokens=args.max_prompt_tokens)
    chosen, selection_diagnostics = select_balanced_global(
        discovered, args.per_dataset, max_encodes=args.max_encodes,
        max_semantic_prefix_rows=args.max_semantic_prefix_rows)
    print(f'global_encode_limit={args.max_encodes} '
          f'feasible_balanced_selection={str(chosen is not None).lower()}')
    if args.dry_run:
        minimum = minimum_balanced_plan(
            discovered, args.per_dataset, max_encodes=args.max_encodes,
            max_semantic_prefix_rows=args.max_semantic_prefix_rows)
        print(f'balanced_per_dataset={args.per_dataset} '
              f'minimum_encode_cap_excess={minimum["minimum_encode_cap_excess"]} '
              f'minimum_balanced_encode_upper_bound='
              f'{minimum["minimum_balanced_encode_upper_bound"]} '
              f'minimum_balanced_episode_ids='
              f'{minimum["minimum_balanced_episode_ids"]}')
        cumulative_minimum = 0
        dataset_cumulative = dict(snips=0, multiwoz=0)
        for episode in minimum['minimum_episodes']:
            cumulative_minimum += episode['expected_compressed_admissions']
            dataset_cumulative[episode['dataset']] += episode['expected_compressed_admissions']
            spans = episode['spans']
            print(f"  minimum_cost_episode={episode['episode_id']} "
                  f"source_id={episode['source_id']} target_id={episode['target_id']} "
                  f"expected_compressed_admissions={episode['expected_compressed_admissions']} "
                  f"cumulative_minimum_encode_upper_bound={cumulative_minimum} "
                  f"dataset_cumulative_encode_upper_bound={dataset_cumulative[episode['dataset']]} "
                  f"source_positions={[[s['source_start'], s['source_end']] for s in spans]} "
                  f"target_positions={[[s['target_start'], s['target_end']] for s in spans]} "
                  f"cross_user={str(episode['cross_user']).lower()} "
                  f"semantic_cluster={episode['cluster_id']} "
                  f"exact_w3_token_ids={[s['token_ids'] for s in spans]}")
        if chosen is None:
            reason = ('insufficient_prefix_eligible_episodes_for_balanced_selection'
                      if any(selection_diagnostics[name]['episodes_passing_prefix_guard'] <
                             args.per_dataset for name in ('snips', 'multiwoz'))
                      else 'no_global_balanced_combination_within_encode_cap')
            print(f'planner_reason={reason}')
    selected = []
    for name in ('snips', 'multiwoz'):
        picked = chosen[name] if chosen else []
        selected.extend(picked)
        print(f'{name}: natural_episodes_after_nonprefix_filter={len(discovered[name])} '
              f'candidates_considered={selection_diagnostics[name]["candidates_considered"]} '
              f'selected={len(picked)}/{args.per_dataset}')
        if args.dry_run:
            diagnostic = selection_diagnostics[name]
            dataset_minimum = minimum['dataset_minima'][name]
            print(f'{name}: balanced_per_dataset={args.per_dataset} '
                  f'minimum_single_episode_encode_cost='
                  f'{dataset_minimum["minimum_single_episode_encode_cost"]} '
                  f'minimum_single_episode_id={dataset_minimum["minimum_single_episode_id"]} '
                  f'minimum_valid_n_episode_combined_encode_cost='
                  f'{dataset_minimum["minimum_valid_n_episode_combined_encode_cost"]} '
                  f'minimum_valid_n_episode_ids='
                  f'{dataset_minimum["minimum_valid_n_episode_ids"]}')
            print(f'{name}: episodes_passing_prefix_guard={diagnostic["episodes_passing_prefix_guard"]} '
                  f'episodes_passing_encode_guard={diagnostic["episodes_passing_encode_guard"]} '
                  f'individual_candidates_passing_max_encodes='
                  f'{diagnostic["individual_candidates_passing_max_encodes"]} '
                  f'candidates_participating_in_feasible_balanced_selection='
                  f'{diagnostic["candidates_participating_in_feasible_balanced_selection"]} '
                  f'final_selectable_episodes={diagnostic["final_selectable_episodes"]} '
                  f'prefix_guard_limit_rows={args.max_semantic_prefix_rows} '
                  f'global_encode_limit={args.max_encodes} '
                  'encode_guard_count_scope=individual_candidate_within_prefix_guard '
                  f'rejection_counts={diagnostic["rejection_counts"]}')
            for reason in ('rejected_by_max_encodes',
                           'rejected_by_max_semantic_prefix_rows',
                           'cannot_participate_in_balanced_selection_within_encode_cap',
                           'not_selected_by_deterministic_priority'):
                examples = [x for x in diagnostic['rejected'] if x['reason'] == reason][:3]
                for item in examples:
                    episode = item['episode']
                    source_positions = [[s['source_start'], s['source_end']]
                                        for s in episode['spans']]
                    target_positions = [[s['target_start'], s['target_end']]
                                        for s in episode['spans']]
                    token_ids = [s['token_ids'] for s in episode['spans']]
                    prefix_rows_required = episode['target_index'] + 1
                    print(f"  rejected {episode['source_id']} -> {episode['target_id']} "
                          f"source_positions={source_positions} "
                          f"target_positions={target_positions} "
                          f"token_ids={token_ids} "
                          f"expected_compressed_admissions={episode['expected_compressed_admissions']} "
                          f"prefix_rows_required={prefix_rows_required} reason={reason}")
        for e in picked:
            cumulative_encode_upper_bound += e['expected_compressed_admissions']
            print(f"  [{e['source_index']}->{e['target_index']}] "
                  f"{e['source_id']} ({e['source_user']}) -> {e['target_id']} ({e['target_user']}) "
                  f"source_ne_target=true cross_user={str(e['cross_user']).lower()} "
                  f"semantic_cluster_match={e['source_cluster_id'] == e['target_cluster_id']} "
                  f"cluster={e['cluster_id']} "
                  f"exact_w3_spans={e['spans']} expected_hits={e['expected_natural_hits']} "
                  f"initial_token_id_avoided={str(e['initial_special_token_id_avoided']).lower()} "
                  f"m9b_strict={e['m9b_strict_safety']['decision']} "
                  f"expected_compressed_admissions<={e['expected_compressed_admissions']} "
                  f"cumulative_compressed_encode_upper_bound={cumulative_encode_upper_bound}")
    if chosen is None:
        raise ValueError(f'No balanced {args.per_dataset}+{args.per_dataset} non-prefix natural '
                         f'episode combination fits global encode limit {args.max_encodes} '
                         'and semantic-prefix guard; prefix reuse is not a fallback')
    encodes = sum(e['expected_compressed_admissions'] for e in selected)
    if encodes > args.max_encodes:
        raise AssertionError('C5 encode guard exceeded')
    print(f'total_selected_episodes={len(selected)} expected_model_requests={len(selected)*len(MODES)*2} '
          f'expected_compressed_entry_encodes_upper_bound={encodes} '
          f'expected_compressed_hits={sum(e["expected_natural_hits"] for e in selected)} '
          f'max_semantic_prefix_rows={args.max_semantic_prefix_rows} '
          f'profile_sha256={profile["sha256"]}')
    return dict(rows=rows, selected=selected,
                discovered_counts={name: len(value) for name, value in discovered.items()},
                workload_paths={name: str(path.resolve()) for name, path in paths.items()},
                workload_hashes={name: file_hash(path) for name, path in paths.items()},
                preparation_manifest_hashes={name: file_hash(Path(str(path)+'.manifest.json'))
                    if Path(str(path)+'.manifest.json').is_file() else None
                    for name, path in paths.items()},
                preparation=preparation, profile=profile,
                selection_diagnostics=selection_diagnostics,
                expected_encodes_upper_bound=encodes)


def semantic_snapshots(rows, episodes, encoder, *, batch_size=32):
    """Replay the actual M9-B TinyBERT/IntentClusterer stream, once per dataset."""
    from semcache.semantic.intent_clusterer import IntentClusterer
    requested = {i for episode in episodes for i in (episode['source_index'], episode['target_index'])}
    if not requested:
        return {}
    limit = max(requested)
    c = CLUSTERS[episodes[0]['dataset']]
    vectors = []
    for offset in range(0, limit+1, batch_size):
        batch = encoder.encode([row['query_text'] for row in rows[offset:offset+batch_size]])
        if len(batch) != min(batch_size, len(rows)-offset):
            raise ValueError('TinyBERT semantic replay length mismatch')
        vectors.extend(batch)
    for i, vector in enumerate(vectors):
        if rows[i].get('semantic_embedding_sha256') != digest(vector):
            raise ValueError(f'Live TinyBERT vector differs from prepared row {i}')
    clusterer = IntentClusterer(c, initialization='first_k', update_mode='buffered',
                                update_interval=100)
    clusterer.initialize(vectors[:c])
    snapshots = {}
    for i, vector in enumerate(vectors):
        if i in requested:
            snapshots[i] = copy.deepcopy(clusterer)
        diagnostic = clusterer.observe_with_diagnostics(vector)
        if diagnostic['cluster_id'] != rows[i]['cluster_id']:
            raise ValueError(f'Live semantic cluster differs from prepared row {i}')
    return snapshots


def verify_pair(episode, raw, compressed, raw_cache, compressed_cache, device):
    """Inspect the real engine's selected decoded hit views, not a forced key."""
    import torch
    raw_hits, compressed_hits = raw['reuse_hits'], compressed['reuse_hits']
    if not raw_hits or len(raw_hits) != len(compressed_hits):
        raise ValueError('RAW/COMPRESSED natural selected-hit count differs or is zero')
    signatures = []
    for raw_hit, compressed_hit in zip(raw_hits, compressed_hits):
        a, b = raw_hit.entry, compressed_hit.entry
        span = (raw_hit.window.start, raw_hit.window.end)
        if (a.key != b.key or span != (compressed_hit.window.start, compressed_hit.window.end)
                or len(raw_hit.window.token_ids) != WINDOW or
                a.qkv_metadata['source_query_id'] != episode['source_id'] or
                b.qkv_metadata['source_query_id'] != episode['source_id'] or
                not b.compressed_kv or b.resident.tensors is not None):
            raise ValueError('Natural semantic/exact-token hit or compressed residency differs')
        source_ids = raw_cache.entries[a.key].qkv_metadata['source_query_token_ids']
        if (tuple(source_ids[a.positions[0]:a.positions[1]]) != raw_hit.window.token_ids or
                tuple(compressed['summary']['token_ids'][span[0]:span[1]]) != raw_hit.window.token_ids):
            raise ValueError('Exact-token w=3 safety failed')
        original = tuple(torch.cat([a.tensors[layer][index] for layer in range(32)], dim=0).to(device)
                         for index in (1, 2))
        for index, role in enumerate(('K', 'V')):
            direct = POLICY.quantize(original[index], role)
            reconstructed = torch.cat([b.tensors[layer][index+1] for layer in range(32)], dim=0)
            if (not torch.equal(b.kv_symbols[index], direct.symbols) or
                    reconstructed.shape != original[index].shape or reconstructed.dtype != torch.float16 or
                    not torch.equal(reconstructed, direct.reconstructed) or
                    not torch.isfinite(reconstructed).all()):
                raise ValueError(f'{role} symbol/reconstruction mismatch')
        for layer in range(32):
            if not torch.equal(a.tensors[layer][0], b.tensors[layer][0]):
                raise ValueError('Q differs between RAW and COMPRESSED')
        signatures.append(dict(cache_key=[a.cluster_id, list(a.token_ids)],
            token_ids=list(raw_hit.window.token_ids), source_start=a.positions[0],
            source_end=a.positions[1], target_start=span[0], target_end=span[1]))
    for cache in (raw_cache, compressed_cache):
        if cache.capacity_bytes != LOGICAL_CAPACITY_BYTES:
            raise ValueError('C5 logical capacities differ')
    assert_compressed_residency(compressed_cache)
    if signatures != episode['spans']:
        raise ValueError('Executed natural reuse spans differ from C5 discovery')
    return signatures


def _event_signature(result, kinds):
    return [(e['event_type'], e.get('cache_key'), e.get('target_start'), e.get('target_end'))
            for e in result['events'] if e['event_type'] in kinds]


def assert_compressed_residency(cache):
    if not cache.entries or any(e.tensors is not None or e.compressed_kv is None
                                for e in cache.entries.values()):
        raise ValueError('Compressed cache retains raw K/V or lacks frozen B2 payload')


def assert_logical_pair(raw_requests, compressed_requests):
    """Fail before interpreting output if capacity/semantic trajectories differ."""
    kinds = {'CLUSTER_ASSIGN', 'CACHE_LOOKUP', 'HIT', 'MISS', 'REUSE_PROVENANCE',
             'ADMIT', 'DENY', 'INSERT', 'EVICT', 'FETCH'}
    for raw, compressed in zip(raw_requests, compressed_requests):
        a, b = raw['summary'], compressed['summary']
        for field in ('cluster_id', 'block_hit_count', 'executed_nonoverlap_hits',
                      'admitted_block_count', 'rejected_block_count', 'reused_unique_token_count'):
            if a[field] != b[field]:
                raise ValueError(f'RAW/COMPRESSED logical {field} differs')
        if _event_signature(raw, kinds) != _event_signature(compressed, kinds):
            raise ValueError('RAW/COMPRESSED semantic/key/admission/eviction events differ')


def _generate(model, state, max_new_tokens, device):
    import torch
    start = time.perf_counter()
    past, logits, tokens = state['past_key_values'], state['next_logits'], []
    with torch.inference_mode():
        for step in range(max_new_tokens):
            if not torch.isfinite(logits).all():
                raise ValueError('Nonfinite greedy-generation logits')
            next_token = logits.argmax(-1, keepdim=True)
            token_id = int(next_token.item())
            tokens.append(token_id)
            if token_id == model.config.eos_token_id:
                break
            if step+1 < max_new_tokens:
                output = model(input_ids=next_token, past_key_values=past, use_cache=True)
                past, logits = output.past_key_values, output.logits[:, -1]
    if str(device).startswith('cuda'):
        torch.cuda.synchronize(device)
    return tokens, 1000*(time.perf_counter()-start)


def _timings(cache, source, target, generation_ms):
    records = cache.storage_timing_records or []
    inserts = [r for r in records if r['operation'] == 'insert']
    lookups = [r for r in records if r['operation'] == 'lookup']
    amount = lambda records, key: sum(r.get(key) or 0. for r in records)
    return dict(source_request_ms=source['summary']['timing'].get('request_wall_ms'),
                target_request_ms=target['summary']['timing'].get('request_wall_ms'),
                generation_ms=generation_ms,
                quantization_ms=amount(inserts, 'quantize_ms'),
                encode_ms=amount(inserts, 'encode_ms'),
                cache_lookup_ms=amount(lookups, 'lookup_ms'),
                decode_ms=amount(lookups, 'decode_ms'),
                dequantization_ms=amount(lookups, 'dequantize_ms'),
                model_reuse_inference_ms=(source['summary']['timing'].get('prefill_wall_ms') or 0.)+
                    (target['summary']['timing'].get('prefill_wall_ms') or 0.)+generation_ms,
                total_episode_ms=(source['summary']['timing'].get('request_wall_ms') or 0.)+
                    (target['summary']['timing'].get('request_wall_ms') or 0.)+generation_ms)


def _run_mode(episode, rows, model, tokenizer, adapter, encoder, snapshots, codec,
              mode, max_new_tokens, device):
    from semcache.semcache_engine import SemCacheEngine
    source, target = rows[episode['source_index']], rows[episode['target_index']]
    if source['source_id'] == target['source_id'] or source['token_ids'] == target['token_ids']:
        raise ValueError('C5 source and target must be distinct non-repeat queries')
    cache = GlobalCache(LOGICAL_CAPACITY_BYTES, physical_storage_mode=(
        MODE_RAW if mode == MODES[0] else MODE_COMPRESSED),
        physical_codec=None if mode == MODES[0] else codec, instrument_storage=True)
    engine = SemCacheEngine(model, tokenizer, adapter, encoder,
        copy.deepcopy(snapshots[episode['source_index']]), cache,
        window_size=WINDOW, storage_device='cpu', pbr_interval_queries=None,
        metadata=dict(model=MODEL_ID, resolved_model_revision=MODEL_REVISION))
    adapter_names = {'user_000': 'user_a', 'user_001': 'user_b'}
    user_source, user_target = (adapter_names[episode[key]] for key in ('source_user', 'target_user'))
    observed = []
    for row, name, user, generation in ((source, 'source', user_source, False),
                                         (target, 'target', user_target, True)):
        if name == 'target':
            # The intervening prepared requests update only the measured semantic
            # centroid state. This episode has a deliberately cold physical cache.
            engine.clusterer = copy.deepcopy(snapshots[episode['target_index']])
        actual_ids = list(tokenizer(row['query_text'], add_special_tokens=True,
                                    truncation=False)['input_ids'])
        if actual_ids != row['token_ids']:
            raise ValueError('Live OPT tokenizer differs from prepared workload')
        result = engine.query(row['query_text'], user, row['source_id'],
            execution_mode='SEMCACHE_PHYSICAL_REUSE', collect_timing=True,
            return_generation_state=generation, return_reuse_hits=True)
        if result['summary']['cluster_id'] != row['cluster_id']:
            raise ValueError('Live TinyBERT cluster differs from prepared semantic assignment')
        observed.append(result)
    source_result, target_result = observed
    if source_result['summary']['executed_nonoverlap_hits'] != 0:
        raise ValueError('Cold C5 source unexpectedly reused cache')
    if (target_result['summary']['executed_nonoverlap_hits'] < 1 or
            not target_result['summary']['physical_reuse_used'] or
            not target_result['summary']['projection_skip_used']):
        raise ValueError('C5 target did not execute a natural physical reuse hit')
    if target_result['summary']['executed_nonoverlap_hits'] != episode['expected_natural_hits']:
        raise ValueError('Live natural hit count differs from model-free discovery')
    if any(p['source_query_id'] != source['source_id'] for p in
           target_result['summary']['reuse_block_provenance']):
        raise ValueError('Target hit did not come from selected natural source')
    source_keys = [e['cache_key'] for e in source_result['events'] if e['event_type'] == 'INSERT']
    if not source_keys:
        raise ValueError('Source was not admitted by SemCache')
    source_entries = {key: cache.entries[key] for key in source_keys if key in cache.entries}
    if not source_entries:
        raise ValueError('Source entries vanished before target')
    source_accounting = [e.storage_accounting for e in source_entries.values()]
    resident_raw_kv = any(e.tensors is not None for e in source_entries.values())
    if mode == MODES[1]:
        assert_compressed_residency(cache)
    generated, generation_ms = _generate(model, target_result.pop('generation_state'),
                                          max_new_tokens, device)
    import torch
    if (not torch.isfinite(source_result['logits']).all() or
            not torch.isfinite(target_result['logits']).all()):
        raise ValueError('Nonfinite OPT logits')
    return dict(mode=mode, source=source_result, target=target_result, cache=cache,
                generated_token_ids=generated, timings=_timings(cache, source_result,
                    target_result, generation_ms), source_accounting=source_accounting,
                raw_kv_resident=resident_raw_kv)


def _episode_rows(episode, observations, signatures):
    raw, compressed = (observations[mode] for mode in MODES)
    comparison = generation_comparison(raw['generated_token_ids'],
                                       compressed['generated_token_ids'])
    strict = episode['m9b_strict_safety']
    safety_fields = dict(source_query_ne_target_query=episode['source_id'] != episode['target_id'],
        physical_safety_contract=SAFETY_CONTRACT,
        semantic_cluster_id=episode['cluster_id'],
        source_cluster_id=episode.get('source_cluster_id', episode['cluster_id']),
        target_cluster_id=episode.get('target_cluster_id', episode['cluster_id']),
        semantic_candidate_match=True,
        exact_reused_w3_token_ids=json.dumps([s['token_ids'] for s in signatures]),
        source_positions=json.dumps([[s['source_start'], s['source_end']] for s in signatures]),
        target_positions=json.dumps([[s['target_start'], s['target_end']] for s in signatures]),
        m9b_strict_safety_eligible=strict['eligible'],
        m9b_strict_decision=strict['decision'],
        m9b_strict_diagnostic_only=True,
        m9b_strict_diagnostic=json.dumps(strict, sort_keys=True))
    rows = []
    for item in (raw, compressed):
        target = item['target']['summary']
        accounting = item['source_accounting']
        reused_entry_bytes = [item['cache'].entries[(s['cache_key'][0],
            tuple(s['cache_key'][1]))].physical_tensor_bytes for s in signatures]
        rows.append(dict(episode_id=episode['episode_id'], dataset=episode['dataset'],
            storage_mode=item['mode'], source_id=episode['source_id'],
            target_id=episode['target_id'], source_user=episode['source_user'],
            target_user=episode['target_user'], cross_user=episode['cross_user'],
            **safety_fields,
            semantic_candidates=target['block_hit_count'],
            admissions=sum(r['summary']['admitted_block_count'] for r in
                           (item['source'], item['target'])),
            hits=target['block_hit_count'], misses=target['block_lookup_count']-target['block_hit_count'],
            reused_blocks=target['executed_nonoverlap_hits'],
            source_resident_entry_count=len(accounting),
            raw_qkv_entry_bytes=RAW_ENTRY_BYTES,
            mean_physical_entry_bytes=statistics.mean(x['actual_qkv_entry_bytes'] for x in accounting),
            reused_entry_physical_bytes=json.dumps(reused_entry_bytes),
            mean_reused_entry_physical_bytes=statistics.mean(reused_entry_bytes),
            source_raw_qkv_resident_bytes=sum(x['raw_qkv_bytes'] for x in accounting),
            source_physical_resident_bytes=sum(x['actual_qkv_entry_bytes'] for x in accounting),
            compression_ratio=(sum(x['raw_qkv_bytes'] for x in accounting)/
                               sum(x['actual_qkv_entry_bytes'] for x in accounting)),
            raw_kv_resident=item['raw_kv_resident'],
            reused_spans=json.dumps(signatures), generated_token_ids=json.dumps(item['generated_token_ids']),
            **item['timings']))
    pair = dict(episode_id=episode['episode_id'], dataset=episode['dataset'],
        source_id=episode['source_id'], target_id=episode['target_id'],
        source_user=episode['source_user'], target_user=episode['target_user'],
        cross_user=episode['cross_user'], **safety_fields,
        natural_hit_count=len(signatures),
        logical_behavior_identical=True, symbol_mismatches=0, reconstruction_failures=0,
        **comparison, raw_target_request_ms=raw['timings']['target_request_ms'],
        compressed_target_request_ms=compressed['timings']['target_request_ms'])
    return rows, pair


def summarize(pairs, rows):
    result = {}
    for name in ('snips', 'multiwoz', 'overall'):
        group = [p for p in pairs if name == 'overall' or p['dataset'] == name]
        if not group:
            continue
        physical = [r for r in rows if r['storage_mode'] == MODES[1] and
                    (name == 'overall' or r['dataset'] == name)]
        result[name] = dict(episode_count=len(group), natural_hit_count=sum(p['natural_hit_count'] for p in group),
            cross_user_episodes=sum(p['cross_user'] for p in group),
            greedy_exact_match_rate=statistics.mean(p['exact_sequence_match'] for p in group),
            mean_prefix_agreement=statistics.mean(p['prefix_agreement_length'] for p in group),
            mean_position_agreement=statistics.mean(p['token_position_agreement'] for p in group),
            mean_normalized_edit_distance=statistics.mean(p['normalized_token_edit_distance'] for p in group),
            raw_qkv_bytes=sum(r['source_raw_qkv_resident_bytes'] for r in physical),
            compressed_qkv_bytes=sum(r['source_physical_resident_bytes'] for r in physical),
            symbol_mismatches=0, reconstruction_failures=0, logical_mismatches=0)
    return result


def run(args):
    plan_data = plan(args)
    if args.dry_run:
        return plan_data
    import torch
    from unittest.mock import patch
    from semcache.models.loader import load_model
    from semcache.models.lora_fixtures import create_controlled_users
    from semcache.models.model_adapter import OPTModelAdapter
    from semcache.semantic.encoder import TinyBERTSemanticEncoder
    from semcache.utils.seed import seed_everything
    output = Path(args.output_dir).resolve()
    if output != OUTPUT.resolve() and not output.is_relative_to(OUTPUT.resolve()):
        raise ValueError('C5 outputs must remain under results/cachegen/c5/e2e_smoke')
    if (output/'manifest.json').exists():
        raise ValueError('C5 output exists; choose a new subdirectory')
    if not str(args.device).startswith('cuda:') or not torch.cuda.is_available():
        raise ValueError('Real C5 requires SERAPH CUDA')
    if args.max_new_tokens < 1 or args.max_new_tokens > 16:
        raise ValueError('C5 greedy generation is limited to 1..16 tokens')
    seed_everything(args.seed)
    model, tokenizer, model_metadata = load_model(dict(name=MODEL_ID, tokenizer=MODEL_ID,
        revision=MODEL_REVISION, tokenizer_revision=MODEL_REVISION, dtype='float16',
        device=args.device, local_files_only=True, attention_implementation='eager'))
    if (model_metadata['resolved_model_revision'] != MODEL_REVISION or
            model_metadata['resolved_tokenizer_revision'] != MODEL_REVISION or
            model.config.num_hidden_layers != 32 or model.config.hidden_size != 2560):
        raise ValueError('Loaded OPT/tokenizer revision or dimensions differ from M9-B')
    model, lora_metadata = create_controlled_users(model)
    adapter = OPTModelAdapter(model)
    encoder = TinyBERTSemanticEncoder(model_id=ENCODER_ID, revision=ENCODER_REVISION,
        device=args.semantic_device, dtype='float32', local_files_only=True)
    if (encoder.metadata.get('model_id') != ENCODER_ID or
            encoder.metadata.get('resolved_revision') != ENCODER_REVISION):
        raise ValueError('Live TinyBERT checkpoint/revision differs from M9-B preparation')
    codec = FrozenK20V16Codec(args.profile_path, quantization_device=args.device,
        decode_device=args.device, instrument=True, coder_backend=args.coder_backend)
    if codec.profile.sha256 != PROFILE_SHA256:
        raise ValueError('Frozen CDF profile changed')
    if any(plan_data['preparation'][name] is None for name in ('snips', 'multiwoz')):
        raise ValueError('Real C5 requires both M9-B preparation sidecar manifests')
    for name in ('snips', 'multiwoz'):
        provenance_device = plan_data['preparation'][name]['metadata']['semantic_encoder'].get('device')
        if provenance_device != args.semantic_device:
            raise ValueError(f'{name} semantic preparation used {provenance_device}; '
                             f'C5 --semantic-device={args.semantic_device} differs')
    snapshots = {name: semantic_snapshots(plan_data['rows'][name],
        [e for e in plan_data['selected'] if e['dataset'] == name], encoder,
        batch_size=plan_data['preparation'][name]['batch_size'])
        for name in ('snips', 'multiwoz')}
    output.mkdir(parents=True, exist_ok=True)
    manifest = dict(stage='C5-SEMCACHE-E2E', status='RUNNING', git=git_state(),
        model=MODEL_ID, model_revision=MODEL_REVISION, tokenizer=MODEL_ID,
        tokenizer_revision=MODEL_REVISION, workload_paths=plan_data['workload_paths'],
        workload_sha256=plan_data['workload_hashes'],
        preparation_manifest_sha256=plan_data['preparation_manifest_hashes'],
        compression_policy=POLICY.name, transform='B2_ANCHOR_MOD_RESIDUAL_KV',
        profile=plan_data['profile'], coder_backend=args.coder_backend,
        shared_profile_bytes=codec.profile_bytes, profile_bytes_charged_per_entry=0,
        selected_episodes=plan_data['selected'], source_ne_target_asserted=True,
        semantic_match_provenance='measured M9-B TinyBERT assignments verified by live TinyBERT/IntentClusterer prefix replay',
        physical_safety_contract=SAFETY_CONTRACT,
        m9b_strict_safety_role='diagnostic_only_not_an_execution_gate',
        m9b_strict_safety_description=('M9-B safety_eligible additionally requires same user, '
            'adapter, full prompt, position and external fixture evidence; C5 does not equate '
            'it with the physical engine exact-w3 rule'),
        logical_capacity_bytes=LOGICAL_CAPACITY_BYTES,
        episode_scope='cold source then distinct later target; intervening requests update semantic state only',
        users='M9-B deterministic two-user assignment mapped to controlled untrained PEFT user_a/user_b',
        lora_metadata=lora_metadata, device=args.device, semantic_device=args.semantic_device,
        software=dict(python=platform.python_version(), torch=torch.__version__,
                      transformers=model_metadata['transformers_version']),
        max_new_tokens=args.max_new_tokens, cdf_fit_calls=0,
        completed_episodes=0)
    write_json(output/'manifest.json', manifest)
    episode_rows, paired = [], []
    actual_encodes = 0
    try:
        with patch.object(fmt, 'fit', side_effect=AssertionError('C5 must not fit CDFs')):
            for index, episode in enumerate(plan_data['selected'], 1):
                rows = plan_data['rows'][episode['dataset']]
                results = {mode: _run_mode(episode, rows, model, tokenizer, adapter, encoder,
                    snapshots[episode['dataset']], codec, mode, args.max_new_tokens, args.device)
                           for mode in MODES}
                assert_logical_pair([results[MODES[0]][x] for x in ('source', 'target')],
                                    [results[MODES[1]][x] for x in ('source', 'target')])
                if (set(results[MODES[0]]['cache'].entries) !=
                        set(results[MODES[1]]['cache'].entries) or
                        results[MODES[0]]['cache'].logical_cache_bytes !=
                        results[MODES[1]]['cache'].logical_cache_bytes):
                    raise ValueError('RAW/COMPRESSED final logical residency differs')
                signatures = verify_pair(episode, results[MODES[0]]['target'],
                    results[MODES[1]]['target'], results[MODES[0]]['cache'],
                    results[MODES[1]]['cache'], args.device)
                observed_strict = m9b_strict_diagnostic(
                    rows[episode['source_index']], rows[episode['target_index']],
                    episode['source_user'], episode['target_user'], signatures)
                if observed_strict != episode['m9b_strict_safety']:
                    raise ValueError('M9-B strict diagnostic changed between discovery and execution')
                flat, comparison = _episode_rows(episode, results, signatures)
                actual_encodes += flat[1]['admissions']
                if (flat[1]['admissions'] > episode['expected_compressed_admissions'] or
                        actual_encodes > args.max_encodes):
                    raise ValueError('C5 compressed-admission runtime guard exceeded')
                episode_rows.extend(flat)
                paired.append(comparison)
                manifest['completed_episodes'] = index
                write_json(output/'manifest.json', manifest)
                print(f'[{index}/{len(plan_data["selected"])}] {episode["episode_id"]} '
                      f'hits={len(signatures)} compressed_encodes={flat[1]["admissions"]}', flush=True)
        write_csv(output/'episode_results.csv', episode_rows, episode_rows[0].keys())
        write_csv(output/'paired_results.csv', paired, paired[0].keys())
        write_json(output/'summary.json', summarize(paired, episode_rows))
        manifest.update(status='COMPLETED', output_sha256={name: file_hash(output/name)
            for name in ('episode_results.csv', 'paired_results.csv', 'summary.json')})
    except BaseException as exc:
        manifest.update(status='FAILED', reason=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        write_json(output/'manifest.json', manifest)
    return summarize(paired, episode_rows)
