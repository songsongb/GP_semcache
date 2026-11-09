"""Phase 2 TEST_FIXTURE_ONLY data; no fixture is a real sharing/GPU finding."""
from collections import Counter
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import os
import random
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from semcache.edgelora.communication import digest
from semcache.edgelora.r5_multi_artifacts import registry_for
from semcache.edgelora.r5_i5_workload import select, pack_groups, requests, prepare_prompt
from semcache.edgelora.r5_i5_contracts import request
from semcache.edgelora.r5_continuous_natural import build_sessions, runtime_task, CONDITIONS, CPUProbe, static_analysis
from semcache.edgelora import r5_continuous_natural_phase2 as api
from semcache.edgelora.r5_continuous_natural_sharing import (
    CATEGORIES, SCOPES, analyze, pair_windows, relation, token_regions, window_region)
from semcache.edgelora.r5_subsequence_natural_io import verify, seal, put
from test_r5_continuous_natural import identities, reseal

ROOT = Path(__file__).resolve().parents[1]


class FixtureTokenizer:
    """Reversible character offsets, solely TEST_FIXTURE_ONLY; never an OPT substitute."""
    def encode(self, text, **kwargs): return list(map(ord, text))
    def decode(self, ids, **kwargs): return ''.join(map(chr, ids))
    def __call__(self, text, **kwargs):
        return dict(input_ids=self.encode(text), offset_mapping=[(n, n+1) for n in range(len(text))])


def fixture_inputs(count=43, context_multiplier=1):
    tok = FixtureTokenizer(); users = registry_for(['Adapter_A', 'Adapter_B', 'Adapter_C'], 'expanded')
    datasets = {}; records = {}; blocks = []; special = dict(bos_token_id=2, eos_token_id=2, pad_token_id=1)
    for name in api.DATASETS:
        samples = [dict(id=f'TEST_FIXTURE_ONLY_{name}_{n}', dataset=name, version='LongBench-v1',
                        group_id=f'group_{n}', context=(f'Document fixture {n}. Shared document words. '*context_multiplier),
                        question=f'Question fixture {n}?') for n in range(count)]
        datasets[name] = samples
        budget = 96 if name == 'multi_news' else 32
        selected, report = select(samples, tok, count=10, seed=42, context_limit=2048,
                                  budget=budget, allow_truncation=True, input_limit=512)
        records[name] = dict(selections=[dict(report, track='NATURAL', prefix=None)], prepared_path='TEST_FIXTURE_ONLY')
        for repetition in range(2):
            offset = 0
            for cohort in pack_groups(deepcopy(selected), 5):
                chosen = min(range(len(cohort)), key=lambda n: digest([42, cohort[n]['sample']['group_id']]))
                for n, p in enumerate(cohort): p['strict'] = n == chosen
                tasks = requests(cohort, track='NATURAL', prefix=None, offset=offset, special=special)
                blocks.append(dict(dataset=name, track='NATURAL', priming_requests=0, repetition=repetition,
                                   routing_seed=42, tasks=tasks))
                offset += len(cohort)
    w = reseal(dict(blocks=blocks, seed=42, context_limit=2048, special_tokens=special,
                    block_size=5, clusters=1, TEST_FIXTURE_ONLY=True))
    return w, records, build_sessions(w, users), datasets, tok, users


class Phase2(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.w, cls.records, cls.baseline, cls.datasets, cls.tok, cls.users = fixture_inputs()
        cls.sessions, cls.selection = api.expand(cls.w, cls.records, cls.baseline, cls.datasets, cls.tok,
                                               input_limit=512, context_truncation='allow')

    def test_counts_and_exact_phase1_prefix_provenance(self):
        self.assertEqual(len(self.sessions), 3)
        self.assertEqual(sum(len(s['invocations']) for s in self.sessions), 150)
        for s, old in zip(self.sessions, self.baseline):
            self.assertEqual(s['invocations'][:20], old['invocations'])
            counts = Counter(i['original_sample_id'] for i in s['invocations'])
            self.assertEqual(Counter(counts.values()), {1: 30, 2: 10})
            self.assertEqual(len(s['checkpoints']), 10)
            for i in s['invocations'][20:]:
                self.assertIsNone(i['original_block_index']); self.assertIsNone(i['original_block_id'])
                self.assertTrue(i['extension_checkpoint_id'].startswith('D5C2_EXTENSION_'))
                self.assertEqual(i['task']['role'], 'SCORED_TARGET'); self.assertTrue(i['task']['scored'])
            ids = [c['checkpoint_id'] for c in s['checkpoints']]
            self.assertEqual(len(ids), len(set(ids)))

    def test_deterministic_next_eligible_selection_and_no_overlap_selection(self):
        actual, _ = api.expand(self.w, self.records, self.baseline, self.datasets, self.tok,
                               input_limit=512, context_truncation='allow')
        self.assertEqual(actual, self.sessions)
        for s in actual:
            selected, _ = select(self.datasets[s['dataset']], self.tok, count=40, **self.selection[s['dataset']]['settings'])
            self.assertEqual([i['original_sample_id'] for i in s['invocations'][20:]], [p['sample']['id'] for p in selected[10:]])

    def test_shortage_missing_invalid_recipe_and_dataset_fail_closed(self):
        for size in (39, 10):
            w, d, b, rows, tok, _ = fixture_inputs(size)
            with self.assertRaises(api.Blocked): api.expand(w, d, b, rows, tok, input_limit=512, context_truncation='allow')
        rows = deepcopy(self.datasets); rows['qasper'].append(rows['qasper'][0])
        with self.assertRaises(ValueError): api.expand(self.w, self.records, self.baseline, rows, self.tok, input_limit=512, context_truncation='allow')
        with self.assertRaises(ValueError): api.expand(self.w, self.records, self.baseline, self.datasets, self.tok, input_limit=128, context_truncation='allow')
        records = deepcopy(self.records); records['qasper']['selections'][0]['eligible'] += 1
        with self.assertRaises(ValueError): api.expand(self.w, records, self.baseline, self.datasets, self.tok, input_limit=512, context_truncation='allow')

    def test_every_preserved_invocation_is_checked_against_original(self):
        baseline = deepcopy(self.baseline); baseline[0]['invocations'][15]['task']['input_text'] = 'TEST_FIXTURE_MUTATION'
        with self.assertRaisesRegex(ValueError, 'Phase 1 prefix'):
            api.expand(self.w, self.records, baseline, self.datasets, self.tok, input_limit=512, context_truncation='allow')

    def test_unique_ids_and_unchanged_strict_request_schema(self):
        ids = []; pair_ids = []
        for s in self.sessions:
            for n, i in enumerate(s['invocations']):
                pair_ids.append(i['invocation_pair_id'])
                self.assertEqual(i['session_sequence_number'], n)
                for c in CONDITIONS:
                    t = runtime_task(i, c); ids.append(t['request_id'])
                    req = request(t, t['token_ids'], t['policy'], identities()[t['adapter_id']], fixture=True)
                    self.assertEqual(req['input_text'], i['task']['input_text'])
                    self.assertEqual(req['token_ids'], i['task']['token_ids'])
        self.assertEqual(len(ids), len(set(ids))); self.assertEqual(len(ids), 300)
        self.assertEqual(len(pair_ids), len(set(pair_ids))); self.assertEqual(len(pair_ids), 150)

    def test_cpu_lifecycle_no_reset_and_no_inference(self):
        original = deepcopy(self.sessions)
        with patch('semcache.edgelora.r5_subsequence_physical.PhysicalOwner.begin', side_effect=AssertionError('not CPU probe')):
            r = api.lifecycle(self.sessions, self.users, identities())
        self.assertEqual(r['fresh_initializations'], 6); self.assertEqual(r['checkpoints'], 60)
        self.assertEqual(r['internal_cache_resets'], 0); self.assertEqual(self.sessions, original)
        self.assertEqual(r['actual_gpu_cache_continuity'], 'NOT_MEASURED')

    def test_real_arrival_clock_and_history_persist_through_extension_checkpoints(self):
        outer = self
        class Arrivals(CPUProbe):
            def execute(p, state, adapter, invocation, task, condition):
                if adapter.owner is None: return
                o = adapter.owner; expected = getattr(adapter, 'expected', 0)
                outer.assertEqual(o.cache.now, expected)
                keys = [o.bridge.matcher.key(0, w) for w in o.bridge.extractor.extract(task['token_ids'])]
                o.bridge.metrics.arrive(keys); adapter.expected = expected+1
                outer.assertEqual(o.cache.now, adapter.expected)
                outer.assertFalse(o.cache.entries)
            def close(p, state):
                if any(a.owner for a in state.adapters.values()):
                    outer.assertEqual(sum(a.owner.cache.now for a in state.adapters.values()), 50)
                super().close(state)
        with patch.object(api, 'CPUProbe', Arrivals): api.lifecycle(self.sessions, self.users, identities())

    def test_pair_category_counts_against_cartesian_oracle(self):
        rng = random.Random(13)
        for _ in range(30):
            a = [rng.randrange(3) for _ in range(9)]; b = [rng.randrange(3) for _ in range(9)]
            labels = ['DOCUMENT_CONTEXT']*9
            cells = pair_windows(a, b, labels, labels); expected = Counter(); actual = Counter()
            prefix = 0
            for x, y in zip(a, b):
                if x != y: break
                prefix += 1
            for sp in range(7):
                for tp in range(7):
                    if a[sp:sp+3] == b[tp:tp+3]:
                        c = CATEGORIES[0 if sp == tp and tp+3 <= prefix else 1 if sp == tp else 3 if tp+3 <= prefix else 2]
                        expected[c] += 1
            for (c, _, _), (n, _) in cells.items(): actual[c] += n
            self.assertEqual(actual, expected)

    def test_replay_relationships_and_three_denominators(self):
        session = deepcopy(self.sessions[0])
        for n, i in enumerate(session['invocations']):
            i['task']['token_ids'] = [1,2,3,4,n%10,6]
        regions = [dict(token_labels=['UNKNOWN']*6, status='UNKNOWN') for _ in session['invocations']]
        report = analyze(session, self.users, regions)
        self.assertEqual([report['views'][s]['denominators']['ordered_request_pairs'] for s in SCOPES], [1225,1215,780])
        self.assertEqual(report['views'][SCOPES[0]]['denominators']['EXPERIMENT_REPLAY'], 10)
        self.assertEqual(report['views'][SCOPES[1]]['denominators'].get('EXPERIMENT_REPLAY', 0), 0)
        a = deepcopy(session['invocations'][0]['task']); b = deepcopy(a); b['sample_id'] = 'DISTINCT_TEST_SAMPLE'
        self.assertEqual(relation(a,b), 'NATURALLY_IDENTICAL_DISTINCT_SAMPLE')
        b['input_text'] += '?'; self.assertEqual(relation(a,b), 'SAME_ORIGINAL_CONTEXT_DIFFERENT_REQUEST')
        b['prepared']['sample']['context'] += 'other'; self.assertEqual(relation(a,b), 'DIFFERENT_CONTEXT_DIFFERENT_REQUEST')
        old = static_analysis(session, self.users)['occurrence_counts']
        self.assertEqual(old, {c: report['views'][SCOPES[0]]['axes']['aggregate']['all'][c]['raw_matching_occurrence_pairs'] for c in CATEGORIES})

    def test_coverage_denominators_and_sharing_rejection(self):
        session = deepcopy(self.sessions[0]); session['invocations'] = session['invocations'][:2]
        for i in session['invocations']: i['task'].update(token_ids=[1]*5, user_id='User_01', adapter_id='Adapter_A')
        session['invocations'][1]['task'].update(user_id='User_11', adapter_id='Adapter_B')
        report = analyze(session, self.users, [dict(token_labels=['DOCUMENT_CONTEXT']*5, status='VERIFIED_OFFSETS')]*2)
        axes = report['views'][SCOPES[0]]['axes']['aggregate']; a = axes['all'][CATEGORIES[0]]; d = axes['all'][CATEGORIES[3]]
        self.assertEqual((a['raw_matching_occurrence_pairs'],d['raw_matching_occurrence_pairs']), (3,6))
        self.assertEqual(a['unique_target_window_positions'], 3)
        self.assertEqual(a['nonoverlapping_target_token_coverage'], 3)
        self.assertEqual(a['target_token_union_coverage'], 5)
        self.assertEqual(axes['authorized'][CATEGORIES[0]]['raw_matching_occurrence_pairs'], 0)
        self.assertEqual(axes['rejected:CROSS_ADAPTER_REUSE_FORBIDDEN'][CATEGORIES[0]]['raw_matching_occurrence_pairs'],3)

    def test_template_context_offsets_truncation_and_unknown(self):
        p = self.baseline[0]['invocations'][0]['task']['prepared']
        r = token_regions(p, self.tok)
        self.assertEqual(r['status'], 'VERIFIED_OFFSETS')
        self.assertIn('DOCUMENT_CONTEXT', r['region_tokens']); self.assertIn('TEMPLATE_INSTRUCTION', r['region_tokens'])
        sample = deepcopy(p['sample']); sample['context'] *= 40
        truncated = prepare_prompt(sample, self.tok, 2048, 32, allow_truncation=True, input_limit=512)
        r = token_regions(truncated, self.tok)
        self.assertEqual(r['status'], 'VERIFIED_OFFSETS'); self.assertIn('TEMPLATE_OMISSION', r['region_tokens'])
        class Slow(FixtureTokenizer):
            def __call__(self, *args, **kwargs): raise NotImplementedError('TEST_FIXTURE_NO_OFFSETS')
        unknown = token_regions(p, Slow())
        self.assertEqual(set(unknown['token_labels']), {'UNKNOWN'})
        self.assertEqual(window_region(['TEMPLATE_FORMAT','DOCUMENT_CONTEXT','DOCUMENT_CONTEXT'],0),'MIXED_BOUNDARY')
        broken = deepcopy(p); broken['token_ids'][0] += 1
        self.assertEqual(token_regions(broken, self.tok)['status'], 'UNKNOWN')

    def test_template_exclusion_requires_both_windows_verified_document(self):
        session = deepcopy(self.sessions[0]); session['invocations'] = session['invocations'][:2]
        for i in session['invocations']: i['task']['token_ids'] = [1,2,3,4,5,6]
        regions = [dict(token_labels=['TEMPLATE_INSTRUCTION']*3+['DOCUMENT_CONTEXT']*3, status='VERIFIED_OFFSETS')]*2
        a = analyze(session, self.users, regions)['views'][SCOPES[1]]['axes']['aggregate']
        self.assertEqual(a['all'][CATEGORIES[0]]['raw_matching_occurrence_pairs'],4)
        self.assertEqual(a['verified_context_to_context'][CATEGORIES[0]]['raw_matching_occurrence_pairs'],1)
        regions[1] = dict(token_labels=['UNKNOWN']*6,status='UNKNOWN')
        a = analyze(session, self.users, regions)['views'][SCOPES[1]]['axes']['aggregate']
        self.assertEqual(a['verified_context_to_context'][CATEGORIES[0]]['raw_matching_occurrence_pairs'],0)
        self.assertEqual(a['unknown_region'][CATEGORIES[0]]['raw_matching_occurrence_pairs'],4)

    def test_extended_resource_guard_failures_are_diagnostic(self):
        tasks = [deepcopy(i['task']) for i in self.sessions[0]['invocations']]
        for t in tasks: t['token_ids'] = list(range(512)); t['policy']['max_new_tokens'] = 96
        r = api.resource_assessment(tasks)
        self.assertFalse(r['limit_checks']['file_count']); self.assertFalse(r['limit_checks']['condition_evidence'])
        self.assertEqual(r['per_adapter_ud_limit_bytes'],4*2**30); self.assertEqual(len(tasks),50)

    def test_missing_assets_sealed_blocked_and_cpu_environment(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, CUDA_VISIBLE_DEVICES=''):
            root = Path(tmp)
            r = api.prepare(root/'missing',root/'multi',root/'phase1',512,'allow',root/'out')
            self.assertEqual(r['status'],'BLOCKED'); verify(root/'out')
            self.assertFalse((root/'out/phase2_plan.json').exists())
            with self.assertRaises(ValueError): api.prepare(root/'missing',root/'multi',root/'phase1',512,'allow',root/'out')
        with patch.dict(os.environ, CUDA_VISIBLE_DEVICES='0'):
            with self.assertRaises(ValueError): api.cpu_validate('missing','unused')

    def test_source_dataset_protected_even_when_original_loader_fails(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, CUDA_VISIBLE_DEVICES=''):
            root = Path(tmp); (root/'original').mkdir(); (root/'dataset').mkdir()
            put(root/'original/i5_dataset_manifest.json',{'qasper':{'prepared_path':str(root/'dataset')}})
            with patch.object(api,'load_original',side_effect=ValueError('invalid original fixture')):
                with self.assertRaises(ValueError):
                    api.prepare(root/'original',root/'multi',root/'phase1',512,'allow',root/'dataset/forbidden')
            self.assertFalse((root/'dataset/forbidden').exists())

    def test_complete_preparation_load_and_validation_with_explicit_fixture_plumbing(self):
        # Mock ONLY unavailable real-asset/native metadata boundaries. Production
        # loader is not relaxed; these fixture outputs are temporary and deleted.
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, CUDA_VISIBLE_DEVICES=''):
            root = Path(tmp)
            for name in ('original','multi','phase1'): (root/name).mkdir()
            loaded = (self.w,self.records,{'TEST_FIXTURE_ONLY':True},{},{},{},
                      {a:{} for a in identities()},self.users,SimpleNamespace(tokenizer=self.tok,identity={'TEST_FIXTURE_ONLY':True}))
            original = (loaded,self.baseline,self.datasets,{'TEST_FIXTURE_ONLY':True})
            def predicted(t,*args): return {'physical_identity':identities()[t['adapter_id']]}
            with patch.object(api,'load_original',return_value=original), \
                 patch.object(api.Evidence,'verify',return_value={'TEST_FIXTURE_ONLY':True}), \
                 patch('semcache.edgelora.r5_i5r3b_transport.predicted_request',side_effect=predicted), \
                 patch.object(api,'transport',return_value={'TEST_FIXTURE_ONLY':True}):
                r = api.prepare(root/'original',root/'multi',root/'phase1',512,'allow',root/'plan')
                self.assertEqual(r['status'],'PASS_CPU_PREPARED',r)
                self.assertEqual(api.read_sessions(root/'plan'),self.sessions)
                with self.assertRaises(ValueError): api.cpu_validate(root/'plan',root/'original/forbidden')
                self.assertFalse((root/'original/forbidden').exists())
                v = api.cpu_validate(root/'plan',root/'audit')
                self.assertEqual(v['status'],'PASS',v); self.assertFalse(v['future_gpu_ready'])
                self.assertEqual(v['gpu_physical_hits'],'NOT_MEASURED'); verify(root/'audit')
                (root/'plan/qasper/invocations/020.json').write_text('{}')
                with self.assertRaises(ValueError): api.load_plan(root/'plan')

    def test_real_loader_rejects_nonpassing_phase1_and_cli_rejects_gpu_action(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ('original','multi','phase1'): (root/name).mkdir()
            put(root/'phase1/phase1_validation.json',dict(schema='D5_CONTINUOUS_NATURAL_CPU_V1',phase1_status='BLOCKED',real_data_validation='BLOCKED'))
            seal(root/'phase1')
            with self.assertRaisesRegex(ValueError,'real Phase 1 PASS'):
                api.load_original(root/'original',root/'multi',root/'phase1')
        p = subprocess.run(['python3',str(ROOT/'scripts/155_r5_continuous_natural_phase2.py'),'run'],capture_output=True,text=True)
        self.assertNotEqual(p.returncode,0); self.assertIn('invalid choice',p.stderr)


if __name__ == '__main__': unittest.main()
