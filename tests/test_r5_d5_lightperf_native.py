"""Required Tiny OPT CPU parity. Missing native packages FAIL, never SKIP.

All assets here are synthetic TEST_FIXTURE_ONLY Tiny OPT/LoRA inputs. No model
download or GPU operation. Production CLI cannot select this test setup.
"""
from copy import deepcopy
from pathlib import Path
import hashlib
import importlib.util
import multiprocessing
import os
import subprocess
import tempfile
import time
import unittest
from unittest.mock import patch

from semcache.edgelora.communication import connect, digest
from semcache.edgelora.r5_d5_lightperf_workload import CONDITIONS, knobs
from semcache.edgelora.r5_d5_lightperf_io import SummaryWriter, read_json, verify


class ProcessHandle:
    def __init__(self, process):
        self.process = process
    def poll(self):
        return self.process.exitcode
    def wait(self, timeout):
        self.process.join(timeout)
        if self.process.is_alive():
            raise subprocess.TimeoutExpired('CPU test UD', timeout)
        return self.process.exitcode
    def terminate(self):
        self.process.terminate()
    def kill(self):
        self.process.kill()


def native_ud(c, output, matrix_values, registry, performance_knobs):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    from semcache.edgelora.user_device import configure_cpu
    from semcache.edgelora.r5_d5_lightperf_runtime import serve_ud_session
    import torch
    cpu = configure_cpu(True)
    matrices = {key: torch.tensor(value, dtype=torch.float32).reshape((8, 4) if key[2] == 'A' else (4, 8))
                for key, value in matrix_values.items()}
    Path(output).mkdir()
    writer = SummaryWriter(output)
    p = dict(resources=dict(worker_deadline_seconds=180), performance_knobs=performance_knobs,
             source={'TEST_FIXTURE_ONLY': True})
    serve_ud_session(c, p, {}, {}, writer, cpu, matrices, registry)


class NativeParity(unittest.TestCase):
    def setUp(self):
        self.assertEqual(os.environ.get('CUDA_VISIBLE_DEVICES'), '', 'native parity must hide CUDA')
        missing = [n for n in ('torch', 'transformers', 'numpy') if importlib.util.find_spec(n) is None]
        self.assertFalse(missing, 'BLOCKED_NATIVE_DEPENDENCIES: ' + ','.join(missing))
        import torch
        self.assertFalse(torch.cuda.is_initialized())
        torch.set_num_threads(1)

    def run_case(self, condition, *, strict=False, performance_knobs=None, corrupt=False, expire=False, eos=None):
        import torch
        from transformers import OPTConfig, OPTForCausalLM
        from semcache.models.model_adapter import OPTModelAdapter
        from semcache.semantic.encoder import ControlledEncoder
        from semcache.edgelora.r5_semantic_index import SemanticIndex
        from semcache.edgelora.r5_subsequence_physical import PhysicalOwner, tensor_proof
        from semcache.edgelora.r5_subsequence_integrity import SCOPED
        from semcache.edgelora.r5_d5_lightperf_runtime import execute_stream, wait_file, reap
        from semcache.edgelora.r5_d5_lightperf_receipts import WireSummary, PolicyCounter, wire_equal
        from semcache.edgelora.r5_d5_lightperf_emulation import LightChannel
        from semcache.edgelora.r5_d5_lightperf_timing import generate_native, LightProfiler
        from semcache.edgelora.r5_subsequence_generation import generate as strict_generate
        from semcache.edgelora.r3a_artifacts import unpack
        from semcache.edgelora.r5_i5_contracts import namespace, request
        from semcache.edgelora.r5_multi_artifacts import registry_for, runtime_view
        from semcache.edgelora.r4c_workload import policy
        from semcache.edgelora.r4c_evidence import Evidence
        performance_knobs = knobs() if performance_knobs is None else performance_knobs
        torch.manual_seed(42)
        cfg = OPTConfig(vocab_size=32, hidden_size=4, word_embed_proj_dim=4, ffn_dim=16,
                        num_hidden_layers=2, num_attention_heads=2, max_position_embeddings=64,
                        dropout=0., attention_dropout=0.)
        cfg._attn_implementation = 'eager'
        model = OPTForCausalLM(cfg).eval()
        adapter = OPTModelAdapter(model)
        registry = registry_for(['Adapter_A', 'Adapter_B', 'Adapter_C'], 'expanded')
        pm = dict(identity=dict(model='TEST_FIXTURE_ONLY', base_revision='fixture', base_weight_sha256='a' * 64,
                                dtype='float32', hidden_dtype='float32', layers=2, hidden_size=4))
        tasks = []
        pattern = [('Adapter_A', 'User_01', [1, 2, 3, 4, 5, 6]), ('Adapter_B', 'User_11', [1, 2, 3, 4, 5, 6]),
                   ('Adapter_A', 'User_02', [1, 2, 3, 4, 5, 6]), ('Adapter_B', 'User_12', [1, 2, 3, 4, 5, 6]),
                   ('Adapter_A', 'User_01', [9, 2, 3, 4, 5, 6]), ('Adapter_A', 'User_01', [0, 1, 2, 3, 4, 5, 6])]
        special = dict(bos_token_id=2, eos_token_id=2 if eos is None else eos, pad_token_id=1)
        for i, (aid, user, ids) in enumerate(pattern):
            tasks.append(dict(request_id=f'TEST_{i:03d}', sample_id=f'CPU_NATIVE_SAMPLE_{i}', user_id=user, adapter_id=aid,
                              input_text='native CPU test', dataset='qasper', token_ids=ids, request_sequence_number=i,
                              policy=policy(special, 'fixed' if eos is None else 'quality', 3)))
        weights_before = [tensor_proof(x) for x in model.parameters()]
        sessions, logits, physicals, snapshots = {}, {}, {}, {}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            es = root / 'es'
            es.mkdir()
            writer = SummaryWriter(es)
            deadline = time.monotonic() + 180
            try:
                for aid in ('Adapter_A', 'Adapter_B'):
                    spec = dict(adapter_id=aid, matrix_sha256=('b' if aid == 'Adapter_A' else 'c') * 64,
                                rank=8, matrix_dtype='float32')
                    identity = namespace(runtime_view(pm, spec), {'TEST_FIXTURE_ONLY': 'integer'}, 'cpu', 'CPU_FIXTURE', {'test': True})
                    subset = [t for t in tasks if t['adapter_id'] == aid]
                    reqs = [request(t, t['token_ids'], t['policy'], identity, fixture=True) for t in subset]
                    values = {(layer, role, part): [(.01 if aid == 'Adapter_A' else .02)] * 32
                              for layer in range(2) for role in 'qkv' for part in ('A', 'B')}
                    out = root / aid
                    c = dict(identity=identity, session='native_cpu_test', requests=reqs, deadline_monotonic=deadline)
                    process = multiprocessing.get_context('spawn').Process(target=native_ud,
                                 args=(c, str(out), values, registry, performance_knobs))
                    process.start()
                    handle = ProcessHandle(process)
                    s = dict(process=handle, ud_out=out, identity=identity, ids=[r['request_id'] for r in reqs],
                             completed=0, stopped=False, owner=None, channel=None, policy=PolicyCounter(), wire=WireSummary([r['request_id'] for r in reqs]))
                    sessions[aid] = s
                    ready = wait_file(out / 'ready.json', handle, min(deadline, time.monotonic() + 45))
                    channel = LightChannel(connect(ready['port']), c['session'], spec['matrix_sha256'],
                                           timeout=20, bandwidth_mbps=performance_knobs['bandwidth_mbps'], trace=s['wire'])
                    channel.condition_deadline = deadline
                    s.update(channel=channel, index=None if condition == CONDITIONS[0] else
                             SemanticIndex(ControlledEncoder({'native CPU test': [1., 0.]}), clusters=1))
                    if condition != CONDITIONS[0]:
                        evidence = None
                        if strict:
                            d = root / (aid + '_strict')
                            d.mkdir()
                            evidence = Evidence(d)
                        s['owner'] = PhysicalOwner(identity, registry, capacity=1152, evidence=evidence,
                                                   validation_mode=SCOPED, trace=s['policy'])
                        s['owner'].bridge.pbr_interval = 2  # explicit test-only PBR exercise

                def generate(*args, **kwargs):
                    req = args[3]
                    owner = args[5]
                    rid = req['request_id']
                    if corrupt and rid == tasks[2]['request_id']:
                        entry = next(iter(owner.cache.entries.values()))
                        with torch.inference_mode():
                            entry.tensors[0][0].data.add_(1.)
                    if expire and rid == tasks[2]['request_id']:
                        kwargs['deadline'] = time.monotonic() - 1
                    if strict:
                        translated = dict(kwargs)
                        translated.pop('deadline')
                        translated['condition'] = 'FULL_RECOMPUTE' if condition == CONDITIONS[0] else 'SEMCACHE_SUBSEQUENCE'
                        profiler = LightProfiler()
                        r = strict_generate(*args, **translated, evidence=None if owner is None else owner.evidence, profiler=profiler)
                        shape = r['logits_shape']
                        values = unpack(r.pop('logits_blob'), 'float32')
                        logits[rid] = [tuple(values[i:i + shape[2]]) for i in range(0, len(values), shape[2])]
                        r.update(row_summary=dict(fresh_qkv_rows=3 * sum(x['fresh_rows_per_role'] for x in r['sparse_receipts']),
                                                  skipped_qkv_rows=3 * sum(x['skipped_rows_per_role'] for x in r['sparse_receipts'])),
                                 first_available_ns=r['steps'][0]['available_ns'], condition=condition, components=r['diagnostics'],
                                 matching_candidate_count=None if r['matching'] is None else len(r['matching']['candidates']),
                                 generated_token_ids_sha256=digest(r['generated_token_ids']),
                                 generated_text_sha256=hashlib.sha256(r['generated_text'].encode()).hexdigest())
                    else:
                        logits[rid] = []
                        r = generate_native(*args, **kwargs, cpu_logit_observer=lambda step, history, values: logits[rid].append(values))
                    physicals[rid] = r['completion']['physical']
                    impact = r['completion']['impact']
                    if impact is not None:
                        self.assertEqual(len(impact['records']), 2)
                        self.assertTrue(all(v > 0 for layer in impact['records'] for v in layer['span_values'].values()))
                    if owner is not None:
                        snapshots[rid] = sorted((repr(key), entry.frequency, entry.impact,
                                                  tuple(tensor_proof(t)['sha256'] for layer in entry.tensors.values() for t in layer))
                                                 for key, entry in owner.cache.entries.items())
                    return r

                if corrupt or expire:
                    with self.assertRaises(ValueError if corrupt else TimeoutError):
                        execute_stream(tasks, sessions, model, adapter, self.tokenizer(), condition, writer, deadline,
                            request_fn=lambda t, ids, p, ident: request(t, ids, p, ident, fixture=True), generate_fn=generate,
                            score_fn=lambda t, text: ({'metric': 'TEST_FIXTURE_ONLY', 'score': 1.}, 'CPU_TEST'))
                    self.assertIsNone(sessions['Adapter_A']['owner'].active)
                    self.assertFalse(sessions['Adapter_A']['owner'].validation.open)
                    return None
                rows = execute_stream(tasks, sessions, model, adapter, self.tokenizer(), condition, writer, deadline,
                    request_fn=lambda t, ids, p, ident: request(t, ids, p, ident, fixture=True), generate_fn=generate,
                    score_fn=lambda t, text: ({'metric': 'TEST_FIXTURE_ONLY', 'score': 1.}, 'CPU_TEST'))
                for aid, s in sessions.items():
                    self.assertTrue(s['stopped'])
                    self.assertEqual(s['process'].wait(15), 0)
                    verify(s['ud_out'])
                    ud = read_json(s['ud_out'] / 'ud.json')
                    self.assertEqual(ud['completed_request_ids'], s['ids'])
                    self.assertTrue(ud['final_bye_sent'])
                    wire_equal(s['wire'].rows['CONTROL'], ud['wire']['CONTROL'])
                self.assertEqual(weights_before, [tensor_proof(x) for x in model.parameters()])
                self.assertFalse(torch.cuda.is_initialized())
                return dict(rows=rows, logits=logits, snapshots=snapshots,
                            policy={aid: dict(s['policy'].counts) for aid, s in sessions.items()}, physicals=physicals)
            finally:
                for s in sessions.values():
                    if s['owner'] is not None:
                        s['owner'].dispose()
                    if s['channel'] is not None:
                        s['channel'].close()
                    reap(s['process'])

    @staticmethod
    def tokenizer():
        class Tokenizer:
            def decode(self, ids, **kwargs):
                return ' '.join(map(str, ids))
        return Tokenizer()

    def test_strict_light_full_exact_approx_logits_tokens_impact_policy_routing(self):
        for condition in CONDITIONS:
            with self.subTest(condition=condition):
                strict, light = self.run_case(condition, strict=True), self.run_case(condition)
                self.assertEqual(strict['logits'], light['logits'], 'full-vocabulary in-memory logits differ')
                self.assertEqual(strict['snapshots'], light['snapshots'])
                self.assertEqual(strict['policy'], light['policy'])
                fields = ('request_id', 'sample_id', 'user_id', 'adapter_id', 'generated_token_ids', 'termination',
                          'fresh_qkv_rows', 'skipped_qkv_rows', 'physical_hit', 'categories', 'cache_occupancy_bytes', 'policy_counts')
                for a, b in zip(strict['rows'], light['rows']):
                    self.assertEqual({k: a[k] for k in fields}, {k: b[k] for k in fields})
                    self.assertEqual(len(b['generated_token_ids']), 3)
                if condition != CONDITIONS[0]:
                    self.assertTrue(light['rows'][2]['physical_hit'])
                    self.assertGreater(sum(x['EVICT'] for x in light['policy'].values()), 0)
                    self.assertGreater(sum(x['PBR'] for x in light['policy'].values()), 0)

    def test_native_bandwidth_and_ud_service_preserve_logits_and_rows(self):
        for condition in CONDITIONS:
            baseline = self.run_case(condition)
            for bandwidth, factor in ((200, 1.), (100, .5), (50, .25)):
                with self.subTest(condition=condition, bandwidth=bandwidth, factor=factor):
                    other = self.run_case(condition, performance_knobs=knobs(bandwidth, factor))
                    self.assertEqual(baseline['logits'], other['logits'])
                    self.assertEqual([r['skipped_qkv_rows'] for r in baseline['rows']], [r['skipped_qkv_rows'] for r in other['rows']])
                    for r in other['rows']:
                        self.assertEqual(r['ud_compute']['requested_factor'], factor)
                        if r['ud_compute']['operations']:
                            self.assertGreater(r['ud_compute']['measured_compute_ms'], 0)

    def test_native_tensor_corruption_aborts_without_fallback(self):
        self.run_case(CONDITIONS[1], corrupt=True)

    def test_native_deadline_aborts_without_admission_or_retry(self):
        self.run_case(CONDITIONS[1], expire=True)

    def test_native_eos_termination_matches_strict(self):
        baseline = self.run_case(CONDITIONS[0])
        eos = baseline['rows'][0]['generated_token_ids'][0]
        strict, light = self.run_case(CONDITIONS[0], strict=True, eos=eos), self.run_case(CONDITIONS[0], eos=eos)
        self.assertEqual(strict['logits'], light['logits'])
        self.assertEqual(light['rows'][0]['termination'], 'EOS')
        self.assertEqual(light['rows'][0]['output_tokens'], 1)


if __name__ == '__main__':
    unittest.main()
