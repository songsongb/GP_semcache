"""Model-free strict profile regression; no downloads, model creation or torch calls."""
import copy
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

from test_m9a_cost_model import DIMS, calibration, profiles
from semcache.metrics.m8 import token_ids_sha256
from semcache.system_cost.es_base_profile import (assert_no_lora, attach_profile, control_plane,
    validate_base_profile, PAPER_SETTINGS, MEASUREMENT_LABEL, PROXY_LABEL)
from semcache.system_cost.base_runtime import span_plan
from semcache.system_cost.model import compare_request
from semcache.system_cost.profiles import pair_key

ROOT = Path(__file__).resolve().parents[1]


def strict_profiles():
    rows = profiles()
    ids = list(range(32))
    for row in rows:
        row.update(PAPER_SETTINGS, requested_prompt_tokens=32, actual_prompt_tokens=32,
            warmup_runs=1, measured_runs=1, warmup_state_semantics='fresh_discarded_trace',
            physical_cache_storage_device='cpu', cache_transfer_path='cpu_to_cuda_on_reuse',
            prompt_token_ids_sha256=token_ids_sha256(ids), semantic_encode_ms=.1,
            cluster_assign_update_ms=.1, subsequence_extract_ms=.1, cache_lookup_ms=.1,
            hit_selection_ms=.1, attention_impact_ms=.1, cache_policy_ms=.1,
            chu_update_ms=.05, pbr_update_ms=.05)
    rows[2]['reuse_block_provenance'] = [dict(destination_start=0, destination_end=16,
        source_start=0, source_end=16, token_ids=ids[:16], cache_key=[0, ids[:16]],
        source_query_id='cold_miss', source_user='user_a', source_adapter='user_a')]
    attached = []
    for row, mode, cost in [(rows[0], 'ES_BASE_NATIVE', 3.), (rows[2], 'ES_BASE_SEMCACHE_REUSE', 2.)]:
        profile = dict(mode=mode, measurement_label=MEASUREMENT_LABEL, provenance='MEASURED',
            lora_absent=True, active_lora_projection_count=0, personalized_output_quality_claimed=False,
            source_pair_key=list(pair_key(row)), fresh_m85_profile_id='fresh-fixture',
            source_m8_sha256='fixture-hash', source_environment_sha256='fixture-env-hash',
            warmup_runs=1, measured_runs=1, dtype=row['dtype'], attention_implementation='eager',
            model_revision=row['model_revision'], tokenizer_revision=row['tokenizer_revision'],
            executed_reused_tokens=row['reused_tokens'], executed_fresh_tokens=row['recomputed_tokens'],
            cache_storage_device='cpu', cache_transfer_path='cpu_to_cuda_on_reuse', prefill_wall_ms=cost,
            cache_dtype=row['dtype'], control_plane_inside_es_compute=False,
            hidden_size=DIMS.hidden_size, layers=DIMS.layers)
        attached.append(attach_profile(row, profile, 'fresh-fixture'))
    return attached[0], rows[1], attached[1]


class StrictESProfileTest(unittest.TestCase):
    def test_no_lora_guard_rejects_disabled_merged_and_other_layer_lora(self):
        linear = SimpleNamespace(weight=object(), in_features=4, out_features=4)
        model = SimpleNamespace(named_modules=lambda: [('q_proj', linear)])
        adapter = SimpleNamespace(layers=[0], projection_modules=lambda _: {'q': linear, 'k': linear, 'v': linear})
        # HF's inactive PeftAdapterMixin method by itself is not a LoRA module.
        model.active_adapters = lambda: []
        self.assertTrue(assert_no_lora(model, adapter)['lora_absent'])
        model.peft_config = {}
        with self.assertRaises(ValueError):
            assert_no_lora(model, adapter)
        del model.peft_config
        linear.lora_A, linear.disable_adapters, linear.merged = {}, True, True
        with self.assertRaises(ValueError):
            assert_no_lora(model, adapter)
        del linear.lora_A
        model.named_modules = lambda: [('ffn', SimpleNamespace(lora_B={}))]
        with self.assertRaises(ValueError):
            assert_no_lora(model, adapter)

    def test_strict_rejects_peft_and_legacy_self_declared_base_fields(self):
        rows = profiles()
        for row in rows:
            row.update(es_base_compute_ms=1., es_base_compute_provenance='MEASURED',
                       es_base_compute_scope='prefill_base_only_excluding_lora_control')
        for policy in ('strict-base-only', 'require-base-only'):
            with self.assertRaisesRegex(ValueError, 'strict-base-only'):
                compare_request(*rows, calibration(), DIMS, es_compute_policy=policy)

    def test_proxy_totals_cannot_be_primary(self):
        result = compare_request(*profiles(), calibration(), DIMS, es_compute_policy='peft-prefill-proxy')
        row = result['row']
        self.assertEqual(row['result_label'], PROXY_LABEL)
        self.assertFalse(row['primary_comparison_eligible'])
        for field in ('edge_lora_total_ms', 'semcache_total_ms'):
            self.assertEqual(row[field+'_label'], PROXY_LABEL)
            self.assertEqual(result['components'][field]['interpretation_label'], PROXY_LABEL)

    def test_base_measured_provenance_and_ud_added_exactly_once(self):
        result = compare_request(*strict_profiles(), calibration(), DIMS)
        row = result['row']
        self.assertTrue(row['primary_comparison_eligible'])
        self.assertEqual(row['es_compute_ms'], 3.)  # PEFT fixture was 10 ms
        self.assertEqual(row['semcache_es_ms'], 2.)  # PEFT fixture was 5 ms
        self.assertEqual(row['es_compute_ms_provenance'], 'MEASURED')
        self.assertEqual(row['es_compute_ms_measurement_label'], MEASUREMENT_LABEL)
        self.assertAlmostEqual(row['ud_lora_ms'], .2*32)
        self.assertAlmostEqual(row['semcache_ud_ms'], .1*32)  # not multiplied by reuse ratio again
        self.assertAlmostEqual(row['edge_lora_total_ms'], 3 + .2*32 + row['network_ms'])
        self.assertAlmostEqual(row['semcache_total_ms'], 1 + 2 + .1*32 + row['semcache_network_ms'])
        self.assertEqual(result['components']['es_compute_ms']['measurement_label'], MEASUREMENT_LABEL)

    def test_control_plane_is_separate_and_nonoverlapping(self):
        row = strict_profiles()[2]
        row['cache_materialization_ms'] = 999.  # nested timer must never be summed separately
        measured = control_plane(row)
        self.assertAlmostEqual(measured['policy_ms'], .2)
        self.assertAlmostEqual(measured['total_control_ms'], 1.)
        self.assertAlmostEqual(measured['control_unaccounted_ms'], .2)
        row['semantic_encode_ms'] = 100
        with self.assertRaises(ValueError):
            control_plane(row)

    def test_span_plan_and_savings_use_only_physical_reuse(self):
        rows = strict_profiles()
        physical = rows[2]
        physical['block_hits'] = 999
        self.assertEqual(sum(p['end']-p['start'] for p in span_plan(physical, list(range(32)))), 16)
        result = compare_request(*rows, calibration(), DIMS)
        self.assertEqual(result['row']['communication_saved_bytes'], 16*2560*32*8)
        physical['reuse_block_provenance'][0]['destination_end'] = 17
        with self.assertRaises(ValueError):
            span_plan(physical, list(range(32)))
        physical = strict_profiles()[2]
        physical['reuse_block_provenance'].append(copy.deepcopy(physical['reuse_block_provenance'][0]))
        with self.assertRaises(ValueError):
            span_plan(physical, list(range(32)))

    def test_profile_identity_dtype_cache_and_skipped_rows_must_match(self):
        for field, value in [('dtype', 'torch.float32'), ('model_revision', 'old'),
                ('cache_storage_device', 'cuda:0'), ('cache_dtype', 'torch.float32'),
                ('control_plane_inside_es_compute', True), ('executed_reused_tokens', 17),
                ('active_lora_projection_count', 1), ('personalized_output_quality_claimed', True)]:
            row = strict_profiles()[2]
            row['es_base_profile'][field] = value
            with self.assertRaises(ValueError):
                validate_base_profile(row, 'ES_BASE_SEMCACHE_REUSE')
        rows = strict_profiles()
        for row in rows:
            row['impact_reducer_type'] = 'reproduction_frobenius_mean'
        with self.assertRaises(ValueError):
            compare_request(*rows, calibration(), DIMS)

    def test_default_inference_path_is_not_replaced(self):
        from semcache.edgelora.mixed_projection import mixed_projection_path
        from semcache.system_cost.base_projection import base_projection_path
        self.assertIsNot(mixed_projection_path, base_projection_path)
        self.assertNotIn('base_projection_path', (ROOT/'src/semcache/semcache_engine.py').read_text())
        self.assertNotIn('base_projection', (ROOT/'scripts/30_run_m8_inference_timing.py').read_text())

    def test_fresh_driver_pins_paper_settings_and_has_no_download_flag(self):
        spec = importlib.util.spec_from_file_location('base_driver', ROOT/'scripts/35_profile_m9a1_es_base.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        args = SimpleNamespace(model='facebook/opt-125m', device='cuda', dtype='float32',
                               warmup_runs=1, measured_runs=2, revision='pinned')
        command = module.fresh_m8_command(args, Path('/tmp/unused-test-output'))
        self.assertNotIn('--allow-download', command)
        self.assertEqual(command[command.index('--prompt-lengths')+1], '32')
        self.assertEqual(command[command.index('--impact-reducer')+1], 'paper_row_l2_sum')
        self.assertEqual(command[command.index('--revision')+1], 'pinned')

    def test_strict_cost_cli_roundtrip_with_fixture_profiles(self):
        import json
        import subprocess
        import tempfile
        from semcache.system_cost.common import write_json
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_json(root/'strict.json', list(strict_profiles()))
            write_json(root/'calibration.json', calibration())
            write_json(root/'config.json', dict(model_type='opt', hidden_size=2560, num_hidden_layers=32))
            run = subprocess.run([sys.executable, str(ROOT/'scripts/34_run_m9a_single_user_cost.py'),
                '--es-input', str(root/'strict.json'), '--ud-calibration', str(root/'calibration.json'),
                '--model-config', str(root/'config.json'), '--es-compute-policy', 'strict-base-only',
                '--output-dir', str(root/'out')], capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            summary = json.loads((root/'out/m9a_summary.json').read_text())
            self.assertTrue(summary['primary_comparison_eligible'])
            self.assertEqual(summary['result_label'], 'STRICT_BASE_ONLY_SYSTEM_MODEL')
            self.assertEqual(summary['comparisons'][0]['components']['es_compute_ms']['measurement_label'],
                             MEASUREMENT_LABEL)


class BaseProjectionDispatchTest(unittest.TestCase):
    def test_only_fresh_rows_reach_plain_linear_and_forwards_are_restored(self):
        """Exercise actual context manager with shape-only stubs, not torch/models."""
        from contextlib import nullcontext
        from unittest.mock import patch
        from semcache.system_cost.base_projection import base_projection_path

        class Vector:
            def __init__(self, values): self.values = list(values)
            def __len__(self): return len(self.values)
            def __getitem__(self, item): return Vector(self.values[item])
            def __setitem__(self, item, value):
                self.values[item] = [value]*len(self.values[item])
            def any(self): return any(self.values)
            def __invert__(self): return Vector([not v for v in self.values])
            def nonzero(self): return Vector([i for i, value in enumerate(self.values) if value])
            def flatten(self): return self
            def sum(self): return sum(self.values)
            def to(self, *args): return self
            def tolist(self): return self.values

        transfers = []
        class Tensor:
            dtype, device, is_cuda = 'float16', 'cpu', False
            def __init__(self, shape): self.shape = shape
            def index_select(self, dim, index):
                result = Tensor((1, len(index), self.shape[-1]))
                result.positions = index.tolist()
                return result
            def index_copy_(self, *args): pass
            def __setitem__(self, *args): pass
            def to(self, target):
                transfers.append(self.shape)
                return self
            def detach(self): return self

        class Linear:
            in_features, out_features = 4, 4
            _forward_hooks, _forward_pre_hooks = {}, {}
            weight = SimpleNamespace(dtype='float16')
            def __init__(self): self.calls = []
            def forward(self, hidden):
                self.calls.append(hidden.positions)
                return Tensor(hidden.shape)
            def __call__(self, hidden): return self.forward(hidden)

        torch_stub = SimpleNamespace(zeros=lambda n, **kw: Vector([False]*n), bool=bool,
            inference_mode=nullcontext, empty=lambda shape, **kw: Tensor(shape),
            nn=SimpleNamespace(Linear=Linear))
        for cached in (0, 2, 5):
            modules = {name: Linear() for name in 'qkv'}
            model = SimpleNamespace(named_modules=lambda: modules.items())
            adapter = SimpleNamespace(model=model, layers=[0], projection_modules=lambda _: modules)
            hits = []
            if cached:
                ids = tuple(range(cached))
                entry = SimpleNamespace(token_ids=ids, qkv_metadata={'component_scope': 'base_qkv_latency_only'},
                                        tensors={0: tuple(Tensor((1, cached, 4)) for _ in 'qkv')})
                hits = [SimpleNamespace(window=SimpleNamespace(start=0, end=cached, token_ids=ids), entry=entry)]
            with patch.dict(sys.modules, {'torch': torch_stub}):
                with base_projection_path(adapter, hits, 5) as audit:
                    for module in modules.values():
                        self.assertEqual(module(Tensor((1, 5, 4))).shape, (1, 5, 4))
                for module in modules.values():
                    self.assertEqual(module.calls, [list(range(cached, 5))] if cached < 5 else [])
                    self.assertNotIn('forward', module.__dict__)
                self.assertEqual(len(audit.records), 3)
                self.assertTrue(all(r['reused_projection_rows'] == cached for r in audit.records.values()))
        self.assertEqual(len(transfers), 6)  # three cached Q/K/V copies for each nonempty hit case


if __name__ == '__main__':
    unittest.main()
