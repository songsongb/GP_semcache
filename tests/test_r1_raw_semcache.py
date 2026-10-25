"""Run without pytest: python3 -m unittest discover -s tests -p test_r1_raw_semcache.py -v."""
import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from semcache.metrics.r1 import (MODES, ENGINE_MODES, atomic_json, digest,
    audit_projection_records, percentile, raw_only_guard, summarize, write_reports, comparison_checks)
from semcache.metrics.r1 import timing_scopes
from semcache.semantic.hit_selection import CacheHit, select_nonoverlapping, select_position_aligned_diagnostic
from semcache.semantic.subsequence import Subsequence


class UtilityTests(unittest.TestCase):
    def records(self):
        return [dict(layer=i, tensor_type=n, native_positions=[6,7], native_projection_rows=2,
                     reused_projection_rows=6, native_call_count=1) for i in range(2) for n in "qkv"]

    def test_execution_counter_rejects_full_gemm_disguised_as_reuse(self):
        records = self.records()
        result = audit_projection_records(records, 8, 2, 6)
        self.assertEqual(result["skipped_projection_rows_all_layers_roles"], 36)
        records[0]["native_projection_rows"] = 8
        with self.assertRaises(AssertionError):
            audit_projection_records(records, 8, 2, 6)

    def test_counter_rejects_missing_projection_and_duplicate_positions(self):
        with self.assertRaises(AssertionError):
            audit_projection_records(self.records()[:-1], 8, 2, 6)
        records = self.records()
        records[0]["native_positions"] = [6,6]
        with self.assertRaises(AssertionError):
            audit_projection_records(records, 8, 2, 6)
        records = self.records()
        records[0]["native_positions"] = [0,1]
        with self.assertRaisesRegex(AssertionError, "consumed cached"):
            audit_projection_records(records, 8, 2, 6, list(range(6)))

    def test_w3_duplicate_positions_and_strict_fixture(self):
        ids = (1,2,3,1,2,3,7)
        entry = SimpleNamespace(key=(0,(1,2,3)), token_ids=(1,2,3), positions=(0,3),
            qkv_metadata=dict(source_query_token_ids=ids, source_user="a", source_adapter="a"))
        hits = [CacheHit(Subsequence((1,2,3), start, start+3), entry) for start in (0,3)]
        selected, mask = select_nonoverlapping(hits, len(ids))
        self.assertEqual(len(selected), 2)
        self.assertEqual(sum(mask), 6)  # w=3 does not mean only three tokens total.
        strict, mask = select_position_aligned_diagnostic(hits, len(ids), ids, "a")
        self.assertEqual(len(strict), 1)
        self.assertEqual(sum(mask), 3)
        self.assertFalse(select_position_aligned_diagnostic(hits, len(ids), ids, "b")[0])
        self.assertFalse(select_position_aligned_diagnostic(hits, len(ids), (9,)+ids[1:], "a")[0])

    def test_compression_import_is_blocked(self):
        with self.assertRaisesRegex(RuntimeError, "forbids compression"):
            with raw_only_guard():
                __import__("semcache.experiments.cachegen.c9a_runtime")
        with raw_only_guard() as guard:
            # Transformers performs optional-package discovery during import.
            importlib.util.find_spec("bitsandbytes")
            self.assertEqual(guard.evidence()["codec_calls"], 0)

    def rows(self):
        return [dict(pair_id="pair", phase="measured", repeat=0, storage="RAW_CPU", instrumentation="minimal",
            requested_prompt_tokens=32, condition="same_user_exact", mode=mode, coverage=0.5 if mode != MODES[0] else 0,
            timing=dict(request_wall_ms=10+i, prefill_gpu_ms=None), correctness=dict(allclose=(i==0)),
            source_signature="s", reuse_signature="r", token_ids_sha256="t") for i, mode in enumerate(MODES)]

    def test_paired_statistics_and_artifacts(self):
        rows = self.rows()
        summary = summarize(rows)
        physical = next(r for r in summary if r["mode"] == "RAW_PHYSICAL_REUSE")
        self.assertEqual(physical["matched_delta_ms_mean"], 2)
        self.assertEqual(physical["reuse_validated_cases"], 0)
        self.assertAlmostEqual(percentile([10,20], .95), 19.5)
        scopes = timing_scopes(dict(request_wall_ms=10, prefill_gpu_ms=8, native_qkv_gpu_ms=1,
                                    cache_materialization_ms=.5, quality_diagnostics_ms=3))
        self.assertEqual(scopes["cache_materialization_ms"]["parent"], "cache_policy_ms")
        self.assertFalse(scopes["native_qkv_gpu_ms"]["additive_to_parent"])
        self.assertEqual(scopes["quality_diagnostics_ms"]["kind"], "outside_request")
        with tempfile.TemporaryDirectory() as tmp:
            write_reports(tmp, rows)
            self.assertEqual(len(Path(tmp, "raw.jsonl").read_text().splitlines()), 4)
            self.assertEqual(len(json.loads(Path(tmp, "correctness.json").read_text())), 4)
            self.assertTrue(Path(tmp, "summary.csv").read_text().startswith("condition,"))
            self.assertIn("No safety claim", Path(tmp, "summary.md").read_text())

    def test_storage_matching_and_atomic_checkpoint(self):
        rows = self.rows()
        other = dict(rows[-1], storage="RAW_GPU_DIAGNOSTIC", reuse_signature="different")
        with self.assertRaises(AssertionError):
            comparison_checks(rows + [other])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "checkpoint.json")
            atomic_json(path, dict(identity=digest(rows), rows=rows))
            self.assertEqual(json.loads(path.read_text())["rows"], rows)
            self.assertFalse(path.with_suffix(".json.tmp").exists())

    def test_cli_and_length_guard_without_model_import(self):
        script = str(ROOT / "scripts/69_run_r1_raw_semcache.py")
        help_result = subprocess.run([sys.executable, script, "--help"], capture_output=True, text=True)
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        result = subprocess.run([sys.executable, script, "--output-dir", "/tmp/not-created-r1",
                                 "--prompt-lengths", "512"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("512+", result.stderr)

    def test_missing_dependency_failure_artifact_and_isolation(self):
        if importlib.util.find_spec("torch"):
            self.skipTest("This test checks the no-torch failure path")
        script = str(ROOT / "scripts/69_run_r1_raw_semcache.py")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp, "run")
            command = [sys.executable, script, "--output-dir", str(root), "--smoke", "--warmup-runs", "0", "--measured-runs", "1"]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 1)
            manifest = json.loads((root / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "FAILED")
            self.assertIn("NOT EXECUTED", manifest["failure"])
            original = (root / "manifest.json").read_bytes()
            subprocess.run(command, capture_output=True)
            self.assertEqual((root / "manifest.json").read_bytes(), original)
            resumed = subprocess.run(command + ["--resume"], capture_output=True, text=True)
            self.assertEqual(resumed.returncode, 1)
            self.assertIn("NOT EXECUTED", resumed.stderr)
            changed = subprocess.run(command + ["--resume", "--seed", "43"], capture_output=True, text=True)
            self.assertIn("Resume rejected", changed.stderr)

    def test_legacy_interfaces_and_no_c9_imports(self):
        engine = ast.parse((ROOT / "src/semcache/semcache_engine.py").read_text())
        query = next(n for n in ast.walk(engine) if isinstance(n, ast.FunctionDef) and n.name == "query")
        self.assertTrue({"execution_mode", "collect_timing"} <= {a.arg for a in query.args.args})
        literals = {n.value for n in ast.walk(engine) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
        self.assertTrue(set(ENGINE_MODES.values()) <= literals)
        for relative in ("scripts/69_run_r1_raw_semcache.py", "src/semcache/metrics/r1.py", "src/semcache/metrics/r1_runtime.py"):
            for node in ast.walk(ast.parse((ROOT / relative).read_text())):
                if isinstance(node, ast.ImportFrom):
                    self.assertNotIn("cachegen", node.module or "")


HAS_MODELS = all(importlib.util.find_spec(x) for x in ("torch", "transformers", "peft"))


@unittest.skipUnless(HAS_MODELS, "NOT EXECUTED: torch/transformers/peft unavailable; tiny random OPT uses no downloads")
class TinyModelTests(unittest.TestCase):
    def setUp(self):
        import torch
        from transformers import OPTConfig, OPTForCausalLM
        from semcache.models.lora_fixtures import create_controlled_users
        from semcache.models.model_adapter import OPTModelAdapter
        torch.manual_seed(7)
        config = OPTConfig(vocab_size=32, hidden_size=16, word_embed_proj_dim=16,
            ffn_dim=32, num_hidden_layers=2, num_attention_heads=2,
            max_position_embeddings=64, dropout=0., attention_dropout=0., use_cache=False)
        config._attn_implementation = "eager"
        self.model = create_controlled_users(OPTForCausalLM(config).eval())[0]
        self.adapter = OPTModelAdapter(self.model)

    def test_real_base_and_lora_rows_and_legacy_equivalence(self):
        import torch
        from semcache.cache.cache_entry import CacheEntry
        from semcache.edgelora.mixed_projection import mixed_projection_path
        from semcache.metrics.r1_runtime import detailed_mixed, Intervals
        hidden = torch.randn(1,9,16)
        with torch.inference_mode():
            sources = {i: tuple(m(hidden) for m in self.adapter.projection_modules(i).values()) for i in range(2)}
            for spans in ([], [(0,3), (3,6)], [(0,3), (3,6), (6,9)]):
                hits = []
                for start, end in spans:
                    ids = tuple(range(start,end))
                    entry = CacheEntry.from_tensors(0, ids, (start,end),
                        {i: tuple(t[:,start:end] for t in qkv) for i,qkv in sources.items()})
                    entry.qkv_metadata["component_scope"] = "total_qkv"
                    hits.append(CacheHit(Subsequence(ids,start,end), entry))
                calls, handles = [], []
                for i in range(2):
                    for module in self.adapter.projection_modules(i).values():
                        for child in (module.base_layer, module.lora_A["user_a"], module.lora_B["user_a"]):
                            handles.append(child.register_forward_pre_hook(lambda m,args: calls.append(args[0].shape[1])))
                try:
                    with detailed_mixed(self.adapter, "user_a", hits, 9, book=Intervals("cpu")) as audit:
                        actual = {i: tuple(m(hidden) for m in self.adapter.projection_modules(i).values()) for i in range(2)}
                finally:
                    for handle in handles:
                        handle.remove()
                expected_rows = 9 - 3*len(spans)
                self.assertEqual(calls, [expected_rows]*18 if expected_rows else [])
                audit_projection_records(list(audit.records.values()), 9, 2, 3*len(spans))
                with mixed_projection_path(self.adapter, "user_a", hits, 9):
                    legacy = {i: tuple(m(hidden) for m in self.adapter.projection_modules(i).values()) for i in range(2)}
                for i in actual:
                    for a,b in zip(actual[i],legacy[i]):
                        torch.testing.assert_close(a,b,atol=0,rtol=0)

    def test_engine_cold_strict_repeat_and_scoped_restore(self):
        import torch
        from semcache.semantic.encoder import ControlledEncoder
        from semcache.semantic.intent_clusterer import IntentClusterer
        from semcache.cache.global_cache import GlobalCache
        from semcache.semcache_engine import SemCacheEngine
        import semcache.semcache_engine as module
        from semcache.metrics.r1_runtime import engine_request, native_request
        text = "2 3 4 5 6 7 8 9"
        tokenizer = lambda text: {"input_ids": list(map(int,text.split()))}
        original = module.mixed_projection_path
        for detailed in (False, True):
            clusterer = IntentClusterer(1)
            clusterer.initialize([[0.]])
            engine = SemCacheEngine(self.model, tokenizer, self.adapter, ControlledEncoder({text:[0.]}),
                                   clusterer, GlobalCache(100000))
            reference = native_request(self.model, tokenizer, self.adapter, text, "user_a", detailed)
            cold = engine_request(engine,text,"user_a","source",ENGINE_MODES["RAW_PHYSICAL_REUSE"],detailed)
            self.assertEqual(cold["summary"]["reused_unique_token_count"],0)
            torch.testing.assert_close(cold["logits"],reference["logits"],atol=1e-6,rtol=1e-5)
            exact = engine_request(engine,text,"user_a","target",ENGINE_MODES["POSITION_ALIGNED_DIAGNOSTIC"],detailed)
            self.assertGreater(exact["summary"]["reused_unique_token_count"],0)
            torch.testing.assert_close(exact["logits"],reference["logits"],atol=1e-6,rtol=1e-5)
            self.assertIs(module.mixed_projection_path,original)


if __name__ == "__main__":
    unittest.main()
