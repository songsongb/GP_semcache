"""Standard-library tests: never import transformers, torch, or load any model."""
import copy
import json
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest

from semcache.experiments.m9b_semantic_workload import (
    MODEL_ID, MODEL_REVISION, ENCODER_ID, ENCODER_REVISION,
    convert_rows, file_hash, prepare, read_raw, serialize, validate_generated,
)
from semcache.simulation.multi_user import read_workload, simulate


class StubTokenizer:
    def __call__(self, text, *, add_special_tokens=True, truncation=False):
        assert add_special_tokens is True and truncation is False
        # Character fixture is explicitly TEST_STUB, never a real workload encoder.
        return {'input_ids': [2] + [ord(c) for c in text]}


class StubEncoder:
    tokenizer = StubTokenizer()

    def encode(self, texts):
        return [[float(sum(map(ord, text)) % 43), float(len(text))] for text in texts]


def metadata():
    return dict(tokenizer=dict(tokenizer_source_id=MODEL_ID, tokenizer_revision=MODEL_REVISION,
        tokenizer_revision_source='tokenizer._commit_hash',
        tokenizer_revision_evidence={'resolved_commit':MODEL_REVISION}),
        semantic_encoder=dict(model_id=ENCODER_ID, resolved_revision=ENCODER_REVISION,
            pooling='masked_mean', max_length=512, backend='tinybert_huggingface'),
        execution_provenance='TEST_STUB')


def raw_rows(dataset='snips'):
    return [dict(dataset=dataset, source_id=f'q{i}', global_query_index=i,
                 original_order_index=119-i, conversation_id=None, domain_or_intent='intent',
                 query_text=f'query {i%17}', user_id=f'original_{i%50}') for i in range(120)]


class SemanticPreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root/'raw.jsonl'
        self.source.write_text(serialize(raw_rows()), encoding='utf-8')
        self.output = self.root/'semantic.jsonl'

    def run_preparation(self, output=None, **kwargs):
        return prepare(self.source, output or self.output, 'snips', StubTokenizer(), StubEncoder(), metadata(), **kwargs)

    def test_conversion_preserves_rows_order_and_source(self):
        before = self.source.read_bytes()
        manifest = self.run_preparation(expected_rows=120)
        rows = read_workload(self.output, 'snips')
        self.assertEqual(len(rows),120)
        for source, row in zip(raw_rows(),rows):
            for key in ('dataset','source_id','global_query_index','original_order_index',
                        'conversation_id','domain_or_intent','query_text'):
                self.assertEqual(row[key],source[key])
            self.assertEqual(row['raw_user_id'], source['user_id'])
            self.assertNotIn('user_id',row)
            self.assertEqual(row['token_count'],len(row['token_ids']))
            self.assertTrue(0 <= row['cluster_id'] < 30)
        self.assertEqual(self.source.read_bytes(),before)
        self.assertTrue(manifest['validation']['valid'])
        self.assertEqual(manifest['validation']['simulator_prefix_rows'],10)
        self.assertEqual(manifest['metadata']['execution_provenance'],'TEST_STUB')
        self.assertEqual(manifest['state']['pending_update_count'],20)
        self.assertEqual(sum(manifest['state']['final_counts']),130)  # C anchors + 100 flushed queries

    def test_deterministic_hashes_and_rerun_comparison(self):
        first = self.run_preparation()
        other = self.root/'second.jsonl'
        second = self.run_preparation(other, previous_manifest=first)
        self.assertEqual(self.output.read_bytes(),other.read_bytes())
        for key in ('source_raw_sha256','output_semantic_sha256','query_order_sha256',
                    'token_sequence_stream_sha256','semantic_assignment_stream_sha256'):
            self.assertEqual(first[key],second[key])
        changed = copy.deepcopy(first)
        changed['query_order_sha256']='changed'
        with self.assertRaisesRegex(ValueError,'Rerun differs'):
            validate_generated(self.source, other, second, previous_manifest=changed)

    def test_semantic_fields_independent_of_user_count_and_raw_users(self):
        manifest = self.run_preparation()
        rows = read_workload(self.output,'snips')
        traces = [simulate(rows,users) for users in (10,25,50)]
        self.assertEqual(len({t['query_order_hash'] for t in traces}),1)
        self.assertEqual(len({t['user_assignment_hash'] for t in traces}),3)
        alternate = raw_rows()
        for row in alternate:
            row['user_id']='different_original_user'
        converted, _ = convert_rows(alternate,'snips',StubTokenizer(),StubEncoder(),metadata())
        self.assertEqual([(r['token_ids'],r['cluster_id']) for r in rows],
                         [(r['token_ids'],r['cluster_id']) for r in converted])
        self.assertFalse(manifest['safe_reuse_claimed'])

    def test_missing_semantic_provenance_rejected(self):
        bad = metadata()
        del bad['semantic_encoder']['resolved_revision']
        with self.assertRaisesRegex(ValueError,'semantic provenance'):
            convert_rows(raw_rows(),'snips',StubTokenizer(),StubEncoder(),bad)
        manifest = self.run_preparation()
        rows = [json.loads(line) for line in self.output.read_text().splitlines()]
        del rows[0]['semantic_assignment_source']
        self.output.write_text(serialize(rows))
        with self.assertRaisesRegex(ValueError,'namespace'):
            read_workload(self.output,'snips')
        with self.assertRaises(ValueError):
            validate_generated(self.source,self.output,manifest)

    def test_invalid_tokens_cluster_counts_and_order_rejected(self):
        manifest = self.run_preparation()
        original = [json.loads(line) for line in self.output.read_text().splitlines()]
        mutations = [('token_count',0),('cluster_id',30),('global_query_index',9),
                     ('token_ids',[]),('semantic_encoder',{}),('tokenizer_revision','main')]
        for key, value in mutations:
            with self.subTest(key=key):
                rows = copy.deepcopy(original)
                rows[0][key]=value
                self.output.write_text(serialize(rows))
                updated = dict(manifest, output_semantic_sha256=file_hash(self.output))
                with self.assertRaises(ValueError):
                    validate_generated(self.source,self.output,updated)
        self.output.write_text(serialize(original[:-1]))
        with self.assertRaisesRegex(ValueError,'row count'):
            validate_generated(self.source,self.output,manifest)

    def test_raw_protection_and_current_count_is_optional(self):
        before = self.source.read_bytes()
        with self.assertRaisesRegex(ValueError,'raw workload'):
            self.run_preparation(self.source)
        with self.assertRaisesRegex(ValueError,'Expected 13784'):
            self.run_preparation(expected_rows=13784)
        self.run_preparation()
        with self.assertRaisesRegex(ValueError,'already exists'):
            self.run_preparation()
        self.assertEqual(self.source.read_bytes(),before)

    def test_model_free_validate_only_cli(self):
        self.run_preparation()
        result = subprocess.run([sys.executable, 'scripts/38_prepare_m9b_semantic_workload.py',
            '--dataset','snips','--source',str(self.source),'--output',str(self.output),
            '--validate-only','--expect-rows','120'], text=True, capture_output=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertTrue(json.loads(result.stdout)['valid'])

    def test_multiwoz_cluster_bound_and_semantic_truncation(self):
        rows = raw_rows('multiwoz')
        rows[-1]['query_text']='x'*600
        converted, _ = convert_rows(rows,'multiwoz',StubTokenizer(),StubEncoder(),metadata())
        self.assertEqual(len(converted),120)
        self.assertTrue(all(0 <= r['cluster_id'] < 20 for r in converted))
        self.assertEqual(converted[-1]['token_count'],601)
        self.assertFalse(converted[-1]['tokenizer_truncation'])
        self.assertTrue(converted[-1]['semantic_truncated'])
        self.assertEqual(converted[-1]['semantic_input_token_count'],601)


if __name__ == '__main__':
    unittest.main()
