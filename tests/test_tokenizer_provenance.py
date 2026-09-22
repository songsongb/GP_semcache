"""Offline provenance regressions: fake loader dependencies, no model execution."""
import copy
import sys
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

from semcache.models.tokenizer_provenance import (
    INHERITED, resolve_tokenizer_provenance, tokenizer_artifact_fields,
    validate_tokenizer_snapshot, compare_snapshot_provenance)
from semcache.system_cost.es_base_profile import validate_fresh_m85
from test_m9a1_base_profile import strict_profiles

REPO = 'facebook/opt-125m'
MODEL = 'a' * 40
TOKEN = 'b' * 40


def resolve(tokenizer=None, **overrides):
    args = dict(model_source_id=REPO, tokenizer_source_id=REPO,
        model_requested_revision='main', tokenizer_requested_revision='main',
        model_resolved_revision=MODEL, model_revision_explicit=True,
        tokenizer_revision_explicit=True)
    args.update(overrides)
    return resolve_tokenizer_provenance(tokenizer or NS(init_kwargs={}), **args)


def row_for(metadata):
    return dict(model_id=REPO, model_revision=MODEL, **tokenizer_artifact_fields(metadata))


class TokenizerProvenanceTest(unittest.TestCase):
    def test_independent_commit_has_priority(self):
        metadata = resolve(NS(init_kwargs={'_commit_hash': TOKEN}, _commit_hash=MODEL))
        self.assertEqual(metadata['tokenizer_revision'], TOKEN)
        self.assertEqual(metadata['tokenizer_revision_source'], 'tokenizer.init_kwargs._commit_hash')
        validate_tokenizer_snapshot(row_for(metadata))

    def test_alternate_authoritative_commit(self):
        metadata = resolve(NS(init_kwargs={}, _commit_hash=TOKEN))
        self.assertEqual(metadata['tokenizer_revision_source'], 'tokenizer._commit_hash')
        validate_tokenizer_snapshot(row_for(metadata))

    def test_snapshot_asset_path(self):
        path = f'/cache/models--facebook--opt-125m/snapshots/{TOKEN}/vocab.json'
        metadata = resolve(NS(init_kwargs={'vocab_file': path}))
        self.assertEqual(metadata['tokenizer_revision'], TOKEN)
        validate_tokenizer_snapshot(row_for(metadata))
        malformed = row_for(metadata)
        malformed['tokenizer_source_id'] = 'other/repo'
        with self.assertRaises(ValueError):
            validate_tokenizer_snapshot(malformed)

    def test_explicit_same_repo_same_revision_inheritance(self):
        metadata = resolve()
        self.assertEqual(metadata['tokenizer_revision_source'], INHERITED)
        self.assertEqual(metadata['tokenizer_revision'], MODEL)
        validate_tokenizer_snapshot(row_for(metadata))

    def test_inheritance_rejects_other_repo_revision_or_implicit_request(self):
        for overrides in ({'tokenizer_source_id': 'other/repo'},
                          {'tokenizer_requested_revision': 'other'},
                          {'tokenizer_revision_explicit': False},
                          {'model_revision_explicit': False}):
            with self.subTest(overrides=overrides):
                metadata = resolve(**overrides)
                self.assertIsNone(metadata['tokenizer_revision'])
                with self.assertRaises(ValueError):
                    validate_tokenizer_snapshot(row_for(metadata))

    def test_missing_source_or_evidence_is_not_a_resolved_snapshot(self):
        row = row_for(resolve())
        for field in ('tokenizer_revision', 'tokenizer_revision_source', 'tokenizer_revision_evidence'):
            broken = copy.deepcopy(row)
            broken.pop(field)
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_tokenizer_snapshot(broken)
        row['tokenizer_source_id'] = 'other/repo'
        with self.assertRaises(ValueError):
            validate_tokenizer_snapshot(row)

    def test_independent_other_repo_requires_actual_resolution(self):
        metadata = resolve(NS(init_kwargs={'_commit_hash': TOKEN}), tokenizer_source_id='other/repo')
        validate_tokenizer_snapshot(row_for(metadata))

    def test_fresh_validation_requires_provenance(self):
        row = strict_profiles()[0]
        validate_fresh_m85(row)
        inherited = resolve(model_source_id=row['model_id'], tokenizer_source_id=row['model_id'])
        row.update(tokenizer_artifact_fields(inherited))
        validate_fresh_m85(row)
        row['tokenizer_revision'] = None
        with self.assertRaises(ValueError):
            validate_fresh_m85(row)
        row['tokenizer_revision'] = MODEL
        row.pop('tokenizer_revision_source')
        with self.assertRaises(ValueError):
            validate_fresh_m85(row)

    def test_comparison_records_and_rejects_mismatch(self):
        fresh = row_for(resolve())
        base = dict(model=REPO, resolved_model_revision=MODEL, **resolve())
        self.assertTrue(compare_snapshot_provenance(fresh, base)['matched'])
        base.update(resolve(NS(init_kwargs={'_commit_hash': TOKEN})))
        result = compare_snapshot_provenance(fresh, base, require_match=False)
        self.assertFalse(result['tokenizer_revision_matches'])
        self.assertTrue(result['model_revision_matches'])
        with self.assertRaisesRegex(ValueError, 'mismatch'):
            compare_snapshot_provenance(fresh, base)
        base.update(resolve())
        base['resolved_model_revision'] = TOKEN
        with self.assertRaises(ValueError):
            compare_snapshot_provenance(fresh, base)

    def test_provenance_survives_raw_and_summary_artifacts(self):
        from semcache.metrics.m8 import raw_record, aggregate_raw
        metadata = resolve()
        fields = tokenizer_artifact_fields(metadata)
        raw = raw_record(experiment_id='test', model_id=REPO, model_revision=MODEL,
            mode='NATIVE', query_id='same_user_exact', **fields)
        for record in (raw, *aggregate_raw([raw])):
            for key, value in fields.items():
                self.assertEqual(record[key], value)

    def test_historical_requested_revision_not_reinterpreted(self):
        fields = tokenizer_artifact_fields({'tokenizer_revision': 'main'})
        self.assertIsNone(fields['tokenizer_revision'])
        self.assertIsNone(fields['tokenizer_revision_source'])

    def test_loader_preserves_revision_selection_and_inference_arguments(self):
        from semcache.models.loader import load_model
        import semcache.models.runtime  # Import before restoring the fake sys.modules context.
        for model_request, token_request in ((None, None), (MODEL, None), ('release', 'release')):
            with self.subTest(model_request=model_request, token_request=token_request):
                calls = {}
                config = NS(model_type='opt', _commit_hash=MODEL)
                tokenizer = NS(init_kwargs={})
                class FakeModel:
                    def __init__(self):
                        self.config = config
                    def to(self, device):
                        calls['device'] = device
                        return self
                    def eval(self):
                        calls['eval'] = True
                        return self
                    def parameters(self):
                        return iter([NS(dtype='float32', device='cpu')])
                def loader(name, result):
                    def load(repo, **kwargs):
                        calls[name] = (repo, kwargs)
                        return result
                    return NS(from_pretrained=load)
                fake_transformers = NS(__version__='test', AutoConfig=loader('config', config),
                    AutoTokenizer=loader('tokenizer', tokenizer),
                    AutoModelForCausalLM=loader('model', FakeModel()))
                fake_torch = NS(float16='float16', float32='float32', bfloat16='bfloat16', __version__='test')
                with patch.dict(sys.modules, torch=fake_torch, transformers=fake_transformers), \
                     patch('semcache.models.runtime.require_models'), \
                     patch('semcache.models.runtime.deterministic_cuda_preflight'):
                    _, _, metadata = load_model(dict(name=REPO, tokenizer=REPO,
                        revision=model_request, tokenizer_revision=token_request,
                        dtype='float32', device='cpu', local_files_only=True,
                        attention_implementation='eager'))
                self.assertEqual(calls['model'][1]['revision'], model_request or 'main')
                self.assertEqual(calls['tokenizer'][1]['revision'], token_request or 'main')
                self.assertTrue(calls['model'][1]['local_files_only'])
                self.assertFalse(calls['model'][1]['use_safetensors'])
                self.assertEqual(calls['model'][1]['attn_implementation'], 'eager')
                self.assertEqual(calls['device'], 'cpu')
                self.assertTrue(calls['eval'])
                if (model_request or 'main') == (token_request or 'main'):
                    self.assertEqual(metadata['tokenizer_revision_source'], INHERITED)
                else:
                    self.assertIsNone(metadata['tokenizer_revision'])


if __name__ == '__main__':
    unittest.main()
