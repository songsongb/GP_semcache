"""Historical schemas, SHA integrity, correctness and cross-cohort gates."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from r2d_fixtures import ROOT, make_all, make_m9a, read, write
from semcache.diagnostics.r2d_evidence import EvidenceAudit, load_m9a, load_c4, compatibility, sha256


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.m9a, self.c4 = make_all(self.root)

    def audit(self):
        return EvidenceAudit(ROOT)

    def test_valid_original_schemas_separate_cohorts_and_strict_fail_retained(self):
        audit = self.audit()
        m = load_m9a(self.m9a,audit)
        c = load_c4(self.c4,audit)
        self.assertEqual(len(m['records']),1)  # two old bandwidths, one measurement
        self.assertEqual(len(c['sessions']),3)
        high = next(r for r in c['conditions'] if r['requested_coverage'] == 'max')
        self.assertFalse(high['full_relative_strict']['RAW_REPLAY_BASELINE'])
        zero = next(r for r in c['conditions'] if r['actual_reused_tokens'] == 0)
        self.assertEqual(zero['comparisons']['baseline_minus_snapshot_ms']['status'],'NOT_APPLICABLE')
        self.assertIsNone(zero['comparisons']['baseline_minus_snapshot_ms']['attributable_difference_ms'])
        matrix = compatibility(m,c)
        self.assertEqual(matrix['status'],'NOT_COMPARABLE')
        self.assertFalse(matrix['direct_latency_addition_permitted'])
        self.assertIn('length',matrix['reasons'])
        self.assertTrue(all(r['validity'] == 'VALID' for r in audit.record()['files']))

    def test_m9a_summary_provenance_sign_units_duplicate_and_eligibility(self):
        original = read(self.m9a)
        for mutation in ('schema','provenance','sign','units','duplicate','eligibility','correctness','proxy_label'):
            with self.subTest(mutation=mutation):
                s = copy.deepcopy(original)
                r = s['comparisons'][0]['row']
                if mutation == 'schema': s['schema'] = 'cachegen_c9b'
                if mutation == 'provenance': s['total_provenance'] = 'MEASURED'
                if mutation == 'sign': r['system_delta_ms'] *= -1
                if mutation == 'units': r['network_ms'] /= 1000
                if mutation == 'duplicate': s['comparisons'][1] = copy.deepcopy(s['comparisons'][0])
                if mutation == 'eligibility': s['primary_comparison_eligible'] = False
                if mutation == 'correctness': r['correctness_gate_passed'] = False
                if mutation == 'proxy_label': r['result_label'] = 'SIMULATED_PROXY_DOUBLE_COUNTS_LORA'
                write(self.m9a,s)
                with self.assertRaises(ValueError): load_m9a(self.m9a,self.audit())
        write(self.m9a,original)

    def test_m9a_source_tampering_missing_evidence_and_nonfinite(self):
        es = self.m9a.parent/'strict_es_input.jsonl'
        es.write_text(es.read_text()+'\n')
        audit = self.audit()
        with self.assertRaisesRegex(ValueError,'SHA256'): load_m9a(self.m9a,audit)
        self.assertTrue(any(r['validity'] == 'INVALID' for r in audit.record()['files']))
        path = self.root/'nan.json'
        path.write_text('{"value": NaN}')
        with self.assertRaisesRegex(ValueError,'nonfinite'): self.audit().read(path)
        path.write_text('{"value": 1e999}')
        with self.assertRaisesRegex(ValueError,'nonfinite'): self.audit().read(path)
        path.write_text('{"value": 1, "value": 2}')
        with self.assertRaisesRegex(ValueError,'duplicate'): self.audit().read(path)

    def test_wrong_m9a_source_adapter_length_endpoint_model_and_prior_gate(self):
        # Update the outer file SHA to exercise semantic validation, not just SHA.
        es = self.m9a.parent/'strict_es_input.jsonl'
        envpath = self.m9a.parent/'m9a_environment.json'
        original = es.read_text()
        env = read(envpath)
        for mutation in ('adapter','model','length','endpoint','gate'):
            with self.subTest(mutation=mutation):
                rows = [json.loads(line) for line in original.splitlines()]
                for row in rows:
                    if mutation == 'adapter': row['adapter_name'] = 'user_b'
                    if mutation == 'model': row['model_id'] = 'facebook/opt-125m'
                    if mutation == 'length': row['prompt_tokens'] = 64
                    if mutation == 'endpoint' and 'es_base_profile' in row: row['es_base_profile']['compute_scope'] = 'one_token_head'
                    if mutation == 'gate' and row['mode'] == 'SEMCACHE_PHYSICAL_REUSE': row['controlled_exact_parity_passed'] = False
                es.write_text(''.join(json.dumps(r)+'\n' for r in rows))
                env['input_files'][str(es)] = sha256(es)
                write(envpath,env)
                with self.assertRaises((ValueError,KeyError)): load_m9a(self.m9a,self.audit())

    def test_proxy_is_never_promoted_to_strict(self):
        p = make_m9a(self.root/'proxy',proxy=True)
        m = load_m9a(p,self.audit())
        self.assertFalse(m['cohort']['strict_primary_eligible'])
        self.assertEqual(m['records'][0]['row']['result_label'],'SIMULATED_PROXY_DOUBLE_COUNTS_LORA')

    def test_c4_required_hashes_and_decision_copy_rejection(self):
        mp = self.c4/'manifest.json'
        manifest = read(mp)
        data = Path(manifest['breakdown_evidence']['directory'])/'cache_path_breakdown.json'
        data.write_text(data.read_text()+' ')
        with self.assertRaisesRegex(ValueError,'SHA256'): load_c4(self.c4,self.audit())
        data.unlink()
        with self.assertRaises(ValueError): load_c4(self.c4,self.audit())
        d = read(self.c4/'cache_path_decision.json')
        d['conditions'][0]['actual_reused_tokens'] = 3
        write(self.c4/'cache_path_decision.json',d)
        with self.assertRaisesRegex(ValueError,'copies disagree'): load_c4(self.c4,self.audit())

    def test_c4_wrong_branch_adapter_model_endpoint_length_and_duplicate_session(self):
        mp = self.c4/'manifest.json'
        original = read(mp)
        for mutation in ('branch','adapter','model','endpoint','length','duplicate'):
            with self.subTest(mutation=mutation):
                m = copy.deepcopy(original)
                ep = m['input_evidence'][1]
                p = Path(ep['directory'])/'manifest.json'
                source = read(p)
                old = copy.deepcopy(source)
                if mutation == 'branch': m['code']['git_branch'] = 'main'
                if mutation == 'adapter': source['settings']['adapter'] = 'user_b'
                if mutation == 'model': source['benchmark_config']['model'] = 'facebook/opt-125m'
                if mutation == 'endpoint': source['settings']['head'] = 'one_token_logits'
                if mutation == 'length': source['benchmark_config']['lengths'] = [32]
                if mutation == 'duplicate': m['input_evidence'][1] = copy.deepcopy(m['input_evidence'][0])
                write(p,source)
                ep['manifest_sha256'] = sha256(p)
                write(mp,m)
                with self.assertRaises((ValueError,KeyError)): load_c4(self.c4,self.audit())
                write(p,old)
        write(mp,original)

    def test_different_head_allowed_and_relative_breakdown_is_repo_relative(self):
        mp = self.c4/'manifest.json'
        m = read(mp)
        m['code']['git_head'] = 'f'*40
        write(mp,m)
        self.assertTrue(load_c4(self.c4,self.audit())['conditions'])
        # Resolve ../ paths through the declared repository root, never CWD.
        import os
        bd = Path(m['breakdown_evidence']['directory'])
        m['breakdown_evidence']['directory'] = os.path.relpath(bd,ROOT)
        write(mp,m)
        previous = Path.cwd()
        try:
            os.chdir(self.root)
            self.assertTrue(load_c4(self.c4,self.audit())['breakdown'])
        finally:
            os.chdir(previous)

    def test_c4_rejects_rehashed_wrong_raw_endpoint_duplicate_reps_and_correctness(self):
        mp = self.c4/'manifest.json'
        m = read(mp)
        ep = m['input_evidence'][1]
        path = Path(ep['directory'])/'cache_path_raw_pairs.json'
        original = read(path)
        for mutation in ('endpoint','duplicate','correctness','oracle'):
            with self.subTest(mutation=mutation):
                rows = copy.deepcopy(original)
                if mutation == 'endpoint': rows[-1]['endpoint_policy'] = 'one_token_logits_use_cache_true'
                if mutation == 'duplicate': rows[-1] = copy.deepcopy(rows[-2])
                if mutation == 'correctness': rows[-1]['full_relative_quality']['RAW_REPLAY_BASELINE']['allclose'] = True
                if mutation == 'oracle': rows[-1]['raw_equivalence']['RAW_REPLAY_BOTH_ORACLE']['exact_qkv'] = False
                write(path,rows)
                ep['raw_samples_sha256'] = sha256(path)
                write(mp,m)
                with self.assertRaises((ValueError,KeyError)): load_c4(self.c4,self.audit())

    def test_c4_zero_control_cannot_claim_removed_work_and_reference_manifest_tamper(self):
        ap = self.c4/'cache_path_aggregate_summary.json'
        dp = self.c4/'cache_path_decision.json'
        agg = read(ap)
        zero = next(c for c in agg if c['actual_reused_tokens'] == 0)
        zero['comparisons']['baseline_minus_snapshot_ms'].update(status='APPLICABLE',attributable_difference_ms=1.)
        write(ap,agg)
        decision = read(dp)
        decision['conditions'] = agg
        write(dp,decision)
        with self.assertRaises(ValueError): load_c4(self.c4,self.audit())
        # A changed referenced manifest is rejected before semantic reconstruction.
        m = read(self.c4/'manifest.json')
        path = Path(m['input_evidence'][0]['directory'])/'manifest.json'
        path.write_text(path.read_text()+' ')
        with self.assertRaisesRegex(ValueError,'SHA256'): load_c4(self.c4,self.audit())


if __name__ == '__main__':
    unittest.main()
