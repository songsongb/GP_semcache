"""Fail-closed, read-only reconciliation of the original M9-A and R2-C4 schemas.

The source cohorts remain distinct. A newly computed SHA is a snapshot of the
supplied file, not an independently authenticated measurement certificate.
"""
import hashlib
import json
import math
from pathlib import Path
import re

from semcache.system_cost.common import nonnegative, read_dimensions
from semcache.system_cost.model import compare_request
from semcache.system_cost.profiles import pair_rows

BRANCH = 'exp/raw-semcache-r2'
MODEL = 'facebook/opt-2.7b'
ENDPOINT = 'all_token_logits_use_cache_false_attention_true'
MISSING_E2E = [
    'actual ES/UD adapter partition and matching personalized output quality',
    'application message framing, counts, dependencies and scheduling',
    'socket bytes, achieved directional throughput, RTT and queueing',
    'wire conversion, protocol, synchronization and CPU/GPU handoffs',
    'cold cache build/fill, refresh, residency and reuse frequency',
    'overlap policy and distributed request wall clock',
    'TTFT, autoregressive decode, token quality and multi-user contention',
]


class EvidenceError(ValueError):
    """Required evidence is absent, contradictory or corrupted."""


def require(condition, reason):
    if not condition:
        raise EvidenceError(reason)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    # Historical metrics.r1.digest contract (including its JSON whitespace).
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def valid_sha(value):
    require(isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None,
            'invalid SHA256 identity')
    return value


def resolve(path, repo_root):
    path = Path(path)
    return (path if path.is_absolute() else Path(repo_root)/path).resolve()


def _unique(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'duplicate JSON key: '+key)
        result[key] = value
    return result


def _finite_tree(value):
    if isinstance(value, float):
        require(math.isfinite(value), 'nonfinite JSON number')
    elif isinstance(value, dict):
        for item in value.values():
            _finite_tree(item)
    elif isinstance(value, list):
        for item in value:
            _finite_tree(item)


def same(actual, expected, location='record'):
    """Compare historical schema values, including units, signs and provenance."""
    if isinstance(expected, dict):
        require(isinstance(actual, dict), location+': expected object')
        for key, value in expected.items():
            require(key in actual, location+': missing '+key)
            same(actual[key], value, location+'.'+key)
    elif isinstance(expected, (list, tuple)):
        require(isinstance(actual, (list, tuple)) and len(actual) == len(expected), location+': list mismatch')
        for i, value in enumerate(expected):
            same(actual[i], value, f'{location}[{i}]')
    elif isinstance(expected, bool) or expected is None:
        require(actual is expected, location+': boolean/null mismatch')
    elif isinstance(expected, (int, float)):
        require(not isinstance(actual, bool) and isinstance(actual, (int, float)) and
                math.isfinite(actual) and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-8),
                location+': numeric value/units/sign mismatch')
    else:
        require(actual == expected, location+': mismatch')


class EvidenceAudit:
    def __init__(self, repo_root):
        self.root = Path(repo_root).resolve()
        self.files = {}
        self.notes = []

    def file(self, path, expected=None):
        path = resolve(path, self.root)
        record = self.files.setdefault(str(path), dict(path=str(path), validity='UNVERIFIED',
            validity_scope='file integrity/readability; see cohort_validation for semantic qualification',
            cohort_validation='NOT_ESTIMABLE_UNTIL_COHORT_RECONSTRUCTION'))
        try:
            actual = sha256(path)
            if 'sha256' in record:
                require(actual == record['sha256'], 'input changed during analysis: '+str(path))
            record.update(sha256=actual, expected_sha256=expected or record.get('expected_sha256'),
                          integrity='RECORDED_SHA256_MATCH' if expected else record.get('integrity', 'SNAPSHOT_ONLY'))
            if expected is not None:
                valid_sha(expected)
                require(actual == expected, 'SHA256 mismatch: '+str(path))
            record['validity'] = 'VALID'
        except (OSError, ValueError) as exc:
            record.update(validity='INVALID', error=str(exc))
            raise EvidenceError(str(exc)) from exc
        return path

    def read(self, path, expected=None, *, jsonl=False):
        path = self.file(path, expected)
        try:
            def parse(text):
                result = json.loads(text, object_pairs_hook=_unique,
                    parse_constant=lambda v: require(False, 'nonfinite JSON constant: '+v))
                _finite_tree(result)
                return result
            text = path.read_text()
            result = [parse(line) for line in text.splitlines() if line.strip()] if jsonl else parse(text)
            self.files[str(path)]['json_parse_validity'] = 'VALID'
            return result
        except (ValueError, TypeError) as exc:
            self.files[str(path)].update(validity='INVALID', error=str(exc))
            raise EvidenceError(f'{path}: {exc}') from exc

    def record(self):
        return dict(files=list(self.files.values()), notes=self.notes,
                    hash_scope='supplied artifacts and recorded source identities; not independent attestation',
                    missing_e2e_components=MISSING_E2E)


def _source_identity(manifest, audit):
    code = manifest['code']
    require(code.get('git_branch') == BRANCH, 'C4 recorded branch mismatch')
    valid_sha(code['sha256'])
    if 'files' in code:
        require(code['files'] and digest(code['files']) == code['sha256'], 'C4 code file-map digest mismatch')
    else:
        audit.notes.append('C4 code.files absent; full historical code-map authentication NOT_ESTIMABLE')
    for collection in (code.get('files', {}), manifest['implementation_sha256']):
        require(isinstance(collection, dict), 'source hash map required')
        for name, expected in collection.items():
            path = resolve(name, audit.root)
            require(path.is_relative_to(audit.root) and path.suffix == '.py', 'invalid source path: '+name)
            audit.file(path, expected)
    require(manifest['implementation_sha256'], 'missing C4 implementation SHA256')
    # A different HEAD alone is not a content mismatch. Only recorded file hashes
    # above are compared with source bytes, and additions to the tree are allowed.


def load_m9a(path, audit):
    path = resolve(path, audit.root)
    summary = audit.read(path)
    env = audit.read(path.parent/'m9a_environment.json')
    require(summary.get('schema') == 'm9a_single_user_v1' and summary.get('total_provenance') == 'SIMULATED',
            'original m9a_single_user_v1 summary with SIMULATED provenance required')
    require(summary.get('safe_reuse_claimed') is False and summary.get('decisions_applied') is False,
            'M9-A must remain analytical, without safe reuse/decision claims')
    require(env.get('model_execution_performed') is False and env.get('distributed_execution') is False and
            env.get('user_count') == 1, 'M9-A composition scope mismatch')
    cfg = env['configuration']
    require(cfg['model'] == MODEL and cfg['query_id'] == 'same_user_exact', 'M9-A model/query mismatch')
    require(cfg['reuse_mode'] == 'SEMCACHE_PHYSICAL_REUSE', 'M9-A RAW physical reuse input required')
    require(cfg.get('allow_invalid_reuse') is False, 'unverified prior correctness / bypass forbidden')
    policy = cfg['es_compute_policy']
    require(policy in ('strict-base-only', 'require-base-only', 'peft-prefill-proxy'), 'unsupported ES provenance/policy')
    require(type(cfg['projection_exchange_only']) is bool, 'boundary configuration must be boolean')
    require(all(cfg.get(k) is None or type(cfg[k]) is int and cfg[k] in (2, 4)
                for k in ('hidden_element_bytes', 'delta_element_bytes')),
            'M9-A uncompressed floating point wire element widths required')
    for name, expected in env['input_files'].items():
        audit.file(name, expected)
    es_path = resolve(cfg['es_input'], audit.root)
    cal_path = resolve(cfg['ud_calibration'], audit.root)
    require(all(str(p) in {str(resolve(k, audit.root)) for k in env['input_files']} for p in (es_path, cal_path)),
            'M9-A inputs missing recorded SHA256')
    data = audit.read(es_path, jsonl=es_path.suffix == '.jsonl')
    rows = data if isinstance(data, list) else data.get('rows', [data])
    cal = audit.read(cal_path)
    require(cal.get('schema') == 'm9a_ud_lora_calibration_v1' and
            cal.get('target_modules') == ['q_proj', 'k_proj', 'v_proj'] and
            cal.get('parameter_element_bytes') == cal.get('temporary_element_bytes') == 4,
            'M9-A exact rank8 QKV FP32 calibration schema required')
    source = env['dimension_source']
    config_path = audit.file(source['path'], source['sha256'])
    audit.read(config_path)
    require(source.get('source') == 'LOCAL_CONFIG_ONLY' and source.get('model_weights_loaded') is False,
            'M9-A config-only dimension provenance required')
    dims, _ = read_dimensions(MODEL, config_path)
    pairs = pair_rows(rows, MODEL, cfg['query_id'], cfg['reuse_mode'])
    require(len({p[0]['experiment_id'] for p in pairs}) == 1, 'multiple M9-A experiments cannot be pooled')
    require(sorted(p[0]['repeat_index'] for p in pairs) == list(range(len(pairs))), 'duplicate/missing M9-A repetitions')
    strict = policy != 'peft-prefill-proxy'
    for triplet in pairs:
        for row in triplet:
            require(row['user_id'] == row['adapter_name'] == 'user_a', 'M9-A wrong user/adapter')
            require(row['dtype'] == 'torch.float16', 'M9-A dtype mismatch: FP16 required')
            require(row.get('measured_runs') == len(pairs), 'M9-A measured repetition count mismatch')
        if strict:
            for row in (triplet[0], triplet[2]):
                profile = row['es_base_profile']
                require(profile.get('compute_scope') == 'full OPT base prefill, including local embedding/logits; no LoRA/control',
                        'M9-A strict endpoint/compute scope mismatch')
                require(isinstance(profile.get('device'), str) and profile['device'], 'M9-A profile device unavailable')
                fresh = audit.read(es_path.parent/'fresh_m85/inference_raw.jsonl', profile['source_m8_sha256'], jsonl=True)
                matches = [r for r in fresh if all(r.get(k) == row.get(k) for k in
                    ('experiment_id', 'model_id', 'query_id', 'user_id', 'mode', 'repeat_index'))]
                require(len(matches) == 1, 'M9-A fresh source pairing missing/duplicate')
                same(row, matches[0], 'M9-A unchanged fresh source fields')
                audit.read(es_path.parent/'fresh_m85/inference_environment.json', profile['source_environment_sha256'])
            from semcache.system_cost.base_runtime import span_plan
            span_plan(triplet[2], triplet[2]['es_base_profile']['prompt_token_ids'])
    if strict and (es_path.parent/'profile_manifest.json').exists():
        pm = audit.read(es_path.parent/'profile_manifest.json')
        require(pm['state'] == 'COMPLETE' and pm['strict_input_sha256'] == sha256(es_path), 'M9-A profile manifest mismatch')
    bands = cfg['bandwidth_mbps']
    require(isinstance(bands, list) and bands and len(set(bands)) == len(bands), 'invalid historical bandwidth grid')
    for b in bands:
        nonnegative(b, 'historical bandwidth_mbps')
        require(b > 0, 'historical bandwidth must be positive')
    expected = []
    for triplet in pairs:
        for bandwidth in bands:
            result = compare_request(*triplet, cal, dims, bandwidth_mbps=bandwidth,
                hidden_element_bytes=cfg['hidden_element_bytes'], delta_element_bytes=cfg['delta_element_bytes'],
                boundary_transfers=not cfg['projection_exchange_only'], es_compute_policy=policy)
            require(result['row']['correctness_gate_passed'] is True and result['row']['invalid_reuse_override'] is False,
                    'M9-A unverified prior correctness')
            expected.append(result)
    comparisons = summary['comparisons']
    require(summary['comparison_count'] == len(expected) == len(comparisons), 'M9-A comparison count mismatch')
    indexed = {}
    for result in comparisons:
        r = result['row']
        key = (r['repeat_index'], r['bandwidth_mbps'])
        require(key not in indexed, 'duplicate M9-A repetition/bandwidth')
        indexed[key] = result
    for result in expected:
        r = result['row']
        key = (r['repeat_index'], r['bandwidth_mbps'])
        require(key in indexed, 'missing M9-A repetition/bandwidth')
        same(indexed[key], result, 'M9-A comparison')
    require(summary.get('result_label') == expected[0]['row']['result_label'], 'M9-A interpretation label mismatch')
    require(summary.get('primary_comparison_eligible') is all(r['row']['primary_comparison_eligible'] for r in expected),
            'M9-A primary eligibility contradicts source records')
    # Bandwidth points in the old summary are repeated accounting of the same
    # measurements. They are not independent repetitions in the new analysis.
    unique = [r for r in expected if r['row']['bandwidth_mbps'] == bands[0]]
    profiles = unique[0]['base_profiles']
    p = profiles[0] or {}
    cohort = dict(model=MODEL, length=sorted({r['row']['prompt_tokens'] for r in unique}), dtype='float16',
        accelerator=unique[0]['row']['es_profile_gpu'], lora_peft_placement='bare OPT ES + calibrated FP32 UD LoRA' if strict else 'PEFT ES proxy + calibrated UD LoRA',
        adapter='user_a synthetic rank8 fixture; UD calibration uses synthetic matrices',
        reference_endpoint='bare OPT latency surrogate; personalized correctness from fresh M8.5 only' if strict else 'PEFT prefill proxy',
        output_head='local full OPT logits included; not physically placed on UD',
        attention_impact_flags='eager/output_attentions=True/use_cache=False; actual M8.5 control',
        cache_residency=p.get('cache_storage_device', 'UNKNOWN'),
        timing_boundary='base prefill + separate exclusive M8.5 control + calibrated UD + analytical payload',
        source_cache_build_exclusion='cold priming/materialization outside ES hit prefill',
        admission_status='fresh M8.5 policy; controlled same_user_exact, not natural hit rate',
        correctness_qualification='correctness_gated_fixture_only; base logits are not quality evidence',
        measurement_provenance='MEASURED ES/control; CALIBRATED UD; SIMULATED network/total',
        strict_primary_eligible=summary['primary_comparison_eligible'])
    return dict(summary=summary, environment=env, records=unique, pairs=pairs, dimensions=dims,
                cohort=cohort, boundary_transfers=not cfg['projection_exchange_only'])


def _c4_source(manifest, rows, references, audit, *, breakdown=False):
    from semcache.diagnostics.r2c3_impact_replay import SETTINGS, BASES, canonical
    from semcache.diagnostics.r2c4_cache_path import ARMS
    require(manifest['experiment'] == 'R2-C4' and manifest['arms'] == list(ARMS), 'C4 experiment/arms mismatch')
    require(manifest['settings'] == SETTINGS, 'C4 adapter/endpoint/head/attention settings mismatch')
    require(manifest.get('environment_verified') is True, 'C4 environment not verified')
    raw = manifest.get('raw_only')
    if raw is not None:
        require(all(raw.get(k) is False for k in ('quantization_enabled', 'storage_compression', 'transport_compression')) and
                raw.get('codec_calls') == 0 and not raw.get('loaded_forbidden_modules'), 'C4 RAW-only provenance mismatch')
    valid_sha(manifest['model_weight_identity'])
    valid_sha(manifest['environment_identity'])
    _source_identity(manifest, audit)
    cfg = manifest['benchmark_config']
    require(cfg['model'] == MODEL and cfg['dtype'] == 'float16', 'C4 wrong model/dtype')
    require(set(cfg['lengths']).issubset({512, 1024}) and 1024 in cfg['lengths'] and
            len(cfg['lengths']) == len(set(cfg['lengths'])), 'C4 wrong length/configuration')
    require(set(cfg['coverage_values']) == {0., .5, 'max'}, 'C4 zero/half/MAX controls required')
    require(cfg['cache_device'] == 'cuda' and cfg['device'].startswith('cuda'), 'C4 device/residency mismatch')
    require(set(references) == set(manifest['conditions']), 'C4 reference condition set mismatch')
    require({(c['length'], c['requested_coverage']) for c in manifest['conditions'].values()} ==
            {(n, v) for n in cfg['lengths'] for v in cfg['coverage_values']} and
            len(manifest['conditions']) == len(cfg['lengths'])*len(cfg['coverage_values']), 'C4 condition grid mismatch')
    for case, condition in manifest['conditions'].items():
        for base in BASES:
            ref = references[case][base]
            binding = ref['provenance']
            same({k: binding[k] for k in ('adapter','dtype','use_cache','output_attentions','attention_backend','impact_execution')},
                 dict(adapter='user_a',dtype='float16',use_cache=False,output_attentions=True,
                      attention_backend='eager',impact_execution='token_precompute'), 'C4 reference binding')
            require(binding['base_mode'] == base and binding['length'] == condition['length'] and
                    binding['requested_coverage'] == condition['requested_coverage'] and
                    binding['model_weight_identity'] == manifest['model_weight_identity'] and
                    binding['implementation_sha256'] == manifest['code']['sha256'] and
                    binding['device'] == cfg['device'] and binding['capacity_mib'] == cfg['capacity_mib'] and
                    binding['storage'].startswith(cfg['cache_device']) and binding.get('adapter_weight_identity'),
                    'C4 reference identity mismatch')
            require(ref['binding_digest'] == digest(binding), 'C4 binding digest mismatch')
            windows = ref['candidate_windows']
            for w in windows:
                nonnegative(w['value'], 'reference impact value')
                require(float(w['value']).hex() == w['value_hex'], 'C4 reference float identity mismatch')
            data = dict(binding_digest=ref['binding_digest'],
                windows=[(w['start'], w['end'], w['token_ids']) for w in windows],
                values=[w['value'] for w in windows], attention=ref['attention_metadata'],
                mask=ref['valid_mask'], policy_digest=ref['policy_digest'])
            require(windows and ref['digest'] == digest(canonical(data)) == condition['reference_digests'][base],
                    'C4 reference value digest mismatch')
    for row in rows:
        require(row['model'] == MODEL and row['dtype'] == 'float16' and row['endpoint_policy'] == ENDPOINT and
                row['projection_layers'] == 32, 'C4 sample model/length/endpoint/projection mismatch')
        c = manifest['conditions'][row['case_id']]
        require(row['length'] == c['length'] and row['requested_coverage'] == c['requested_coverage'] and
                row['condition_identity'] == c['condition_identity'], 'C4 observation identity mismatch')
        n, coverage = c['length'], c['requested_coverage']
        expected = (n//3 if coverage == 'max' else math.floor(n*coverage/3))*3
        require(c['actual_reused_tokens'] == expected and math.isclose(c['actual_coverage'], expected/n, abs_tol=1e-12),
                'C4 actual coverage mismatch')
        if breakdown:
            require(row['components'], 'C4 diagnostic components missing')
            for component in row['components']:
                for field in ('host_ms', 'cuda_ms'):
                    if component.get(field) is not None:
                        nonnegative(component[field], 'C4 diagnostic '+field)
                require(component.get('do_not_sum_nested') is True, 'C4 nested timer label missing')
            nonnegative(row['instrumented_request_wall_ms'], 'C4 breakdown wall')
            require(type(row['full_relative_quality']['allclose']) is bool, 'C4 breakdown correctness missing')
        else:
            require(type(row['projection_layers']) is int, 'C4 projection count must be integer')
            for arm in ARMS[2:]:
                valid_sha(row['arms'][arm]['qkv_digest'])
    if breakdown:
        for key, condition in manifest['conditions'].items():
            observed = {a: all(r['full_relative_quality']['allclose'] for r in rows if r['case_id'] == key and r['arm'] == a) for a in ARMS}
            same(condition['full_relative_strict'], observed, 'C4 breakdown correctness copy')


def load_c4(directory, audit):
    from semcache.diagnostics.r2c4_analysis import aggregate
    from semcache.diagnostics.r2c4_cache_path import ARMS
    directory = resolve(directory, audit.root)
    manifest = audit.read(directory/'manifest.json')
    summaries = audit.read(directory/'cache_path_aggregate_summary.json')
    decision = audit.read(directory/'cache_path_decision.json')
    require(manifest['experiment'] == 'R2-C4' and manifest['execution_status'] == 'COMPLETE' and
            manifest['GPU_VALIDATION'] == 'AGGREGATED_GPU_RUNS' and
            manifest['analysis_process']['gpu_inference_performed'] is False, 'completed CPU C4 aggregate required')
    require(manifest['arms'] == list(ARMS), 'C4 aggregate arm mismatch')
    _source_identity(manifest, audit)
    for obj in (manifest, decision):
        require(obj['classification'] == 'ORACLE_DIAGNOSTIC' and
                all(obj.get(k) is False for k in ('safe_reuse_claimed','correctness_qualified_speedup','production_speedup_claimed')),
                'C4 diagnostic qualification mismatch')
    require(decision['execution_status'] == 'COMPLETE' and decision['breakdown_not_pooled'] is True,
            'C4 decision incomplete or breakdown pooled')
    require(decision['conditions'] == summaries and summaries, 'C4 decision/aggregate copies disagree')
    sessions = []
    require(len(manifest['input_evidence']) == 3, 'three C4 session references required')
    directories = set()
    for evidence in manifest['input_evidence']:
        source = resolve(evidence['directory'], audit.root)
        require(source not in directories, 'duplicate C4 source directory')
        directories.add(source)
        m = audit.read(source/'manifest.json', evidence['manifest_sha256'])
        rows = audit.read(source/'cache_path_raw_pairs.json', evidence['raw_samples_sha256'])
        refs = audit.read(source/'impact_references.json', m['impact_reference_artifact_sha256'])
        require(evidence['execution_status'] == m['execution_status'] == 'COMPLETE' and
                evidence['GPU_VALIDATION'] == m['GPU_VALIDATION'] == 'EXECUTED', 'C4 source status mismatch')
        _c4_source(m, rows, refs, audit)
        sessions.append((m, rows))
    be = manifest['breakdown_evidence']
    bd = resolve(be['directory'], audit.root)
    require(bd not in directories and be['pooled_with_minimal'] is False, 'C4 breakdown mixed with latency')
    bm = audit.read(bd/'manifest.json', be['manifest_sha256'])
    rows = audit.read(bd/'cache_path_breakdown.json', be['data_sha256'])
    refs = audit.read(bd/'impact_references.json', bm['impact_reference_artifact_sha256'])
    _c4_source(bm, rows, refs, audit, breakdown=True)
    seeds = {s['bootstrap_seed'] for s in summaries}
    draws = {s['bootstrap_draws'] for s in summaries}
    require(len(seeds) == len(draws) == 1, 'inconsistent C4 bootstrap settings')
    rebuilt = aggregate(sessions, (bm, rows), seed=next(iter(seeds)), draws=next(iter(draws)))
    require(len(rebuilt) == len(summaries), 'C4 missing aggregate conditions')
    same(summaries, rebuilt, 'C4 aggregate reconstruction')
    config = manifest['aggregated_experimental_config']
    first = sessions[0][0]
    expected = {k: first['benchmark_config'][k] for k in ('model','dtype','lengths','coverage_values','cache_device','capacity_mib','seed')}
    expected.update(settings=first['settings'], hardware=first['hardware_identity'],
        order_seeds=[m['benchmark_config']['order_seed'] for m, _ in sessions], source_code_sha256=first['code']['sha256'])
    same(config, expected, 'C4 aggregated configuration')
    require(set(manifest['conditions']) == {s['case_id'] for s in summaries}, 'C4 aggregate condition set mismatch')
    for condition in manifest['conditions'].values():
        require(condition.get('status') == 'AGGREGATED' and condition.get('ablation_equivalence_pass') is True and
                condition.get('impact_replay_reduction_calls') == 0, 'C4 aggregate condition gate mismatch')
    cohort = dict(model=MODEL, length=config['lengths'], dtype='float16', accelerator=config['hardware'],
        lora_peft_placement='local ES PEFT user_a QKV + full decoder; no UD execution',
        adapter='user_a synthetic rank8 fixture', reference_endpoint=ENDPOINT,
        output_head='native_all_token_logits [1,L,V]',
        attention_impact_flags='eager/output_attentions=True/use_cache=False; exact Impact Replay',
        cache_residency='cuda resident fixed-source RAW QKV',
        timing_boundary='conditional synchronized local request; six-arm minimal observation',
        source_cache_build_exclusion='source capture/catalog/index/descriptors and oracle validation outside request',
        admission_status='controlled occurrence eligibility bypasses production Admission',
        correctness_qualification={s['case_id']: s['full_relative_strict'] for s in summaries},
        measurement_provenance='MEASURED local conditional GPU sessions; separate nonminimal breakdown')
    return dict(manifest=manifest, conditions=summaries, breakdown=rows, cohort=cohort,
                sessions=[m['benchmark_config']['independent_run_id'] for m, _ in sessions])


def compatibility(m9a, c4):
    matrix = []
    for field in m9a['cohort']:
        if field == 'strict_primary_eligible':
            continue
        a, b = m9a['cohort'][field], c4['cohort'][field]
        match = a == b
        matrix.append(dict(field=field, m9a=a, r2c4=b,
            status='MATCH' if match else 'NOT_COMPARABLE',
            reason='same recorded value' if match else 'distinct conditions or measurement scope; no cross-cohort substitution'))
    # Matching model/dtype alone cannot qualify pooled timing or safe reuse.
    return dict(status='NOT_COMPARABLE', cohorts_kept_separate=True, matrix=matrix,
        reasons=[r['field'] for r in matrix if r['status'] != 'MATCH'],
        direct_latency_addition_permitted=False, pooled_e2e_claim_permitted=False)
