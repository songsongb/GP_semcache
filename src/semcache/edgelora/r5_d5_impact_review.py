"""CPU-only paired reference/candidate Light review; no new numerical audit."""
from pathlib import Path
from .communication import require
from .r5_d5_impact import REFERENCE, HOST_BATCH, implementation
from .r5_d5_lightperf_analysis import verify_run, compare, describe
from .r5_d5_lightperf_io import hidden_cuda, read_json, SummaryWriter, fresh


def _compare(reference, candidate):
    require(reference['condition'] == candidate['condition'] == 'SEMCACHE_D5_EXACT', 'paired EXACT runs required')
    require(implementation(reference) == REFERENCE and implementation(candidate) == HOST_BATCH, 'reference/candidate implementation pair required')
    keys = ('dataset', 'scale', 'tasks_sha256', 'source', 'performance_knobs', 'backend', 'versions',
            'precision_policy', 'cpu_concurrency', 'warmup_requests', 'measurement_scope')
    require(all(reference[k] == candidate[k] for k in keys), 'NOT_COMPARABLE: source/device/backend/workload/knobs/resources differ')
    require(all(reference['gpu'].get(k) == candidate['gpu'].get(k) for k in ('uuid', 'name', 'device', 'total_bytes')),
            'NOT_COMPARABLE: physical GPU differs')
    require(reference.get('workload_mode', 'NATURAL') == candidate.get('workload_mode', 'NATURAL') and
            reference.get('replay_stream_sha256') == candidate.get('replay_stream_sha256'), 'continuous replay contract differs')
    left, right = reference['requests'], candidate['requests']
    require(len(left) == len(right) == reference['scale'], 'incomplete paired stream')
    fields = ('request_id', 'sample_id', 'user_id', 'adapter_id', 'input_tokens', 'output_tokens',
              'generated_token_ids', 'generated_text_sha256', 'quality', 'scorer_version', 'physical_hit',
              'physical_reuse_sources', 'fresh_qkv_rows', 'skipped_qkv_rows', 'cache_occupancy_bytes',
              'cache_entry_count', 'policy_counts', 'matching_candidate_count', 'categories')
    proofs = ('full_entry_proof_count', 'consumed_tensor_proof_count', 'admission_entry_proof_count',
              'full_entry_serialized_bytes', 'consumed_tensor_serialized_bytes', 'admission_entry_serialized_bytes',
              'integrity_check_failures', 'records_expired')
    rows = []
    for a, b in zip(left, right):
        require(all(r['integrity_validation']['integrity_check_failures'] == 0 and
                    r['integrity_validation']['records_expired'] is True for r in (a, b)), 'failed/open integrity validation')
        changed = [k for k in fields if a[k] != b[k]]
        if a['attention_impact']['values_sha256'] != b['attention_impact']['values_sha256']:
            changed.append('attention_impact.values_sha256')
        changed.extend('integrity_validation.' + k for k in proofs if a['integrity_validation'][k] != b['integrity_validation'][k])
        require(a['request_id'] == b['request_id'], 'paired request order differs')
        rows.append(dict(request_id=a['request_id'], preserved=not changed, differences=changed,
            reference_e2e_ms=a['application_e2e_ms'], candidate_e2e_ms=b['application_e2e_ms'],
            reference_impact_elapsed_ms=a['attention_impact'].get('elapsed_ms'),
            candidate_impact_elapsed_ms=b['attention_impact'].get('elapsed_ms'),
            delta_e2e_ms=b['application_e2e_ms'] - a['application_e2e_ms'],
            reference_ttft_ms=a['application_ttft_ms'], candidate_ttft_ms=b['application_ttft_ms']))
    return dict(status='IMPACT_TOKEN_AND_LIGHT_SUMMARY_PARITY_PASS' if all(r['preserved'] for r in rows) else 'PARITY_FAILED',
        requests=rows, reference_e2e_ms=describe([r['application_e2e_ms'] for r in left]),
        candidate_e2e_ms=describe([r['application_e2e_ms'] for r in right]),
        total_reference_e2e_ms=sum(r['application_e2e_ms'] for r in left),
        total_candidate_e2e_ms=sum(r['application_e2e_ms'] for r in right),
        full_policy_identity_audit='NOT_EVALUATED_LIGHT_HAS_COUNTS_NOT_FULL_POLICY_TRACE',
        qkv_value_audit='NOT_EVALUATED_LIGHT_HAS_NO_DENSE_VALUE_DUMP',
        independent_full_vocabulary_audit='NOT_EVALUATED', correctness_qualified_speedup='NOT_ESTABLISHED')


def review(reference, candidate, output_dir, full=None):
    hidden_cuda()
    require(reference['run'] != candidate['run'], 'distinct fresh runs required')
    loaded, bindings, protected = [], [], []
    for triple in (reference, candidate) + (() if full is None else (full,)):
        require(set(triple) == {'run', 'plan', 'cpu_audit'}, 'explicit receipt triple required')
        run, binding = verify_run(triple['run'], triple['plan'], triple['cpu_audit'])
        plan = read_json(Path(triple['plan']) / 'plan.json')
        loaded.append((run, plan))
        bindings.append(binding)
        protected.extend(triple.values())
        protected.extend(x['path'] for x in run['ud'].values())
        protected.extend(plan[k] for k in ('original_experiment', 'multi', 'base_prepared', 'baseline_plan', 'strict_pilot') if plan.get(k))
        protected.extend(plan.get('dataset_paths', {}).values())
    a, ap = loaded[0]; b, bp = loaded[1]
    require(ap['original_inventory'] == bp['original_inventory'] and ap['multi_inventory'] == bp['multi_inventory'] and
            ap['stream']['tasks'] == bp['stream']['tasks'], 'original prepared inputs/tasks differ')
    result = _compare(a, b)
    result.update(bindings=bindings, full_vs_candidate='NOT_EVALUATED_NO_MATCHED_FULL_PROVIDED',
                  reference_resources=a.get('resources'), candidate_resources=b.get('resources'),
                  measurement='SINGLE_TRIAL_DESCRIPTIVE; variant wall scopes contain synchronization effects')
    if full is not None:
        baseline, baseline_plan = loaded[2]
        require(baseline_plan['original_inventory'] == bp['original_inventory'] and
                baseline_plan['multi_inventory'] == bp['multi_inventory'] and
                baseline_plan['stream']['tasks'] == bp['stream']['tasks'], 'FULL prepared inputs/tasks differ')
        value = compare(baseline, b)
        require(value['status'] == 'DESCRIPTIVE_MATCHED_IDENTITY', 'FULL/candidate conditions or identities differ')
        result['full_vs_candidate'] = value
    triples = (reference, candidate) + (() if full is None else (full,))
    for triple, binding in zip(triples, bindings):
        require(verify_run(triple['run'], triple['plan'], triple['cpu_audit'])[1] == binding,
                'receipt inputs changed during paired review')
    writer = SummaryWriter(fresh(output_dir, protected))
    writer.write('impact_comparison.json', result)
    writer.seal()
    return result
