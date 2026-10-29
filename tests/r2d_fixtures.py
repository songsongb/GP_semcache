"""TEST_FIXTURE_ONLY scalar artifacts; no measurements, inference or networking.

Reuse the existing scalar schema builders, then supply real file hashes and
consistent serialized reference digests. Values never certify GPU execution.
"""
import copy
import json
from pathlib import Path
import sys

from test_m9a1_base_profile import strict_profiles
from test_m9a_cost_model import calibration, DIMS
from test_r2c4_analysis import session
from semcache.diagnostics.r2c3_impact_replay import BASES, canonical
from semcache.diagnostics.r2c4_cache_path import ARMS, RAW
from semcache.diagnostics.r2c4_analysis import aggregate, differences
from semcache.diagnostics.r2d_evidence import BRANCH, digest, sha256
from semcache.system_cost.common import read_dimensions
from semcache.system_cost.model import compare_request

ROOT = Path(__file__).resolve().parents[1]


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')


def read(path):
    return json.loads(path.read_text())


def make_m9a(root, *, proxy=False, zero=False):
    root = Path(root)/'m9a_TEST_FIXTURE_ONLY'
    root.mkdir(parents=True)
    rows = list(strict_profiles())
    if zero:
        r = rows[2]
        r.update(reused_tokens=0, recomputed_tokens=32, token_reuse_ratio=0., physical_reuse_used=False,
                 projection_skip_used=False, saved_communication_bytes=0, reuse_block_provenance=[])
        r['es_base_profile'].update(executed_reused_tokens=0, executed_fresh_tokens=32)
    source = [{k: copy.deepcopy(v) for k, v in r.items() if not k.startswith('es_base_') and
               k not in ('fresh_m85_profile_id', 'semcache_control_profile')} for r in rows]
    fresh = root/'fresh_m85'
    fresh.mkdir()
    (fresh/'inference_raw.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in source))
    write(fresh/'inference_environment.json', dict(TEST_FIXTURE_ONLY=True))
    for row in (rows[0], rows[2]):
        row['es_base_profile'].update(source_m8_sha256=sha256(fresh/'inference_raw.jsonl'),
            source_environment_sha256=sha256(fresh/'inference_environment.json'),
            device='cuda:0', prompt_token_ids=list(range(32)),
            compute_scope='full OPT base prefill, including local embedding/logits; no LoRA/control')
    es = root/'strict_es_input.jsonl'
    es.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    cal = calibration()
    cal.update(schema='m9a_ud_lora_calibration_v1', target_modules=['q_proj','k_proj','v_proj'],
               parameter_element_bytes=4, temporary_element_bytes=4, TEST_FIXTURE_ONLY=True)
    write(root/'ud.json', cal)
    write(root/'config.json', dict(model_type='opt', hidden_size=2560, num_hidden_layers=32))
    _, ds = read_dimensions(DIMS.model_id, root/'config.json')
    policy = 'peft-prefill-proxy' if proxy else 'strict-base-only'
    cfg = dict(model=DIMS.model_id, es_input=str(es), ud_calibration=str(root/'ud.json'),
        model_config=str(root/'config.json'), query_id='same_user_exact', reuse_mode='SEMCACHE_PHYSICAL_REUSE',
        bandwidth_mbps=[200., 500.], hidden_element_bytes=None, delta_element_bytes=None,
        projection_exchange_only=False, es_compute_policy=policy, allow_invalid_reuse=False)
    comparisons = [compare_request(*rows, cal, DIMS, bandwidth_mbps=b, es_compute_policy=policy) for b in cfg['bandwidth_mbps']]
    summary = dict(schema='m9a_single_user_v1', total_provenance='SIMULATED',
        result_label=comparisons[0]['row']['result_label'],
        primary_comparison_eligible=all(c['row']['primary_comparison_eligible'] for c in comparisons),
        safe_reuse_claimed=False, decisions_applied=False, comparison_count=len(comparisons),
        comparisons=comparisons, TEST_FIXTURE_ONLY=True)
    write(root/'m9a_summary.json', summary)
    write(root/'m9a_environment.json', dict(configuration=cfg, dimension_source=ds,
        model_execution_performed=False, distributed_execution=False, user_count=1,
        input_files={str(p):sha256(p) for p in (es, root/'ud.json')}, TEST_FIXTURE_ONLY=True))
    return root/'m9a_summary.json'


def make_c4(root):
    root = Path(root)/'c4_TEST_FIXTURE_ONLY'
    root.mkdir(parents=True)
    name = 'src/semcache/diagnostics/r2c4_analysis.py'
    files = {name: sha256(ROOT/name)}
    code = dict(files=files, sha256=digest(files), git_branch=BRANCH, git_head='0'*40,
                git_status='', TEST_FIXTURE_ONLY=True)
    sessions, evidences, all_refs = [], [], None
    for i in range(3):
        combined, observations, refs = None, [], {}
        for coverage in (0., .5, 'max'):
            m, rows = session(f'run{i}', coverage=coverage)
            key = next(iter(m['conditions']))
            m.update(code=copy.deepcopy(code), implementation_sha256=files, arms=list(ARMS),
                model_weight_identity=digest('TEST_FIXTURE_ONLY weights'),
                environment_identity=digest('TEST_FIXTURE_ONLY environment'), TEST_FIXTURE_ONLY=True)
            c = m['conditions'][key]
            c['condition_identity'] = digest(dict(TEST_FIXTURE_ONLY=True, case=key))
            c['reference_digests'] = {}
            refs[key] = {}
            for base in BASES:
                binding = dict(base_mode=base, adapter='user_a', dtype='float16', device='cuda:0',
                    storage='cuda:0', capacity_mib=512, length=1024, requested_coverage=coverage,
                    use_cache=False, output_attentions=True, attention_backend='eager',
                    impact_execution='token_precompute', adapter_weight_identity=digest('TEST_FIXTURE_ONLY adapter'),
                    model_weight_identity=m['model_weight_identity'], implementation_sha256=code['sha256'])
                data = dict(binding_digest=digest(binding), windows=[(0,3,[0,1,2])], values=[1.],
                    attention=[[[1,32,1024,1024],'torch.float16','cuda:0']], mask=[[1]*1024],
                    policy_digest=digest('TEST_FIXTURE_ONLY policy'))
                ref = dict(provenance=binding, binding_digest=data['binding_digest'],
                    digest=digest(canonical(data)), candidate_windows=[dict(start=0,end=3,token_ids=[0,1,2],value=1.,value_hex=float(1).hex())],
                    attention_metadata=data['attention'], valid_mask=data['mask'], policy_digest=data['policy_digest'],
                    TEST_FIXTURE_ONLY=True)
                refs[key][base] = ref
                c['reference_digests'][base] = ref['digest']
            for r in rows:
                r.update(projection_layers=32, condition_identity=c['condition_identity'])
                if coverage == .5:
                    walls = [100., 237.4, 561., 439.7, 558.66, 438.]
                elif coverage == 'max':
                    walls = [100.,248.68,784.96,539.34,776.38,526.44]
                else:
                    walls = [100.,110.,90.,89.,91.,88.]
                r['wall_ms'] = dict(zip(ARMS, walls))
                r['differences'] = differences(r['wall_ms'])
                for arm, a in r['arms'].items():
                    a['native_projection_rows'] = (1024-a['reused_tokens'])*32*3
                    a['skipped_projection_rows'] = a['reused_tokens']*32*3
                    if arm != ARMS[0]:
                        a['reference_digest'] = c['reference_digests'][BASES[0] if arm == ARMS[1] else BASES[1]]
                    if arm in RAW:
                        a['qkv_digest'] = digest('TEST_FIXTURE_ONLY QKV')
                        r['raw_equivalence'][arm]['qkv_digest'] = a['qkv_digest']
                observations.append(r)
            if combined is None:
                combined = m
            else:
                combined['conditions'].update(m['conditions'])
        combined['benchmark_config'].update(coverage_values=[0.,.5,'max'])
        directory = root/f'session{i}'
        write(directory/'impact_references.json', refs)
        combined['impact_reference_artifact_sha256'] = sha256(directory/'impact_references.json')
        write(directory/'manifest.json', combined)
        write(directory/'cache_path_raw_pairs.json', observations)
        sessions.append((combined, observations))
        evidences.append(dict(directory=str(directory), manifest_sha256=sha256(directory/'manifest.json'),
            raw_samples_sha256=sha256(directory/'cache_path_raw_pairs.json'), execution_status='COMPLETE', GPU_VALIDATION='EXECUTED'))
        all_refs = refs
    bm = copy.deepcopy(sessions[0][0])
    bm['benchmark_config'].update(phase='breakdown', warmup_runs=1, measured_runs=3)
    bdrows = []
    for key, c in bm['conditions'].items():
        c.update(status='DIAGNOSED', timing_boundary_verified=True)
        cov = c['requested_coverage']
        for arm in ARMS:
            for repeat in range(3):
                components = [dict(component=name, host_ms=value, cuda_ms=None, location='request_internal',
                    parent=parent, status='MEASURED', inclusive=True, do_not_sum_nested=True)
                    for name, value, parent in (
                        ('source_snapshot_validate',226.79 if cov == 'max' else 118.19,'executor_setup'),
                        ('fast_index_guard',127.89,None), ('prepared_mapping_get',.92,'request_mapping_select'),
                        ('full_mixed_projection',313.25,None))]
                if cov == 0:
                    for component in components:
                        if component['component'] in ('source_snapshot_validate', 'prepared_mapping_get'):
                            component.update(host_ms=None, status='NOT_APPLICABLE')
                bdrows.append(dict(case_id=key, length=1024, requested_coverage=cov, model=DIMS.model_id,
                    dtype='float16', endpoint_policy='all_token_logits_use_cache_false_attention_true',
                    projection_layers=32, condition_identity=c['condition_identity'], arm=arm, repeat=repeat,
                    instrumentation='breakdown', timing_boundary_verified=True, components=components,
                    instrumented_request_wall_ms=999., full_relative_quality={'allclose':c['full_relative_strict'][arm]},
                    path={'qkv_memory_counters':{'contiguous_copy_bytes':502824960}}, TEST_FIXTURE_ONLY=True))
    directory = root/'breakdown'
    write(directory/'impact_references.json', all_refs)
    bm['impact_reference_artifact_sha256'] = sha256(directory/'impact_references.json')
    write(directory/'manifest.json', bm)
    write(directory/'cache_path_breakdown.json', bdrows)
    summaries = aggregate(sessions, (bm, bdrows), seed=4101, draws=100)
    first = sessions[0][0]
    cfg = {k: first['benchmark_config'][k] for k in ('model','dtype','lengths','coverage_values','cache_device','capacity_mib','seed')}
    cfg.update(settings=first['settings'], hardware=first['hardware_identity'],
        order_seeds=[m['benchmark_config']['order_seed'] for m, _ in sessions], source_code_sha256=code['sha256'])
    manifest = dict(experiment='R2-C4', execution_status='COMPLETE', GPU_VALIDATION='AGGREGATED_GPU_RUNS',
        analysis_process={'gpu_inference_performed':False}, code=code, arms=list(ARMS), implementation_sha256=files,
        classification='ORACLE_DIAGNOSTIC', safe_reuse_claimed=False, correctness_qualified_speedup=False,
        production_speedup_claimed=False, input_evidence=evidences, aggregated_experimental_config=cfg,
        conditions={s['case_id']:dict(status='AGGREGATED',ablation_equivalence_pass=True,impact_replay_reduction_calls=0) for s in summaries},
        breakdown_evidence=dict(directory=str(directory), manifest_sha256=sha256(directory/'manifest.json'),
            data_sha256=sha256(directory/'cache_path_breakdown.json'), pooled_with_minimal=False), TEST_FIXTURE_ONLY=True)
    out = root/'aggregate'
    write(out/'manifest.json', manifest)
    write(out/'cache_path_aggregate_summary.json', summaries)
    write(out/'cache_path_decision.json', dict(execution_status='COMPLETE',conditions=summaries,breakdown_not_pooled=True,
        classification='ORACLE_DIAGNOSTIC',safe_reuse_claimed=False,correctness_qualified_speedup=False,production_speedup_claimed=False))
    return out


def make_all(root):
    return make_m9a(root), make_c4(root)


if __name__ == '__main__':
    base = Path(sys.argv[1]).resolve()
    m9a, c4 = make_all(base)
    print(json.dumps(dict(TEST_FIXTURE_ONLY=True, m9a_summary=str(m9a), c4_aggregate_dir=str(c4))))
