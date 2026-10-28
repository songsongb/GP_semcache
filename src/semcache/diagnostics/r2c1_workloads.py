"""Bounded controlled RAW-QKV workloads and explicit physical memory estimates."""
import math
from semcache.metrics.r1b2 import coverage_value

MODES=('FULL_NATIVE','SEMCACHE_NO_REUSE','RAW_SEMCACHE_QKV')


def lengths(value,phase,max_positions=2048):
    result=[int(v) for v in (value or ('32,64' if phase=='smoke' else '512,1024')).split(',')]
    allowed=(32,64) if phase=='smoke' else (512,1024,1536,2048)
    if not result or len(set(result))!=len(result) or any(n not in allowed or n>max_positions for n in result):
        raise ValueError('unsupported/duplicate length or actual OPT positional limit exceeded')
    return result


def coverages(value):
    result=[coverage_value(x) for x in value.split(',')]
    if not result or len(set(result))!=len(result): raise ValueError('unique coverages required')
    return result


def expected_tokens(length,coverage):
    v=coverage_value(str(coverage))
    return (length//3 if v=='max' else min(length//3,math.floor(length*v/3)))*3


def memory_estimate(config,length,element_bytes,capacity_bytes,profile=False):
    # Persistent captured TOTAL Q/K/V, not a causal K/V prefix cache.
    source=3*config.num_hidden_layers*length*config.hidden_size*element_bytes
    estimates=dict(source_qkv_backing_bytes=source,
        engine_capacity_budget_bytes=capacity_bytes,
        passive_target_qkv_bytes=source,
        source_capture_and_copy_transient_bytes=2*source,
        assembly_workspace_bytes=6*length*config.hidden_size*element_bytes,
        attention_reserve_bytes=config.num_hidden_layers*config.num_attention_heads*length*length*12,
        activation_reserve_bytes=config.num_hidden_layers*length*config.hidden_size*32,
        all_token_logits_reserve_bytes=length*config.vocab_size*8,
        profiler_reserve_bytes=1024**3 if profile else 0,
        fragmentation_reserve_bytes=512*1024**2)
    return dict(**estimates,estimated_additional_bytes=sum(estimates.values()),
        metadata_python_bytes=None,metadata_status='measured source index/descriptors after build; no invented fixed estimate',
        scope='conservative simultaneous-allocation screen, not measured peak or OOM proof')


def check_coverage(record,ids,layers):
    selected=record.get('selected_provenance',[]); positions=[]
    for p in selected:
        a,b=p['target']; s,e=p['source']
        if (not 0<=a<b<=len(ids) or b-a!=3 or [a,b]!=[s,e] or not p['prefix_token_ids_equal'] or
            not p['adapter_identity_match'] or list(ids[a:b])!=list(p['tokens'])):
            raise ValueError('physical token/position/prefix/adapter alignment violation')
        positions.extend(range(a,b))
    if len(set(positions))!=len(positions) or len(positions)!=record['reused_tokens']:
        raise ValueError('coverage overlaps or differs from physical rows')
    audit=record['projection_audit']; reused=len(positions)
    if audit['skipped_projection_rows_all_layers_roles']!=3*layers*reused or audit['native_projection_rows_all_layers_roles']!=3*layers*(len(ids)-reused):
        raise ValueError('skipped/native QKV rows do not match coverage')
    return dict(reused_tokens=reused,recomputed_tokens=len(ids)-reused,actual_coverage=reused/len(ids),
        covered_positions=positions,selected_windows=len(selected),selected_provenance=selected,
        skipped_projection_rows=audit['skipped_projection_rows_all_layers_roles'],
        native_projection_rows=audit['native_projection_rows_all_layers_roles'])
