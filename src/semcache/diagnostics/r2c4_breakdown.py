"""Separate instrumented cache-path trials; no nested total is manufactured."""
from semcache.diagnostics.r2c4_cache_path import ARMS,RAW,PARENTS
from semcache.diagnostics.r2c4_benchmark import equivalence
from semcache.diagnostics.r2b3_benchmark import strict_metrics

EXECUTOR_PARENT={s:'full_mixed_projection' for s in ('source_reference_resolution','source_conversion',
    'window_conversion_assignment','fresh_index_preparation','fresh_native_qkv','output_allocation',
    'fresh_assembly','cached_gather','cached_scatter')}
SETUP_PARENT=dict(source_snapshot_validation='executor_setup',target_hit_to_position_mapping='prepared_mapping_get',
    source_group_construction='prepared_mapping_get',cpu_index_tensor_construction='prepared_mapping_get',
    device_index_preparation='prepared_mapping_get',module_forward_patch_setup='executor_setup',context_cleanup='executor_cleanup',
    batched_gather_scatter_execution='full_mixed_projection')


def flatten(record):
    rows=list(record['cache_path_host_breakdown'])
    for kind,parent in [('setup',SETUP_PARENT),('executor',EXECUTOR_PARENT)]:
        book=record.get(kind+'_breakdown')
        if book is None:
            rows.append(dict(component=kind+'_book',location='request_internal',host_ms=None,cuda_ms=None,
                calls=None,status='UNAVAILABLE',parent=None,inclusive=True,clock=None,do_not_sum_nested=True))
            continue
        for name,value in book.items():
            status=value['status']
            if name=='source_snapshot_validation' and record['cache_path']['snapshot_bypassed']:status='BYPASSED_ORACLE'
            if record['reused_tokens']==0 and name in ('source_snapshot_validation','target_hit_to_position_mapping','source_group_construction','cpu_index_tensor_construction','device_index_preparation'):
                status='NOT_APPLICABLE'
            rows.append(dict(component=name,book=kind,location='request_internal',host_ms=value.get('host_ms'),
                cuda_ms=value.get('cuda_ms'),calls=value.get('calls'),status=status,parent=parent.get(name),
                inclusive=True,clock='CPU host interval / CUDA event interval (separate clocks)',do_not_sum_nested=True,
                limitation='CUDA event sum includes stream/host dispatch gaps; not kernel self-time or E2E latency'))
    return rows


def collect(runtime,args,identity,save):
    output=[]
    for _ in range(args.warmup_runs):
        for arm in ARMS:runtime.invoke_cache_arm(arm)
    for repeat in range(args.measured_runs):
        outputs={};records={}
        for arm in ARMS:
            save(context=dict(case=identity['case_id'],phase='breakdown',repeat=repeat,arm=arm))
            outputs[arm],records[arm]=runtime.invoke_cache_arm(arm,detailed=True)
        for arm in RAW:equivalence(outputs[ARMS[2]],outputs[arm],records[ARMS[2]],records[arm])
        for arm,record in records.items():
            output.append(dict(identity,arm=arm,repeat=repeat,instrumentation='breakdown',
                timing_boundary_verified=record['cache_path']['timing_boundary_verified'],
                instrumented_request_wall_ms=record['request_ms'],path=record['cache_path'],
                components=flatten(record),full_relative_quality=strict_metrics(outputs['FULL_NATIVE'],outputs[arm]),
                scope='nonminimal diagnostic only; never pooled with primary latency; nested parent/child and CPU/CUDA timings not summed'))
        outputs.clear();save(breakdown=output)
    return output
