"""I5 allocation receipts and read-only diagnostics; no historical live cache restore."""
import os
from .communication import require
from .gpu_identity import validate_gpu,SAME_COMPATIBLE_GPU_CLASS


def receipt(pm,gpu,policy,*,fixture=False):
    if fixture:
        return dict(gpu_identity_policy=policy,TEST_FIXTURE_ONLY=True,preparation_gpu_uuid=None,execution_gpu_uuid=None,
                    gpu_uuid_matched=None,preparation_gpu_name=None,execution_gpu_name=None,gpu_class_compatible=None,
                    base_fingerprint_verified=True,adapter_fingerprints_verified=True,
                    physical_kv_created_in_current_execution=False,historical_live_kv_restored=False)
    return dict(validate_gpu(pm['gpu'],gpu,'cuda:0',policy),TEST_FIXTURE_ONLY=False,base_fingerprint_verified=True,
                adapter_fingerprints_verified=True,physical_kv_created_in_current_execution=True,historical_live_kv_restored=False)


def verify_receipt(value,pm,gpu,policy,*,fixture=False):
    require(value==receipt(pm,gpu,policy,fixture=fixture),'GPU provenance receipt differs')
    require(value['historical_live_kv_restored'] is False,'historical live KV restore forbidden')
    return value


def reject_restore(plan):
    require(not plan.get('historical_live_kv_restored',False) and not plan.get('restore_cache_from') and not plan.get('cache_checkpoint'),'historical live cache restore is not supported')


def match_execution(expected,current):
    require(expected==current,'execution GPU changed within matched A/B/C block')


def diagnostics(pm,device='cuda:0',policy=SAME_COMPATIBLE_GPU_CLASS):
    result=dict(gpu_identity_policy=policy,preparation_gpu_uuid=pm.get('gpu',{}).get('uuid'),
                cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),cuda_available=False,allocated_gpu_checked=False,eligible=False)
    try:
        import torch
        result['cuda_available']=torch.cuda.is_available()
        if not result['cuda_available']:return dict(result,reason='CUDA unavailable; allocated GPU NOT_CHECKED')
        require(torch.cuda.device_count()==1,'one CUDA-visible GPU required')
        from .r3a_runtime import memory
        current=memory(device)
        result.update(allocated_gpu_checked=True,current_device=device,gpu=current)
        result.update(validate_gpu(pm['gpu'],current,device,policy));result['eligible']=True
    except Exception as exc:result['reason']=str(exc)
    return result
