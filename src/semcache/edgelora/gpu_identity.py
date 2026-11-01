"""Explicit allocation policy; preparation artifacts are never rewritten."""
import re
from .communication import require

STRICT_PREPARATION_GPU = 'STRICT_PREPARATION_GPU'
SAME_COMPATIBLE_GPU_CLASS = 'SAME_COMPATIBLE_GPU_CLASS'


def validate_gpu(prepared, current, device, policy=STRICT_PREPARATION_GPU):
    require(policy in (STRICT_PREPARATION_GPU,SAME_COMPATIBLE_GPU_CLASS),'unknown GPU identity policy')
    if policy==STRICT_PREPARATION_GPU:
        # Preserve the historical predicate for every default setup caller.
        require(current['uuid']!='UNAVAILABLE' and current['uuid']==prepared['uuid'] and '3090' in current['name'],
                'fresh allocated RTX3090 UUID required')
    else:
        require(isinstance(current.get('uuid'),str) and current['uuid'].strip() and
                current['uuid'].strip().upper() not in ('UNAVAILABLE','NONE','UNKNOWN','NULL'),'execution GPU UUID unavailable')
        require(str(device)=='cuda:0' and current.get('device')=='cuda:0','I5 requires the expected cuda:0 device')
        require(bool(re.search(r'\bRTX\s*3090\b(?!\s*Ti)',prepared.get('name',''),re.I)),'unsupported preparation GPU class')
        require(bool(re.search(r'\bRTX\s*3090\b(?!\s*Ti)',current.get('name',''),re.I)),'unsupported execution GPU class')
    return dict(gpu_identity_policy=policy,preparation_gpu_uuid=prepared['uuid'],execution_gpu_uuid=current['uuid'],
                gpu_uuid_matched=prepared['uuid']==current['uuid'],preparation_gpu_name=prepared['name'],
                execution_gpu_name=current['name'],gpu_class_compatible=True)
