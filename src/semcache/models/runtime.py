import importlib.util


def require_models():
    missing = [name for name in ('torch', 'transformers') if importlib.util.find_spec(name) is None]
    if missing:
        raise SystemExit('Missing dependencies: '+', '.join(missing)+". Install: python3 -m pip install -e '.[models,test]'")


def deterministic_cuda_preflight(device):
    import os
    import torch
    if str(device).startswith('cuda') and torch.are_deterministic_algorithms_enabled():
        if os.environ.get('CUBLAS_WORKSPACE_CONFIG') not in (':4096:8', ':16:8'):
            raise RuntimeError('Deterministic CUDA requires CUBLAS_WORKSPACE_CONFIG. '
                               'Before starting Python: export CUBLAS_WORKSPACE_CONFIG=:4096:8')
