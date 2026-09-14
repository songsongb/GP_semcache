"""Explicit manifests, canonical hashes, allowlisted environment (never env dumps)."""
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess


def now():
    return datetime.now(timezone.utc).isoformat()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def reject_secrets(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if any(s in str(key).lower() for s in ('password', 'secret', 'api_key', 'access_token', 'auth_token', 'hf_token')):
                raise ValueError(f'Secret-bearing field forbidden: {key}')
            reject_secrets(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            reject_secrets(item)
    elif isinstance(value, str) and ('hf_' in value and len(value.split('hf_')[-1]) > 25 or 'sk-proj-' in value):
        raise ValueError('Possible credential in manifest')


def write_manifest(path, value):
    reject_secrets(value)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(canonical(value)+'\n', encoding='utf-8')


def _command(args):
    try:
        return subprocess.check_output(args, stderr=subprocess.DEVNULL, text=True, timeout=5).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def environment():
    versions = {}
    for package in ('torch', 'transformers', 'peft', 'datasets', 'sacrebleu'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    cuda, gpu = False, None
    if versions['torch']:
        import torch
        cuda = torch.cuda.is_available()
        gpu = torch.cuda.get_device_name(0) if cuda else None
    return dict(git_commit_sha=_command(['git', 'rev-parse', 'HEAD']),
                dirty_git_status=_command(['git', 'status', '--porcelain']), host=platform.node(),
                python_version=platform.python_version(), package_versions=versions,
                cuda_available=cuda, gpu_model=gpu,
                driver=_command(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader']))


def run_manifest(run_id, config, workload_manifest, **details):
    from .provenance import configuration_provenance
    result = dict(run_id=run_id, timestamp=now(), environment=environment(), config=config,
                  configuration_provenance=configuration_provenance(config),
                  workload=workload_manifest, **details)
    reject_secrets(result)
    return result
