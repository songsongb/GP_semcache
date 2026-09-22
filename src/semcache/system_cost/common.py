"""Strict provenance, units, config-only dimensions and JSON utilities."""
from dataclasses import dataclass, asdict
import hashlib
import json
import math
import os
from pathlib import Path

PROVENANCE = frozenset({'MEASURED', 'CALIBRATED', 'SIMULATED', 'PAPER_REFERENCE'})
MODELS = ('facebook/opt-2.7b', 'facebook/opt-125m')
PAPER_REFERENCE = {
    'provenance': 'PAPER_REFERENCE',
    'es_gpu': 'NVIDIA A100 80GB', 'es_cpu': 'Intel Xeon Gold 6338',
    'ud_cpu_cores': 4, 'ud_cpu_frequency_ghz': 2.3,
    'ud_memory_capacity_bytes': 8 * 1024**3, 'bandwidth_mbps': 200,
}


def nonnegative(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f'{name} must be finite and nonnegative')
    return value


def integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return value


def tagged_ms(value, provenance, source, *, signed=False, dependencies=()):
    if provenance not in PROVENANCE:
        raise ValueError(f'Invalid provenance: {provenance}')
    if signed:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError('Latency must be numeric, not boolean')
        nonnegative(abs(value), source)
    else:
        nonnegative(value, source)
    if not source or any(p not in PROVENANCE for p in dependencies):
        raise ValueError('Every component needs a source and valid dependency provenance')
    return dict(value_ms=value, provenance=provenance, source=source,
                dependency_provenance=sorted(set(dependencies)))


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass(frozen=True)
class Dimensions:
    model_id: str
    hidden_size: int
    layers: int
    rank: int = 8

    def __post_init__(self):
        if self.model_id not in MODELS:
            raise ValueError('M9-A supports only OPT-2.7B and OPT-125M')
        integer(self.hidden_size, 'hidden_size', 1)
        integer(self.layers, 'layers', 1)
        if self.rank != 8:
            raise ValueError('M9-A calibrates rank 8 only')

    def record(self):
        return asdict(self)


def read_dimensions(model_id, config_path=None):
    """Read config.json directly, never instantiate AutoModel or access network."""
    if model_id not in MODELS:
        raise ValueError('Unsupported M9-A model')
    if config_path is None:
        cache = Path(os.environ.get('HF_HUB_CACHE',
            str(Path(os.environ.get('HF_HOME', str(Path.home()/'.cache/huggingface'))) / 'hub')))
        repo = cache / ('models--' + model_id.replace('/', '--'))
        main = repo / 'refs/main'
        if main.exists():
            config_path = repo / 'snapshots' / main.read_text().strip() / 'config.json'
        else:
            candidates = list((repo / 'snapshots').glob('*/config.json'))
            if len(candidates) != 1:
                raise ValueError('Supply --model-config with a local OPT config.json; no downloads are allowed')
            config_path = candidates[0]
    path = Path(config_path)
    config = json.loads(path.read_text())
    if config.get('model_type') != 'opt':
        raise ValueError('Expected an OPT config.json')
    dims = Dimensions(model_id, config['hidden_size'], config['num_hidden_layers'])
    # Prevent an accidentally supplied tiny/other config from being labelled 2.7B.
    expected = {'facebook/opt-2.7b': (2560, 32), 'facebook/opt-125m': (768, 12)}
    if (dims.hidden_size, dims.layers) != expected[model_id]:
        raise ValueError('Config dimensions disagree with selected model ID')
    return dims, dict(path=str(path.resolve()), sha256=file_sha256(path),
                      source='LOCAL_CONFIG_ONLY', model_weights_loaded=False)
