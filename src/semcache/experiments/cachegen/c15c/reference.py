"""Independent released reference; optional AST-only execution, never imports CUDA.

Only isolated pure functions and the configuration dataclass are evaluated.
The bin constructors' final .cuda() is replaced by identity for CPU parity.
Source hashes must match before any AST is compiled.
"""
import ast
from dataclasses import dataclass
from pathlib import Path
import subprocess
from types import SimpleNamespace
from typing import Tuple

from ..common import file_hash
from ..shared.source_audit import REVISION, SOURCE_SHA256

BASE = 'LMCache/lmcache/storage_backend/serde/'
HASHES = {BASE+n: SOURCE_SHA256[BASE+n] for n in ('cachegen_basics.py', 'cachegen_encoder.py')}
HASHES[BASE+'cachegen_decoder.py'] = 'fcd529f0c889cd69d15fe9061d2b63b864e691455d964d21896cc327a3b4fc3e'
FUNCTIONS = {
    BASE+'cachegen_basics.py': ['CacheGenConfig.from_model_name'],
    BASE+'cachegen_encoder.py': ['CacheGenSerializer.make_key_bins', 'CacheGenSerializer.make_value_bins',
                                '_split_kv', 'torch_quant_vectorized', 'encode_function'],
    BASE+'cachegen_decoder.py': ['do_dequantize', 'CacheGenDeserializer.from_bytes'],
}


def source_manifest(repo=None):
    result = dict(commit=REVISION, files_functions=FUNCTIONS, source_sha256=HASHES,
        checkout_status='NOT_MOUNTED; faithfully reproduced pinned formulas',
        schedule_origin='Mistral-7B QUANT_LEVEL=2 branch copied literally to 32-layer OPT; OPT is not whitelisted',
        formula='C=bins//2-1; m=amax(abs(x),-1,keepdim=True); q=int8(round(x*(C/m)+C)); xhat=(float(q)-C)/C*m',
        scale_axis='merged heads*head_dim, independently per layer and token',
        rounding='torch.round (ties to even), shift BEFORE rounding',
        clipping='none', epsilon=None, zero_guard=False,
        all_zero_vector='0*inf -> NaN -> int8 conversion is device dependent; no source repair applied',
        input_dtype='FP16 capture; maxabs inherits input dtype; bins/factor FP32',
        reconstruction_dtype='do_dequantize FP32; final huggingface deserializer casts FP16')
    if repo is not None:
        repo = Path(repo).resolve()
        revision = subprocess.run(['git', '-C', str(repo), 'rev-parse', 'HEAD'],
                                  text=True, check=True, capture_output=True).stdout.strip()
        if revision != REVISION:
            raise ValueError('CacheGen checkout must match pinned reference commit')
        for path, expected in HASHES.items():
            if file_hash(repo/path) != expected:
                raise ValueError(f'CacheGen reference source differs: {path}')
        result.update(checkout_status='PINNED_REVISION_AND_SOURCE_HASHES_VERIFIED', checkout_path=str(repo))
    return result


def _node(path, name):
    tree = ast.parse(Path(path).read_text())
    for part in name.split('.'):
        tree = next(n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == part)
    if isinstance(tree, ast.FunctionDef):
        tree.decorator_list = []
    return tree


def _function(path, name):
    import torch
    node = _node(path, name)
    scope = dict(torch=torch, Tuple=Tuple)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(path), 'exec'), scope)
    return scope[node.name]


def reference_functions(repo=None):
    if repo is not None:
        source_manifest(repo)
        repo = Path(repo)
        return (_function(repo/(BASE+'cachegen_encoder.py'), 'torch_quant_vectorized'),
                _function(repo/(BASE+'cachegen_decoder.py'), 'do_dequantize'))
    # Direct transcription of the active released functions (no old torch_quant).
    import torch
    def quantize(bins, input_groups):
        MAX = (bins // 2 - 1)[:, None, None]
        max1 = torch.amax(torch.abs(input_groups), dim=-1, keepdim=True)
        factor = MAX / max1
        xq = torch.round(input_groups * factor + MAX).to(torch.int8)
        return xq, max1
    def dequantize(t, bins, maxtensors):
        C = (bins // 2 - 1)[:, None, None]
        t = t - C
        t = t / C
        t = t * maxtensors
        return t
    return quantize, dequantize


def reference_bins(repo=None):
    import torch
    if repo is None:
        # Literal released configuration slices, independent of policy.bins_for.
        k, v = torch.zeros(32), torch.zeros(32)
        k.fill_(16); k[:20] = 16; k[:10] = 32
        v.fill_(16); v[:2] = 32
        return dict(K=k, V=v)
    source_manifest(repo)
    repo = Path(repo)
    config = _node(repo/(BASE+'cachegen_basics.py'), 'CacheGenConfig')
    scope = dict(torch=torch, dataclass=dataclass, os=SimpleNamespace(environ={'QUANT_LEVEL': '2'}))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[config], type_ignores=[])), '<released-config>', 'exec'), scope)
    cfg = scope['CacheGenConfig'].from_model_name('mistralai/Mistral-7B-Instruct-v0.2')
    result = {}
    for role, method in [('K', 'make_key_bins'), ('V', 'make_value_bins')]:
        node = _node(repo/(BASE+'cachegen_encoder.py'), 'CacheGenSerializer.'+method)
        # Only device transfer changes. Formula, fills and boundary slices stay.
        last = node.body[-1]
        if not isinstance(last, ast.Return) or ast.unparse(last.value) != 'ret.cuda()':
            raise ValueError('Unexpected released bin-constructor device transfer')
        last.value = ast.Name(id='ret', ctx=ast.Load())
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), '<released-bins-cpu>', 'exec'), scope)
        result[role] = scope[method](None, cfg)
    return result
