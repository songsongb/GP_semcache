"""R3-A artifact schema, identities, safe raw arrays and fixed parity thresholds."""
import array
import csv
from datetime import datetime, timezone
import importlib.metadata
import math
import os
from pathlib import Path
import struct
import subprocess
import sys

from .communication import canonical, digest, digest_bytes, json_loads, payload_size, require

ROOT = Path(__file__).resolve().parents[3]
CLASSIFICATION = 'PHYSICAL_LOCALHOST_TCP_NO_REUSE'
THRESHOLDS = dict(atol=1e-3, rtol=1e-3, max_abs=1e-2, relative_l2=1e-3,
                  last_token_kl=1e-5, last_token_argmax_agreement=True)
ENDPOINT = dict(batch_size=1, prompt_tokens=32, attention='eager', use_cache=False,
                output_attentions=True, output_head='native_all_token_logits')


def now(): return datetime.now(timezone.utc).isoformat()
def sha(path): return digest_bytes(Path(path).read_bytes())


def read(path):
    return json_loads(Path(path).read_bytes())


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_bytes(canonical(value)+b'\n')
    temporary.replace(path)


def append(path, value):
    with Path(path).open('ab') as stream: stream.write(canonical(value)+b'\n')


def write_csv(path, rows):
    fields = sorted({k for r in rows for k in r} or {'status','error'})
    with Path(path).open('w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def source_identity():
    paths = [ROOT/'scripts/95_run_r3a_distributed_no_reuse.py',
             *[ROOT/'src/semcache/edgelora'/name for name in
               ('edge_server.py','user_device.py','communication.py','collaborative_inference.py')],
             *sorted((ROOT/'src/semcache/edgelora').glob('r3a_*.py')),
             *[ROOT/'src/semcache/models'/n for n in ('loader.py','model_adapter.py','lora_decomposition.py','lora_fixtures.py')],
             ROOT/'src/semcache/system_cost/es_base_profile.py',ROOT/'src/semcache/system_cost/network.py']
    files = {str(p.relative_to(ROOT)):sha(p) for p in paths}
    def git(*args):
        run = subprocess.run(['git',*args],cwd=ROOT,capture_output=True,text=True)
        return run.stdout.strip() if run.returncode == 0 else None
    return dict(files=files, sha256=digest(files), git_head=git('rev-parse','HEAD'),
                git_branch=git('branch','--show-current'), dirty=git('status','--short'))


def versions():
    def version(name):
        try: return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: return None
    return dict(python=sys.version, **{n:version(n) for n in ('torch','transformers','peft','numpy')})


def process_environment():
    return dict(hostname=os.uname().nodename,pid=os.getpid(),variables={k:os.environ.get(k) for k in
        ('HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES',
         'CUBLAS_WORKSPACE_CONFIG','CUDA_LAUNCH_BLOCKING','OMP_NUM_THREADS','MKL_NUM_THREADS')})


def flags(*, socket_used=False, gpu=False, fixture=False):
    return dict(classification=CLASSIFICATION, actual_network_socket_used=socket_used,
        actual_remote_network_measured=False, cache_reuse_enabled=False,
        gpu_inference_performed=gpu, user_device_cuda_used=False, TEST_FIXTURE_ONLY=fixture,
        safe_reuse_claimed=False, correctness_qualified_reuse_speedup=False,
        r2d_break_even_validated=False, full_generation_measured=False, trained_adapter=False)


def pack(values, dtype):
    require(dtype in ('float16','float32'), 'unsupported array dtype')
    values = list(values)
    require(all(math.isfinite(v) for v in values), 'nonfinite tensor')
    if dtype == 'float32':
        result = array.array('f',values)
        if sys.byteorder != 'little': result.byteswap()
        return result.tobytes()
    return struct.pack('<'+'e'*len(values),*values)


def unpack(data, dtype):
    require(dtype in ('float16','float32'), 'unsupported array dtype')
    width = 2 if dtype == 'float16' else 4
    require(len(data)%width == 0, 'misaligned array')
    if dtype == 'float32':
        result = array.array('f'); result.frombytes(data)
        if sys.byteorder != 'little': result.byteswap()
        result = list(result)
    else:
        result = list(struct.unpack('<'+'e'*(len(data)//2),data))
    require(all(math.isfinite(v) for v in result), 'nonfinite tensor')
    return result


def tensor_bytes(tensor):
    import torch
    require(tensor.dtype in (torch.float16,torch.float32) and sys.byteorder == 'little', 'unsupported tensor dtype/byte order')
    cpu = tensor.detach().to('cpu').contiguous()
    require(bool(torch.isfinite(cpu).all()), 'nonfinite tensor')
    return cpu.view(torch.uint8).numpy().tobytes()


def tensor_from_bytes(data, dtype, shape):
    import torch
    require(len(data) == payload_size(dtype,shape) and sys.byteorder == 'little', 'tensor size/endianness mismatch')
    tensor = torch.frombuffer(bytearray(data),dtype=getattr(torch,dtype)).reshape(shape).clone()
    require(bool(torch.isfinite(tensor).all()), 'nonfinite tensor')
    return tensor


def full_base_fingerprint(model, *, peft_wrapped=False):
    import torch
    records = {}
    for name, value in sorted(model.state_dict().items()):
        if peft_wrapped and 'lora_' in name:
            continue
        if peft_wrapped:
            name=name.replace('.base_layer.','.')
        require('lora_' not in name and 'base_layer' not in name and name not in records, 'base fingerprint requires unambiguous base tensors')
        cpu = value.detach().cpu().contiguous()
        records[name] = dict(dtype=str(cpu.dtype),shape=list(cpu.shape),
                            sha256=digest_bytes(cpu.view(torch.uint8).numpy().tobytes()))
    return dict(sha256=digest(records), tensors=records, scope='complete bare OPT state_dict including embeddings/LM head/buffers')


def check_parity(values, reference, shape):
    require(len(values) == len(reference) == math.prod(shape) and shape[0] == 1 and len(shape) == 3,
            'reference logit shape mismatch')
    require(all(math.isfinite(v) for v in values+reference), 'nonfinite logits')
    errors = [abs(a-b) for a,b in zip(values,reference)]
    norm = math.sqrt(sum(v*v for v in reference))
    relative = math.sqrt(sum((a-b)**2 for a,b in zip(values,reference)))/max(norm,1e-30)
    width = shape[-1]
    actual, expected = values[-width:],reference[-width:]
    def logsoftmax(row):
        maximum = max(row)
        logsum = maximum+math.log(sum(math.exp(v-maximum) for v in row))
        return [v-logsum for v in row]
    a,b = logsoftmax(actual),logsoftmax(expected)
    kl = sum(math.exp(y)*(y-x) for x,y in zip(a,b))
    agreement = max(range(width),key=actual.__getitem__) == max(range(width),key=expected.__getitem__)
    violations = sum(e > THRESHOLDS['atol']+THRESHOLDS['rtol']*abs(v) for e,v in zip(errors,reference))
    passed = (violations == 0 and max(errors) <= THRESHOLDS['max_abs'] and relative <= THRESHOLDS['relative_l2'] and
              kl <= THRESHOLDS['last_token_kl'] and agreement)
    return dict(status='PASS' if passed else 'PARITY_FAIL',passed=passed, thresholds=dict(THRESHOLDS),
        max_abs=max(errors), relative_l2=relative, last_token_kl=kl,
        last_token_argmax_agreement=agreement, tolerance_violations=violations)


def validate_prepared(directory):
    directory = Path(directory).resolve()
    manifest = read(directory/'r3a_prepare_manifest.json')
    require(manifest['execution_status'] in ('COMPLETE','PREPARED') and manifest['preparation_process_exited'] is True,
            'completed isolated prepare required')
    require(manifest['code']['files'] == source_identity()['files'], 'prepared implementation hash mismatch')
    require({'r3a_reference.json','r3a_adapter_manifest.json','r3a_reference_logits.bin','r3a_adapter.bin','r3a_preflight.json'}.
            issubset(manifest['artifact_hashes']), 'missing required prepared artifact hash')
    for name, expected in manifest['artifact_hashes'].items():
        require(Path(name).name == name and sha(directory/name) == expected, 'prepared artifact SHA mismatch: '+name)
    reference = read(directory/'r3a_reference.json')
    adapter = read(directory/'r3a_adapter_manifest.json')
    require(adapter['matrix_file']=='r3a_adapter.bin' and reference['logits_file']=='r3a_reference_logits.bin',
            'unexpected local tensor artifact name')
    require(reference['identity'] == adapter['identity'] == manifest['identity'], 'prepared identity mismatch')
    identity=manifest['identity']
    require(identity['token_ids_sha256']==digest(identity['token_ids']) and len(identity['token_ids'])==identity['length'] and
            all(type(v) is int and v>=0 for v in identity['token_ids']), 'prepared token identity mismatch')
    require(identity['adapter_matrix_sha256']==adapter['matrix_sha256'], 'prepared adapter tensor identity mismatch')
    require(reference['logits_dtype']=='float32' and len(reference['logits_shape'])==3 and
            reference['logits_shape'][:2]==[1,identity['length']] and
            (directory/reference['logits_file']).stat().st_size==payload_size(reference['logits_dtype'],reference['logits_shape']),
            'bounded all-token reference logit shape/length mismatch')
    require(reference['thresholds'] == THRESHOLDS and adapter['adapter'] == 'user_a' and adapter['rank'] == 8 and
            adapter['trained_adapter'] is False and adapter['all_matrices_nonzero'] is True, 'adapter/reference gate mismatch')
    require(sha(directory/adapter['matrix_file']) == adapter['matrix_sha256'], 'adapter matrix SHA mismatch')
    require(sha(directory/reference['logits_file']) == reference['logits_sha256'], 'reference logits SHA mismatch')
    require(manifest['identity_sha256'] == digest(manifest['identity']), 'prepared identity digest mismatch')
    require(reference['native_projection_outputs'] == 3*manifest['identity']['layers'], 'incomplete native QKV reference')
    fixture = manifest.get('TEST_FIXTURE_ONLY') is True
    if not fixture:
        require('r3a_base_fingerprint.json' in manifest['artifact_hashes'], 'complete base fingerprint artifact required')
        require(manifest['cleanup']['all_children_exited'] is True, 'reference process exit unverified')
        require(manifest['identity']['model'] in ('facebook/opt-2.7b','facebook/opt-125m') and
                manifest['identity']['endpoint'] == ENDPOINT and manifest['identity']['dtype'] == 'float16', 'unsupported prepared endpoint')
        require(manifest['identity']['layers'] == (32 if manifest['identity']['model'].endswith('2.7b') else 12), 'wrong OPT layer count')
        require(identity['length']==32 and identity['hidden_size']==(2560 if identity['model'].endswith('2.7b') else 768) and
                identity['hidden_dtype']=='float16' and identity['delta_dtype']=='float32','unsupported OPT precision/shape')
    return manifest,reference,adapter


def load_matrices(directory, manifest, *, torch_backend):
    path = Path(directory)/manifest['matrix_file']
    require(path.stat().st_size <= 128*1024*1024 and sha(path) == manifest['matrix_sha256'], 'invalid adapter blob')
    data = path.read_bytes(); result = {}; next_offset = 0
    identity = manifest['identity']; h = identity['hidden_size']
    for record in manifest['matrices']:
        layer,role,matrix = record['layer'],record['role'],record['matrix']
        require(type(layer) is int and 0 <= layer < identity['layers'] and role in ('q','k','v') and matrix in ('A','B'), 'invalid matrix identity')
        require(record['shape'] == ([8,h] if matrix == 'A' else [h,8]) and record['offset'] == next_offset,
                'matrix shape/offset mismatch')
        size = payload_size(record['dtype'],record['shape'])
        blob = data[next_offset:next_offset+size]; next_offset += size
        require(len(blob) == size and record['sha256'] == digest_bytes(blob), 'matrix tensor SHA mismatch')
        key = (layer,role,matrix)
        require(key not in result and any(unpack(blob,record['dtype'])), 'duplicate or zero LoRA matrix')
        result[key] = tensor_from_bytes(blob,record['dtype'],record['shape']) if torch_backend else unpack(blob,record['dtype'])
    require(next_offset == len(data) and len(result) == identity['layers']*6, 'incomplete adapter matrix set')
    require(manifest['scale'] == 1. and set(manifest['dtypes']) == {identity['delta_dtype']}, 'unsupported LoRA scale/precision path')
    return result
