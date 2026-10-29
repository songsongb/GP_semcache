"""Explicit stdlib TEST_FIXTURE_ONLY tiny causal decoder; never an OPT fallback.

32 decoder layers, scalar linear QKV and causal softmax attention. This allows
real independent-process TCP tests without Torch, GPU or model downloads.
"""
import math
from pathlib import Path
import random
import struct

from .communication import digest, digest_bytes, require
from .r3a_artifacts import ENDPOINT, THRESHOLDS, flags, now, pack, sha, source_identity, write


def f32(value): return struct.unpack('<f',struct.pack('<f',value))[0]


def linear(x, weight, inputs, outputs):
    require(len(x)%inputs == 0 and len(weight) == inputs*outputs, 'tiny linear shape mismatch')
    return [f32(sum(x[row*inputs+k]*weight[j*inputs+k] for k in range(inputs)))
            for row in range(len(x)//inputs) for j in range(outputs)]


def forward(tokens, base, delta_provider):
    h,n = base['hidden'],len(tokens)
    x = [base['embedding'][token*h+j] for token in tokens for j in range(h)]
    for layer,weights in enumerate(base['layers']):
        outputs = {}
        for role in 'qkv':
            b = linear(x,weights[role],h,h)
            delta = delta_provider(layer,role,x)
            outputs[role] = [f32(v+d) for v,d in zip(b,delta)]
        attended = []
        for i in range(n):
            scores = [sum(outputs['q'][i*h+k]*outputs['k'][j*h+k] for k in range(h))/math.sqrt(h) for j in range(i+1)]
            maximum = max(scores); exp = [math.exp(s-maximum) for s in scores]; total = sum(exp)
            attended.extend(f32(math.tanh(x[i*h+k]+sum(e/total*outputs['v'][j*h+k] for j,e in enumerate(exp)))) for k in range(h))
        x = attended
    return linear(x,base['head'],h,base['vocabulary'])


def prepare_tiny(directory):
    """Only a test fixture generator; the production prepare CLI never calls it."""
    directory = Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    require(not any(directory.iterdir()), 'new/empty tiny fixture directory required')
    rng = random.Random(42); h,layers,length,vocab = 4,32,4,7
    def randoms(n): return [f32(rng.uniform(-.1,.1)) for _ in range(n)]
    base = dict(hidden=h,vocabulary=vocab,embedding=randoms(vocab*h),head=randoms(vocab*h),
                layers=[{r:randoms(h*h) for r in 'qkv'} for _ in range(layers)])
    identity = dict(model='TEST_FIXTURE_ONLY_TINY_CAUSAL_DECODER',dtype='float32',layers=layers,length=length,
        hidden_size=h,hidden_dtype='float32',delta_dtype='float32',adapter='user_a',rank=8,
        base_weight_sha256=digest(base),token_ids=[1,2,3,4],token_ids_sha256=digest([1,2,3,4]),
        endpoint=dict(ENDPOINT,prompt_tokens=length),precision='stdlib f32 linear outputs; not PEFT/CUDA',
        base_revision='TEST_FIXTURE_ONLY',tokenizer_source='TEST_FIXTURE_ONLY_INTEGER_IDS',tokenizer_revision='TEST_FIXTURE_ONLY')
    matrices,records,blob = {},[],bytearray()
    for layer in range(layers):
        for role in 'qkv':
            for matrix,shape in (('A',[8,h]),('B',[h,8])):
                values = randoms(math.prod(shape)); matrices[layer,role,matrix] = values
                payload = pack(values,'float32')
                records.append(dict(layer=layer,role=role,matrix=matrix,shape=shape,dtype='float32',
                    offset=len(blob),sha256=digest_bytes(payload)))
                blob.extend(payload)
    (directory/'r3a_adapter.bin').write_bytes(blob)
    identity['adapter_matrix_sha256'] = sha(directory/'r3a_adapter.bin')
    def delta(layer,role,x): return linear(linear(x,matrices[layer,role,'A'],h,8),matrices[layer,role,'B'],8,h)
    logits = forward(identity['token_ids'],base,delta)
    (directory/'r3a_reference_logits.bin').write_bytes(pack(logits,'float32'))
    write(directory/'r3a_tiny_base.json',base)
    write(directory/'r3a_adapter_manifest.json',dict(identity=identity,adapter='user_a',rank=8,scale=1.,dtypes=['float32'],
        all_matrices_nonzero=True,trained_adapter=False,matrices=records,matrix_file='r3a_adapter.bin',matrix_sha256=sha(directory/'r3a_adapter.bin')))
    write(directory/'r3a_reference.json',dict(identity=identity,thresholds=THRESHOLDS,logits_file='r3a_reference_logits.bin',
        logits_sha256=sha(directory/'r3a_reference_logits.bin'),logits_dtype='float32',logits_shape=[1,length,vocab],native_projection_outputs=3*layers))
    write(directory/'r3a_preflight.json',dict(TEST_FIXTURE_ONLY=True,execution_status='COMPLETE',real_model=False))
    names = ['r3a_reference.json','r3a_adapter_manifest.json','r3a_reference_logits.bin','r3a_adapter.bin','r3a_tiny_base.json','r3a_preflight.json']
    write(directory/'r3a_prepare_manifest.json',dict(execution_status='COMPLETE',identity=identity,identity_sha256=digest(identity),
        code=source_identity(),created_at=now(),preparation_process_exited=True,
        artifact_hashes={n:sha(directory/n) for n in names},**flags(fixture=True)))
    return directory


if __name__ == '__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,required=True)
    print(prepare_tiny(p.parse_args().output_dir))
