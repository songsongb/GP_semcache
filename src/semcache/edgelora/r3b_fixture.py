"""Explicit stdlib-only TEST_FIXTURE_ONLY tensors and causal decoder.

Not an OPT/FP16/GPU substitute. Exercises actual CacheEntry/GlobalCache and TCP.
"""
import math
from .r3a_artifacts import pack
from .r3a_tiny import linear,f32


class ArrayTensor:
    def __init__(self,values,shape):
        self.values=list(values);self.shape=tuple(shape);self.ndim=len(shape);self.dtype='float32';self.device='cpu'
        assert len(values)==math.prod(shape)
    def detach(self):return self
    def to(self,device):
        if str(device)!='cpu':raise ValueError('fixture CPU only')
        return self
    def clone(self):return ArrayTensor(list(self.values),self.shape)
    def contiguous(self):return self
    def is_contiguous(self):return True
    def numel(self):return len(self.values)
    def element_size(self):return 4
    def fixture_bytes(self):return pack(self.values,'float32')
    def untyped_storage(self):
        tensor=self
        class Storage:
            def data_ptr(self):return id(tensor.values)
            def nbytes(self):return len(tensor.values)*4
        return Storage()


def forward(tokens,base,projection):
    """Same scalar decoder equations; HIT bypasses base Linear as well as UD."""
    h=base['hidden'];n=len(tokens);x=[base['embedding'][token*h+j] for token in tokens for j in range(h)]
    for layer,weights in enumerate(base['layers']):
        outputs={role:projection(layer,role,x,lambda role=role:linear(x,weights[role],h,h)) for role in 'qkv'}
        attended=[]
        for i in range(n):
            scores=[sum(outputs['q'][i*h+k]*outputs['k'][j*h+k] for k in range(h))/math.sqrt(h) for j in range(i+1)]
            maximum=max(scores);exp=[math.exp(s-maximum) for s in scores];total=sum(exp)
            attended.extend(f32(math.tanh(x[i*h+k]+sum(e/total*outputs['v'][j*h+k] for j,e in enumerate(exp)))) for k in range(h))
        x=attended
    return linear(x,base['head'],h,base['vocabulary'])
