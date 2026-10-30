"""Actual tiny causal KV-cached CPU decoder, explicitly TEST_FIXTURE_ONLY."""
import math
from types import SimpleNamespace
from .communication import require
from .r3a_tiny import f32,linear
from .r3b_fixture import ArrayTensor
from .r4a_session import positions


class TinyBackend:
    def __init__(self,base,matrices=None):self.base=base;self.matrices=matrices;self.vocab=base['vocabulary'];self.heads=1;self.hidden=base['hidden']
    def forward(self,tokens,mask,past=None,project=None,use_cache=True):
        h=self.hidden;old=0 if past is None else past[0][0].shape[2];n=len(tokens)
        require(len(mask)==old+n,'fixture KV length/mask mismatch')
        x=[f32(self.base['embedding'][t*h+k]+positions(mask)[old+i]*.0001) for i,t in enumerate(tokens) for k in range(h)]
        updated=[]
        for layer,w in enumerate(self.base['layers']):
            out={}
            for role in 'qkv':
                def native(role=role):return linear(x,w[role],h,h)
                if project is not None:out[role]=project(layer,role,x,native)
                else:
                    a,b=self.matrices[layer,role,'A'],self.matrices[layer,role,'B']
                    d=linear(linear(x,a,h,8),b,8,h);out[role]=[f32(v+z) for v,z in zip(native(),d)]
            keys=([] if past is None else list(past[layer][0].values))+out['k']
            vals=([] if past is None else list(past[layer][1].values))+out['v'];attended=[]
            for i in range(n):
                visible=[j for j in range(old+i+1) if mask[j]]
                scores=[sum(out['q'][i*h+k]*keys[j*h+k] for k in range(h))/math.sqrt(h) for j in visible]
                exps=[] if not scores else [math.exp(s-max(scores)) for s in scores];total=sum(exps)
                attended.extend(f32(math.tanh(x[i*h+k]+(sum(e/total*vals[j*h+k] for j,e in zip(visible,exps)) if total else 0.))) for k in range(h))
            x=attended;updated.append((ArrayTensor(keys,[1,1,old+n,h]),ArrayTensor(vals,[1,1,old+n,h])))
        logits=linear(x,self.base['head'],h,self.vocab)
        return SimpleNamespace(logits=logits,shape=[1,n,self.vocab],past_key_values=tuple(updated) if use_cache else None)
    def decode_text(self,tokens):return ' '.join('fixture_token_'+str(t) for t in tokens)
