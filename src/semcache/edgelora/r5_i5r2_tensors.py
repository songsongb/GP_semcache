"""Bounded, aligned historical tensor comparisons. No Torch/GPU imports."""
import hashlib
import math
from pathlib import Path,PurePosixPath
import struct

from .communication import digest,require
from .r3a_artifacts import THRESHOLDS
from .r5_i5_protocol import validate_context
from .r5_i5r0_io import parse

UNKNOWN='NOT_OBSERVABLE_FROM_AVAILABLE_EVIDENCE'
DTYPES={'float16':('<e',2),'float32':('<f',4)}


def safe_path(root,name):
    require(isinstance(name,str) and name and '\\' not in name and ':' not in name,'unsafe evidence path')
    p=PurePosixPath(name)
    require(not p.is_absolute() and p.as_posix()==name and all(x not in ('','.','..') for x in name.split('/')),'unsafe evidence path')
    root=Path(root).resolve();path=root/name
    require(path.resolve().is_relative_to(root) and not any(x.is_symlink() for x in (path,*path.parents) if x!=root and x.is_relative_to(root)),
            'evidence symlink/path escape')
    return path


def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda:stream.read(1024*1024),b''):h.update(data)
    return h.hexdigest()


class Evidence:
    """Verifies every sealed byte, but only reads bounded tensors into memory."""
    def __init__(self,root,*,max_tensor_bytes=32*2**20,max_input_bytes=32*2**30):
        self.root=Path(root)
        require(not self.root.is_symlink() and self.root.is_dir(),'missing/unsafe evidence directory: '+str(root))
        self.root=self.root.resolve();self.max_tensor_bytes=max_tensor_bytes;self.max_input_bytes=max_input_bytes
        require(0<max_tensor_bytes<=128*2**20 and 0<max_input_bytes<=128*2**30,'invalid evidence resource bounds')
        self.files={};total=0
        for path in sorted(self.root.rglob('*')):
            require(not path.is_symlink(),'evidence symlink forbidden')
            if path.is_file():
                name=path.relative_to(self.root).as_posix();safe_path(self.root,name)
                total+=path.stat().st_size;require(total<=max_input_bytes,'input evidence exceeds configured byte bound')
                self.files[name]=path.stat().st_size
        self.total_bytes=total
    def json(self,name):
        p=safe_path(self.root,name);require(name in self.files,'missing required evidence: '+name)
        require(self.files[name]<=16*2**20,'JSON metadata exceeds 16 MiB bound')
        return parse(p.read_bytes())
    def jsonl(self,name,*,optional=False):
        if name not in self.files:
            require(optional,'missing required evidence: '+name);return []
        require(self.files[name]<=128*2**20,'JSONL metadata exceeds 128 MiB bound')
        records=[]
        with safe_path(self.root,name).open('rb') as stream:
            while True:
                line=stream.readline(16*2**20+1)
                if not line:break
                require(len(line)<=16*2**20,'JSONL record exceeds 16 MiB bound')
                if line.strip():records.append(parse(line))
        return records
    def verify(self,manifest='i5_output_manifest.json'):
        inventory=self.json(manifest)['files']
        require(set(inventory)==set(self.files)-{manifest},'sealed evidence inventory differs')
        for name,expected in sorted(inventory.items()):
            require(file_sha(safe_path(self.root,name))==expected,'corrupt sealed evidence: '+name)
        self.inventory=inventory
        return dict(status='SHA256_INVENTORY_PASS',manifest=manifest,manifest_sha256=file_sha(self.root/manifest),
                    files=len(inventory),bytes=self.total_bytes)
    def validate_tensor_reference(self,base,item,*,required_file=False):
        require(item['dtype'] in DTYPES,'unsupported tensor dtype')
        shape=item['shape'];require(1<=len(shape)<=4 and all(type(n) is int and n>0 for n in shape),'invalid tensor shape')
        size=math.prod(shape)*DTYPES[item['dtype']][1]
        require(size<=self.max_tensor_bytes,'tensor exceeds configured byte bound')
        if 'file' not in item:
            require(not required_file,'tensor bytes unavailable; hash-only evidence cannot supply values');return None
        relative=(PurePosixPath(base)/item['file']).as_posix();safe_path(self.root,item['file'])
        p=safe_path(self.root,relative);require(relative in self.files,'missing tensor evidence: '+relative)
        require(self.files[relative]==size,'tensor shape/dtype/byte count differs')
        if hasattr(self,'inventory'):require(self.inventory[relative]==item['sha256'],'corrupt tensor reference: '+relative)
        return p
    def tensor(self,base,item):
        p=self.validate_tensor_reference(base,item,required_file=True)
        relative=p.relative_to(self.root).as_posix()
        data=p.read_bytes();require(hashlib.sha256(data).hexdigest()==item['sha256'],'corrupt tensor evidence: '+relative)
        return data


def slice_axis(data,shape,axis,start,stop,width):
    require(type(axis) is int and 0<=axis<len(shape) and 0<=start<stop<=shape[axis],'incompatible tensor position slice')
    inner=math.prod(shape[axis+1:])*width;outer=math.prod(shape[:axis]);stride=shape[axis]*inner
    selected=b''.join(data[i*stride+start*inner:i*stride+stop*inner] for i in range(outer))
    result=list(shape);result[axis]=stop-start
    return selected,result


def metrics(reference,actual,dtype,*,allow_negative_infinity=False):
    """Chunked float reductions, bounded independently of tensor element count."""
    fmt,width=DTYPES[dtype];require(reference and len(reference)==len(actual) and len(reference)%width==0,'tensor byte counts differ')
    squared=[];norms=[];absolute=[];max_abs=0.;violations=finite=unequal=nonfinite_mismatch=0;first_violation=None
    for offset in range(0,len(reference),65536*width):
        a=struct.iter_unpack(fmt,reference[offset:offset+65536*width]);b=struct.iter_unpack(fmt,actual[offset:offset+65536*width])
        sq=[];norm=[];ab=[]
        for index,((x,),(y,)) in enumerate(zip(a,b),start=offset//width):
            unequal+=x!=y
            if not (math.isfinite(x) and math.isfinite(y)):
                require(allow_negative_infinity and all(math.isfinite(v) or v==-math.inf for v in (x,y)),'unexpected nonfinite tensor value')
                nonfinite_mismatch+=x!=y;continue
            error=abs(y-x);max_abs=max(max_abs,error);finite+=1
            violation=error>THRESHOLDS['atol']+THRESHOLDS['rtol']*abs(x);violations+=violation
            if violation and first_violation is None:first_violation=index
            sq.append(error*error);norm.append(x*x);ab.append(error)
        squared.append(math.fsum(sq));norms.append(math.fsum(norm));absolute.append(math.fsum(ab))
    relative=math.sqrt(math.fsum(squared))/max(math.sqrt(math.fsum(norms)),1e-30)
    first_byte=next((i for i,(x,y) in enumerate(zip(reference,actual)) if x!=y),None) if reference!=actual else None
    return dict(bitwise_equal=reference==actual,max_abs=max_abs if not nonfinite_mismatch else None,
        mean_abs=math.fsum(absolute)/finite if finite and not nonfinite_mismatch else None,
        relative_l2=relative if not nonfinite_mismatch else None,elementwise_tolerance_violations=violations,
        nonfinite_pattern_mismatches=nonfinite_mismatch,finite_aligned_elements=finite,elements=len(reference)//width,differing_values=unequal,
        first_bitwise_difference_element=None if first_byte is None else first_byte//width,first_tolerance_violation_element=first_violation,
        activation_tolerance_pass=not nonfinite_mismatch and violations==0 and max_abs<=THRESHOLDS['max_abs'] and relative<=THRESHOLDS['relative_l2'],
        reference_slice_sha256=hashlib.sha256(reference).hexdigest(),actual_slice_sha256=hashlib.sha256(actual).hexdigest(),
        tolerance_scope='Frozen atol/rtol/max_abs/relative_l2; activation checks do not substitute for frozen logit KL/argmax gates')


def align(reference,actual,*,relation,prefix,history_reference=(),history_actual=()):
    """Return query token intervals in stored arrays; never align unrelated tokens."""
    a=validate_context(reference);b=validate_context(actual)
    require(a['pass_index']==b['pass_index'],'unmatched generation step')
    step=a['pass_index']
    if relation=='FULL_PREFIX_VS_SOURCE':
        require(step==0 and a['past_length']==b['past_length']==0 and len(b['target_token_ids'])==prefix>0 and
                b['target_token_ids']==a['target_token_ids'][:prefix],'standalone/full-target prefix differs')
        left=right=(0,prefix)
    else:
        require(relation in ('FULL_SUFFIX_VS_HIT','MATCHED') and a['target_token_ids']==b['target_token_ids'] and a['policy']==b['policy'],
                'matched target token/policy differs')
        require(list(history_reference[:step])==list(history_actual[:step]) and len(history_reference)>=step and len(history_actual)>=step,
                'unmatched generated history')
        if step:
            require(a['input_token_ids']==b['input_token_ids'],'unmatched decode input');left=right=(0,1)
        elif relation=='FULL_SUFFIX_VS_HIT':
            require(a['past_length']==0 and b['past_length']==prefix and b['input_token_ids']==a['input_token_ids'][prefix:],'suffix/full token alignment differs')
            left=(prefix,a['input_length']);right=(0,b['input_length'])
        else:
            require(a['past_length']==b['past_length'] and a['input_token_ids']==b['input_token_ids'],'matched input context differs')
            left=right=(0,a['input_length'])
    require(a['positions'][left[0]:left[1]]==b['positions'][right[0]:right[1]] and
            a['cache_positions'][left[0]:left[1]]==b['cache_positions'][right[0]:right[1]],'semantic positions differ')
    require((a['attention_mask'][:prefix]==b['attention_mask']) if relation=='FULL_PREFIX_VS_SOURCE' else a['attention_mask']==b['attention_mask'],
            'attention-mask contexts differ')
    return left,right


def compare(evidence,a,base_a,b,base_b,*,axis_a=1,axis_b=1,range_a=None,range_b=None,key_range=None):
    require(a['dtype']==b['dtype'],'incompatible tensor dtypes')
    x=evidence.tensor(base_a,a);y=evidence.tensor(base_b,b);sx=list(a['shape']);sy=list(b['shape']);width=DTYPES[a['dtype']][1]
    if range_a is not None:x,sx=slice_axis(x,sx,axis_a,*range_a,width)
    if range_b is not None:y,sy=slice_axis(y,sy,axis_b,*range_b,width)
    if key_range is not None:
        x,sx=slice_axis(x,sx,len(sx)-1,*key_range,width);y,sy=slice_axis(y,sy,len(sy)-1,*key_range,width)
    require(sx==sy,'incompatible aligned tensor shapes')
    result=metrics(x,y,a['dtype'],allow_negative_infinity=a.get('allow_negative_infinity',False) or b.get('allow_negative_infinity',False))
    def position(index,axis,interval):
        if index is None:return None
        return (index//math.prod(sx[axis+1:]))%sx[axis]+(interval[0] if interval else 0)
    return dict(aligned_shape=sx,dtype=a['dtype'],reference=dict(shape=a['shape'],stride=a.get('observed_stride',UNKNOWN),device=a['device'],
        sha256=a['sha256'],token_axis=axis_a,token_interval=range_a),actual=dict(shape=b['shape'],stride=b.get('observed_stride',UNKNOWN),device=b['device'],
        sha256=b['sha256'],token_axis=axis_b,token_interval=range_b),coverage='ALL_ALIGNED_ELEMENTS',sampling=False,
        first_differing_stored_token_indices=dict(reference=position(result['first_bitwise_difference_element'],axis_a,range_a),
            actual=position(result['first_bitwise_difference_element'],axis_b,range_b)),**result)


def verdict(records,*,field='bitwise_equal'):
    if not records:return UNKNOWN
    return 'PASS' if all(r['metrics'].get(field) is True for r in records) else 'FAIL'


def first(records):
    rank={'layer_input':0,'norm1':1,'projection_input':2,**{x:3 for x in ('base_q','base_k','base_v','delta_q','delta_k','delta_v')},
          **{x:4 for x in ('q','k','v','raw_q','raw_k','raw_v')},'attention_scores':5,'attention_masked_scores':6,'attention_probabilities':7,
          'attention_context':8,'attention':9,'attention_output':9,'attention_residual':10,'norm2':11,'fc1':12,'activation':13,
          'ffn':14,'ffn_residual':15,'layer':16,'layer_output':16}
    values=[r for r in records if r['metrics'].get('bitwise_equal') is False]
    if not values:return dict(status='NO_OBSERVED_DIFFERENCE',scope='Only captured and aligned tensors; missing boundaries are not exonerated')
    key=lambda r:(r.get('step',0),r['layer'],rank.get(r['role'],100))
    found=min(values,key=key);return dict(status='FIRST_OBSERVED_DIFFERENCE',record=found,
        causal_peers=[dict(layer=r['layer'],role=r['role'],family=r['family']) for r in values if key(r)==key(found)],
        scope='Partial computational order, not cross-condition wall-clock order or a proven root cause')
