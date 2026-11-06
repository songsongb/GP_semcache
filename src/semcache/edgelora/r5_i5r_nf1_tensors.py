"""Bounded CPU-only mmap comparisons of sealed canonical tensor exports."""
import hashlib,json,math,mmap,os,struct
from pathlib import Path
from .communication import require
from .r5_i5r2_tensors import Evidence,DTYPES,file_sha,safe_path,metrics
from .r4b2_analysis import raw_head_bytes

UNKNOWN='NOT_OBSERVABLE_FROM_AVAILABLE_EVIDENCE'
MAX_FILES=100000

class StreamEvidence(Evidence):
 def __init__(self,root,max_input_bytes=64*2**30):
  root=Path(root);require(root.is_dir() and not root.is_symlink(),'missing/unsafe evidence directory: '+str(root))
  count=0
  for parent,dirs,files in os.walk(root,followlinks=False):
   require(not any((Path(parent)/n).is_symlink() for n in dirs+files),'evidence symlink forbidden')
   count+=len(files);require(count<=MAX_FILES,'evidence file-count bound exceeded')
  super().__init__(root,max_input_bytes=max_input_bytes,max_tensor_bytes=32*2**20)
 def records(self,name):
  require(name in self.files,'missing required JSONL: '+name)
  count=0
  with safe_path(self.root,name).open('rb') as stream:
   while True:
    line=stream.readline(16*2**20+1)
    if not line:break
    require(len(line)<=16*2**20,'JSONL line exceeds bound')
    if not line.strip():continue
    count+=1;require(count<=10000,'JSONL record bound exceeded')
    def pairs(items):
     result={}
     for k,v in items:require(k not in result,'duplicate JSON key');result[k]=v
     return result
    yield json.loads(line,object_pairs_hook=pairs,parse_constant=lambda x:(_ for _ in ()).throw(ValueError('nonfinite JSON')))

def aligned_ranges(a,b,relation,prefix=512,target=1280):
 require(a['pass_index']==b['pass_index']==0,'prefill comparison requires step zero')
 ta,tb=a['target_token_ids'],b['target_token_ids']
 if relation=='PREFIX':
  require(len(ta)==prefix and len(tb)==target and ta==tb[:prefix] and a['past_length']==b['past_length']==0,'source/full token prefix differs');left=right=(0,prefix)
 elif relation=='SUFFIX':
  require(ta==tb and len(ta)==target and a['past_length']==0 and b['past_length']==prefix and b['input_token_ids']==ta[prefix:],'full/HIT suffix context differs');left,right=(prefix,target),(0,target-prefix)
 else:
  require(relation=='MATCHED' and ta==tb and a['past_length']==b['past_length'] and a['input_token_ids']==b['input_token_ids'],'repeat/matched context differs');left=right=(0,len(a['input_token_ids']))
 require(a['positions'][left[0]:left[1]]==b['positions'][right[0]:right[1]] and a['cache_positions'][left[0]:left[1]]==b['cache_positions'][right[0]:right[1]],'absolute position mismatch')
 require(a['attention_mask']==[1]*len(ta) and b['attention_mask']==[1]*len(tb),'unsupported mask semantics')
 return left,right

def region(ev,base,item,start,stop,*,head_transform=False,heads=32):
 shape=item['shape'];require(item['dtype'] in DTYPES and len(shape) in (3,4) and shape[0]==1,'tensor dtype/rank/batch differs')
 require(0<=start<stop<=shape[-2],'tensor token range differs');width=DTYPES[item['dtype']][1]
 path=ev.validate_tensor_reference(base,item,required_file=True)
 require(file_sha(path)==item['sha256'],'tensor SHA changed before mmap')
 with path.open('rb') as f,mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ) as view:
  stride=shape[-1]*width;n=shape[-2];outer=1 if len(shape)==3 else shape[1]
  data=b''.join(view[(h*n+start)*stride:(h*n+stop)*stride] for h in range(outer))
 selected=list(shape);selected[-2]=stop-start
 if head_transform:
  require(len(shape)==3 and shape[-1]%heads==0,'raw-to-head layout differs')
  data=raw_head_bytes(data,selected,heads,width);selected=[1,heads,stop-start,shape[-1]//heads]
 return data,selected

def numerical(a,b,dtype):
 fmt,width=DTYPES[dtype]
 require(a and len(a)==len(b) and len(a)%width==0,'numerical byte counts differ')
 if a==b:
  # Exact zero errors after a real byte comparison and finite scan; avoid
  # allocating/reducing millions of zero error values for integrity controls.
  require(all(math.isfinite(v[0]) for v in struct.iter_unpack(fmt,a)),'unexpected nonfinite tensor value')
  result=metrics(a[:width],b[:width],dtype)
  result.update(elements=len(a)//width,finite_aligned_elements=len(a)//width,reference_slice_sha256=hashlib.sha256(a).hexdigest(),actual_slice_sha256=hashlib.sha256(b).hexdigest())
 else:result=metrics(a,b,dtype)
 if dtype=='float16':
  def ordered(v):return 0x8000-(v&0x7fff) if v&0x8000 else 0x8000+v
  total=maximum=nonzero=bitwise_differences=0
  for (x,),(y,) in (() if a==b else zip(struct.iter_unpack('<H',a),struct.iter_unpack('<H',b))):
   d=abs(ordered(x)-ordered(y));maximum=max(maximum,d);total+=d;nonzero+=d>0;bitwise_differences+=x!=y
  result['FP16_ULP']=dict(max=maximum,mean=total/(len(a)//2),nonzero=nonzero,scope='finite binary16 monotone encoding; signed zero distance zero')
  result['bitwise_differing_elements']=bitwise_differences
 result['differing_element_percentage']=100*result['differing_values']/result['elements']
 return result

def metadata(base,item,interval):
 return dict(base=base,shape=item['shape'],dtype=item['dtype'],device=item.get('device',UNKNOWN),sha256=item['sha256'],file=item.get('file',UNKNOWN),token_range=list(interval),observed_stride=item.get('observed_stride',UNKNOWN),storage_offset=item.get('storage_offset',UNKNOWN),serialization='canonical contiguous logical values; not proof of live layout')

def compare(ev,base_a,a,base_b,b,range_a,range_b,*,head_a=False,head_b=False,heads=32,required=True):
 require(a is not None and b is not None,'missing required tensor metadata')
 require(a['dtype']==b['dtype'],'tensor dtype mismatch')
 # Validate references even for hash-only records. Named missing/corrupt files
 # always fail; an intentionally omitted value is a coverage observation only.
 ev.validate_tensor_reference(base_a,a);ev.validate_tensor_reference(base_b,b)
 meta=dict(reference=metadata(base_a,a,range_a),candidate=metadata(base_b,b,range_b),coverage='ALL_ALIGNED_TOKENS',sampling=False)
 if 'file' not in a or 'file' not in b:
  require(not required,'required strict tensor values missing')
  full=range_a==(0,a['shape'][-2]) and range_b==(0,b['shape'][-2]) and not head_a and not head_b and a['shape']==b['shape']
  return dict(meta,status='HASH_ONLY_VALUES_UNAVAILABLE',bitwise_equal_by_full_sha256=a['sha256']==b['sha256'] if full else UNKNOWN,metrics=UNKNOWN)
 x,sx=region(ev,base_a,a,*range_a,head_transform=head_a,heads=heads);y,sy=region(ev,base_b,b,*range_b,head_transform=head_b,heads=heads)
 require(sx==sy,'aligned tensor shape mismatch')
 return dict(meta,status='OBSERVED_VALUES',aligned_shape=sx,metrics=numerical(x,y,a['dtype']))
