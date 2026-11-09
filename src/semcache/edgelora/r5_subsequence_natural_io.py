"""NATURAL-only bounded sharding. Historical Evidence/serializers stay unchanged."""
import json
from pathlib import Path
from .communication import require
from .r3a_artifacts import ROOT,write
from .r5_i5r2_tensors import Evidence,file_sha,safe_path
from .r5_i5r0_io import parse
from .r4c_evidence import Evidence as Writer

MANIFEST='natural_inventory.json'
JSON_LIMIT=16*2**20
MAX_BYTES=64*2**30
MAX_FILES=500000
NAMESPACE=ROOT/'results/r5/i5r/d5_natural'

def output(path,protected=()):
    p=Path(path).resolve()
    require(not Path(path).is_symlink() and not p.exists(),'fresh NATURAL output required')
    require(not p.is_relative_to(ROOT) or p.is_relative_to(NAMESPACE),'new D5 NATURAL namespace required')
    require(all(not p.is_relative_to(Path(x).resolve()) and not Path(x).resolve().is_relative_to(p) for x in protected),'output overlaps input')
    p.mkdir(parents=True);return p

def put(path,value):
    data=json.dumps(value,sort_keys=True,indent=2,allow_nan=False).encode()
    require(len(data)<=JSON_LIMIT,'NATURAL JSON exceeds16MiB: '+str(path))
    Path(path).parent.mkdir(parents=True,exist_ok=True);Path(path).write_bytes(data)

def seal(root):
    """Hash immediate children; seal every subdirectory, no giant flat manifest."""
    root=Path(root);require(not (root/MANIFEST).exists(),'refusing reseal')
    files={};dirs={}
    for p in sorted(root.iterdir()):
        require(not p.is_symlink(),'symlink evidence forbidden')
        if p.is_dir():
            if not (p/MANIFEST).exists():seal(p)
            dirs[p.name]=file_sha(p/MANIFEST)
        else:files[p.name]=file_sha(p)
    put(root/MANIFEST,dict(schema='D5_NATURAL_TREE_V1',files=files,directories=dirs))

def verify(root):
    root=Path(root);count=total=0
    def visit(path,depth=0):
        nonlocal count,total
        require(depth<=16,'evidence directory depth exceeded')
        require(not path.is_symlink() and path.is_dir(),'missing/unsafe shard')
        m=path/MANIFEST;require(m.is_file() and m.stat().st_size<=JSON_LIMIT,'missing/oversized shard inventory')
        count+=1;total+=m.stat().st_size
        require(count<=MAX_FILES and total<=MAX_BYTES,'NATURAL evidence budget exceeded')
        record=parse(m.read_bytes());require(record['schema']=='D5_NATURAL_TREE_V1','unknown shard seal')
        require(set(record['files']).isdisjoint(record['directories']),'file/directory collision')
        require({p.name for p in path.iterdir()}==set(record['files'])|set(record['directories'])|{MANIFEST},'shard inventory differs')
        for name,sha in record['files'].items():
            p=safe_path(path,name);require(p.parent==path and p.is_file() and not p.is_symlink() and file_sha(p)==sha,'corrupt shard file: '+str(p))
            count+=1;total+=p.stat().st_size
            require(count<=MAX_FILES and total<=MAX_BYTES,'NATURAL evidence budget exceeded')
            if p.suffix=='.json':require(p.stat().st_size<=JSON_LIMIT,'NATURAL JSON exceeds16MiB: '+str(p))
        for name,sha in record['directories'].items():
            p=safe_path(path,name);require(p.parent==path and not p.is_symlink() and file_sha(p/MANIFEST)==sha,'corrupt child seal');visit(p,depth+1)
    visit(root)
    return dict(manifest_sha256=file_sha(root/MANIFEST),files=count,bytes=total)

class RequestWriter(Writer):
    """One physical writer; request-directory paths returned to original replay."""
    def __init__(self,*args,request_ids=(),**kwargs):
        super().__init__(*args,**kwargs);self.request_ids=tuple(request_ids);self.current=None
    def blob(self,data,name):
        rid=self.current
        if rid is None:rid=next((r for r in self.request_ids if Path(name).name.startswith(r+'_')),None)
        require(rid in self.request_ids,'unbound evidence request')
        return super().blob(data,rid+'/'+name)

def save_result(root,result):
    """Remove only redundant metadata; every per-pass receipt remains sealed."""
    rid=result['request_id'];steps=result.pop('steps');receipts=result.pop('sparse_receipts')
    require(receipts==[r for s in steps for r in s['record']['sparse']],'step receipt coverage differs')
    names=[]
    for i,step in enumerate(steps):
        name=rid+f'/step_{i:03d}.json';put(Path(root)/name,step);names.append(name)
    result['step_files']=names;put(Path(root)/rid/'result.json',result)

def load_result(root,rid):
    root=Path(root);result=parse(safe_path(root,rid+'/result.json').read_bytes())
    require(result['request_id']==rid and 1<=len(result['step_files'])<=128,'invalid request result shard')
    result['steps']=[];used=0
    for i,name in enumerate(result.pop('step_files')):
        require(name==rid+f'/step_{i:03d}.json','step order/path differs')
        p=safe_path(root,name);used+=p.stat().st_size
        require(p.stat().st_size<=JSON_LIMIT and used<=256*2**20,'request metadata budget exceeded')
        result['steps'].append(parse(p.read_bytes()))
    result['sparse_receipts']=[r for s in result['steps'] for r in s['record']['sparse']]
    return result
