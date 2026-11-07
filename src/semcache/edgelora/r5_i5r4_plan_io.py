"""I4-only content-addressed JSON shards; historical Evidence limits unchanged."""
import json
import re
from .communication import canonical, digest, require
from .r3a_artifacts import write

LIMIT=16*2**20
SHARD=256*2**10
TAG='$i4_shard'
FORMAT='I4_SHARDED_PLAN_V1'

def checked_json(ev,name):
    size=ev.files.get(name)
    require(size is not None, 'missing I4 JSON artifact: '+name)
    require(size<=LIMIT,f'I4 JSON artifact {name}: measured={size} bytes, limit={LIMIT} bytes; prepare a fresh sharded plan; never edit sealed evidence')
    return ev.json(name)

def write_plan(root,plan):
    """Deduplicate repeated contexts, samples and tasks without changing values."""
    shards=root/'plan_shards';shards.mkdir()
    seen=set()
    def pack(value):
        if isinstance(value,dict):
            require(TAG not in value and '$i4_string_chunks' not in value,'reserved I4 shard key')
            node={k:pack(v) for k,v in value.items()}
        elif isinstance(value,list):node=[pack(v) for v in value]
        elif isinstance(value,str) and len(value.encode())>SHARD:
            node={'$i4_string_chunks':[pack(value[i:i+32768]) for i in range(0,len(value),32768)]}
        elif isinstance(value,str) and len(value.encode())>4096:node=value
        else:return value
        if not isinstance(node,str) and not (isinstance(value,dict) and ('token_ids' in value or 'context' in value)) and len(canonical(node))<SHARD:return node
        key=digest(node)
        if key not in seen:
            require(len(seen)<8192,'I4 shard count exceeds bound')
            require(len(canonical(node))+1<=LIMIT,'I4 individual shard exceeds JSON bound')
            write(shards/(key+'.json'),node);seen.add(key)
        return {TAG:key}
    payload=pack(plan)
    write(root/'i4_plan.json',dict(storage_format=FORMAT,payload=payload,logical_plan_sha256=digest(plan)))
    require((root/'i4_plan.json').stat().st_size<=LIMIT,'I4 plan index exceeds JSON bound')
    require(sum(p.stat().st_size for p in root.rglob('*.json'))<120*2**20,'I4 packed plan exceeds bounded inventory budget')

def read_plan(ev):
    index=checked_json(ev,'i4_plan.json')
    if index.get('storage_format')!=FORMAT:return index # Small legacy plans retain strict verification.
    expanded=0;nodes=0;active=set()
    def unpack(node,depth=0):
        nonlocal expanded,nodes
        nodes+=1;require(nodes<=8000000,'I4 reconstructed node count exceeds bound')
        require(depth<=128,'I4 shard nesting exceeds bound')
        if isinstance(node,dict) and TAG in node:
            key=node[TAG];require(set(node)=={TAG} and isinstance(key,str) and re.fullmatch('[0-9a-f]{64}',key),'invalid I4 shard reference')
            require(key not in active,'cyclic I4 shard reference');active.add(key)
            raw=checked_json(ev,'plan_shards/'+key+'.json')
            require(digest(raw)==key,'I4 shard content identity differs')
            value=unpack(raw,depth+1);active.remove(key);return value
        if isinstance(node,dict):
            if set(node)=={'$i4_string_chunks'}:return ''.join(unpack(v,depth+1) for v in node['$i4_string_chunks'])
            return {k:unpack(v,depth+1) for k,v in node.items()}
        if isinstance(node,list):return [unpack(v,depth+1) for v in node]
        expanded+=len(canonical(node));require(expanded<=512*2**20,'I4 reconstructed plan exceeds 512 MiB bound')
        return node
    plan=unpack(index['payload'])
    require(digest(plan)==index['logical_plan_sha256'],'I4 logical plan identity differs')
    return plan
