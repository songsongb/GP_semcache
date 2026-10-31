"""Versioned multi-adapter artifacts. Legacy user_a validation remains frozen."""
from copy import deepcopy
import math
from pathlib import Path
import random

from .communication import digest, digest_bytes, require
from .r3a_artifacts import ROOT, pack, read, sha, tensor_from_bytes, unpack, validate_prepared, write
from .r4c_prepare import TinyTokenizer, opt_tokenizer
from .r5_contracts import namespace
from .r5_evidence import seal, verify

PREP='R5_MULTI_PREPARED_V1'
ADAPTER='R5_MULTI_ADAPTER_V1'
REGISTRY='R5_MULTI_REGISTRY_V1'
SEEDS={'Adapter_A':104729,'Adapter_B':130363,'Adapter_C':155921}
METHOD='CONTROLLED_SYNTHETIC_MT19937_UNIFORM_F32_V1'


def fresh(path,protected=()):
    path=Path(path).resolve()
    require(not path.is_relative_to(ROOT) or path.is_relative_to(ROOT/'results/r5/multiuser'),'I4A outputs belong under results/r5/multiuser')
    require(all(not path.is_relative_to(Path(p).resolve()) for p in protected if p),'output overlaps input')
    require(not path.exists() or path.is_dir() and not any(path.iterdir()),'refusing nonempty output directory')
    path.mkdir(parents=True,exist_ok=True);path.chmod(0o700);return path


def tokenizer_identity(pm,tokenizer=None):
    if pm.get('TEST_FIXTURE_ONLY'):
        return dict(scope='TEST_FIXTURE_ONLY_MOD5_CHAR_TOKENIZER',vocab_sha256=digest(list(range(7))),tokenizer_revision='TEST_FIXTURE_ONLY')
    return opt_tokenizer(pm,tokenizer).identity


def registry_for(adapter_ids,stage='pilot'):
    require(stage in ('pilot','expanded'),'unsupported integration stage')
    require(list(adapter_ids)==list(SEEDS)[:2 if stage=='pilot' else 3],'stage adapter coverage mismatch')
    per_adapter=2 if stage=='pilot' else 10
    users={}
    for a,aid in enumerate(adapter_ids):
        for j in range(per_adapter):users[f'User_{a*per_adapter+j+1:02d}']=dict(adapter_id=aid,share_group='shared_'+aid,allow_cross_user_reuse=True)
    value=dict(schema=REGISTRY,stage=stage,policy='EXPLICIT_SHARED_ADAPTER_GROUP_V1',users=users)
    return dict(value,sha256=digest(value))


def validate_registry(registry,adapter_ids):
    value=dict(registry);seal_value=value.pop('sha256')
    require(value.get('schema')==REGISTRY and value.get('policy')=='EXPLICIT_SHARED_ADAPTER_GROUP_V1' and digest(value)==seal_value,'corrupt user registry')
    require(value['users'] and len(value['users'])<=30,'bounded logical users required')
    for user,m in value['users'].items():
        require(isinstance(user,str) and user and set(m)=={'adapter_id','share_group','allow_cross_user_reuse'} and
                m['adapter_id'] in adapter_ids and isinstance(m['share_group'],str) and m['share_group'] and
                type(m['allow_cross_user_reuse']) is bool,'invalid user/adapter authorization')
    return registry


def resolve(registry,user,claimed=None):
    m=registry['users'].get(user)
    require(m is not None and (claimed is None or claimed==m['adapter_id']),'UNAUTHORIZED_USER_OR_ADAPTER')
    return m['adapter_id']


def sharing(registry,source_user,target_user,source_adapter,target_adapter):
    if source_adapter!=target_adapter:return 'CROSS_ADAPTER_REUSE_FORBIDDEN'
    a=registry['users'].get(source_user);b=registry['users'].get(target_user)
    if a is None or b is None or a['adapter_id']!=source_adapter or b['adapter_id']!=target_adapter:return 'UNAUTHORIZED_USER_OR_ADAPTER'
    if source_user==target_user:return None
    if a['share_group']!=b['share_group'] or not a['allow_cross_user_reuse'] or not b['allow_cross_user_reuse']:return 'SHARING_NOT_AUTHORIZED'
    return None


def prepare(base_prepared,output,stage='pilot'):
    output=fresh(output,(base_prepared,));base_prepared=Path(base_prepared).resolve();pm,_,_=validate_prepared(base_prepared)
    i=pm['identity'];tokenizer=tokenizer_identity(pm);records={}
    require(stage in ('pilot','expanded'),'unsupported adapter preparation stage')
    for aid,seed in list(SEEDS.items())[:2 if stage=='pilot' else 3]:
        rng=random.Random(seed);blob=bytearray();matrices=[]
        for layer in range(i['layers']):
            for role in 'qkv':
                for matrix,shape in (('A',[8,i['hidden_size']]),('B',[i['hidden_size'],8])):
                    payload=pack((rng.uniform(-.05,.05) for _ in range(math.prod(shape))),'float32')
                    matrices.append(dict(layer=layer,role=role,matrix=matrix,shape=shape,dtype='float32',offset=len(blob),sha256=digest_bytes(payload)))
                    blob.extend(payload)
        (output/(aid+'.bin')).write_bytes(blob)
        spec=dict(schema=ADAPTER,adapter_id=aid,rank=8,lora_alpha=8,scale=1.,matrix_dtype='float32',layers=i['layers'],
            hidden_size=i['hidden_size'],roles=list('qkv'),layout='A[rank,hidden];B[hidden,rank];total_qkv',
            source_seed=seed,preparation_method=METHOD,trained_adapter=False,classification='CONTROLLED_SYNTHETIC_RESEARCH_ADAPTER_NOT_PERSONALIZED',
            base_fingerprint=i['base_weight_sha256'],model_revision=i['base_revision'],tokenizer_sha256=digest(tokenizer),
            matrix_file=aid+'.bin',matrix_sha256=digest_bytes(blob),matrices=matrices,TEST_FIXTURE_ONLY=pm.get('TEST_FIXTURE_ONLY',False))
        write(output/(aid+'.json'),spec);records[aid]=sha(output/(aid+'.json'))
    registry=registry_for(records,stage);write(output/'registry.json',registry)
    manifest=dict(schema=PREP,stage=stage,base_prepared=str(base_prepared),base_manifest_sha256=sha(base_prepared/'r3a_prepare_manifest.json'),
        base_identity=pm['identity'],tokenizer=tokenizer,adapters=records,registry_sha256=sha(output/'registry.json'),
        TEST_FIXTURE_ONLY=pm.get('TEST_FIXTURE_ONLY',False))
    write(output/'multi_prepare.json',manifest);validate(output,sealed=False);seal(output)
    return output


def validate_matrix(path,spec):
    aid=spec['adapter_id'];require(aid in SEEDS and spec['schema']==ADAPTER,'unsupported adapter artifact')
    require(spec['rank']==spec['lora_alpha']==8 and spec['scale']==1. and spec['matrix_dtype']=='float32' and spec['roles']==list('qkv') and
            spec['layout']=='A[rank,hidden];B[hidden,rank];total_qkv' and spec['source_seed']==SEEDS[aid] and
            spec['preparation_method']==METHOD and spec['trained_adapter'] is False and
            spec['classification']=='CONTROLLED_SYNTHETIC_RESEARCH_ADAPTER_NOT_PERSONALIZED','unsupported LoRA contract')
    require(spec['matrix_file']==aid+'.bin','unsafe adapter matrix path')
    blob=(Path(path)/spec['matrix_file']).read_bytes();require(len(blob)<=128*2**20 and digest_bytes(blob)==spec['matrix_sha256'],'incorrect matrix loading/fingerprint')
    offset=0;keys=set()
    for r in spec['matrices']:
        key=(r['layer'],r['role'],r['matrix']);shape=[8,spec['hidden_size']] if r['matrix']=='A' else [spec['hidden_size'],8]
        require(type(r['layer']) is int and 0<=r['layer']<spec['layers'] and r['role'] in ('q','k','v') and r['matrix'] in ('A','B') and
                key not in keys and r['shape']==shape and r['dtype']=='float32' and r['offset']==offset,'invalid adapter matrix layout/coverage')
        size=math.prod(shape)*4;part=blob[offset:offset+size];offset+=size;keys.add(key)
        require(len(part)==size and digest_bytes(part)==r['sha256'] and any(unpack(part,'float32')),'corrupt or zero adapter matrix')
    require(offset==len(blob) and len(keys)==spec['layers']*6,'incomplete adapter matrices')
    return blob


def validate(path,*,sealed=True):
    path=Path(path).resolve()
    if sealed:verify(path)
    m=read(path/'multi_prepare.json')
    stage=m.get('stage','pilot')
    require(stage in ('pilot','expanded') and m['schema']==PREP and list(m['adapters'])==sorted(SEEDS)[:2 if stage=='pilot' else 3],
            'unsupported multi-adapter preparation')
    pm,_,_=validate_prepared(m['base_prepared'])
    if not m['TEST_FIXTURE_ONLY']:
        require(pm['identity']['model']=='facebook/opt-2.7b' and pm['identity']['dtype']=='float16' and pm['identity']['hidden_dtype']=='float16' and
                pm['identity']['layers']==32 and pm['identity']['hidden_size']==2560 and pm['identity']['seed']==42,'I4A requires frozen OPT-2.7B FP16 base')
    require(sha(Path(m['base_prepared'])/'r3a_prepare_manifest.json')==m['base_manifest_sha256'] and
            m['base_identity']==pm['identity'] and m['TEST_FIXTURE_ONLY']==pm.get('TEST_FIXTURE_ONLY',False),'multi-adapter base identity differs')
    adapters={}
    for aid,expected in m['adapters'].items():
        require(sha(path/(aid+'.json'))==expected,'corrupt adapter manifest')
        s=read(path/(aid+'.json'));validate_matrix(path,s)
        require(s['adapter_id']==aid and s['base_fingerprint']==pm['identity']['base_weight_sha256'] and
                s['model_revision']==pm['identity']['base_revision'] and s['layers']==pm['identity']['layers'] and
                s['hidden_size']==pm['identity']['hidden_size'] and s['tokenizer_sha256']==digest(m['tokenizer']) and
                s['TEST_FIXTURE_ONLY']==m['TEST_FIXTURE_ONLY'],'adapter/base/tokenizer compatibility mismatch')
        adapters[aid]=s
    fps=[s['matrix_sha256'] for s in adapters.values()]
    require(len(set(fps))==len(fps) and pm['identity']['adapter_matrix_sha256'] not in fps,'adapters must be genuinely distinct; no legacy renaming')
    require(sha(path/'registry.json')==m['registry_sha256'],'corrupt registry artifact')
    registry=validate_registry(read(path/'registry.json'),adapters)
    require(registry.get('stage','pilot')==stage,'registry/preparation stage mismatch')
    return m,pm,adapters,registry


def load(path,spec,*,torch_backend):
    blob=validate_matrix(path,spec);result={}
    for r in spec['matrices']:
        size=math.prod(r['shape'])*4;part=blob[r['offset']:r['offset']+size]
        result[r['layer'],r['role'],r['matrix']]=tensor_from_bytes(part,'float32',r['shape']) if torch_backend else unpack(part,'float32')
    return result


def runtime_view(pm,spec):
    # This is a validated view for the shared math/storage primitives, never a
    # forged legacy prepared artifact and never passed to validate_prepared.
    i=dict(pm['identity'],adapter=spec['adapter_id'],adapter_matrix_sha256=spec['matrix_sha256'],rank=spec['rank'],delta_dtype=spec['matrix_dtype'])
    return dict(schema='R5_MULTI_VALIDATED_RUNTIME_VIEW_V1',identity=i,identity_sha256=digest(i))


def physical(view,tokenizer,device,gpu,contract,spec,registry):
    contract=deepcopy(contract)
    contract['r5_multi']=dict(schema=ADAPTER,adapter_manifest_sha256=digest(spec),rank=spec['rank'],scale=spec['scale'],
        matrix_dtype=spec['matrix_dtype'],layout=spec['layout'],registry_sha256=registry['sha256'],cache_format='UNCHANGED_R3B_TOTAL_RAW_QKV',
        sharing_policy=registry['policy'])
    return namespace(view,tokenizer,device,gpu,contract),contract
