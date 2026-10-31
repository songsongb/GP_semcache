"""I5 adapter preparation destination; unchanged I4A matrices and validation schema."""
from .r5_multi_artifacts import (SEEDS,METHOD,ADAPTER,PREP,registry_for,tokenizer_identity,validate,seal)
from .r5_i5_evidence import fresh
from .r3a_artifacts import validate_prepared,pack,write,sha
from .communication import digest,digest_bytes,require
from pathlib import Path
import math
import random

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

