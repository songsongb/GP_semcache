"""R5 versioned application contracts. Logical users never select adapter weights."""
from copy import deepcopy
import re

from .communication import digest, digest_bytes, require
from .r4c_workload import policy

REQUEST_SCHEMA = 'R5_REQUEST_V1'
ARTIFACT_SCHEMA = 'R5_ARTIFACT_V1'
OUTCOMES = ('CACHE_MISS', 'PHYSICAL_PREFIX_HIT', 'SEMANTIC_CANDIDATE_REJECTED',
            'ADMISSION_DENIED', 'CAPACITY_EVICTION', 'INVALID_CACHE_INTEGRITY')


def namespace(pm, tokenizer, device, gpu_uuid, contract):
    i = pm['identity']
    return dict(schema='R5_PHYSICAL_IDENTITY_V1', model=i['model'],
        model_revision=i.get('base_revision', 'TEST_FIXTURE_ONLY'),
        base_fingerprint=i['base_weight_sha256'], tokenizer=deepcopy(tokenizer),
        adapter_id=i['adapter'], adapter_matrix_fingerprint=i['adapter_matrix_sha256'],
        dtype=i['hidden_dtype'], layers=i['layers'], hidden_size=i['hidden_size'],
        rank=i['rank'], device=str(device), gpu_uuid=gpu_uuid, opt_contract=digest(contract),
        attention_context='standalone causal; all-ones mask; positions from zero')


def request(raw, token_ids, p, physical_identity, *, fixture=False):
    ids = list(token_ids)
    row = dict(schema=REQUEST_SCHEMA, **{k:raw[k] for k in
        ('request_id','user_id','adapter_id','input_text','dataset','sample_id','request_sequence_number')},
        token_ids=ids, attention_mask=[1]*len(ids), positions=list(range(len(ids))),
        generation_policy=deepcopy(p), physical_identity=deepcopy(physical_identity),
        input_text_sha256=digest_bytes(raw['input_text'].encode()),
        token_ids_sha256=digest(ids), TEST_FIXTURE_ONLY=fixture)
    row['request_sha256'] = digest(row)
    return validate_request(row)


def validate_request(row):
    required = {'schema','request_id','user_id','adapter_id','input_text','dataset','sample_id',
        'request_sequence_number','token_ids','attention_mask','positions','generation_policy',
        'physical_identity','input_text_sha256','token_ids_sha256','TEST_FIXTURE_ONLY','request_sha256'}
    require(set(row)==required and row['schema']==REQUEST_SCHEMA, 'unsupported R5 request schema')
    require(isinstance(row['request_id'],str) and re.fullmatch(r'[A-Za-z0-9_:-]{1,60}',row['request_id']), 'unsafe request ID')
    require(all(isinstance(row[k],str) and row[k] for k in ('user_id','adapter_id','input_text','dataset','sample_id')), 'missing request identity/text')
    require(type(row['request_sequence_number']) is int and row['request_sequence_number']>=0, 'invalid request sequence')
    ids=row['token_ids']; require(0<len(ids)<=512 and all(type(t) is int and t>=0 for t in ids), 'invalid bounded tokens')
    require(row['attention_mask']==[1]*len(ids) and row['positions']==list(range(len(ids))), 'unsupported mask/position context')
    p=row['generation_policy']
    require(p==policy({k:p[k] for k in ('bos_token_id','eos_token_id','pad_token_id')},p['mode'],p['max_new_tokens']), 'invalid generation policy')
    copy=dict(row);seal=copy.pop('request_sha256')
    require(digest(copy)==seal and row['input_text_sha256']==digest_bytes(row['input_text'].encode()) and row['token_ids_sha256']==digest(ids), 'request content seal differs')
    n=row['physical_identity']
    fields={'schema','model','model_revision','base_fingerprint','tokenizer','adapter_id','adapter_matrix_fingerprint',
        'dtype','layers','hidden_size','rank','device','gpu_uuid','opt_contract','attention_context'}
    require(set(n)==fields and n['schema']=='R5_PHYSICAL_IDENTITY_V1', 'invalid physical identity schema')
    require(all(isinstance(n[k],str) and re.fullmatch('[0-9a-f]{64}',n[k]) for k in
        ('base_fingerprint','adapter_matrix_fingerprint','opt_contract')), 'invalid physical fingerprint')
    require(n['rank']==8 and all(type(n[k]) is int and n[k]>0 for k in ('layers','hidden_size')) and
        n['dtype'] in ('float32','float16') and isinstance(n['tokenizer'],dict) and n['tokenizer'] and
        n['attention_context']=='standalone causal; all-ones mask; positions from zero','unsupported physical layout/context')
    require(n['adapter_id']==row['adapter_id'], 'unprepared adapter identity')
    if not row['TEST_FIXTURE_ONLY']:
        require(n['model']=='facebook/opt-2.7b' and n['dtype']=='float16' and n['layers']==32 and n['hidden_size']==2560 and n['device'].startswith('cuda') and
            re.fullmatch('[0-9a-f]{40}',n['model_revision']) and
            re.fullmatch('[0-9a-f]{40}',n['tokenizer']['tokenizer_revision']), 'immutable OPT/tokenizer revision required')
    return row


def prompt(row):
    return dict(token_ids=list(row['token_ids']),attention_mask=list(row['attention_mask']),
        positions=list(row['positions']),prompt_sha256=digest(dict(tokens=row['token_ids'],mask=row['attention_mask'])))


def gates(fixture=False):
    names=('R5_I0_CONTRACT_PASS','R5_I1_SEMANTIC_PATH_PASS','R5_I2_CACHE_LIFECYCLE_PASS',
        'R5_I3_DISTRIBUTED_GENERATION_PASS','R5_CPU_FIXTURE_PASS','R5_GPU_PHYSICAL_EXECUTION_PASS',
        'R5_INDEPENDENT_AUDIT_PASS','R5_TOKEN_FIDELITY_PASS','R5_STRICT_BITWISE_FIDELITY_PASS',
        'NATIVE_PEFT_STRICT_PARITY_PASS','correctness_qualified_speedup')
    return dict({n:False for n in names}, schema='R5_DECISION_V1',TEST_FIXTURE_ONLY=fixture,
        R5_I4_FULL_MULTI_ADAPTER_NOT_IMPLEMENTED=True, R5_I4_CONCURRENT_MULTI_USER_NOT_IMPLEMENTED=True,
        R5_I5_LARGE_SCALE_EVALUATION_NOT_RUN=True)
