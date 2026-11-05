"""CPU envelope predictions only. No change to frozen framing or request schemas."""
from .communication import canonical,header,payload_size,MAX_HEADER,require,digest
from .r5_i5_limits import wire_limits
from .r5_i5_contracts import namespace,request
from .r5_multi_artifacts import runtime_view
from .r5_i5_protocol import context
from .r5_i5_session import VERSION

HEADROOM=8192
LABEL='TRANSPORT_LIMITED_DERIVED_WORKLOAD'

def predicted_request(task,pm,spec,tokenizer):
 # Actual model/adapter/tokenizer metadata; only unknown runtime identity values
 # are conservative placeholders. Contract hashes always occupy 64 hex bytes.
 view=runtime_view(pm,spec)
 physical=namespace(view,tokenizer,'cuda:0','PREDICTED_GPU_UUID_'+'X'*110,{'prediction_only':True})
 return request(task,task['token_ids'],task['policy'],physical,fixture=False)

def envelope(req,users,prefix=512):
 # Invoke the unchanged production preflight, including its near-full-prefix case.
 base=wire_limits(req,users,'0'*64,2048)
 ids=req['token_ids'];policy=req['generation_policy'];sizes={}
 for n in ([0,prefix] if len(ids)>prefix else [0]):
  grant=None if not n else dict(entry_id='0'*64,source_request_id='P000000S',prefix_length=n,proof_sha256='0'*64)
  for step in range(policy['max_new_tokens']):
   tokens=ids[n:] if not step else [50271] # conservative five-digit OPT decode ID
   past=n if not step else len(ids)+step-1
   ctx=context(req['request_id'],step,tokens,ids,past,policy,'PARTIAL_PREFIX_HIT' if n else 'COLD_MISS','0'*64,grant,context_limit=2048)
   h=header('HIDDEN',req['request_id']+f'.s{step:02d}',31,req['physical_identity']['dtype'],[1,len(tokens),req['physical_identity']['hidden_size']],r4c_forward=ctx,r5_multi=dict(schema=VERSION,registry_sha256=users['sha256'],request=req))
   h.update(session_id='0'*32,adapter_sha256='0'*64,sequence=999999,payload_bytes=payload_size(h['dtype'],h['shape']),payload_sha256='0'*64)
   sizes[('prefix_512' if n else 'cold')+f'_step_{step}']=len(canonical(h))
 maximum=max(base['maximum_header_bytes'],*sizes.values())
 require(maximum+HEADROOM<=MAX_HEADER,f'TCP_HEADER_HEADROOM: predicted={maximum}, reserve={HEADROOM}, limit={MAX_HEADER}; explicit new plan required; no automatic retry')
 return dict(production_wire_limits=base,scenario_header_bytes=sizes,maximum_header_bytes=maximum,header_limit=MAX_HEADER,remaining_headroom_bytes=MAX_HEADER-maximum,required_headroom_bytes=HEADROOM,maximum_utilization=maximum/MAX_HEADER,predicted_identity='Actual model/adapter/tokenizer; placeholder GPU UUID longer than expected; digest lengths fixed; not an execution receipt',scope='ENVELOPE_ONLY_NOT_A_CACHE_GRANT')

def preflight(plans,pm,specs,tokenizer,users):
 rows=[]
 for control,plan in plans.items():
  for ordinal,block in enumerate(plan['blocks']):
   for task in block['tasks']:
    req=predicted_request(task,pm,specs[task['adapter_id']],tokenizer)
    rows.append(dict(control=control,block_index=ordinal,request_id=task['request_id'],sample_id=task['sample_id'],role=task['role'],input_tokens=len(task['token_ids']),token_ids_sha256=digest(task['token_ids']),conditions=list(plan['conditions']),**envelope(req,users)))
 return dict(rows=rows,maximum_header_bytes=max(r['maximum_header_bytes'] for r in rows),maximum_header_utilization=max(r['maximum_utilization'] for r in rows),required_headroom_bytes=HEADROOM,header_limit=MAX_HEADER,scope='CPU_PREDICTED; runtime preflight remains unchanged and mandatory')
