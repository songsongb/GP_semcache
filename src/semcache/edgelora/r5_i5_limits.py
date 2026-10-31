"""Account for unchanged TCP payload/header bounds before the first inference request."""
from .communication import canonical,header,payload_size,MAX_HEADER,require
from .r5_i5_protocol import context
from .r5_i5_session import VERSION


def wire_limits(req, registry, cohort, context_limit):
    maxima=[];ids=req['token_ids'];p=req['generation_policy'];hidden=req['physical_identity']['hidden_size']
    for prefix in sorted({0,len(ids)-1}):
        grant=None if not prefix else dict(entry_id='0'*64,source_request_id='P000000S',prefix_length=prefix,proof_sha256='0'*64)
        c=context(req['request_id'],0,ids[prefix:],ids,prefix,p,'COLD_MISS' if not prefix else 'PARTIAL_PREFIX_HIT',cohort,grant,context_limit=context_limit)
        h=header('HIDDEN',req['request_id']+'.s00',0,req['physical_identity']['dtype'],[1,len(ids)-prefix,hidden],
            r4c_forward=c,r5_multi=dict(schema=VERSION,registry_sha256=registry['sha256'],request=req))
        h.update(session_id='0'*32,adapter_sha256='0'*64,sequence=999999,payload_bytes=payload_size(h['dtype'],h['shape']),payload_sha256='0'*64)
        maxima.append(len(canonical(h)))
    delta=payload_size('float32',[3,1,len(ids),hidden])
    require(max(maxima)<=MAX_HEADER, f'TCP_HEADER_LIMIT: required={max(maxima)}, limit={MAX_HEADER}; reduce input limit; no silent truncation')
    return dict(maximum_header_bytes=max(maxima),header_limit=MAX_HEADER,maximum_delta_payload_bytes=delta)
