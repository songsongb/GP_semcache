"""Independent value primitive and read-only Phase 2.5A static feasibility.

Not a full model/GPU numerical auditor. Never authorize from static token counts.
"""
from collections import Counter
from pathlib import Path
import json
import math

from .communication import require, digest, digest_bytes, canonical
from .r3a_tiny import f32
from .r3a_artifacts import pack, unpack, THRESHOLDS
from .r5_i5_contracts import validate_request
from .r5_multi_artifacts import sharing
from .r5_subsequence_audit import rows
from .r5_subsequence_natural_io import verify, put
from .r5_layer_input_arithmetic import REFERENCE, SCALAR
from .r5_layer_input_physical import MODE, SCHEMA


def replay_projection(ev, record, req, physical, sources):
    """Replay saved construction, source auth/input equality, and output bytes.

    ev implements tensor('es', spec). sources maps completed request IDs to
    {request, projections[(layer,role)], completion_control}. Caller must also
    verify inventories, wire/UD completion and numerical model evidence. This
    primitive does not prove base GEMM values correct from model weights.
    """
    validate_request(req)
    require(record['schema'] == physical['schema'] == SCHEMA and record['mode'] == physical['mode'] == MODE and
            record['request_sha256'] == req['request_sha256'] and record['request_id'] == req['request_id'], 'verified record/request differs')
    arithmetic = record['arithmetic']; guard = arithmetic['reference_guard']
    require(arithmetic['contract'] in (REFERENCE,SCALAR) and
            (guard and arithmetic['contract'] == REFERENCE or not guard and arithmetic['contract'] == SCALAR and req['TEST_FIXTURE_ONLY']),
            'unsupported replay arithmetic contract')
    inp, out = record['input'], record['output']; x, y = ev.tensor('es',inp), ev.tensor('es',out)
    n,h = inp['shape'][1:]; dtype = inp['dtype']; width = 2 if dtype == 'float16' else 4
    require(dtype in ('float16','float32') and inp['shape'] == out['shape'] == [1,n,h] and out['dtype'] == dtype and
            n == len(req['token_ids']) and h == req['physical_identity']['hidden_size'], 'replay input/output shape/dtype differs')
    require(all(math.isfinite(v) for v in unpack(x,dtype)+unpack(y,dtype)), 'nonfinite replay projection')
    compute = record['fresh_positions']; output_fresh = record['output_fresh_positions']
    require(compute == sorted(set(compute)) and output_fresh == sorted(set(output_fresh)) and
            all(0 <= p < n for p in compute+output_fresh) and record['past_kv_grant'] is False, 'invalid projection coordinates')
    require(record['native_projection_rows'] == record['cpu_lora_rows'] == len(compute) and
            record['projection_rows_skipped'] == n-len(compute) and
            record['reference_projection_rows'] == (n if guard else 0), 'false work accounting')
    computed = b''
    if compute:
        base, delta = record['fresh_base'], record['fresh_delta']
        require(base['shape'] == delta['shape'] == [1,len(compute),h] and base['dtype'] == dtype and delta['dtype'] == 'float32', 'reference/fresh arithmetic shape differs')
        computed = pack([f32(a+b) for a,b in zip(unpack(ev.tensor('es',base),dtype),unpack(ev.tensor('es',delta),'float32'))],dtype)
    if guard:
        require(compute == list(range(n)) and ev.tensor('es',record['full_shape_reference']) == computed == y,
                'paid full-shape reference differs')
    positions_in_computed = {p:j for j,p in enumerate(compute)}
    for p in output_fresh:
        require(p in positions_in_computed, 'uncomputed output row')
        j = positions_in_computed[p]
        require(y[p*h*width:(p+1)*h*width] == computed[j*h*width:(j+1)*h*width], 'fresh row value differs')
    authorizations = {}
    for auth in physical['integrity_validation']['authorizations']:
        body = dict(auth); aid = body.pop('authorization_id')
        require(digest(body) == aid and aid not in authorizations and auth['request_sha256'] == req['request_sha256'] and
                auth['request_epoch'] == physical['integrity_validation']['request_epoch'], 'authorization digest/epoch differs')
        authorizations[aid] = auth
    used = set(output_fresh); counts = Counter({c:0 for c in 'ABCD'})
    for item in record['consumed']:
        sid = item['source_request_id']; require(sid in sources, 'unknown/uncompleted source')
        source = sources[sid]; sr = validate_request(source['request']); control = source['completion_control']; projections = source['projections']
        require(control['request'] == sr and sr['physical_identity'] == req['physical_identity'] and
                sharing(ev.registry,sr['user_id'],req['user_id'],sr['adapter_id'],req['adapter_id']) is None, 'source ownership/completion differs')
        from .r4c_workload import termination
        require(termination(sr['generation_policy'],control['generated_token_ids']) == control['termination'] and control['termination'] is not None,
                'source termination differs')
        require(all(not r['consumed'] and r['mode'] == MODE and r['arithmetic'] == arithmetic for r in projections.values()),
                'transitive/foreign source provenance')
        lo,hi = item['source_positions']; start,end = item['target_positions']
        require(hi-lo == end-start == 3 and 0 <= lo < hi <= len(sr['token_ids']) and 0 <= start < end <= n and
                sr['token_ids'][lo:hi] == req['token_ids'][start:end], 'w3 tokens/coordinates differ')
        auth = authorizations[item['integrity_validation']['authorization_id']]
        proof_dtype = auth['tensors']['0'][0]['dtype']; tag = proof_dtype.removeprefix('torch.')
        require(tag == dtype, 'source tensor dtype differs')
        def proof(data): return dict(shape=[1,3,h],dtype=proof_dtype,sha256=digest_bytes(data))
        inputs = {}; tensors = {}; input_bytes = 0
        for layer in range(req['physical_identity']['layers']):
            specs = [projections[layer,r]['input'] for r in 'qkv']
            data = [rows(ev.tensor('es',s),s['shape'],s['dtype'],list(range(lo,hi))) for s in specs]
            require(data[0] == data[1] == data[2], 'source QKV input contract differs')
            inputs[str(layer)] = proof(data[0]); input_bytes += len(data[0]); tensors[str(layer)] = []
            for r in 'qkv':
                spec = projections[layer,r]['output']
                tensors[str(layer)].append(proof(rows(ev.tensor('es',spec),spec['shape'],spec['dtype'],list(range(lo,hi)))))
        extra = dict(schema=SCHEMA,mode=MODE,arithmetic=arithmetic,source_request_sha256=sr['request_sha256'],
            completion_sha256=digest(control),source_root='COMPLETED_REQUEST_WITH_NO_CACHED_CONSUMPTION',
            qkv_input_contract='ACTUAL_FULL_INPUT_BYTES_EQUAL_ALL_THREE_ROLES',transitive_reuse=False)
        meta = dict(component_scope='total_qkv',completed=True,provenance_mode=MODE,physical_identity=sr['physical_identity'],
            source_user=sr['user_id'],source_adapter=sr['adapter_id'],source_request_id=sid,source_token_ids=sr['token_ids'],
            input_proofs=inputs,layer_input_verified=extra)
        size = 3*3*req['physical_identity']['layers']*h*width+input_bytes+len(canonical(extra))
        expected_header = dict(metadata=meta,positions=[lo,hi],tokens=sr['token_ids'][lo:hi],size=size)
        require(digest(auth['header']) == digest(expected_header) and auth['tensors'] == tensors and
                auth['admission_proof'] == digest(dict(expected_header,tensors=tensors)), 'source admission proof does not reproduce')
        layer,role = record['layer'],record['role']; original = projections[layer,role]
        source_x = rows(ev.tensor('es',original['input']),original['input']['shape'],dtype,list(range(lo,hi)))
        target_x = rows(x,inp['shape'],dtype,list(range(start,end)))
        require(source_x == target_x and item['source_input_proof'] == item['target_input_proof'] == proof(source_x), 'false layer input equality')
        snapshot = ev.tensor('es',item['tensor'])
        source_y = rows(ev.tensor('es',original['output']),original['output']['shape'],dtype,list(range(lo,hi)))
        check = item['integrity_validation']; tensor_expected = tensors[str(layer)]['qkv'.index(role)]
        require(item['tensor']['shape'] == tensor_expected['shape'] and item['tensor']['dtype'] == dtype and
                item['tensor']['sha256'] == tensor_expected['sha256'], 'consumed tensor evidence spec differs')
        require(snapshot == source_y == rows(y,out['shape'],dtype,list(range(start,end))) and
                check['private_snapshot'] and check['expected'] == check['actual'] == tensor_expected and
                list(auth['header']['positions']) == [lo,hi] and digest(auth['key']) == digest(item['key']), 'consumed snapshot differs')
        prefix = 0
        for a,b in zip(sr['token_ids'],req['token_ids']):
            if a != b: break
            prefix += 1
        c = ('A' if end <= prefix else 'B') if (lo,hi) == (start,end) else ('D' if end <= prefix else 'C')
        require(c == item['category'] and not used & set(range(start,end)), 'category/overlap differs')
        counts[c] += 3; used.update(range(start,end))
    require(used == set(range(n)) and sorted(set(range(n))-set(output_fresh)) == record['reused_positions'], 'output coverage differs')
    if not guard: require(set(compute) == set(output_fresh), 'scalar work-skipping partition differs')
    return dict(status='CPU_PROJECTION_VALUE_REPLAY_PASS',reused_by_category=dict(counts),
                computed_rows=len(compute),skipped_rows=n-len(compute),
                full_model_numerical_validation='NOT_RUN',scope='value primitive; inventories/wire/UD/GEMM/full-model numerical audit still required')


def static_feasibility(phase2_plan, phase2_audit):
    """Use sealed Phase 2 results without invoking its source-pinned loader.

    Later implementation additions must not require resealing historical inputs.
    This reader verifies inventories, plan ID, request sharding/digests and audit
    linkage; it does not claim to rerun the prior full real-data validation.
    """
    require(not Path(phase2_plan).is_symlink() and not Path(phase2_audit).is_symlink(), 'source root symlink forbidden')
    phase2_plan=Path(phase2_plan).resolve();phase2_audit=Path(phase2_audit).resolve()
    if not phase2_plan.is_dir() or not phase2_audit.is_dir():
        return dict(status='BLOCKED',reason='existing sealed Phase 2 plan/audit unavailable locally')
    plan_receipt = verify(phase2_plan); audit_receipt = verify(phase2_audit)
    p = json.loads((Path(phase2_plan)/'phase2_plan.json').read_bytes())
    body = dict(p); pid = body.pop('plan_id')
    require(p['schema'] == 'D5_CONTINUOUS_NATURAL_PHASE2_CPU_V1' and digest(body) == pid and not p['TEST_FIXTURE_ONLY'] and
            p['context_mode'] == 'EXACT_CONTEXT_VERIFIED' and p['allow_approximate'] is False and p['thresholds'] == THRESHOLDS,
            'real sealed Phase 2 plan required')
    v = json.loads((Path(phase2_audit)/'phase2_validation.json').read_bytes())
    require(v['status'] == v['real_data_validation'] == 'PASS' and not v['TEST_FIXTURE_ONLY'] and
            v['plan_id'] == pid and v['plan_inventory'] == plan_receipt, 'Phase 2 audit/plan binding differs')
    from .r5_continuous_natural_phase2 import read_sessions
    sessions = read_sessions(phase2_plan); reports = []
    for session in sessions:
        name = session['dataset']; inv = session['invocations']
        require(digest(session) == p['session_digests'][name] and len(inv) == 50 and len({i['original_sample_id'] for i in inv}) == 40 and
                len({i['invocation_id'] for i in inv}) == 50, 'sealed Phase 2 session differs')
        sharing_report = json.loads((Path(phase2_audit)/name/'phase2/sharing_summary.json').read_bytes())
        scopes = {}
        for scope, expected in (('replay_inclusive',1225),('distinct_sample_only',1215),('first_occurrence_per_original_sample',780)):
            view = sharing_report['views'][scope]
            require(view['denominators']['ordered_request_pairs'] == expected,'static pair denominator differs')
            scopes[scope] = {category:view['axes']['aggregate']['authorized'][category]
                for category in ('same_position_nonprefix_windows','shifted_position_nonprefix_windows')}
            scopes[scope]['verified_document_to_document'] = {category:view['axes']['aggregate']['authorized_verified_context_to_context'][category]
                for category in ('same_position_nonprefix_windows','shifted_position_nonprefix_windows')}
        reports.append(dict(dataset=name,scopes=scopes,layer_input_equality_rate='NOT_MEASURED',resident_candidates='NOT_MEASURED',
            physical_reuse='NOT_MEASURED',limitations='one resident owner per gram/cluster; 64 entries/candidate scan; non-overlap selection; F/A/I/S residency not modeled'))
    return dict(status='REAL_DATA_STATIC_ANALYSIS_PASS',plan_id=pid,plan_inventory=plan_receipt,audit_inventory=audit_receipt,datasets=reports,
                scope='read-only extraction of verified Phase 2 static evidence; no runtime grant or new numerical validation')
