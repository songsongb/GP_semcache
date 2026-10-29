"""Read-only R3-A failed-pilot audit. Intentionally excluded from r3a_*.py.

No successful-run validator is weakened: this audits an explicitly FAILED run,
and never makes its data eligible for the normal pilot/latency path.
"""
import csv
import math
from pathlib import Path

from .communication import PREFIX, canonical, digest, json_loads, payload_size, require, validate
from .r3a_artifacts import ROOT, THRESHOLDS, load_matrices, read, sha, source_identity, validate_prepared

CLASSIFICATION = 'R3A_PARITY_DIAGNOSTIC'
CONCLUSIONS = {'REFERENCE_NONREPRODUCIBLE','GPU_RECONSTRUCTION_MISMATCH','CPU_GPU_DELTA_NUMERICS',
               'WIRE_BYTES_OR_LAYOUT_MISMATCH','ES_ASSEMBLY_MISMATCH','UNRESOLVED'}


class Evidence:
    def __init__(self):
        self.files = {}

    def file(self, path, expected=None):
        path = Path(path).resolve()
        record = self.files.setdefault(str(path), dict(path=str(path), validity='UNVERIFIED'))
        try:
            actual = sha(path)
            require('sha256' not in record or record['sha256']==actual, 'evidence changed while reading: '+str(path))
            record.update(sha256=actual,expected_sha256=expected or record.get('expected_sha256'))
            if expected is not None:
                require(actual==expected, 'evidence SHA mismatch: '+str(path))
            record['validity']='VALID'
        except (OSError,ValueError) as exc:
            record.update(validity='INVALID',reason=str(exc))
            raise
        return path

    def read(self, path, expected=None): return read(self.file(path,expected))

    def jsonl(self, path):
        return [json_loads(line) for line in self.file(path).read_bytes().splitlines() if line.strip()]

    def unchanged(self):
        for name, record in self.files.items():
            require(record['validity']=='VALID' and sha(name)==record['sha256'], 'input changed during diagnostic: '+name)


def verify_prepared(prepared, evidence):
    prepared=Path(prepared).resolve()
    pm=evidence.read(prepared/'r3a_prepare_manifest.json')
    recorded=pm['code']['files']
    require(pm['code']['sha256']==digest(recorded), 'recorded production source-map digest mismatch')
    require(recorded==source_identity()['files'],
            'untouched original source hashes required; restore the recorded checkout or obtain a fresh prepare; never rewrite provenance')
    for name, expected in recorded.items():
        path=(ROOT/name).resolve()
        require(path.is_relative_to(ROOT), 'source path outside repository')
        evidence.file(path,expected)
    # Use the unchanged production gate, including its source identity checks.
    checked,reference,adapter=validate_prepared(prepared)
    require(pm==checked, 'prepare manifest changed')
    for name,expected in pm['artifact_hashes'].items(): evidence.file(prepared/name,expected)
    matrices=load_matrices(prepared,adapter,torch_backend=False)
    require(len(matrices)==pm['identity']['layers']*6,'incomplete original matrix evidence')
    del matrices
    return pm,reference,adapter


def compare_values(candidate, reference, *, atol=None, rtol=None):
    """Scalar CPU counterpart for metric tests; relative L2 uses reference norm."""
    require(len(candidate)==len(reference) and len(reference)>0, 'comparison shape mismatch')
    require(all(math.isfinite(v) for v in candidate+reference), 'nonfinite comparison')
    atol=THRESHOLDS['atol'] if atol is None else atol
    rtol=THRESHOLDS['rtol'] if rtol is None else rtol
    errors=[abs(a-b) for a,b in zip(candidate,reference)]
    norm=math.sqrt(sum(v*v for v in reference))
    return dict(max_abs=max(errors),relative_l2=math.sqrt(sum(v*v for v in errors))/max(norm,1e-30),
        unequal_elements=sum(a!=b for a,b in zip(candidate,reference)),
        elementwise_threshold_failures=sum(e>atol+rtol*abs(v) for e,v in zip(errors,reference)),
        reference_norm=norm,candidate_norm=math.sqrt(sum(v*v for v in candidate)),elements=len(reference))


def first_difference(records, key):
    return next((dict(layer=r['layer'],role=r.get('role'),evidence=r[key]) for r in records if r.get(key) is False),None)


def audit_failure(prepared, pilot, evidence=None):
    evidence=evidence if evidence is not None else Evidence()
    prepared,pilot=Path(prepared).resolve(),Path(pilot).resolve()
    pm,reference,adapter=verify_prepared(prepared,evidence)
    m=evidence.read(pilot/'r3a_manifest.json')
    require(m['execution_status']=='FAILED' and m['configuration']['phase']=='pilot', 'explicit FAILED pilot required')
    require(m['prepared_manifest_sha256']==sha(prepared/'r3a_prepare_manifest.json') and
            m['identity_sha256']==pm['identity_sha256'] and m['code']['files']==pm['code']['files'] and
            m['code']['sha256']==pm['code']['sha256'], 'pilot/prepared identity mismatch')
    required={'r3a_es_result.json','r3a_ud_result.json','r3a_environment.json','r3a_iterations.jsonl',
        'r3a_wire_trace.jsonl','r3a_ud_wire_trace.jsonl','r3a_parity.json','r3a_latency_raw.csv','r3a_lifecycle.json','r3a_decision.json'}
    require(required.issubset(m['artifact_hashes']), 'required failed-pilot artifacts/hashes missing')
    for name,expected in m['artifact_hashes'].items():
        require(Path(name).name==name, 'invalid pilot artifact path')
        evidence.file(pilot/name,expected)
    es=evidence.read(pilot/'r3a_es_result.json'); ud=evidence.read(pilot/'r3a_ud_result.json')
    environment=evidence.read(pilot/'r3a_environment.json')
    require(environment.get('es')==es and environment.get('ud')==ud, 'environment/result copies disagree')
    require(es['execution_status']=='FAILED' and 'PARITY_FAIL' in es.get('error','') and
            es['pid']!=ud['pid'] and es['identity_sha256']==ud['identity_sha256']==pm['identity_sha256'],
            'not a verified isolated ES parity failure')
    require(ud['adapter_matrix_sha256']==adapter['matrix_sha256'] and ud['user_device_cuda_used'] is False and
            es['actual_network_socket_used'] is ud['actual_network_socket_used'] is True, 'placement/adapter/socket evidence mismatch')
    fixture=pm.get('TEST_FIXTURE_ONLY') is True
    require(es['gpu_inference_performed'] is (not fixture), 'GPU/fixture provenance mismatch')
    iterations=evidence.jsonl(pilot/'r3a_iterations.jsonl')
    require(len(iterations)==1, 'audit contract requires exactly the single failing first warmup; no omitted observations')
    item=iterations[0];row,parity,layers=item['row'],item['parity'],item['layers']
    require((row['phase'],row['repeat'],row['request_id'])==('warmup',0,'warmup_000000') and row['parity_passed'] is False,
            'first failing warmup identity mismatch')
    require(parity['thresholds']==THRESHOLDS and parity['passed'] is False and parity['status']=='PARITY_FAIL', 'frozen strict parity gate mismatch')
    require(parity['max_abs']>THRESHOLDS['max_abs'] or parity['relative_l2']>THRESHOLDS['relative_l2'] or
            parity['tolerance_violations']>0 or parity['last_token_kl']>THRESHOLDS['last_token_kl'] or
            parity['last_token_argmax_agreement'] is False, 'recorded failure contradicts its metrics')
    require(es['rows']==[row] and es['parities']==[parity] and
            es['per_layer_trace']==[dict(request_id=row['request_id'],layers=layers)], 'ES raw/result copies disagree')
    parity_file=evidence.read(pilot/'r3a_parity.json')
    require(parity_file['observations']==[parity] and parity_file['parity_passed'] is False and
            parity_file['thresholds']==THRESHOLDS, 'parity copies disagree')
    with evidence.file(pilot/'r3a_latency_raw.csv').open() as stream: csv_rows=list(csv.DictReader(stream))
    require(csv_rows==[{k:'' if v is None else str(v) for k,v in row.items()}], 'CSV/raw copies disagree')
    n=pm['identity']['layers'];shape=[1,pm['identity']['length'],pm['identity']['hidden_size']]
    require(set(reference.get('projections',{}))=={f'{layer}.{role}' for layer in range(n) for role in 'qkv'},
            'complete original per-layer/role reference hashes required; cannot impute missing reference evidence')
    require([l['layer_id'] for l in layers]==list(range(n)) and row['layers_observed']==n and
            row['qkv_outputs_observed']==3*n and row['es_pid']==es['pid'] and row['ud_pid']==ud['pid'], 'incomplete forward/PID accounting')
    frames=evidence.jsonl(pilot/'r3a_wire_trace.jsonl');ud_frames=evidence.jsonl(pilot/'r3a_ud_wire_trace.jsonl')
    complete=[f for f in frames if f['complete']];ud_complete=[f for f in ud_frames if f['complete']]
    require(len(complete)==len(ud_complete)==2*n+2 and len(complete)==len(frames), 'unexpected missing/extra/incomplete ES frames')
    require([(f['operation'],f['header']['kind']) for f in complete[:2]]==[('send','HELLO'),('recv','READY')], 'bad handshake')
    for ef,uf in zip(complete,ud_complete):
        validate(ef['header'],ef['header']['payload_bytes'])
        require(ef['operation']!=uf['operation'] and ef['pid']==es['pid'] and uf['pid']==ud['pid'] and
                all(ef[k]==uf[k] for k in ('header','frame_sha256','wire_bytes')), 'ES/UD frame digest/metadata mismatch')
        require(ef['wire_bytes']==PREFIX.size+len(canonical(ef['header']))+ef['header']['payload_bytes'], 'wire byte accounting mismatch')
        require(ef['header']['adapter_sha256']==adapter['matrix_sha256'], 'wire adapter identity mismatch')
    for operation in ('send','recv'):
        sequence=[f['header']['sequence'] for f in complete if f['operation']==operation]
        require(sequence==list(range(len(sequence))), 'out-of-order complete-frame evidence')
    require(len({f['header']['session_id'] for f in complete})==1,'mixed sessions')
    hidden_comparisons=[]
    for l in layers:
        layer=l['layer_id'];sent,received=complete[2+2*layer:4+2*layer]
        require(l['sent_frame']==sent and l['received_frame']==received and l['qkv_hidden_input_equal']==dict(q=True,k=True,v=True),
                'layer/role trace mismatch')
        for f,kind,dtype,tshape,operation in ((sent,'HIDDEN',pm['identity']['hidden_dtype'],shape,'send'),
                (received,'DELTAS',pm['identity']['delta_dtype'],[3]+shape,'recv')):
            h=f['header']
            require(h['kind']==kind and h['request_id']==row['request_id'] and h['layer_id']==layer and
                    h['dtype']==dtype and h['shape']==tshape and h['payload_bytes']==payload_size(dtype,tshape) and
                    f['operation']==operation, 'actual layer/request/tensor schedule mismatch')
            require(row['es_start_monotonic_ns']<=f['monotonic_start_ns']<=f['monotonic_end_ns']<=row['es_end_monotonic_ns'],
                    'data frame outside original request interval')
        require(l['ud']==received['header']['metadata'] and l['hidden_payload_sha256']==sent['header']['payload_sha256'] and
                l['delta_payload_sha256']==received['header']['payload_sha256'] and
                l['ud']['hidden_payload_sha256']==l['hidden_payload_sha256'], 'hidden/delta provenance mismatch')
        for role in 'qkv':
            native=reference.get('projections',{}).get(f'{layer}.{role}')
            original=None if native is None else native['hidden_sha256']
            hidden_comparisons.append(dict(layer=layer,role=role,native_hidden_sha256=original,
                es_hidden_sha256=l['hidden_payload_sha256'],bitwise_equal=None if original is None else original==l['hidden_payload_sha256'],
                error_magnitude_status='NOT_ESTIMABLE_FROM_SHA_ONLY'))
    require(ud['layers']==[dict(request_id=row['request_id'],layer_id=l['layer_id'],metrics=l['ud'],
        hidden_frame_sha256=l['sent_frame']['frame_sha256'],delta_frame_sha256=l['received_frame']['frame_sha256']) for l in layers],
        'UD layer result copies disagree')
    counts={}
    for direction,operation in (('ES_TO_UD','send'),('UD_TO_ES','recv')):
        block=[f for f in complete[2:] if f['operation']==operation]
        prefix='es_to_ud' if operation=='send' else 'ud_to_es'
        counts[direction]=dict(data_frames=len(block),payload_bytes=sum(f['header']['payload_bytes'] for f in block),
                               wire_bytes=sum(f['wire_bytes'] for f in block))
        require(row[prefix+'_frames']==len(block)==n and row[prefix+'_payload_bytes']==counts[direction]['payload_bytes'] and
                row[prefix+'_wire_bytes']==counts[direction]['wire_bytes'],'directional row/trace counts mismatch')
    cleanup=evidence.read(pilot/'r3a_lifecycle.json')
    require(cleanup==m['cleanup'] and {c['pid'] for c in cleanup['children']}=={es['pid'],ud['pid']}, 'cleanup PID/evidence mismatch')
    errors=[f for f in ud_frames if not f['complete']]
    shutdown_supported=(len(errors)==1 and errors[0]['operation']=='recv' and errors[0]['wire_bytes']==0 and
        errors[0]['header']=={} and 'truncated frame' in (errors[0].get('error') or '') and
        ud_frames[-1]==errors[0] and errors[0]['monotonic_start_ns']>=ud_complete[-1]['monotonic_end_ns'] and
        errors[0]['monotonic_end_ns']>=row['es_end_monotonic_ns'] and len(ud['layers'])==n)
    first=first_difference(hidden_comparisons,'bitwise_equal')
    evidence.unchanged()
    return dict(status='COMPLETE',evidence_validity='VERIFIED_RECORDED_FAILED_PILOT',TEST_FIXTURE_ONLY=fixture,
        prepared_manifest_sha256=sha(prepared/'r3a_prepare_manifest.json'),reference_identity=pm['identity'],
        original_source_hashes=pm['code']['files'],pilot_manifest_sha256=sha(pilot/'r3a_manifest.json'),
        process_metadata=dict(es_pid=es['pid'],ud_pid=ud['pid'],es_status=es['execution_status'],ud_status=ud['execution_status'],
            gpu_inference_performed=es['gpu_inference_performed'],user_device_cuda_used=ud['user_device_cuda_used'],
            original_es_gpu=es.get('gpu'),original_ud_cpu=ud['cpu'],
            original_es_environment=es.get('environment'),original_ud_environment=ud.get('environment')),
        parity=parity,actual_schedule=dict(layers=n,qkv_outputs=3*n,directions=counts),cleanup=cleanup,
        hidden_comparisons=hidden_comparisons,first_bitwise_hidden_divergence=first,
        localization='between layer-0 hidden ingress and layer-1 hidden ingress' if first and first['layer']==1 and
            all(c['bitwise_equal'] is True for c in hidden_comparisons if c['layer']==0) else 'see per-role hidden evidence; no numeric magnitude inferred',
        root_failure='STRICT_LOGIT_PARITY_FAIL',numerical_cause='UNRESOLVED',
        shutdown=dict(status='POST_PARITY_CLOSE_SYMPTOM_SUPPORTED' if shutdown_supported else 'UNRESOLVED',errors=errors,
            exact_es_parity_raise_timestamp='NOT_RECORDED',
            basis='all data completed before ES parity evaluation; verified source closes in finally after PARITY_FAIL; EOF has zero next-frame bytes' if shutdown_supported else 'insufficient shutdown ordering evidence',
            limit='not a directly timestamped parity-raise event; recv start may precede parity while waiting for the next frame'),
        wire_audit=dict(endpoint_frame_digests_agree=True,raw_payloads_stored=False,
            limit='reported end-to-end digests agree; no raw bytes remain to independently recompute tensor contents, branch order or ES combined QKV',
            es_combined_qkv_magnitudes='NOT_ESTIMABLE_FROM_ORIGINAL_TRACE'),
        hypotheses={h:dict(status='UNRESOLVED',reason=reason) for h,reason in (
            ('H1','requires native repeats using exact prepared weights'),('H2','requires same-hidden native vs independent GPU reconstruction'),
            ('H3','requires exact same-hidden, same-A/B CPU/GPU numerical comparison'),
            ('H4','complete frame digests alone do not prove correct tensor assembly/branch interpretation'),
            ('H5','original trace lacks assembled QKV output hashes/magnitudes'))})


def conclude(native_a=None,native_b=None,es_replay=None):
    """Return every evidenced distinction, never infer a cause from argmax/SHA alone."""
    findings=[]
    for native in (native_a,native_b):
        if native is None: continue
        for control in native.get('controls',[]):
            if (control['vs_original']['unequal_elements'] or control.get('original_projection_mismatches') or
                    control.get('logits_sha256')!=control.get('original_logits_sha256')):
                findings.append(dict(conclusion='REFERENCE_NONREPRODUCIBLE',evidence=control['label'],
                    strict_gate_failed=not control['vs_original']['passed'],scope='native reproduction differs under verified recorded conditions'))
        repeat=native.get('repeat_comparison')
        if repeat and repeat['unequal_elements']:
            findings.append(dict(conclusion='REFERENCE_NONREPRODUCIBLE',evidence='within-worker native repeat differs',strict_gate_failed=not repeat['passed']))
        observed=native.get('instrumented_forward')
        if observed and observed['vs_control']['unequal_elements']:
            findings.append(dict(conclusion='REFERENCE_NONREPRODUCIBLE',evidence='instrumented native differs from native control',
                strict_gate_failed=not observed['vs_control']['passed'],scope='observer-dependent reference; do not attribute original failure to CPU numerics'))
    if native_a:
        for p in native_a.get('projections',[]):
            c=p['comparisons'];where=dict(layer=p['layer'],role=p['role'])
            if c['native_vs_gpu_reconstruction']['unequal_elements']:
                findings.append(dict(conclusion='GPU_RECONSTRUCTION_MISMATCH',evidence=where,scope='identical native hidden and original weights'))
            if c['gpu_delta_vs_cpu_delta']['unequal_elements']:
                findings.append(dict(conclusion='CPU_GPU_DELTA_NUMERICS',evidence=where,
                    fp16_output_flip_count=p['fp16_output_flip_count'],scope='difference observed; full-failure causality requires controls/replay'))
            if not p['serialization_byte_identity'] or c['cpu_delta_vs_unpacked_cuda']['unequal_elements']:
                findings.append(dict(conclusion='WIRE_BYTES_OR_LAYOUT_MISMATCH',evidence=where,scope='local production packing/unpacking control'))
    if es_replay:
        for p in es_replay.get('projections',[]):
            if not p['wire_matches_local_cpu']:
                findings.append(dict(conclusion='WIRE_BYTES_OR_LAYOUT_MISMATCH',evidence=dict(layer=p['layer'],role=p['role']),
                    scope='actual TCP versus local CPU verification',
                    limit='consistency failure; CPU-process numerics versus serialization/layout not isolated by this contrast alone'))
            if not p['observed_output_matches_assembly']:
                findings.append(dict(conclusion='ES_ASSEMBLY_MISMATCH',evidence=dict(layer=p['layer'],role=p['role']),scope='actual module output versus independently computed assembly'))
    contrasts=[];path_bounds=[]
    if native_a and es_replay:
        native_points={(p['layer'],p['role']):p for p in native_a.get('projections',[])}
        for p in es_replay.get('projections',[]):
            native=native_points.get((p['layer'],p['role']))
            if native and 'hidden' in native and 'hidden' in p:
                contrasts.append(dict(layer=p['layer'],role=p['role'],
                    same_hidden=native['hidden']['sha256']==p['hidden']['sha256'],
                    same_base=native['independent_gpu_base']['sha256']==p['base']['sha256'],
                    same_cpu_delta=native['cpu_delta']['sha256']==p['received_delta']['sha256'],
                    observed_es_matches_native_cpu_assembly=p['observed_output']['sha256']==native['predicted_cpu_assembled']['sha256'],
                    native_vs_cpu_assembly_metrics=native['comparisons']['native_vs_cpu_assembled'],
                    scope='matched-input/tensor binding between independent diagnostic workers; not original ES QKV capture'))
        grouped={}
        for p in es_replay.get('projections',[]):grouped.setdefault(p['layer'],[]).append(p)
        for layer,block in sorted(grouped.items()):
            following=grouped.get(layer+1)
            if following and len(block)==3 and all(p.get('output_matches_original_native') is True for p in block) and any(
                    p.get('hidden_matches_original_native') is False for p in following):
                path_bounds.append(dict(layer=layer,scope='after matching QKV outputs and before next-layer input',
                    interpretation='attention/rest-of-decoder boundary difference; no specific arithmetic cause or magnitude inferred from SHA'))
    return dict(conclusions=sorted({f['conclusion'] for f in findings}) or ['UNRESOLVED'],findings=findings,
        same_hidden_es_contrasts=contrasts,post_qkv_path_localizations=path_bounds,
        causal_resolution='UNRESOLVED',
        interpretation='Numeric differences, bitwise reproducibility and strict logit failure are separate observations. No forced single root or threshold change.')
