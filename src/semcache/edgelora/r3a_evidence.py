"""Read-only R3-A trial/aggregate validation; no imputation or model imports."""
import csv
from pathlib import Path
import statistics

from .communication import PREFIX,canonical,payload_size,require,validate
from .r3a_artifacts import CLASSIFICATION, THRESHOLDS, read, sha, validate_prepared


def jsonl(path):
    from .communication import json_loads
    return [json_loads(line) for line in Path(path).read_bytes().splitlines() if line.strip()]


def validate_run(directory, prepared, *, phase=None, _candidate=False):
    directory=Path(directory);pm,ref,am=validate_prepared(prepared)
    m=read(directory/'r3a_manifest.json');decision=read(directory/'r3a_decision.json')
    c=m['configuration'];identity=pm['identity']
    require(c['phase'] in ('pilot','latency') and (phase is None or phase==c['phase']), 'wrong trial phase')
    require(type(c['warmup_runs']) is int and type(c['measured_runs']) is int and
            (1 if c['phase']=='pilot' else 3)<=c['warmup_runs']<=1000 and
            (3 if c['phase']=='pilot' else 10)<=c['measured_runs']<=1000, 'insufficient/invalid trial counts')
    require(m['execution_status']==('VALIDATING' if _candidate else 'PILOT_VALIDATED' if c['phase']=='pilot' else 'COMPLETE'), 'missing/failed pilot or latency evidence')
    require(m['prepared_manifest_sha256']==sha(Path(prepared)/'r3a_prepare_manifest.json') and
            m['identity_sha256']==pm['identity_sha256'], 'stable prepared identity mismatch')
    require(m['code']['files']==pm['code']['files'], 'trial code identity mismatch')
    required={'r3a_environment.json','r3a_wire_trace.jsonl','r3a_ud_wire_trace.jsonl','r3a_iterations.jsonl',
              'r3a_latency_raw.csv','r3a_parity.json','r3a_boundary_audit.json','r3a_decision.json',
              'r3a_es_result.json','r3a_ud_result.json','r3a_lifecycle.json'}
    require(required.issubset(m['artifact_hashes']), 'missing required trial artifact hash')
    for name,expected in m['artifact_hashes'].items():
        require(Path(name).name==name and sha(directory/name)==expected, 'trial artifact SHA mismatch: '+name)
    for obj in (m,decision):
        require(obj['classification']==CLASSIFICATION and obj['actual_network_socket_used'] is True and
                obj['actual_remote_network_measured'] is False and obj['cache_reuse_enabled'] is False and
                obj['user_device_cuda_used'] is False and obj['parity_passed'] is (not _candidate) and
                obj['safe_reuse_claimed'] is False and obj['correctness_qualified_reuse_speedup'] is False,
                'trial classification/parity/isolation gate mismatch')
    require(m['cleanup']['all_children_exited'] is True, 'process cleanup unverified')
    fixture=pm.get('TEST_FIXTURE_ONLY') is True
    require(m['TEST_FIXTURE_ONLY']==fixture and m['gpu_inference_performed'] is (not fixture), 'fixture/GPU provenance mismatch')
    environment=read(directory/'r3a_environment.json')
    es,ud=environment['es'],environment['ud']
    require(type(es['pid']) is int and type(ud['pid']) is int and es['pid']!=ud['pid'] and
            es['execution_status']==ud['execution_status']=='COMPLETE' and es['base_only'] is True and
            es['adapter_parameters_on_es']==0 and ud['full_model_loaded'] is False and
            ud['user_device_cuda_used'] is False, 'independent process separation not established')
    require(ud['identity_sha256']==es['identity_sha256']==pm['identity_sha256'] and
            ud['adapter_matrix_sha256']==am['matrix_sha256'], 'ES/UD tensor identity mismatch')
    require(es['environment']['hostname']==ud['environment']['hostname'] and
            ud['environment']['variables']['CUDA_VISIBLE_DEVICES']=='' and
            all(r['environment']['variables'][k]=='1' for r in (es,ud) for k in ('HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE')),
            'one-host/offline/UD CUDA visibility gate failed')
    require(es['gpu_inference_performed'] is (not fixture) and
            ud['cpu']['effective_affinity'] and len(ud['cpu']['effective_affinity'])<=4,
            'GPU execution or effective UD affinity unverified')
    if not fixture:
        require(es['gpu']['free_bytes']>=512*1024**2 and es['gpu']['peak_allocated_bytes']<es['gpu']['total_bytes'] and
                es['gpu']['uuid']!='UNAVAILABLE', 'GPU memory/identity gate failed')
    observations=jsonl(directory/'r3a_iterations.jsonl')
    require(es['rows']==[o['row'] for o in observations] and es['parities']==[o['parity'] for o in observations] and
            es['per_layer_trace']==[dict(request_id=o['row']['request_id'],layers=o['layers']) for o in observations],
            'ES result/raw copies disagree')
    expected=[(p,i) for p,count in (('warmup',c['warmup_runs']),('measured',c['measured_runs'])) for i in range(count)]
    require([(o['row']['phase'],o['row']['repeat']) for o in observations]==expected, 'missing/duplicate/warmup measured observations')
    ids=set();frames=jsonl(directory/'r3a_wire_trace.jsonl');ud_frames=jsonl(directory/'r3a_ud_wire_trace.jsonl')
    require(len(frames)==len(ud_frames)==2*identity['layers']*len(observations)+4,'actual frame count mismatch')
    controls=[f for f in frames if f['header']['layer_id']==-1]
    require([(f['operation'],f['header']['kind'],f['header']['request_id']) for f in controls]==
            [('send','HELLO','handshake'),('recv','READY','handshake'),('send','STOP','shutdown'),('recv','BYE','shutdown')],
            'control message schedule mismatch')
    require(len({f['header']['session_id'] for f in frames})==1,'mixed socket sessions')
    for operation in ('send','recv'):
        sequences=[f['header']['sequence'] for f in frames if f['operation']==operation]
        require(sequences==list(range(len(sequences))),'out-of-order wire sequence evidence')
    for ef,uf in zip(frames,ud_frames):
        require(ef['complete'] is True and uf['complete'] is True and ef['error'] is uf['error'] is None and
                ef['operation']!=uf['operation'] and ef['pid']==es['pid'] and uf['pid']==ud['pid'] and
                all(ef[k]==uf[k] for k in ('header','wire_bytes','frame_sha256')), 'ES/UD wire evidence mismatch')
        require(ef['monotonic_end_ns']>=ef['monotonic_start_ns'] and
                uf['monotonic_end_ns']>=uf['monotonic_start_ns'],'invalid per-process clock interval')
        require(ef['header']['adapter_sha256']==am['matrix_sha256'], 'wrong adapter frame identity')
        validate(ef['header'],ef['header']['payload_bytes'])
        require(ef['wire_bytes']==PREFIX.size+len(canonical(ef['header']))+ef['header']['payload_bytes'], 'framed wire byte count mismatch')
    for o in observations:
        row,parity,layers=o['row'],o['parity'],o['layers'];rid=row['request_id']
        require(row['es_pid']==es['pid'] and row['ud_pid']==ud['pid'], 'row PID identity mismatch')
        for key,value in row.items():
            if key.endswith('_ms') and value is not None:
                require(type(value) in (int,float) and value>=0,'invalid timing component: '+key)
        require(rid not in ids and row['identity_sha256']==pm['identity_sha256'], 'duplicate/mismatched request identity')
        ids.add(rid)
        require(parity['passed'] is True and parity['status']=='PASS' and parity['thresholds']==THRESHOLDS and
                parity['tolerance_violations']==0 and parity['max_abs']<=THRESHOLDS['max_abs'] and
                parity['relative_l2']<=THRESHOLDS['relative_l2'] and parity['last_token_kl']<=THRESHOLDS['last_token_kl'] and
                parity['last_token_argmax_agreement'] is True and row['parity_passed'] is True, 'strict parity gate failed')
        require([l['layer_id'] for l in layers]==list(range(identity['layers'])) and
                row['layers_observed']==identity['layers'] and row['qkv_outputs_observed']==3*identity['layers'], 'missing real forward layers/QKV outputs')
        block=[f for f in frames if f['header']['request_id']==rid]
        require(len(block)==2*identity['layers'],'request data frame count mismatch')
        for layer,l in enumerate(layers):
            sent,received=block[2*layer:2*layer+2]
            require(l['request_id']==rid and l['sent_frame']==sent and l['received_frame']==received and
                    l['qkv_hidden_input_equal']==dict(q=True,k=True,v=True), 'QKV input identity or layer trace mismatch')
            for f,kind,dtype,shape in ((sent,'HIDDEN',identity['hidden_dtype'],[1,identity['length'],identity['hidden_size']]),
                 (received,'DELTAS',identity['delta_dtype'],[3,1,identity['length'],identity['hidden_size']])):
                h=f['header']
                require(h['kind']==kind and h['layer_id']==layer and h['dtype']==dtype and h['shape']==shape and
                        h['payload_bytes']==payload_size(dtype,shape), 'layer wire tensor mismatch')
                require(row['es_start_monotonic_ns']<=f['monotonic_start_ns']<=f['monotonic_end_ns']<=row['es_end_monotonic_ns'], 'frame outside request clock')
            require(l['ud']==received['header']['metadata'] and l['ud']['user_device_cuda_used'] is False and
                    l['ud']['hidden_payload_sha256']==sent['header']['payload_sha256'], 'transported hidden/delta provenance mismatch')
            require(l['ud']['pid']==ud['pid'] and l['ud']['monotonic_end_ns']>=l['ud']['monotonic_start_ns'], 'UD clock/PID mismatch')
            for key in ('cpu_lora_ms','input_deserialization_ms','output_serialization_ms'):
                require(type(l['ud'][key]) in (int,float) and l['ud'][key]>=0,'invalid UD timing')
            if not fixture:
                precision=l['ud']['actual_precision']
                require(set(precision)==set('qkv') and all(
                    all(p[k]=='torch.float32' for k in ('input_dtype','A_weight_dtype','A_output_dtype','B_weight_dtype','B_output_dtype'))
                    for p in precision.values()), 'unverified actual CPU A/B precision')
        for prefix,operation in (('es_to_ud','send'),('ud_to_es','recv')):
            selected=[f for f in block if f['operation']==operation]
            require(row[prefix+'_frames']==len(selected)==identity['layers'] and
                    row[prefix+'_payload_bytes']==sum(f['header']['payload_bytes'] for f in selected) and
                    row[prefix+'_wire_bytes']==sum(f['wire_bytes'] for f in selected),'directional byte/frame accounting mismatch')
        require(abs(row['es_synchronized_wall_ms']-(row['es_end_monotonic_ns']-row['es_start_monotonic_ns'])/1e6)<1e-9,
                'request wall/time unit mismatch')
        require(abs(row['ud_cpu_lora_ms']-sum(l['ud']['cpu_lora_ms'] for l in layers))<1e-8,'UD processing aggregate mismatch')
    expected_ud=[dict(request_id=o['row']['request_id'],layer_id=l['layer_id'],metrics=l['ud'],
        hidden_frame_sha256=l['sent_frame']['frame_sha256'],delta_frame_sha256=l['received_frame']['frame_sha256']) for o in observations for l in o['layers']]
    require(ud['layers']==expected_ud and ud['completed_requests']==len(observations),'UD result/raw copies disagree')
    parity_copy=read(directory/'r3a_parity.json')
    require(parity_copy['observations']==[o['parity'] for o in observations], 'parity copies disagree')
    with (directory/'r3a_latency_raw.csv').open() as stream: csv_rows=list(csv.DictReader(stream))
    require(len(csv_rows)==len(observations),'CSV observation count mismatch')
    for row,o in zip(csv_rows,observations):
        require(row=={k:'' if v is None else str(v) for k,v in o['row'].items()}, 'CSV/raw row copies disagree')
    boundary=read(directory/'r3a_boundary_audit.json')
    require(boundary['h0_transported'] is False and boundary['hL_transported'] is False and
            boundary['reference_inside_distributed_request_wall'] is False and boundary['nested_timers_may_be_summed'] is False,
            'timing/ownership boundary mismatch')
    return dict(manifest=m,observations=observations,environment=environment,files={p.name:sha(p) for p in directory.iterdir() if p.is_file()})


def aggregate(prepared,pilot,inputs):
    from .r3a_artifacts import flags
    pm,_,_=validate_prepared(prepared)
    pilot_data=validate_run(pilot,prepared,phase='pilot')
    latency=validate_run(inputs,prepared,phase='latency')
    require(latency['manifest']['pilot_manifest_sha256']==sha(Path(pilot)/'r3a_manifest.json'), 'aggregate pilot binding mismatch')
    require(latency['environment']['es']['device']==pilot_data['environment']['es']['device'], 'pilot/latency device mismatch')
    if not pm.get('TEST_FIXTURE_ONLY'):
        require(latency['environment']['es']['gpu']['uuid']==pilot_data['environment']['es']['gpu']['uuid'], 'pilot/latency accelerator mismatch')
    measured=[o['row'] for o in latency['observations'] if o['row']['phase']=='measured']
    values=[r['es_synchronized_wall_ms'] for r in measured]
    result=dict(execution_status='COMPLETE',measured_count=len(values),warmup_count=len(latency['observations'])-len(values),
        es_synchronized_wall_ms=dict(mean=statistics.mean(values),median=statistics.median(values),min=min(values),max=max(values)),
        directional_bytes_per_measured_request=[{k:r[k] for k in r if k.endswith('_bytes') or k.endswith('_frames')} for r in measured],
        timing_interpretation='synchronized warm distributed prefill only; nested host/CUDA and UD spans are not summed',
        identity_sha256=pm['identity_sha256'],parity_passed=True,**flags(socket_used=True,fixture=pm.get('TEST_FIXTURE_ONLY') is True))
    audit=dict(execution_status='COMPLETE',prepared_manifest_sha256=sha(Path(prepared)/'r3a_prepare_manifest.json'),
        prepared_directory=str(Path(prepared).resolve()),pilot_directory=str(Path(pilot).resolve()),
        latency_directory=str(Path(inputs).resolve()),
        pilot=pilot_data['files'],latency=latency['files'],raw_rows_verified=True,wire_both_endpoints_verified=True,
        no_imputation=True,reference_excluded_from_request_clock=True,sha_scope='artifact consistency, not independent hardware attestation')
    return result,audit
