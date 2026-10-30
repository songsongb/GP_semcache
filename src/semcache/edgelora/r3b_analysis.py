"""R3-B evidence gates and exact-count descriptive statistics (CPU only)."""
import math
import statistics
from .communication import require
from .r3a_artifacts import read

MODES=('NO_REUSE','COLD_MISS','WARM_HIT')


def flags(fixture=False):
    return dict(classification='R3B_PHYSICAL_RAW_QKV_DIAGNOSTIC',PRIMARY_PERFORMANCE_ELIGIBLE=False,
        correctness_qualified_reuse_speedup=False,r2d_break_even_validated=False,cachegen_enabled=False,
        compression_enabled=False,semantic_generalization_validated=False,actual_remote_network_measured=False,
        TEST_FIXTURE_ONLY=fixture)


def validate_row(row,layers):
    require(all(row.get(k)==v for k,v in flags(row.get('TEST_FIXTURE_ONLY',False)).items()),'R3-B qualification flags changed')
    require(row['mode'] in MODES and row['observed_qkv_outputs']==3*layers,'incomplete QKV forward')
    hit=row['mode']=='WARM_HIT'
    require(row['reused_qkv_outputs']==(3*layers if hit else 0) and row['cache_status']==('HIT' if hit else
            'MISS' if row['mode']=='COLD_MISS' else 'BYPASS'),'wrong physical HIT/MISS behavior')
    for field in ('hidden_frames','delta_frames'):
        require(row[field]==(0 if hit else layers),'wrong observed data frame count: '+field)
    if hit:
        require(row['ud_cpu_lora_ms']==0 and row['hidden_payload_bytes']==row['delta_payload_bytes']==
                row['application_wire_bytes']==0,'HIT performed UD work or data traffic')
    else:require(row['application_wire_bytes']>row['hidden_payload_bytes']+row['delta_payload_bytes']>0,'wire byte ledger invalid')
    require(row['cache_bitwise_passed'] is True,'cache/full-logit/QKV bitwise parity failed')
    require(all(math.isfinite(row[k]) and row[k]>=0 for k in ('request_wall_ms','cache_lookup_ms','cache_insert_ms')),
            'invalid measured request timing')
    if row['mode']=='COLD_MISS':require(row['cache_bytes_before']==0 and row['cache_admitted'] is True,'cold cache not empty/admitted')
    return True


def q90(values):
    ordered=sorted(values);position=.9*(len(ordered)-1);lo=int(position);hi=min(lo+1,len(ordered)-1)
    return ordered[lo]+(ordered[hi]-ordered[lo])*(position-lo)


def summarize(rows,measured,layers):
    result={}
    for mode in MODES:
        selected=[r for r in rows if r['stage']=='B3' and r['phase']=='measured' and r['mode']==mode]
        require(len(selected)==measured and len({r['request_id'] for r in selected})==measured,'missing/duplicate measured rows')
        for row in selected:validate_row(row,layers)
        values=[r['request_wall_ms'] for r in selected]
        result[mode]=dict(samples=len(values),median_request_wall_ms=statistics.median(values),p90_request_wall_ms=q90(values),
            min_request_wall_ms=min(values),max_request_wall_ms=max(values),
            hidden_frames=sum(r['hidden_frames'] for r in selected),delta_frames=sum(r['delta_frames'] for r in selected),
            application_wire_bytes=sum(r['application_wire_bytes'] for r in selected),
            native_strict_passes=sum(r['native_parity']['passed'] for r in selected))
    return dict(by_mode=result,p90_method='linear interpolation at 0.9*(n-1)',
        timing_scope='synchronized application request wall; nested host/CUDA spans are not additive',
        historical_comparison='NOT_COMPARABLE_TO_R2D_COMPUTE_ENDPOINT; no R3-C/R2-D break-even validation')


def audit_run(directory,pm,result,warmup,measured):
    """Independent row, stage, endpoint-wire and UD work reconciliation."""
    from pathlib import Path
    from .r3a_evidence import jsonl
    from .r3a_artifacts import check_parity,unpack,sha
    from .communication import canonical,PREFIX,validate
    directory=Path(directory);rows=jsonl(directory/'r3b_requests.jsonl');n=pm['identity']['layers']
    physical=read(directory/'r3b_no_reuse_reference.json')
    native_directory=Path(read(directory/'r3b_plan.json')['prepared'])
    native=read(native_directory/'r3a_reference.json')
    physical_raw=(directory/'r3b_no_reuse_logits.f32.bin').read_bytes()
    require(sha(directory/'r3b_no_reuse_logits.f32.bin')==physical['logits_sha256'] and
            len(physical_raw)==4*math.prod(native['logits_shape']),'physical reference logit SHA/size mismatch')
    require(set(physical['qkv_sha256'])=={f'{l}.{r}' for l in range(n) for r in 'qkv'},'incomplete physical QKV reference')
    expected_native_parity=check_parity(unpack(physical_raw,'float32'),
        unpack((native_directory/native['logits_file']).read_bytes(),'float32'),native['logits_shape'])
    require(len(rows)==3+3*(warmup+measured) and len({r['request_id'] for r in rows})==len(rows),'incomplete/duplicate request ledger')
    require([(r['stage'],r['mode']) for r in rows[:3]]==[('B1','NO_REUSE'),('B1','COLD_MISS'),('B2','WARM_HIT')],
            'B1/B2 order mismatch')
    require(all(result['stages'][s]['status']=='PASS' for s in ('B1','B2','B3')),'stage gate failed')
    require(read(directory/'r3b_stage_gates.json')==result['stages'],'stage gate copies differ')
    for name in ('B1','B2','B3'):require(read(directory/f'{name}_gate.json')==result['stages'][name],'per-stage gate copy mismatch')
    require(len(result['stages']['B2']['invalid_guards'])==9 and all(g['rejected'] for g in result['stages']['B2']['invalid_guards']),
            'invalid-hit guards missing')
    expected=[(phase,repeat,mode) for phase,count in (('warmup',warmup),('measured',measured))
              for repeat in range(count) for mode in MODES]
    require([(r['phase'],r['repeat'],r['mode']) for r in rows[3:]]==expected,'B3 sample schedule incomplete')
    for before,after in zip(rows,rows[1:]):
        require(before['request_end_monotonic_ns']<=after['request_start_monotonic_ns'],'request clocks overlap/reordered')
    for row in rows:
        validate_row(row,n)
        require(row['logits_sha256']==physical['logits_sha256'] and row['qkv_sha256']==physical['qkv_sha256'],
                'full-logit or all-layer physical QKV hash mismatch')
        require(row['native_parity']==expected_native_parity,'native strict gate metrics/copy mismatch')
        if row['stage']=='B2':require(row['request_start_monotonic_ns']>=result['stages']['B1']['completed_monotonic_ns'],'B2 before B1 gate')
        if row['stage']=='B3':require(row['request_start_monotonic_ns']>=result['stages']['B2']['completed_monotonic_ns'],'B3 before B2 gate')
    es=jsonl(directory/'r3b_wire_trace.jsonl');ud=jsonl(directory/'r3a_ud_wire_trace.jsonl')
    for stage in ('B1','B2','B3'):
        require(jsonl(directory/f'{stage}_wire_trace.jsonl')==[r for r in es if r['header']['request_id'].startswith(stage+'_')],
                'per-stage wire trace copy mismatch')
    require(len(es)==len(ud),'wire endpoint counts differ')
    for left,right in zip(es,ud):
        require(left['complete'] and right['complete'] and left['operation']!=right['operation'] and
                all(left[k]==right[k] for k in ('header','frame_sha256','wire_bytes')),'wire endpoint digest/copy mismatch')
        h=left['header'];validate(h,h['payload_bytes'])
        require(left['wire_bytes']==PREFIX.size+len(canonical(h))+h['payload_bytes'],'wire framing byte ledger incorrect')
        require(h['adapter_sha256']==pm['identity']['adapter_matrix_sha256'],'wire adapter SHA mismatch')
    for op in ('send','recv'):
        require([r['header']['sequence'] for r in es if r['operation']==op]==list(range(sum(r['operation']==op for r in es))),
                'wire sequence discontinuity')
    require([(r['header']['kind']) for r in es[:2]]==['HELLO','READY'] and [r['header']['kind'] for r in es[-2:]]==['STOP','BYE'],
            'missing handshake/shutdown')
    ud_result=read(directory/'r3a_ud_result.json')
    require(ud_result['execution_status']=='COMPLETE' and ud_result['user_device_cuda_used'] is False and
            ud_result['full_model_loaded'] is False and ud_result['pid']==result['ud_pid']!=result['pid'],'process isolation failed')
    require(ud_result['environment']['variables']['CUDA_VISIBLE_DEVICES']=='','UD CUDA visibility not hidden')
    for row in rows:
        records=[r for r in es if r['header']['request_id']==row['request_id']]
        layers=[l for l in ud_result['layers'] if l['request_id']==row['request_id']]
        require(len(records)==row['hidden_frames']+row['delta_frames'] and len(layers)==row['hidden_frames'],'per-request UD/data mismatch')
        require(sum(l['metrics']['cpu_lora_ms'] for l in layers)==row['ud_cpu_lora_ms'],'UD computation accounting mismatch')
        for record in records:
            require(row['request_start_monotonic_ns']<=record['monotonic_start_ns']<=record['monotonic_end_ns']<=row['request_end_monotonic_ns'],
                    'frame outside synchronized request wall')
        require(sum(r['wire_bytes'] for r in records)==row['application_wire_bytes'],'request wire sum mismatch')
        for kind,field in (('HIDDEN','hidden_payload_bytes'),('DELTAS','delta_payload_bytes')):
            subset=[r for r in records if r['header']['kind']==kind]
            require(sum(r['header']['payload_bytes'] for r in subset)==row[field],'directional payload mismatch')
            require([r['header']['layer_id'] for r in subset]==([] if row['mode']=='WARM_HIT' else list(range(n))),
                    'physical layer message schedule mismatch')
    require(sum(r['hidden_frames'] for r in rows)==len(ud_result['layers']),'unmatched UD work outside requests')
    expected_summary=summarize(rows,measured,n)
    require(expected_summary==result['summary'],'worker/recomputed summary mismatch')
    return dict(status='PASS',requests=len(rows),endpoint_frame_hashes_agree=True,
        output_cache_bitwise_all_pass=True,native_strict_passes=sum(r['native_parity']['passed'] for r in rows),
        native_strict_failures=sum(not r['native_parity']['passed'] for r in rows),
        actual_network_socket_used=True,actual_remote_network_measured=False,summary=expected_summary)
