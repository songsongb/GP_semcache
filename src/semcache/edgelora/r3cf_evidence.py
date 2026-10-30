"""Read-only legacy validation and independent focused socket/schedule audit."""
import csv
from functools import lru_cache
import importlib.util
from pathlib import Path

from .communication import PREFIX,canonical,require,validate
from .r3a_artifacts import ROOT,check_parity,read,sha,unpack,validate_prepared
from .r3a_evidence import jsonl
from .r3b_analysis import flags as bflags,validate_row
from .r3c_analysis import profiles,contrast,schedule as legacy_schedule
from .r3c_calibration import validate_calibration
from .r3c_runtime import proxy_request_metrics
from .r3cf_schedule import audit_schedule,budget,flags,make_schedule
from .r3cf_analysis import bracket,describe,fixed_count,mixture_sensitivity


@lru_cache(maxsize=1)
def legacy_runner():
    spec=importlib.util.spec_from_file_location('r3cf_original100',ROOT/'scripts/100_run_r3c_network_one_shot.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def verify_sources(sources):
    require(sources,'empty source SHA manifest')
    for name,expected in sources.items():
        path=(ROOT/name).resolve()
        require(path.is_relative_to(ROOT) and sha(path)==expected,'source integrity mismatch: '+name)


def legacy_reanalysis(directory,*,allow_fixture=False,seed=42,resamples=10000):
    directory=Path(directory).resolve()
    if not directory.exists():return dict(execution_status='NOT_RUN_MISSING_INPUT',legacy_input=str(directory),
        reason='original legacy directory unavailable; no rows reconstructed or fetched',**flags())
    result_sha=sha(directory/'r3c_one_shot_results.json')
    original,plan=legacy_runner().load_run(directory)
    fixture=original.get('TEST_FIXTURE_ONLY') is True
    require(not fixture or allow_fixture,'TEST_FIXTURE_ONLY legacy input requires explicit opt-in')
    require(original['classification']=='R3C_APPLICATION_LAYER_NETWORK_DIAGNOSTIC' and
        original['actual_network_socket_used'] is True,'unsupported legacy cohort classification')
    require(plan['profiles']==profiles([0,10],[None,200]) and plan['budget']['model_request_bound']==228 and
        plan['schedule']==legacy_schedule(plan['profiles'],seed=plan['seed']),'original 228-request factorial contract required')
    rows=jsonl(directory/'r3c_requests.jsonl')
    require(len(rows)==228 and len({r['request_id'] for r in rows})==228,'228 unique audited requests required')
    if not fixture:require(original['gpu_inference_performed'] is True,'real GPU legacy provenance required')
    old_decision=read(directory/'r3c_break_even_decision.json')
    with (directory/'r3c_break_even.csv').open(newline='') as stream:old_csv=list(csv.DictReader(stream))
    require(len(old_csv)==20,'missing old cache-arm contrasts')
    points=[];candidates={};models={}
    for p in plan['profiles']:
        cid=p['id'];block=[r for r in rows if r['condition_id']==cid and r['phase']=='measured']
        controls=[r for r in block if r['arm']=='NO_REUSE'];require(len(controls)==8,'eight legacy controls per condition required')
        current=[]
        for task in (t for t in plan['schedule'] if t['condition']['id']==cid and t['hit_fraction'] is not None):
            arm=[r for r in block if r['arm']==task['arm']]
            hits=[r['request_wall_ms'] for r in arm if r['cache_status']=='HIT']
            misses=[r['request_wall_ms'] for r in arm if r['cache_status']=='MISS']
            require(len(arm)==8 and len(hits)==int(8*task['hit_fraction']) and len(hits)+len(misses)==8,'legacy realized HIT counts mismatch')
            prior=[r for r in old_csv if r['condition_id']==cid and r['arm']==task['arm']]
            require(len(prior)==1,'missing/duplicate historical bootstrap arm')
            unconditional=contrast([r['request_wall_ms'] for r in arm],[r['request_wall_ms'] for r in controls],plan['seed'])
            for key in ('delta_mean_ms','ci95_low_ms','ci95_high_ms','status'):
                require(prior[0][key]==str(unconditional[key]),'historical unconditional bootstrap copy differs from original algorithm')
            point=dict(condition_id=cid,arm=task['arm'],**fixed_count(hits,misses,[r['request_wall_ms'] for r in controls],seed,resamples),
                cache=describe(arm),no_reuse=describe(controls),raw_hit_ms=hits,raw_miss_ms=misses,
                original_unconditional_csv_row=prior[0],original_unconditional_recomputed=unconditional,
                interpretation='RETROSPECTIVE_EXPLORATORY_ONLY')
            current.append(point);points.append(point)
        candidates[cid]=bracket(current,retrospective=True)
        models[cid]=mixture_sensitivity([r for r in block if r['arm']=='CACHE_00_OF_8'],
            [r for r in block if r['arm']=='CACHE_08_OF_8'],controls,seed,resamples)
    require(sha(directory/'r3c_one_shot_results.json')==result_sha,'legacy result changed during read-only analysis')
    for name,expected in original['output_hashes'].items():require(sha(directory/name)==expected,'legacy artifact changed during analysis: '+name)
    return dict(execution_status='COMPLETE',interpretation='RETROSPECTIVE_EXPLORATORY_ONLY',legacy_input=str(directory),
        legacy_result_sha256=sha(directory/'r3c_one_shot_results.json'),legacy_ledger_sha256=sha(directory/'r3c_requests.jsonl'),
        legacy_original_output_hashes=original['output_hashes'],legacy_source_hashes=original['sources'],
        original_classification=original['classification'],original_decision=old_decision,
        original_decision_sha256=sha(directory/'r3c_break_even_decision.json'),original_csv_sha256=sha(directory/'r3c_break_even.csv'),
        independent_request_ids=228,independent_deployments=1,native_parity_preserved=original['native_peft_strict_parity_passed'],
        arms=points,candidate_intervals=candidates,mean_mixture_sensitivity=models,seed=seed,resamples=resamples,
        warning='one-HIT strata have zero estimated HIT variance; no retrospective definitive crossover claim',**flags(fixture))


def audit_physical(directory,plan,result):
    """Own schedule audit; original frame/tensor arithmetic and validators reused."""
    directory=Path(directory);es=directory/'es';manifest=read(directory/'r3cf_schedule_manifest.json')
    require(manifest==plan['schedule_manifest']==make_schedule(plan['seed']) and plan['budget']==budget(manifest),
        'immutable schedule/budget differs from predeclared manifest')
    require(plan['schedule']==manifest['schedule'] and plan['profiles']==manifest['profiles'],'runtime schedule copies differ')
    verify_sources(plan['sources'])
    rows=jsonl(es/'r3cf_requests.jsonl');accounting=audit_schedule(rows,manifest)
    raw=jsonl(es/'r3c_requests.jsonl');require(len(raw)==len(rows),'component/focused row count mismatch')
    for a,b in zip(raw,rows):
        expected=dict(a,**flags(plan['fixture']),analysis_role=b['analysis_role'],schedule_sha256=manifest['schedule_sha256'])
        require(expected==b,'focused wrapper altered original computation/timing/counter evidence')
    pm,native,am=validate_prepared(plan['prepared'])
    identity=dict(identity=pm['identity'],identity_sha256=pm['identity_sha256'],
        prepared_manifest_sha256=sha(Path(plan['prepared'])/'r3a_prepare_manifest.json'),adapter_matrix_sha256=am['matrix_sha256'],
        gpu_uuid=pm.get('gpu',{}).get('uuid','TEST_FIXTURE_ONLY'))
    require(identity==plan['reference_identity'] and pm.get('TEST_FIXTURE_ONLY',False)==plan['fixture'],'prepared/reference scope mismatch')
    require(result['execution_status']=='COMPLETE' and result['requests']==126 and
        len({result['pid'],result['ud_pid'],result['proxy_pid']})==3 and result['proxy_pid']==plan['proxy']['pid'],
        'worker status/count/process identity failure')
    require(all(result.get(k)==v for k,v in flags(plan['fixture']).items()),'worker qualification flags changed')
    if not plan['fixture']:require(result['gpu']['uuid']==identity['gpu_uuid']!='UNAVAILABLE','fresh GPU UUID mismatch')
    cal=read(directory/'r3cf_network_calibration.json')
    require(cal==plan['calibration'] and cal['execution_status']=='COMPLETE' and cal['proxy_pid']==result['proxy_pid'] and
        cal['conditions']==validate_calibration(cal['raw_probes'],manifest['profiles']) and
        cal['independent_source_relay_sink_sha_and_bytes_passed'] is True and all(c['status']=='VALID' for c in cal['conditions']),
        'calibration identity/realization gate failed')
    physical=read(es/'r3b_no_reuse_reference.json');physical_file=es/'r3b_no_reuse_logits.f32.bin'
    require(sha(physical_file)==physical['logits_sha256'] and
        set(physical['qkv_sha256'])=={f'{l}.{r}' for l in range(32) for r in 'qkv'},'physical reference corrupted/incomplete')
    parity=check_parity(unpack(physical_file.read_bytes(),'float32'),
        unpack((Path(plan['prepared'])/native['logits_file']).read_bytes(),'float32'),native['logits_shape'])
    source=jsonl(es/'es_wire.jsonl');sink=jsonl(es/'r3a_ud_wire_trace.jsonl')
    proxy=[f for f in jsonl(directory/'proxy/proxy_frames.jsonl') if f.get('header',{}).get('session_id')==result['session_id']]
    relay={(f['direction'],f['header']['sequence']):f for f in proxy}
    require(len(source)==len(sink)==len(proxy)==len(relay),'independent frame ledgers differ or repeat')
    for a,b in zip(source,sink):
        require(a['complete'] and b['complete'] and a['operation']!=b['operation'] and
            all(a[k]==b[k] for k in ('header','frame_sha256','wire_bytes')),'independent ES/UD frame SHA mismatch')
        h=a['header'];validate(h,h['payload_bytes']);direction='es_to_ud' if a['operation']=='send' else 'ud_to_es'
        f=relay[direction,h['sequence']]
        require(h['adapter_sha256']==am['matrix_sha256'] and h['session_id']==result['session_id'] and
            a['wire_bytes']==PREFIX.size+len(canonical(h))+h['payload_bytes'] and
            f['header']==h and f['complete'] and f['frame_sha256']==a['frame_sha256'] and
            f['wire_bytes_read']==f['wire_bytes_forwarded']==a['wire_bytes'],'proxy/socket frame identity/bytes mismatch')
    for op in ('send','recv'):
        require([r['header']['sequence'] for r in source if r['operation']==op]==list(range(sum(r['operation']==op for r in source))),
            'wire sequence discontinuity')
    ud=read(es/'r3a_ud_result.json')
    require(ud['execution_status']=='COMPLETE' and ud['pid']==result['ud_pid'] and ud['user_device_cuda_used'] is False and
        ud['full_model_loaded'] is False and ud['environment']['variables']['CUDA_VISIBLE_DEVICES']=='','UD isolation/status failed')
    require(read(es/'r3c_condition_gates.json')==result['condition_gates'] and
        set(result['condition_gates'])=={'C00','C03'} and all(g['status']=='PASS' for g in result['condition_gates'].values()),'condition gates failed')
    for i,row in enumerate(rows):
        validate_row(dict(row,**bflags(plan['fixture'])),32)
        require(row['reference_identity']==identity and row['es_pid']==result['pid'] and row['ud_pid']==result['ud_pid'] and
            row['proxy_pid']==result['proxy_pid'] and all(row.get(k)==v for k,v in flags(plan['fixture']).items()),'request identity/flags mismatch')
        require(row['logits_sha256']==physical['logits_sha256'] and row['qkv_sha256']==physical['qkv_sha256'] and
            row['native_parity']==parity,'cache equality or unchanged Native strict metrics failed')
        profile=next(p for p in manifest['profiles'] if p['id']==row['condition_id'])
        frames=[f for f in source if f['header']['request_id']==row['request_id']]
        forwarded=[f for f in proxy if f['header']['request_id']==row['request_id']]
        require(row['condition']==profile and row['proxy_frames']==forwarded and
            row['proxy']==proxy_request_metrics(forwarded,profile) and row['proxy']['valid'] and
            row['calibration_status']==row['impairment_status']=='VALID','per-request impairment evidence invalid')
        require(len(frames)==len(forwarded)==row['hidden_frames']+row['delta_frames'] and
            sum(f['wire_bytes'] for f in frames)==row['application_wire_bytes'],'request frame count/byte mismatch')
        for kind,field in (('HIDDEN','hidden_payload_bytes'),('DELTAS','delta_payload_bytes')):
            selected=[f for f in frames if f['header']['kind']==kind]
            require([f['header']['layer_id'] for f in selected]==([] if row['mode']=='WARM_HIT' else list(range(32))) and
                sum(f['header']['payload_bytes'] for f in selected)==row[field],'layer schedule/directional payload mismatch')
        layers=[l for l in ud['layers'] if l['request_id']==row['request_id']]
        require(len(layers)==row['hidden_frames'] and sum(l['metrics']['cpu_lora_ms'] for l in layers)==row['ud_cpu_lora_ms'],
            'HIT did UD work or per-request UD accounting mismatch')
        require(all(row['request_start_monotonic_ns']<=f['monotonic_start_ns']<=f['monotonic_end_ns']<=row['request_end_monotonic_ns'] for f in frames),
            'model data I/O outside synchronized wall')
        if row['phase']!='gate':require(row['request_start_monotonic_ns']>=result['condition_gates'][row['condition_id']]['completed_monotonic_ns'],'measurement preceded gate')
        if i:require(rows[i-1]['request_end_monotonic_ns']<=row['request_start_monotonic_ns'],'overlapping request clocks')
    require(ud['completed_requests']==sum(r['hidden_frames']>0 for r in rows)==101 and len(ud['layers'])==3232,'unaccounted/missing UD work')
    require([f['header']['kind'] for f in source[:2]]==['HELLO','READY'] and [f['header']['kind'] for f in source[-2:]]==['STOP','BYE'] and
        sum(f['header']['kind'] not in ('HIDDEN','DELTAS') for f in source)==4,'handshake/shutdown control ledger mismatch')
    for path in ('proxy/proxy_parent_lifecycle.json','calibration/echo_parent_lifecycle.json','es/ud_lifecycle.json'):
        life=read(directory/path);require(life['returncode']==0 and not life['forced_termination'] and life['all_children_exited'],'unclean '+path)
    life=read(es/'lifecycle.json');require(life['returncode']==0 and life['tracked_descendants']['all_stopped'],'unclean ES descendants')
    relay_life=read(directory/'proxy/proxy_lifecycle.json')
    require(relay_life['threads_stopped'] and not relay_life['errors'],'proxy transport/cleanup failed')
    return dict(status='PASS',rows=126,accounting=accounting,independent_es_ud_relay_sha_bytes_reconciled=True,
        native_peft_strict_parity_passed=parity['passed'],native_parity=parity,completed_ud_requests=101,full_hits=25,
        model_data_frames=6464,control_frames_outside_wall=4)


def load_focused(directory):
    directory=Path(directory).resolve();report=read(directory/'r3cf_results.json')
    require(report['execution_status']=='COMPLETE' and report.get('audit',{}).get('status')=='PASS' and not report.get('analysis_only'),
        'completed physical focused run required')
    require(type(report.get('TEST_FIXTURE_ONLY')) is bool and all(report.get(k)==v for k,v in flags(report['TEST_FIXTURE_ONLY']).items()),
        'focused result qualification flags changed')
    required={'r3cf_requests.jsonl','r3cf_schedule_manifest.json','r3cf_network_calibration.json','es/es_result.json',
        'es/es_plan.json','es/r3cf_requests.jsonl','es/r3c_requests.jsonl','es/es_wire.jsonl','es/r3a_ud_wire_trace.jsonl',
        'es/r3a_ud_result.json','es/r3b_no_reuse_reference.json','es/r3b_no_reuse_logits.f32.bin','es/r3c_condition_gates.json',
        'proxy/proxy_frames.jsonl','proxy/proxy_lifecycle.json','proxy/proxy_parent_lifecycle.json',
        'calibration/echo_parent_lifecycle.json','es/ud_lifecycle.json','es/lifecycle.json'}
    require(required.issubset(report['output_hashes']),'missing required focused artifact hashes')
    for name,expected in report['output_hashes'].items():
        p=(directory/name).resolve();require(p.is_relative_to(directory) and sha(p)==expected,'focused artifact SHA mismatch: '+name)
    verify_sources(report['sources']);plan=read(directory/'es/es_plan.json');result=read(directory/'es/es_result.json')
    require(report['sources']==plan['sources'] and (directory/'r3cf_requests.jsonl').read_bytes()==(directory/'es/r3cf_requests.jsonl').read_bytes(),
        'focused source/ledger export copies differ')
    require(audit_physical(directory,plan,result)==report['audit'],'recorded/recomputed audit differs')
    return report,plan
