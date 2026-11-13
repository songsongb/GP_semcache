"""Native OPT ES/isolated CPU UD orchestration; no implicit GPU execution."""
from collections import Counter
from pathlib import Path
import importlib.metadata
import os
import signal
import shutil
import subprocess
import sys
import time
import uuid

from .communication import require, digest, header, connect
from .r3a_artifacts import ROOT
from .r5_d5_lightperf_io import (ENGINE, MODE, SCHEMA, SummaryWriter, fresh, read_json,
                               source_identity, verify, hidden_cuda)
from .r5_d5_lightperf_workload import CONDITIONS, knobs
from .r5_d5_lightperf_emulation import LightChannel, UDServiceEmulator, serve_light_sparse
from .r5_d5_lightperf_receipts import WireSummary, PolicyCounter, compact_request, wire_equal
from .r5_d5_lightperf_timing import generate_native, SessionLifecycle

SCRIPT = ROOT / 'scripts/159_r5_d5_lightperf.py'


def native_versions():
    return {n: importlib.metadata.version(n) for n in ('torch', 'transformers', 'numpy')}


def bound_receipts(experiment, cpu_audit):
    """UD-safe receipt validation: never import transformers or load the model."""
    pi, ai = verify(experiment), verify(cpu_audit)
    p = read_json(Path(experiment) / 'plan.json')
    a = read_json(Path(cpu_audit) / 'cpu_validation.json')
    body = dict(p)
    pid = body.pop('plan_id')
    require(p['engine'] == ENGINE and digest(body) == pid and p['mode'] == MODE and p['schema'] == SCHEMA,
            'native Light plan identity differs')
    s = p['source']
    require(s['approval_scope'] == 'CLEAN_REVIEWED_SOURCE' and s == source_identity(
        reviewed_commit=s['commit'], reviewed_fingerprint=s['content_fingerprint']), 'clean source approval differs')
    from .r5_d5_lightperf import TEST_MODULES
    require(a['engine'] == ENGINE and a['status'] == 'PASS' and a['scope'] == 'CLEAN_NATIVE_CPU_APPROVAL' and
            a['failed'] == a['errors'] == a['skipped'] == 0 and a['tests'] > 0 and
            a['source'] == s and a['plan_inventory'] == pi and a['modules'] == list(TEST_MODULES) and
            a['native_parity'] == 'TINY_OPT_STRICT_LIGHT_CPU_PASS' and a['versions'] == native_versions(),
            'missing/stale/non-native CPU approval')
    require(p['resources']['status'] == 'STATIC_BOUNDS_PASS_RUNTIME_CHECKS_PENDING', 'BLOCKED_INFEASIBLE')
    return p, pi, ai


def allocation_guard(environ=None):
    e = os.environ if environ is None else environ
    require(e.get('SLURM_JOB_ID') and not e.get('SLURM_ARRAY_TASK_ID') and
            e.get('R5_ENABLE_GPU') == e.get('R5_I5_ENABLE_GPU') == e.get('D5L_ENABLE_GPU') == '1',
            'manual non-array allocation and three explicit GPU opt-ins required')
    from .r5_subsequence_natural_runtime import validate_node
    validate_node(e)
    require(int(e.get('SLURM_CPUS_PER_TASK', '0')) >= 4 and int(e.get('SLURM_MEM_PER_NODE', '0')) >= 65536 and
            len(os.sched_getaffinity(0)) >= 4, '4 allocated CPUs/64GiB required')
    require(e.get('HF_HUB_OFFLINE') == e.get('TRANSFORMERS_OFFLINE') == '1' and
            e.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8' and e.get('PYTHONHASHSEED') == '42',
            'offline/deterministic environment differs')
    # User provides remaining allocation time explicitly; no squeue/scontrol query.
    seconds = int(e.get('D5L_ALLOCATION_REMAINING_SECONDS', '0'))
    require(0 < seconds <= 16200, 'explicit remaining allocation seconds required')
    return seconds


def host_available():
    values = {line.split(':')[0]: int(line.split()[1]) * 1024
              for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')}
    result = values['MemAvailable']
    cg = Path('/sys/fs/cgroup')
    if (cg / 'memory.max').is_file() and (cg / 'memory.current').is_file():
        limit = (cg / 'memory.max').read_text().strip()
        if limit != 'max':
            result = min(result, int(limit) - int((cg / 'memory.current').read_text()))
    return result


def wait_file(path, process, deadline):
    while not Path(path).exists():
        require(process.poll() is None, 'UD exited before metadata: ' + str(path.parent))
        if time.monotonic() >= deadline:
            raise TimeoutError('UD metadata deadline exceeded: ' + str(path))
        time.sleep(min(.02, max(.001, deadline - time.monotonic())))
    return read_json(path)


def reap(process, *, timeout=30):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)


def seal_after(writer, primary, cleanup_errors):
    """Keep the original failure even when a bounded receipt or seal fails."""
    try:
        if primary is not None or cleanup_errors:
            writer.write('failure.json', dict(status='FAILED', reason=str(primary)[:4096], cleanup_errors=cleanup_errors))
        writer.seal()
    except BaseException as exc:
        if primary is None:
            raise
        if hasattr(primary, 'add_note'):
            primary.add_note('Light finalization also failed: ' + repr(exc))


def ud_worker(config, output_dir):
    hidden_cuda()
    c = read_json(config)
    p, pi, ai = bound_receipts(c['experiment'], c['cpu_audit'])
    require(c['source'] == p['source'] and c['plan_inventory'] == pi and c['cpu_inventory'] == ai and
            c['performance_knobs'] == p['performance_knobs'] and str(Path(output_dir).resolve()) == c['output_dir'],
            'UD config/source/plan/knobs differ')
    from .r5_multi_artifacts import validate, load as load_matrices
    from .r5_i5_contracts import request
    from .user_device import configure_cpu
    from .communication import listener
    m, pm, specs, registry = validate(p['multi'])
    identity = c['identity']
    spec = specs[identity['adapter_id']]
    tasks = [t for t in p['stream']['tasks'] if t['adapter_id'] == identity['adapter_id']]
    require(c['requests'] == [request(t, t['token_ids'], t['policy'], identity) for t in tasks] and
            identity['adapter_matrix_fingerprint'] == spec['matrix_sha256'] and
            identity['base_fingerprint'] == pm['identity']['base_weight_sha256'] and
            identity['model_revision'] == pm['identity']['base_revision'] and identity['tokenizer'] == m['tokenizer'],
            'UD model/adapter/original requests differ')
    writer = SummaryWriter(fresh(output_dir, [config, c['experiment'], c['cpu_audit'], p['multi'], p['base_prepared']]))
    try:
        cpu = configure_cpu(True)
        matrices = load_matrices(p['multi'], spec, torch_backend=True)
        import torch
        require(not torch.cuda.is_initialized() and not any(n.startswith('transformers') for n in sys.modules) and
                all(t.device.type == 'cpu' for t in matrices.values()), 'UD isolation violated')
        serve_ud_session(c, p, pi, ai, writer, cpu, matrices, registry)
    except BaseException as exc:
        if not writer.sealed:
            seal_after(writer, exc, [])
        raise


def serve_ud_session(c, p, pi, ai, writer, cpu, matrices, registry):
    """Shared native CPU UD session used by production and Tiny OPT tests."""
    from .communication import listener
    identity = c['identity']
    spec = dict(matrix_sha256=identity['adapter_matrix_fingerprint'])
    connection = sock = channel = None
    primary = None
    try:
        import torch
        hidden_cuda()
        require(not torch.cuda.is_initialized() and not any(n.startswith('transformers') for n in sys.modules) and
                all(t.device.type == 'cpu' for t in matrices.values()), 'native UD session isolation violated')
        deadline = c['deadline_monotonic']
        require(0 < deadline - time.monotonic() <= p['resources']['worker_deadline_seconds'], 'UD deadline invalid')
        ids = [r['request_id'] for r in c['requests']]
        wire, service = WireSummary(ids), UDServiceEmulator(p['performance_knobs']['ud_throughput_factor'])
        compute = {rid: dict(cpu_projection_rows=0, measured_compute_ms=0., measured_service_ms=0., pacing_ms=0.,
                             operations=0, serialization_ms=0., deserialization_ms=0., hash_ms=0.) for rid in ids}
        completed = []

        def trace(record):
            rid = record['request_id']
            require(rid in compute, 'unbound UD compute')
            if record.get('kind') == 'COMPLETE_REQUEST':
                require(rid == ids[len(completed)], 'UD completion order differs')
                v = compute[rid]
                v['achieved_service_factor'] = None if v['measured_service_ms'] <= 0 else v['measured_compute_ms'] / v['measured_service_ms']
                v['requested_factor'] = service.factor
                v['scope'] = 'EMULATED_UD_SERVICE_TIME'
                writer.write(f'request_{len(completed):03d}.json', dict(request_id=rid, wire=wire.request(rid),
                             cpu_projection_rows=v['cpu_projection_rows'], compute=v))
                completed.append(rid)
                return
            compute[rid]['cpu_projection_rows'] += record['cpu_projection_rows']
            for k, value in record.get('service_delta', {}).items():
                compute[rid][k] += value
            for k in ('serialization_ms', 'deserialization_ms', 'hash_ms'):
                compute[rid][k] += record.get(k, 0.)

        sock = listener()
        sock.settimeout(min(120., max(.001, deadline - time.monotonic())))
        writer.write('ready.json', dict(port=sock.getsockname()[1], cpu=cpu, pid=os.getpid(), source=p['source'],
                                       user_device_cuda_used=False, full_model_loaded=False))
        connection, peer = sock.accept()
        require(peer[0] == '127.0.0.1', 'loopback required')
        sock.close()
        sock = None
        channel = LightChannel(connection, c['session'], spec['matrix_sha256'], bandwidth_mbps=p['performance_knobs']['bandwidth_mbps'],
                               timeout=min(1800., deadline - time.monotonic()), trace=wire)
        channel.condition_deadline = deadline
        serve_light_sparse(channel, identity, registry, matrices, service=service, deadline=deadline,
                           expected_requests=c['requests'], trace=trace)
        require(completed == ids and not wire.errors and not torch.cuda.is_initialized(), 'incomplete/failed UD session')
        writer.write('ud.json', dict(engine=ENGINE, status='COMPLETE', source=p['source'], plan_inventory=pi, cpu_inventory=ai,
                                    identity=identity, cpu=cpu, completed_request_ids=completed, wire=wire.rows,
                                    service=service.report(), pacing=channel.pacer.report(),
                                    user_device_cuda_used=False, full_model_loaded=False, final_bye_sent=True))
    except BaseException as exc:
        primary = exc
        raise
    finally:
        cleanup_errors = []
        for resource in (connection, sock):
            if resource is not None:
                try:
                    resource.close()
                except BaseException as exc:
                    cleanup_errors.append(str(exc)[:512])
        seal_after(writer, primary, cleanup_errors)
        if primary is None and cleanup_errors:
            raise RuntimeError('UD cleanup failed: ' + str(cleanup_errors))


def stop_after_ack(session, req, control, lifecycle):
    require(session['completed'] < len(session['ids']) and control['request'] == req and
            req['request_id'] == session['ids'][session['completed']], 'ACK request differs')
    final = lifecycle.ack(req['request_id'])
    session['completed'] += 1
    if not final:
        return
    require(not session['stopped'] and session['completed'] == len(session['ids']), 'duplicate/premature STOP')
    lifecycle.stop_after_ack(req['adapter_id'])
    ch = session['channel']
    ch.send(header('STOP', 'shutdown', -1))
    h, b, _ = ch.recv(kind='BYE', request_id='shutdown', layer_id=-1)
    require(not b and h['dtype'] == 'none' and h['shape'] == [] and h['metadata'] == {}, 'invalid BYE')
    lifecycle.bye(req['adapter_id'])
    session['stopped'] = True
    ch.close()


def execute_stream(tasks, sessions, model, adapter, tokenizer, condition, writer, deadline, *,
                   score_fn, request_fn=None, generate_fn=generate_native):
    """Shared serial engine, used by the real ES worker and native CPU tests.

    Injection is only for CPU unit tests; production worker supplies real inputs
    and generate_native. Never a fixture option in the production CLI.
    """
    from .r5_i5_contracts import request
    from .r5_subsequence_physical import EXACT, APPROX
    request_fn = request if request_fn is None else request_fn
    lifecycle, rows = SessionLifecycle(tasks), []
    for ordinal, task in enumerate(tasks):
        arrival = time.monotonic_ns()
        if time.monotonic() >= deadline:
            raise TimeoutError('condition request deadline exceeded')
        s = sessions[task['adapter_id']]
        req = request_fn(task, task['token_ids'], task['policy'], s['identity'])
        lifecycle.begin(task)
        before = dict(s['policy'].counts)
        owner = s['owner']
        result = generate_fn(model, adapter, tokenizer, req, s['channel'], owner, s['index'], condition=condition,
                             mode=APPROX if condition == CONDITIONS[2] else EXACT,
                             allow_approximate=condition == CONDITIONS[2], deadline=deadline,
                             on_tcp_complete=lambda control, s=s, req=req: stop_after_ack(s, req, control, lifecycle))
        ud = wait_file(s['ud_out'] / f'request_{s["completed"] - 1:03d}.json', s['process'], min(deadline, time.monotonic() + 30))
        require(ud['request_id'] == req['request_id'], 'UD request receipt differs')
        row = compact_request(task, ordinal, result, owner,
                              {k: s['policy'].counts[k] - before[k] for k in before}, tasks,
                              s['wire'].request(req['request_id']), ud)
        qstart = time.monotonic_ns()
        quality, scorer_version = score_fn(task, result['generated_text'])
        row.update(quality=quality, scorer_version=scorer_version, quality_ms=(time.monotonic_ns() - qstart) / 1e6)
        row['wire_summary_observer_cumulative_ms'] = s['wire'].observer_ms
        row['arrival_ns'] = arrival
        io_start = time.monotonic_ns()
        name = f'request_{ordinal:03d}.json'
        writer.write(name, dict(row, receipt_scope='PRIMARY_BODY_FINAL_TIMING_IN_RUN_JSON'))
        end = time.monotonic_ns()
        row.update(completion_ns=end, application_ttft_ms=(row['first_available_ns'] - arrival) / 1e6,
                   application_e2e_ms=(end - arrival) / 1e6, request_persistence_ms=(end - io_start) / 1e6,
                   primary_receipt=name, primary_receipt_sha256=writer.hashes[name])
        rows.append(row)
    lifecycle.finish()
    return rows


def es_worker(experiment, cpu_audit, condition, output_dir, source_commit, expected_gpu_uuid,
              enable_approximate_experiment=False, enable_extended_experiment=False):
    remaining = allocation_guard()
    from .r5_d5_lightperf import approval
    p, pi, ai = approval(experiment, cpu_audit, cpu_only=False)
    launch_scope(p, condition, source_commit, expected_gpu_uuid, enable_approximate_experiment, enable_extended_experiment)
    require(remaining >= p['resources']['worker_deadline_seconds'] + 120, 'remaining Slurm time insufficient')
    require(host_available() >= 32 * 2**30 and shutil.disk_usage(Path(output_dir).resolve().parent).free >= 2**30,
            'host RAM/disk reserve insufficient')
    begin = time.monotonic_ns()
    deadline = time.monotonic() + p['resources']['worker_deadline_seconds']
    writer = SummaryWriter(fresh(output_dir, protected(p, experiment, cpu_audit)))
    sessions, primary = {}, None
    try:
        from .r4a_runtime import setup
        from .r4b2_kv import installed_contract
        from .r5_multi_artifacts import validate, runtime_view, physical, tokenizer_identity
        from .r5_i5_contracts import request
        from .r5_i5_gpu import receipt
        from .r5_i5r2b_capture import backend
        from .r5_runtime import encoder_for
        from .r5_semantic_index import SemanticIndex
        from .r5_perf_schedule import TINYBERT_REVISION
        from .r5_subsequence_physical import PhysicalOwner
        from .r5_subsequence_integrity import SCOPED
        from .r5_sparse_qkv import VERSION
        from .r5_i5_quality import score, SCORING_VERSION
        m, pm, specs, registry = validate(p['multi'])
        _, _, _, driver, tokenizer, gpu = setup(dict(base_prepared=p['base_prepared'], fixture=False,
                device='cuda:0', i5_gpu_policy_scope='R5_I5'), native=False, gpu_identity_policy='SAME_COMPATIBLE_GPU_CLASS')
        import torch
        free = torch.cuda.mem_get_info(0)[0]
        require(torch.cuda.device_count() == 1 and gpu['uuid'] == expected_gpu_uuid and free >= 12 * 2**30 and
                driver.model.config.max_position_embeddings == 2048, 'GPU UUID/device/context/12GiB reserve differs')
        require(tokenizer_identity(pm, tokenizer) == p['tokenizer_identity'], 'loaded tokenizer differs')
        torch.cuda.reset_peak_memory_stats(0)
        contract = installed_contract(driver)
        contract.update(sparse_protocol=VERSION, impact='streamed_original_M_row_l2_sum', generation_scope='native full prefill and native decode')
        tasks = p['stream']['tasks']
        encoder = None if condition == CONDITIONS[0] else encoder_for(dict(fixture=False, semantic_revision=TINYBERT_REVISION), tasks)
        require(encoder is None or encoder.metadata == p['semantic_identity'], 'semantic checkpoint differs from CPU plan')
        for aid in sorted({t['adapter_id'] for t in tasks}):
            spec = specs[aid]
            identity, _ = physical(runtime_view(pm, spec), p['tokenizer_identity'], driver.device, gpu['uuid'], contract, spec, registry)
            selected = [t for t in tasks if t['adapter_id'] == aid]
            ids = [t['request_id'] for t in selected]
            ud_out = Path(str(writer.root) + '__ud_' + aid)
            config = dict(engine=ENGINE, experiment=str(Path(experiment).resolve()), cpu_audit=str(Path(cpu_audit).resolve()),
                          source=p['source'], plan_inventory=pi, cpu_inventory=ai, performance_knobs=p['performance_knobs'],
                          session=uuid.uuid4().hex, identity=identity, output_dir=str(ud_out), deadline_monotonic=deadline,
                          requests=[request(t, t['token_ids'], t['policy'], identity) for t in selected])
            config_name = 'config_' + aid + '.json'
            writer.write(config_name, config)
            proc = subprocess.Popen([sys.executable, str(SCRIPT), '_ud', '--config', str(writer.root / config_name),
                    '--output-dir', str(ud_out)], cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES='', HIP_VISIBLE_DEVICES=''),
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            s = dict(process=proc, ud_out=ud_out, owner=None, channel=None, identity=identity,
                     ids=ids, completed=0, stopped=False, policy=PolicyCounter(), wire=WireSummary(ids))
            sessions[aid] = s
            ready = wait_file(ud_out / 'ready.json', proc, min(deadline, time.monotonic() + 120))
            require(ready['source'] == p['source'] and ready['user_device_cuda_used'] is False and ready['full_model_loaded'] is False,
                    'UD startup identity/isolation differs')
            ch = LightChannel(connect(ready['port']), config['session'], spec['matrix_sha256'], timeout=min(1800., deadline - time.monotonic()),
                              bandwidth_mbps=p['performance_knobs']['bandwidth_mbps'], trace=s['wire'])
            ch.condition_deadline = deadline
            s.update(channel=ch, index=None if encoder is None else SemanticIndex(encoder, clusters=1))
            if encoder is not None:
                s['owner'] = PhysicalOwner(identity, registry, capacity=512 * 2**20, evidence=None,
                                           trace=s['policy'], validation_mode=SCOPED)
        effective_backend = backend()
        host = host_available()
        require(host >= 32 * 2**30 and torch.cuda.mem_get_info(0)[0] >= 12 * 2**30, 'post-load reserves insufficient')
        initialized = time.monotonic_ns()
        rows = execute_stream(tasks, sessions, driver.model, driver.adapter, tokenizer, condition, writer, deadline,
                              score_fn=lambda t, text: (score(t['prepared']['sample'], text, t['prepared']['template_version']), SCORING_VERSION))
        sequence_end = time.monotonic_ns()
        uds = {}
        for aid, s in sessions.items():
            require(s['stopped'], 'missing final ACK/STOP/BYE')
            s['process'].wait(timeout=min(60., max(.001, deadline - time.monotonic())))
            require(s['process'].returncode == 0, 'UD failed during finalization')
            ui = verify(s['ud_out'])
            u = read_json(s['ud_out'] / 'ud.json')
            require(u['status'] == 'COMPLETE' and u['completed_request_ids'] == s['ids'] and
                    u['source'] == p['source'] and u['final_bye_sent'] is True, 'UD final receipt differs')
            wire_equal(s['wire'].rows['CONTROL'], u['wire']['CONTROL'])
            require(not s['wire'].errors, 'ES failed wire summary')
            uds[aid] = dict(path=str(s['ud_out']), inventory=ui, cpu=u['cpu'], service=u['service'], pacing=u['pacing'],
                            control_wire=u['wire']['CONTROL'], es_pacing=s['channel'].pacer.report())
        require(time.monotonic() < deadline and torch.cuda.max_memory_allocated(0) < 22 * 2**30 and backend() == effective_backend,
                'condition deadline/22GiB/backend guard failed')
        require(bound_receipts(experiment, cpu_audit) == (p, pi, ai), 'source/approval changed during condition')
        r = dict(engine=ENGINE, schema=SCHEMA, mode=MODE, status='COMPLETE', source=p['source'], plan_id=p['plan_id'],
                 plan_inventory=pi, cpu_inventory=ai, condition=condition, dataset=p['dataset'], scale=p['count'],
                 tasks_sha256=p['stream']['tasks_sha256'], performance_knobs=p['performance_knobs'], measurement_scope='NATIVE_LIGHT',
                 gpu=gpu, gpu_driver_version='NOT_MEASURED',
                 gpu_identity_receipt=receipt(pm, gpu, 'SAME_COMPATIBLE_GPU_CLASS', fixture=False),
                 backend=effective_backend, versions=native_versions(), precision_policy=dict(base='float16', ud='float32', validation=SCOPED),
                 cpu_concurrency=dict(es_threads=torch.get_num_threads(), affinity=sorted(os.sched_getaffinity(0)),
                                      ud={aid: u['cpu'] for aid, u in uds.items()}), warmup_requests=0,
                 initialized_excluded_ms=(initialized - begin) / 1e6, sequence_span_ms=(sequence_end - initialized) / 1e6,
                 run_wall_before_seal_ms=(time.monotonic_ns() - begin) / 1e6, final_sealing_ms='NOT_MEASURED',
                 resources=dict(gpu_free_after_load_bytes=free, peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(0),
                                peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(0), host_available_bytes=host,
                                cpu_count=len(os.sched_getaffinity(0)), primary_files_before_final=writer.files,
                                primary_bytes_before_final=writer.bytes, deadline_seconds=p['resources']['worker_deadline_seconds']),
                 requests=rows, ud=uds, persistence_before_final=writer.timings, errors=[],
                 independent_full_vocabulary_audit='NOT_EVALUATED', correctness_qualified_speedup='NOT_ESTABLISHED')
        writer.write('run.json', r)
    except BaseException as exc:
        primary = exc
        raise
    finally:
        cleanup_errors = []
        for s in sessions.values():
            for fn in (lambda s=s: s['owner'].dispose() if s['owner'] is not None else None,
                       lambda s=s: s['channel'].close() if s['channel'] is not None else None,
                       lambda s=s: reap(s['process'])):
                try:
                    fn()
                except BaseException as exc:
                    cleanup_errors.append(str(exc)[:512])
        seal_after(writer, primary, cleanup_errors)
        if primary is None and cleanup_errors:
            raise RuntimeError('native Light cleanup failed: ' + str(cleanup_errors))


def protected(p, experiment, cpu_audit):
    return [experiment, cpu_audit, p['original_experiment'], p['multi'], p['base_prepared'], *p['dataset_paths'].values()]


def launch_scope(p, condition, source_commit, expected_gpu_uuid, approximate, extended):
    require(condition in CONDITIONS and source_commit == p['source']['commit'] and
            isinstance(expected_gpu_uuid, str) and expected_gpu_uuid.startswith('GPU-'), 'condition/reviewed source/GPU UUID required')
    require(condition != CONDITIONS[2] or approximate, 'APPROX explicit opt-in required')
    require(extended or p['count'] == 5 and p['performance_knobs'] == knobs(), '20/50 or L3/L4 require separate extended experiment approval')


def run(experiment, cpu_audit, condition, output_dir, source_commit, expected_gpu_uuid, *,
        enable_gpu_experiment=False, enable_approximate_experiment=False, enable_extended_experiment=False):
    require(enable_gpu_experiment, 'explicit --enable-gpu-experiment required')
    remaining = allocation_guard()
    from .r5_d5_lightperf import approval
    p, _, _ = approval(experiment, cpu_audit, cpu_only=False)
    launch_scope(p, condition, source_commit, expected_gpu_uuid, enable_approximate_experiment, enable_extended_experiment)
    require(remaining >= p['resources']['worker_deadline_seconds'] + 120, 'insufficient remaining allocation')
    # Parent does not create output; child enforces fresh resolved path and input protection.
    argv = [sys.executable, str(SCRIPT), '_worker', '--experiment', str(Path(experiment).resolve()), '--cpu-audit',
            str(Path(cpu_audit).resolve()), '--condition', condition, '--output-dir', str(Path(output_dir).absolute()),
            '--source-commit', source_commit, '--expected-gpu-uuid', expected_gpu_uuid]
    if enable_approximate_experiment:
        argv.append('--enable-approximate-experiment')
    if enable_extended_experiment:
        argv.append('--enable-extended-experiment')
    process = subprocess.Popen(argv, cwd=ROOT, start_new_session=True)
    try:
        rc = process.wait(timeout=p['resources']['worker_deadline_seconds'])
        require(rc == 0, 'native Light worker failed; no retry/fallback')
    except BaseException:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
        raise
    from .r5_d5_lightperf_analysis import verify_run
    _, binding = verify_run(output_dir, experiment, cpu_audit)
    return dict(status='NATIVE_LIGHT_COMPLETE', binding=binding, correctness_qualified_speedup='NOT_ESTABLISHED')
