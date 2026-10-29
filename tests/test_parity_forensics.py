"""Synthetic CPU artifact contracts ONLY; never substitute these for SERAPH data."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from semcache.edgelora.communication import digest,digest_bytes,encode,header
from semcache.edgelora.r3a_artifacts import ROOT,THRESHOLDS,pack,read,sha,source_identity,write,write_csv
from semcache.edgelora.r3a_tiny import prepare_tiny
from semcache.edgelora.parity_forensics_audit import Evidence,audit_failure,compare_values,conclude

SCRIPT=ROOT/'scripts/96_diagnose_r3a_parity.py'


def fixture(root):
    """Build clearly synthetic FAILED records; NO GPU/network/benchmark execution."""
    prepared=prepare_tiny(root/'prepare_TEST_FIXTURE_ONLY');pilot=root/'pilot_TEST_FIXTURE_ONLY';pilot.mkdir()
    pm=read(prepared/'r3a_prepare_manifest.json');ref=read(prepared/'r3a_reference.json');ref['projections']={}
    identity=pm['identity'];es_pid,ud_pid=101,102;es_frames=[];ud_frames=[];layers=[]
    def frame(operation,h,payload,sequence,clock):
        h.update(session_id='TEST_FIXTURE_ONLY',adapter_sha256=identity['adapter_matrix_sha256'],sequence=sequence)
        packet,h=encode(h,payload)
        record=dict(operation=operation,pid=es_pid,monotonic_start_ns=clock,monotonic_end_ns=clock+100,
            wire_bytes=len(packet),header_bytes=len(packet)-len(payload),frame_sha256=digest_bytes(packet),
            header=h,complete=True,error=None)
        peer=dict(record,operation='recv' if operation=='send' else 'send',pid=ud_pid)
        es_frames.append(record);ud_frames.append(peer)
        return record
    frame('send',header('HELLO','handshake',-1),b'',0,0)
    frame('recv',header('READY','handshake',-1),b'',0,1000)
    for layer in range(32):
        payload=pack([float(layer+1)]*16,'float32');delta=pack([.01]*48,'float32')
        sent=frame('send',header('HIDDEN','warmup_000000',layer,'float32',[1,4,4]),payload,layer+1,2_000_000+layer*1_000_000)
        metadata=dict(pid=ud_pid,hidden_payload_sha256=digest_bytes(payload),user_device_cuda_used=False,
                      cpu_lora_ms=.01,monotonic_start_ns=1,monotonic_end_ns=2)
        received=frame('recv',header('DELTAS','warmup_000000',layer,'float32',[3,1,4,4],**metadata),delta,layer+1,2_500_000+layer*1_000_000)
        layers.append(dict(layer_id=layer,request_id='warmup_000000',hidden_payload_sha256=digest_bytes(payload),
            delta_payload_sha256=digest_bytes(delta),qkv_hidden_input_equal=dict(q=True,k=True,v=True),
            sent_frame=sent,received_frame=received,ud=metadata))
        for role in 'qkv':
            ref['projections'][f'{layer}.{role}']=dict(hidden_sha256=digest_bytes(payload) if layer==0 else digest('TEST_FIXTURE_ONLY other hidden'),
                native_output_sha256=digest(dict(TEST_FIXTURE_ONLY=True,layer=layer,role=role)),delta_sha256=digest('TEST_FIXTURE_ONLY native delta'))
    write(prepared/'r3a_reference.json',ref);pm['artifact_hashes']['r3a_reference.json']=sha(prepared/'r3a_reference.json');write(prepared/'r3a_prepare_manifest.json',pm)
    row=dict(phase='warmup',repeat=0,request_id='warmup_000000',parity_passed=False,layers_observed=32,qkv_outputs_observed=96,
        es_pid=es_pid,ud_pid=ud_pid,es_start_monotonic_ns=1_000_000,es_end_monotonic_ns=1_000_000_000)
    for direction,op in (('es_to_ud','send'),('ud_to_es','recv')):
        data=[f for f in es_frames[2:] if f['operation']==op]
        row.update({direction+'_frames':len(data),direction+'_payload_bytes':sum(f['header']['payload_bytes'] for f in data),
                    direction+'_wire_bytes':sum(f['wire_bytes'] for f in data)})
    parity=dict(thresholds=dict(THRESHOLDS),passed=False,status='PARITY_FAIL',max_abs=.1,relative_l2=.01,
        tolerance_violations=1,last_token_kl=0.,last_token_argmax_agreement=True)
    error=dict(operation='recv',pid=ud_pid,complete=False,error='ProtocolError: unexpected close / truncated frame',header={},wire_bytes=0,
        monotonic_start_ns=34_000_000,monotonic_end_ns=1_100_000_000)
    ud_frames.append(error)
    # EOF can be observed before the ES records its post-close timestamp.
    es=dict(execution_status='FAILED',error='RuntimeError: PARITY_FAIL',pid=es_pid,identity_sha256=pm['identity_sha256'],
        actual_network_socket_used=True,gpu_inference_performed=False,closed_monotonic_ns=1_100_000_100,
        rows=[row],parities=[parity],per_layer_trace=[dict(request_id=row['request_id'],layers=layers)],TEST_FIXTURE_ONLY=True)
    ud=dict(execution_status='FAILED',pid=ud_pid,identity_sha256=pm['identity_sha256'],
        adapter_matrix_sha256=identity['adapter_matrix_sha256'],user_device_cuda_used=False,actual_network_socket_used=True,
        cpu=dict(effective_threads=None),layers=[dict(request_id=row['request_id'],layer_id=l['layer_id'],metrics=l['ud'],
            hidden_frame_sha256=l['sent_frame']['frame_sha256'],delta_frame_sha256=l['received_frame']['frame_sha256']) for l in layers],TEST_FIXTURE_ONLY=True)
    cleanup=dict(all_children_exited=True,children=[dict(pid=es_pid,returncode=1),dict(pid=ud_pid,returncode=1)])
    for name,value in (('r3a_es_result.json',es),('r3a_ud_result.json',ud),('r3a_environment.json',dict(es=es,ud=ud)),
        ('r3a_parity.json',dict(parity_passed=False,thresholds=dict(THRESHOLDS),observations=[parity])),
        ('r3a_lifecycle.json',cleanup),('r3a_decision.json',dict(execution_status='FAILED',parity_passed=False))):write(pilot/name,value)
    for name,records in (('r3a_iterations.jsonl',[dict(row=row,parity=parity,layers=layers)]),
                         ('r3a_wire_trace.jsonl',es_frames),('r3a_ud_wire_trace.jsonl',ud_frames)):
        (pilot/name).write_text(''.join(json.dumps(r)+'\n' for r in records))
    write_csv(pilot/'r3a_latency_raw.csv',[row])
    write(pilot/'r3a_manifest.json',dict(execution_status='FAILED',configuration=dict(phase='pilot'),code=pm['code'],
        prepared_manifest_sha256=sha(prepared/'r3a_prepare_manifest.json'),identity_sha256=pm['identity_sha256'],cleanup=cleanup,
        artifact_hashes={p.name:sha(p) for p in pilot.iterdir()},TEST_FIXTURE_ONLY=True))
    return prepared,pilot


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.prepared,self.pilot=fixture(Path(self.tmp.name))

    def test_verified_fixture_failure_remains_unresolved_and_localizes_only_a_range(self):
        report=audit_failure(self.prepared,self.pilot)
        self.assertTrue(report['TEST_FIXTURE_ONLY']);self.assertEqual(report['numerical_cause'],'UNRESOLVED')
        self.assertEqual(report['first_bitwise_hidden_divergence']['layer'],1)
        self.assertEqual(report['localization'],'between layer-0 hidden ingress and layer-1 hidden ingress')
        self.assertEqual(report['actual_schedule']['directions']['ES_TO_UD']['data_frames'],32)
        self.assertEqual(report['actual_schedule']['qkv_outputs'],96)
        self.assertEqual(report['root_failure'],'STRICT_LOGIT_PARITY_FAIL')
        self.assertEqual(report['shutdown']['status'],'POST_PARITY_CLOSE_SYMPTOM_SUPPORTED')
        self.assertEqual(report['shutdown']['exact_es_parity_raise_timestamp'],'NOT_RECORDED')
        self.assertTrue(all(v['status']=='UNRESOLVED' for v in report['hypotheses'].values()))

    def test_source_identity_and_thresholds_unchanged_by_new_module_names(self):
        recorded=read(self.prepared/'r3a_prepare_manifest.json')['code']['files']
        self.assertEqual(recorded,source_identity()['files'])
        self.assertFalse(any('parity_forensics' in name or '96_diagnose' in name for name in recorded))
        self.assertEqual(THRESHOLDS,dict(atol=1e-3,rtol=1e-3,max_abs=1e-2,relative_l2=1e-3,last_token_kl=1e-5,last_token_argmax_agreement=True))

    def test_source_hash_mismatch_fail_closed_without_altering_sources(self):
        path=self.prepared/'r3a_prepare_manifest.json';m=read(path)
        name=next(iter(m['code']['files']));m['code']['files'][name]='0'*64;m['code']['sha256']=digest(m['code']['files']);write(path,m)
        with self.assertRaisesRegex(ValueError,'untouched original source'):audit_failure(self.prepared,self.pilot)

    def test_pilot_tamper_is_rejected(self):
        path=self.pilot/'r3a_wire_trace.jsonl';path.write_bytes(path.read_bytes()+b'\n')
        with self.assertRaisesRegex(ValueError,'SHA mismatch'):audit_failure(self.prepared,self.pilot)

    def test_unrecorded_shutdown_causality_is_not_assumed(self):
        path=self.pilot/'r3a_ud_wire_trace.jsonl';records=[json.loads(s) for s in path.read_text().splitlines()]
        records[-1]['wire_bytes']=4;path.write_text(''.join(json.dumps(r)+'\n' for r in records))
        mp=self.pilot/'r3a_manifest.json';m=read(mp);m['artifact_hashes'][path.name]=sha(path);write(mp,m)
        report=audit_failure(self.prepared,self.pilot)
        self.assertEqual(report['shutdown']['status'],'UNRESOLVED')

    def test_evidence_reader_detects_input_changed_during_analysis(self):
        e=Evidence();path=self.pilot/'r3a_decision.json';e.read(path);write(path,dict(changed=True))
        with self.assertRaises(ValueError):e.unchanged()

    def test_metrics_keep_negative_values_and_reference_norm(self):
        m=compare_values([-1.,3.],[-2.,4.]);self.assertEqual(m['max_abs'],1.)
        self.assertAlmostEqual(m['relative_l2'],(2/20)**.5);self.assertEqual(m['unequal_elements'],2)
        with self.assertRaises(ValueError):compare_values([float('nan')],[1.])
        with self.assertRaises(ValueError):compare_values([1.],[1.,2.])


class ConclusionTests(unittest.TestCase):
    def control(self):
        return dict(label='control',vs_original=dict(unequal_elements=0,passed=True),
            logits_sha256='same',original_logits_sha256='same',original_projection_mismatches=[])

    def projection(self):
        zero=dict(unequal_elements=0)
        return dict(layer=0,role='q',comparisons={n:dict(zero) for n in
            ('native_vs_gpu_reconstruction','gpu_delta_vs_cpu_delta','cpu_delta_vs_unpacked_cuda')},
            fp16_output_flip_count=0,serialization_byte_identity=True)

    def test_no_gpu_or_argmax_only_is_unresolved(self):
        self.assertEqual(conclude()['conclusions'],['UNRESOLVED'])
        a=dict(controls=[self.control()],repeat_comparison=dict(unequal_elements=0,passed=True),projections=[])
        self.assertEqual(conclude(a)['conclusions'],['UNRESOLVED'])

    def test_multiple_objective_failure_classes_retained(self):
        c=self.control();c['vs_original']=dict(unequal_elements=1,passed=True)
        p=self.projection();p['comparisons']['gpu_delta_vs_cpu_delta']['unequal_elements']=1
        p['comparisons']['native_vs_gpu_reconstruction']['unequal_elements']=1;p['serialization_byte_identity']=False
        a=dict(controls=[c],projections=[p])
        result=conclude(a)
        self.assertEqual(set(result['conclusions']),{'REFERENCE_NONREPRODUCIBLE','GPU_RECONSTRUCTION_MISMATCH','CPU_GPU_DELTA_NUMERICS','WIRE_BYTES_OR_LAYOUT_MISMATCH'})
        self.assertEqual(result['causal_resolution'],'UNRESOLVED')

    def test_assembly_mismatch_requires_actual_observed_vs_assembled_difference(self):
        self.assertEqual(conclude(es_replay=dict(projections=[dict(layer=0,role='q',wire_matches_local_cpu=True,
            observed_output_matches_assembly=True)]))['conclusions'],['UNRESOLVED'])
        self.assertEqual(conclude(es_replay=dict(projections=[dict(layer=0,role='q',wire_matches_local_cpu=True,
            observed_output_matches_assembly=False)]))['conclusions'],['ES_ASSEMBLY_MISMATCH'])

    def test_observer_drift_is_a_reference_control_failure(self):
        a=dict(controls=[self.control()],instrumented_forward=dict(vs_control=dict(unequal_elements=1,passed=True)))
        r=conclude(a);self.assertEqual(r['conclusions'],['REFERENCE_NONREPRODUCIBLE'])
        self.assertIn('observer-dependent',r['findings'][0]['scope'])


class CLITests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.prepared,self.pilot=fixture(Path(self.tmp.name))
        base=ROOT/'results/r3/a';base.mkdir(parents=True,exist_ok=True)
        self.output=tempfile.TemporaryDirectory(prefix='parity_forensics_TEST_FIXTURE_ONLY_',dir=base)
        self.addCleanup(self.output.cleanup);self.out=Path(self.output.name)

    def run_cli(self,phase,extra=()):
        return subprocess.run([sys.executable,str(SCRIPT),'--phase',phase,'--prepared',str(self.prepared),
            '--pilot-inputs',str(self.pilot),'--output-dir',str(self.out),*extra],capture_output=True,text=True,timeout=15.)

    def test_cpu_audit_cli_exports_two_reports_and_never_mutates_inputs(self):
        before={str(p):sha(p) for d in (self.prepared,self.pilot) for p in d.iterdir()}
        run=self.run_cli('audit');self.assertEqual(run.returncode,0,run.stderr)
        r=read(self.out/'r3a_parity_forensics.json')
        self.assertEqual(r['classification'],'R3A_PARITY_DIAGNOSTIC');self.assertFalse(r['PRIMARY_PERFORMANCE_ELIGIBLE'])
        self.assertEqual(r['gpu_findings_status'],'NOT_RUN');self.assertEqual(r['assessment']['conclusions'],['UNRESOLVED'])
        self.assertEqual(json.loads(run.stdout)['original_parity'],r['offline_audit']['parity'])
        self.assertGreater((self.out/'r3a_parity_forensics_summary.txt').stat().st_size,0)
        self.assertEqual(before,{str(p):sha(p) for d in (self.prepared,self.pilot) for p in d.iterdir()})
        outputs={p.name:sha(p) for p in self.out.iterdir()}
        self.assertNotEqual(self.run_cli('audit').returncode,0);self.assertEqual(outputs,{p.name:sha(p) for p in self.out.iterdir()})

    def test_dry_run_is_explicitly_no_model_and_unaudited(self):
        run=self.run_cli('dry-run');self.assertEqual(run.returncode,0,run.stderr)
        r=read(self.out/'r3a_parity_forensics.json');self.assertFalse(r['readiness']['original_evidence_audited'])
        self.assertTrue(r['readiness']['no_model_or_network_executed']);self.assertFalse(r['gpu_inference_performed'])

    def test_gpu_requires_explicit_opt_in_without_loading_torch(self):
        run=self.run_cli('gpu-probe');self.assertNotEqual(run.returncode,0)
        self.assertIn('--enable-gpu-probes',run.stderr)
        r=read(self.out/'r3a_parity_forensics.json');self.assertEqual(r['gpu_findings_status'],'NOT_RUN')

    def test_missing_original_is_visible_not_substituted(self):
        self.prepared=Path(self.tmp.name)/'absent_original'
        run=self.run_cli('audit');self.assertNotEqual(run.returncode,0)
        r=read(self.out/'r3a_parity_forensics.json')
        self.assertEqual(r['execution_status'],'FAILED');self.assertEqual(r['gpu_findings_status'],'NOT_RUN')
        self.assertEqual(r['assessment']['conclusions'],['UNRESOLVED']);self.assertNotIn('offline_audit',r)

    def test_help_and_no_benchmark_in_cli(self):
        run=subprocess.run([sys.executable,str(SCRIPT),'--help'],capture_output=True,text=True)
        self.assertEqual(run.returncode,0,run.stderr);self.assertIn('gpu-probe',run.stdout)
        self.assertNotIn('--measured-runs',run.stdout)


@unittest.skipUnless(os.environ.get('SEMCACHE_R3A_PARITY_GPU')=='1','GPU forensics NOT_RUN; requires original artifacts and explicit SERAPH opt-in')
class AllocatedGPUProbe(unittest.TestCase):
    def test_original_seraph_probe(self):
        run=subprocess.run([sys.executable,str(SCRIPT),'--phase','gpu-probe',
            '--prepared',os.environ['SEMCACHE_R3A_PARITY_PREPARED'],'--pilot-inputs',os.environ['SEMCACHE_R3A_PARITY_PILOT'],
            '--output-dir',os.environ['SEMCACHE_R3A_PARITY_OUTPUT'],'--enable-gpu-probes'],timeout=5400.)
        self.assertEqual(run.returncode,0)
        result=read(Path(os.environ['SEMCACHE_R3A_PARITY_OUTPUT'])/'r3a_parity_forensics.json')
        self.assertFalse(result['PRIMARY_PERFORMANCE_ELIGIBLE']);self.assertEqual(result['classification'],'R3A_PARITY_DIAGNOSTIC')


if __name__=='__main__':unittest.main()
