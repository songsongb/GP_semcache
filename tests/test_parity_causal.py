"""TEST_FIXTURE_ONLY evidence/arithmetic tests. No OPT, Torch, GPU or benchmark."""
import copy
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

from semcache.edgelora.communication import ProtocolError,digest,digest_bytes
from semcache.edgelora.r3a_artifacts import (ENDPOINT,ROOT,THRESHOLDS,check_parity,load_matrices,
    pack,read,sha,source_identity,unpack,write)
from semcache.edgelora.parity_forensics_audit import CLASSIFICATION,Evidence
from semcache.edgelora.parity_causal_evidence import (SERIALIZATION,assess,audit_prior_forensics,
    compare_projections,control_failure,descriptor_valid,projection_map,validate_identity)
from semcache.edgelora.parity_causal_process import finish

SCRIPT=ROOT/'scripts/97_attribute_r3a_cpu_delta.py'


def synthetic_workers():
    """Synthetic schema sentinels, NOT model or experimental evidence."""
    reference=[float(i%7) for i in range(32*7)];candidate=[v+.1 for v in reference]
    strict=check_parity(candidate,reference,[1,32,7]);strict['unequal_elements']=224
    exact=check_parity(reference,reference,[1,32,7]);exact['unequal_elements']=0
    ident=dict(model='facebook/opt-2.7b',layers=32,length=32,hidden_size=2560,dtype='float16',
        hidden_dtype='float16',delta_dtype='float32',adapter='user_a',rank=8,endpoint=ENDPOINT,
        token_ids=list(range(32)),token_ids_sha256=digest(list(range(32))),TEST_FIXTURE_ONLY=True)
    cpu=dict(TEST_FIXTURE_ONLY=True,affinity=[0,1,2,3],torch_threads=4,interop_threads=1)
    expected=dict(identity=ident,identity_sha256=digest(ident),gpu_uuid='TEST_FIXTURE_ONLY_UUID',
        reference_logits_sha256=digest_bytes(pack(reference,'float32')),reference_logits_shape=[1,32,7],
        prepared_manifest_sha256=digest('TEST_FIXTURE_ONLY prepare'),pilot_manifest_sha256=digest('TEST_FIXTURE_ONLY pilot'),
        thresholds=dict(THRESHOLDS),cpu_affinity=[0,1,2,3],cpu_threads=4,
        pilot_parity={k:strict[k] for k in ('max_abs','relative_l2','tolerance_violations','last_token_kl','last_token_argmax_agreement','passed')})
    def tensor(tag,layer,role,dtype='torch.float16'):
        return dict(sha256=digest(dict(TEST_FIXTURE_ONLY=True,tag=tag,layer=layer,role=role)),
                    shape=[1,32,2560],dtype=dtype)
    records=[];points=[];layers=[]
    for layer in range(32):
        for role in 'qkv':
            hidden=tensor('cpu hidden',layer,'shared');base=tensor('cpu base',layer,role)
            delta=tensor('cpu delta',layer,role,'torch.float32');qkv=tensor('cpu qkv',layer,role)
            records.append(dict(layer=layer,role=role,hidden=hidden,base=base,local_cpu_delta=delta,received_delta=delta,
                observed_output=qkv,observed_output_matches_assembly=True,wire_matches_local_cpu=True))
            points.append(dict(layer=layer,role=role,hidden=hidden if layer==0 else tensor('native hidden',layer,'shared'),
                independent_gpu_base=base if layer==0 else tensor('native base',layer,role),cpu_delta=delta,
                predicted_cpu_assembled=qkv,native_output=tensor('native qkv',layer,role),
                comparisons=dict(gpu_delta_vs_cpu_delta=dict(max_abs=1e-7,unequal_elements=1,TEST_FIXTURE_ONLY=True)),
                fp16_output_flip_count=1))
        layers.append(dict(layer_id=layer,hidden_payload_sha256=records[-1]['hidden']['sha256'],
                           delta_payload_sha256=digest(dict(TEST_FIXTURE_ONLY=True,layer=layer,bundle=True))))
    expected['pilot_layer_payload_hashes']=[dict(layer=l['layer_id'],hidden=l['hidden_payload_sha256'],delta=l['delta_payload_sha256']) for l in layers]
    native_desc=dict(sha256=expected['reference_logits_sha256'],dtype='float32',shape=[1,32,7],bytes=224*4,serialization=SERIALIZATION)
    desc=dict(native_desc,sha256=digest_bytes(pack(candidate,'float32')))
    hashes={f"{p['layer']}.{p['role']}":dict(hidden_sha256=p['hidden']['sha256'],native_output_sha256=p['native_output']['sha256']) for p in points}
    controls=[dict(label=label,vs_original=copy.deepcopy(exact),logits=copy.deepcopy(native_desc),
        projection_hashes={} if label=='native_no_hooks' else copy.deepcopy(hashes),original_projection_mismatches=[])
        for label in ('native_no_hooks','native_passive_hashes','native_gpu_delta_control')]
    def common(pid):
        return dict(TEST_FIXTURE_ONLY=True,execution_status='COMPLETE',classification=CLASSIFICATION,PRIMARY_PERFORMANCE_ELIGIBLE=False,
            comparison_identity=copy.deepcopy(expected),process_metadata=dict(pid=pid,gpu=dict(uuid=expected['gpu_uuid']),
                cpu_runtime=copy.deepcopy(cpu),effective_affinity=[0,1,2,3]),cpu_runtime=copy.deepcopy(cpu))
    a=common(101);a.update(controls=controls,all_gpu_reconstructions_exact=True,gpu_reconstruction_count=96,
                          same_hidden_projections=points)
    b=copy.deepcopy(a);b['process_metadata']['pid']=102
    local=common(103);local.update(logits=copy.deepcopy(desc),vs_original_logits=copy.deepcopy(strict),projections=copy.deepcopy(records),
        weights_verified_after=True,gpu_lora_on_es=False,actual_network_socket_used=False,distinct_ud_process_used=False,cpu_delta_is_forward_source=True)
    replay=common(104);replay.update(logits=copy.deepcopy(desc),vs_original_logits=copy.deepcopy(strict),projections=copy.deepcopy(records),
        weights_verified_after=True,gpu_lora_on_es=False,actual_network_socket_used=True,distinct_ud_process_used=True,layers=layers,
        ud_process_metadata=dict(pid=105,user_device_cuda_used=False,cpu_runtime=copy.deepcopy(cpu),cpu=dict(effective_affinity=[0,1,2,3],effective_threads=4)),
        wire_accounting=dict(hidden_frames=32,delta_frames=32))
    return a,b,local,replay,expected,copy.deepcopy(replay)


class CausalGates(unittest.TestCase):
    def setUp(self):self.a,self.b,self.local,self.replay,self.expected,self.prior=synthetic_workers()
    def assess(self):return assess(self.a,self.b,self.local,self.replay,self.expected,self.prior)

    def test_synthetic_positive_requires_all96_and_full_logits_but_not_old_unrecorded_logits(self):
        result=self.assess()
        self.assertTrue(result['sufficiency_established']);self.assertEqual(result['qkv_hashes_matched'],96)
        self.assertTrue(result['full_logits_bitwise_matched']);self.assertTrue(result['first_cpu_numeric_effect_isolated'])
        self.assertFalse(result['local_strict_parity_passed'])
        self.assertIn('NOT_ESTIMABLE',result['original_failed_trace_bitwise_status'])

    def test_native_control_drift_stops_attribution_even_if_gate_passes(self):
        self.a['controls'][0]['vs_original']['unequal_elements']=1
        self.assertTrue(self.a['controls'][0]['vs_original']['passed'])
        self.assertEqual(self.assess()['conclusion'],'REFERENCE_NONREPRODUCIBLE')

    def test_gpu_reconstruction_is_a_distinct_failure(self):
        self.a['all_gpu_reconstructions_exact']=False
        self.assertEqual(control_failure(self.a),'GPU_RECONSTRUCTION_MISMATCH')
        self.assertFalse(self.assess()['sufficiency_established'])

    def test_full_logits_hash_difference_is_not_hidden_by_equal_strict_metrics(self):
        self.local['logits']['sha256']=digest('TEST_FIXTURE_ONLY other full logits')
        result=self.assess();self.assertFalse(result['sufficiency_established'])
        self.assertFalse(result['full_logits_bitwise_matched'])

    def test_missing_duplicate_out_of_order_and_extra_roles_rejected(self):
        for operation in ('missing','duplicate','badrole','order'):
            with self.subTest(operation=operation):
                x=copy.deepcopy(self.local['projections'])
                if operation=='missing':x.pop()
                if operation=='duplicate':x[-1]=copy.deepcopy(x[0])
                if operation=='badrole':x[0]['role']='w'
                if operation=='order':x[0],x[1]=x[1],x[0]
                with self.assertRaises(ProtocolError):compare_projections(x,self.replay['projections'])

    def test_same_hidden_first_cpu_delta_difference_localized_without_magnitude_invention(self):
        self.replay['projections'][0]['received_delta']['sha256']=digest('TEST_FIXTURE_ONLY changed delta')
        r=self.assess();p=r['local_vs_replay']['first_difference']
        self.assertTrue(p['same_hidden']);self.assertEqual(p['first_different_component'],'cpu_delta')
        self.assertEqual(p['numeric_magnitude'],'NOT_ESTIMABLE_HASH_ONLY');self.assertFalse(r['sufficiency_established'])

    def test_later_hidden_mismatch_is_downstream_not_matched_input_arithmetic(self):
        for role in 'qkv':
            self.replay['projections'][3+'qkv'.index(role)]['hidden']['sha256']=digest('TEST_FIXTURE_ONLY layer1 different hidden')
        r=self.assess();p=r['local_vs_replay']['first_difference']
        self.assertEqual(p['layer'],1);self.assertFalse(p['same_hidden'])
        self.assertFalse(p['matched_input_arithmetic_claim_allowed']);self.assertFalse(r['sufficiency_established'])

    def test_original_replay_or_pilot_difference_prevents_original_failure_attribution(self):
        self.prior['projections'][0]['base']['sha256']=digest('TEST_FIXTURE_ONLY prior base differs')
        self.assertFalse(self.assess()['sufficiency_established'])
        self.prior=copy.deepcopy(self.replay)
        self.replay['layers'][0]['delta_payload_sha256']=digest('TEST_FIXTURE_ONLY original payload differs')
        self.assertFalse(self.assess()['sufficiency_established'])

    def test_changed_identity_uuid_model_dtype_adapter_tokens_reference_and_thresholds_rejected(self):
        for field in ('gpu_uuid','identity_sha256','reference_logits_sha256','prepared_manifest_sha256','cpu_affinity','thresholds'):
            bad=copy.deepcopy(self.expected);bad[field]='TEST_FIXTURE_ONLY bad'
            with self.subTest(field=field),self.assertRaises(ProtocolError):validate_identity(bad,self.expected)
        for field,value in (('model','facebook/opt-125m'),('dtype','float32'),('adapter','user_b'),('layers',31),('length',16)):
            bad=copy.deepcopy(self.expected);bad['identity'][field]=value;bad['identity_sha256']=digest(bad['identity'])
            with self.subTest(field=field),self.assertRaises(ProtocolError):validate_identity(bad,bad)

    def test_shape_endianness_noncanonical_logit_metadata_rejected(self):
        for field,value in (('shape',[1,1,7]),('dtype','float16'),('bytes',1),('serialization','native endian'),('sha256','bad')):
            bad=dict(self.local['logits']);bad[field]=value
            with self.subTest(field=field),self.assertRaises(ProtocolError):descriptor_valid(bad)
        self.a['controls'][0]['logits']['serialization']='TEST_FIXTURE_ONLY wrong layout'
        with self.assertRaises(ProtocolError):self.assess()

    def test_process_cpu_runtime_placement_assembly_counts_fail_closed(self):
        for field in ('pid','cpu','cuda','assembly','frames'):
            replay=copy.deepcopy(self.replay)
            if field=='pid':replay['process_metadata']['pid']=self.local['process_metadata']['pid']
            if field=='cpu':replay['cpu_runtime']['torch_threads']=1
            if field=='cuda':replay['ud_process_metadata']['user_device_cuda_used']=True
            if field=='assembly':replay['projections'][0]['observed_output_matches_assembly']=False
            if field=='frames':replay['wire_accounting']['hidden_frames']=31
            with self.subTest(field=field),self.assertRaises(ProtocolError):assess(self.a,self.b,self.local,replay,self.expected,self.prior)

    def test_bitwise_differences_flips_strict_failure_sufficiency_are_separate_gates(self):
        self.a['same_hidden_projections'][0]['fp16_output_flip_count']=0
        self.assertFalse(self.assess()['sufficiency_established'])
        self.a['same_hidden_projections'][0]['fp16_output_flip_count']=1
        self.local['vs_original_logits']['passed']=True
        self.assertFalse(self.assess()['sufficiency_established'])

    def test_matched_magnitude_is_kept_separate_from_unmatched_downstream_magnitude(self):
        self.replay['projections'][0]['local_oracle_numeric_comparisons']=dict(same_hidden=True,metrics=dict(qkv=dict(max_abs=0.)))
        r=self.assess();self.assertTrue(r['local_vs_replay']['rows'][0]['measured_numeric_comparisons']['same_hidden'])


class SyntheticArithmetic(unittest.TestCase):
    def test_nonzero_all32_scalar_local_vs_serialization_roundtrip_and_negative_delta_intervention(self):
        from semcache.edgelora.r3a_tiny import prepare_tiny,forward,linear
        with tempfile.TemporaryDirectory() as temporary:
            directory=prepare_tiny(Path(temporary)/'TEST_FIXTURE_ONLY')
            base=read(directory/'r3a_tiny_base.json');am=read(directory/'r3a_adapter_manifest.json')
            matrices=load_matrices(directory,am,torch_backend=False);identity=am['identity'];traces=[[],[],[]]
            def provider(mode):
                def delta(layer,role,x):
                    result=linear(linear(x,matrices[layer,role,'A'],4,8),matrices[layer,role,'B'],8,4)
                    self.assertTrue(any(result));traces[mode].append((layer,role,digest_bytes(pack(x,'float32'))))
                    if mode==1:result=unpack(pack(result,'float32'),'float32')
                    if mode==2 and layer==0 and role=='v':result=[v+.1 for v in result]
                    return result
                return delta
            a,b,c=[forward(identity['token_ids'],base,provider(mode)) for mode in range(3)]
            self.assertEqual(pack(a,'float32'),pack(b,'float32'));self.assertNotEqual(pack(a,'float32'),pack(c,'float32'))
            self.assertEqual(traces[0],traces[1]);self.assertEqual(len(traces[0]),96)
            self.assertTrue(check_parity(b,a,[1,4,7])['passed']);self.assertFalse(check_parity(c,a,[1,4,7])['passed'])

    def test_fp32_tiny_delta_can_flip_fp16_without_itself_proving_full_logits_failure(self):
        midpoint=1.+2.**-11
        left=unpack(pack([midpoint-2.**-24],'float16'),'float16')[0]
        right=unpack(pack([midpoint+2.**-24],'float16'),'float16')[0]
        self.assertNotEqual(left,right)
        self.assertTrue(check_parity([right],[left],[1,1,1])['passed'])


class PriorAudit(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.directory=Path(self.tmp.name)
        a,b,local,replay,expected,_=synthetic_workers()
        pm=dict(identity=expected['identity'],code=source_identity(),artifact_hashes={'r3a_reference_logits.bin':expected['reference_logits_sha256']})
        self.pm=pm
        self.offline=dict(prepared_manifest_sha256=expected['prepared_manifest_sha256'],pilot_manifest_sha256=expected['pilot_manifest_sha256'],
            reference_identity=expected['identity'],parity=replay['vs_original_logits'],actual_schedule=dict(layers=32,qkv_outputs=96),
            process_metadata=dict(original_es_gpu=dict(uuid=expected['gpu_uuid']),original_ud_cpu=dict(effective_threads=4,effective_affinity=[0,1,2,3])))
        sources={str((ROOT/'scripts/96_diagnose_r3a_parity.py').relative_to(ROOT)):sha(ROOT/'scripts/96_diagnose_r3a_parity.py')}
        report=dict(TEST_FIXTURE_ONLY=True,execution_status='COMPLETE',gpu_findings_status='COMPLETE',classification=CLASSIFICATION,
            PRIMARY_PERFORMANCE_ELIGIBLE=False,prepared_manifest_sha256=expected['prepared_manifest_sha256'],
            production_source_identity=source_identity(),diagnostic_source_hashes=sources,offline_audit=self.offline)
        for key,folder,current in (('native_a','native-a',a),('native_b','native-b',b),('es_replay','es',replay)):
            worker=copy.deepcopy(current)
            if key!='es_replay':
                worker['controls']=[dict(vs_original=c['vs_original'],logits_sha256=c['logits']['sha256'],
                    projection_hashes=c['projection_hashes'],original_projection_mismatches=[]) for c in current['controls'][1:]]
                worker.update(repeat_comparison=dict(unequal_elements=0),projections=current['same_hidden_projections'][:6])
            (self.directory/folder).mkdir()
            write(self.directory/folder/'worker_result.json',worker)
            write(self.directory/folder/'lifecycle.json',dict(all_children_exited=True,children=[dict(returncode=0,forced_termination=False)]))
            report[key]=worker
        report['diagnostic_output_hashes']={str(p.relative_to(self.directory)):sha(p) for p in self.directory.rglob('*') if p.is_file()}
        write(self.directory/'r3a_parity_forensics.json',report)

    def test_original_hashes_checked_and_missing_logit_sha_not_invented(self):
        r=audit_prior_forensics(self.directory,Evidence(),self.offline,self.pm)
        self.assertEqual(r['prior_selected_probe_count'],6);self.assertIn('NOT_ESTIMABLE',r['original_full_logit_bitwise_comparison'])

    def test_stored_worker_tamper_rejected(self):
        with (self.directory/'es/worker_result.json').open('ab') as stream:stream.write(b' ')
        with self.assertRaisesRegex(ProtocolError,'SHA'):audit_prior_forensics(self.directory,Evidence(),self.offline,self.pm)

    def test_required_worker_hash_cannot_be_omitted(self):
        report=read(self.directory/'r3a_parity_forensics.json')
        report['diagnostic_output_hashes'].pop('es/worker_result.json')
        write(self.directory/'r3a_parity_forensics.json',report)
        with self.assertRaisesRegex(ProtocolError,'required worker hashes'):audit_prior_forensics(self.directory,Evidence(),self.offline,self.pm)

    def test_different_prepare_and_diagnostic_source_hash_rejected(self):
        report=read(self.directory/'r3a_parity_forensics.json')
        report['prepared_manifest_sha256']=digest('other TEST_FIXTURE_ONLY prepare')
        write(self.directory/'r3a_parity_forensics.json',report)
        with self.assertRaisesRegex(ProtocolError,'mismatch'):audit_prior_forensics(self.directory,Evidence(),self.offline,self.pm)
        report['prepared_manifest_sha256']=self.offline['prepared_manifest_sha256']
        report['diagnostic_source_hashes']['scripts/96_diagnose_r3a_parity.py']='0'*64
        write(self.directory/'r3a_parity_forensics.json',report)
        with self.assertRaisesRegex(ProtocolError,'SHA'):audit_prior_forensics(self.directory,Evidence(),self.offline,self.pm)


class CLIAndLifecycle(unittest.TestCase):
    def test_help_and_model_free_import(self):
        run=subprocess.run([sys.executable,str(SCRIPT),'--help'],capture_output=True,text=True)
        self.assertEqual(run.returncode,0,run.stderr);self.assertIn('gpu-intervention',run.stdout)
        code="import semcache.edgelora.parity_causal_probe; import sys; assert not any(n in sys.modules for n in ('torch','transformers','peft'))"
        run=subprocess.run([sys.executable,'-c',code],env=dict(os.environ,PYTHONPATH=str(ROOT/'src')),capture_output=True,text=True)
        self.assertEqual(run.returncode,0,run.stderr)

    def test_dry_run_missing_audit_and_no_overwrite(self):
        with tempfile.TemporaryDirectory(prefix='parity_causal_TEST_FIXTURE_ONLY_',dir=ROOT/'results/r3/a') as temporary:
            base=Path(temporary);output=base/'dry';missing=base/'ABSENT_ORIGINAL_TEST'
            command=[sys.executable,str(SCRIPT),'--phase','dry-run','--prepared',str(missing),'--pilot-inputs',str(missing),
                     '--forensic-inputs',str(missing),'--output-dir',str(output)]
            first=subprocess.run(command,capture_output=True,text=True);self.assertEqual(first.returncode,0,first.stderr)
            report=read(output/'r3a_cpu_delta_causal.json');self.assertFalse(report['original_evidence_audited'])
            self.assertEqual(report['gpu_findings_status'],'NOT_RUN');self.assertFalse(report['assessment']['sufficiency_established'])
            original_hashes={p.name:sha(p) for p in output.iterdir()}
            second=subprocess.run(command,capture_output=True,text=True);self.assertNotEqual(second.returncode,0)
            self.assertEqual(original_hashes,{p.name:sha(p) for p in output.iterdir()})
            command[command.index('dry-run')]='audit';command[-1]=str(base/'audit')
            failure=subprocess.run(command,capture_output=True,text=True);self.assertNotEqual(failure.returncode,0)
            failed=read(base/'audit/r3a_cpu_delta_causal.json');self.assertEqual(failed['execution_status'],'FAILED')
            self.assertEqual(failed['gpu_findings_status'],'NOT_RUN');self.assertNotIn('local_cpu_oracle',failed)

    def test_nested_input_output_rejected_and_explicit_gpu_opt_in_parsing(self):
        spec=importlib.util.spec_from_file_location('causal_cli',SCRIPT);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory(prefix='parity_causal_TEST_FIXTURE_ONLY_',dir=ROOT/'results/r3/a') as temporary:
            with self.assertRaisesRegex(ProtocolError,'immutable'):module.new_output(Path(temporary)/'nested',(Path(temporary),))
        parsed=module.args(['--phase','gpu-intervention','--prepared','a','--pilot-inputs','b','--forensic-inputs','c','--output-dir','d'])
        self.assertFalse(parsed.enable_gpu_probes)

    def test_process_cleanup_reaps_isolated_and_inherited_groups(self):
        for group in (True,False):
            with self.subTest(group=group),tempfile.TemporaryDirectory() as temporary:
                stdout=(Path(temporary)/'stdout').open('w');stderr=(Path(temporary)/'stderr').open('w')
                process=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'],stdout=stdout,stderr=stderr,start_new_session=group)
                try:
                    r=finish((process,stdout,stderr),group=group)
                    self.assertTrue(r['all_children_exited']);self.assertIsNotNone(process.poll());self.assertTrue(stdout.closed)
                finally:
                    if process.poll() is None:process.kill();process.wait(timeout=2.)

    def test_cleanup_kills_ud_like_descendant_after_es_like_group_leader_exit(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary);stdout=(path/'stdout').open('w');stderr=(path/'stderr').open('w')
            code="import subprocess,sys,pathlib; p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); pathlib.Path(sys.argv[1]).write_text(str(p.pid))"
            process=subprocess.Popen([sys.executable,'-c',code,str(path/'descendant')],stdout=stdout,stderr=stderr,start_new_session=True)
            try:
                process.wait(timeout=5.);descendant=int((path/'descendant').read_text())
                result=finish((process,stdout,stderr),group=True)
                self.assertTrue(result['descendant_group_cleanup_applied'])
                deadline=time.monotonic()+3.
                while Path(f'/proc/{descendant}/stat').exists() and time.monotonic()<deadline:
                    if Path(f'/proc/{descendant}/stat').read_text().split()[2]=='Z':break
                    time.sleep(.01)
                self.assertTrue(not Path(f'/proc/{descendant}/stat').exists() or
                                Path(f'/proc/{descendant}/stat').read_text().split()[2]=='Z')
            finally:
                if process.poll() is None:finish((process,stdout,stderr),group=True)

    def test_new_diagnostics_do_not_expand_original_source_glob_or_thresholds(self):
        identity=source_identity()
        self.assertFalse(any('parity_causal' in name or '/97_' in name for name in identity['files']))
        self.assertEqual(len(identity['files']),15)
        self.assertEqual(THRESHOLDS,dict(atol=1e-3,rtol=1e-3,max_abs=1e-2,relative_l2=1e-3,last_token_kl=1e-5,last_token_argmax_agreement=True))


@unittest.skipUnless(os.environ.get('SEMCACHE_R3A_CAUSAL_GPU')=='1','explicit original-evidence SERAPH GPU diagnostic opt-in required')
class GPUOptIn(unittest.TestCase):
    def test_original_all_layer_causal_diagnostic(self):
        required=['SEMCACHE_R3A_CAUSAL_PREPARED','SEMCACHE_R3A_CAUSAL_PILOT','SEMCACHE_R3A_CAUSAL_FORENSICS','SEMCACHE_R3A_CAUSAL_OUTPUT']
        self.assertTrue(all(os.environ.get(n) for n in required),'all immutable original paths and a fresh output required')
        result=subprocess.run([sys.executable,str(SCRIPT),'--phase','gpu-intervention','--enable-gpu-probes',
            '--prepared',os.environ[required[0]],'--pilot-inputs',os.environ[required[1]],'--forensic-inputs',os.environ[required[2]],
            '--output-dir',os.environ[required[3]]],capture_output=True,text=True,timeout=7500)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        report=read(Path(os.environ[required[3]])/'r3a_cpu_delta_causal.json')
        self.assertEqual(report['gpu_findings_status'],'COMPLETE')
        # UNRESOLVED and reference failures remain valid diagnostic outcomes,
        # never silently promoted to a successful functional pilot.
        self.assertFalse(report['PRIMARY_PERFORMANCE_ELIGIBLE'])


if __name__=='__main__':unittest.main()
