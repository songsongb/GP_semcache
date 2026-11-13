"""CPU fixtures only: prepared-source compatibility, D5 transport, node gates."""
from pathlib import Path
import hashlib,os,subprocess,tempfile,unittest
from unittest.mock import Mock,patch
from semcache.edgelora import communication as legacy
from semcache.edgelora import r5_subsequence_natural_runtime as runtime
from semcache.edgelora.r5_d5_channel import Channel
from semcache.edgelora.r3a_artifacts import ROOT,read,write,source_identity,validate_prepared,pack
from semcache.edgelora.r3a_tiny import prepare_tiny
from semcache.edgelora.gpu_identity import validate_gpu,SAME_COMPATIBLE_GPU_CLASS
from test_r5_natural_shutdown import FixtureSocket

PREPARED_COMM_SHA='a4bf2740338be1d821612f3f0037cf68858721d5923e25368326bf5ccbc02007'
INCOMPATIBLE_COMM_SHA='b2865f648d7a4d505d55470d0ac9beb9b965b5694c9d31a2018de79c0392929c'
NODES=('moana-y1','moana-y2','moana-y3','moana-y4','moana-y5','moana-y7')


class PreparedCompatibility(unittest.TestCase):
    def test_original_communication_bytes_and_r3_fingerprint_are_restored(self):
        key='src/semcache/edgelora/communication.py'
        self.assertEqual(hashlib.sha256((ROOT/key).read_bytes()).hexdigest(),PREPARED_COMM_SHA)
        files=source_identity()['files'];self.assertEqual(files[key],PREPARED_COMM_SHA)
        self.assertNotIn('src/semcache/edgelora/r5_d5_channel.py',files)
        self.assertIs(Channel.__bases__[0],legacy.Channel);self.assertIs(runtime.Channel,Channel)
        self.assertFalse(hasattr(legacy.Channel,'_fail'))  # The common class was not monkeypatched.

    def test_real_validate_prepared_accepts_pinned_fixture_without_rewriting_it(self):
        # A new tiny TEST_FIXTURE_ONLY preparation; no production assets accessed.
        with tempfile.TemporaryDirectory() as tmp:
            prepared=prepare_tiny(Path(tmp)/'prepared');manifest=prepared/'r3a_prepare_manifest.json'
            before=manifest.read_bytes();m,reference,adapter=validate_prepared(prepared)
            self.assertTrue(m['TEST_FIXTURE_ONLY']);self.assertEqual(reference['identity'],m['identity'])
            self.assertEqual(m['code']['files']['src/semcache/edgelora/communication.py'],PREPARED_COMM_SHA)
            self.assertEqual(manifest.read_bytes(),before)

    def test_incompatible_hash_is_still_rejected_by_unmodified_validator(self):
        with tempfile.TemporaryDirectory() as tmp:
            prepared=prepare_tiny(Path(tmp)/'prepared');manifest=prepared/'r3a_prepare_manifest.json'
            m=read(manifest);m['code']['files']['src/semcache/edgelora/communication.py']=INCOMPATIBLE_COMM_SHA
            write(manifest,m)  # Deliberate corruption of this disposable fixture only.
            with self.assertRaisesRegex(legacy.ProtocolError,'prepared implementation hash mismatch'):validate_prepared(prepared)

    def test_d5_and_prepared_channel_send_identical_valid_frames_and_records(self):
        payload=pack([1.,2.,3.,4.],'float32');h=legacy.header('HIDDEN','TEST_FIXTURE_ONLY',0,'float32',[1,1,4])
        outputs=[]
        for cls in (legacy.Channel,Channel):
            sock=FixtureSocket();ch=cls(sock,'fixture_session','a'*64,timeout=1800)
            r=ch.send(h,payload)
            outputs.append((b''.join(sock.sent),{k:r[k] for k in ('header','wire_bytes','header_bytes','frame_sha256','complete','error')}))
        self.assertEqual(outputs[0],outputs[1])

    def test_d5_does_not_suppress_success_evidence_export_failure(self):
        error=OSError('TEST_FIXTURE_ONLY disk quota failure');ch=Channel(FixtureSocket(),'fixture_session','a'*64,trace=Mock(side_effect=error))
        with self.assertRaises(OSError) as raised:ch.send(legacy.header('STOP','shutdown',-1))
        self.assertIs(raised.exception,error);self.assertTrue(ch.failed);self.assertTrue(ch.sock.closed)
        self.assertFalse(ch.records[-1]['complete']);self.assertIn('disk quota failure',ch.records[-1]['error'])


class NodeGates(unittest.TestCase):
    def test_exact_six_nodes_and_actual_allocation_match(self):
        self.assertEqual(runtime.D5_NODES,NODES)
        for node in NODES:
            runtime.validate_node(dict(D5N_NODE=node,SLURMD_NODENAME=node))
            runtime.validate_node(dict(D5N_NODE=node,SLURM_JOB_NODELIST=node))
            for actual in ('moana-y6','foreign','moana-y1,moana-y2','moana-y[1-2]','',None):
                with self.assertRaises(legacy.ProtocolError):runtime.validate_node(dict(D5N_NODE=node,SLURMD_NODENAME=actual,SLURM_JOB_NODELIST=node))
        for node in ('moana-y6','moana-y8','moana-y0','MOANA-Y1','',None):
            with self.assertRaises(legacy.ProtocolError):runtime.validate_node(dict(D5N_NODE=node,SLURMD_NODENAME=node))
        with self.assertRaises(legacy.ProtocolError):runtime.validate_node({})

    def test_worker_rejects_wrong_node_before_model_setup_or_output(self):
        env=dict(SLURM_JOB_ID='TEST_FIXTURE_ONLY',R5_ENABLE_GPU='1',R5_I5_ENABLE_GPU='1',
            SLURM_CPUS_PER_TASK='4',SLURM_MEM_PER_NODE='65536',D5N_NODE='moana-y1',SLURMD_NODENAME='moana-y7')
        with tempfile.TemporaryDirectory() as tmp,patch.dict(os.environ,env,clear=True), \
             patch.object(runtime,'approval',return_value=({}, {},[{},None,None,None,None,None],{})), \
             patch.object(runtime.os,'sched_getaffinity',return_value={0,1,2,3}), \
             patch('semcache.edgelora.r4a_runtime.setup',side_effect=AssertionError('GPU setup forbidden')) as setup:
            with self.assertRaisesRegex(legacy.ProtocolError,'node differs'):
                runtime.worker('fixture_plan',Path(tmp)/'out','FULL_RECOMPUTE','fixture_cpu')
            setup.assert_not_called();self.assertFalse((Path(tmp)/'out').exists())

    def test_job_shell_node_fragment_only_and_resource_manual_guards_unchanged(self):
        path=ROOT/'jobs/r5_natural_phase3a.sh';text=path.read_text();lines=text.splitlines()
        case=next(x for x in lines if x.startswith('case "$D5N_NODE"'))
        allocation=next(x for x in lines if x.startswith('[[ "${SLURMD_NODENAME:'))
        # Execute ONLY the extracted two node guards, never a job or Slurm command.
        guard=case+'\n'+allocation+'\n'
        for node in (*NODES,'moana-y6','bad'):
            env=dict(os.environ,D5N_NODE=node,SLURMD_NODENAME=node,SLURM_ARRAY_TASK_ID='')
            p=subprocess.run(['bash','-c',guard],env=env,capture_output=True)
            self.assertEqual(p.returncode,0 if node in NODES else 2)
        for changes in (dict(SLURMD_NODENAME='moana-y2'),dict(SLURM_ARRAY_TASK_ID='0')):
            env=dict(os.environ,D5N_NODE='moana-y1',SLURMD_NODENAME='moana-y1',SLURM_ARRAY_TASK_ID='');env.update(changes)
            self.assertEqual(subprocess.run(['bash','-c',guard],env=env,capture_output=True).returncode,2)
        old=subprocess.check_output(['git','show','HEAD:jobs/r5_natural_phase3a.sh'],cwd=ROOT,text=True)
        self.assertEqual([x for x in lines if x.startswith('#SBATCH')],[x for x in old.splitlines() if x.startswith('#SBATCH')])
        for required in ('#SBATCH --partition=batch_eebme_ugrad','#SBATCH --nodes=1','#SBATCH --gres=gpu:1',
            '#SBATCH --cpus-per-task=4','#SBATCH --mem=64G','#SBATCH --time=04:30:00','${SLURM_JOB_ID:?manual allocation required}',
            '"${R5_ENABLE_GPU:-0}" == 1','"${R5_I5_ENABLE_GPU:-0}" == 1'):
            self.assertIn(required,text)

    def test_allowed_nodes_do_not_override_rtx3090_uuid_or_cuda0_contract(self):
        prepared=dict(uuid='TEST_FIXTURE_ONLY_PREPARED',name='NVIDIA GeForce RTX 3090')
        current=dict(uuid='TEST_FIXTURE_ONLY_EXECUTION',name='NVIDIA GeForce RTX 3090',device='cuda:0')
        for node in NODES:
            runtime.validate_node(dict(D5N_NODE=node,SLURMD_NODENAME=node))
            r=validate_gpu(prepared,current,'cuda:0',SAME_COMPATIBLE_GPU_CLASS)
            self.assertEqual(r['execution_gpu_uuid'],current['uuid']);self.assertFalse(r['gpu_uuid_matched'])
            for changes in (dict(name='NVIDIA A100'),dict(name='RTX 3090 Ti'),dict(uuid='UNAVAILABLE'),dict(device='cuda:1')):
                with self.assertRaises(legacy.ProtocolError):validate_gpu(prepared,dict(current,**changes),'cuda:0',SAME_COMPATIBLE_GPU_CLASS)
            with self.assertRaises(legacy.ProtocolError):validate_gpu(prepared,current,'cuda:1',SAME_COMPATIBLE_GPU_CLASS)


if __name__=='__main__':unittest.main()
