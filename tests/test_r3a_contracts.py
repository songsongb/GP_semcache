"""Pure CPU artifact and numerical gate tests; no Torch import."""
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest

from semcache.edgelora.communication import ProtocolError,digest
from semcache.edgelora.r3a_artifacts import THRESHOLDS,ROOT,check_parity,load_matrices,read,sha,validate_prepared,write
from semcache.edgelora.r3a_tiny import prepare_tiny


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.prepared=prepare_tiny(Path(self.tmp.name)/'prepared')

    def test_adapter_safe_blob_actual_nonzero_shapes_scales_and_identity(self):
        pm,ref,am=validate_prepared(self.prepared)
        matrices=load_matrices(self.prepared,am,torch_backend=False)
        self.assertEqual(len(matrices),32*6)
        self.assertTrue(all(any(m) for m in matrices.values()))
        self.assertTrue(pm['TEST_FIXTURE_ONLY'])
        self.assertEqual(ref['native_projection_outputs'],96)
        self.assertEqual(ref['thresholds'],THRESHOLDS)

    def test_matrix_shape_offset_hash_and_zero_rejected(self):
        _,_,am=validate_prepared(self.prepared)
        for mutation in ('shape','offset','hash','dtype','role'):
            with self.subTest(mutation=mutation):
                m=copy.deepcopy(am)
                if mutation=='shape':m['matrices'][0]['shape']=[1,32]
                if mutation=='offset':m['matrices'][0]['offset']=1
                if mutation=='hash':m['matrices'][0]['sha256']='0'*64
                if mutation=='dtype':m['matrices'][0]['dtype']='pickle'
                if mutation=='role':m['matrices'][0]['role']='qk'
                with self.assertRaises(ProtocolError):load_matrices(self.prepared,m,torch_backend=False)

    def test_artifact_tamper_and_wrong_prepared_identity_rejected(self):
        manifest=self.prepared/'r3a_prepare_manifest.json';m=read(manifest)
        m['identity']['token_ids'][0]=6;write(manifest,m)
        with self.assertRaises(ProtocolError):validate_prepared(self.prepared)
        m['identity']['token_ids'][0]=1;write(manifest,m)
        blob=self.prepared/'r3a_adapter.bin';blob.write_bytes(blob.read_bytes()+b'x')
        with self.assertRaisesRegex(ProtocolError,'SHA'):validate_prepared(self.prepared)

    def test_predeclared_parity_detects_fail_and_nonfinite(self):
        self.assertTrue(check_parity([1.,2.,3.],[1.,2.,3.],[1,1,3])['passed'])
        bad=check_parity([2.,2.,3.],[1.,2.,3.],[1,1,3])
        self.assertFalse(bad['passed']);self.assertEqual(bad['status'],'PARITY_FAIL')
        self.assertEqual(bad['thresholds'],THRESHOLDS)
        with self.assertRaises(ProtocolError):check_parity([float('nan')],[1.],[1,1,1])
        with self.assertRaises(ProtocolError):check_parity([1.],[1.],[1,1,2])

    def test_projection_coordinator_rejects_stale_layer_duplicate_and_mutated_input(self):
        from semcache.edgelora.edge_server import ProjectionSession
        from semcache.edgelora.communication import digest_bytes
        from semcache.edgelora.r3a_artifacts import pack
        class Courier:
            def send(self,h,payload):
                self.payload=payload
                return {'header':h}
            def recv(self,**expected):
                data=pack([.01]*48,'float32')
                return dict(payload_sha256=digest_bytes(data),metadata=dict(hidden_payload_sha256=digest_bytes(self.payload),
                    user_device_cuda_used=False)),data,{}
        identity=dict(layers=32,length=4,hidden_size=4,hidden_dtype='float32',delta_dtype='float32')
        x=[.1]*16
        for failure in ('layer','duplicate','mutation'):
            s=ProjectionSession(Courier(),identity,'req',torch_backend=False)
            s.correction(0,'q',x)
            with self.subTest(failure=failure),self.assertRaises(ProtocolError):
                if failure=='layer':s.correction(1,'k',x)
                if failure=='duplicate':s.correction(0,'q',x)
                if failure=='mutation':s.correction(0,'k',[.2]+x[1:])
        with self.assertRaises(ProtocolError):s.complete()

    def test_cli_exact_launcher_contract_and_rejected_options(self):
        spec=importlib.util.spec_from_file_location('r3a_cli',ROOT/'scripts/95_run_r3a_distributed_no_reuse.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        a=module.parse_args(['--phase','prepare','--model','facebook/opt-2.7b','--dtype','float16','--prompt-lengths','32',
            '--device','cuda:0','--adapter','user_a','--seed','42','--output-dir','prepare'])
        self.assertEqual(a.prompt_lengths,'32')
        module.parse_args(['--phase','pilot','--prepared','prepare','--device','cuda:0','--warmup-runs','1','--measured-runs','3','--output-dir','pilot'])
        module.parse_args(['--phase','latency','--prepared','prepare','--pilot-inputs','pilot','--device','cuda:0',
            '--warmup-runs','3','--measured-runs','10','--output-dir','latency'])
        module.parse_args(['--phase','analyze','--prepared','prepare','--pilot-inputs','pilot','--inputs','latency','--output-dir','aggregate'])
        import contextlib,io
        for args in (['--phase','prepare','--prompt-lengths','64'],['--phase','latency','--prepared','p'],
                     ['--phase','prepare','--device','cpu'],['--phase','analyze','--prepared','p','--pilot-inputs','q']):
            with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):module.parse_args(args+['--output-dir','out'])


if __name__=='__main__':unittest.main()
