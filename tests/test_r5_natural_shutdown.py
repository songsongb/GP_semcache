"""TEST_FIXTURE_ONLY CPU timeout/last-use shutdown tests; no GPU run evidence."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json,os,socket,subprocess,tempfile,threading,unittest
from unittest.mock import Mock,patch
from semcache.edgelora.communication import Channel,ProtocolError,PREFIX,MAGIC,canonical,encode,header,digest,connect,listener
from semcache.edgelora.r5_subsequence_natural_runtime import ledger,stop_adapter,join_adapter,adapter_completion_callback
from semcache.edgelora.r5_subsequence_generation import complete_request,generate
from semcache.edgelora.r5_sparse_qkv import SparseSession,serve_sparse
from semcache.edgelora.r3b_fixture import ArrayTensor
import test_r5_sparse_physical as fixture_module


class FixtureSocket:
    family=socket.AF_INET
    def __init__(self,data=b'',failure=None):
        self.data=data;self.failure=failure;self.closed=False;self.recv_calls=0;self.sent=[];self.timeouts=[]
    def getpeername(self):return ('127.0.0.1',12345)
    def settimeout(self,value):self.timeouts.append(value)
    def recv(self,size):
        self.recv_calls+=1
        if self.data:
            part,self.data=self.data[:size],self.data[size:];return part
        if self.failure is not None:raise self.failure
        return b''
    def send(self,data):self.sent.append(bytes(data));return len(data)
    def close(self):self.closed=True


def packet(h,sequence=0):
    return encode(dict(h,session_id='TEST_FIXTURE_ONLY',adapter_sha256='a'*64,sequence=sequence),b'')[0]


def fixture_session(req,ids=None,ready=True):
    control=dict(request=req,generated_token_ids=[3],termination='LENGTH_CAP')
    data=packet(header('READY',req['request_id']+'_COMPLETE',-1,completion_sha256=digest(control))) if ready else b''
    data+=packet(header('BYE','shutdown',-1),sequence=1 if ready else 0)
    sock=FixtureSocket(data);channel=Channel(sock,'TEST_FIXTURE_ONLY','a'*64,timeout=1800)
    return dict(channel=channel,requests=[req],ids=ids or [req['request_id']],process=SimpleNamespace(wait=Mock(),returncode=0)),control


class Logging(unittest.TestCase):
    def test_timeout_before_or_during_header_keeps_original_and_failed_control_record(self):
        for data in (b'',b'R3A',PREFIX.pack(MAGIC,20,0)+b'{'):
            with self.subTest(bytes_received=len(data)),tempfile.TemporaryDirectory() as tmp:
                primary=TimeoutError('TEST_FIXTURE_ONLY timed out');sock=FixtureSocket(data,primary)
                ch=Channel(sock,'TEST_FIXTURE_ONLY','a'*64,timeout=1800,trace=lambda r:ledger(tmp,r,['FAILED','FAILED_FRAME','R']))
                with self.assertRaises(TimeoutError) as raised:ch.recv()
                self.assertIs(raised.exception,primary);self.assertTrue(ch.failed);self.assertTrue(sock.closed)
                record=json.loads((Path(tmp)/'wire/CONTROL/FAILED_FRAME.jsonl').read_text())
                self.assertEqual(record['header'],{});self.assertFalse(record['complete'])
                self.assertEqual(record['error'],'TimeoutError: TEST_FIXTURE_ONLY timed out');self.assertEqual(record['wire_bytes'],len(data))
                calls=sock.recv_calls
                with self.assertRaises(ProtocolError):ch.recv()
                self.assertEqual(sock.recv_calls,calls);self.assertEqual(ch.timeout,1800)

    def test_broken_error_logger_and_close_cannot_mask_original_exception(self):
        primary=TimeoutError('original timeout');sock=FixtureSocket(failure=primary)
        sock.close=Mock(side_effect=OSError('close failed'))
        ch=Channel(sock,'TEST_FIXTURE_ONLY','a'*64,trace=lambda r:r['header']['request_id'])
        with self.assertRaises(TimeoutError) as raised:ch.recv()
        self.assertIs(raised.exception,primary);self.assertTrue(ch.failed);sock.close.assert_called_once()
        self.assertEqual(ch.records[-1]['error'],'TimeoutError: original timeout')
        if hasattr(primary,'__notes__'):
            self.assertTrue(any('KeyError' in n for n in primary.__notes__));self.assertTrue(any('close failed' in n for n in primary.__notes__))

    def test_successful_request_id_and_schema_are_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            sock=FixtureSocket(packet(header('READY','R_s000',-1)))
            ch=Channel(sock,'TEST_FIXTURE_ONLY','a'*64,trace=lambda r:ledger(tmp,r,['R']))
            h,b,record=ch.recv(kind='READY',request_id='R_s000',layer_id=-1)
            self.assertEqual(h['request_id'],'R_s000');self.assertEqual(b,b'');self.assertTrue(record['complete']);self.assertIsNone(record['error'])
            self.assertEqual(json.loads((Path(tmp)/'wire/R/R_s000.jsonl').read_text()),record)
            self.assertNotIn('FAILED_FRAME',str(list(Path(tmp).rglob('*.jsonl'))))

    def test_malformed_headers_remain_invalid_without_fabricating_request_ids(self):
        for h in ({},[],{'kind':'READY'},{'request_id':42}):
            with self.subTest(header=h),tempfile.TemporaryDirectory() as tmp:
                raw=canonical(h);sock=FixtureSocket(PREFIX.pack(MAGIC,len(raw),0)+raw)
                ch=Channel(sock,'TEST_FIXTURE_ONLY','a'*64,trace=lambda r:ledger(tmp,r,[]))
                with self.assertRaises(ProtocolError):ch.recv()
                record=json.loads((Path(tmp)/'wire/CONTROL/FAILED_FRAME.jsonl').read_text())
                self.assertEqual(record['header'],h);self.assertIn('ProtocolError:',record['error'])
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ProtocolError):ledger(tmp,dict(header={},complete=True,error=None),[])
            self.assertFalse(list(Path(tmp).rglob('*.jsonl')))

    def test_send_failure_is_also_fail_closed_when_tracing_fails(self):
        primary=OSError('original send failure');sock=FixtureSocket();sock.send=Mock(side_effect=primary)
        ch=Channel(sock,'TEST_FIXTURE_ONLY','a'*64,trace=Mock(side_effect=RuntimeError('logger failed')))
        with self.assertRaises(OSError) as raised:ch.send(header('STOP','shutdown',-1))
        self.assertIs(raised.exception,primary);self.assertTrue(ch.failed);self.assertTrue(sock.closed)
        self.assertEqual(ch.records[-1]['error'],'OSError: original send failure')


class Shutdown(unittest.TestCase):
    def setUp(self):self.f=fixture_module.Physical();self.f.setUp();self.req=self.f.req('LAST',[1,2,3])

    def test_stop_after_ack_precedes_1972_seconds_of_local_completion_work(self):
        session,control=fixture_session(self.req);clock=[0];events=[]
        class Owner:
            def complete(_,**kwargs):
                self.assertTrue(session['stopped']);self.assertTrue(session['channel'].sock.closed)
                self.assertEqual([r['header']['kind'] for r in session['channel'].records],['HELLO','READY','STOP','BYE'])
                events.append(('admission_start',clock[0]));clock[0]+=1972.123
                return {'TEST_FIXTURE_ONLY':True,'completed':kwargs['completed']}
        result=complete_request(session['channel'],self.req,[3],owner=Owner(),attentions='TEST_FIXTURE_ONLY',
            on_tcp_complete=adapter_completion_callback(session,self.req))
        self.assertEqual(result[0],control);self.assertEqual(events,[('admission_start',0)])
        self.assertEqual(clock[0],1972.123);join_adapter(session);session['process'].wait.assert_called_once_with(timeout=60)

    def test_nonfinal_request_retains_connection_and_legacy_default_completion_order(self):
        session,control=fixture_session(self.req,ids=['LAST','FUTURE']);self.assertIsNone(adapter_completion_callback(session,self.req))
        owner=Mock();owner.complete.return_value={'TEST_FIXTURE_ONLY':True}
        self.assertEqual(complete_request(session['channel'],self.req,[3],owner=owner), (control,owner.complete.return_value))
        self.assertEqual([r['header']['kind'] for r in session['channel'].records],['HELLO','READY'])
        self.assertFalse(session['channel'].sock.closed);owner.complete.assert_called_once_with(completed=True,attentions=None)

    def test_premature_duplicate_or_invalid_stop_is_rejected(self):
        session,control=fixture_session(self.req,ids=['LAST','FUTURE'])
        with self.assertRaises(ProtocolError):stop_adapter(session,control)
        self.assertEqual(session['channel'].sent,0)
        session,control=fixture_session(self.req,ready=False);stop_adapter(session,control)
        with self.assertRaises(ProtocolError):stop_adapter(session,control)
        session,control=fixture_session(self.req,ready=False)
        session['channel'].sock.data=packet(header('BYE','shutdown',-1,unexpected=True))
        with self.assertRaisesRegex(ProtocolError,'shutdown acknowledgement'):stop_adapter(session,control)
        self.assertFalse(session.get('stopped',False))

    def test_ack_failure_or_shutdown_failure_cannot_admit_or_report_success(self):
        for phase in ('completion','shutdown'):
            session,_=fixture_session(self.req);owner=Mock();primary=TimeoutError('TEST_FIXTURE_ONLY '+phase)
            if phase=='completion':session['channel'].sock.data=b''
            else:
                control=dict(request=self.req,generated_token_ids=[3],termination='LENGTH_CAP')
                session['channel'].sock.data=packet(header('READY','LAST_COMPLETE',-1,completion_sha256=digest(control)))
            session['channel'].sock.failure=primary
            with self.assertRaises(TimeoutError) as raised:
                complete_request(session['channel'],self.req,[3],owner=owner,on_tcp_complete=adapter_completion_callback(session,self.req))
            self.assertIs(raised.exception,primary);owner.complete.assert_not_called()
            with self.assertRaises(ProtocolError):join_adapter(session)

    def test_ud_exit_and_bounded_wait_errors_reach_runner_without_retry(self):
        for rc,failure in ((7,None),(0,subprocess.TimeoutExpired('TEST_FIXTURE_ONLY_UD',60))):
            session,_=fixture_session(self.req);session['stopped']=True;session['process'].returncode=rc
            if failure:session['process'].wait.side_effect=failure
            with self.assertRaises((ProtocolError,subprocess.TimeoutExpired)):join_adapter(session)
            session['process'].wait.assert_called_once_with(timeout=60)

    def test_generation_wrapper_aborts_and_reraises_shutdown_error(self):
        primary=TimeoutError('TEST_FIXTURE_ONLY shutdown failure');owner=Mock()
        with patch('semcache.edgelora.r5_subsequence_generation._generate',side_effect=primary):
            with self.assertRaises(TimeoutError) as raised:
                generate(None,None,None,self.req,None,owner,None,condition='SEMCACHE_SUBSEQUENCE',on_tcp_complete=Mock())
        self.assertIs(raised.exception,primary);owner.abort.assert_called_once()

    def test_condition_runner_stops_before_approx_when_exact_worker_fails(self):
        from semcache.edgelora import r5_natural_approx as runner
        from semcache.edgelora.r5_subsequence_natural_io import output,put,seal,verify
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);calls=[]
            p=dict(implementation={'source':{'commit':'TEST_FIXTURE_ONLY'}},multi=str(root/'multi'),
                original_experiment=str(root/'original'),dataset_paths={},resources=dict(required_free_disk_bytes=0,
                total_run_timeout_seconds=600,worker_timeout_seconds=60,run_limit_bytes=64*2**30))
            def cpu_worker(argv,timeout):
                c=argv[argv.index('--condition')+1];calls.append(c)
                if c==runner.CONDITIONS[1]:return 1  # Worker reports UD/shutdown failure.
                out=output(argv[argv.index('--output-dir')+1]);put(out/'identity.json',dict(gpu={'uuid':'TEST_FIXTURE_ONLY'}));seal(out)
                return 0
            with patch.dict(os.environ,dict(SLURM_JOB_ID='TEST_FIXTURE_ONLY',R5_ENABLE_GPU='1',R5_I5_ENABLE_GPU='1'),clear=True), \
                 patch.object(runner,'approval',return_value=(p,{},[],{})), \
                 patch.object(runner.shutil,'disk_usage',return_value=SimpleNamespace(free=100*2**30)), \
                 patch('semcache.edgelora.r5_subsequence_pilot.wait_worker',side_effect=cpu_worker):
                with self.assertRaisesRegex(ProtocolError,'no retry/fallback'):
                    runner.run(root/'plan',root/'run','TEST_FIXTURE_ONLY',root/'cpu',True,True)
            self.assertEqual(calls,list(runner.CONDITIONS[:2]));verify(root/'run')
            self.assertEqual(json.loads((root/'run/status.json').read_text())['status'],'FAILED')


class RealTCP(unittest.TestCase):
    def test_five_requests_three_adapters_stop_at_last_use_and_keep_receipt_validation(self):
        # Explicit toy scalar fixtures. Source/request ordering is a test input,
        # never a reconstruction of the unavailable MultiNews production plan.
        f=fixture_module.Physical();f.setUp();sessions={};servers={};events=[];errors=[]
        order=[('A','A0'),('C','C0'),('B','B0'),('A','A1'),('B','B1')]
        def server_main(sock,identity,done,records):
            conn=None
            try:
                conn,_=sock.accept();sock.close()
                ch=Channel(conn,'TEST_FIXTURE_ONLY',identity['adapter_matrix_fingerprint'],timeout=5)
                serve_sparse(ch,identity,f.registry,f.matrices,fixture=True,trace=records.append)
            except BaseException as exc:errors.append(exc)
            finally:
                if conn:conn.close()
                done.set()
        try:
            for aid in 'ABC':
                identity=deepcopy(f.ident);identity['adapter_id']='Adapter_'+aid
                sock=listener();done=threading.Event();records=[];port=sock.getsockname()[1]
                thread=threading.Thread(target=server_main,args=(sock,identity,done,records),daemon=True);thread.start()
                ch=Channel(connect(port),'TEST_FIXTURE_ONLY',identity['adapter_matrix_fingerprint'],timeout=5)
                sessions[aid]=dict(channel=ch,ids=[rid for a,rid in order if a==aid],requests=[])
                servers[aid]=(thread,done,records)
            for aid,rid in order:
                req=f.req(rid,[1,2,3]);req['adapter_id']=req['physical_identity']['adapter_id']='Adapter_'+aid
                req['user_id']={'A':'User_01','B':'User_11','C':'User_21'}[aid]
                req.pop('request_sha256');req['request_sha256']=digest(req)
                session=sessions[aid];session['requests'].append(req)
                sparse=SparseSession(session['channel'],req,rid+'_s000',fixture=True)
                x=ArrayTensor([1.]*12,[1,3,4])
                for layer in range(2):
                    for role in 'qkv':self.assertEqual(sparse.correction(layer,role,x,[0,1,2]).shape,(1,3,4))
                sparse.complete()
                callback=adapter_completion_callback(session,req)
                complete_request(session['channel'],req,[3],on_tcp_complete=callback)
                if callback:
                    self.assertTrue(servers[aid][1].wait(2));events.append(aid)
                if rid=='B0':self.assertTrue(servers['C'][1].is_set())  # C is gone while A/B still need work.
            self.assertEqual(events,['C','A','B']);self.assertFalse(errors)
            for aid,(thread,done,records) in servers.items():
                thread.join(2);self.assertFalse(thread.is_alive());self.assertTrue(sessions[aid]['stopped'])
                complete=[r for r in records if r.get('kind')=='COMPLETE_REQUEST']
                self.assertEqual([r['control']['request']['request_id'] for r in complete],sessions[aid]['ids'])
                self.assertEqual(sum(r.get('cpu_projection_rows',0) for r in records),18*len(sessions[aid]['ids']))
        finally:
            for session in sessions.values():session['channel'].close()
            for thread,done,records in servers.values():thread.join(6)


if __name__=='__main__':unittest.main()
