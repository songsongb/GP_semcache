"""Cross-process exact writer accounting with bounded independent reconciliation.

The ledger is SQLite (DELETE journal, bounded page count), inside a reserved
64 MiB accounting namespace. Every ES/UD metadata/blob/pipe write is serialized
with the same BEGIN IMMEDIATE transaction. Unknown writers are detected by full
inventory/hash reconciliation, periodically and before success. This is process
failure safety, not a guarantee against arbitrary external disk writers.
"""
from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import threading
import time

from .communication import canonical,digest_bytes,require
from .r4c_evidence import Evidence
from .r3b_cache import spec,raw

RESERVE=64*2**20


class Writer:
    def __init__(self,root,budget,*,create=False):
        self.root=Path(root).resolve();self.budget=budget;self.hashes={};self.io_ms=0.;self.lock=threading.RLock()
        self.db=self.root/'accounting/ledger.sqlite';self.db.parent.mkdir(parents=True,exist_ok=True)
        require(budget>RESERVE,'evidence budget must exceed 64 MiB ledger reservation')
        if create:
            require(not self.db.exists(),'accounting namespace already exists')
            with self.connection() as c:
                c.executescript('CREATE TABLE meta(total INTEGER, files INTEGER, failed TEXT, budget INTEGER); CREATE TABLE files(path TEXT PRIMARY KEY,size INTEGER,sha TEXT);')
                c.execute('INSERT INTO meta VALUES(0,0,NULL,?)',(budget,))
        self.scans=[]

    def connection(self):
        c=sqlite3.connect(self.db,timeout=120,isolation_level=None)
        c.execute('PRAGMA journal_mode=DELETE');c.execute('PRAGMA synchronous=FULL');c.execute('PRAGMA max_page_count=8192')
        return c

    def relative(self,path):
        p=Path(path).resolve();require(p.is_relative_to(self.root) and not p.is_relative_to(self.root/'accounting'),'writer path escapes data namespace')
        return p,str(p.relative_to(self.root))

    def status(self):
        # Snapshot counters and transient journal metadata under the same
        # cross-process lock; a UD commit must not remove the journal between
        # exists/stat or change data bytes halfway through this observation.
        with self.lock,self.connection() as c:
            c.execute('BEGIN IMMEDIATE')
            total,files,failed,budget=c.execute('SELECT * FROM meta').fetchone()
            accounting_bytes=sum(p.stat().st_size for p in (self.db,self.db.with_name('ledger.sqlite-journal')) if p.exists())
            c.commit()
        require(failed is None,'EVIDENCE_ACCOUNTING_FAILED: '+str(failed))
        require(budget==self.budget and total+RESERVE<=budget,'EVIDENCE_BUDGET_EXCEEDED')
        return dict(data_bytes=total,data_files=files,reserved_accounting_bytes=RESERVE,budget_bytes=budget,
            accounting_file_bytes=accounting_bytes,total_evidence_bytes=total+accounting_bytes)

    def bytes(self,path,data,*,append=False,exclusive=False):
        start=time.monotonic_ns();p,name=self.relative(path);p.parent.mkdir(parents=True,exist_ok=True)
        with self.lock,self.connection() as c:
            c.execute('BEGIN IMMEDIATE')
            try:
                total,files,failed,budget=c.execute('SELECT * FROM meta').fetchone();require(failed is None,'writer already failed')
                old=c.execute('SELECT size,sha FROM files WHERE path=?',(name,)).fetchone()
                require(not exclusive or old is None,'immutable blob collision')
                require((p.exists() and p.stat().st_size==old[0]) if old else not p.exists(),'untracked/missing/changed writer file: '+name)
                old_size=old[0] if old else 0;new_size=(old_size if append else 0)+len(data)
                require(total-old_size+new_size+RESERVE<=budget,'EVIDENCE_BUDGET_EXCEEDED: '+name)
                if append and old:
                    h=self.hashes.get(name)
                    if h is None:
                        b=p.read_bytes();require(digest_bytes(b)==old[1],'existing append digest corrupted');h=hashlib.sha256(b)
                    else:require(h.hexdigest()==old[1],'concurrent writer ownership changed')
                    h=h.copy();h.update(data)
                else:h=hashlib.sha256(data)
                with p.open('ab' if append else 'xb' if exclusive else 'wb') as f:
                    require(f.write(data)==len(data),'short evidence write')
                require(p.stat().st_size==new_size,'writer length mismatch')
                c.execute('INSERT OR REPLACE INTO files VALUES(?,?,?)',(name,new_size,h.hexdigest()))
                c.execute('UPDATE meta SET total=?,files=?',(total-old_size+new_size,files+(old is None)))
                c.commit();self.hashes[name]=h
            except BaseException as exc:
                c.rollback();c.execute('UPDATE meta SET failed=?',(str(exc),));raise
        self.io_ms+=(time.monotonic_ns()-start)/1e6
        return name

    def write(self,path,value):return self.bytes(path,canonical(value)+b'\n')
    def append(self,path,value):return self.bytes(path,canonical(value)+b'\n',append=True)

    def reconcile(self):
        start=time.monotonic_ns()
        with self.lock,self.connection() as c:
            c.execute('BEGIN IMMEDIATE')
            try:
                expected={n:(s,h) for n,s,h in c.execute('SELECT * FROM files')};actual={}
                for p in self.root.rglob('*'):
                    require(not p.is_symlink(),'symlink in evidence inventory')
                    if p.is_file() and not p.is_relative_to(self.root/'accounting'):
                        b=p.read_bytes();actual[str(p.relative_to(self.root))]=(len(b),digest_bytes(b))
                require(actual==expected,'EVIDENCE_RECONCILIATION_MISMATCH: missing, extra, corrupted or changed data')
                managed=list((self.root/'accounting').iterdir())
                require(all(p.name in ('ledger.sqlite','ledger.sqlite-journal') for p in managed),'unexpected accounting file')
                require(sum(p.stat().st_size for p in managed if p.is_file())<=RESERVE,'accounting reservation exceeded')
                total=sum(s for s,h in actual.values());meta=c.execute('SELECT * FROM meta').fetchone()
                require(meta[0]==total and meta[1]==len(actual) and meta[2] is None and total+RESERVE<=meta[3],'accounting total/overflow/corruption')
                c.commit()
            except BaseException as exc:
                c.rollback();c.execute('UPDATE meta SET failed=?',(str(exc),));raise
        result=dict(elapsed_ms=(time.monotonic_ns()-start)/1e6,files=len(actual),bytes=total,monotonic_ns=time.monotonic_ns())
        self.scans.append(result);return result


@contextmanager
def accounted_modules(writer):
    """Opt-in process-local IO dependency injection; frozen source is unchanged."""
    from . import r3a_artifacts as artifacts
    original_write,original_append=artifacts.write,artifacts.append;changed=[]
    for module in list(sys.modules.values()):
        if module is None or not getattr(module,'__name__','').startswith('semcache.edgelora.'):continue
        for name,old,new in (('write',original_write,writer.write),('append',original_append,writer.append)):
            if getattr(module,name,None) is old:changed.append((module,name,old));setattr(module,name,new)
    try:yield
    finally:
        for module,name,old in reversed(changed):setattr(module,name,old)


class PerformanceEvidence(Evidence):
    def __init__(self,output,writer,*,profile='SCALABLE_PERFORMANCE',reconcile_every=16,legacy_scan=False,perf=None):
        super().__init__(output,budget_bytes=writer.budget);self.writer=writer;self.profile=profile
        self.every=reconcile_every;require(self.every>=1,'positive reconciliation interval required')
        self.checks=0;self.legacy_scan=legacy_scan;self.perf=perf;self.write_ms=0.;self.legacy_scans=[]

    def check(self,additional=0):
        start=time.monotonic_ns();self.checks+=1
        try:
            if self.legacy_scan:
                from .r4c_evidence import directory_bytes
                tick=time.monotonic_ns();n=directory_bytes(self.writer.root)
                self.legacy_scans.append(dict(elapsed_ms=(time.monotonic_ns()-tick)/1e6,bytes=n,files=self.writer.status()['data_files']))
            state=self.writer.status();require(state['data_bytes']+RESERVE+additional<=self.writer.budget,'EVIDENCE_BUDGET_EXCEEDED: forward headroom')
            if self.checks%self.every==0:self.writer.reconcile()
            return state['data_bytes']
        finally:
            if self.perf is not None:self.perf.events.append(dict(name='evidence_budget_check_nested',start_ns=start,end_ns=time.monotonic_ns(),
                request_id=self.perf.request_id,request_index=self.perf.request_index,step=self.perf.step))

    def blob(self,blob,name):
        start=time.monotonic_ns();self.writer.bytes(self.output/name,blob,exclusive=True);self.bytes+=len(blob)
        self.write_ms+=(time.monotonic_ns()-start)/1e6
        if self.perf is not None:self.perf.events.append(dict(name='tensor_logit_file_writing_nested',start_ns=start,end_ns=time.monotonic_ns(),
            request_id=self.perf.request_id,request_index=self.perf.request_index,step=self.perf.step))
        return name

    def save(self,tensor,name,*,force=False):
        start=time.monotonic_ns()
        try:
            result=spec(tensor)
            if self.profile=='STRICT_AUDIT' or force:result['file']=self.blob(raw(tensor),'tensors/'+name+'.bin')
            return result
        finally:
            if self.perf is not None:self.perf.events.append(dict(name='forward_evidence_capture_nested',start_ns=start,end_ns=time.monotonic_ns(),
                request_id=self.perf.request_id,request_index=self.perf.request_index,step=self.perf.step))

    def diagnostics(self):
        return dict(**self.writer.status(),budget_checks=self.checks,recursive_scans=len(self.writer.scans)+len(self.legacy_scans),
            cumulative_scan_ms=sum(s['elapsed_ms'] for s in self.writer.scans+self.legacy_scans),
            blob_write_ms=self.write_ms,writer_io_ms=self.writer.io_ms)


def spawn(command,writer,stdout,stderr,*,env=None):
    """Bounded, ledger-accounted pipe drains for both independent processes."""
    from .r3a_artifacts import ROOT
    process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True)
    errors=[]
    def drain(pipe,name):
        try:
            writer.bytes(name,b'',exclusive=True)
            while True:
                b=os.read(pipe.fileno(),8192)
                if not b:break
                writer.bytes(name,b,append=True)
        except BaseException as exc:errors.append(str(exc));process.terminate()
        finally:pipe.close()
    threads=[threading.Thread(target=drain,args=(pipe,path),daemon=True) for pipe,path in ((process.stdout,stdout),(process.stderr,stderr))]
    for t in threads:t.start()
    return process,threads,errors


def finish(child,*,timeout=7500):
    import signal
    p,threads,errors=child;forced=False
    try:code=p.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        forced=True;os.killpg(p.pid,signal.SIGTERM)
        try:p.wait(timeout=5)
        except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait(timeout=5)
        code=p.returncode
    except BaseException:
        os.killpg(p.pid,signal.SIGTERM)
        try:p.wait(timeout=5)
        except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait(timeout=5)
        for t in threads:t.join(timeout=10)
        raise
    for t in threads:t.join(timeout=10)
    require(not any(t.is_alive() for t in threads),'evidence pipe did not drain')
    return dict(pid=p.pid,returncode=code,forced_termination=forced,all_children_exited=True,pipe_errors=errors)
