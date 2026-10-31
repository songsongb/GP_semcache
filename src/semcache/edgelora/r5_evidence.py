"""R5 immutable evidence and implementation identity (no heavy imports)."""
from pathlib import Path
import time
from .communication import digest_bytes, require
from .r3a_artifacts import ROOT, read, write
from .r4c_evidence import Evidence, directory_bytes


class TimedEvidence(Evidence):
    """Identifiable write overhead; hashing/live validation remain instrumented."""
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.write_ms=0.

    def blob(self,blob,name):
        started=time.monotonic_ns()
        try:return super().blob(blob,name)
        finally:self.write_ms+=(time.monotonic_ns()-started)/1e6


def sources():
    paths=[*sorted((ROOT/'src').rglob('*.py')),ROOT/'scripts/110_run_r5_semcache_e2e.py',
        ROOT/'scripts/111_analyze_r5_semcache_e2e.py',ROOT/'jobs/r5_opt_semcache_e2e.sh',ROOT/'configs/r5_integration.json']
    return {str(p.relative_to(ROOT)):digest_bytes(p.read_bytes()) for p in paths if p.is_file()}


def fresh(path,protected=()):
    p=Path(path).resolve()
    require(not p.is_relative_to(ROOT) or p.is_relative_to(ROOT/'results/r5'),'R5 repository outputs belong under results/r5')
    require(all(not p.is_relative_to(Path(x).resolve()) for x in protected if x),'output overlaps immutable input')
    require(not p.exists() or p.is_dir() and not any(p.iterdir()),'refusing nonempty output directory')
    p.mkdir(parents=True,exist_ok=True);return p


def seal(path):
    p=Path(path);name='r5_output_manifest.json'
    write(p/name,dict(schema='R5_OUTPUT_MANIFEST_V1',files={str(x.relative_to(p)):digest_bytes(x.read_bytes())
        for x in sorted(p.rglob('*')) if x.is_file() and x!=p/name}))


def verify(path):
    p=Path(path);m=read(p/'r5_output_manifest.json')
    actual={str(x.relative_to(p)):digest_bytes(x.read_bytes()) for x in sorted(p.rglob('*')) if x.is_file() and x!=p/'r5_output_manifest.json'}
    require(m['files']==actual,'immutable R5 manifest differs')
