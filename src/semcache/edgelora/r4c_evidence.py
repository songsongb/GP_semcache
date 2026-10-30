"""Bounded C evidence, immutable output manifests, and fail-closed IO."""
from pathlib import Path

from .communication import digest_bytes, require
from .r3a_artifacts import tensor_bytes, write
from .r3b_cache import spec

PROFILES = ('STRICT_AUDIT', 'SCALABLE_EVALUATION')


def fresh(path, protected=()):
    path = Path(path).resolve()
    from .r3a_artifacts import ROOT
    require(not path.is_relative_to(ROOT) or path.is_relative_to(ROOT/'results/r4/c'), 'repository outputs must be separate under results/r4/c')
    require(all(not path.is_relative_to(Path(p).resolve()) for p in protected if p is not None), 'output would modify input directory')
    require(not path.exists() or path.is_dir() and not any(path.iterdir()), 'refusing nonempty output directory')
    path.mkdir(parents=True, exist_ok=True); return path


def directory_bytes(path):
    return sum(p.stat().st_size for p in Path(path).rglob('*') if p.is_file())


class Evidence:
    def __init__(self, output, *, budget_bytes=2**30):
        self.output=Path(output); self.budget_bytes=budget_bytes; self.profile='STRICT_AUDIT'; self.bytes=0
        self.checked_size=None;self.checked_blob_bytes=0
        require(type(budget_bytes) is int and 0 < budget_bytes <= 16*2**30, 'evidence budget must be in (0,16 GiB]')

    def check(self, additional=0):
        actual=directory_bytes(self.output.parent if self.output.name=='es' else self.output)
        require(actual+additional <= self.budget_bytes, f'EVIDENCE_BUDGET_EXCEEDED: used={actual}, next={additional}, budget={self.budget_bytes}')
        self.checked_size=actual;self.checked_blob_bytes=self.bytes
        return actual

    def blob(self, blob, name):
        path=(self.output/name).resolve(); require(path.is_relative_to(self.output.resolve()) and not path.exists(), 'evidence path collision/escape')
        if self.checked_size is None:self.check()
        # All tensor/logit writes are owned here and cannot overwrite files.
        # At each forward checkpoint reserve 16 MiB for both endpoint ledgers
        # (bounded 64 KiB headers, <=32 layer exchanges), controls and JSON.
        # Actual directory usage is rescanned at every forward and request end.
        accounted=self.checked_size+self.bytes-self.checked_blob_bytes
        require(accounted+len(blob)+min(16*2**20,self.budget_bytes//4)<=self.budget_bytes,'EVIDENCE_BUDGET_EXCEEDED: reserved write accounting')
        path.parent.mkdir(parents=True, exist_ok=True);path.write_bytes(blob);require(path.stat().st_size==len(blob),'short evidence write');self.bytes+=len(blob)
        return str(path.relative_to(self.output))

    def save(self, tensor, name, *, force=False):
        result=spec(tensor)
        if self.profile == 'STRICT_AUDIT' or force:
            n=tensor.numel()*tensor.element_size()
            blob=tensor.fixture_bytes() if hasattr(tensor,'fixture_bytes') else tensor_bytes(tensor)
            require(len(blob)==n and digest_bytes(blob)==result['sha256'], 'tensor evidence serialization differs')
            result['file']=self.blob(blob, 'tensors/'+name+'.bin')
        return result


def seal_directory(path, name='r4c_output_manifest.json'):
    path=Path(path); hashes={str(p.relative_to(path)):digest_bytes(p.read_bytes()) for p in sorted(path.rglob('*')) if p.is_file() and p!=path/name}
    write(path/name, dict(files=hashes)); return hashes


def verify_directory(path):
    import json
    path=Path(path); recorded=json.loads((path/'r4c_output_manifest.json').read_text())['files']
    actual={str(p.relative_to(path)):digest_bytes(p.read_bytes()) for p in sorted(path.rglob('*')) if p.is_file() and p!=path/'r4c_output_manifest.json'}
    require(actual==recorded, 'immutable output manifest differs: changed/missing/extra evidence')
    return actual
