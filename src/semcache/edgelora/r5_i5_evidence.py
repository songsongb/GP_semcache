"""I5 immutable block inventory and source provenance, separate from frozen evidence."""
from pathlib import Path
import json
from .communication import digest, require
from .r3a_artifacts import ROOT, sha, write


def fresh(path, protected=()):
    path = Path(path).resolve()
    allowed=(ROOT/'results/r5/i5',ROOT/'results/r5/i5r/controlled_hit')
    require(not path.is_relative_to(ROOT) or any(path.is_relative_to(p) for p in allowed),
            'I5 outputs belong under results/r5/i5 or results/r5/i5r/controlled_hit')
    require(all(not path.is_relative_to(Path(p).resolve()) and not Path(p).resolve().is_relative_to(path)
                for p in protected if p), 'output overlaps input')
    require(not path.exists(), 'refusing existing output directory; restart into a fresh attempt')
    path.mkdir(parents=True, mode=0o700); return path


def inventory(path):
    path = Path(path)
    require(not any(p.is_symlink() for p in path.rglob('*')), 'symlinks forbidden in evidence')
    return {str(p.relative_to(path)): sha(p) for p in sorted(path.rglob('*'))
            if p.is_file() and p != path/'i5_output_manifest.json'}


def seal(path):
    write(Path(path)/'i5_output_manifest.json', dict(schema='I5_OUTPUT_V1', files=inventory(path)))


def verify(path):
    require(read_json(Path(path)/'i5_output_manifest.json')['files'] == inventory(path), 'evidence inventory mismatch')


def read_json(path):
    return json.loads(Path(path).read_text())


def jsonl(path):
    p = Path(path)
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def sources():
    paths = [*sorted((ROOT/'src').rglob('*.py')), *sorted((ROOT/'scripts').glob('*r5_i5*.py')),
             *sorted((ROOT/'jobs').glob('r5_i5*.sh')), *sorted((ROOT/'configs').glob('r5_i5*.json'))]
    return {str(p.relative_to(ROOT)): sha(p) for p in paths}


def verify_sources(expected):
    require(sources() == expected, 'I5 execution source hashes changed')
