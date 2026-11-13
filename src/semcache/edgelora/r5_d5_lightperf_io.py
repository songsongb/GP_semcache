"""Independent LIGHT receipts and source identity; Strict IO is untouched."""
from pathlib import Path
import hashlib
import json
import os
import subprocess

from .communication import canonical, digest, require, json_loads
from .r3a_artifacts import ROOT

SCHEMA = 'D5_LIGHTPERF_V1'
MODE = 'LIGHTWEIGHT_TIMING'
BASE = '4916836ce328e119f3a8d85c6c1afc7cf8d363e5'
BRANCH = 'exp/d5-l1-l5-lightperf'
NAMESPACE = ROOT / 'results/r5/d5_lightperf'
MAX_JSON = 8 * 2**20
MAX_FILES = 2048
MAX_BYTES = 64 * 2**20
MANIFEST = 'light_inventory.json'


def hidden_cuda():
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU command requires CUDA_VISIBLE_DEVICES empty')


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def no_symlinks(path):
    p = Path(path).absolute()
    require(not any(x.is_symlink() for x in (p, *p.parents)), 'symlink component forbidden: ' + str(path))
    return p.resolve()


def fresh(path, protected=(), *, namespace=NAMESPACE):
    p = no_symlinks(path)
    ns = no_symlinks(namespace)
    require(p.is_relative_to(ns) and p != ns, 'new d5_lightperf namespace required')
    require(not p.exists(), 'fresh output required')
    for raw in protected:
        q = Path(raw).resolve()
        require(not p.is_relative_to(q) and not q.is_relative_to(p), 'output overlaps protected input')
    p.mkdir(parents=True, exist_ok=False)
    return p


class SummaryWriter:
    """Write-once bounded summaries. Incremental accounting; scan only at seal.

    No tensor.save or blob method: passing this object to Strict evidence is an
    error. Native physical owner uses evidence=None while keeping its own proofs.
    """
    def __init__(self, root, *, max_bytes=MAX_BYTES, max_files=MAX_FILES):
        self.root = no_symlinks(root)
        require(self.root.is_dir() and not any(self.root.iterdir()), 'empty owned output required')
        require(type(max_bytes) is int and 0 < max_bytes <= MAX_BYTES and
                type(max_files) is int and 0 < max_files <= MAX_FILES, 'invalid light output bounds')
        self.max_bytes, self.max_files = max_bytes, max_files
        self.bytes, self.files, self.hashes, self.sealed = 0, 0, {}, False

    def write(self, name, value):
        self._write(name, canonical(value), {'.json'})

    def write_text(self, name, value):
        require(isinstance(value, str), 'text export required')
        self._write(name, value.encode(), {'.csv', '.md'})

    def _write(self, name, data, suffixes):
        require(not self.sealed and isinstance(name, str) and Path(name).name == name and
                Path(name).suffix in suffixes and name != MANIFEST, 'unsafe/closed light write')
        require(name not in self.hashes, 'receipt overwrite forbidden')
        require(len(data) <= MAX_JSON and self.bytes + len(data) <= self.max_bytes and
                self.files + 1 < self.max_files, 'light summary budget exceeded')
        path = no_symlinks(self.root / name)
        with path.open('xb') as f:
            require(f.write(data) == len(data), 'short light write')
        self.hashes[name] = hashlib.sha256(data).hexdigest()
        self.bytes += len(data)
        self.files += 1

    def seal(self):
        require(not self.sealed and {p.name for p in self.root.iterdir()} == set(self.hashes),
                'unexpected output or repeated sealing')
        require(all(not p.is_symlink() and file_hash(p) == self.hashes[p.name]
                    for p in self.root.iterdir()), 'owned output changed before seal')
        value = dict(schema=SCHEMA, mode=MODE, files=self.hashes, data_bytes=self.bytes)
        data = canonical(value)
        require(len(data) <= MAX_JSON and self.bytes + len(data) <= self.max_bytes, 'seal budget exceeded')
        with (self.root / MANIFEST).open('xb') as f:
            f.write(data)
        self.sealed = True
        return verify(self.root)


def read_json(path):
    path = no_symlinks(path)
    require(path.is_file() and path.stat().st_size <= MAX_JSON, 'missing/oversized light JSON')
    return json_loads(path.read_bytes())


def verify(root):
    root = no_symlinks(root)
    m = read_json(root / MANIFEST)
    require(m['schema'] == SCHEMA and m['mode'] == MODE and
            type(m['files']) is dict and len(m['files']) + 1 <= MAX_FILES, 'not a bounded LIGHT receipt')
    require({p.name for p in root.iterdir()} == set(m['files']) | {MANIFEST}, 'receipt inventory differs')
    total = 0
    for name, expected in m['files'].items():
        require(Path(name).name == name and Path(name).suffix in {'.json', '.csv', '.md'} and name != MANIFEST,
                'invalid light inventory path')
        p = no_symlinks(root / name)
        require(p.is_file() and p.stat().st_size <= MAX_JSON, 'missing/oversized light export')
        if p.suffix == '.json':
            read_json(p)
        total += p.stat().st_size
        require(total <= MAX_BYTES and file_hash(p) == expected, 'changed light receipt')
    require(total == m['data_bytes'] and total + (root / MANIFEST).stat().st_size <= MAX_BYTES,
            'receipt byte accounting differs')
    return dict(manifest_sha256=file_hash(root / MANIFEST), files=len(m['files']) + 1,
                bytes=total + (root / MANIFEST).stat().st_size)


def git(root, *args):
    return subprocess.check_output(['git', *args], cwd=root, text=True).strip()


def source_identity(*, development=False, reviewed_commit=None, reviewed_fingerprint=None,
                    expected_root=ROOT, expected_branch=BRANCH):
    """Development receipts never authorize GPU execution or clean CPU approval."""
    root = no_symlinks(expected_root)
    require(Path(git(root, 'rev-parse', '--show-toplevel')).resolve() == root, 'wrong worktree root')
    branch, head = git(root, 'branch', '--show-current'), git(root, 'rev-parse', 'HEAD')
    require(branch == expected_branch, 'wrong Light branch')
    require(subprocess.run(['git', 'merge-base', '--is-ancestor', BASE, head], cwd=root,
                           capture_output=True).returncode == 0, 'missing Strict baseline fixes')
    dirty = git(root, 'status', '--porcelain', '--untracked-files=no')
    tracked = set(git(root, 'ls-files').splitlines())
    # Include all potentially executable local code/configuration, not results.
    paths = sorted(p for folder in ('src', 'scripts', 'jobs', 'tests', 'configs', 'docs')
                   for p in (root / folder).rglob('*')
                   if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc')
    hashes = {str(p.relative_to(root)): file_hash(no_symlinks(p)) for p in paths}
    for name in ('pyproject.toml', 'requirements.txt'):
        if (root / name).exists():
            hashes[name] = file_hash(no_symlinks(root / name))
    fingerprint = digest(hashes)
    if not development:
        require(not dirty and set(hashes) <= tracked, 'clean committed Light source required')
        require(reviewed_commit == head and reviewed_fingerprint == fingerprint,
                'fresh reviewed HEAD/content fingerprint required')
        for name, sha in hashes.items():
            blob = subprocess.check_output(['git', 'show', head + ':' + name], cwd=root)
            require(hashlib.sha256(blob).hexdigest() == sha, 'committed source bytes differ')
    return dict(root=str(root), branch=branch, commit=head, strict_baseline=BASE,
                tracked_dirty=bool(dirty), files=hashes, content_fingerprint=fingerprint,
                approval_scope='DEVELOPMENT_NOT_GPU_APPROVED' if development else 'CLEAN_REVIEWED_SOURCE')
