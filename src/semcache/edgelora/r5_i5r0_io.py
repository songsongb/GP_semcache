"""Portable, read-only I5R-0 input validation. No historical analyzer/source bypass."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import zipfile

UNKNOWN = 'NOT_OBSERVABLE_FROM_AVAILABLE_EVIDENCE'
ROOT = Path(__file__).resolve().parents[3]
CONDITIONS = ('FULL_RECOMPUTE','SEMANTIC_NO_REUSE','SEMCACHE_INTEGRATED')


def require(ok, message):
    if not ok: raise ValueError(message)


def sha(data): return hashlib.sha256(data).hexdigest()


def canonical(value): return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def safe_name(name):
    p=PurePosixPath(name)
    require(name and not p.is_absolute() and '\\' not in name and ':' not in name and
            all(x not in ('..','.') for x in name.split('/')),'unsafe evidence path: '+name)
    return p


def parse(data):
    def unique(pairs):
        d={}
        for k,v in pairs:
            require(k not in d,'duplicate JSON key: '+k);d[k]=v
        return d
    return json.loads(data,object_pairs_hook=unique,parse_constant=lambda x: (_ for _ in ()).throw(ValueError('nonfinite JSON')))


class Files:
    def __init__(self,path,zip_mode=False):
        self.path=Path(path).resolve();self.archive=None
        require(self.path.exists(),'missing input: '+str(self.path))
        if zip_mode:
            self.archive=zipfile.ZipFile(self.path);self.entries={};total=0
            for item in self.archive.infolist():
                safe_name(item.filename.rstrip('/'))
                require(not stat.S_ISLNK(item.external_attr>>16),'ZIP symlink forbidden')
                require(item.filename not in self.entries,'duplicate ZIP member')
                require(not item.flag_bits&1,'encrypted ZIP unsupported')
                total+=item.file_size
                require(total<=8*2**30 and item.file_size<=2**30,'archive exceeds bounded 8 GiB/1 GiB member budget')
                self.entries[item.filename]=item
            self.names={n for n,i in self.entries.items() if not i.is_dir()}
        else:
            require(self.path.is_dir(),'runs-root must be a directory')
            self.names=set()
            for child in self.path.iterdir():
                if re.fullmatch(r'block_\d+_\d+',child.name):
                    require(child.is_dir() and not child.is_symlink(),'unsafe block directory')
                    require((child/'i5_execution_config.json').is_file(),'incomplete block: missing execution config')
                    for p in child.rglob('*'):
                        require(not p.is_symlink(),'evidence symlink forbidden')
                        if p.is_file():self.names.add(p.relative_to(self.path).as_posix())
    def close(self):
        if self.archive:self.archive.close()
    def read(self,name):
        safe_name(name);require(name in self.names,'missing evidence: '+name)
        return self.archive.read(name) if self.archive else (self.path/name).read_bytes()
    def json(self,name):return parse(self.read(name))
    def jsonl(self,name):return [parse(line) for line in self.read(name).splitlines() if line.strip()]
    def file_hash(self,name):
        h=hashlib.sha256()
        with (self.archive.open(name) if self.archive else (self.path/name).open('rb')) as f:
            for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
        return h.hexdigest()
    def verify(self,root,allow_partial=False):
        manifest=root+'/i5_output_manifest.json'
        if manifest not in self.names:
            require(allow_partial,'missing sealed evidence manifest: '+root)
            return dict(status='UNSEALED_PARTIAL',files={})
        expected=self.json(manifest)['files']
        actual={n[len(root)+1:] for n in self.names if n.startswith(root+'/') and n!=manifest}
        require(set(expected)==actual,'manifest inventory differs: '+root)
        for name,digest in expected.items():
            safe_name(name);require(self.file_hash(root+'/'+name)==digest,'corrupt sealed file: '+root+'/'+name)
        return dict(status='SHA256_INVENTORY_PASS',files=expected)


def load_blocks(files,mode,allow_partial=False,expected_blocks=None):
    pattern=r'(?:^|/)(block_(\d+)_(\d+)'+('_audit' if mode=='AUDIT_ZIP' else '')+r')/i5_execution_config.json$'
    roots=[]
    for n in sorted(files.names):
        match=re.search(pattern,n)
        if match:roots.append((int(match[2]),n.rsplit('/',1)[0]))
    require(roots,'no matching block evidence directories')
    require(len({i for i,_ in roots})==len(roots),'ambiguous/duplicate block indices or retries')
    indices={i for i,_ in roots}
    require(indices==set(range(max(indices)+1)),'missing block index in observed sequence')
    if expected_blocks is not None:require(indices==set(range(expected_blocks)),'missing/unexpected blocks against --expected-blocks')
    blocks=[];ids=set();experiment=None;plan_ids=None
    for index,root in sorted(roots):
        integrity=files.verify(root,allow_partial and mode=='AUDIT_ZIP');config=files.json(root+'/i5_execution_config.json');block=config['block']
        require(block['block_id'] not in ids,'duplicate block ID');ids.add(block['block_id'])
        b=dict(block);bid=b.pop('block_id');require(sha(canonical(b))==bid,'invalid block identity hash')
        if experiment is None:experiment=config['experiment_id']
        require(config['experiment_id']==experiment,'mixed experiment IDs')
        item=dict(mode=mode,index=index,root=root,config=config,integrity=integrity,requests={},registry=None,missing=[])
        if integrity['status']!='SHA256_INVENTORY_PASS':item['missing'].append(root+':UNSEALED_EXPORTED_EVIDENCE')
        if root+'/i5_adapter_registry.json' in files.names:item['registry']=files.json(root+'/i5_adapter_registry.json')['registry']
        if mode=='RAW_RUNS':
            require(files.json(root+'/i5_block_status.json')['status']=='COMPLETE','incomplete/failed block')
            workload=files.json(root+'/i5_workload_manifest.json')
            # Older exporters may contain the full plan rather than a block view.
            if 'blocks' in workload:
                plan_ids={x['block_id'] for x in workload['blocks']}
            rows=[];events=[]
            for c in CONDITIONS:
                require(files.json(root+'/'+c+'/result.json')['status']=='COMPLETE','incomplete condition')
                for x in files.jsonl(root+'/'+c+'/requests.jsonl'):
                    if x['kind']=='ARRIVAL':
                        req=x['request'];key=(c,req['request_id']);require(key not in item['requests'],'duplicate request arrival');item['requests'][key]=req
                rows.extend(files.jsonl(root+'/'+c+'/generation.jsonl'))
                event_path=root+'/'+c+'/events.jsonl'
                if c=='FULL_RECOMPUTE':
                    # A disables both semantic matching and cache events. The
                    # sealed inventory may legitimately omit this empty stream.
                    condition_events=files.jsonl(event_path) if event_path in files.names else []
                    require(not condition_events,'FULL_RECOMPUTE must not contain semantic/cache events')
                else:
                    condition_events=files.jsonl(event_path)
                events.extend(dict(e,condition=c) for e in condition_events)
        else:
            audit=files.json(root+'/i5_independent_audit.json');require(audit['status']=='PASS','exported audit not PASS')
            item['reported_audit']=audit
            rows=files.jsonl(root+'/i5_request_metrics.jsonl');events=files.jsonl(root+'/i5_cache_events.jsonl')
            for name in ('cache_summary','latency_summary','network_summary'):
                path=root+'/i5_'+name+'.json'
                if path not in files.names:
                    require(allow_partial,'missing required summary '+path);item['missing'].append(path)
                else:item[name]=files.json(path)
        gate_path=root+'/i5_decision.json'
        item['reported_decision']=files.json(gate_path) if gate_path in files.names else UNKNOWN
        item.update(rows=rows,events=events)
        blocks.append(item)
    if plan_ids is not None:require(ids==plan_ids,'missing blocks against embedded full experiment plan')
    return blocks


def provenance(blocks):
    result=[];groups={};cache={};availability={}
    for b in blocks:
        c=b['config'];commit=c.get('source_commit');hashes=c.get('sources',{})
        groups.setdefault(str(commit),[]).append(b['index']);checks={};available=False
        if isinstance(commit,str) and re.fullmatch('[0-9a-f]{40}',commit):
            if commit not in availability:availability[commit]=subprocess.run(['git','cat-file','-e',commit+'^{commit}'],cwd=ROOT,capture_output=True).returncode==0
            available=availability[commit]
        if available:
            for path,digest in sorted(hashes.items()):
                safe_name(path)
                key=(commit,path,digest)
                if key not in cache:
                    blob=subprocess.run(['git','show',commit+':'+path],cwd=ROOT,capture_output=True)
                    cache[key]='PASS' if blob.returncode==0 and sha(blob.stdout)==digest else 'HASH_MISMATCH_OR_MISSING_BLOB'
                checks[path]=cache[key]
        result.append(dict(block_index=b['index'],block_id=c['block']['block_id'],recorded_commit=commit,
            recorded_source_inventory=hashes,historical_git_object_available=available,
            blob_checks=checks,verification='ALL_RECORDED_BLOBS_MATCH' if checks and all(x=='PASS' for x in checks.values()) else
            'RECORDED_SOURCE_MISMATCH' if checks else 'HISTORICAL_OBJECT_UNAVAILABLE'))
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    current_files=[Path(__file__),ROOT/'src/semcache/edgelora/r5_i5r0.py',ROOT/'scripts/125_diagnose_r5_i5r0.py']
    return dict(historical_groups=groups,blocks=result,current_diagnostic=dict(git_head=head,
        tracked_status=subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True),
        files={str(p.relative_to(ROOT)):sha(p.read_bytes()) for p in current_files if p.exists()}),
        fresh_independent_GPU_audit_performed=False,interpretation='Exported audit PASS is a historical claim; file integrity and available Git blobs checked here, not GPU tensors or strict current-source audit.')
