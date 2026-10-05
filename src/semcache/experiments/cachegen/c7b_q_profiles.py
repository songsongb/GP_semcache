"""C7-B1: four Q-only profiles, fit-only CDFs, CPU held-out calibration.

The inspected B2 modulus-255 representation is projection-role generic.
B2's four-stream K/V frame is not reused: Q has its own two-stream namespace.
Existing CPU arithmetic primitives are loaded read-only from the C6 export.
"""
from __future__ import annotations
import argparse
from collections import Counter
from contextlib import contextmanager
import csv
import hashlib
import importlib
import inspect
import json
import math
from pathlib import Path
import struct
import sys

from .c7b_q_capture import (require,sha,digest,read,write,git,verify_files,validate_cohort,
    validate_q_block,seraph_paths,MODEL,REVISION,VERSION,WEIGHTS,Q_OBJECT,DEFAULT_ROOT,PROFILE_SHA)

CANDIDATES=(16,20,24,32)
TRANSFORM='B2_ANCHOR_MOD_RESIDUAL_Q'
PROFILE_HEADER=struct.Struct('<8sI')
FRAME_HEADER=struct.Struct('<8sBHHI32sII')


def load_backend(storage_src):
    import semcache.experiments.cachegen as package
    if storage_src:
        external=Path(storage_src).resolve()/'semcache/experiments/cachegen'
        require(external.is_dir(), 'Existing C6 storage src export required: '+str(external))
        if str(external) not in package.__path__:package.__path__.append(str(external))
    try:
        fmt=importlib.import_module('semcache.experiments.cachegen.b2.format')
        core=importlib.import_module('semcache.experiments.cachegen.shared.core')
        policy=importlib.import_module('semcache.experiments.cachegen.c15c.policy')
    except ImportError as exc:
        raise RuntimeError('C7-B1 requires existing CPU B2 transform/arithmetic primitives via --storage-src; no substitute backend') from exc
    # Check the actual loaded single-projection transform against the inspected
    # algebra. K/V pairing belongs only to framing, not this mathematical map.
    shape=(32,3,4);domain=bytes((19*i+63)%255 for i in range(math.prod(shape)))
    expected=bytearray(domain)
    for layer in range(32):
        for token in (1,2):
            for channel in range(4):
                anchor=domain[(layer*3)*4+channel];index=(layer*3+token)*4+channel
                expected[index]=(domain[index]-anchor+127)%255
    representation=fmt._representation(domain,shape,residual=True)
    require(representation==bytes(expected) and fmt._representation(representation,shape,residual=True,inverse=True)==domain,
            'Loaded B2 transform no longer matches reviewed role-generic modulus255 algebra')
    anchor,residual=fmt._roles(representation,shape)
    require(anchor==b''.join(representation[l*12:l*12+4] for l in range(32))
        and residual==b''.join(representation[l*12+4:l*12+12] for l in range(32)), 'B2 single-projection stream contract changed')
    for name in ('cdf_from_counts','validate_cdf','arithmetic_encode_fast','arithmetic_decode_fast'):
        require(callable(getattr(core,name,None)), 'CPU arithmetic backend unavailable: '+name)
    hashes={inspect.getfile(m):sha(inspect.getfile(m)) for m in (fmt,core,policy)}
    return dict(fmt=fmt,core=core,provenance=dict(transform=TRANSFORM,
        transform_provenance='EXISTING_B2_ROLE_GENERIC_MATHEMATICS; Q-only two-stream frame',
        role_generic_verified=True,kv_framing_reused=False,backend='FAST_PY_BITEXACT',source_hashes=hashes,
        quantization='C1.5C released shifted-rounding operation order; Q-only uniform bins; zero-max rows use centered codes and zero reconstruction'))


def quantize(q,bins):
    import torch
    require(bins in CANDIDATES, 'Only Q16/Q20/Q24/Q32 bin counts allowed')
    validate_q_block(q,hidden=q.shape[-1])
    limit=bins//2-1
    maximum=q.abs().amax(dim=-1,keepdim=True)
    # New Q-only zero handling. Frozen K/V code/profile stays untouched.
    denominator=torch.where(maximum==0,torch.ones_like(maximum),maximum)
    limits=torch.full((32,1,1),limit,dtype=torch.float32)
    symbols=torch.round(q*(limits/denominator)+limits).to(torch.int16)
    require(((symbols>=0)&(symbols<=2*limit)).all().item(), 'Q symbols outside quantizer support')
    domain=bytes((symbols-limit+127).to(torch.uint8).flatten().tolist())
    scales=struct.pack('<96f',*maximum.float().flatten().tolist())
    return domain,scales,tuple(q.shape)


def streams(domain,shape,backend):
    return backend['fmt']._roles(backend['fmt']._representation(domain,shape,residual=True),shape)


def inverse_streams(parts,shape,backend):
    layers,tokens,hidden=shape
    require(len(parts)==2 and tuple(map(len,parts))==(layers*hidden,layers*(tokens-1)*hidden), 'Q stream shape mismatch')
    a,r=parts
    packed=b''.join(a[l*hidden:(l+1)*hidden]+r[l*(tokens-1)*hidden:(l+1)*(tokens-1)*hidden] for l in range(layers))
    return backend['fmt']._representation(packed,shape,residual=True,inverse=True)


class QProfile:
    def __init__(self,metadata,cdfs,backend):
        require(metadata['bins'] in CANDIDATES and metadata['transform']==TRANSFORM and metadata['layers']==32, 'Invalid Q profile metadata')
        require(len(cdfs)==2, 'Q profile needs anchor and residual CDFs')
        for cdf in cdfs:backend['core'].validate_cdf(cdf)
        self.metadata=metadata;self.cdfs=tuple(tuple(c) for c in cdfs)
    def to_bytes(self):
        metadata=json.dumps(self.metadata,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
        return PROFILE_HEADER.pack(b'SCQPRO01',len(metadata))+metadata+b''.join(struct.pack('<256I',*cdf) for cdf in self.cdfs)
    @classmethod
    def from_bytes(cls,data,backend):
        require(len(data)>=PROFILE_HEADER.size, 'Truncated Q profile')
        magic,length=PROFILE_HEADER.unpack_from(data)
        require(magic==b'SCQPRO01' and len(data)==PROFILE_HEADER.size+length+2048, 'Invalid Q profile frame')
        meta=json.loads(data[PROFILE_HEADER.size:PROFILE_HEADER.size+length])
        cdfs=tuple(struct.unpack_from('<256I',data,PROFILE_HEADER.size+length+i*1024) for i in range(2))
        return cls(meta,cdfs,backend)


def fit_profile(blocks,loader,bins,backend,fit_cohort_sha):
    require(len(blocks)==96 and all(b['split']=='profile_fit' for b in blocks), 'Q profile fitting accepts only96 profile_fit blocks')
    require(len({b['calibration_id'] for b in blocks})==96, 'Duplicate fit blocks')
    histograms=[[0]*255 for _ in range(2)]
    shape=None
    for block in blocks:
        domain,_,current=quantize(loader(block['calibration_id']),bins)
        require(shape is None or shape==current, 'Fit shapes differ');shape=current
        for target,stream in zip(histograms,streams(domain,shape,backend)):
            for symbol,count in Counter(stream).items():target[symbol]+=count
    cdfs=tuple(backend['core'].cdf_from_counts(h) for h in histograms)
    return QProfile(dict(bins=bins,layers=32,tokens=3,hidden=shape[-1],transform=TRANSFORM,
        fit_cohort_sha256=fit_cohort_sha,fit_block_ids=[b['calibration_id'] for b in blocks],
        backend_provenance=backend['provenance'],profile_fit_call_count=1,cdf_fit_call_count=2),cdfs,backend)


@contextmanager
def no_runtime_fitting(counter):
    """Detect and abort actual Python/C fitting calls during selection coding."""
    previous=sys.getprofile()
    def guard(frame,event,arg):
        name=(getattr(arg,'__name__','') if event=='c_call' else frame.f_code.co_name).lower()
        if event in ('call','c_call') and (name in ('fit','fit_profile','fit_profiles','cdf_from_counts') or
            ('cdf' in name and any(word in name for word in ('fit','calculate','build','train')))):
            counter['runtime_profile_fit_count']+=1
            raise RuntimeError('Candidate-select profile fitting forbidden: '+name)
    sys.setprofile(guard)
    try:yield
    finally:sys.setprofile(previous)


def encode(q,profile,backend):
    domain,scales,shape=quantize(q,profile.metadata['bins'])
    require(shape==(32,3,profile.metadata['hidden']), 'Q profile/entry shape mismatch')
    payloads=[backend['core'].arithmetic_encode_fast(s,cdf) for s,cdf in zip(streams(domain,shape,backend),profile.cdfs)]
    fingerprint=hashlib.sha256(profile.to_bytes()).digest()
    header=FRAME_HEADER.pack(b'SCQBLK01',profile.metadata['bins'],*shape,fingerprint,*map(len,payloads))
    body=header+scales+b''.join(payloads)
    frame=body+hashlib.sha256(body).digest()
    sizes=dict(raw_q_bytes=q.numel()*q.element_size(),compressed_q_bitstream_bytes=sum(map(len,payloads)),
        compressed_q_local_metadata_bytes=len(header)+len(scales)+32,compressed_q_total_resident_bytes=len(frame))
    return frame,sizes


def decode(frame,profile,backend):
    import torch
    require(len(frame)>=FRAME_HEADER.size+384+32 and hashlib.sha256(frame[:-32]).digest()==frame[-32:], 'Corrupt Q frame')
    magic,bins,layers,tokens,hidden,fingerprint,na,nr=FRAME_HEADER.unpack_from(frame)
    require(magic==b'SCQBLK01' and bins==profile.metadata['bins'] and (layers,tokens,hidden)==(32,3,profile.metadata['hidden'])
        and fingerprint==hashlib.sha256(profile.to_bytes()).digest(), 'Q frame/profile mismatch')
    require(na>0 and nr>0 and len(frame)==FRAME_HEADER.size+384+na+nr+32, 'Q frame length mismatch')
    offset=FRAME_HEADER.size;maxima=torch.tensor(struct.unpack_from('<96f',frame,offset)).reshape(32,3,1)
    require(torch.isfinite(maxima).all().item() and (maxima>=0).all().item(), 'Invalid Q maxima')
    offset+=384;parts=[]
    for n,count,cdf in zip((na,nr),(32*hidden,64*hidden),profile.cdfs):
        parts.append(backend['core'].arithmetic_decode_fast(frame[offset:offset+n],count,cdf));offset+=n
    domain=inverse_streams(parts,(32,3,hidden),backend)
    symbols=torch.frombuffer(bytearray(domain),dtype=torch.uint8).to(torch.int16).reshape(32,3,hidden)-127
    limit=bins//2-1
    require((symbols.abs()<=limit).all().item(), 'Q reconstruction symbols outside candidate support')
    q=(symbols.float()/limit*maxima).to(torch.float16)
    validate_q_block(q,hidden=hidden)
    return q


def metrics(raw,reconstructed):
    import torch
    x=raw.double().flatten();y=reconstructed.double().flatten()
    require(x.shape==y.shape and torch.isfinite(x).all().item() and torch.isfinite(y).all().item(), 'Nonfinite/mismatched Q metric input')
    norm=float(x.norm());ynorm=float(y.norm());error=y-x
    relative=float(error.norm())/norm if norm else (0.0 if float(error.norm())==0 else float('inf'))
    cosine=float(torch.dot(x,y))/(norm*ynorm) if norm and ynorm else (1.0 if norm==ynorm==0 else 0.0)
    result=dict(mse=float(error.square().mean()),relative_l2=relative,
                cosine_similarity=max(-1.0,min(1.0,cosine)),max_absolute_error=float(error.abs().max()))
    require(all(math.isfinite(v) for v in result.values()), 'Nonfinite distortion metric')
    return result


def percentile(values,p):
    values=sorted(values);require(values and all(math.isfinite(v) for v in values), 'Nonfinite/empty aggregate')
    at=(len(values)-1)*p;lo=math.floor(at);hi=math.ceil(at)
    return values[lo]+(values[hi]-values[lo])*(at-lo)


def aggregate_metrics(rows):
    from statistics import mean,median
    result={}
    for key in ('mse','relative_l2','cosine_similarity','max_absolute_error'):
        values=[r[key] for r in rows]
        require(values and all(math.isfinite(v) for v in values), 'Invalid metrics population')
        result[key]=dict(mean=mean(values),median=median(values),p95=percentile(values,0.95),max=max(values),min=min(values))
    return result


def accounting(raw,payload,metadata):
    require(raw>0 and payload>0 and metadata>=0, 'Invalid resident byte accounting')
    total=payload+metadata
    return dict(raw_q_bytes=raw,compressed_q_bitstream_bytes=payload,compressed_q_local_metadata_bytes=metadata,
        compressed_q_total_resident_bytes=total,q_payload_compression_ratio=raw/payload,
        q_resident_compression_ratio=raw/total,q_resident_byte_reduction_percentage=100*(1-total/raw))


def pareto(rows):
    # Distortion axis is held-out block mean relative L2, predeclared here.
    return [r['candidate'] for r in rows if not any(
        other['q_resident_compression_ratio']>=r['q_resident_compression_ratio'] and other['mean_relative_l2']<=r['mean_relative_l2']
        and (other['q_resident_compression_ratio']>r['q_resident_compression_ratio'] or other['mean_relative_l2']<r['mean_relative_l2']) for other in rows)]


def load_capture(root):
    import torch
    cohort=read(root/'cohort.json');overlap=validate_cohort(cohort)
    manifest=read(root/'capture/capture_manifest.json')
    expected=dict(stage='C7-B0',status='COMPLETE',model=MODEL,model_revision=REVISION,tokenizer_revision=REVISION,
        prompt_version=VERSION,adapter_hashes=WEIGHTS,q_object=Q_OBJECT,layers=32,hidden_dim=2560,dtype='float16',blocks=128,
        model_inference_performed=True,training_performed=False,**overlap)
    for k,v in expected.items():require(manifest.get(k)==v, 'Capture manifest mismatch: '+k)
    require(manifest['cohort_sha256']==sha(root/'cohort.json'), 'Capture cohort changed')
    for name,expected_sha in manifest['output_hashes'].items():
        require(sha(root/'capture'/name)==expected_sha, 'Capture output changed: '+name)
    require(manifest['output_hashes'].get('q_blocks.pt')==sha(root/'capture/q_blocks.pt'), 'Unbound capture tensor file')
    require(manifest['input_hashes']==cohort['provenance']['input_hashes'], 'Capture input provenance differs')
    verify_files(manifest['input_hashes'])
    corpus=torch.load(root/'capture/q_blocks.pt',map_location='cpu',weights_only=True)
    require(corpus['schema']=='c7b_total_q_v1' and corpus['q_object']==Q_OBJECT and corpus['cohort_sha256']==manifest['cohort_sha256'], 'Q capture schema mismatch')
    require(set(corpus['blocks'])=={b['calibration_id'] for b in cohort['blocks']}, 'Q capture IDs incomplete')
    for q in corpus['blocks'].values():validate_q_block(q)
    require(sum(q.numel()*q.element_size() for q in corpus['blocks'].values())==manifest['raw_q_bytes'], 'Raw Q bytes differ')
    return cohort,manifest,corpus['blocks']


def write_csv(path,rows):
    require(rows, 'Empty CSV')
    with path.open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def calibrate(args):
    root=args.output_root;capture=args.capture_root
    seraph_paths(root)
    import torch
    allowed={'cohort.json','capture'} if root.resolve()==capture.resolve() else set()
    require(not root.exists() or {p.name for p in root.iterdir()}<=allowed, 'Refusing existing B1/unknown output files')
    cohort,manifest,corpus=load_capture(capture)
    capture_hashes={str((capture/p).resolve()):sha(capture/p) for p in
        ('cohort.json','capture/capture_manifest.json','capture/q_blocks.pt')}
    backend=load_backend(args.storage_src)
    fit=[b for b in cohort['blocks'] if b['split']=='profile_fit'];select=[b for b in cohort['blocks'] if b['split']=='candidate_select']
    require(len(fit)==96 and len(select)==32, 'Capture split changed')
    # Validate the frozen KV artifact read-only; it is never passed to fitting.
    require(sha(args.kv_profile)==PROFILE_SHA, 'Frozen UNIFORM_K20_V16 profile SHA differs')
    kv_before=sha(args.kv_profile)
    root.mkdir(parents=True,exist_ok=True);profiles=root/'profiles';profiles.mkdir(exist_ok=False)
    per_block=[];per_layer=[];summary=[];profiles_meta={};counter={'runtime_profile_fit_count':0}
    fit_sha=digest(fit)
    for bins in CANDIDATES:
        candidate='Q'+str(bins)
        # Loader closure exposes only fit IDs during fitting.
        fit_ids={b['calibration_id'] for b in fit}
        def fit_loader(cid):
            require(cid in fit_ids, 'Candidate-select tensor requested by fitter')
            return corpus[cid]
        profile=fit_profile(fit,fit_loader,bins,backend,fit_sha)
        path=profiles/f'q{bins}.bin';path.write_bytes(profile.to_bytes())
        profile=QProfile.from_bytes(path.read_bytes(),backend)
        profiles_meta[candidate]=dict(path=str(path.resolve()),sha256=sha(path),shared_profile_bytes=path.stat().st_size,
            fit_block_ids=profile.metadata['fit_block_ids'],fit_cohort_sha256=fit_sha,bins=bins,layers=32,transform=TRANSFORM,
            profile_fit_call_count=1,cdf_fit_call_count=2)
        block_rows=[];layer_rows=[]
        with no_runtime_fitting(counter):
            for block in select:
                q=corpus[block['calibration_id']]
                frame,sizes=encode(q,profile,backend);decoded=decode(frame,profile,backend)
                row=dict(candidate=candidate,calibration_id=block['calibration_id'],user=block['user'],history_depth=block['history_depth'],
                    **sizes,**metrics(q,decoded))
                block_rows.append(row)
                for layer in range(32):layer_rows.append(dict(candidate=candidate,calibration_id=block['calibration_id'],
                    layer=layer,**metrics(q[layer],decoded[layer])))
        require(counter['runtime_profile_fit_count']==0, 'Forbidden held-out fitting')
        rates=accounting(sum(r['raw_q_bytes'] for r in block_rows),sum(r['compressed_q_bitstream_bytes'] for r in block_rows),
                         sum(r['compressed_q_local_metadata_bytes'] for r in block_rows))
        distortion=aggregate_metrics(block_rows);layer_distortion=aggregate_metrics(layer_rows)
        row=dict(candidate=candidate,bins=bins,transform=TRANSFORM,**rates,
            byte_reduction_percentage=rates['q_resident_byte_reduction_percentage'],mean_mse=distortion['mse']['mean'],
            mean_relative_l2=distortion['relative_l2']['mean'],p95_relative_l2=distortion['relative_l2']['p95'],
            mean_cosine=distortion['cosine_similarity']['mean'],min_cosine=distortion['cosine_similarity']['min'],
            max_absolute_error=distortion['max_absolute_error']['max'],shared_profile_bytes=path.stat().st_size)
        summary.append(row);per_block.extend(block_rows);per_layer.extend(layer_rows)
        profiles_meta[candidate].update(block_metric_aggregates=distortion,layer_metric_aggregates=layer_distortion)
        profiles_meta[candidate]['per_layer_metric_aggregates']={str(layer):aggregate_metrics(
            [r for r in layer_rows if r['layer']==layer]) for layer in range(32)}
        print(f'{candidate}: held-out32 completed; no runtime profile fits',flush=True)
    require(sha(args.kv_profile)==kv_before, 'Frozen KV profile changed')
    verify_files(manifest['input_hashes'])
    verify_files(capture_hashes)
    write_csv(root/'candidate_summary.csv',summary);write(root/'candidate_summary.json',dict(candidates=summary,profiles=profiles_meta,selected_q_candidate=None))
    write_csv(root/'per_block_metrics.csv',per_block);write_csv(root/'per_layer_metrics.csv',per_layer)
    frontier=pareto(summary)
    write(root/'rate_distortion.json',dict(candidates=summary,metric_aggregates=profiles_meta,pareto_frontier=frontier,
        pareto_axes=['maximize q_resident_compression_ratio','minimize mean_relative_l2'],p95_definition='linear interpolation at (n-1)*0.95',selected_q_candidate=None))
    text=['# C7-B1 TOTAL-Q calibration','',
        'C7-A2 established that resident TOTAL Q is an active SemCache HIT payload alongside K/V. C7-B evaluates its storage compression.',
        '128 conversation-disjoint training-pool blocks exclude frozen32, capability64 and all discovered hash-bound B3 evaluation cohorts.',
        'Only96 profile_fit blocks fit Q CDFs. Only32 candidate_select blocks measure held-out rate/distortion.',
        'Q16/Q20/Q24/Q32 are bin counts. The existing role-generic B2 anchor/modulus255 math uses a separate Q profile/frame namespace.',
        'Results are tensor-level rate/distortion only. No BLEU/generation-quality preservation claim is made. No candidate is automatically frozen.','',
        '| Candidate | Resident ratio | Reduction % | Mean MSE | Mean rel L2 | p95 rel L2 | Mean cosine | Min cosine | Max abs error | Profile bytes |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in summary:text.append('| '+r['candidate']+' | '+' | '.join(f'{r[k]:.7g}' for k in ('q_resident_compression_ratio','byte_reduction_percentage',
        'mean_mse','mean_relative_l2','p95_relative_l2','mean_cosine','min_cosine','max_absolute_error','shared_profile_bytes'))+' |')
    text += ['', 'Pareto frontier (compression vs mean relative L2): '+', '.join(frontier)+'. Manual review required; C7-B2 freeze not performed.',
        'Shared profile bytes are separate; resident entries count entropy payload plus local frame/scale metadata.','']
    (root/'summary.md').write_text('\n'.join(text))
    write(root/'manifest.json',dict(stage='C7-B1',status='COMPLETE',research_scope='resident TOTAL-Q rate-distortion calibration only',source_stage='C7-A2',
        semcache_contract='PHYSICAL_EXACT_W3_WITHIN_SEMANTIC_CLUSTER; HIT payload TOTAL_QKV',q_candidates=list(CANDIDATES),
        fit_blocks=96,candidate_select_blocks=32,**validate_cohort(cohort),downstream_quality_evaluated=False,
        q_profile_frozen=False,selected_q_candidate=None,existing_kv_profile_modified=False,
        existing_kv_profile=dict(name='UNIFORM_K20_V16',path=str(args.kv_profile.resolve()),sha256=PROFILE_SHA),
        model=MODEL,model_revision=REVISION,tokenizer_revision=REVISION,adapter_hashes=WEIGHTS,
        capture_artifact_sha256=sha(capture/'capture/q_blocks.pt'),capture_manifest_sha256=sha(capture/'capture/capture_manifest.json'),
        cohort_sha256=sha(capture/'cohort.json'),transform_provenance=backend['provenance'],profile_hashes={k:v['sha256'] for k,v in profiles_meta.items()},
        runtime_profile_fit_count_on_candidate_select=counter['runtime_profile_fit_count'],model_inference_performed=False,training_performed=False,
        gpu_required=False,software=dict(torch=torch.__version__,python=sys.version),git=git(),
        output_hashes={str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file()}))


def main(argv=None):
    from .c6_quality import PROFILE
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture-root',type=Path,default=Path(DEFAULT_ROOT));p.add_argument('--output-root',type=Path,default=Path(DEFAULT_ROOT))
    p.add_argument('--storage-src',type=Path,default=Path('/data/khuss/repos/GP_semcache/.c6_storage_src/src'))
    p.add_argument('--kv-profile',type=Path,default=Path(PROFILE))
    calibrate(p.parse_args(argv))
