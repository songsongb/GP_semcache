"""Baseline codecs and a narrow adapter to the external official legacy API."""
import importlib
import importlib.util
import subprocess
import sys
import time
from pathlib import Path
from .common import environment, file_hash, from_heads, to_heads

SERDE = 'lmcache.storage_backend.serde'


class Unavailable(RuntimeError):
    pass


def audit(repo=None):
    """Read sources before importing codec code. Never builds an extension."""
    report = dict(status='UNKNOWN', opt_support='UNKNOWN', c1_full_available=False,
        c1_full_reason='No reviewed compatible external codec',
        cachegen_quant_available=False,
        cachegen_quant_reason='No clean official anchor/delta quantize-only boundary verified',
        serializer_import=SERDE+'.cachegen_encoder.CacheGenSerializer',
        deserializer_import=SERDE+'.cachegen_decoder.CacheGenDeserializer',
        cachegen_revision=None, source_hashes={}, environment=environment(),
        paper_parameters=dict(group_tokens=10, anchor_bits=8, bins=[0.5, 1.0, 1.5],
                              provenance='PAPER_DEFINED', applied=False),
        padding_tokens=0, automatic_padding=False)
    if repo is None:
        return report
    repo = Path(repo).resolve()
    report['repo'] = str(repo)
    def git(*args):
        p = subprocess.run(['git', '-C', str(repo), *args], text=True, capture_output=True)
        return p.stdout.strip() if p.returncode == 0 else None
    report.update(cachegen_revision=git('rev-parse', 'HEAD'), git_dirty=bool(git('status', '--porcelain')),
                  git_remote=git('remote', 'get-url', 'origin'))
    candidates = [repo/'LMCache', repo]
    base = next((p for p in candidates if (p/'lmcache/storage_backend/serde/cachegen_encoder.py').exists()), None)
    if base is None:
        report['reason'] = 'Legacy serializer API not found; newer LMCache APIs need separate review'
        return report
    report['python_root'] = str(base)
    report['torchac_cuda_build_sources_present'] = (base/'third_party/torchac_cuda/setup.py').exists()
    sources = {}
    for name in ('encoder', 'decoder', 'basics'):
        path = base/f'lmcache/storage_backend/serde/cachegen_{name}.py'
        if not path.exists():
            report['reason'] = f'Missing {path}'
            return report
        sources[name] = path.read_text()
        report['source_hashes'][str(path)] = file_hash(path)
    enc, dec, basics = (sources[n] for n in ('encoder', 'decoder', 'basics'))
    self_cdf = 'calculate_cdf(new_key' in enc and 'calculate_cdf(new_value' in enc
    lookup = 'from_model_name(metadata.model_name)' in enc
    opt_literal = 'facebook/opt-2.7b' in basics
    # Positive support is proven only by a future reviewed adapter/runtime probe.
    opt = 'UNSUPPORTED' if lookup and not opt_literal and 'is not supported' in basics else 'UNKNOWN'
    report.update(status='SOURCE_AUDITED', opt_support=opt,
        arbitrary_model_metadata=False if lookup else 'UNKNOWN',
        model_specific_config_lookup=lookup,
        cdf_policy='INPUT_SELF_FITTED' if self_cdf else 'UNKNOWN',
        external_cdf_profile_assets='No external CDF assets in reviewed active path' if self_cdf else 'UNKNOWN',
        layout='[L,2,H,T,Dh] for fmt=huggingface; internal [L,2,T,H,Dh]',
        head_constraints='Python derives H and Dh from input; CUDA constraints require smoke validation',
        cuda_required='.cuda()' in enc or '.cuda()' in dec,
        anchor_delta_boundary='Not exposed in reviewed active encode_function',
        artifact_parameters='Integer layer-wise bin counts from CacheGenConfig; not paper 0.5/1.0/1.5',
        c1_full_reason=('OPT model configuration is unsupported; ' if opt == 'UNSUPPORTED' else '') +
            ('official encoder fits CDF on evaluation input, forbidden by C1 split policy' if self_cdf
             else 'CDF fit policy and OPT support require explicit source review'),
        provenance='OFFICIAL_ARTIFACT source audit; OPT adaptation REPRODUCTION_CHOICE')
    return report


def timing(call, device, cuda_events=False):
    import torch
    gpu = str(device).startswith('cuda')
    if gpu:
        torch.cuda.synchronize(device)
    start = end = None
    if cuda_events:
        if not gpu:
            raise ValueError('CUDA events require CUDA execution')
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    wall = time.perf_counter_ns()
    if start is not None:
        start.record()
    value = call()
    if end is not None:
        end.record()
    if gpu:
        torch.cuda.synchronize(device)
    elapsed = (time.perf_counter_ns()-wall)/1e6
    return value, elapsed, start.elapsed_time(end) if start is not None else None


class Baseline:
    def __init__(self, mode):
        if mode not in ('FP16_RAW', 'UNIFORM_INT8'):
            raise Unavailable(f'{mode}: no verified official implementation available')
        self.mode = mode

    def encode(self, k, v):
        import torch
        if self.mode == 'FP16_RAW':
            return (k.clone(), v.clone())
        def quant(x):
            xf = x.float()
            scale = xf.abs().amax(-1, keepdim=True)/127
            scale = torch.where(scale == 0, torch.ones_like(scale), scale)
            return torch.round(xf/scale).clamp(-127, 127).to(torch.int8), scale
        return quant(k), quant(v)

    def decode(self, encoded):
        if self.mode == 'FP16_RAW':
            return tuple(x.clone() for x in encoded)
        return tuple((symbols.float()*scale).half() for symbols, scale in encoded)

    def sizes(self, encoded):
        if self.mode == 'FP16_RAW':
            return sum(x.numel()*x.element_size() for x in encoded), 0
        return (sum(x.numel()*x.element_size() for x, _ in encoded),
                sum(s.numel()*s.element_size() for _, s in encoded))


class Official:
    """Only imports an externally installed codec; no fallback or monkey patches."""
    def __init__(self, report, model, tokens):
        import torch
        if report.get('status') != 'SOURCE_AUDITED' or not report.get('cachegen_revision'):
            raise Unavailable('A git-tracked external source audit is required')
        for path, expected in report['source_hashes'].items():
            if file_hash(path) != expected:
                raise Unavailable('Codec sources changed since audit')
        spec = importlib.util.find_spec('torchac_cuda')
        if spec is None or not spec.origin or not spec.origin.endswith(('.so', '.pyd')):
            raise Unavailable('Prebuilt torchac_cuda extension required; no build is attempted')
        extension = importlib.import_module('torchac_cuda')
        self.extension = extension
        if not torch.cuda.is_available():
            raise Unavailable('Official artifact requires CUDA')
        sys.path.insert(0, report['python_root'])
        self.enc = importlib.import_module(SERDE+'.cachegen_encoder')
        self.dec = importlib.import_module(SERDE+'.cachegen_decoder')
        self.basics = importlib.import_module(SERDE+'.cachegen_basics')
        for module in (self.enc, self.dec, self.basics):
            if str(Path(module.__file__).resolve()) not in report['source_hashes']:
                raise Unavailable('Imported codec does not match audited checkout')
        config = importlib.import_module('lmcache.config')
        cfg = config.LMCacheEngineConfig.from_defaults(chunk_size=tokens)
        metadata = config.LMCacheEngineMetadata(model_name=model, fmt='huggingface', world_size=1, worker_id=0)
        self.serializer = self.enc.CacheGenSerializer(cfg, metadata)
        self.deserializer = self.dec.CacheGenDeserializer(cfg, metadata)
        self.report = report

    def encode(self, k, v, heads=32):
        import torch
        return self.serializer.to_bytes(torch.stack((to_heads(k, heads), to_heads(v, heads)), dim=1))

    def decode(self, encoded):
        x = self.deserializer.from_bytes(encoded)
        return from_heads(x[:, 0]).half(), from_heads(x[:, 1]).half()

    def sizes(self, encoded):
        obj = self.basics.CacheGenGPUEncoderOutput.from_bytes(encoded)
        payload = sum(c.bytestream.numel()*c.bytestream.element_size() for c in obj.data_chunks)
        if not 0 < payload <= len(encoded):
            raise ValueError('Unexpected official serialized accounting')
        # Everything other than arithmetic stream bytes: CDF, scales, lengths,
        # shape and Python serialization framing. No overhead is omitted.
        return payload, len(encoded)-payload

    def symbols_exact(self, k, v, encoded):
        """Separate correctness pass; never inside measured encode/decode regions."""
        import torch
        if not all(hasattr(m, n) for m, n in ((self.enc, 'torch_quant_vectorized'),
                                             (self.dec, 'decode_function_gpu'))):
            return None
        qk, _ = self.enc.torch_quant_vectorized(self.serializer.key_bins, k)
        qv, _ = self.enc.torch_quant_vectorized(self.serializer.value_bins, v)
        obj = self.basics.CacheGenGPUEncoderOutput.from_bytes(encoded)
        buf = torch.empty((k.shape[1], 2*k.shape[0]*k.shape[2]), dtype=torch.uint8, device=k.device)
        dk, dv = self.dec.decode_function_gpu(obj.cdf, obj.data_chunks, k.shape[0], k.shape[1], buf)
        return bool(torch.equal(qk.to(dk.dtype), dk) and torch.equal(qv.to(dv.dtype), dv))


def stage_status(mode, report=None):
    if mode in ('FP16_RAW', 'UNIFORM_INT8'):
        return True, None
    if mode not in ('CACHEGEN_QUANT', 'CACHEGEN_FULL'):
        raise ValueError('Unknown compression mode')
    report = report or audit()
    key = 'cachegen_quant' if mode == 'CACHEGEN_QUANT' else 'c1_full'
    return bool(report[key+'_available']), report[key+'_reason']
