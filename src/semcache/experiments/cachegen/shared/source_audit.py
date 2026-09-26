"""Read-only audit of the pinned external source. Never imports/builds it."""
from pathlib import Path
import subprocess
from ..common import file_hash

REVISION = '6bed34ca9d495289955ed754cf2ad0c43346ee48'
# Independently fetched from commit-pinned raw.githubusercontent.com sources.
SOURCE_SHA256 = {
    'LMCache/third_party/torchac_cuda/main.cpp':
        '0fbb7827efbe7904669264a56304b3c01d5f44cb24b06d8a8c02b40437086767',
    'LMCache/third_party/torchac_cuda/torchac_kernel_enc_new.cu':
        '724e59dba0f1921792f7ab60205886a35cac8c369185ebf769c63826c8e39bcd',
    'LMCache/third_party/torchac_cuda/torchac_kernel_dec_new.cu':
        'a78b1320d3838f6f2028c6401d79b4eef8301ec90025913cc07a41268ca7d9d1',
    'LMCache/lmcache/storage_backend/serde/cachegen_encoder.py':
        'b8a0ec7b4d1c787ac885f03b86d361d2e64e88c0b12486c0ac271e1c9ec71832',
    'LMCache/lmcache/storage_backend/serde/cachegen_basics.py':
        '82303500bb5fd44f6c601ff01fec5b7de919d160bb8fe4f429a609c268816ea7',
}


def audit(repo):
    repo = Path(repo)
    result = dict(official_repo=str(repo), official_revision=REVISION,
        source_sha256=SOURCE_SHA256, local_checkout_status='NOT_MOUNTED; pinned upstream source audited',
        imported_arithmetic_coder_path=None, official_code_reused=False,
        boundary='torchac_cuda.encode_fast_new / decode_fast_new / decode_fast_prefsum',
        cdf_representation='CUDA int16 [layers,channels,Lp], interpreted unsigned; 16-bit cumulative '
                           'counts, terminal 65536 implicit/wrapped; strictly increasing unsigned support',
        symbol_representation='CUDA int8 bits reinterpreted as uint8, [layers,tokens,channels]',
        requirements='prebuilt torchac_cuda CUDA extension; all buffers on CUDA; channel/block divisibility',
        integration_status='UNAVAILABLE_FOR_C15_ALPHABET',
        reason='Encoder MAX_LP=48, decoder MAX_LP=64; unchanged 255-symbol UNIFORM_INT8 needs Lp=256. '
               'No clean direct boundary for this alphabet without changing official kernels or symbol factorization.',
        selected_coder='semcache.experiments.cachegen.shared.core; deterministic CPU; RESEARCH_EXTENSION',
        high_level_encoder_used=False, cdf_fit_policy='calibration only; never calculate_cdf on evaluation',
        c15b_status='DEFERRED: active pinned encoder has no anchor/delta transform or group-of-10 semantics')
    if repo.exists():
        revision = subprocess.run(['git', '-C', str(repo), 'rev-parse', 'HEAD'], check=True,
                                  capture_output=True, text=True).stdout.strip()
        if revision != REVISION:
            raise ValueError('Official CacheGen revision differs from pinned audit')
        for name, expected in SOURCE_SHA256.items():
            if file_hash(repo/name) != expected:
                raise ValueError(f'Official source differs from pinned audit: {name}')
        result['local_checkout_status'] = 'PINNED_REVISION_AND_SOURCE_HASHES_VERIFIED'
    return result
