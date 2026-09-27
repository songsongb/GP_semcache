"""Tiny deterministic CPU comparisons; never loads captures or models."""
from .policy import CACHEGEN_RELEASED_QL2
from .reference import reference_bins, reference_functions, source_manifest

LAYERS = {'K': (0, 9, 10, 19, 20, 31), 'V': (0, 1, 2, 31)}


def run_parity(repo=None):
    import torch
    quant, dequant = reference_functions(repo)
    reference = reference_bins(repo)
    rows = []
    for dtype in (torch.float16, torch.float32):
        for role, layers in LAYERS.items():
            bins = reference[role][list(layers)]
            limits = bins // 2 - 1
            # Exact half-integer shifted ties, asymmetric signs, extrema, zero,
            # independent token scales, near-extrema and an all-zero vector.
            x = torch.stack([torch.tensor([
                [-c, c, 0, .5, -.5, 1.5, -1.5, c-.125],
                [-2*c, 2*c, 0, 1, -1, 3, -3, 2*c-.25],
                [-.25, .25, 0, .2499, -.2499, .0625, -.0625, .125],
                [0, 0, 0, 0, 0, 0, 0, 0]], dtype=dtype)
                for c in limits.tolist()])
            actual = CACHEGEN_RELEASED_QL2.quantize(x, role, layer_indices=layers)
            rq, rm = quant(bins, x)
            reconstruction = dequant(rq.float(), bins, rm)
            # Keep the released vectorized FP32 promotion. A scalar limits[i]
            # denominator would instead round this diagnostic step to FP16.
            reference_scale = rm / limits[:, None, None]
            for i, layer in enumerate(layers):
                scale_diff = (actual.scale[i]-reference_scale[i]).abs().max().item()
                maxabs_diff = (actual.maxabs[i]-rm[i]).abs().max().item()
                diff = (actual.dequantize()[i]-reconstruction[i]).abs().max().item()
                fp16_diff = (actual.reconstructed[i]-reconstruction.to(dtype)[i]).abs().max().item()
                equal = torch.equal(actual.symbols[i], rq[i])
                expected_bins = int(bins[i].item())
                passed = (actual.bins[i].item() == expected_bins and equal and
                          scale_diff == maxabs_diff == diff == fp16_diff == 0)
                rows.append(dict(role=role, layer=layer, input_dtype=str(dtype), bins=expected_bins,
                    quantization_limit=int(limits[i].item()), integer_range=[0, 2*int(limits[i].item())],
                    signed_storage_range=[-int(limits[i].item()), int(limits[i].item())],
                    scale_max_abs_difference=scale_diff, maxabs_max_abs_difference=maxabs_diff,
                    symbol_equality=equal, reconstruction_max_abs_difference=diff,
                    final_cast_max_abs_difference=fp16_diff, passed=passed,
                    all_zero_vector_symbols=rq[i, -1].tolist(),
                    zero_behavior='Released unguarded NaN-to-int8 conversion; measured on CPU only'))
    return dict(stage='C1.5C-0', status='PASS' if all(r['passed'] for r in rows) else 'FAIL',
        device='cpu', seed=42, torch_version=torch.__version__, synthetic_only=True, source=source_manifest(repo),
        reference_mode='ISOLATED_RELEASED_AST; bin .cuda() replaced with identity' if repo else 'DIRECT_PINNED_FORMULA',
        policy=CACHEGEN_RELEASED_QL2.name, layer_profile=CACHEGEN_RELEASED_QL2.profile(), tests=rows,
        real_opt_smoke_status='NOT RUN — requires SERAPH')


def print_parity(report):
    print('role layer dtype          bins limit range   scale-diff symbol-equal recon-diff pass')
    for r in report['tests']:
        print(f"{r['role']:4} {r['layer']:5} {r['input_dtype']:14} {r['bins']:4} "
              f"{r['quantization_limit']:5} 0..{r['integer_range'][1]:2} "
              f"{r['scale_max_abs_difference']:10.2g} {str(r['symbol_equality']):12} "
              f"{r['reconstruction_max_abs_difference']:10.2g} {r['passed']}")
    print('C1.5C-0:', report['status'])
