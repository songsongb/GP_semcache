"""Float64 reductions; relative L2 uses the first (reference) tensor norm."""
def similarity(reference, candidate):
    import torch
    if reference.shape != candidate.shape or reference.numel() == 0:
        raise ValueError('Need equal nonempty tensor shapes')
    a = reference.detach().to(device='cpu', dtype=torch.float64).flatten()
    b = candidate.detach().to(device='cpu', dtype=torch.float64).flatten()
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError('Nonfinite projection values')
    delta = (a-b).abs()
    na, nb = a.norm().item(), b.norm().item()
    return {'cosine_similarity': (float(torch.dot(a,b)/(na*nb)) if na and nb else (1.0 if na == nb else 0.0)),
            'relative_l2': delta.norm().item()/na if na else (0.0 if not nb else None),
            'max_abs_error': delta.max().item(), 'mean_abs_error': delta.mean().item()}
