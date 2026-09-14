"""CPU float64 reductions; KL direction is baseline || injected throughout."""
from .qkv_metrics import similarity


def compare_logits(baseline, injected, target_start):
    import torch
    if baseline.ndim != 3 or baseline.shape[0] != 1 or baseline.shape != injected.shape:
        raise ValueError('Expected equal batch-one [batch, sequence, vocabulary] logits')
    if type(target_start) is not int or not 0 <= target_start < baseline.shape[1]:
        raise ValueError('Invalid affected suffix start')
    m = similarity(baseline, injected)
    a, b = (x.detach().to(device='cpu', dtype=torch.float64) for x in (baseline, injected))
    la, lb = a.log_softmax(-1), b.log_softmax(-1)
    kl = (la.exp()*(la-lb)).sum(-1)
    ai, bi = a[0, -1].argmax().item(), b[0, -1].argmax().item()
    return dict(max_abs_logit_diff=m['max_abs_error'], mean_abs_logit_diff=m['mean_abs_error'],
                relative_l2_logit_diff=m['relative_l2'], logit_cosine_similarity=m['cosine_similarity'],
                last_position_kl_baseline_to_injected=kl[0, -1].item(),
                affected_suffix_mean_kl=kl[:, target_start:].mean().item(),
                baseline_last_argmax_token_id=ai, injected_last_argmax_token_id=bi,
                last_argmax_agreement=ai == bi,
                prefix_max_abs_logit_diff=(a[:, :target_start]-b[:, :target_start]).abs().max().item() if target_start else 0.0)
