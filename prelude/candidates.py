import torch


def prior_candidates(eps, n_cand=16, max_lag=72):
    """Prior candidates and source-lag prior from lagged correlations of training-segment innovations.
    eps: standardized innovations [T, N]. Returns candidates [N, C] and prior |rho_ik(l)| [N, C, max_lag]."""
    n, N = eps.shape
    corr = torch.stack([eps[l:].T @ eps[:n - l] / (n - l) for l in range(1, max_lag + 1)])  # [lag, target, source]
    score = corr.abs().max(0)[0]
    score.fill_diagonal_(-1)
    _, cand = score.topk(min(n_cand, N - 1), 1)
    prior = corr.abs()[:, torch.arange(N, device=eps.device)[:, None], cand].permute(1, 2, 0)
    return cand.cpu(), prior.cpu()
