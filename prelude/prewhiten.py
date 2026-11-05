import numpy as np
import torch
import torch.nn as nn


def fit_prewhitening(x_train, cycle, order=24, device="cpu"):
    """Training-segment initial values: periodic profile, cross-sectional loadings, AR(P) coefficients and
    innovation scale; also returns the standardized training innovations used for the prior candidates."""
    T, N = x_train.shape
    P = order
    prof = np.stack([x_train[p::cycle].mean(0) for p in range(cycle)])
    r = x_train - prof[np.arange(T) % cycle]
    m = r.mean(1)
    b = r.T @ m / (m @ m)
    e = torch.tensor(r - np.outer(m, b), device=device)
    X = torch.stack([e[P - p:T - p] for p in range(1, P + 1)], -1)
    coef = torch.linalg.lstsq(X.permute(1, 0, 2), e[P:].T[..., None]).solution[..., 0]
    eps = e[P:] - torch.einsum("tnp,np->tn", X, coef)
    us = eps.std(0)
    eps = (eps - eps.mean(0)) / us
    return dict(prof=prof, b=b, coef=coef.cpu(), us=us.cpu()), eps


class Prewhitening(nn.Module):
    """Removes the periodic profile and the cross-sectional factor, then the target's own AR(P) dynamics (Eq. 22)."""

    def __init__(self, n_vars, seq_len, cycle, order=24, one_step_w=0.1):
        super().__init__()
        assert seq_len > order
        self.L, self.cyc, self.P, self.one_step_w = seq_len, cycle, order, one_step_w
        self.prof = nn.Parameter(torch.zeros(cycle, n_vars))
        self.b = nn.Parameter(torch.zeros(n_vars))
        self.coef = nn.Parameter(torch.zeros(n_vars, order))
        self.register_buffer("us", torch.ones(n_vars))
        self.aux_loss = 0.0

    def set_statistics(self, prof, b, coef, us):
        with torch.no_grad():
            self.prof.copy_(torch.tensor(prof, dtype=torch.float32))
            self.b.copy_(torch.tensor(b, dtype=torch.float32))
            self.coef.copy_(coef.float())
            self.us.copy_(us.float())

    def forward(self, x, pos):
        # x: [B, L, N] -> innovations scaled by the training-segment innovation std, [B, L - P, N]
        P = self.P
        pin = pos.long()[:, None] + torch.arange(-self.L, 0, device=x.device)
        r = x - self.prof[pin % self.cyc]
        e = r - r.mean(2, keepdim=True) * self.b
        U = e.unfold(1, P + 1, 1)  # last dim: e(t-P) .. e(t)
        u = (U[..., P] - (U[..., :P].flip(-1) * self.coef).sum(-1)) / self.us
        self.aux_loss = self.one_step_w * u.pow(2).mean() if self.training and self.one_step_w > 0 else 0.0
        return u
