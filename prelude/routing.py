import torch
import torch.nn as nn
import torch.nn.functional as F

PRIOR_SCALE = 20.0


class SourceLagRouting(nn.Module):
    """Spectral source retention (Eq. 24) and source-lag routing (Eq. 25) over the prior candidates."""

    def __init__(self, n_vars, n_innov, n_cand=16, max_lag=72, window=24, n_spec=16, temp=0.5, ratio=0.25, ratio_w=0.1):
        super().__init__()
        C, S = min(n_cand, n_vars - 1), window
        assert max_lag <= n_innov
        self.lmax, self.S, self.ls = max_lag, S, min(max_lag, n_innov - S)
        self.temp, self.ratio, self.ratio_w = temp, ratio, ratio_w
        self.register_buffer("cand", torch.zeros(n_vars, C, dtype=torch.long))
        self.theta = nn.Parameter(torch.zeros(n_vars, C, max_lag))  # source-lag score table
        self.wq, self.wk = nn.Linear(S, S, bias=False), nn.Linear(S, S, bias=False)
        with torch.no_grad():
            self.wq.weight.copy_(torch.eye(S))
            self.wk.weight.copy_(torch.eye(S))
        self.tau = nn.Parameter(torch.tensor(10.0))
        nf = n_innov // 2 + 1
        self.spec_A = nn.Linear(nf, n_spec, bias=False)
        nn.init.normal_(self.spec_A.weight, std=nf ** -0.5)
        self.spec_s0 = nn.Parameter(torch.tensor(3.0))
        self.aux_loss = 0.0

    def set_candidates(self, cand, prior):
        with torch.no_grad():
            self.cand.copy_(cand)
            self.theta.copy_(PRIOR_SCALE * prior.float())

    def retention(self, u):
        spec = torch.fft.rfft(u, dim=1).abs().transpose(1, 2)  # [B, N, nf]
        q = self.spec_A(spec / (spec.norm(dim=-1, keepdim=True) + 1e-6))
        d = (q[:, :, None] - q[:, self.cand]).pow(2).sum(-1)  # [B, N, C]
        return torch.sigmoid(self.spec_s0 - d)

    def forward(self, u):
        # u: innovations [B, U, N] -> routing weights [B, N, C, lmax]
        B, U, N = u.shape
        S, ls = self.S, self.ls
        segs = u[:, U - S - ls:U - 1].unfold(1, S, 1).flip(1)  # [B, ls, N, S]; entry l-1 is the source window at lag l
        logit = self.theta.expand(B, -1, -1, -1)
        q = self.wq(u[:, -S:].transpose(1, 2))
        k = self.wk(segs)[:, :, self.cand]
        sim = F.pad(torch.einsum("bnd,blncd->bncl", q, k), (0, self.lmax - ls))  # lags beyond ls use the score table only
        logit = logit + self.tau * sim / S
        p = self.retention(u)
        if self.training:  # relaxed Bernoulli mask
            g1 = -torch.log(-torch.log(torch.rand_like(p).clamp(1e-6, 1 - 1e-6)))
            g2 = -torch.log(-torch.log(torch.rand_like(p).clamp(1e-6, 1 - 1e-6)))
            mask = torch.sigmoid((torch.log(p + 1e-6) - torch.log(1 - p + 1e-6) + g1 - g2) / self.temp)
        else:
            mask = p
        self.aux_loss = self.ratio_w * (p.mean(-1) - self.ratio).pow(2).mean() if self.training and self.ratio_w > 0 else 0.0
        logit = logit + torch.log(mask.clamp_min(1e-6))[..., None]
        return torch.softmax(logit.reshape(B, N, -1), dim=-1).view_as(logit)
