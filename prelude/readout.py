import numpy as np
import torch
import torch.nn as nn


class TransferReadout(nn.Module):
    """Rational transfer readout (Eq. 26) with time- and horizon-aware modulation (Eq. 27)."""

    def __init__(self, n_vars, pred_len, cycle, n_cand=16, max_lag=72, window=24, gate_b0=0.0):
        super().__init__()
        C = min(n_cand, n_vars - 1)
        self.lmax, self.hn, self.S, self.spd = max_lag, min(max_lag, pred_len), window, cycle
        self.a = nn.Parameter(torch.zeros(n_vars, C))  # pairwise reading coefficients
        V = torch.zeros(max_lag, max_lag, self.hn)  # transfer kernel, initialized as the shift operator
        for l in range(1, max_lag + 1):
            for h in range(min(l, self.hn)):
                V[l - 1, max_lag - l + h, h] = 1.0
        self.W = nn.Parameter(V)
        self.tf_p = nn.Parameter(torch.zeros(n_vars))  # denominator coefficient: delta = 0.95 tanh(tf_p)
        self.gam = nn.Parameter(torch.exp(-torch.arange(self.hn, dtype=torch.float32) / max_lag))  # step-wise coefficients
        self.gmlp = nn.Sequential(nn.Linear(6, 16), nn.GELU(), nn.Linear(16, self.hn))
        with torch.no_grad():
            self.gmlp[2].weight.zero_()
            self.gmlp[2].bias.zero_()
        self.gb0 = nn.Parameter(torch.tensor(float(gate_b0)))
        self.gbv = nn.Parameter(torch.zeros(n_vars))

    def forward(self, u, w, cand, us, pos, pred_len):
        # u: innovations [B, U, N]; w: routing weights [B, N, C, lmax]; us: innovation scale [N] -> correction [B, H, N]
        B, _, N = u.shape
        f = u[:, -self.lmax:, cand]  # [B, lmax, N, C]
        Z = torch.einsum("bncl,btnc->bnlt", w * self.a[None, :, :, None], f)
        dl = 0.95 * torch.tanh(self.tf_p)
        idx = torch.arange(self.lmax, device=u.device)
        diff = idx[:, None] - idx[None, :]
        D = (dl[:, None, None] ** diff.clamp_min(0)) * (diff >= 0)  # geometric recursion
        Z = torch.einsum("bnls,nts->bnlt", Z, D)
        out = torch.einsum("bnlt,lth->bnh", Z, self.W) * us[None, :, None]
        out = 2 * torch.tanh(out / 2)
        e_sel = (w * f.pow(2).mean(1)[..., None]).sum((2, 3))
        e_self = u[:, -self.S:].pow(2).mean(1)
        feat = [e_sel.clamp_min(1e-6).log(), e_self.clamp_min(1e-6).log()]
        p = pos.float()
        for per in (self.spd, 7 * self.spd):
            ang = 2 * np.pi * (p % per) / per
            feat += [ang.sin()[:, None].expand(B, N), ang.cos()[:, None].expand(B, N)]
        g = torch.sigmoid(self.gb0 + self.gbv[None, :, None] + self.gmlp(torch.stack(feat, -1)))
        delta = torch.zeros(B, pred_len, N, device=u.device, dtype=out.dtype)
        delta[:, :self.hn] = (out * g * self.gam).transpose(1, 2)
        return delta
