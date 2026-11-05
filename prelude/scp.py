import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SharedComponentPathway(nn.Module):
    """Shared-component pathway: periodic basis table, cross-sectional pooling with K common factors,
    linear autoregressive bypass and seasonal term (Section VI-B)."""

    def __init__(self, n_vars, seq_len, pred_len, cycle, d_model, dropout, n_factors=0, seasonal_lags=0, scale="raw",
                 n_heads=4, pool_drop=0.5):
        super().__init__()
        assert scale in ("raw", "last"), scale
        L, H, N, d = seq_len, pred_len, n_vars, d_model
        self.L, self.H, self.N, self.cyc, self.scale = L, H, N, cycle, scale
        self.nh, self.pool_drop, self.n_factors, self.seasonal_lags = n_heads, pool_drop, n_factors, seasonal_lags
        self.spd = cycle
        self.table = nn.Parameter(torch.zeros(cycle, N))  # periodic basis table
        self.pz = nn.Linear(L, d)
        self.mv, self.mo = nn.Linear(L, L), nn.Linear(L, L)  # pooling projections W_v, W_o
        self.mlp = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d), nn.GELU())
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d, H))
        self.sa = nn.Linear(L, H)  # linear autoregressive bypass
        nn.init.zeros_(self.sa.weight)
        nn.init.zeros_(self.sa.bias)
        if n_factors:
            self.fac_w = nn.Parameter(torch.full((n_factors, N), 1.0 / N))  # pooling weights Omega
            self.fac_l = nn.Parameter(torch.ones(N, n_factors))  # loadings Lambda
            self.fac_g = nn.Parameter(torch.zeros(n_factors))  # per-factor gates
        if seasonal_lags:
            self.sea_v = nn.Parameter(torch.zeros(N, seasonal_lags))
            self.sea_h = nn.Parameter(torch.ones(pred_len, seasonal_lags))
        self.full_x = None  # full normalized series [T, N]; the seasonal term only reads values before the forecast origin

    def init_factors(self, x_train):
        # leading eigenvectors of the training-segment correlation matrix
        x = (x_train - x_train.mean(0)) / (x_train.std(0) + 1e-8)
        _, v = np.linalg.eigh(np.nan_to_num(x.T @ x / len(x)))
        v = np.ascontiguousarray(v[:, ::-1][:, :self.n_factors])
        with torch.no_grad():
            self.fac_w.copy_(torch.tensor(v.T, dtype=torch.float32))
            self.fac_l.copy_(torch.tensor(v, dtype=torch.float32))

    def normalize(self, x):
        if self.scale == "raw":
            m0, s0 = x.mean(1), (x.var(1) + 1e-5).sqrt()
        else:
            m0, s0 = x[:, -1], torch.ones_like(x[:, -1])
        return ((x - m0[:, None]) / s0[:, None]).transpose(1, 2), m0, s0

    def forward(self, x, pos):
        # x: [B, L, N]; pos: [B], index of the first forecast step in the full series
        pos = pos.long()
        z, m0, s0 = self.normalize(x)  # z: [B, N, L]
        zin = z + self.table[(pos[:, None] + torch.arange(self.L, device=x.device)) % self.cyc].transpose(1, 2)
        B, n = z.shape[:2]
        mz = self.mv(z)
        A = F.dropout(torch.full((B, self.nh, n, n), 1.0 / n, device=x.device), p=self.pool_drop, training=self.training)
        pooled = torch.einsum("bhij,bjhd->bihd", A, mz.view(B, n, self.nh, -1)).reshape(B, n, -1)
        if self.n_factors:
            pooled = pooled + torch.einsum("ik,bkd->bid", self.fac_l * self.fac_g[None], torch.einsum("kj,bjd->bkd", self.fac_w, mz))
        zin = zin + self.mo(pooled)
        u = self.pz(zin)
        h = self.mlp(u) + u
        out = self.head(h) + self.sa(z)
        if self.seasonal_lags:
            k = torch.arange(self.H, device=x.device)
            lag = self.spd * ((k + self.spd) // self.spd)
            idx = (pos[:, None, None] + k[None, :, None] - lag[None, :, None]
                   - self.spd * torch.arange(self.seasonal_lags, device=x.device)).clamp_min(0)  # [B, H, M]
            vn = (self.full_x[idx] - m0[:, None, None]) / s0[:, None, None]
            out = out + torch.einsum("bhmn,nm,hm->bnh", vn, self.sea_v, self.sea_h)
        return (out * s0[..., None] + m0[..., None]).transpose(1, 2)  # [B, H, N]
