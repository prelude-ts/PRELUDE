import torch
import torch.nn as nn

from .candidates import prior_candidates
from .prewhiten import Prewhitening, fit_prewhitening
from .readout import TransferReadout
from .routing import SourceLagRouting
from .scp import SharedComponentPathway


class PRELUDE(nn.Module):
    """Separated reading: Y = SCP(X) + alpha * ISP(Prewhiten(X)) (Eq. 18)."""

    def __init__(self, n_vars, seq_len, pred_len, cycle, d_model, dropout, n_factors=0, seasonal_lags=0, scale="raw",
                 order=24, n_cand=16, max_lag=72, window=24, ratio=0.25, one_step_w=0.1, ratio_w=0.1):
        super().__init__()
        self.N, self.L, self.H, self.cyc = n_vars, seq_len, pred_len, cycle
        self.order, self.n_cand, self.max_lag = order, n_cand, max_lag
        self.scp = SharedComponentPathway(n_vars, seq_len, pred_len, cycle, d_model, dropout, n_factors, seasonal_lags, scale)
        self.prewhiten = Prewhitening(n_vars, seq_len, cycle, order, one_step_w)
        self.routing = SourceLagRouting(n_vars, seq_len - order, n_cand, max_lag, window, ratio=ratio, ratio_w=ratio_w)
        self.readout = TransferReadout(n_vars, pred_len, cycle, n_cand, max_lag, window)
        self.scp_only = False  # first training stage: base forecast only

    def fit_statistics(self, x_train, full_x=None):
        """x_train: normalized training segment [T, N] (float64 numpy); full_x: normalized full series for the seasonal term."""
        stats, eps = fit_prewhitening(x_train, self.cyc, self.order)
        self.prewhiten.set_statistics(**stats)
        self.routing.set_candidates(*prior_candidates(eps, self.n_cand, self.max_lag))
        if self.scp.n_factors:
            self.scp.init_factors(x_train)
        if full_x is not None:
            self.scp.full_x = torch.as_tensor(full_x, dtype=torch.float32, device=self.prewhiten.us.device)

    def forward(self, x, pos):
        """x: [B, L, N]; pos: [B] index of the first forecast step. Returns base forecast and correction, both [B, H, N]."""
        y_base = self.scp(x, pos)
        u = self.prewhiten(x, pos)
        w = self.routing(u)
        delta = self.readout(u, w, self.routing.cand, self.prewhiten.us, pos, self.H)
        if self.scp_only:
            delta = delta * 0
        return y_base, delta

    def aux_loss(self):
        """One-step prediction loss of the innovations and retention-ratio constraint (Eq. 29)."""
        return self.prewhiten.aux_loss + self.routing.aux_loss

    def base_parameters(self):
        return list(self.scp.parameters())

    def predict(self, x, pos, alpha=None):
        y_base, delta = self(x, pos)
        return y_base + (delta if alpha is None else alpha * delta)
