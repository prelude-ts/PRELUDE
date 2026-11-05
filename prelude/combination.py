import torch


class CombinationWeight:
    """Target-wise combination weight solved in closed form on the validation set (Eq. 28):
    alpha_i = clip(sum <Y_i - Y^s_i, Delta_i> / sum ||Delta_i||^2, 0, 1)."""

    def __init__(self):
        self.num, self.den = 0.0, 0.0

    def update(self, y, y_base, delta):
        # y, y_base, delta: [B, H, N]
        d = delta.double()
        self.num = self.num + ((y - y_base).double() * d).sum((0, 1))
        self.den = self.den + (d ** 2).sum((0, 1))

    def compute(self):
        return (self.num / self.den.clamp_min(1e-12)).clamp(0, 1).float()  # [N]
