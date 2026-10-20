"""Compact MNIST UNets for denoising and flow matching."""
from __future__ import annotations

import torch
from torch import Tensor, nn


def _block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.GELU(),
        nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.GELU(),
    )


class _UNetCore(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.enc1 = _block(1, dim)
        self.down1 = nn.Sequential(nn.AvgPool2d(2), _block(dim, dim * 2))
        self.down2 = nn.Sequential(nn.AvgPool2d(2), _block(dim * 2, dim * 4))
        self.up1 = nn.ConvTranspose2d(dim * 4, dim * 2, 2, stride=2)
        self.dec1 = _block(dim * 4, dim * 2)
        self.up2 = nn.ConvTranspose2d(dim * 2, dim, 2, stride=2)
        self.dec2 = _block(dim * 2, dim)
        self.out = nn.Conv2d(dim, 1, 1)

    def forward(self, x: Tensor, mod1: Tensor | None = None, mod2: Tensor | None = None,
                shift1: Tensor | None = None, shift2: Tensor | None = None) -> Tensor:
        e1 = self.enc1(x)                    # B,D,28,28
        e2 = self.down1(e1)                  # B,2D,14,14
        z = self.down2(e2)                   # B,4D,7,7
        z = self.up1(z)                      # B,2D,14,14
        if mod1 is not None:
            z = z * mod1
        if shift1 is not None:
            z = z + shift1
        z = self.dec1(torch.cat([z, e2], dim=1))
        z = self.up2(z)                      # B,D,28,28
        if mod2 is not None:
            z = z * mod2
        if shift2 is not None:
            z = z + shift2
        z = self.dec2(torch.cat([z, e1], dim=1))
        return self.out(z)


class SingleStepDenoiser(nn.Module):
    """Unconditional 28×28 denoiser, trained with additive Gaussian noise."""
    def __init__(self, dim: int = 128) -> None:
        super().__init__()
        self.net = _UNetCore(dim)

    def forward(self, noisy: Tensor) -> Tensor:
        return self.net(noisy)


class MNISTUNet(nn.Module):
    """Time- and optionally class-conditioned MNIST velocity predictor."""
    def __init__(self, dim: int = 64, num_classes: int = 10, class_conditioned: bool = False) -> None:
        super().__init__()
        self.class_conditioned = class_conditioned
        self.net = _UNetCore(dim)
        self.t1 = nn.Sequential(nn.Linear(1, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim * 2))
        self.t2 = nn.Sequential(nn.Linear(1, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))
        if class_conditioned:
            self.c1 = nn.Sequential(nn.Linear(num_classes, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim * 2))
            self.c2 = nn.Sequential(nn.Linear(num_classes, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))

    def forward(self, x: Tensor, t: Tensor, labels: Tensor | None = None) -> Tensor:
        t = t.to(device=x.device, dtype=x.dtype).reshape(-1, 1).clamp(0, 1)
        # Broadcast channel-wise modulation across spatial dimensions.
        t1 = 1 + self.t1(t).unsqueeze(-1).unsqueeze(-1)
        t2 = 1 + self.t2(t).unsqueeze(-1).unsqueeze(-1)
        if self.class_conditioned:
            if labels is None:
                labels = torch.full((x.shape[0],), -1, device=x.device, dtype=torch.long)
            c = torch.zeros(x.shape[0], 10, device=x.device, dtype=x.dtype)
            valid = labels >= 0
            if valid.any():
                c[valid] = torch.nn.functional.one_hot(labels[valid], 10).to(x.dtype)
            c1 = self.c1(c).unsqueeze(-1).unsqueeze(-1)
            c2 = self.c2(c).unsqueeze(-1).unsqueeze(-1)
            t1 = (t1 - 1)  # assignment modulation: c_scale * activation + t_shift
            t2 = (t2 - 1)
            return self.net(x, c1, c2, t1, t2)
        return self.net(x, t1, t2)
