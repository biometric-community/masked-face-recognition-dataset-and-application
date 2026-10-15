"""Simple lower-face mask overlay (paper Fig. 3 template substitute)."""

from __future__ import annotations

import torch


def apply_mask_aug(
    images: torch.Tensor,
    prob: float = 0.5,
    y0_frac: float = 0.52,
    color: float = 0.85,
) -> torch.Tensor:
    """Occlude the lower face with a solid rectangle (N,C,H,W in [-1,1] or [0,1])."""
    if prob <= 0:
        return images
    n, _c, h, w = images.shape
    device = images.device
    mask = (torch.rand(n, device=device) < prob).view(n, 1, 1, 1)
    y0 = int(h * y0_frac)
    out = images.clone()
    # map color into same value range as normalized faces (~[-1,1] after Normalize)
    fill = torch.full((1, 1, 1, 1), color * 2.0 - 1.0, device=device, dtype=images.dtype)
    out[:, :, y0:, :] = torch.where(mask, fill.expand_as(out[:, :, y0:, :]), out[:, :, y0:, :])
    return out
