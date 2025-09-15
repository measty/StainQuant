import torch
import torch.nn as nn
import torch.nn.functional as F

from stainquant.data.transforms import rgb_to_od


class StainBasis(nn.Module):
    """Global learnable stain dictionary."""

    def __init__(self, K: int, init_rgb: torch.Tensor = None):
        super().__init__()
        if init_rgb is not None:
            rgb = init_rgb.float().view(K, 3) / 255.0
        else:
            rgb = torch.rand(K, 3)

        od = rgb_to_od(rgb)

        self.register_buffer("init_rgb", rgb.clone())
        self._basis = nn.Parameter(od)

    def forward(self) -> torch.Tensor:
        return F.normalize(self._basis, dim=1)

    @property
    def basis(self) -> torch.Tensor:
        return self.forward()
