import torch
import torch.nn as nn
from .stain_basis import StainBasis


class Decoder(nn.Module):
    def __init__(self, basis: StainBasis):
        super().__init__()
        self.basis_module = basis

    @property
    def K(self):
        return self.basis_module.basis.shape[0]

    def forward(self, C: torch.Tensor) -> torch.Tensor:
        """Mix concentrations with basis and convert to RGB."""
        basis = self.basis_module.basis  # (K,3)
        B = basis.view(1, self.K, 3, 1, 1)
        # C: (N,K,H,W)
        od = (C.unsqueeze(2) * B).sum(1)  # (N,3,H,W)
        rgb = torch.exp(-od)
        return rgb
