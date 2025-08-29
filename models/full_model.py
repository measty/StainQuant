import torch
import torch.nn as nn

from .encoder import Encoder
from .stain_basis import StainBasis
from .decoder import Decoder


class StainSeparationModel(nn.Module):
    def __init__(
        self,
        K: int = 3,
        init_rgb: torch.Tensor = None,
        arch: str = "simple",
        encoder_kwargs: dict | None = None,
    ):
        super().__init__()
        encoder_kwargs = encoder_kwargs or {}
        self.encoder = Encoder(K=K, arch=arch, **encoder_kwargs)
        self.basis = StainBasis(K, init_rgb)
        self.decoder = Decoder(self.basis)

    def forward(self, x: torch.Tensor):
        C = self.encoder(x)
        recon = self.decoder(C)
        return recon, C
