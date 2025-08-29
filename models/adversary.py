import torch
import torch.nn as nn


class PermutationAdversary(nn.Module):
    """Adversary predicting which stain channels to drop."""

    def __init__(self, K: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(K, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, K),
            nn.Sigmoid(),
        )

    def forward(self, C: torch.Tensor) -> torch.Tensor:
        """Return drop probabilities for each channel."""
        return self.net(C)
