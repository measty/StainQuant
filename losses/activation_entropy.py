import torch


def activation_entropy_loss(C: torch.Tensor) -> torch.Tensor:
    """Encourage per-pixel concentration distributions to be nearly one-hot.

    Concentrations are normalised across stain channels for each pixel to
    obtain a categorical distribution. The Shannon entropy of this
    distribution is minimised so that one channel dominates most pixels.
    """
    csum = (C.sum(dim=1, keepdim=True) + 1e-8)
    P = C / csum
    entropy = -(P * torch.log(P + 1e-8)).sum(dim=1, keepdim=True)
    entropy = entropy[csum > 0.01]  # ignore pixels with almost zero concentration
    return entropy.mean() if entropy.numel() > 0 else C.new_tensor(0.0)
