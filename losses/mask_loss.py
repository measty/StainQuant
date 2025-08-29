import torch


def mask_loss(C: torch.Tensor, mask: torch.Tensor, channel: int) -> torch.Tensor:
    """Penalty when masked pixels are not dominated by a specific channel."""
    if mask.sum() <= 10: # prob. not a useful mask
        return C.sum() * 0.0
    masked = C * mask.unsqueeze(1)
    total = masked.sum(dim=1)
    ch = masked[:, channel]
    ratio = ch / (total + 1e-8)
    inds = [i for i in range(C.shape[1]) if i != channel]
    return (1.0 - ratio).mean() #+ 0.5 * masked[:, inds].mean()
    #return masked[masked > 0].mean()  # penalise all non-zero concentrations from other channels in masked pixels
