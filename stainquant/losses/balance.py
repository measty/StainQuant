import torch


def channel_entropy_loss(C: torch.Tensor) -> torch.Tensor:
    """Promote balanced utilisation of the available stain channels.

    The mean activation of each channel over the batch is treated as a
    categorical distribution ``p``. Its Shannon entropy ``H(p)`` is maximised,
    or equivalently ``-H(p)`` is minimised, so that all channels are used
    roughly equally often. This counteracts collapse where only a subset of
    stains are ever active.
    """

    mean_act = C.mean(dim=(0, 2, 3))
    p = mean_act / (mean_act.sum() + 1e-8)
    entropy = -(p * torch.log(p + 1e-8)).sum()
    return -entropy

