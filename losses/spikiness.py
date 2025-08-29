import torch


def spikiness_loss(C: torch.Tensor, channels: list[int] | None = None) -> torch.Tensor:
    """Encourage sparse stain channels to form sharp peaks.

    For each image and selected channel the spatial kurtosis of the
    concentration distribution is computed. High kurtosis indicates that
    only a few pixels carry most of the mass -- matching the expected
    behaviour of sparse stains such as CD8. The loss minimises the
    reciprocal of this kurtosis so that spikier activations yield a
    smaller penalty.

    Parameters
    ----------
    C: torch.Tensor
        Concentration maps of shape ``(B, K, H, W)``.
    channels: list[int] | None, optional
        Indices of channels to apply the penalty to. ``None`` means all
        channels are considered.
    """

    if channels is not None:
        C = C[:, channels]

    mu = C.mean(dim=(2, 3), keepdim=True)
    diff = C - mu
    var = (diff ** 2).mean(dim=(2, 3)) + 1e-8
    fourth = (diff ** 4).mean(dim=(2, 3))
    kurt = fourth / (var ** 2)
    return (1.0 / (kurt + 1e-8)).mean()
