import torch


def total_correlation_loss(C: torch.Tensor) -> torch.Tensor:
    """Discourage statistical dependence between stain channels.

    Total correlation, also known as multi-variate mutual information, measures
    how much information is shared among variables. Here we approximate it using
    the sum of absolute off-diagonal elements of the empirical covariance of the
    concentration maps. When this term is small, stain channels are linearly
    uncorrelated and therefore capture distinct features in the data.
    """

    N, K, H, W = C.shape
    flat = C.view(N, K, -1)
    mean = flat.mean(dim=2, keepdim=True)
    centered = flat - mean
    cov = torch.bmm(centered, centered.transpose(1, 2)) / flat.shape[2]
    off_diag = cov - torch.diag_embed(torch.diagonal(cov, dim1=1, dim2=2))
    return off_diag.abs().mean()

