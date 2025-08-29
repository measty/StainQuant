import torch
import torch.nn.functional as F


def orthogonality_loss(basis: torch.Tensor) -> torch.Tensor:
    """Encourage basis vectors to be mutually orthogonal.

    The current stain basis is first normalised and its Gram matrix computed.
    Deviation of this matrix from the identity is penalised using the mean
    absolute error. Conceptually this prevents different stains from collapsing
    onto similar directions in colour space, keeping them well separated and
    numerically stable.
    """

    norm_B = F.normalize(basis, dim=1)
    prod = torch.matmul(norm_B, norm_B.t())
    identity = torch.eye(basis.size(0), device=basis.device)
    diff = prod - identity
    return diff.abs().mean()
