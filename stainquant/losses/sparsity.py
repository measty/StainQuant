import torch
import torch.nn.functional as F


def sparsity_loss(C: torch.Tensor, power=0.5) -> torch.Tensor:
    """Encourage each pixel to be dominated by as few stains as possible.

    Concentration values are sorted per pixel and multiplied with increasing
    quadratic weights. This approximates a soft ``\\ell_0`` penalty – the first
    (largest) channel in a pixel is barely penalised while subsequent channels
    contribute progressively more. By minimising the weighted sum the model is
    nudged toward representations where most pixels activate only a single stain
    channel.
    """
    # minimise geometric mean stain concentration over C
    return C.pow(power).mean()
    # sort concentrations descending per pixel
    # sorted_C, _ = torch.sort(C, dim=1, descending=True)
    # K = C.size(1)

    # # quadratic weights emphasising lower ranked stains
    # weights = torch.linspace(0.0, 1.0, steps=K, device=C.device, dtype=C.dtype)
    # weights = weights.pow(2).view(1, K, 1, 1)
    # weights[0, 2, 0, 0] = weights[0, 1, 0, 0]
    # weights[0, 1, 0, 0] = 0.0 # only penalize mixing of > 2 stains

    # loss = (sorted_C * weights).sum(dim=1).mean()
    # return loss

