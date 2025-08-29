import torch
import torch.nn.functional as F

from data.transforms import rgb_to_od


def colour_consistency_loss(basis: torch.Tensor, init_rgb: torch.Tensor) -> torch.Tensor:
    """Keep learned stain colours close to a desired reference.

    When rough RGB priors for the stains are provided, this loss penalises the
    L1 distance between the current basis vectors and those reference colours.
    Conceptually this anchors the basis in colour space, preventing arbitrary
    rotations that might otherwise occur when optimisation relies solely on
    reconstruction error. If no priors are given the loss is zero.
    """

    if init_rgb is None:
        return torch.tensor(0.0, device=basis.device)
    target_rgb = init_rgb.to(basis.device).float() / 255.0
    target_od = rgb_to_od(target_rgb).view(basis.shape)
    target_od = F.normalize(target_od, dim=1)
    return (basis - target_od).abs().mean()
