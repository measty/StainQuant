import torch
import torch.nn.functional as F
from torchvision import models

_vgg = None


def _get_vgg() -> torch.nn.Module:
    """Load VGG19 network for perceptual features (cached)."""
    global _vgg
    if _vgg is None:
        vgg = models.vgg19(weights=None).features.eval()
        for p in vgg.parameters():
            p.requires_grad = False
        _vgg = vgg
    return _vgg


_feat_layers = [2, 7, 16, 25]


def _vgg_features(x: torch.Tensor) -> list[torch.Tensor]:
    """Extract perceptual features from VGG19."""
    vgg = _get_vgg().to(x.device)
    if x.shape[-1] < 64 or x.shape[-2] < 64:
        x = F.interpolate(x, size=(64, 64), mode="bilinear", align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
    x = (x - mean) / std

    feats = []
    out = x
    for i, layer in enumerate(vgg):
        out = layer(out)
        if i in _feat_layers:
            feats.append(out)
        if i >= max(_feat_layers):
            break
    return feats


def reconstruction_loss(img: torch.Tensor, recon: torch.Tensor) -> torch.Tensor:
    """Compute reconstruction loss encouraging fidelity to the input image.

    This loss is the sum of two terms: an L1 distance in pixel space and a
    perceptual distance based on VGG19 features. Minimising the L1 term ensures
    that the reconstructed image matches the exact optical density values of the
    target. The perceptual term compares features extracted from a fixed VGG19
    network, capturing higher-level structure. Both terms correspond to the
    ``L_1`` norm of their respective feature representations, making the overall
    objective differentiable and favouring accurate, visually consistent
    reconstructions.
    """

    l1 = F.l1_loss(recon, img)

    img_feats = _vgg_features(img)
    recon_feats = _vgg_features(recon)
    perc = sum(F.l1_loss(rf, tf) for rf, tf in zip(recon_feats, img_feats)) / len(img_feats)

    return l1 + 2 * perc
