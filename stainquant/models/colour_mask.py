import torch
import torch.nn as nn
from torchvision.transforms import GaussianBlur

def _rgb_to_hsv(img: torch.Tensor) -> torch.Tensor:
    """Convert an RGB image in ``[0, 1]`` to HSV."""
    r, g, b = img.unbind(1)
    maxc = img.amax(dim=1)
    minc = img.amin(dim=1)
    deltac = maxc - minc

    s = torch.where(maxc == 0, torch.zeros_like(deltac), deltac / (maxc + 1e-8))

    h = torch.zeros_like(maxc)
    mask = deltac != 0
    h = torch.where(mask & (r == maxc), (g - b) / (deltac + 1e-8), h)
    h = torch.where(mask & (g == maxc), 2.0 + (b - r) / (deltac + 1e-8), h)
    h = torch.where(mask & (b == maxc), 4.0 + (r - g) / (deltac + 1e-8), h)
    h = (h / 6.0) % 1.0

    hsv = torch.stack((h, s, maxc), dim=1)
    return hsv

class ColourMask(nn.Module):
    """Binary mask for pixels near a target RGB colour based on hue."""

    def __init__(self, target_rgb, threshold: float = 0.05, *, sat_thresh: float = 0.55, val_thresh: float = 0.7):
        super().__init__()
        rgb = (
            torch.tensor(target_rgb, dtype=torch.float32).view(1, 3, 1, 1) / 255.0
        )
        self.register_buffer("target_hue", _rgb_to_hsv(rgb)[:, 0])
        self.threshold = float(threshold)
        self.sat_thresh = float(sat_thresh)
        self.val_thresh = float(val_thresh)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """Return a mask of pixels within ``threshold`` hue distance of the target, softened by a slight
        gaussian blur."""
        hsv = _rgb_to_hsv(img)
        hue, sat, val = hsv[:, 0], hsv[:, 1], hsv[:, 2]
        diff = torch.abs(hue - self.target_hue)
        diff = torch.minimum(diff, 1.0 - diff)
        mask = (diff <= self.threshold) & (sat >= self.sat_thresh) & (val <= self.val_thresh)
        # Apply a small Gaussian blur to the mask
        mask = mask.float()
        mask = GaussianBlur(kernel_size=3, sigma=(0.1, 0.4))(mask.unsqueeze(1)).squeeze(1)
        return mask
    
class ColourMaskRGB(nn.Module):
    """Binary mask for pixels near a target RGB colour."""

    def __init__(self, target_rgb, threshold=0.25):
        super().__init__()
        rgb = torch.tensor(target_rgb, dtype=torch.float32) / 255.0
        self.register_buffer("target", rgb.view(1, 3, 1, 1))
        self.threshold = float(threshold)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """Return a mask of pixels within ``threshold`` of ``target`` colour."""
        diff = torch.norm(img - self.target, dim=1)
        mask = (diff <= self.threshold).float()
        return mask
