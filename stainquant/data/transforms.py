import torch
import torchvision.transforms.functional as TF
from torchvision import transforms


def rgb_to_od(img: torch.Tensor) -> torch.Tensor:
    """Convert an RGB image in [0,1] to optical density."""
    img = img.clamp(min=1e-6)
    od = -torch.log(img)
    return od


def od_to_rgb(od: torch.Tensor) -> torch.Tensor:
    return torch.exp(-od)


class BasicAugment(transforms.Compose):
    def __init__(self):
        aug = [transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()]
        super().__init__(aug)

