from pathlib import Path
from typing import Optional, Sequence
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from PIL import Image
from strong_augment import StrongAugment, get_augment_space

from .transforms import rgb_to_od, BasicAugment


class DummyDataset(Dataset):
    def __init__(self, n_samples: int = 10, image_size: int = 32, augment: bool = False):
        self.n_samples = n_samples
        self.image_size = image_size
        self.augment = BasicAugment() if augment else transforms.Compose([])

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        img = torch.rand(3, self.image_size, self.image_size)
        img = self.augment(img)
        od = rgb_to_od(img)
        return img, od


class FolderDataset(Dataset):
    """Wrap torchvision ImageFolder to return (img, od)."""

    def __init__(self, root: str, augment: bool = False):
        tfms = [transforms.ToTensor()]
        if augment:
            tfms.append(BasicAugment())
        self.dataset = datasets.ImageFolder(root, transform=transforms.Compose(tfms))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, _ = self.dataset[idx]
        od = rgb_to_od(img)
        return img, od


class FlatFolderDataset(Dataset):
    """Dataset that loads all images under ``root`` without class subfolders."""

    def __init__(self, root: str, augment: bool = True, strong_aug: bool = True, extensions: Sequence[str] = datasets.folder.IMG_EXTENSIONS):
        self.root = Path(root)
        self.extensions = tuple(ext.lower() for ext in extensions)
        tfms = [transforms.ToTensor()]
        if augment:
            tfms.append(BasicAugment()) # fix so same augment is applied to both img and tar
        self.tar_transform = transforms.Compose(tfms)
        if strong_aug:
            default_augs = get_augment_space()
            default_augs.pop('solarize')
            tfms = [StrongAugment(operations=[0,2,3,4], probabilites=[0.3, 0.3, 0.3, 0.1], augment_space=default_augs)] + tfms
        self.transform = transforms.Compose(tfms)

        self.paths = [p for p in self.root.rglob("*") if p.is_file() and p.suffix.lower() in self.extensions]

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        orig = Image.open(path).convert("RGB")
        tar = self.tar_transform(orig)
        img = self.transform(orig)
        return img, tar


def create_loader(
    batch_size: int = 4,
    n_samples: int = 10,
    augment: bool = False,
    strong_aug: bool = True,
    data_dir: Optional[str] = None,
    use_class_folders: bool = False,
    num_workers: int = 2,
) -> DataLoader:
    if data_dir is None:
        ds = DummyDataset(n_samples=n_samples, augment=augment)
    else:
        if use_class_folders:
            ds = FolderDataset(data_dir, augment=augment)
        else:
            ds = FlatFolderDataset(data_dir, augment=augment, strong_aug=strong_aug)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
