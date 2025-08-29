from torch.optim.lr_scheduler import CosineAnnealingLR


def create_scheduler(optimizer, max_epochs):
    return CosineAnnealingLR(optimizer, T_max=max_epochs)
