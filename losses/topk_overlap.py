import torch


def topk_overlap_loss(C: torch.Tensor, top_p: float = 0.05) -> torch.Tensor:
    """Penalise overlap of top activations across stain channels.

    For each channel the ``top_p`` fraction of pixels with highest
    concentration values is selected. Locations that belong to this
    set for more than one channel contribute to the loss. The final
    value corresponds to the fraction of pixels in the batch that are
    among the strongest activations of multiple stains.
    """
    B, K, H, W = C.shape
    N = H * W
    flat = C.view(B, K, N)
    k = max(1, int(N * top_p))
    thresh, _ = flat.topk(k, dim=2)
    thresh = thresh[..., -1:].expand_as(flat)
    masks = (flat >= thresh).float()
    overlap = masks.sum(dim=1) - 1.0
    overlap = overlap.clamp(min=0)
    return overlap.mean() * (1.0 / top_p)
