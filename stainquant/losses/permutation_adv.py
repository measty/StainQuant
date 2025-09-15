import torch
import torch.nn.functional as F


def permutation_adversary_loss(
    adversary,
    hist_C: torch.Tensor,
    C: torch.Tensor,
    img: torch.Tensor,
    decoder,
):
    """Channel-drop adversarial loss.

    ``adversary`` predicts which stain channels to keep based on a historical
    concentration map ``hist_C``. The predicted mask is then applied to the
    current concentration map ``C`` when reconstructing the input image. The
    adversary seeks a mask that still yields a good reconstruction, whereas the
    generator is penalised if such a reconstruction is possible.
    """

    keep = adversary(hist_C)
    keep = keep.view(keep.size(0), keep.size(1), 1, 1)

    recon_adv = decoder(C.detach() * keep)
    recon_whole = decoder(C)
    # rewrad the adversary for dropping stains that do not contribute to the reconstruction
    d_loss = F.l1_loss(recon_adv, recon_whole.detach())

    # reward the generator if all stain reconstruction is closer to the input image than masked
    # reconstruction
    recon_adv_det = decoder(C * keep.detach())
    g_loss = -F.l1_loss(recon_adv_det, img) + F.l1_loss(recon_whole, img)

    return d_loss, g_loss

