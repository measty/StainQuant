import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import random
from tqdm import tqdm
from torch.optim import Adam
from pathlib import Path

from models.full_model import StainSeparationModel
from models.adversary import PermutationAdversary
from losses.reconstruction import reconstruction_loss
from losses.sparsity import sparsity_loss
from losses.activation_entropy import activation_entropy_loss
from losses.total_correlation import total_correlation_loss
from losses.colour_consistency import colour_consistency_loss
from losses.orthogonality import orthogonality_loss
from losses.permutation_adv import permutation_adversary_loss
from losses.balance import channel_entropy_loss
from losses.spikiness import spikiness_loss
from losses.topk_overlap import topk_overlap_loss
from losses.mask_loss import mask_loss
from models.colour_mask import ColourMask, ColourMaskRGB
from .utils import (
    knockout_visualization,
    one_only_visualization,
    hue_histogram,
    basis_swatches_html,
    update_html,
    loss_curve_image,
    _tensor_to_base64,
)

def add_watches(model):
    """
    Register lightweight hooks that print the first op whose activation or
    gradient contains NaN/Inf.  Safe for view‑producing layers; no cloning or
    grad modification involved.
    """
    def make_forward_hook(name):
        def fwd_hook(module, inputs, output):
            # ---------- watch the forward pass ----------
            tensors = output if isinstance(output, tuple) else (output,)
            for t in tensors:
                if isinstance(t, torch.Tensor):
                    if torch.isnan(t).any() or torch.isinf(t).any():
                        print(f'💥 NaN/Inf **activation** in {name}')
                        #raise SystemExit

                    # ---------- attach a grad hook to this tensor ----------
                    def grad_hook(grad, _name=name):
                        if torch.isnan(grad).any() or torch.isinf(grad).any():
                            print(f'💥 NaN/Inf **gradient** in {_name}')
                            #raise SystemExit
                    if t.requires_grad:
                        # Register the hook only if the tensor requires gradients
                        t.register_hook(grad_hook)
        return fwd_hook

    for n, m in model.named_modules():
        # Skip modules that do nothing (they’ll just duplicate prints)
        if n == '' or isinstance(m, (nn.Sequential, nn.ModuleList)):
            continue
        m.register_forward_hook(make_forward_hook(n))


class Trainer:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        trainer_cfg = cfg.get("trainer", {})
        model_cfg = cfg.get("model", {})

        K = model_cfg.get("K", cfg.get("K", 3))
        arch = model_cfg.get("arch", trainer_cfg.get("arch", "simple"))
        encoder_cfg = model_cfg.get("encoder", {})

        lr = float(trainer_cfg.get("lr", 1e-3))
        wd = float(trainer_cfg.get("weight_decay", 0.0))

        init = cfg.get("init_rgb")
        init_rgb = torch.tensor(init) if init is not None else None

        if trainer_cfg.get("use_unet"):
            arch = "unet"

        # store weights for each loss component
        self.loss_cfg = cfg.get("losses", {})
        self._weights = {}
        for key, d in self.loss_cfg.items():
            w = d.get("weight", 0.0)
            if isinstance(w, (list, tuple)) and len(w) == 2:
                start, final = float(w[0]), float(w[1])
            else:
                start = final = float(w)
            self._weights[key] = (start, final)
            d["weight"] = final

        spike_cfg = self.loss_cfg.get("spike", {})
        self.spike_channels = spike_cfg.get("channels")

        mask_cfg = self.loss_cfg.get("mask", {})
        init_list = cfg.get("init_rgb", [])
        default_rgb = [init_list[c] for c in mask_cfg.get("channel", [4])] if len(init_list) > max(mask_cfg.get("channel", [4])) else [[70, 40, 10]]
        mask_rgb = mask_cfg.get("rgb", default_rgb)
        thresh = mask_cfg.get("thresh", [0.25])
        self.mask_channels = mask_cfg.get("channel", [4])
        self.maskers = [ColourMask(mask_rgb[i], thresh[i]).to(self.device) for i in range(len(self.mask_channels))]

        self.warmup_iters = int(trainer_cfg.get("warmup_iters", 0))
        self.step = 0

        def weight_fn(key: str) -> float:
            start, final = self._weights.get(key, (0.0, 0.0))
            if self.warmup_iters <= 0:
                return final
            t = min(self.step, self.warmup_iters) / self.warmup_iters
            return start + (final - start) * t

        self.w = weight_fn

        self.model = StainSeparationModel(
            K=K,
            init_rgb=init_rgb,
            arch=arch,
            encoder_kwargs=encoder_cfg,
        ).to(self.device)

        # debugging:
        #add_watches(self.model)

        self.opt = Adam(self.model.parameters(), lr=lr, weight_decay=wd)
        if self.w("perm") > 0:
            self.adversary = PermutationAdversary(K=K).to(self.device)
            self.opt_D = Adam(self.adversary.parameters(), lr=lr, weight_decay=wd)

        self.C_history: list[torch.Tensor] = []
        self.history_size = int(trainer_cfg.get("adv_history", 10))

        # visualisation update frequency
        self.update_every = int(trainer_cfg.get("update_every", 0))
        self.loss_history = []

    def _check_for_nans(self):
        """Print names of parameters or gradients that contain NaNs."""
        for name, param in self.model.named_parameters():
            if torch.isnan(param).any():
                print(f"NaN in parameter: {name} at step {self.step}")
                return True
            if param.grad is not None and torch.isnan(param.grad).any():
                print(f"NaN in gradient of {name} at step {self.step}")
                return True
        return False
    
    def _check_huge_weights(self):
        """Print names of parameters or gradients that contain unusually large values."""
        for name, param in self.model.named_parameters():
            if torch.isinf(param).any() or (param.abs() > 1e5).any():
                print(f"Huge value in parameter: {name} at step {self.step}")
                return True
            if param.grad is not None and (torch.isinf(param.grad).any() or (param.grad.abs() > 1e5).any()):
                print(f"Huge value in gradient of {name} at step {self.step}")
                return True
        return False

    def train_step(self, batch):
        img, tar = batch
        img = img.to(self.device)
        tar = tar.to(self.device)
        recon, C = self.model(img)

        if self.C_history:
            hist_C = random.choice(self.C_history)
        else:
            hist_C = C.detach()

        if self.w("perm") > 0:
            d_loss, g_adv = permutation_adversary_loss(
                self.adversary, hist_C, C, tar, self.model.decoder
            )
        
        # rec = reconstruction_loss(tar, recon)
        # l1 = sparsity_loss(C) if self.w("l1") > 0 else C.new_tensor(0.0)
        # entropy_loss = activation_entropy_loss(C) if self.w("entropy") > 0 else C.new_tensor(0.0)
        # tc = total_correlation_loss(C) if self.w("tc") > 0 else C.new_tensor(0.0)
        # col = colour_consistency_loss(self.model.basis.basis, self.model.basis.init_rgb) if self.w("col") > 0 else C.new_tensor(0.0)
        # orth = orthogonality_loss(self.model.basis.basis) if self.w("orth") > 0 else C.new_tensor(0.0)
        # bal = channel_entropy_loss(C) if self.w("balance") > 0 else C.new_tensor(0.0)
        # spike = spikiness_loss(C, channels=self.spike_channels) if self.w("spike") > 0 else C.new_tensor(0.0)
        # overlap = topk_overlap_loss(C) if self.w("overlap") > 0 else C.new_tensor(0.0)
        if self.w("mask") > 0:
            m_loss = 0.0
            for i, masker in enumerate(self.maskers):
                m_loss += mask_loss(C, masker(tar), self.mask_channels[i])
        else:
            m_loss = C.new_tensor(0.0)

        loss = self.w("rec") * reconstruction_loss(tar, recon)
        if self.w("l1") > 0:
            loss += self.w("l1") * sparsity_loss(C)
        if self.w("entropy") > 0:
            loss += self.w("entropy") * activation_entropy_loss(C)
        if self.w("tc") > 0:
            loss += self.w("tc") * total_correlation_loss(C)
        if self.w("col") > 0:
            loss += self.w("col") * colour_consistency_loss(self.model.basis.basis, self.model.basis.init_rgb)
        if self.w("orth") > 0:
            loss += self.w("orth") * orthogonality_loss(self.model.basis.basis)
        if self.w("balance") > 0:
            loss += self.w("balance") * channel_entropy_loss(C)
        if self.w("spike") > 0:
            loss += self.w("spike") * spikiness_loss(C, channels=self.spike_channels)
        if self.w("overlap") > 0:
            loss += self.w("overlap") * topk_overlap_loss(C)
        if self.w("mask") > 0:
            loss += self.w("mask") * m_loss
            
        if self.w("perm") > 0:
            loss += self.w("perm") * g_adv

        # warning if loss nan
        if torch.isnan(loss).any():
            print(f"Loss is NaN at step {self.step}.")

        self.opt.zero_grad()
        with torch.autograd.set_detect_anomaly(True):
                loss.backward()
        #loss.backward()
        # clip large gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0, error_if_nonfinite=True)
        self.opt.step()
        #self._check_for_nans()
        #self._check_huge_weights()

        if self.w("perm") > 0:
            # update discriminator using fresh forward pass to avoid stale graphs
            d_loss, _ = permutation_adversary_loss(
                self.adversary, hist_C, C.detach(), tar.detach(), self.model.decoder
            )
            self.opt_D.zero_grad()
            (self.w("perm") * d_loss).backward()
            self.opt_D.step()

        self.C_history.append(C.detach())
        if len(self.C_history) > self.history_size:
            self.C_history.pop(0)

        self.step += 1
        return loss.item()

    def _random_sample(self, loader: DataLoader) -> torch.Tensor:
        ds = loader.dataset
        idx = random.randint(0, len(ds) - 1)
        img, tar = ds[idx]
        return img.unsqueeze(0).to(self.device), tar.unsqueeze(0).to(self.device)

    def fit(
        self,
        loader: DataLoader,
        epochs: int = 1,
        vis_path: str = "train_vis.html",
        ckpt_dir: str = "checkpoints",
    ):
        """Train the model and optionally save checkpoints."""
        self.model.train()
        self.step = 0
        ckpt_path = Path(ckpt_dir)
        ckpt_path.mkdir(parents=True, exist_ok=True)
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            for batch in tqdm(loader, desc="Training", leave=False):
                loss = self.train_step(batch)
                self.loss_history.append(loss)
                if self.update_every and self.step % self.update_every == 0:
                    sample_img, sample_tar = self._random_sample(loader)
                    ko_b64, labels = knockout_visualization(self.model, sample_img)
                    oo_b64, oo_labels = one_only_visualization(self.model, sample_img)
                    hist = hue_histogram(self.model.basis.basis)
                    swatch_html = basis_swatches_html(self.model.basis.basis)
                    lc = loss_curve_image(self.loss_history)
                    mask_b64 = None
                    if self.w("mask") > 0:
                        # Generate mask visualization
                        mask = [masker(sample_tar).repeat(1, 3, 1, 1) for masker in self.maskers]
                        # catenate along width dimension
                        mask = torch.cat(mask, dim=3)
                        mask_b64 = _tensor_to_base64(mask[0])
                    update_html(
                        vis_path,
                        ko_b64,
                        labels,
                        hist,
                        swatch_html,
                        lc,
                        oo_b64,
                        oo_labels,
                        mask_b64,
                    )
                    print(f"Updated visualisations saved to {vis_path}")

            # save checkpoint at the end of each epoch
            torch.save(
                self.model.state_dict(),
                ckpt_path / f"epoch_{epoch + 1}.pt",
            )

        # save final model
        torch.save(self.model.state_dict(), ckpt_path / "final.pt")

