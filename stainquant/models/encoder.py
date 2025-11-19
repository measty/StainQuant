import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_norm(norm: str | None, channels: int) -> nn.Module:
    if not norm or norm == "none":
        return nn.Identity()
    norm = norm.lower()
    if norm == "batchnorm":
        return nn.BatchNorm2d(channels)
    if norm == "instancenorm":
        # approximate instance norm with group norm using many groups
        groups = min(channels, 32)
        return nn.GroupNorm(groups, channels)
    if norm == "layernorm":
        return nn.GroupNorm(1, channels)
    raise ValueError(f"Unknown norm type: {norm}")


class _ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, norm: str | None = None, residual=False, dropout=0.0):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, 3, padding=1, padding_mode="reflect")]
        norm_layer = _get_norm(norm, out_ch)
        if not isinstance(norm_layer, nn.Identity):
            layers.append(norm_layer)
        layers.append(nn.ReLU(inplace=True))
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        layers.append(nn.Conv2d(out_ch, out_ch, 3, padding=1, padding_mode="reflect"))
        norm_layer2 = _get_norm(norm, out_ch)
        if not isinstance(norm_layer2, nn.Identity):
            layers.append(norm_layer2)
        layers.append(nn.ReLU(inplace=True))
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)
        self.residual = residual
        if residual and in_ch != out_ch:
            self.res_conv = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.res_conv = nn.Identity()

    def forward(self, x):
        out = self.block(x)
        if self.residual:
            out = out + self.res_conv(x)
        return out


class CrossStainAttention(nn.Module):
    """Simple squeeze–excitation style attention across stain channels."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.attn = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.avg_pool(x)
        weights = self.attn(weights)
        return x * weights
    
class CrossStainAttentionV2(nn.Module):
    """
    Flexible cross-stain attention block.

    Parameters
    ----------
    channels : int
        Number of input channels (and output channels - the block is identity-shaped).
    reduction : int | None, default 4
        Bottleneck reduction ratio.  None → auto: max(channels // 16, 1).
    pooling : {'avg', 'avg_std', 'avg_max'}, default 'avg'
        Global descriptor type.
          • 'avg'      - classic SE (mean only).  
          • 'avg_std'  - mean + (std + ε)½  (captures 1st & 2nd-order stats).  
          • 'avg_max'  - mean + max (robust to outliers).
    activation : {'relu', 'gelu', 'silu'}, default 'relu'
        Non-linearity in the bottleneck MLP.
    gating : {'sigmoid', 'softmax'}, default 'sigmoid'
        Channel gating function.  Softmax makes channels compete (∑c w_c = 1).
    spatial_attention : bool, default False
        Add CBAM-style spatial attention after channel attention.
    residual : bool, default False
        Return `x * attn + x` instead of `x * attn`.
    eps : float, default 1e-6
        Numerical stabiliser for std pooling.
    bias : bool, default False
        Use bias terms in the 1 x 1 convs.
    """

    def __init__(
        self,
        channels: int,
        reduction: int | None = 1,
        pooling: str = "avg",
        activation: str = "relu",
        gating: str = "sigmoid",
        spatial_attention: bool = False,
        residual: bool = False,
        eps: float = 1e-6,
        bias: bool = False,
    ):
        super().__init__()
        assert pooling in {"avg", "avg_std", "avg_max"}
        assert activation in {"relu", "gelu", "silu"}
        assert gating in {"sigmoid", "softmax"}

        self.channels = channels
        self.pooling = pooling
        self.gating = gating
        self.spatial_attention = spatial_attention
        self.residual = residual
        self.eps = eps

        # ---- global pooling(s) -------------------------------------------------
        self.gap = nn.AdaptiveAvgPool2d(1)
        if pooling == "avg_max":
            self.gmp = nn.AdaptiveMaxPool2d(1)  # max pool needed
        # 'avg_std' uses the same GAP result for µ; σ is computed on‑the‑fly

        # ---- channel‑wise excitation path --------------------------------------
        in_feats = channels * (1 if pooling == "avg" else 2)
        if reduction is None:
            reduction = max(channels // 16, 1)
        hidden = max(in_feats // reduction, 1)

        self.fc1 = nn.Conv2d(in_feats, hidden, kernel_size=1, bias=bias)
        self.act = {"relu": nn.ReLU(inplace=True),
                    "gelu": nn.GELU(),
                    "silu": nn.SiLU(inplace=True)}[activation]
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1, bias=bias)

        self.channel_gate = nn.Sigmoid() if gating == "sigmoid" else nn.Identity()  # softmax applied in forward

        # ---- optional spatial attention path -----------------------------------
        if spatial_attention:
            # CBAM: 2‑ch (avg & max) -> 1‑ch mask
            self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
            self.spatial_gate = nn.Sigmoid()

    # --------------------------------------------------------------------------
    def _make_descriptor(self, x: torch.Tensor) -> torch.Tensor:
        """Return a (B, D=C or 2C, 1, 1) global descriptor."""
        mu = self.gap(x)  # (B, C, 1, 1)

        if self.pooling == "avg":
            return mu

        if self.pooling == "avg_std":
            # unbiased=False for efficiency
            sigma = torch.sqrt(
                (x.var(dim=(2, 3), keepdim=True, unbiased=False) + self.eps)
            )
            return torch.cat([mu, sigma], dim=1)

        if self.pooling == "avg_max":  # mean + max
            mx = self.gmp(x)
            return torch.cat([mu, mx], dim=1)

        raise RuntimeError("Unknown pooling mode")

    # --------------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ---- channel attention -------------------------------------------------
        desc = self._make_descriptor(x)              # B × D × 1 × 1
        w = self.fc2(self.act(self.fc1(desc)))       # B × C × 1 × 1

        if self.gating == "sigmoid":
            w = self.channel_gate(w)
        else:  # softmax along channel dim
            w = F.softmax(w.squeeze(-1).squeeze(-1), dim=1).unsqueeze(-1).unsqueeze(-1)

        out = x * w                                  # broadcast

        # ---- spatial attention -------------------------------------------------
        if self.spatial_attention:
            avg_pool = torch.mean(out, dim=1, keepdim=True)
            max_pool, _ = torch.max(out, dim=1, keepdim=True)
            s = self.spatial_gate(self.spatial_conv(torch.cat([avg_pool, max_pool], dim=1)))
            out = out * s

        # ---- residual mix ------------------------------------------------------
        if self.residual:
            out = out + x

        return out


class Encoder(nn.Module):
    """Configurable convolutional encoder producing concentration maps."""

    def __init__(
        self,
        in_ch: int = 3,
        K: int = 3,
        arch: str = "simple",
        base_channels: int = 16,
        depth: int = 2,
        norm: str | None = "layernorm",
        residual: bool = False,
        dropout: float = 0.0,
        cross_corr: bool = False,
    ):
        super().__init__()
        self.arch = arch
        self.norm = norm
        self.residual = residual
        self.dropout = dropout
        self.cross_corr = cross_corr

        if arch == "unet":
            # encoder path
            self.enc_blocks = nn.ModuleList()
            self.pools = nn.ModuleList()
            ch = in_ch
            out_ch = base_channels
            for _ in range(depth):
                self.enc_blocks.append(
                    _ConvBlock(ch, out_ch, self.norm, residual, dropout)
                )
                self.pools.append(nn.MaxPool2d(2))
                ch = out_ch
                out_ch *= 2

            self.bottleneck = _ConvBlock(ch, out_ch, self.norm, residual, dropout)

            # decoder path
            self.upconvs = nn.ModuleList()
            self.dec_blocks = nn.ModuleList()
            for _ in range(depth):
                self.upconvs.append(nn.ConvTranspose2d(out_ch, out_ch // 2, 2, stride=2))
                self.dec_blocks.append(
                    _ConvBlock(out_ch, out_ch // 2, self.norm, residual, dropout)
                )
                out_ch //= 2

            final_layers = [nn.Conv2d(base_channels, K, 1)]
            if cross_corr:
                final_layers.append(CrossStainAttention(K))
            final_layers.append(nn.ReLU())
            self.final = nn.Sequential(*final_layers)
        else:
            layers = []
            ch = in_ch
            for _ in range(max(1, depth)):
                layers.append(_ConvBlock(ch, base_channels, self.norm, residual, dropout))
                ch = base_channels
            layers.append(nn.Conv2d(ch, K, 1))
            if cross_corr:
                layers.append(CrossStainAttention(K))
            layers.append(nn.ReLU())
            self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.arch == "unet":
            skips = []
            out = x
            for enc, pool in zip(self.enc_blocks, self.pools):
                out = enc(out)
                skips.append(out)
                out = pool(out)
            out = self.bottleneck(out)
            for up, dec, skip in zip(self.upconvs, self.dec_blocks, reversed(skips)):
                out = up(out)
                out = torch.cat([out, skip], dim=1)
                out = dec(out)
            return self.final(out)
        else:
            return self.net(x)
