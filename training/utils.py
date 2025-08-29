import base64
from io import BytesIO
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch
from torchvision.transforms.functional import to_pil_image
from data.transforms import od_to_rgb


def _tensor_to_base64(img: torch.Tensor) -> str:
    """Convert a CHW tensor in [0,1] to a base64 PNG string."""
    buf = BytesIO()
    to_pil_image(img.clamp(0, 1)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def knockout_visualization(model, img: torch.Tensor) -> (List[str], List[str]):
    """Return base64 strings for input, reconstruction and per-channel knockouts."""
    model.eval()
    with torch.no_grad():
        recon, C = model(img)
        outs: List[torch.Tensor] = [img[0].cpu(), recon[0].cpu()]
        labels = ["Input", "Reconstruction"]
        for k in range(C.size(1)):
            C_ko = C.clone()
            C_ko[:, k] = 0
            out = model.decoder(C_ko)[0].cpu()
            outs.append(out)
            labels.append(f"KO {k+1}")
    b64 = [_tensor_to_base64(o) for o in outs]
    return b64, labels


def one_only_visualization(model, img: torch.Tensor) -> (List[str], List[str]):
    """Return base64 strings for input, reconstruction and per-channel single-stain images."""
    model.eval()
    with torch.no_grad():
        recon, C = model(img)
        outs: List[torch.Tensor] = [img[0].cpu(), recon[0].cpu()]
        labels = ["Input", "Reconstruction"]
        for k in range(C.size(1)):
            C_oo = torch.zeros_like(C)
            C_oo[:, k] = C[:, k]
            out = model.decoder(C_oo)[0].cpu()
            outs.append(out)
            labels.append(f"Only {k+1}")
    b64 = [_tensor_to_base64(o) for o in outs]
    return b64, labels


def hue_histogram(basis: torch.Tensor) -> str:
    """Return a histogram of the stain basis hues as a base64 PNG.

    The input ``basis`` is expected to be in optical density space. The vectors
    are first converted back to RGB so that the hue of each stain can be
    visualised. A histogram of these hue values is plotted and encoded as a
    base64 string so it can be embedded in HTML reports.
    """
    rgb = od_to_rgb(basis.detach().cpu()).clamp(0, 1).numpy()
    hsv = mcolors.rgb_to_hsv(rgb.reshape(-1, 1, 3)).reshape(rgb.shape)
    hue = hsv[:, 0]

    fig, ax = plt.subplots()
    ax.hist(hue, bins=20, range=(0, 1), color="orange")
    ax.set_xlabel("Hue value (0-1)")
    ax.set_ylabel("Number of basis vectors")
    ax.set_xlim(0, 1)
    buf = BytesIO()
    fig.savefig(buf, format="PNG")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def loss_curve_image(loss_history: List[float]) -> str:
    """Return base64-encoded loss curve plot."""
    fig, ax = plt.subplots()
    ax.plot(loss_history)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    buf = BytesIO()
    fig.savefig(buf, format="PNG")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


_basis_history: List[torch.Tensor] = []


def basis_swatches_html(basis: torch.Tensor, *, max_len: int = 100, height: int = 20) -> str:
    """Return HTML showing the evolution of each basis colour as a horizontal bar.

    The bars are stacked vertically in a column with the first basis at the top
    and the last one at the bottom.

    The function keeps an in-memory history of the basis vectors across calls.
    Each time it is invoked, the latest colours are appended to a per-basis bar
    image.  When a bar exceeds ``max_len`` pixels in length, the oldest colours
    are discarded so the rightmost part always corresponds to the most recent
    training step.
    """

    global _basis_history

    rgb = (od_to_rgb(basis.detach().cpu()).clamp(0, 1) * 255).to(torch.uint8)

    if not _basis_history or len(_basis_history) != len(rgb):
        _basis_history = [torch.empty(3, 0, dtype=torch.uint8) for _ in range(len(rgb))]

    scale = 4  # repeat pixels horizontally for better visibility
    blocks = []
    for i, color in enumerate(rgb):
        bar = torch.cat([_basis_history[i], color[:, None]], dim=1)
        if bar.shape[1] > max_len:
            bar = bar[:, -max_len:]
        _basis_history[i] = bar

        img = bar[:, None, :].repeat(1, height, 1)
        img = img.repeat_interleave(scale, dim=2).float() / 255.0
        b64 = _tensor_to_base64(img)
        width = bar.shape[1] * scale
        blocks.append(
            f"<div style='margin:2px;display:flex;align-items:center'>"
            f"<span style='margin-right:4px'>{i+1}</span>"
            f"<img src='data:image/png;base64,{b64}' width='{width}' height='{height}' style='image-rendering:pixelated'/></div>"
        )

    return "\n".join(blocks)


def update_html(
    path: str,
    knockout_b64: List[str],
    labels: List[str],
    hue_b64: str,
    basis_html: str = "",
    loss_b64: Optional[str] = None,
    one_only_b64: Optional[List[str]] = None,
    one_only_labels: Optional[List[str]] = None,
    mask_b64: Optional[str] = None,
) -> None:
    """Write a simple HTML page showing training visualisations.

    Parameters
    ----------
    path: str
        Output HTML file path.
    knockout_b64: List[str]
        Base64 encoded images for input/reconstruction and knockouts.
    labels: List[str]
        Labels corresponding to ``knockout_b64``.
    hue_b64: str
        Base64 encoded hue histogram from :func:`hue_histogram`.
    basis_html: str, optional
        HTML snippet showing the basis colours, typically produced by
        :func:`basis_swatches_html`.
    loss_b64: str, optional
        Base64 encoded loss curve image.
    one_only_b64: list of str, optional
        Base64 encoded images for input/reconstruction and single stain channels.
    one_only_labels: list of str, optional
        Labels corresponding to ``one_only_b64``.
    mask_b64: str, optional
        Base64 encoded visualisation of the colour-based mask.
    """
    loss_section = ""
    if loss_b64:
        loss_section = f"<h2>Training Loss</h2><img src='data:image/png;base64,{loss_b64}'/>"
    ko_html = "".join(
        f"<div style='display:inline-block;text-align:center;margin:4px'>"
        f"<img src='data:image/png;base64,{b64}'/><br/>{lab}</div>"
        for b64, lab in zip(knockout_b64, labels)
    )
    only_section = ""
    if one_only_b64:
        if one_only_labels is None:
            one_only_labels = labels
        oo_html = "".join(
            f"<div style='display:inline-block;text-align:center;margin:4px'>"
            f"<img src='data:image/png;base64,{b64}'/><br/>{lab}</div>"
            for b64, lab in zip(one_only_b64, one_only_labels)
        )
        only_section = f"<h2>Single Stain Only</h2>{oo_html}"
    mask_section = ""
    if mask_b64:
        mask_section = f"<h2>Brown Mask</h2><img src='data:image/png;base64,{mask_b64}'/>"
    html = f"""
    <html>
    <head><meta charset='utf-8'><title>Stain Visualisation</title></head>
    <body>
    <h1>Stain Separation Visualisation</h1>
    <h2>Knockout Reconstructions</h2>
    {ko_html}
    {only_section}
    <h2>Stain Basis Hue Histogram</h2>
    <img src='data:image/png;base64,{hue_b64}'/>
    <h2>Basis Colours</h2>
    {basis_html}
    {loss_section}
    {mask_section}
    </body>
    </html>
    """
    Path(path).write_text(html)
