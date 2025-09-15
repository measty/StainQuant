from typing import List

import torch
import yaml
import gradio as gr
from torchvision import transforms
from PIL import Image
from pathlib import Path
from importlib import resources

from stainquant.models.full_model import StainSeparationModel
from stainquant.training.utils import knockout_visualization, one_only_visualization


def _default_config_path() -> str:
    try:
        return str(resources.files("stainquant.configs").joinpath("default.yaml"))
    except Exception:
        return str(Path(__file__).resolve().parents[1] / "configs" / "default.yaml")


def load_model(ckpt_path: str, config_path: str | None = None):
    """Load model checkpoint and configuration."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if config_path is None:
        config_path = _default_config_path()
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    model_cfg = cfg.get("model", {})
    K = model_cfg.get("K", cfg.get("K", 3))
    arch = model_cfg.get("arch", "simple")
    encoder_cfg = model_cfg.get("encoder", {})
    init = cfg.get("init_rgb")
    init_rgb = torch.tensor(init) if init is not None else None
    model = StainSeparationModel(K=K, init_rgb=init_rgb, arch=arch, encoder_kwargs=encoder_cfg).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, device


def run_inference(model: StainSeparationModel, device: torch.device, image: Image.Image) -> List[Image.Image]:
    """Return reconstruction, knockout and single-channel images."""
    to_tensor = transforms.ToTensor()
    img = to_tensor(image).unsqueeze(0).to(device)

    ko_b64, ko_labels = knockout_visualization(model, img)
    oo_b64, oo_labels = one_only_visualization(model, img)

    # decode base64 strings back to PIL images for gradio
    def b64_to_pil(b64: str) -> Image.Image:
        import base64
        from io import BytesIO
        buf = BytesIO(base64.b64decode(b64))
        return Image.open(buf)

    ko_imgs = [b64_to_pil(b) for b in ko_b64]
    oo_imgs = [b64_to_pil(b) for b in oo_b64]

    # arrange outputs: first pair of input and reconstruction already present in both
    return ko_labels, ko_imgs, oo_labels, oo_imgs


def build_interface(model: StainSeparationModel, device: torch.device) -> gr.Interface:
    """Create the Gradio interface for stain separation."""

    def separate_stains(image: Image.Image):
        """Run stain separation on ``image`` using the loaded model.

        Parameters
        ----------
        image : PIL.Image
            Input RGB image.

        Returns
        -------
        tuple[list[tuple[str, PIL.Image]], list[tuple[str, PIL.Image]]]
            ``(knockout, single)`` where each item is a list of ``(label, image)``
            pairs visualising the effect of removing or isolating each stain.
        """

        ko_labels, ko_imgs, oo_labels, oo_imgs = run_inference(model, device, image)
        return list(zip(ko_imgs, ko_labels)), list(zip(oo_imgs, oo_labels))

    return gr.Interface(
        fn=separate_stains,
        inputs=gr.Image(type="pil", label="Input Image"),
        outputs=[
            gr.Gallery(label="Knockout Reconstructions"),
            gr.Gallery(label="Single Channel Only"),
        ],
        title="Stain Separation Inference",
        description=(
            "Upload a histology patch to see reconstructions with each stain removed "
            "and in isolation."
        ),
    )


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    model, device = load_model(args.ckpt, args.config)
    demo = build_interface(model, device)
    demo.launch(mcp_server=True)


if __name__ == "__main__":
    main()

