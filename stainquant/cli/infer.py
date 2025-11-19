from pathlib import Path
import torch
import yaml
from torchvision.utils import save_image
from tqdm import tqdm
from importlib import resources

from stainquant.models.full_model import StainSeparationModel
from stainquant.data.loader import create_loader


def _default_config_path() -> str:
    try:
        return str(resources.files("stainquant.configs").joinpath("default.yaml"))
    except Exception:
        return str(Path(__file__).resolve().parents[1] / "configs" / "default.yaml")


def main(
    ckpt_path: str,
    config_path: str | None = None,
    out_dir: str = "out",
    num_examples: int | None = None,
    save_channels: bool = False,
):
    if config_path is None:
        config_path = _default_config_path()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading config from {config_path}")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_cfg = cfg.get("model", {})
    K = model_cfg.get("K", cfg.get("K", 3))
    arch = model_cfg.get("arch", "simple")
    encoder_cfg = model_cfg.get("encoder", {})
    init = cfg.get("init_rgb")
    init_rgb = torch.tensor(init) if init is not None else None

    model = StainSeparationModel(K=K, init_rgb=init_rgb, arch=arch, encoder_kwargs=encoder_cfg).to(device)
    if not Path(ckpt_path).exists():
        raise FileNotFoundError(f"Checkpoint '{ckpt_path}' does not exist")
    print(f"Loading checkpoint from {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    loader = create_loader(batch_size=1, data_dir=cfg.get("data_dir"), augment=False, strong_aug=False)
    model.eval()
    with torch.no_grad():
        for i, (img, _) in enumerate(tqdm(loader, desc="Inferring")):
            if num_examples is not None and i >= num_examples:
                break

            img = img.to(device)
            recon, C = model(img)
            save_image(recon.cpu(), out_dir / f"{i}.png")

            if save_channels:
                for k in range(C.shape[1]):
                    Ck = torch.zeros_like(C)
                    Ck[:, k : k + 1] = C[:, k : k + 1]
                    ch_img = model.decoder(Ck)
                    save_image(ch_img.cpu(), out_dir / f"{i}_ch{k}.png")
    print(f"Inference complete. Images saved to {out_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--output_dir", default="out")
    parser.add_argument("--num_examples", type=int, default=None, help="number of random samples to infer")
    parser.add_argument("--save_channels", action="store_true", help="save per-stain channel reconstructions")
    args = parser.parse_args()
    main(
        args.ckpt,
        args.config,
        args.output_dir,
        num_examples=args.num_examples,
        save_channels=args.save_channels,
    )

