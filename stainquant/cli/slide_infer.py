import argparse
import csv
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from tiatoolbox.models.dataset.classification import WSIPatchDataset
from tiatoolbox.wsicore.wsireader import WSIReader

from stainquant.models.full_model import StainSeparationModel


def load_model(ckpt_path: str, cfg: dict, device: torch.device) -> StainSeparationModel:
    model_cfg = cfg.get("model", {})
    K = model_cfg.get("K", cfg.get("K", 3))
    arch = model_cfg.get("arch", "simple")
    encoder_cfg = model_cfg.get("encoder", {})
    init = cfg.get("init_rgb")
    init_rgb = torch.tensor(init) if init is not None else None

    model = StainSeparationModel(
        K=K, init_rgb=init_rgb, arch=arch, encoder_kwargs=encoder_cfg
    ).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, K


def main(
    wsi_path: str,
    ckpt_path: str,
    config_path: str,
    output_csv: str = "stain_quant.csv",
    threshold: float | None = None,
):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    inf_cfg = cfg.get("inference", {})
    patch_size = inf_cfg.get("patch_size", 512)
    mpp = inf_cfg.get("mpp", 0.5)
    reader = WSIReader.open(wsi_path)
    slide_mpp = reader.info.mpp
    scale_factor = mpp / slide_mpp  # we will be saving coords at baseline resolution

    try:
        dataset = WSIPatchDataset(
            img_path=wsi_path,
            mask_path=None,
            patch_input_shape=[patch_size, patch_size],
            stride_shape=[patch_size, patch_size],
            resolution=mpp,
            units="mpp",
            auto_get_mask=True,
        )
    except Exception:  # fallback for slides without mpp metadata
        dataset = WSIPatchDataset(
            img_path=wsi_path,
            mode="tile",
            patch_input_shape=[patch_size, patch_size],
            stride_shape=[patch_size, patch_size],
            auto_get_mask=True,
        )
    loader = DataLoader(dataset, batch_size=1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, K = load_model(ckpt_path, cfg, device)

    results = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Processing patches"):
            if isinstance(batch, dict):
                img = batch["image"]
                coords = batch["coords"]
            else:  # fallback to tuple
                img, coords = batch

            img = torch.as_tensor(img, device=device).float() / 255.0
            if img.dim() == 3:  # HWC
                img = img.permute(2, 0, 1).unsqueeze(0)
            elif img.dim() == 4 and img.shape[-1] == 3:  # NHWC
                img = img.permute(0, 3, 1, 2)

            _, C = model(img)
            if threshold is not None:
                C = torch.where(C < threshold, torch.zeros_like(C), C)
            mean_vec = C.mean(dim=(2, 3)).cpu().squeeze(0).tolist()

            c = coords[0] if isinstance(coords, torch.Tensor) else coords
            c = c.tolist() if hasattr(c, "tolist") else c
            results.append([c[0] * scale_factor[0], c[1] * scale_factor[1], c[2] * scale_factor[0], c[3] * scale_factor[1]] + mean_vec)

    channel_names = cfg.get("channel_names", [f"stain_{i}" for i in range(K)])
    if len(channel_names) < K:
        channel_names += [f"stain_{i}" for i in range(len(channel_names), K)]
    header = ["xmin", "ymin", "xmax", "ymax"] + channel_names[:K]
    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(results)
    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Slide-level stain quantification")
    parser.add_argument("--wsi", required=True, help="Path to input WSI")
    parser.add_argument("--ckpt", required=True, help="Path to model checkpoint")
    parser.add_argument(
        "--config",
        default="ckpt_dir",
        help="Config file path. Use 'ckpt_dir' to load config.yaml from the checkpoint directory.",
    )
    parser.add_argument("--output_csv", default="stain_quant.csv", help="Where to save results")
    parser.add_argument("--threshold", type=float, default=None, help="Ignore stain values below this threshold")
    args = parser.parse_args()

    main(
        wsi_path=args.wsi,
        ckpt_path=args.ckpt,
        config_path=(args.config if args.config != "ckpt_dir" else str(Path(args.ckpt).parent / "config.yaml")),
        output_csv=args.output_csv,
        threshold=args.threshold,
    )
