import os
from importlib import resources
import yaml

from stainquant.data.loader import create_loader
from stainquant.training.trainer import Trainer


def _default_config_path() -> str:
    try:
        return str(resources.files("stainquant.configs").joinpath("default.yaml"))
    except Exception:
        # Fallback if importlib.resources is unavailable/older Python
        return os.path.join(os.path.dirname(__file__), "..", "configs", "default.yaml")


def main(config_path: str | None = None):
    if config_path is None:
        config_path = _default_config_path()
    print(f"Loading config from {config_path}")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    trainer_cfg = cfg.get("trainer", {})
    batch_size = trainer_cfg.get("batch_size", 2)
    ckpt_dir = trainer_cfg.get("ckpt_dir", "checkpoints")
    # save config details in checkpoint directory also
    os.makedirs(ckpt_dir, exist_ok=True)
    with open(os.path.join(ckpt_dir, "config.yaml"), "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    print("Creating data loader...")
    loader = create_loader(
        batch_size=batch_size,
        data_dir=cfg.get("data_dir"),
        strong_aug=cfg.get("trainer", {}).get("strong_augment", False),
    )
    print("Initializing trainer...")
    trainer = Trainer(cfg)
    print("Starting training...")
    trainer.fit(loader, epochs=trainer_cfg.get("max_epochs", 1), ckpt_dir=ckpt_dir)
    print("Training complete.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    main(args.config)

