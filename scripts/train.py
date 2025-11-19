import sys
import os
import yaml
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.loader import create_loader
from training.trainer import Trainer


def main(config_path):
    print(f"Loading config from {config_path}")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    trainer_cfg = cfg.get('trainer', {})
    batch_size = trainer_cfg.get('batch_size', 2)
    ckpt_dir = trainer_cfg.get('ckpt_dir', 'checkpoints')
    # save config details in checkpoint directory also
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    with open(os.path.join(ckpt_dir, 'config.yaml'), 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)
    print("Creating data loader...")
    loader = create_loader(batch_size=batch_size, data_dir=cfg.get('data_dir'), strong_aug=cfg.get("trainer", {}).get('strong_augment', False))
    print("Initializing trainer...")
    trainer = Trainer(cfg)
    print("Starting training...")
    trainer.fit(loader, epochs=trainer_cfg.get('max_epochs', 1), ckpt_dir=ckpt_dir, vis_path=os.path.join(ckpt_dir, 'train_vis.html'))
    print("Training complete.")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/default.yaml')
    args = parser.parse_args()
    main(args.config)
