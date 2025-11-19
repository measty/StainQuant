# StainQuant: Physics-Guided Stain Separation for Multi-IHC Pathology Images

StainLearn implements a physics-guided encoder–decoder for separating multiplex immunohistochemistry (IHC) stains from RGB whole-slide patches. The model operates in the optical-density domain, explicitly learning stain concentration maps and a stain basis that obey the Beer–Lambert law. By incorporating priors from digital pathology domain knowledge via multiple loss terms, the model produces disentangled stain channels suitable for downstream quantification and visualization.

## Approach Overview

- **CNN encoder for concentrations** – A UNet-style encoder produces non-negative stain concentration maps \(C(x) \ge 0\). Sparsity and entropy regularisers make the concentrations sharp and stain-specific.
- **Learned stain dictionary** – A trainable basis \(\Lambda \in \mathbb{R}^{K\times3}\) represents per-stain absorbance.
- **Visualization utilities** – Knockout (remove one stain) and single-channel renderings help interpret learned stains. A Gradio demo allows visualisations for rapid inspection of stain separation.

## Installation & Environment Setup

1. **Clone the repository** and create a Python environment (3.10+ recommended):
   ```bash
   git clone https://github.com/measty/StainQuant.git
   cd StainQuant
   ```
2. **Install package**:
   ```bash
   pip install -e .
   ```

## Data Preparation

Training expects RGB patches stored on disk. `data/loader.py` provides several dataset wrappers:

- **`FlatFolderDataset`** – Loads every image file under `data_dir` recursively (default). Suitable when all patches share a single folder.
- **`FolderDataset`** – Wraps `torchvision.datasets.ImageFolder` to support class-based subdirectories (`use_class_folders: true`).
- **`DummyDataset`** – Generates random tensors for debugging.

To train on your own data:

1. Extract square patches (512-1024 would be a good size choice) as `.png`, `.jpg` files.
2. Organise them in a single directory, for example:
   ```text
   /data/ihc_images/
       patch_000.png
       patch_001.png
       …
   ```
3. Update the configuration file to point `data_dir` to this directory.

## Configuring Training for a New Stain Panel

Configuration files are in `configs/` and are parsed by `scripts/train.py`. Use `configs/default.yaml` as a template. Key sections to edit when adapting to a new dataset:

```yaml
data_dir: /abs/path/to/your/patches

model:
  K: 5                    # change to the number of stain channels in your panel
  arch: unet              
  encoder:                  # defaults should work fine but you can tweak the model architecture here if needed
    base_channels: 32
    depth: 2
    norm: instancenorm
    residual: true
    dropout: 0.1
    cross_corr: true

channel_names:
  - H
  - CDX2
  - …                      # match the names of your stains

init_rgb:
  - [r, g, b]              # approximate RGB hue for each stain (0–255) in same order as channel_names

losses:                       # loss weights - defaults should work well but feel free to adjust
  rec:    {weight: 1.0}
  l1:     {weight: [0.0, 0.05]}       # start/end weights for annealed losses
  entropy:{weight: [0.0, 0.01]}
  col:    {weight: [0.1, 0.05]}
  overlap:{weight: [0.0, 0.05]}

trainer:
  max_epochs: 50
  lr: 2e-4
  batch_size: 4
  strong_augment: false
  ckpt_dir: checkpoints/my_panel
```

### Main things to change

- **Number of channels (`K`) & names** – Match the stains present in your panel. The order here determines how concentration maps are interpreted.
- **`init_rgb` priors** – Supply rough RGB values (0–255) measured from reference slides. They stabilise early training and anchor the colour-consistency loss.
- **Loss schedules** – Many losses accept two-element lists `[start, end]` to linearly ramp weights between warm-up and full training. Set the first element to `0.0` when bootstrapping a new panel.
- **Checkpoint directory** – Every training run writes its final config to `<ckpt_dir>/config.yaml` alongside model weights.

After editing the YAML, either overwrite `configs/default.yaml` or create `configs/<my_panel>.yaml`.

## Training

Run the training script with your configuration:

```bash
python scripts/train.py --config configs/my_panel.yaml
```

Important command-line flags (see `scripts/train.py` for full list):

- `--resume` – continue from an existing checkpoint.
- `--num-workers` – override dataloader workers if the default (2) is too low or high.
- `--amp` – enable mixed-precision training on supported GPUs.

During training the script writes periodic checkpoints to `trainer.ckpt_dir`, and will make a train_vis.html file wwith visualizations of recoonstructions and stain channels to monitor progress during training.

## Evaluating & Visualising Results

### Gradio Sanity Check

Use the Gradio app to inspect a trained checkpoint interactively:

```bash
python scripts/gradio_app.py --ckpt checkpoints/my_panel/final.pt
```

The interface presents two galleries:

1. **Knockout reconstructions** – Each panel shows the input reconstruction with one stain removed, highlighting the contribution of the dropped channel.
2. **Single-channel renderings** – Displays each stain in isolation.

Launch the app locally (default `http://127.0.0.1:7860`) and drag-and-drop validation patches to qualitatively assess separation quality.

### Command-line Inference

For batch evaluation on a folder of patches:

```bash
python scripts/infer.py \
  --ckpt checkpoints/my_panel/final.pt \
  --config configs/my_panel.yaml \
  --input_dir /path/to/test_patches \
  --output_dir outputs/my_panel
```

Outputs include reconstructed images, per-stain concentration maps, and summary metrics (MAE, PSNR, Adjusted Rand Index) computed in `evaluation/metrics.py`.

## Troubleshooting & Tips
- **Inactive stain channels** – If the model is not utilizing a stain channel for a relatively rare stain, try turning on the mask loss for that stain.

## Citation

If you find this repository useful, please cite the associated paper at [url].

