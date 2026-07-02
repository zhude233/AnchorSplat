# AnchorSplat: Fast and Structure Consistent Detail Synthesis for Gaussian Splatting (ECCV 2026)

✨ **ECCV 2026 Code Release** · ⚡ **Fast** · 🌍 **Generalizable** · 🔌 **Plug-and-Play**

Official code release for **AnchorSplat**.

AnchorSplat is a fast, generalizable, and plug-and-play method for enhancing low-quality 3D Gaussian Splatting assets. Given only a coarse 3DGS model, it synthesizes detail-rich Gaussian primitives directly in 3D with a single network forward pass, avoiding the slow render-SR-reoptimize pipeline used by 2D-centric 3DGS super-resolution methods.

## ✨ Highlights

- ⚡ **Fast**: feed-forward 3D-native enhancement without per-scene optimization.
- 🌍 **Generalizable**: transfers to unseen 3DGS assets, including inputs with SH settings not seen during training.
- 🔌 **Plug-and-play**: supports external 3DGS PLY inputs with explicit normalization and coordinate restoration.
- 🎯 **Structure-consistent**: local point anchors keep generated details aligned with the input geometry.

This repository hosts the ECCV 2026 code release. Large assets such as datasets, pretrained checkpoints, generated point clouds, logs, and evaluation outputs are intentionally excluded from git.

## 🔥 News

- ✅ **2026-07-02**: Release training code.
- ✅ **2026-07-02**: Release evaluation code.
- ✅ **2026-07-02**: Release inference code and demo.

## 📌 Release TODO

- ✅ Release training code
- ✅ Release evaluation code
- ✅ Release inference code and demo
- ⏳ Release pretrained model
- ⏳ Release processed third-party datasets
- ⏳ Release 3DGS-SR dataset
- ⏳ Release project page

## 🛠️ Installation

The code is tested around PyTorch, CUDA extension packages, Pointcept, and gsplat. Exact wheels depend on your CUDA and PyTorch versions.

```bash
conda env create -f environment.yml
conda activate anchorsplat

# Point Transformer V3 backbone.
git clone https://github.com/Pointcept/Pointcept.git third_party/Pointcept
pip install -e third_party/Pointcept
```

If you prefer a manual installation:

```bash
conda create -n anchorsplat python=3.8 -y
conda activate anchorsplat
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# Point Transformer V3 backbone.
git clone https://github.com/Pointcept/Pointcept.git third_party/Pointcept
pip install -e third_party/Pointcept
```

If CUDA extension packages fail to install from `requirements.txt`, install versions that match your local PyTorch/CUDA build:

- `spconv-cu118` or the matching `spconv` wheel for your CUDA version
- `torch-scatter`, `torch-sparse`, `torch-cluster`, `torch-geometric`
- `fused-ssim`
- optional `flash-attn` if you keep `PointTransformerV3Model.enable_flash = True`

## 📦 Pretrained Checkpoints

Place the released checkpoint at:

```text
checkpoints/anchorsplat_20x.pth
```

Expected SHA256 for the current 20x checkpoint:

```text
ad05f8b965c002c1f62cea53e4ce10ed4804bbc433375afa5f411f236d1b79a3
```

You can override this path in all scripts with the `WEIGHTS` or `CHECKPOINT` environment variables.

## ⚡ Inference On External PLY Files

AnchorSplat includes a lightweight inference path for Gaussian PLY files exported by LGM-style or Trellis-style pipelines.

```bash
WEIGHTS=checkpoints/anchorsplat_20x.pth \
bash scripts/inference_external.sh examples/lgm_sample.ply outputs/lgm_sample_refined.ply lgm

WEIGHTS=checkpoints/anchorsplat_20x.pth \
bash scripts/inference_external.sh /path/to/trellis_output.ply outputs/trellis_refined.ply trellis
```

Use `NORMALIZATION=centered` for object-centric assets near the origin and `NORMALIZATION=bbox` for translated COLMAP/world-frame assets. The default `NORMALIZATION=auto` chooses between them from the input bounding box.

Equivalent Python entry:

```bash
python inference_external.py \
  --weights checkpoints/anchorsplat_20x.pth \
  --input_ply examples/lgm_sample.ply \
  --output_ply outputs/lgm_sample_refined.ply \
  --model_type lgm \
  --normalization auto
```

The PLY reader expects Inria-style 3DGS attributes: log-space `scale_*`, logit-space `opacity`, SH DC `f_dc_*`, optional `f_rest_*`, and quaternion `rot_*`. Coordinates are normalized internally to `[0, 1]^3`, scales are shifted by the same scalar factor, and the output is written back in the original input coordinate frame. See [docs/input_format.md](docs/input_format.md) for the exact contract and conversion formulas.

## 🗂️ Dataset Layout

Training expects each scene to contain a low-resolution input 3DGS checkpoint and high-resolution target views:

```text
data/3dgs-sr/train/<scene_name>/
  256/
    gaussian_splatting/
      ckpts/
        ckpt_14999_rank0.pt
    sparse/0/
      cameras.bin
      images.bin
      points3D.bin
  1024/
    images/
    cameras.txt
    images.txt

data/3dgs-sr/test/<scene_name>/
  ...
```

The defaults are configured in `configs/dataset/objaverse.gin`:

```gin
SplatfactoDataset.low_resolution = 256
SplatfactoDataset.high_resolution = 1024
SplatfactoDataset.input_ckpt_step = 14999
train_dataset/SplatfactoDataset.dataset_folder = 'data/3dgs-sr/train'
test_dataset/SplatfactoDataset.dataset_folder = 'data/3dgs-sr/test'
```

## 🏋️ Training

The main training script is DDP-based. The default script launches 8 processes:

```bash
bash scripts/train_anchorsplat.sh
```

Common overrides:

```bash
GPUS=0 NPROC=1 ACCUMULATE_STEP=8 OUTPUT_DIR=outputs/anchorsplat_20x_single_gpu \
bash scripts/train_anchorsplat.sh
```

The core model settings are in `configs/model/ptv3.gin`:

```gin
FeaturePredictor.point_multiply_factor = 20
FeaturePredictor.anchor_offset_scale = 0.015
FeaturePredictor.predict_double_features = True
```

The default W&B mode is disabled for open-source runs. Enable online logging with:

```bash
python train.py --disable_wandb=false ...
```

## 📊 Evaluation

Run evaluation with a checkpoint:

```bash
CHECKPOINT=checkpoints/anchorsplat_20x.pth \
GPUS=0 NPROC=1 \
bash scripts/evaluate_anchorsplat.sh
```

Outputs are written under `outputs/` and include rendered comparisons, per-rank metric files, and an evaluation log.

## 🧰 Dataset Preparation Tools

The `tools/` directory contains helper scripts used during internal data conversion and low-resolution 3DGS preparation:

- `tools/convert_3dgs_ply_to_splatformer_ckpt.py`
- `tools/prepare_mvimgnet_splatformer_ready.py`
- `tools/train_mvimgnet_splatformer_ready.sh`
- `tools/prepare_mvimgnet_lowres_3dgs.py`
- `tools/train_mvimgnet_lowres_3dgs.sh`
- `tools/rewrite_colmap_bins_legacy.py`

These tools use relative defaults under `data/` and `third_party/`. Some require external projects such as gsplat, COLMAP preprocessing scripts, or SuperGaussian. Keep those dependencies outside git, for example under `third_party/`.

## 🧭 Repository Structure

```text
configs/              Gin configs for data, model, and training
dataset/              3DGS dataset loading and COLMAP utilities
models/               AnchorSplat predictor and Point Transformer V3 wrapper
scripts/              Main train, eval, and external inference launchers
tools/                Dataset conversion and preprocessing utilities
utils/                Rendering, losses, metrics, logging, and optimizers
train.py              DDP training and evaluation entrypoint
inference_external.py External PLY inference entrypoint
```

## 🙏 Acknowledgements

This code builds on ideas and components from SplatFormer, Pointcept, gsplat, PyTorch Geometric, and 3D Gaussian Splatting tooling. Please also follow the licenses of all third-party dependencies you install locally.

## 📄 License

This repository is released under the MIT License. Third-party dependencies and datasets retain their original licenses.
