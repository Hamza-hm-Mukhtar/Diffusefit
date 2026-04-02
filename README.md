# DiffuseFit

A research-oriented PyTorch implementation of **DiffuseFit**, a three-stage diffusion-based virtual try-on pipeline.

## What this repository includes

- **Stage 1 — Pose-Aligned Garment Alignment**
  - masked garment normalization
  - pose encoding from skeleton + densepose
  - affine style modulation
  - cross-modal fusion (`cross_attention`, `concat`, `add`)
  - dense flow warping
- **Stage 2 — Garment-Aware Parsing Synthesis**
  - layout replacement with warped garment mask + pose prior
  - UNet semantic parser
- **Stage 3 — Limb-Aware Try-On Generation**
  - masked limb autoencoder
  - CLIP garment semantics
  - latent diffusion decoder for final synthesis
- **Evaluation**
  - SSIM, PSNR, FID, LPIPS, GFS
- **Ablations**
  - pose-guided flow
  - co-training / frozen stage interface
  - cross-modal fusion strategy
  - limb-aware generator variants
  - garment replacement
  - diffusion conditioning inputs
  - GP-VTON-style warping replacement
  - diffusion sampling steps
  - latency benchmarking

## Important note

The paper excerpt is detailed but still leaves several implementation choices unspecified, including:
- exact channel counts and hidden widths
- exact label IDs for garment / limb parsing channels across datasets
- exact CLIP token injection granularity inside the diffusion denoiser
- the exact GP-VTON stage-1 replacement implementation
- the exact diffusion sampler schedule and VAE used

This repository therefore implements a **faithful research-grade reference implementation**, not an official reproduction. All such assumptions are documented in `docs/paper_analysis_and_required_files.md`.

## Repository layout

```text
diffusefit_repo/
├── configs/
├── diffusefit/
├── docs/
├── scripts/
└── tests/
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Dataset preparation

This repo expects a manifest-based layout. You can either:
1. use existing VITON / VITON-HD / DressCode folders directly, or
2. generate manifests with:

```bash
python scripts/prepare_dataset.py --dataset viton --root /path/to/VITON --out manifests/viton
python scripts/prepare_dataset.py --dataset viton_hd --root /path/to/VITON-HD --out manifests/viton_hd
python scripts/prepare_dataset.py --dataset dresscode --root /path/to/DressCode --out manifests/dresscode
```

Each manifest item stores paths for:
- person image
- garment image
- garment mask
- parse map
- skeleton map
- densepose map
- optional limb map
- target image
- split metadata

## Training

### Stage 1
```bash
python scripts/train_stage1.py --config configs/viton.yaml
```

### Stage 2
```bash
python scripts/train_stage2.py --config configs/viton.yaml --stage1-ckpt outputs/stage1/best.pt
```

### Stage 3
```bash
python scripts/train_stage3.py --config configs/viton.yaml   --stage1-ckpt outputs/stage1/best.pt   --stage2-ckpt outputs/stage2/best.pt
```

### End-to-end orchestration
```bash
python scripts/train_all.py --config configs/viton.yaml
```

## Evaluation

```bash
python scripts/evaluate.py --config configs/viton_hd.yaml   --checkpoint outputs/full/best.pt   --split test
```

## Inference

```bash
python scripts/infer.py --config configs/viton_hd.yaml   --checkpoint outputs/full/best.pt   --person path/to/person.jpg   --garment path/to/cloth.jpg   --garment-mask path/to/cloth_mask.png   --parse path/to/parse.png   --skeleton path/to/openpose.png   --densepose path/to/densepose.png   --output outputs/demo.png
```

## Ablations

Example:
```bash
python scripts/run_ablation.py --config configs/viton.yaml   --ablation configs/ablations/no_pose_guided_flow.yaml
```

## Benchmarking

```bash
python scripts/benchmark.py --config configs/viton_hd.yaml   --checkpoint outputs/full/best.pt   --steps 25 36 50 100
```

## Tests

```bash
pytest -q
```
