# Remote Sensing Masked Autoencoders (RS-MAE)

This repository contains the codebase for investigating and improving Masked Autoencoders (MAE) for multi-spectral remote sensing applications. It implements several novel modifications to the MAE framework, focusing on masking strategies, reconstruction targets, and loss formulations tailored for Earth observation data.

## Features & Ablations

- **Masking Strategies:** NDI Saliency Curriculum and Stochastic Mixed-Batch Regularization.
- **Reconstruction Targets:** Feature-Guided Targets (HOG + NDI) and Latent Targets (Self-Distillation).
- **Loss Formulations:** HOG-Weighted Loss and Differentiable GLCM Texture Loss.
- **Large-Tile Processing:** Epoch-Shifted Grid Sampling (for training) and Dynamic Sliding-Window Inference (for downstream evaluation).

## Setup & Configuration

Install the required dependencies via the provided conda environment:
```bash
mamba env create -f environment.yml
mamba activate rs_mae
```

Path configurations for datasets and outputs are managed by Hydra. Adjust the paths for your local setup or cluster environment in `configs/paths/local.yaml` or `configs/paths/cluster.yaml`.

## Data

The repository supports several datasets. You must acquire and prepare them according to their respective literature:
- **BigEarthNet v2.0 (reBEN)** / **reBEN 7k**
- **m-cashew-plantation**
- **MMHunan**
- **C2Seg** (Cross-City Semantic Segmentation - Augsburg, Berlin, Beijing, Wuhan)

## Usage

The project relies heavily on Hydra for configuration. You can easily switch between datasets and experiments using the command line.

### Pretraining
To run self-supervised pretraining using `train/pretrain.py`:
```bash
python train/pretrain.py experiment=baseline
```

### Downstream Fine-Tuning
To fine-tune a pretrained model for semantic segmentation using `train/finetune.py`:
```bash
python train/finetune.py experiment=baseline dataset=cashew_plantation
```

### Evaluation & Visualization
Scripts for evaluating models and plotting true-color qualitative comparisons are provided in the `checks/` directory. For example, to generate side-by-side segmentation results:
```bash
python checks/plot_inference.py
```

## Qualitative Results

![Qualitative Results on C2Seg (Beijing/Wuhan)](results/qualitative/c2seg_BW_sample.png)

## Engineering Highlights: Solving OOM on Large Tiles
Beyond the experiments, this codebase offers a robust, model-agnostic pipeline to solve **Out-of-Memory (OOM) errors and grid artifacts** on massive images (e.g., satellite or medical scans).
- **Scalable Data Loading:** `SegmentationDataset` for stochastic grid-sampling on arbitrary large tiles.
- **Seamless Inference:** `windowed_inference` for artifact-free, memory-safe sliding window predictions.
- **OOM-Free Evaluation:** `windowed_evaluation` for calculating metrics on massive predictions without crashing the GPU.

> 💡 **Standalone Utilities:** The `SegmentationDataset` class and the `windowed_inference` / `windowed_evaluation` functions are entirely model-agnostic. Feel free to drop them into your own projects to handle large tiles effortlessly!
> 
> For an interactive demonstration, check out the Jupyter Notebook: [`checks/large_tile_ds.ipynb`](checks/large_tile_ds.ipynb)

### Minimal Working Example
```python
from datasets.segmentation_ds import SegmentationDataset
from utils.spatial_ops import windowed_inference, windowed_evaluation

# 1. Train seamlessly on massive tiles
train_ds = SegmentationDataset(huge_images, huge_masks, grid_sampling=True, crop_size=256, min_margin=128)

# 2. Artifact-free inference (merged on CPU to save VRAM)
pred_logits = windowed_inference(model, huge_images, window_size=256, stride=128, batch_size=16, merger_device="cpu")

# 3. Safely evaluate metrics without crashing the GPU
windowed_evaluation(pred_logits, huge_masks, tile_size=5000, eval_batch_size=3, torchmetrics=[mIoU])
```

### Experiment Tracking
All experiments, training curves, and evaluation logs for this thesis are publicly accessible on Weights & Biases (W&B):
[View Run Logs on W&B](https://wandb.ai/jacer-technical-university-of-berlin/RS_MAE-Thesis/table?nw=nwuserjacer)
