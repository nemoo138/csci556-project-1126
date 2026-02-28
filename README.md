# C3-LDM Architecture and Pipeline Summary

## Overview
C3-LDM (Census-Consistent, Conditional Latent Diffusion Model) maps VIIRS nightlights and WSF settlement masks to high-resolution population density maps. It models residuals over a baseline dasymetric allocation, runs diffusion in a compressed latent space, and can enforce census totals per admin unit. Training supports multiple population products (WorldPop, GHS-POP, HRSL) via product conditioning.

## Data and Inputs
- Tile size: 256x256.
- Inputs: stacked features (2, 256, 256) where channel 0 is VIIRS nightlights (low-res) and channel 1 is WSF settlement/built-up mask (high-res).
- Targets: population grids from WorldPop, GHS-POP, HRSL.
- Product IDs: WorldPop=0, GHS-POP=1, HRSL=2.
- Pairing file: `data/paired_dataset/multi_product_pairing.csv` with 41,896 paired samples (WorldPop 24,020; GHS-POP 14,592; HRSL 3,284).
- Data loader: `data/dataset.py` normalizes inputs to [0, 1], replaces NaN/Inf with zeros, and can optionally return admin IDs and census totals.

## Model Architecture
### 1) Baseline and Residualization
- Default baseline: `BaselineDasymetric` in `models/baseline.py`.
  - Score: `(lights + lambda_L) * (settlement + lambda_S)`.
  - Optional per-admin normalization if census totals are available.
- Optional learned baseline: `CNNBaseline` (selected via `--baseline_type cnn`).
- Residual target: `R = log((Y + eps) / (B + eps))`.
- Population reconstruction: `P = B * exp(R)`.

### 2) Residual VAE (Latent Compression)
- `ResidualVAE` in `models/vae.py`.
- Encodes residual maps from 256x256 to latent 32x32 with 4 channels, and decodes back.
- Training uses KL loss plus direct reconstruction loss on residuals.

### 3) Conditioning Stack
- `DualBranchConditionalEncoder` in `models/conditional_encoder.py`:
  - Low-res branch for lights, high-res branch for settlement.
  - Fused to a 32x32 conditioning tensor (default `cond_channels=256`).
- `ProductEmbedding` in `models/product_embedding.py`:
  - Per-product embedding broadcast to match the conditioning tensor.
- `TimeEmbedding` in `models/time_embedding.py`:
  - Sinusoidal embedding plus MLP (default `time_emb_dim=256`).

### 4) Diffusion Core
- `SimpleUNet` in `models/unet_simple.py`.
- Operates on latent `z_t` (4x32x32), predicts noise with FiLM-style conditioning from time and spatial/product features.

### 5) Census Consistency
- `CensusConsistencyLayerVectorized` in `models/census_layer.py`.
- Rescales predicted population so each admin unit sums to its census total when admin IDs and totals are provided.

## Training Pipeline (train.py)
1. Load batch from `MultiProductDataset` (lights, settlement, target, product_id).
2. Compute baseline (simple or CNN).
3. Compute residual target `R_true`.
4. Encode residual with VAE to `z_0`; sample diffusion timestep `t` and noise to create `z_t`.
5. Build conditioning (spatial + product) and time embedding.
6. U-Net predicts noise; diffusion loss = MSE(pred_noise, true_noise).
7. Decode `z_0` for direct VAE reconstruction loss.
8. Optional census loss on total population sums (`lambda_census`).
9. Total loss = diffusion + beta_kl * KL + lambda_vae_recon * recon + lambda_census * census.
10. Optimizer: AdamW; linear beta schedule (default 1000 steps).
11. Checkpoints saved via `utils/checkpoint.py`; resume supported.

Notes:
- Input and intermediate NaN/Inf checks are in place; corrupted data is sanitized in the dataset loader.
- VAE and KL warmup stages are supported (see CLI args).

## Inference and Sampling Pipeline (inference.py, models/sampler.py)
1. Load checkpoint and build model stack.
2. Normalize condition inputs per channel to match training.
3. Create `C3LDMSampler` with DDPM or DDIM sampler.
4. Sample latent `z_0`, decode to residual, and reconstruct population `P = B * exp(R)`.
5. Optionally apply census consistency layer using admin IDs and totals.
6. Output one or more population maps with shape (num_samples, 1, 256, 256).

## Evaluation Pipeline (eval/)
- `eval/evaluate.py` runs inference on a dataset split and computes metrics.
- Metrics include RMSE, MAE, MAPE, R2, Pearson/Spearman correlation, dense/sparse errors by percentile, and census error stats when admin data is available.
- Outputs: JSON metrics, human-readable summary, and optional saved predictions.

## Key Repository Paths
- `train.py`: training loop and CLI config.
- `inference.py`: inference entry point.
- `models/`: baseline, VAE, conditioning, U-Net, census layer, sampler.
- `data/dataset.py`: multi-product data loader.
- `utils/checkpoint.py`: checkpoint save/load helpers.
- `eval/`: evaluation scripts and metrics.
