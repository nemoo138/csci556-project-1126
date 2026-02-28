"""Evaluation with automatic scale correction."""

import torch
import numpy as np
from data.dataset import MultiProductDataset
from models import (
    BaselineDasymetric, CNNBaseline, ResidualVAE, TimeEmbedding,
    DualBranchConditionalEncoder, ProductEmbedding, SimpleUNet, C3LDMSampler
)
from eval.metrics import rmse, mae, r2_score, spatial_correlation
from tqdm import tqdm

# Configuration
checkpoint_path = 'checkpoints_wp/checkpoint_best.pt'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print("Loading models...")
checkpoint = torch.load(checkpoint_path, map_location=device)

# Detect baseline type from checkpoint
baseline_state = checkpoint['models'].get('baseline', None)
if baseline_state is not None and len(baseline_state) > 0:
    if any('net.' in key for key in baseline_state.keys()):
        print("Detected CNNBaseline from checkpoint")
        baseline = CNNBaseline(hidden_channels=16).to(device)
        baseline.load_state_dict(baseline_state)
    else:
        print("Detected BaselineDasymetric from checkpoint")
        baseline = BaselineDasymetric().to(device)
else:
    print("No baseline in checkpoint, using BaselineDasymetric")
    baseline = BaselineDasymetric().to(device)
vae = ResidualVAE(latent_channels=4, base_channels=64).to(device)
time_emb = TimeEmbedding(dim=256, base_dim=64).to(device)
cond_state = checkpoint['models'].get('cond_encoder', {})
low_w = cond_state.get('low_res_branch.0.weight', None)
high_w = cond_state.get('high_res_branch.0.weight', None)
low_in = low_w.shape[1] if low_w is not None else 1
high_in = high_w.shape[1] if high_w is not None else low_in
cond_encoder = DualBranchConditionalEncoder(
    cond_channels=256, low_res_ch=128, high_res_ch=128,
    low_in_channels=low_in, high_in_channels=high_in
).to(device)
product_emb = ProductEmbedding(num_products=3, d_prod=64, cond_channels=256).to(device)
unet = SimpleUNet(in_channels=4, model_channels=128, time_emb_dim=256, cond_channels=256).to(device)

# Load weights (baseline already loaded above if present)
vae.load_state_dict(checkpoint['models']['vae'])
time_emb.load_state_dict(checkpoint['models']['time_emb'])
cond_encoder.load_state_dict(checkpoint['models']['cond_encoder'])
product_emb.load_state_dict(checkpoint['models']['product_emb'])
unet.load_state_dict(checkpoint['models']['unet'])

# Set to eval
for model in [baseline, vae, time_emb, cond_encoder, product_emb, unet]:
    model.eval()

# Load dataset
dataset = MultiProductDataset(
    pairing_csv='data/paired_dataset/test_split.csv',
    data_root='data',
    normalize=True,
    return_census=False,
    products=['WorldPop']
)

print(f"Test dataset: {len(dataset)} samples")

# Create sampler
betas = torch.linspace(0.0001, 0.02, 1000).to(device)
sampler = C3LDMSampler(baseline, vae, time_emb, cond_encoder, product_emb, unet, None, betas, device)

# Evaluate
all_preds_raw = []
all_preds_rescaled = []
all_targets = []
all_baselines = []

print("\nEvaluating on first 100 samples...")

with torch.no_grad():
    for idx in tqdm(range(min(100, len(dataset)))):
        sample = dataset[idx]
        lights = sample['lights'].unsqueeze(0).to(device)
        settlement = sample['settlement'].unsqueeze(0).to(device)
        lights_mask = sample.get('lights_mask')
        settlement_mask = sample.get('settlement_mask')
        if lights_mask is None:
            lights_mask = torch.ones_like(lights)
        else:
            lights_mask = lights_mask.unsqueeze(0).to(device)
        if settlement_mask is None:
            settlement_mask = torch.ones_like(settlement)
        else:
            settlement_mask = settlement_mask.unsqueeze(0).to(device)
        target = sample['target'].cpu().numpy()[0]
        product_id = sample['product_id']

        # Get baseline
        baseline_map = baseline(lights, settlement).cpu().numpy()[0, 0]

        # Generate prediction
        pred_raw = sampler.sample_population_map(
            lights=lights,
            settlement=settlement,
            lights_mask=lights_mask,
            settlement_mask=settlement_mask,
            product_id=product_id,
            num_samples=1, sampler='ddim', num_steps=50,
            show_progress=False
        ).cpu().numpy()[0, 0, 0]

        # Rescale to match baseline sum (simple heuristic)
        # Better: rescale to match target sum (if available)
        target_sum = target.sum()
        pred_sum = pred_raw.sum()

        if pred_sum > 0.1:  # Avoid division by zero
            scale_factor = target_sum / pred_sum
            pred_rescaled = pred_raw * scale_factor
        else:
            pred_rescaled = pred_raw

        all_preds_raw.append(pred_raw)
        all_preds_rescaled.append(pred_rescaled)
        all_targets.append(target)
        all_baselines.append(baseline_map)

# Stack
all_preds_raw = np.stack(all_preds_raw)
all_preds_rescaled = np.stack(all_preds_rescaled)
all_targets = np.stack(all_targets)
all_baselines = np.stack(all_baselines)

# Compute metrics
print("\n" + "="*80)
print("RESULTS")
print("="*80)

print("\nBaseline (lights × settlement):")
print(f"  RMSE: {rmse(all_baselines, all_targets):.4f}")
print(f"  MAE:  {mae(all_baselines, all_targets):.4f}")
print(f"  R²:   {r2_score(all_baselines, all_targets):.4f}")
corr = spatial_correlation(all_baselines, all_targets, method='pearson')
print(f"  Corr: {corr:.4f}")

print("\nModel (raw predictions, no rescaling):")
print(f"  RMSE: {rmse(all_preds_raw, all_targets):.4f}")
print(f"  MAE:  {mae(all_preds_raw, all_targets):.4f}")
print(f"  R²:   {r2_score(all_preds_raw, all_targets):.4f}")
corr = spatial_correlation(all_preds_raw, all_targets, method='pearson')
print(f"  Corr: {corr:.4f}")

print("\nModel (with sum-matching rescaling):")
print(f"  RMSE: {rmse(all_preds_rescaled, all_targets):.4f}")
print(f"  MAE:  {mae(all_preds_rescaled, all_targets):.4f}")
print(f"  R²:   {r2_score(all_preds_rescaled, all_targets):.4f}")
corr = spatial_correlation(all_preds_rescaled, all_targets, method='pearson')
print(f"  Corr: {corr:.4f}")

print("\n" + "="*80)
print("\nConclusion:")
print("  If rescaled RMSE << raw RMSE:")
print("    → Main issue is SCALE, not spatial pattern")
print("    → Model learned good patterns but wrong magnitude")
print("    → Solution: Add census consistency or scale loss")
print("\n  If rescaled RMSE ≈ raw RMSE:")
print("    → Issue is spatial pattern, not just scale")
print("    → Model hasn't learned proper population distribution")
print("    → Solution: More training or architecture changes")
print("="*80)
