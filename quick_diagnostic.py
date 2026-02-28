"""Quick diagnostic to check model predictions."""

import torch
import numpy as np
from data.dataset import MultiProductDataset
from models import (
    BaselineDasymetric, CNNBaseline, ResidualVAE, TimeEmbedding,
    DualBranchConditionalEncoder, ProductEmbedding, SimpleUNet, C3LDMSampler
)

# Load checkpoint
checkpoint_path = 'checkpoints_wp/checkpoint_best.pt'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print("Loading checkpoint...")
checkpoint = torch.load(checkpoint_path, map_location=device)

# Detect baseline type from checkpoint
# Try to infer from saved baseline state dict
baseline_state = checkpoint['models'].get('baseline', None)
if baseline_state is not None and len(baseline_state) > 0:
    # Check if it's CNNBaseline (has 'net.0.weight') or simple (no parameters)
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

# Set to eval mode
baseline.eval()
vae.eval()
time_emb.eval()
cond_encoder.eval()
product_emb.eval()
unet.eval()

print("Models loaded successfully")

# Load test dataset
dataset = MultiProductDataset(
    pairing_csv='data/paired_dataset/test_split.csv',
    data_root='data',
    normalize=True,
    return_census=False,
    products=['WorldPop']
)

print(f"\nTest dataset: {len(dataset)} samples")

# Test on a few samples
betas = torch.linspace(0.0001, 0.02, 1000).to(device)
sampler = C3LDMSampler(baseline, vae, time_emb, cond_encoder, product_emb, unet, None, betas, device)

print("\nTesting on 5 samples...")
print("=" * 80)

with torch.no_grad():
    for idx in [0, 100, 500, 1000, 2000]:
        if idx >= len(dataset):
            continue

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
        target = sample['target'].cpu().numpy()
        product_id = sample['product_id']

        # Get baseline
        baseline_map = baseline(lights, settlement).cpu().numpy()[0, 0]

        # Generate prediction (DDIM, 50 steps)
        pred = sampler.sample_population_map(
            lights=lights,
            settlement=settlement,
            lights_mask=lights_mask,
            settlement_mask=settlement_mask,
            product_id=product_id,
            num_samples=1, sampler='ddim', num_steps=50,
            show_progress=False
        ).cpu().numpy()[0, 0, 0]

        # Compute errors
        rmse_val = np.sqrt(np.mean((pred - target[0]) ** 2))

        print(f"\nSample {idx}:")
        print(f"  Target:     sum={target.sum():8.1f}, max={target.max():6.3f}, mean={target.mean():6.4f}")
        print(f"  Baseline:   sum={baseline_map.sum():8.1f}, max={baseline_map.max():6.3f}, mean={baseline_map.mean():6.4f}")
        print(f"  Prediction: sum={pred.sum():8.1f}, max={pred.max():6.3f}, mean={pred.mean():6.4f}")
        print(f"  RMSE: {rmse_val:.4f}")
        print(f"  Ratio (pred/target): {pred.sum() / (target.sum() + 1e-6):.3f}")

        # Check for anomalies
        if pred.max() > 100:
            print("  ⚠️  WARNING: Prediction has very large values!")
        if pred.min() < -1:
            print("  ⚠️  WARNING: Prediction has negative values!")
        if np.isnan(pred).any():
            print("  ⚠️  WARNING: Prediction contains NaN!")
        if pred.sum() > target.sum() * 10:
            print("  ⚠️  WARNING: Prediction is 10x larger than target!")
        if pred.sum() < target.sum() * 0.1 and target.sum() > 1:
            print("  ⚠️  WARNING: Prediction is 10x smaller than target!")

print("\n" + "=" * 80)
print("\nDiagnostic complete. Check the warnings above for issues.")
