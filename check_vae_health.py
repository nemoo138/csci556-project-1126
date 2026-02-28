"""Check VAE health from checkpoint."""

import torch
import numpy as np
from data.dataset import MultiProductDataset
from models import BaselineDasymetric, CNNBaseline, ResidualVAE

# Configuration
checkpoint_path = 'checkpoints_cnn/checkpoint_last.pt'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print("Loading checkpoint...")
checkpoint = torch.load(checkpoint_path, map_location=device)

# Detect baseline type
baseline_state = checkpoint['models'].get('baseline', None)
if baseline_state is not None and any('net.' in key for key in baseline_state.keys()):
    baseline = CNNBaseline(hidden_channels=16).to(device)
    baseline.load_state_dict(baseline_state)
else:
    baseline = BaselineDasymetric().to(device)

vae = ResidualVAE(latent_channels=4, base_channels=64).to(device)
vae.load_state_dict(checkpoint['models']['vae'])
vae.eval()
baseline.eval()

# Load dataset
dataset = MultiProductDataset(
    pairing_csv='data/paired_dataset/train_split.csv',
    data_root='data',
    normalize=True,
    return_census=False,
    products=['WorldPop']
)

print(f"\nAnalyzing VAE on {min(100, len(dataset))} training samples...\n")

all_mu = []
all_logvar = []
all_recon_errors = []

with torch.no_grad():
    for idx in range(min(100, len(dataset))):
        sample = dataset[idx]
        lights = sample['lights'].unsqueeze(0).to(device)
        settlement = sample['settlement'].unsqueeze(0).to(device)
        target = sample['target'].unsqueeze(0).to(device)

        # Get baseline
        baseline_map = baseline(lights, settlement)

        # Compute residual
        residual_target = torch.log((target + 1e-6) / (baseline_map + 1e-6))
        residual_target = torch.clamp(residual_target, min=-10, max=10)

        # Encode
        mu, logvar = vae.encode(residual_target)

        # Decode
        z = mu  # Use mean for deterministic check
        residual_recon = vae.decode(z)

        # Reconstruction error
        recon_error = torch.nn.functional.mse_loss(residual_recon, residual_target).item()

        all_mu.append(mu.cpu().numpy())
        all_logvar.append(logvar.cpu().numpy())
        all_recon_errors.append(recon_error)

# Stack
all_mu = np.concatenate(all_mu, axis=0)
all_logvar = np.concatenate(all_logvar, axis=0)

print("="*80)
print("VAE HEALTH REPORT")
print("="*80)

print("\n1. LATENT STATISTICS")
print(f"   mu (mean):     {all_mu.mean():.4f}  [target: ~0.0]")
print(f"   mu (std):      {all_mu.std():.4f}   [target: ~1.0]")
print(f"   logvar (mean): {all_logvar.mean():.4f}  [target: ~0.0]")
print(f"   logvar (std):  {all_logvar.std():.4f}")

# Compute actual variance
actual_var = np.exp(all_logvar).mean()
print(f"   σ² (variance): {actual_var:.4f}  [target: ~1.0]")
print(f"   σ (std dev):   {np.sqrt(actual_var):.4f}  [target: ~1.0]")

print("\n2. RECONSTRUCTION")
print(f"   Avg MSE: {np.mean(all_recon_errors):.6f}")

print("\n3. KL DIVERGENCE (estimated)")
# KL(q||p) = 0.5 * sum(1 + logvar - mu^2 - exp(logvar))
kl_per_sample = -0.5 * (1 + all_logvar - all_mu**2 - np.exp(all_logvar))
kl_mean = kl_per_sample.sum(axis=(1,2,3)).mean()
print(f"   Avg KL: {kl_mean:.4f}  [target: 0.5-2.0]")

print("\n4. DIAGNOSIS")
if all_logvar.mean() > 1.0:
    print("   ⚠️  logvar TOO HIGH (>1.0)")
    print("   → Variances too large, model too uncertain")
    print("   → SOLUTION: Reduce beta_kl (try 0.05-0.1)")
elif all_logvar.mean() > 0.3:
    print("   ⚠️  logvar slightly high (>0.3)")
    print("   → SOLUTION: Reduce beta_kl slightly (try 0.15-0.2)")
elif all_logvar.mean() < -2.0:
    print("   ⚠️  logvar TOO LOW (<-2.0)")
    print("   → Variances too small, posterior collapse")
    print("   → SOLUTION: Increase beta_kl (try 0.5-1.0)")
elif all_logvar.mean() < -0.3:
    print("   ⚠️  logvar slightly low (<-0.3)")
    print("   → SOLUTION: Increase beta_kl slightly (try 0.4-0.5)")
else:
    print("   ✅ logvar in healthy range (-0.3 to +0.3)")

if kl_mean > 5.0:
    print("   ⚠️  KL loss very high (>5.0)")
    print("   → Posterior far from prior")
    print("   → SOLUTION: Reduce beta_kl or increase lambda_vae_recon")
elif kl_mean < 0.2:
    print("   ⚠️  KL loss very low (<0.2)")
    print("   → Posterior collapsed to deterministic")
    print("   → SOLUTION: Increase beta_kl")

print("\n5. RECOMMENDED HYPERPARAMETERS")

current_beta_kl = 0.3  # Your current value
logvar_mean = all_logvar.mean()

if logvar_mean > 1.0:
    suggested_beta_kl = 0.05
elif logvar_mean > 0.5:
    suggested_beta_kl = 0.1
elif logvar_mean > 0.0:
    suggested_beta_kl = 0.2
elif logvar_mean > -1.0:
    suggested_beta_kl = 0.3
elif logvar_mean > -2.0:
    suggested_beta_kl = 0.5
else:
    suggested_beta_kl = 1.0

print(f"   beta_kl: {suggested_beta_kl}")
print(f"   lambda_vae_recon: 1.0")
print(f"   lambda_census: 0.1")

print("\n" + "="*80)
print("\nRun your training with these parameters to fix VAE health!")
print("="*80)
