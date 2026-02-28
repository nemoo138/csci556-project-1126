"""
Compare different baseline models to see which works best.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from data.dataset import MultiProductDataset
from models.baseline import BaselineDasymetric
from models.learned_baseline import LinearRegressionBaseline, CNNBaseline, ScaledDasymetric

# Configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print("Loading test dataset...")
dataset = MultiProductDataset(
    pairing_csv='data/paired_dataset/test_split.csv',
    data_root='data',
    normalize=True,
    return_census=False,
    products=['WorldPop']
)

print(f"Test dataset: {len(dataset)} samples\n")

# Initialize all baseline models
baselines = {
    'Simple Dasymetric (current)': BaselineDasymetric().to(device),
    'Scaled Dasymetric': ScaledDasymetric(init_scale=10.0).to(device),
    'Linear Regression': LinearRegressionBaseline().to(device),
    'Small CNN': CNNBaseline(hidden_channels=16).to(device),
}

# Make them trainable
for model in baselines.values():
    model.train()

# Train each baseline for a few epochs
print("Training baselines for 5 epochs...")
print("=" * 80)

for name, model in baselines.items():
    if name == 'Simple Dasymetric (current)':
        # Skip training the fixed baseline
        model.eval()
        continue

    print(f"\nTraining: {name}")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(5):
        total_loss = 0
        count = 0

        for idx in range(min(1000, len(dataset))):  # Train on first 1000 samples
            sample = dataset[idx]
            lights = sample['lights'].unsqueeze(0).to(device)
            settlement = sample['settlement'].unsqueeze(0).to(device)
            target = sample['target'].unsqueeze(0).to(device)

            # Forward
            pred = model(lights, settlement)

            # Loss: MSE on population
            loss = F.mse_loss(pred, target)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            count += 1

        avg_loss = total_loss / count
        print(f"  Epoch {epoch+1}/5: Loss = {avg_loss:.6f}")

    model.eval()

print("\n" + "=" * 80)
print("Evaluating baselines on test set (first 500 samples)...")
print("=" * 80)

# Evaluate all baselines
results = {}

for name, model in baselines.items():
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for idx in tqdm(range(min(500, len(dataset))), desc=name):
            sample = dataset[idx]
            lights = sample['lights'].unsqueeze(0).to(device)
            settlement = sample['settlement'].unsqueeze(0).to(device)
            target = sample['target'].cpu().numpy()

            # Predict
            pred = model(lights, settlement).cpu().numpy()[0]

            all_preds.append(pred)
            all_targets.append(target)

    # Stack
    all_preds = np.stack(all_preds)
    all_targets = np.stack(all_targets)

    # Compute metrics
    rmse = np.sqrt(np.mean((all_preds - all_targets) ** 2))
    mae = np.mean(np.abs(all_preds - all_targets))

    # Correlation
    from scipy.stats import pearsonr
    corr, _ = pearsonr(all_preds.flatten(), all_targets.flatten())

    # Sum ratio
    pred_sum = all_preds.sum()
    target_sum = all_targets.sum()
    sum_ratio = pred_sum / (target_sum + 1e-6)

    results[name] = {
        'rmse': rmse,
        'mae': mae,
        'corr': corr,
        'sum_ratio': sum_ratio
    }

# Print results
print("\n" + "=" * 80)
print("RESULTS")
print("=" * 80)

print(f"\n{'Baseline Model':<30} {'RMSE':<10} {'MAE':<10} {'Corr':<10} {'Sum Ratio':<10}")
print("-" * 80)

for name, metrics in results.items():
    print(f"{name:<30} {metrics['rmse']:<10.4f} {metrics['mae']:<10.4f} "
          f"{metrics['corr']:<10.4f} {metrics['sum_ratio']:<10.4f}")

print("\n" + "=" * 80)
print("RECOMMENDATIONS")
print("=" * 80)

# Find best
best_rmse = min(results.items(), key=lambda x: x[1]['rmse'])
best_corr = max(results.items(), key=lambda x: x[1]['corr'])

print(f"\nBest RMSE: {best_rmse[0]} ({best_rmse[1]['rmse']:.4f})")
print(f"Best Correlation: {best_corr[0]} ({best_corr[1]['corr']:.4f})")

print("\nRecommendation:")
if results['Linear Regression']['rmse'] < results['Simple Dasymetric (current)']['rmse'] * 0.9:
    print("  → Use Linear Regression baseline (10%+ better than current)")
    print("  → Easy to implement and train")
elif results['Small CNN']['rmse'] < results['Simple Dasymetric (current)']['rmse'] * 0.85:
    print("  → Use Small CNN baseline (15%+ better than current)")
    print("  → More complex but better performance")
elif results['Scaled Dasymetric']['rmse'] < results['Simple Dasymetric (current)']['rmse'] * 0.95:
    print("  → Use Scaled Dasymetric (5%+ better, minimal change)")
    print("  → Just adds one learnable parameter")
else:
    print("  → Keep Simple Dasymetric (learned baselines not significantly better)")
    print("  → Focus on improving diffusion model instead")

print("\n" + "=" * 80)
