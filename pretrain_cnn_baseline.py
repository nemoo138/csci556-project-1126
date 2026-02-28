"""Pre-train CNN baseline before full model training."""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from data.dataset import MultiProductDataset
from models import CNNBaseline

# Configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'
num_epochs = 20
batch_size = 128
lr = 1e-3

print("Loading dataset...")
dataset = MultiProductDataset(
    pairing_csv='data/paired_dataset/train_split.csv',
    data_root='data',
    normalize=True,
    return_census=False,
    products=['WorldPop']
)

loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

print(f"Training set: {len(dataset)} samples")

# Initialize CNN baseline
baseline = CNNBaseline(hidden_channels=16).to(device)
optimizer = torch.optim.Adam(baseline.parameters(), lr=lr)

print(f"\nCNN Baseline parameters: {sum(p.numel() for p in baseline.parameters()):,}")

# Training loop
print("\n" + "="*80)
print("PRE-TRAINING CNN BASELINE")
print("="*80)

for epoch in range(num_epochs):
    baseline.train()

    total_mse = 0
    total_census = 0
    count = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{num_epochs}")

    for batch in pbar:
        lights = batch['lights'].to(device)
        settlement = batch['settlement'].to(device)
        target = batch['target'].to(device)

        # Forward
        pred = baseline(lights, settlement)

        # Loss: MSE + census consistency
        loss_mse = F.mse_loss(pred, target)

        # Census loss: match total population
        pred_total = pred.sum(dim=(1,2,3))
        target_total = target.sum(dim=(1,2,3))
        loss_census = F.mse_loss(pred_total, target_total) / (target_total.mean() + 1e-6)

        # Combined loss
        loss = loss_mse + 0.1 * loss_census

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track
        total_mse += loss_mse.item()
        total_census += loss_census.item()
        count += 1

        # Update progress bar
        pbar.set_postfix({
            'MSE': f'{loss_mse.item():.4f}',
            'Census': f'{loss_census.item():.2f}'
        })

    avg_mse = total_mse / count
    avg_census = total_census / count

    print(f"\nEpoch {epoch+1}: MSE={avg_mse:.4f}, Census={avg_census:.2f}")

    # Evaluate on a few samples
    if (epoch + 1) % 5 == 0:
        baseline.eval()
        with torch.no_grad():
            sample = dataset[0]
            lights = sample['lights'].unsqueeze(0).to(device)
            settlement = sample['settlement'].unsqueeze(0).to(device)
            target = sample['target'].cpu().numpy()

            pred = baseline(lights, settlement).cpu().numpy()[0, 0]

            print(f"  Sample check:")
            print(f"    Target sum:     {target.sum():.1f}")
            print(f"    Prediction sum: {pred.sum():.1f}")
            print(f"    Ratio:          {pred.sum() / (target.sum() + 1e-6):.3f}")

# Save pre-trained baseline
os.makedirs('checkpoints_cnn_pretrain', exist_ok=True)
torch.save({
    'baseline_state_dict': baseline.state_dict(),
    'epoch': num_epochs,
}, 'checkpoints_cnn_pretrain/pretrained_baseline.pt')

print("\n" + "="*80)
print("Pre-training complete!")
print("Saved to: checkpoints_cnn_pretrain/pretrained_baseline.pt")
print("\nNow modify train.py to load this pre-trained baseline, or")
print("just start training with lower lambda_census (0.001-0.01)")
print("="*80)
