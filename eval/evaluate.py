"""
Evaluation Script for C3-LDM
Reference: IMPLEMENTATION_ROADMAP.md Phase 7

Evaluates trained C3-LDM models on test datasets.
Computes comprehensive metrics including RMSE, MAE, correlation, census errors, etc.
"""

import os
import argparse
import json
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path

# Add parent directory to path
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import MultiProductDataset
from models import (
    BaselineDasymetric,
    ResidualVAE,
    TimeEmbedding,
    DualBranchConditionalEncoder,
    ProductEmbedding,
    SimpleUNet,
    C3LDMSampler
)
from eval.metrics import batch_metrics, print_metrics


def load_models(checkpoint_path, device):
    """Load trained models from checkpoint."""
    print(f"\nLoading models from: {checkpoint_path}")

    # Initialize models
    models = {
        'baseline': BaselineDasymetric().to(device),
        'vae': ResidualVAE(latent_channels=4, base_channels=64).to(device),
        'time_emb': TimeEmbedding(dim=256, base_dim=64).to(device),
        'cond_encoder': DualBranchConditionalEncoder(cond_channels=256, low_res_ch=128, high_res_ch=128).to(device),
        'product_emb': ProductEmbedding(num_products=3, d_prod=64, cond_channels=256).to(device),
        'unet': SimpleUNet(in_channels=4, model_channels=128, time_emb_dim=256, cond_channels=256).to(device)
    }

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load learned model states
    learned_models = ['vae', 'time_emb', 'cond_encoder', 'product_emb', 'unet']
    for name in learned_models:
        if name in checkpoint['models']:
            models[name].load_state_dict(checkpoint['models'][name])
            print(f"  Loaded {name}")
        else:
            print(f"  Warning: {name} not found in checkpoint")

    # Set all models to eval mode
    for model in models.values():
        model.eval()

    # Get diffusion schedule from checkpoint
    if 'config' in checkpoint:
        config = checkpoint['config']
        T = getattr(config, 'diffusion_steps', 1000)
        beta_start = getattr(config, 'beta_start', 0.0001)
        beta_end = getattr(config, 'beta_end', 0.02)
    else:
        T, beta_start, beta_end = 1000, 0.0001, 0.02

    betas = torch.linspace(beta_start, beta_end, T).to(device)

    return models, betas


def generate_samples(models, betas, lights, settlement, product_id, device,
                    num_samples=1, sampler='ddim', num_steps=50):
    """
    Generate population map samples.

    Args:
        models: Dictionary of C3-LDM models
        betas: Diffusion noise schedule
        lights: (1, 256, 256) VIIRS nighttime lights
        settlement: (1, 256, 256) WSF settlement
        product_id: Product ID (0=WorldPop, 1=GHS-POP, 2=HRSL)
        device: Device to run on
        num_samples: Number of samples to generate
        sampler: 'ddpm' or 'ddim'
        num_steps: Number of sampling steps (for DDIM)

    Returns:
        samples: (num_samples, 1, 256, 256) generated population maps
    """
    # Create sampler
    c3ldm_sampler = C3LDMSampler(
        baseline=models['baseline'],
        vae=models['vae'],
        time_emb=models['time_emb'],
        cond_encoder=models['cond_encoder'],
        product_emb=models['product_emb'],
        unet=models['unet'],
        census_layer=None,  # No census constraint for evaluation
        betas=betas,
        device=device
    )

    # Prepare inputs (handle both numpy and torch)
    if isinstance(lights, torch.Tensor):
        lights_tensor = lights.unsqueeze(0).float().to(device)  # (1, 1, 256, 256)
    else:
        lights_tensor = torch.from_numpy(lights).unsqueeze(0).float().to(device)

    if isinstance(settlement, torch.Tensor):
        settlement_tensor = settlement.unsqueeze(0).float().to(device)
    else:
        settlement_tensor = torch.from_numpy(settlement).unsqueeze(0).float().to(device)

    # Generate samples using C3LDMSampler.sample_population_map
    pop_maps = c3ldm_sampler.sample_population_map(
        lights_tensor,
        settlement_tensor,
        product_id,  # Pass as int, method will handle it
        admin_ids=None,
        census_totals=None,
        num_samples=num_samples,
        sampler=sampler,
        num_steps=num_steps,
        eta=0.0,
        show_progress=False
    )  # Returns (B, num_samples, 1, 256, 256)

    # Convert to numpy and reshape
    samples = pop_maps.cpu().numpy()  # (1, num_samples, 1, 256, 256)
    samples = samples[0]  # (num_samples, 1, 256, 256)

    return samples


def evaluate_dataset(checkpoint_path, data_root, pairing_csv, products,
                     device, num_samples=1, sampler='ddim', num_steps=50,
                     max_samples=None, output_dir=None, save_predictions=False):
    """
    Evaluate model on a dataset.

    Args:
        checkpoint_path: Path to model checkpoint
        data_root: Root directory for data
        pairing_csv: Path to pairing CSV
        products: List of products to evaluate (e.g., ['WorldPop'])
        device: Device to run on
        num_samples: Number of samples to generate per input
        sampler: 'ddpm' or 'ddim'
        num_steps: Number of DDIM steps
        max_samples: Maximum number of dataset samples to evaluate (None = all)
        output_dir: Directory to save results (None = don't save)
        save_predictions: Whether to save individual predictions

    Returns:
        metrics: Dictionary of evaluation metrics
    """
    # Load models
    models, betas = load_models(checkpoint_path, device)

    # Load dataset
    print(f"\nLoading dataset from: {pairing_csv}")
    dataset = MultiProductDataset(
        pairing_csv=pairing_csv,
        data_root=data_root,
        normalize=True,
        return_census=False,
        products=products
    )

    print(f"Dataset size: {len(dataset)} samples")
    if max_samples is not None:
        eval_size = min(max_samples, len(dataset))
        print(f"Evaluating on: {eval_size} samples")
    else:
        eval_size = len(dataset)

    # Prepare storage for predictions and targets
    all_predictions = []
    all_targets = []

    # Evaluate
    print(f"\nGenerating {num_samples} sample(s) per input using {sampler.upper()}...")
    pbar = tqdm(range(eval_size), desc="Evaluating")

    for idx in pbar:
        # Load sample
        sample = dataset[idx]
        lights = sample['lights']  # (1, 256, 256)
        settlement = sample['settlement']  # (1, 256, 256)
        target = sample['target']  # (1, 256, 256)
        product_id = sample['product_id']

        # Generate predictions
        with torch.no_grad():
            predictions = generate_samples(
                models, betas, lights, settlement, product_id, device,
                num_samples=num_samples, sampler=sampler, num_steps=num_steps
            )  # (num_samples, 1, 256, 256)

        # Store (convert to numpy if needed)
        all_predictions.append(predictions)
        if isinstance(target, torch.Tensor):
            target = target.cpu().numpy()
        all_targets.append(target)

        # Save individual predictions if requested
        if save_predictions and output_dir is not None:
            pred_dir = Path(output_dir) / 'predictions'
            pred_dir.mkdir(parents=True, exist_ok=True)

            # Save mean prediction
            pred_mean = predictions.mean(axis=0)  # (1, 256, 256)
            np.save(pred_dir / f'pred_{idx:06d}.npy', pred_mean)
            np.save(pred_dir / f'target_{idx:06d}.npy', target)

    # Stack all predictions and targets
    all_predictions = np.concatenate(all_predictions, axis=0)  # (N*num_samples, 1, 256, 256)
    all_targets = np.stack(all_targets, axis=0)  # (N, 1, 256, 256)

    # If multiple samples, average them
    if num_samples > 1:
        # Reshape to (N, num_samples, 1, 256, 256) then average
        N = eval_size
        all_predictions = all_predictions.reshape(N, num_samples, 1, 256, 256)
        all_predictions = all_predictions.mean(axis=1)  # (N, 1, 256, 256)

    print(f"\nFinal shapes:")
    print(f"  Predictions: {all_predictions.shape}")
    print(f"  Targets: {all_targets.shape}")

    # Compute metrics
    print("\nComputing metrics...")
    metrics = batch_metrics(all_predictions, all_targets)

    # Print results
    print_metrics(metrics, title=f"Evaluation Results ({eval_size} samples)")

    # Save results
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save metrics
        metrics_clean = {k: float(v) if not np.isnan(v) else None for k, v in metrics.items()}
        with open(output_dir / 'metrics.json', 'w') as f:
            json.dump(metrics_clean, f, indent=2)
        print(f"\nSaved metrics to: {output_dir / 'metrics.json'}")

        # Save summary
        with open(output_dir / 'summary.txt', 'w') as f:
            f.write(f"C3-LDM Evaluation Results\n")
            f.write(f"=" * 70 + "\n\n")
            f.write(f"Checkpoint: {checkpoint_path}\n")
            f.write(f"Dataset: {pairing_csv}\n")
            f.write(f"Products: {products}\n")
            f.write(f"Sampler: {sampler.upper()} ({num_steps} steps)\n")
            f.write(f"Samples per input: {num_samples}\n")
            f.write(f"Evaluated samples: {eval_size}\n\n")

            f.write(f"Metrics:\n")
            f.write(f"-" * 70 + "\n")
            for key, value in metrics.items():
                if np.isnan(value):
                    f.write(f"  {key:25s}: NaN\n")
                elif 'rel_error' in key or 'mape' in key:
                    f.write(f"  {key:25s}: {value:.2f}%\n")
                else:
                    f.write(f"  {key:25s}: {value:.4f}\n")

        print(f"Saved summary to: {output_dir / 'summary.txt'}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate C3-LDM model')

    # Model and data
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data_root', type=str, default='data',
                       help='Root directory for data')
    parser.add_argument('--pairing_csv', type=str,
                       default='data/paired_dataset/test_split.csv',
                       help='Path to test split CSV (default: test_split.csv from stratified 80/20 split)')
    parser.add_argument('--products', type=str, nargs='+', default=['WorldPop'],
                       help='Products to evaluate (WorldPop, GHS-POP, HRSL)')

    # Sampling
    parser.add_argument('--num_samples', type=int, default=1,
                       help='Number of samples to generate per input')
    parser.add_argument('--sampler', type=str, default='ddim',
                       choices=['ddpm', 'ddim'],
                       help='Sampler to use')
    parser.add_argument('--num_steps', type=int, default=50,
                       help='Number of DDIM steps (ignored for DDPM)')

    # Evaluation
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of samples to evaluate (None = all)')

    # Output
    parser.add_argument('--output_dir', type=str, default='eval_results',
                       help='Directory to save results')
    parser.add_argument('--save_predictions', action='store_true',
                       help='Save individual predictions')

    # Device
    parser.add_argument('--device', type=str,
                       default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use')

    args = parser.parse_args()

    print("=" * 70)
    print("C3-LDM Model Evaluation")
    print("=" * 70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Products: {args.products}")
    print(f"Sampler: {args.sampler.upper()} ({args.num_steps} steps)")
    print(f"Device: {args.device}")
    print("=" * 70)

    # Run evaluation
    metrics = evaluate_dataset(
        checkpoint_path=args.checkpoint,
        data_root=args.data_root,
        pairing_csv=args.pairing_csv,
        products=args.products,
        device=args.device,
        num_samples=args.num_samples,
        sampler=args.sampler,
        num_steps=args.num_steps,
        max_samples=args.max_samples,
        output_dir=args.output_dir,
        save_predictions=args.save_predictions
    )

    print("\n✓ Evaluation completed!")


if __name__ == "__main__":
    main()
