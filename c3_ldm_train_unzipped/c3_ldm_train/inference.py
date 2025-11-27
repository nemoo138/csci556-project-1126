"""
C3-LDM Inference Script
Reference: IMPLEMENTATION_ROADMAP.md Phase 6

Generate population maps from VIIRS + WSF features using trained C3-LDM model.
"""

import argparse
import os
from pathlib import Path
import numpy as np
import torch

# Import C3-LDM components
from models import (
    BaselineDasymetric,
    ResidualVAE,
    TimeEmbedding,
    DualBranchConditionalEncoder,
    ProductEmbedding,
    SimpleUNet,
    CensusConsistencyLayerVectorized
)
from models.sampler import C3LDMSampler
from utils.checkpoint import load_model_only


def load_models_from_checkpoint(checkpoint_path, device='cpu'):
    """
    Load all C3-LDM models from a checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load models to

    Returns:
        models_dict: Dictionary of loaded models
    """
    print(f"Loading models from: {checkpoint_path}")

    # Initialize models
    models = {
        'baseline': BaselineDasymetric().to(device),
        'vae': ResidualVAE(latent_channels=4, base_channels=64).to(device),
        'time_emb': TimeEmbedding(dim=256, base_dim=64).to(device),
        'cond_encoder': DualBranchConditionalEncoder(cond_channels=256, low_res_ch=128, high_res_ch=128).to(device),
        'product_emb': ProductEmbedding(num_products=3, d_prod=64, cond_channels=256).to(device),
        'unet': SimpleUNet(in_channels=4, model_channels=128, time_emb_dim=256, cond_channels=256).to(device),
        'census_layer': CensusConsistencyLayerVectorized().to(device)
    }

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load model states
    for name, model in models.items():
        if name in checkpoint['models']:
            model.load_state_dict(checkpoint['models'][name])
            print(f"  Loaded {name}")
        else:
            print(f"  Warning: {name} not found in checkpoint")

    # Set all models to eval mode
    for model in models.values():
        model.eval()

    return models


def generate_population_map(
    condition,
    product_id,
    models,
    betas,
    device='cpu',
    num_samples=1,
    sampler='ddim',
    num_steps=50,
    eta=0.0,
    admin_ids=None,
    census_totals=None
):
    """
    Generate population map from input features.

    Args:
        lights: (1, 256, 256) VIIRS nighttime lights
        settlement: (1, 256, 256) WSF settlement footprint
        product_id: int, product ID (0=WorldPop, 1=GHS-POP, 2=HRSL)
        models: Dictionary of C3-LDM models
        betas: Diffusion noise schedule
        device: Device to run on
        num_samples: Number of samples to generate
        sampler: 'ddpm' or 'ddim'
        num_steps: Number of sampling steps (for DDIM)
        eta: Stochasticity parameter (for DDIM)
        admin_ids: (256, 256) admin unit IDs (optional)
        census_totals: (max_admin_units,) census totals (optional)

    Returns:
        population_maps: (num_samples, 1, 256, 256) generated population maps
    """
    # San check for condition shape
    if condition.shape[0] != 2:
        raise ValueError(
            f"Expected condition shape (2, 256, 256), got {condition.shape}"
        )
    
    lights = condition[0:1, ...]      # (1, 256, 256)
    settlement = condition[1:2, ...]  # (1, 256, 256)
    
    # Create sampler
    c3ldm_sampler = C3LDMSampler(
        baseline=models['baseline'],
        vae=models['vae'],
        time_emb=models['time_emb'],
        cond_encoder=models['cond_encoder'],
        product_emb=models['product_emb'],
        unet=models['unet'],
        census_layer=models['census_layer'],
        betas=betas,
        device=device
    )

    # Prepare inputs
    lights_tensor = torch.from_numpy(lights).unsqueeze(0).float()  # (1, 1, 256, 256)
    settlement_tensor = torch.from_numpy(settlement).unsqueeze(0).float()  # (1, 1, 256, 256)

    if admin_ids is not None:
        admin_ids_tensor = torch.from_numpy(admin_ids).unsqueeze(0).long()  # (1, 256, 256)
    else:
        admin_ids_tensor = None

    if census_totals is not None:
        census_totals_tensor = torch.from_numpy(census_totals).unsqueeze(0).float()  # (1, max_admin_units)
    else:
        census_totals_tensor = None

    # Generate
    with torch.no_grad():
        pop_maps = c3ldm_sampler.sample_population_map(
            lights=lights_tensor,
            settlement=settlement_tensor,
            product_id=product_id,
            admin_ids=admin_ids_tensor,
            census_totals=census_totals_tensor,
            num_samples=num_samples,
            sampler=sampler,
            num_steps=num_steps,
            eta=eta,
            show_progress=True
        )

    # Convert to numpy: (num_samples, 1, 256, 256)
    pop_maps_np = pop_maps.squeeze(0).cpu().numpy()  # (num_samples, 1, 256, 256)

    return pop_maps_np


def main():
    parser = argparse.ArgumentParser(description="C3-LDM Inference")

    # Input/output
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    # parser.add_argument('--lights', type=str, required=True,
    #                    help='Path to VIIRS nighttime lights (.npy)')
    # parser.add_argument('--settlement', type=str, required=True,
    #                    help='Path to WSF settlement footprint (.npy)')
    parser.add_argument('--condition', type=str, required=True,
                    help='Path to stacked VIIRS+WSF input (.npy), shape (2,256,256)')
    parser.add_argument('--output', type=str, required=True,
                       help='Path to save output population map (.npy)')

    # Model settings
    parser.add_argument('--product_id', type=int, default=0,
                       help='Product ID: 0=WorldPop, 1=GHS-POP, 2=HRSL')
    parser.add_argument('--num_samples', type=int, default=1,
                       help='Number of samples to generate')

    # Sampling settings
    parser.add_argument('--sampler', type=str, default='ddim',
                       choices=['ddpm', 'ddim'],
                       help='Sampling method')
    parser.add_argument('--num_steps', type=int, default=50,
                       help='Number of sampling steps (for DDIM)')
    parser.add_argument('--eta', type=float, default=0.0,
                       help='Stochasticity parameter (0=deterministic, 1=stochastic)')

    # Census consistency (optional)
    parser.add_argument('--admin_ids', type=str, default=None,
                       help='Path to admin unit IDs (.npy)')
    parser.add_argument('--census_totals', type=str, default=None,
                       help='Path to census totals (.npy)')

    # Diffusion schedule
    parser.add_argument('--diffusion_steps', type=int, default=1000)
    parser.add_argument('--beta_start', type=float, default=0.0001)
    parser.add_argument('--beta_end', type=float, default=0.02)

    # Device
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()

    # Load input data
    print("\n" + "=" * 70)
    print("C3-LDM Inference")
    print("=" * 70)

    print(f"\nLoading inputs...")
    # lights = np.load(args.lights)  # (1, 256, 256)
    # settlement = np.load(args.settlement)  # (1, 256, 256)
    condition = np.load(args.condition)  # (2, 256, 256)
    print(f"  Condition shape: {condition.shape}")

    if condition.shape[0] != 2:
        raise ValueError(f"Expected 2 channels (VIIRS + WSF), got shape {condition.shape}")
    # print(f"  Lights shape: {lights.shape}")
    # print(f"  Settlement shape: {settlement.shape}")

    # Load census data if provided
    if args.admin_ids and args.census_totals:
        print(f"\nLoading census data...")
        admin_ids = np.load(args.admin_ids)
        census_totals = np.load(args.census_totals)
        print(f"  Admin IDs shape: {admin_ids.shape}")
        print(f"  Census totals shape: {census_totals.shape}")
    else:
        admin_ids = None
        census_totals = None

    # Load models
    models = load_models_from_checkpoint(args.checkpoint, device=args.device)

    # Build diffusion schedule
    betas = torch.linspace(args.beta_start, args.beta_end, args.diffusion_steps)

    # Generate population map
    print(f"\nGenerating population maps...")
    print(f"  Product ID: {args.product_id}")
    print(f"  Sampler: {args.sampler}")
    if args.sampler == 'ddim':
        print(f"  Steps: {args.num_steps}")
        print(f"  Eta: {args.eta}")
    print(f"  Num samples: {args.num_samples}")

    pop_maps = generate_population_map(
        # lights=lights,
        # settlement=settlement,
        condition=condition,
        product_id=args.product_id,
        models=models,
        betas=betas,
        device=args.device,
        num_samples=args.num_samples,
        sampler=args.sampler,
        num_steps=args.num_steps,
        eta=args.eta,
        admin_ids=admin_ids,
        census_totals=census_totals
    )

    # Save output
    print(f"\nSaving output to: {args.output}")
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    np.save(args.output, pop_maps)

    print(f"  Output shape: {pop_maps.shape}")
    print(f"  Population range: [{pop_maps.min():.2f}, {pop_maps.max():.2f}]")
    print(f"  Total population: {pop_maps.sum():.0f}")

    if args.num_samples > 1:
        print(f"\nSample statistics:")
        mean_map = pop_maps.mean(axis=0)
        std_map = pop_maps.std(axis=0)
        print(f"  Mean population: {mean_map.sum():.0f}")
        print(f"  Std dev (spatial avg): {std_map.mean():.2f}")

    print("\n✓ Inference completed!")


if __name__ == "__main__":
    main()
