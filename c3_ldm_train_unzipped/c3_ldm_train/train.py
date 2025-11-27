"""
C3-LDM Training Script
Reference: IMPLEMENTATION_ROADMAP.md Phase 5.2

Trains the complete C3-LDM model with:
- Multi-product supervision (WorldPop, GHS-POP, HRSL)
- Diffusion-based residual modeling
- Census consistency enforcement
- Checkpoint saving/loading
"""

import argparse
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import C3-LDM components
from models import (
    BaselineDasymetric,
    ResidualVAE,
    TimeEmbedding,
    DualBranchConditionalEncoder,
    ProductEmbedding,
    SimpleUNet
)
from models.census_layer import CensusConsistencyLayerVectorized
from data.dataset import MultiProductDataset
from utils.checkpoint import save_checkpoint, load_checkpoint, find_latest_checkpoint


class C3LDMTrainer:
    """Complete C3-LDM training pipeline."""

    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Build models
        self.build_models()

        # Build optimizer
        self.build_optimizer()

        # Build diffusion schedule
        self.build_diffusion_schedule()

        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')

    def build_models(self):
        """Build all C3-LDM components."""
        print("Building models...")

        # Phase 1: Baseline dasymetric
        self.baseline = BaselineDasymetric().to(self.device)

        # Phase 1: VAE for residual encoding/decoding
        self.vae = ResidualVAE(
            latent_channels=4,
            base_channels=64
        ).to(self.device)

        # Phase 2: Time embedding
        self.time_emb = TimeEmbedding(
            dim=self.config.time_emb_dim,
            base_dim=64
        ).to(self.device)

        # Phase 2: Conditional encoder (VIIRS + WSF)
        self.cond_encoder = DualBranchConditionalEncoder(
            cond_channels=self.config.cond_channels,
            low_res_ch=128,
            high_res_ch=128
        ).to(self.device)

        # Phase 2: Product embeddings
        self.product_emb = ProductEmbedding(
            num_products=3,  # WorldPop, GHS-POP, HRSL
            d_prod=64,
            cond_channels=self.config.cond_channels
        ).to(self.device)

        # Phase 3: Diffusion U-Net
        self.unet = SimpleUNet(
            in_channels=4,
            model_channels=128,
            time_emb_dim=self.config.time_emb_dim,
            cond_channels=self.config.cond_channels
        ).to(self.device)

        # Phase 4: Census consistency layer
        self.census_layer = CensusConsistencyLayerVectorized().to(self.device)

        # Initialize weights
        self._initialize_weights()

        # Print model sizes
        total_params = sum(
            sum(p.numel() for p in model.parameters())
            for model in [self.baseline, self.vae, self.time_emb,
                         self.cond_encoder, self.product_emb, self.unet]
        )
        print(f"Total parameters: {total_params:,}")

    def _initialize_weights(self):
        """Initialize model weights for numerical stability."""
        def init_module(m):
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight, gain=0.02)  # Small gain for stability
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        print("Initializing model weights...")
        self.vae.apply(init_module)
        self.time_emb.apply(init_module)
        self.cond_encoder.apply(init_module)
        self.product_emb.apply(init_module)
        self.unet.apply(init_module)

    def build_optimizer(self):
        """Build optimizer for all trainable models."""
        # Combine all trainable parameters
        params = []
        params += list(self.vae.parameters())
        params += list(self.cond_encoder.parameters())
        params += list(self.product_emb.parameters())
        params += list(self.unet.parameters())

        self.optimizer = torch.optim.AdamW(
            params,
            lr=self.config.lr,
            weight_decay=self.config.weight_decay
        )

        print(f"Optimizer: AdamW (lr={self.config.lr}, wd={self.config.weight_decay})")

    def build_diffusion_schedule(self):
        """Build linear diffusion schedule."""
        T = self.config.diffusion_steps
        beta_start = self.config.beta_start
        beta_end = self.config.beta_end

        # Linear schedule
        self.betas = torch.linspace(beta_start, beta_end, T).to(self.device)
        self.alphas = 1 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)

        print(f"Diffusion schedule: T={T}, β=[{beta_start:.4f}, {beta_end:.4f}]")

    def _zero_loss(self):
        """Return zero loss dictionary to skip a problematic batch."""
        return {
            'loss': torch.tensor(0.0, device=self.device),
            'loss_diffusion': 0.0,
            'loss_kl': 0.0,
            'loss_recon': 0.0
        }

    def training_step(self, batch):
        """
        Single training step.

        Follows C3-LDM.md Section "Component 6" training procedure.
        """
        # Move batch to device
        lights = batch['lights'].to(self.device)  # (B, 1, 256, 256)
        settlement = batch['settlement'].to(self.device)  # (B, 1, 256, 256)
        target = batch['target'].to(self.device)  # (B, 1, 256, 256)
        product_id = batch['product_id'].to(self.device)  # (B,)

        B = lights.shape[0]
        epsilon = 1e-3  # Increased epsilon for stability

        # DEBUG: Check input data
        if torch.isnan(lights).any() or torch.isinf(lights).any():
            print(f"\n❌ NaN/Inf in INPUT lights! min={lights.min():.4f}, max={lights.max():.4f}")
            return self._zero_loss()
        if torch.isnan(settlement).any() or torch.isinf(settlement).any():
            print(f"\n❌ NaN/Inf in INPUT settlement! min={settlement.min():.4f}, max={settlement.max():.4f}")
            return self._zero_loss()
        if torch.isnan(target).any() or torch.isinf(target).any():
            print(f"\n❌ NaN/Inf in INPUT target! min={target.min():.4f}, max={target.max():.4f}")
            return self._zero_loss()

        # 1. Baseline dasymetric
        with torch.no_grad():
            baseline = self.baseline(lights, settlement)  # (B, 1, 256, 256)
            baseline = torch.clamp(baseline, min=epsilon)  # Prevent zeros

            # DEBUG: Check baseline
            if torch.isnan(baseline).any() or torch.isinf(baseline).any():
                print(f"\n❌ NaN/Inf in BASELINE! min={baseline.min():.4f}, max={baseline.max():.4f}")
                return self._zero_loss()

        # 2. Compute residual target: R_true = log((Y_true + ε) / (B + ε))
        # Clamp residual to prevent extreme values
        residual_true = torch.log((target + epsilon) / (baseline + epsilon))
        residual_true = torch.clamp(residual_true, min=-10, max=10)

        # DEBUG: Check residual
        if torch.isnan(residual_true).any() or torch.isinf(residual_true).any():
            print(f"\n❌ NaN/Inf in RESIDUAL_TRUE! min={residual_true.min():.4f}, max={residual_true.max():.4f}")
            print(f"  target range: [{target.min():.4f}, {target.max():.4f}]")
            print(f"  baseline range: [{baseline.min():.4f}, {baseline.max():.4f}]")
            return self._zero_loss()

        # 3. Encode to latent space
        mu_z, logvar_z = self.vae.encode(residual_true)

        # DEBUG: Check VAE encoder outputs
        if torch.isnan(mu_z).any() or torch.isinf(mu_z).any():
            print(f"\n❌ NaN/Inf in MU_Z! min={mu_z.min():.4f}, max={mu_z.max():.4f}")
            return self._zero_loss()
        if torch.isnan(logvar_z).any() or torch.isinf(logvar_z).any():
            print(f"\n❌ NaN/Inf in LOGVAR_Z! min={logvar_z.min():.4f}, max={logvar_z.max():.4f}")
            return self._zero_loss()

        # Clamp logvar to prevent explosion
        logvar_z = torch.clamp(logvar_z, min=-10, max=10)
        # Reparameterization
        std = torch.exp(0.5 * logvar_z)
        eps_noise = torch.randn_like(std)
        z_0 = mu_z + eps_noise * std  # (B, 4, 32, 32)

        # DEBUG: Check z_0
        if torch.isnan(z_0).any() or torch.isinf(z_0).any():
            print(f"\n❌ NaN/Inf in Z_0! min={z_0.min():.4f}, max={z_0.max():.4f}")
            print(f"  std range: [{std.min():.4f}, {std.max():.4f}]")
            return self._zero_loss()

        # 4. Diffusion forward process
        # Sample random timestep for each batch element
        t = torch.randint(0, self.config.diffusion_steps, (B,), device=self.device)
        # Sample noise
        noise = torch.randn_like(z_0)
        # Forward diffusion: z_t = sqrt(α_bar_t) * z_0 + sqrt(1 - α_bar_t) * noise
        alpha_bar_t = self.alpha_bar[t].view(B, 1, 1, 1)
        sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t.clamp(min=1e-8))
        sqrt_one_minus_alpha_bar_t = torch.sqrt((1 - alpha_bar_t).clamp(min=1e-8))
        z_t = sqrt_alpha_bar_t * z_0 + sqrt_one_minus_alpha_bar_t * noise

        # DEBUG: Check z_t
        if torch.isnan(z_t).any() or torch.isinf(z_t).any():
            print(f"\n❌ NaN/Inf in Z_T! min={z_t.min():.4f}, max={z_t.max():.4f}")
            return self._zero_loss()

        # 5. Conditioning
        # Spatial conditioning from VIIRS + WSF
        H_cond_spatial = self.cond_encoder(lights, settlement)  # (B, C, 32, 32)

        # DEBUG: Check spatial conditioning
        if torch.isnan(H_cond_spatial).any() or torch.isinf(H_cond_spatial).any():
            print(f"\n❌ NaN/Inf in H_COND_SPATIAL! min={H_cond_spatial.min():.4f}, max={H_cond_spatial.max():.4f}")
            return self._zero_loss()

        # Product conditioning
        H_cond_product = self.product_emb(product_id)  # (B, C, 1, 1)

        # DEBUG: Check product conditioning
        if torch.isnan(H_cond_product).any() or torch.isinf(H_cond_product).any():
            print(f"\n❌ NaN/Inf in H_COND_PRODUCT! min={H_cond_product.min():.4f}, max={H_cond_product.max():.4f}")
            return self._zero_loss()

        # Combined conditioning
        H_cond = H_cond_spatial + H_cond_product  # Broadcast add

        # DEBUG: Check combined conditioning
        if torch.isnan(H_cond).any() or torch.isinf(H_cond).any():
            print(f"\n❌ NaN/Inf in H_COND! min={H_cond.min():.4f}, max={H_cond.max():.4f}")
            return self._zero_loss()

        # Time embedding
        t_emb = self.time_emb(t)  # (B, time_emb_dim)

        # DEBUG: Check time embedding
        if torch.isnan(t_emb).any() or torch.isinf(t_emb).any():
            print(f"\n❌ NaN/Inf in T_EMB! min={t_emb.min():.4f}, max={t_emb.max():.4f}")
            return self._zero_loss()

        # 6. Predict noise
        noise_pred = self.unet(z_t, t_emb, H_cond)  # (B, 4, 32, 32)

        # DEBUG: Check U-Net output
        if torch.isnan(noise_pred).any() or torch.isinf(noise_pred).any():
            print(f"\n❌ NaN/Inf in NOISE_PRED from U-Net!")
            print(f"  z_t range: [{z_t.min():.4f}, {z_t.max():.4f}]")
            print(f"  t_emb range: [{t_emb.min():.4f}, {t_emb.max():.4f}]")
            print(f"  H_cond range: [{H_cond.min():.4f}, {H_cond.max():.4f}]")
            return self._zero_loss()

        # 7. Diffusion loss (MSE between predicted and true noise)
        loss_diffusion = F.mse_loss(noise_pred, noise)

        # 8. KL divergence loss (regularize VAE latent space)
        loss_kl = -0.5 * torch.mean(1 + logvar_z - mu_z.pow(2) - logvar_z.exp())

        # 9. Optional reconstruction loss
        if self.config.lambda_recon > 0:
            # Estimate z_0 from z_t and noise prediction
            # Safe division with clamping
            z_0_hat = (z_t - sqrt_one_minus_alpha_bar_t * noise_pred) / sqrt_alpha_bar_t.clamp(min=1e-8)
            # Decode to residual
            residual_hat = self.vae.decode(z_0_hat)
            # Clamp residual to prevent overflow in exp
            residual_hat = torch.clamp(residual_hat, min=-10, max=10)
            # Convert back to population: P = B * exp(R)
            pop_raw = baseline * torch.exp(residual_hat)
            pop_raw = torch.clamp(pop_raw, min=0, max=1e6)  # Prevent extreme values

            # Reconstruction loss (log space)
            loss_recon = F.l1_loss(torch.log1p(pop_raw), torch.log1p(target))
        else:
            loss_recon = torch.tensor(0.0, device=self.device)

        # 10. Total loss
        loss = (
            loss_diffusion +
            self.config.beta_kl * loss_kl +
            self.config.lambda_recon * loss_recon
        )

        # Return losses
        return {
            'loss': loss,
            'loss_diffusion': loss_diffusion.item(),
            'loss_kl': loss_kl.item(),
            'loss_recon': loss_recon.item() if isinstance(loss_recon, torch.Tensor) else 0.0
        }

    def train_epoch(self, train_loader, epoch):
        """Train for one epoch."""
        self.vae.train()
        self.cond_encoder.train()
        self.product_emb.train()
        self.unet.train()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        epoch_metrics = {
            'loss': 0,
            'loss_diffusion': 0,
            'loss_kl': 0,
            'loss_recon': 0
        }

        for batch_idx, batch in enumerate(pbar):
            # Training step
            losses = self.training_step(batch)

            # Skip batch if zero loss (indicates problematic data)
            if not losses['loss'].requires_grad:
                print(f"\nSkipping batch {batch_idx} (zero loss from bad data)")
                continue

            # Backward
            self.optimizer.zero_grad()
            losses['loss'].backward()

            # Clip gradients for all trainable models
            grad_norm = torch.nn.utils.clip_grad_norm_([
                *self.vae.parameters(),
                *self.cond_encoder.parameters(),
                *self.product_emb.parameters(),
                *self.unet.parameters()
            ], max_norm=1.0)

            # Check for NaN gradients
            if torch.isnan(grad_norm):
                print(f"\nWarning: NaN gradient detected at batch {batch_idx}")
                continue  # Skip this batch

            self.optimizer.step()

            # Update metrics
            for key in epoch_metrics:
                if key in losses:
                    val = losses[key].item() if isinstance(losses[key], torch.Tensor) else losses[key]
                    epoch_metrics[key] += val

            # Update progress bar
            pbar.set_postfix({
                'L_diff': f"{losses['loss_diffusion']:.4f}",
                'L_kl': f"{losses['loss_kl']:.6f}",
                'L_recon': f"{losses['loss_recon']:.4f}"
            })

            self.global_step += 1

        # Average metrics
        for key in epoch_metrics:
            epoch_metrics[key] /= len(train_loader)

        return epoch_metrics

    def save_checkpoint(self, metrics, is_best=False):
        """Save training checkpoint."""
        models_dict = {
            'vae': self.vae,
            'cond_encoder': self.cond_encoder,
            'product_emb': self.product_emb,
            'unet': self.unet
        }

        optimizers_dict = {
            'optimizer': self.optimizer
        }

        save_checkpoint(
            checkpoint_dir=self.config.checkpoint_dir,
            epoch=self.epoch,
            step=self.global_step,
            models_dict=models_dict,
            optimizers_dict=optimizers_dict,
            metrics=metrics,
            is_best=is_best,
            keep_last_n=self.config.keep_last_n
        )

    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint and resume training."""
        models_dict = {
            'vae': self.vae,
            'cond_encoder': self.cond_encoder,
            'product_emb': self.product_emb,
            'unet': self.unet
        }

        optimizers_dict = {
            'optimizer': self.optimizer
        }

        checkpoint_info = load_checkpoint(
            checkpoint_path=checkpoint_path,
            models_dict=models_dict,
            optimizers_dict=optimizers_dict,
            device=self.device
        )

        self.epoch = checkpoint_info['epoch']
        self.global_step = checkpoint_info['step']
        if 'loss' in checkpoint_info['metrics']:
            self.best_loss = checkpoint_info['metrics']['loss']

        return checkpoint_info

    def train(self, train_loader):
        """Main training loop."""
        print("\n" + "=" * 70)
        print("Starting C3-LDM Training")
        print("=" * 70)

        for epoch in range(self.epoch + 1, self.config.num_epochs + 1):
            self.epoch = epoch

            # Train epoch
            metrics = self.train_epoch(train_loader, epoch)

            # Print epoch summary
            print(f"\nEpoch {epoch} Summary:")
            print(f"  Loss: {metrics['loss']:.4f}")
            print(f"  L_diffusion: {metrics['loss_diffusion']:.4f}")
            print(f"  L_kl: {metrics['loss_kl']:.6f}")
            print(f"  L_recon: {metrics['loss_recon']:.4f}")

            # Save checkpoint
            is_best = metrics['loss'] < self.best_loss
            if is_best:
                self.best_loss = metrics['loss']
                # save the best model
                self.save_checkpoint(metrics, is_best=is_best)
                print(f"  ✓ New best model epoch {epoch} saved!")

            if epoch % self.config.save_every == 0:
                self.save_checkpoint(metrics, is_best=is_best)
                print(f"  ✓ Checkpoint epoch {epoch} saved!")

        print("\n✓ Training completed!")


def parse_args():
    parser = argparse.ArgumentParser(description="Train C3-LDM")

    # Data
    parser.add_argument('--data_root', type=str, default='data')
    parser.add_argument('--pairing_csv', type=str,
                       default='data/paired_dataset/multi_product_pairing.csv')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)

    # Model architecture
    parser.add_argument('--time_emb_dim', type=int, default=256)
    parser.add_argument('--cond_channels', type=int, default=256)

    # Diffusion
    parser.add_argument('--diffusion_steps', type=int, default=1000)
    parser.add_argument('--beta_start', type=float, default=0.0001)
    parser.add_argument('--beta_end', type=float, default=0.02)

    # Training
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--beta_kl', type=float, default=0.0001)
    parser.add_argument('--lambda_recon', type=float, default=0.1)

    # Checkpointing
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--save_every', type=int, default=5)
    parser.add_argument('--keep_last_n', type=int, default=3)

    return parser.parse_args()


def main():
    # Parse arguments
    args = parse_args()

    # Create checkpoint directory
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # Create trainer
    trainer = C3LDMTrainer(args)

    # Resume from checkpoint if requested
    if args.resume:
        trainer.load_checkpoint(args.resume)
    elif Path(args.checkpoint_dir).exists():
        # Auto-resume from latest
        latest = find_latest_checkpoint(args.checkpoint_dir)
        if latest:
            print(f"Found checkpoint: {latest}")
            response = input("Resume from this checkpoint? [y/N]: ")
            if response.lower() == 'y':
                trainer.load_checkpoint(latest)

    # Create dataset and dataloader
    print(f"\nLoading dataset from {args.pairing_csv}...")
    dataset = MultiProductDataset(
        pairing_csv=args.pairing_csv,
        data_root=args.data_root,
        normalize=True,
        return_census=False
    )

    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # Train
    trainer.train(train_loader)


if __name__ == "__main__":
    main()
