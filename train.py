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
    CNNBaseline,
    ResidualVAE,
    TimeEmbedding,
    DualBranchConditionalEncoder,
    ProductEmbedding,
    SimpleUNet
)
from models.census_layer import CensusConsistencyLayerVectorized
from data.dataset import MultiProductDataset
from utils.checkpoint import save_checkpoint, load_checkpoint, find_latest_checkpoint
from models.sampler import C3LDMSampler   # NEW: for quick inference sampling
import matplotlib.pyplot as plt     # NEW: to save quick preview images
from scipy.stats import pearsonr    # For correlation computation

def soft_clamp(x, min_val=-10.0, max_val=10.0):
    """Soft clamp using tanh so gradients do not vanish hard at the bounds."""
    scale = (max_val - min_val) / 2.0
    center = (max_val + min_val) / 2.0
    # Map x to (min_val, max_val) smoothly
    return center + scale * torch.tanh((x - center) / scale)

# KL loss computation - free bits can cause collapse, make it optional
def kl_loss_standard(mu, logvar):
    """Standard KL divergence without free bits."""
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

def kl_loss_free_bits(mu, logvar, free_bits=0.1):
    """KL with free bits to prevent collapse - USE WITH CAUTION."""
    # Per-dimension KL
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    # Clamp to minimum (free bits)
    kl_per_dim = torch.clamp(kl_per_dim, min=free_bits)
    return kl_per_dim.mean()

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

        # Phase 1: Baseline - use CNN baseline for better predictions
        baseline_type = getattr(self.config, 'baseline_type', 'simple')
        if baseline_type == 'cnn':
            self.baseline = CNNBaseline(hidden_channels=16).to(self.device)
            print("Using CNNBaseline (learnable, ~500 params)")
        else:
            self.baseline = BaselineDasymetric().to(self.device)
            print("Using BaselineDasymetric (fixed formula)")

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
        use_mask = getattr(self.config, 'use_missingness_mask', True)
        in_ch = 2 if use_mask else 1
        self.cond_encoder = DualBranchConditionalEncoder(
            cond_channels=self.config.cond_channels,
            low_res_ch=128,
            high_res_ch=128,
            low_in_channels=in_ch,
            high_in_channels=in_ch
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
    
    def initialize_decoder_bias(self, target_mean=0.0):
        """
        Initialize VAE decoder's final layer bias to output correct mean.
        This helps avoid the decoder bias problem where it outputs residual ≈ -4
        instead of residual ≈ 0.
        """
        # Find the last conv layer in VAE decoder
        decoder_modules = list(self.vae.decoder.modules())
        for module in reversed(decoder_modules):
            if isinstance(module, nn.Conv2d) and module.bias is not None:
                nn.init.constant_(module.bias, target_mean)
                print(f"Initialized VAE decoder final bias to {target_mean:.3f}")
                return
        print("Warning: Could not find decoder final conv layer to initialize bias")

    def build_optimizer(self):
        """Build optimizer for all trainable models."""
        # Combine all trainable parameters
        params = []
        # Add baseline if it's learnable (CNNBaseline has parameters)
        if hasattr(self.baseline, 'parameters') and len(list(self.baseline.parameters())) > 0:
            params += list(self.baseline.parameters())
            print(f"  Baseline parameters: {sum(p.numel() for p in self.baseline.parameters()):,}")
        params += list(self.vae.parameters())
        # MODIFIED: include time embedding parameters in optimizer
        params += list(self.time_emb.parameters())
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


    def _get_beta_kl_current(self):
        """KL annealing: start from 0, ramp up to config.beta_kl over kl_warmup_epochs.
        
        Starting from 0 is critical to prevent VAE collapse - it lets the encoder
        learn to encode useful information before KL pushes it toward N(0,1).
        """
        warmup = getattr(self.config, "kl_warmup_epochs", 0)
        target_beta = self.config.beta_kl

        # No warmup if kl_warmup_epochs <= 0
        if warmup <= 0:
            return target_beta

        current_epoch = max(1, self.epoch)

        if current_epoch >= warmup:
            return target_beta

        # Linear interpolation: epoch=1 -> 0, epoch=warmup -> target_beta
        # START FROM 0, not 1.0!
        frac = current_epoch / float(warmup)
        beta_kl_current = target_beta * frac  # 0 -> target_beta
        return beta_kl_current


    def _zero_loss(self):
        """Return zero loss dictionary to skip a problematic batch."""
        return {
            'loss': torch.tensor(0.0, device=self.device),
            'loss_diffusion': 0.0,
            'loss_kl': 0.0,
            'loss_recon': 0.0,
            'loss_vae_recon': 0.0
        }

    def training_step(self, batch):
        """
        Single training step.

        Follows C3-LDM.md Section "Component 6" training procedure.
        """
        # Move batch to device
        lights = batch['lights'].to(self.device)  # (B, 1, 256, 256)
        settlement = batch['settlement'].to(self.device)  # (B, 1, 256, 256)
        lights_mask = batch.get('lights_mask')
        settlement_mask = batch.get('settlement_mask')
        if lights_mask is None:
            lights_mask = torch.ones_like(lights)
        else:
            lights_mask = lights_mask.to(self.device)
        if settlement_mask is None:
            settlement_mask = torch.ones_like(settlement)
        else:
            settlement_mask = settlement_mask.to(self.device)
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
        baseline_floor = getattr(self.config, 'baseline_floor', 1e-3)
        baseline_eff = torch.clamp(baseline, min=baseline_floor)  # Prevent tiny baseline explosions

            # DEBUG: Check baseline
        if torch.isnan(baseline_eff).any() or torch.isinf(baseline_eff).any():
            print(f"\n❌ NaN/Inf in BASELINE! min={baseline_eff.min():.4f}, max={baseline_eff.max():.4f}")
            return self._zero_loss()

        # 2. Compute residual target: R_true = log((Y_true + ε) / (B + ε))
        # Clamp residual to prevent extreme values
        residual_true = torch.log((target + epsilon) / (baseline_eff + epsilon))
        # NEW: soft clamp to keep gradients
        residual_true = soft_clamp(residual_true, min_val=-10.0, max_val=10.0)

        # DEBUG: Check residual
        if torch.isnan(residual_true).any() or torch.isinf(residual_true).any():
            print(f"\n❌ NaN/Inf in RESIDUAL_TRUE! min={residual_true.min():.4f}, max={residual_true.max():.4f}")
            print(f"  target range: [{target.min():.4f}, {target.max():.4f}]")
            print(f"  baseline range: [{baseline.min():.4f}, {baseline.max():.4f}]")
            return self._zero_loss()

        # 3. Encode to latent space
        mu_z, logvar_z = self.vae.encode(residual_true)

        # In training_step, after computing residual_true:
        if self.global_step % 200 == 0:
            print(f"[GT residual] range: [{residual_true.min():.2f}, {residual_true.max():.2f}], "
                f"mean={residual_true.mean():.2f}, std={residual_true.std():.2f}")

        # DEBUG: Check VAE encoder outputs
        if torch.isnan(mu_z).any() or torch.isinf(mu_z).any():
            print(f"\n❌ NaN/Inf in MU_Z! min={mu_z.min():.4f}, max={mu_z.max():.4f}")
            return self._zero_loss()
        if torch.isnan(logvar_z).any() or torch.isinf(logvar_z).any():
            print(f"\n❌ NaN/Inf in LOGVAR_Z! min={logvar_z.min():.4f}, max={logvar_z.max():.4f}")
            return self._zero_loss()

        # Clamp logvar to prevent explosion
        logvar_z = torch.clamp(logvar_z, min=-8, max=10)
        # Reparameterization
        std = torch.exp(0.5 * logvar_z)
        eps_noise = torch.randn_like(std)
        z_0 = mu_z + eps_noise * std  # (B, 4, 32, 32)

        # Fix: Add Direct VAE Reconstruction Loss
        # NEVER detach z_0 - encoder must always get reconstruction gradients!
        # Without this, encoder only gets KL loss which pushes toward constant output (collapse)
        residual_direct = self.vae.decode(z_0)
        # Weight reconstruction toward higher-population areas
        weight = torch.sqrt(torch.clamp(target, min=0.0))
        weight = weight / (weight.mean() + 1e-6)
        loss_vae_recon = ((residual_direct - residual_true) ** 2 * weight).sum() / (weight.sum() + 1e-6)

        # NEW: Add census consistency loss (total population matching)
        # Compute predicted population from direct VAE reconstruction
        pop_direct = baseline_eff * torch.exp(residual_direct)
        pop_direct = torch.clamp(pop_direct, min=0, max=1e6)

        # Sum over spatial dimensions (B, 1, H, W) -> (B,)
        pred_total = pop_direct.sum(dim=(1, 2, 3))
        target_total = target.sum(dim=(1, 2, 3))

        # Census consistency loss: match total population
        loss_census = F.mse_loss(pred_total, target_total) / (target_total.mean() + 1e-6)  # Normalize by mean population
        
        vae_warmup = getattr(self.config, 'vae_warmup_epochs', 0)

        if self.global_step % 200 == 0:
            phase = "VAE_WARMUP" if self.epoch <= vae_warmup else "FULL"
            print(f"[Direct VAE] pred_mean={residual_direct.mean():.2f}, "
                f"pred_std={residual_direct.std():.2f}, "
                f"gt_mean={residual_true.mean():.2f}, "
                f"gt_std={residual_true.std():.2f}, "
                f"loss_vae={loss_vae_recon:.4f}, "
                f"phase={phase}")

        # After computing z_0 from encoder (around line where you have z_0 = mu_z + eps_noise * std)
        if self.global_step % 200 == 0:
            print(f"\n[TRAINING z_0] range: [{z_0.min():.3f}, {z_0.max():.3f}], "
                f"mean={z_0.mean():.3f}, std={z_0.std():.3f}")
            print(f"[TRAINING mu_z] range: [{mu_z.min():.3f}, {mu_z.max():.3f}], "
                f"mean={mu_z.mean():.3f}, std={mu_z.std():.3f}")
            
            print(f"[VAE Health] mu_z std={mu_z.std():.4f}, logvar_z mean={logvar_z.mean():.2f}")
            # Check for encoder collapse
            if mu_z.std() < 0.1:
                print("⚠️⚠️⚠️  WARNING: VAE ENCODER may be collapsing (mu_z std too low)!  ⚠️⚠️⚠️")
            # Check for decoder collapse  
            if residual_direct.std() < 0.1:
                print("⚠️⚠️⚠️  WARNING: VAE DECODER is collapsing (pred_std too low)!  ⚠️⚠️⚠️")
                print(f"         Decoder outputs constant value. Try: lower beta_kl, longer kl_warmup")
            

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
        if getattr(self.config, 'use_missingness_mask', True):
            lights_cond = torch.cat([lights, lights_mask], dim=1)
            settlement_cond = torch.cat([settlement, settlement_mask], dim=1)
        else:
            lights_cond = lights
            settlement_cond = settlement
        H_cond_spatial = self.cond_encoder(lights_cond, settlement_cond)  # (B, C, 32, 32)

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
        # Use standard KL - free bits can paradoxically cause collapse
        loss_kl = kl_loss_standard(mu_z, logvar_z)

        # 9. Optional reconstruction loss
        if self.config.lambda_recon > 0:
            # Estimate z_0 from z_t and noise prediction
            # Safe division with clamping
            z_0_hat = (z_t - sqrt_one_minus_alpha_bar_t * noise_pred) / sqrt_alpha_bar_t.clamp(min=1e-8)
            # Decode to residual
            residual_hat = self.vae.decode(z_0_hat)
            # Clamp residual to prevent overflow in exp
            # residual_hat = torch.clamp(residual_hat, min=-10, max=10)
            # NEW: soft clamp to keep gradients and test residual output
            residual_hat = soft_clamp(residual_hat, min_val=-10.0, max_val=10.0)
            if self.global_step % 200 == 0:
                with torch.no_grad():
                    print(
                        f"\n[Step {self.global_step}] residual_hat range: "
                        f"[{residual_hat.min():.2f}, {residual_hat.max():.2f}], "
                        f"mean={residual_hat.mean():.2f}, std={residual_hat.std():.2f}"
                    )
            # Convert back to population: P = B * exp(R)
            pop_raw = baseline_eff * torch.exp(residual_hat)
            pop_raw = torch.clamp(pop_raw, min=0, max=1e6)  # Prevent extreme values

            # Reconstruction loss (log space)
            loss_recon = F.l1_loss(torch.log1p(pop_raw), torch.log1p(target))
        else:
            loss_recon = torch.tensor(0.0, device=self.device)

        # # 10. Total loss
        # loss = (
        #     loss_diffusion +
        #     self.config.beta_kl * loss_kl +
        #     self.config.lambda_recon * loss_recon
        # )
        # 10. Total loss with KL annealing and VAE warmup
        # NOTE: loss_recon REMOVED - it conflicts with loss_vae_recon by pushing decoder
        # in opposite directions through the diffusion path vs direct path
        beta_kl_current = self._get_beta_kl_current()
        
        # Two-phase training: VAE only first, then full model
        vae_warmup = getattr(self.config, 'vae_warmup_epochs', 0)
        lambda_census = getattr(self.config, 'lambda_census', 0.0)  # Census consistency weight

        if self.epoch <= vae_warmup:
            # Phase 1: VAE only - focus on decoder learning correct residual mapping
            loss = (
                beta_kl_current * loss_kl +
                self.config.lambda_vae_recon * loss_vae_recon +
                lambda_census * loss_census  # Add census loss
            )
            if self.global_step % 200 == 0:
                print(f"[VAE Warmup Phase] epoch {self.epoch}/{vae_warmup}")
        else:
            # Phase 2: Full model with diffusion
            loss = (
                loss_diffusion +
                beta_kl_current * loss_kl +
                # self.config.lambda_recon * loss_recon +  # DISABLED - causes gradient conflict
                self.config.lambda_vae_recon * loss_vae_recon +  # Direct VAE supervision
                lambda_census * loss_census  # Add census loss
            )

        # Return losses
        return {
            'loss': loss,
            'loss_diffusion': loss_diffusion.item(),
            'loss_kl': loss_kl.item(),
            'loss_recon': loss_recon.item() if isinstance(loss_recon, torch.Tensor) else 0.0,
            'loss_vae_recon': loss_vae_recon.item(),
            'loss_census': loss_census.item()  # Add census loss to metrics
        }

    def train_epoch(self, train_loader, epoch):
        """Train for one epoch."""
        self.vae.train()
        # MODIFIED: time embedding should also be in train mode
        self.time_emb.train()
        self.cond_encoder.train()
        self.product_emb.train()
        self.unet.train()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        epoch_metrics = {
            'loss': 0,
            'loss_diffusion': 0,
            'loss_kl': 0,
            'loss_recon': 0,
            'loss_vae_recon': 0,
            'loss_census': 0
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
                # MODIFIED: include time embedding in gradient clipping
                *self.time_emb.parameters(),
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
                'L_vae': f"{losses['loss_vae_recon']:.4f}",
                'L_census': f"{losses['loss_census']:.4f}"
            })

            self.global_step += 1

        # Average metrics
        for key in epoch_metrics:
            epoch_metrics[key] /= len(train_loader)

        return epoch_metrics
    

    def quick_inference_epoch(self, epoch: int, num_examples: int = 1, fix_idx: bool = True):
        """Run a small DDPM sampling on a few tiles to monitor training progress."""
        if not hasattr(self, "eval_dataset"):
            print("  [quick_inference] No eval_dataset attached, skipping.")
            return

        os.makedirs(self.config.eval_output_dir, exist_ok=True)

        # Switch to eval mode for all relevant modules
        self.vae.eval()
        self.time_emb.eval()
        self.cond_encoder.eval()
        self.product_emb.eval()
        self.unet.eval()

        # Take first few samples from dataset (fixed index list for consistency)
        indices = list(range(min(num_examples, len(self.eval_dataset))))
        if fix_idx and num_examples <= 5:
            indices = [13850, 8150, 5000, 7500, 22000] # Fixed indices for monitoring
            print(f"  [quick_inference] Using fixed indices: {indices}")

        # Build diffusion schedule on current device
        betas = torch.linspace(
            self.config.beta_start,
            self.config.beta_end,
            self.config.diffusion_steps,
            device=self.device
        )

        sampler = C3LDMSampler(
            baseline=self.baseline,
            vae=self.vae,
            time_emb=self.time_emb,
            cond_encoder=self.cond_encoder,
            product_emb=self.product_emb,
            unet=self.unet,
            census_layer=self.census_layer,
            betas=betas,
            device=self.device
        )

        for idx in indices:
            sample = self.eval_dataset[idx]
            lights = sample["lights"].unsqueeze(0).to(self.device)       # (1,1,256,256)
            settlement = sample["settlement"].unsqueeze(0).to(self.device)
            lights_mask = sample.get("lights_mask")
            settlement_mask = sample.get("settlement_mask")
            if lights_mask is None:
                lights_mask = torch.ones_like(lights)
            else:
                lights_mask = lights_mask.unsqueeze(0).to(self.device)
            if settlement_mask is None:
                settlement_mask = torch.ones_like(settlement)
            else:
                settlement_mask = settlement_mask.unsqueeze(0).to(self.device)
            product_id = torch.tensor([sample["product_id"]], device=self.device, dtype=torch.long)
            gt_pop = sample["target"].squeeze().numpy()  # Ground truth



            
    

            
            with torch.no_grad():
                # Compute baseline for comparison
                baseline_map = self.baseline(lights, settlement).squeeze().cpu().numpy()
                
                # Compute what GT residual should be
                epsilon = 1e-3
                gt_residual = np.log((gt_pop + epsilon) / (baseline_map + epsilon))
                
                pop_maps = sampler.sample_population_map(
                    lights=lights,
                    settlement=settlement,
                    lights_mask=lights_mask,
                    settlement_mask=settlement_mask,
                    product_id=product_id,
                    admin_ids=None,
                    census_totals=None,
                    num_samples=1,
                    sampler="ddpm",
                    num_steps=self.config.eval_num_steps,
                    eta=0.0,
                    show_progress=False
                )  # (1,1,256,256)

            pop_map = pop_maps.squeeze().cpu().numpy()  # (256,256)
            
            # Compute predicted residual (inverse of pop = baseline * exp(residual))
            pred_residual = np.log((pop_map + epsilon) / (baseline_map + epsilon))
            
            # Print comprehensive diagnostics
            print(f"\n  [Tile {idx}] Diagnostics:")
            print(f"    GT pop:       sum={gt_pop.sum():.1f}, max={gt_pop.max():.2f}")
            print(f"    Baseline:     sum={baseline_map.sum():.1f}, max={baseline_map.max():.2f}")
            print(f"    Pred pop:     sum={pop_map.sum():.1f}, max={pop_map.max():.2f}")
            print(f"    Ratio pred/GT: {pop_map.sum() / (gt_pop.sum() + 1e-6):.4f}")
            print(f"    GT residual:   mean={gt_residual.mean():.2f}, std={gt_residual.std():.2f}")
            print(f"    Pred residual: mean={pred_residual.mean():.2f}, std={pred_residual.std():.2f}")
            print(f"    Residual gap:  {pred_residual.mean() - gt_residual.mean():.2f} (should be ~0)")
            
            # Correlation
            corr, _ = pearsonr(pop_map.flatten(), gt_pop.flatten())
            print(f"    Spatial corr:  {corr:.3f}")

            # Save as .npy
            npy_path = os.path.join(
                self.config.eval_output_dir,
                f"epoch{epoch}_idx{idx:06d}.npy"
            )
            np.save(npy_path, pop_map)

            # Save as .png
            png_path = os.path.join(
                self.config.eval_output_dir,
                f"epoch{epoch}_idx{idx:06d}.png"
            )
            plt.figure(figsize=(4, 4))
            plt.imshow(pop_map, cmap="viridis")
            plt.colorbar()
            plt.title(f"Epoch {epoch}, sample {idx}")
            plt.tight_layout()
            plt.savefig(png_path)
            plt.close()

        # Switch back to train mode; train_epoch will call .train() again
        self.vae.train()
        self.time_emb.train()
        self.cond_encoder.train()
        self.product_emb.train()
        self.unet.train()

    

    def save_checkpoint(self, metrics, is_best=False):
        """Save training checkpoint."""
        models_dict = {
            'baseline': self.baseline,  # Save baseline (learned or fixed)
            'vae': self.vae,
            # MODIFIED: save time embedding weights
            'time_emb': self.time_emb,
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
            'baseline': self.baseline,  # Load baseline
            'vae': self.vae,
            # MODIFIED: load time embedding weights
            'time_emb': self.time_emb,
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
        print(f"Training phases:")
        print(f"  Phase 1 (VAE warmup): epochs 1-{self.config.vae_warmup_epochs} - VAE only, no diffusion")
        print(f"  Phase 2 (KL warmup):  epochs 1-{self.config.kl_warmup_epochs} - beta_kl: 0 -> {self.config.beta_kl}")
        print(f"  Phase 3 (Full):       epochs {max(self.config.vae_warmup_epochs, self.config.kl_warmup_epochs)+1}+ - full model")

        for epoch in range(self.epoch + 1, self.config.num_epochs + 1):
            self.epoch = epoch
            
            # Print current phase
            vae_warmup = self.config.vae_warmup_epochs
            kl_warmup = self.config.kl_warmup_epochs
            beta_kl_now = self._get_beta_kl_current()
            if epoch <= vae_warmup:
                print(f"\n[Phase 1: VAE ONLY] Epoch {epoch}/{vae_warmup}, beta_kl={beta_kl_now:.4f}")
            elif epoch <= kl_warmup:
                print(f"\n[Phase 2: KL WARMUP] Epoch {epoch}/{kl_warmup}, beta_kl={beta_kl_now:.4f}")
            else:
                print(f"\n[Phase 3: FULL] Epoch {epoch}, beta_kl={beta_kl_now:.4f}")

            # Train epoch
            metrics = self.train_epoch(train_loader, epoch)

            # Print epoch summary
            print(f"\nEpoch {epoch} Summary:")
            print(f"  Loss: {metrics['loss']:.4f}")
            print(f"  L_diffusion: {metrics['loss_diffusion']:.4f}")
            print(f"  L_kl: {metrics['loss_kl']:.6f}")
            print(f"  L_vae_recon: {metrics['loss_vae_recon']:.4f}")
            print(f"  L_census: {metrics['loss_census']:.4f}")
            print(f"  L_recon (disabled): {metrics['loss_recon']:.4f}")

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

            # NEW: periodic quick inference
            if getattr(self.config, "eval_every", 0) > 0 and epoch % self.config.eval_every == 0:
                print(f"  ▶ Running quick inference at epoch {epoch}...")
                # number of examples to generate can be larger
                self.quick_inference_epoch(epoch, num_examples=5, fix_idx=True)

        print("\n✓ Training completed!")


def parse_args():
    parser = argparse.ArgumentParser(description="Train C3-LDM")

    # Data
    parser.add_argument('--data_root', type=str, default='data')
    parser.add_argument('--pairing_csv', type=str,
                       default='data/paired_dataset/train_split.csv',
                       help='Path to training split CSV (default: train_split.csv from stratified 80/20 split)')
    parser.add_argument('--products', type=str, nargs='+', default=['WorldPop'],
                       help='Products to train on. Default: WorldPop only. '
                            'Options: WorldPop, GHS-POP, HRSL')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)

    # Model architecture
    parser.add_argument('--baseline_type', type=str, default='simple',
                       choices=['simple', 'cnn'],
                       help='Baseline model: simple=fixed dasymetric, cnn=learned CNN')
    parser.add_argument('--time_emb_dim', type=int, default=256)
    parser.add_argument('--cond_channels', type=int, default=256)
    parser.add_argument('--no_missingness_mask', dest='use_missingness_mask',
                        action='store_false',
                        help='Disable missingness mask channels for conditioning')
    parser.set_defaults(use_missingness_mask=True)

    # Diffusion
    parser.add_argument('--diffusion_steps', type=int, default=1000)
    parser.add_argument('--beta_start', type=float, default=0.0001)
    parser.add_argument('--beta_end', type=float, default=0.02)

    # Training
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--beta_kl', type=float, default=0.01,
                        help='KL divergence weight (keep low to prevent collapse)')
    parser.add_argument('--lambda_recon', type=float, default=0.0,
                        help='Indirect reconstruction loss weight (disabled by default)')
    parser.add_argument('--lambda_vae_recon', type=float, default=100.0,
                        help='Direct VAE reconstruction loss weight (high to ensure decoder learns)')
    parser.add_argument('--lambda_census', type=float, default=0.0,
                        help='Census consistency loss weight (match total population, 0=disabled)')
    parser.add_argument('--decoder_bias_init', type=float, default=0.0,
                        help='Initialize VAE decoder bias to this value (0.0 = neutral)')
    parser.add_argument('--baseline_floor', type=float, default=1e-3,
                        help='Minimum baseline used in residualization to avoid explosion')

    parser.add_argument('--kl_warmup_epochs', type=int, default=20,
                        help='How many epochs to anneal KL weight (0 -> beta_kl)')
    parser.add_argument('--vae_warmup_epochs', type=int, default=10,
                        help='Train VAE only (no diffusion) for first N epochs')


    # Checkpointing
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--save_every', type=int, default=5)
    parser.add_argument('--keep_last_n', type=int, default=3)

    # NEW: quick inference / monitoring
    parser.add_argument('--eval_every', type=int, default=0,
                        help='If >0, run quick inference every N epochs')
    parser.add_argument('--eval_output_dir', type=str, default='quick_eval',
                        help='Directory to save quick inference outputs')
    parser.add_argument('--eval_num_steps', type=int, default=50,
                        help='Number of DDPM steps for quick inference')

    return parser.parse_args()

def detect_cond_in_channels(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state = checkpoint.get('models', {}).get('cond_encoder', {})
    low_w = state.get('low_res_branch.0.weight', None)
    high_w = state.get('high_res_branch.0.weight', None)
    low_in = low_w.shape[1] if low_w is not None else 1
    high_in = high_w.shape[1] if high_w is not None else low_in
    return low_in, high_in


def main():
    # Parse arguments
    args = parse_args()

    # Create checkpoint directory
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # Resolve resume path before building models (affects cond encoder channels)
    resume_path = args.resume
    if resume_path is None and Path(args.checkpoint_dir).exists():
        # Auto-resume from latest
        latest = find_latest_checkpoint(args.checkpoint_dir)
        if latest:
            print(f"Found checkpoint: {latest}")
            response = input("Resume from this checkpoint? [y/N]: ")
            if response.lower() == 'y':
                resume_path = latest

    args.resume = resume_path

    # If resuming, align missingness-mask setting with checkpoint
    if resume_path is not None:
        low_in, high_in = detect_cond_in_channels(resume_path)
        use_mask = (low_in > 1) or (high_in > 1)
        if args.use_missingness_mask != use_mask:
            print(f"Overriding use_missingness_mask to {use_mask} based on checkpoint")
        args.use_missingness_mask = use_mask

    # Create trainer
    trainer = C3LDMTrainer(args)
    
    # Initialize decoder bias (only for fresh training, not resume)
    fresh_start = True

    # Resume from checkpoint if requested
    if resume_path:
        trainer.load_checkpoint(resume_path)
        fresh_start = False
    
    # Initialize decoder bias for fresh training
    if fresh_start and args.decoder_bias_init != 0.0:
        trainer.initialize_decoder_bias(args.decoder_bias_init)

    # Create dataset and dataloader
    print(f"\nLoading dataset from {args.pairing_csv}...")
    print(f"Training on products: {args.products}")
    dataset = MultiProductDataset(
        pairing_csv=args.pairing_csv,
        data_root=args.data_root,
        normalize=True,
        return_census=False,
        products=args.products  # Filter to specified products
    )

    # NEW: attach dataset for quick eval
    trainer.eval_dataset = dataset

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
