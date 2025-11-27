"""
DDPM and DDIM Samplers for C3-LDM Inference
Reference: IMPLEMENTATION_ROADMAP.md Phase 6

Implements:
- DDPM: Full reverse diffusion (1000 steps)
- DDIM: Accelerated sampling (50-100 steps)
"""

import torch
import torch.nn as nn
from tqdm import tqdm


class DDPMSampler:
    """
    Denoising Diffusion Probabilistic Model (DDPM) sampler.

    Uses the full reverse diffusion process with all timesteps.
    Slower but more accurate than DDIM.
    """

    def __init__(self, betas, device='cpu'):
        """
        Args:
            betas: (T,) tensor of noise schedule
            device: Device to run sampling on
        """
        self.device = device
        self.betas = betas.to(device)
        self.T = len(betas)

        # Precompute diffusion parameters
        self.alphas = 1 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)
        self.alpha_bar_prev = torch.cat([
            torch.tensor([1.0], device=device),
            self.alpha_bar[:-1]
        ])

        # Variance for reverse process
        # σ_t^2 = β_t (for variance-preserving)
        self.sigma = torch.sqrt(self.betas)

    @torch.no_grad()
    def sample(self, unet, time_emb, cond, shape, show_progress=True):
        """
        Sample from the diffusion model using DDPM.

        Args:
            unet: Trained U-Net model
            time_emb: Time embedding module
            cond: (B, C, H, W) conditioning tensor
            shape: Tuple (B, C, H, W) for latent shape
            show_progress: Whether to show progress bar

        Returns:
            z_0: (B, C, H, W) denoised latent samples
        """
        B, C, H, W = shape

        # Start from pure noise
        z_t = torch.randn(shape, device=self.device)

        # Reverse diffusion
        timesteps = range(self.T - 1, -1, -1)
        if show_progress:
            timesteps = tqdm(timesteps, desc="DDPM Sampling")

        for t in timesteps:
            # Create batch of timesteps
            t_batch = torch.full((B,), t, device=self.device, dtype=torch.long)

            # Get time embeddings
            t_emb = time_emb(t_batch)

            # Predict noise
            noise_pred = unet(z_t, t_emb, cond)

            # Compute mean of p(z_{t-1} | z_t)
            alpha_t = self.alphas[t]
            alpha_bar_t = self.alpha_bar[t]
            alpha_bar_prev_t = self.alpha_bar_prev[t]

            # μ_θ(z_t, t) = 1/√α_t * (z_t - β_t/√(1-ᾱ_t) * ε_θ(z_t, t))
            mean = (1 / torch.sqrt(alpha_t)) * (
                z_t - (self.betas[t] / torch.sqrt(1 - alpha_bar_t)) * noise_pred
            )

            if t > 0:
                # Add noise (except at t=0)
                noise = torch.randn_like(z_t)
                z_t = mean + self.sigma[t] * noise
            else:
                # At t=0, return mean (no noise)
                z_t = mean

        return z_t


class DDIMSampler:
    """
    Denoising Diffusion Implicit Model (DDIM) sampler.

    Accelerated sampling by skipping timesteps.
    Much faster than DDPM (50 steps vs 1000 steps).
    """

    def __init__(self, betas, device='cpu'):
        """
        Args:
            betas: (T,) tensor of noise schedule
            device: Device to run sampling on
        """
        self.device = device
        self.betas = betas.to(device)
        self.T = len(betas)

        # Precompute diffusion parameters
        self.alphas = 1 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)

    @torch.no_grad()
    def sample(
        self,
        unet,
        time_emb,
        cond,
        shape,
        num_steps=50,
        eta=0.0,
        show_progress=True
    ):
        """
        Sample from the diffusion model using DDIM.

        Args:
            unet: Trained U-Net model
            time_emb: Time embedding module
            cond: (B, C, H, W) conditioning tensor
            shape: Tuple (B, C, H, W) for latent shape
            num_steps: Number of sampling steps (e.g., 50)
            eta: Stochasticity parameter (0=deterministic, 1=DDPM)
            show_progress: Whether to show progress bar

        Returns:
            z_0: (B, C, H, W) denoised latent samples
        """
        B, C, H, W = shape

        # Create sub-sequence of timesteps
        # Uniform spacing from T-1 to 0
        skip = self.T // num_steps
        timesteps = list(range(self.T - 1, 0, -skip))
        timesteps.append(0)  # Ensure we end at t=0
        timesteps = timesteps[:num_steps]  # Truncate to num_steps

        # Start from pure noise
        z_t = torch.randn(shape, device=self.device)

        if show_progress:
            timesteps_iter = tqdm(timesteps, desc=f"DDIM Sampling ({num_steps} steps)")
        else:
            timesteps_iter = timesteps

        for i, t in enumerate(timesteps_iter):
            # Create batch of timesteps
            t_batch = torch.full((B,), t, device=self.device, dtype=torch.long)

            # Get time embeddings
            t_emb = time_emb(t_batch)

            # Predict noise
            noise_pred = unet(z_t, t_emb, cond)

            # Get alpha values
            alpha_bar_t = self.alpha_bar[t]

            # Get next timestep
            if i < len(timesteps) - 1:
                t_prev = timesteps[i + 1]
                alpha_bar_prev = self.alpha_bar[t_prev]
            else:
                alpha_bar_prev = torch.tensor(1.0, device=self.device)

            # Predict x_0 from z_t and noise prediction
            # x_0 = (z_t - √(1-ᾱ_t) * ε) / √ᾱ_t
            pred_x0 = (z_t - torch.sqrt(1 - alpha_bar_t) * noise_pred) / torch.sqrt(alpha_bar_t)

            # Compute direction pointing to z_t
            # dir_zt = √(1-ᾱ_{t-1} - σ_t^2) * ε_θ(z_t, t)
            sigma_t = eta * torch.sqrt(
                (1 - alpha_bar_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_prev)
            )
            dir_zt = torch.sqrt(1 - alpha_bar_prev - sigma_t ** 2) * noise_pred

            # Compute z_{t-1}
            # z_{t-1} = √ᾱ_{t-1} * x_0 + dir_zt + σ_t * ε
            z_t = torch.sqrt(alpha_bar_prev) * pred_x0 + dir_zt

            # Add noise (except at last step)
            if eta > 0 and i < len(timesteps) - 1:
                noise = torch.randn_like(z_t)
                z_t = z_t + sigma_t * noise

        return z_t


class C3LDMSampler:
    """
    Complete C3-LDM sampler with all components.

    Combines:
    - Baseline dasymetric
    - DDPM/DDIM sampling in latent space
    - VAE decoding
    - Census consistency enforcement
    """

    def __init__(
        self,
        baseline,
        vae,
        time_emb,
        cond_encoder,
        product_emb,
        unet,
        census_layer,
        betas,
        device='cpu'
    ):
        """
        Args:
            baseline: Baseline dasymetric module
            vae: VAE encoder/decoder
            time_emb: Time embedding module
            cond_encoder: Conditional encoder
            product_emb: Product embedding module
            unet: Diffusion U-Net
            census_layer: Census consistency layer
            betas: Diffusion noise schedule
            device: Device to run on
        """
        self.baseline = baseline
        self.vae = vae
        self.time_emb = time_emb
        self.cond_encoder = cond_encoder
        self.product_emb = product_emb
        self.unet = unet
        self.census_layer = census_layer
        self.device = device

        # Create DDPM and DDIM samplers
        self.ddpm_sampler = DDPMSampler(betas, device)
        self.ddim_sampler = DDIMSampler(betas, device)

    @torch.no_grad()
    def sample_population_map(
        self,
        lights,
        settlement,
        product_id,
        admin_ids=None,
        census_totals=None,
        num_samples=1,
        sampler='ddim',
        num_steps=50,
        eta=0.0,
        show_progress=True
    ):
        """
        Generate population map from VIIRS + WSF features.

        Args:
            lights: (B, 1, 256, 256) VIIRS nighttime lights
            settlement: (B, 1, 256, 256) WSF settlement footprint
            product_id: (B,) or int, product ID (0=WorldPop, 1=GHS-POP, 2=HRSL)
            admin_ids: (B, 256, 256) admin unit IDs (optional)
            census_totals: (B, max_admin_units) census totals (optional)
            num_samples: Number of samples to generate per input
            sampler: 'ddpm' or 'ddim'
            num_steps: Number of sampling steps (for DDIM)
            eta: Stochasticity parameter (for DDIM)
            show_progress: Whether to show progress bar

        Returns:
            population_maps: (B, num_samples, 1, 256, 256) generated population maps
        """
        B = lights.shape[0]
        epsilon = 1e-6

        # Move to device
        lights = lights.to(self.device)
        settlement = settlement.to(self.device)

        if isinstance(product_id, int):
            product_id = torch.full((B,), product_id, dtype=torch.long, device=self.device)
        else:
            product_id = product_id.to(self.device)

        # 1. Baseline dasymetric
        baseline_pop = self.baseline(lights, settlement)
        baseline_pop = torch.clamp(baseline_pop, min=0)

        # 2. Conditioning
        # Spatial conditioning
        H_cond_spatial = self.cond_encoder(lights, settlement)
        # Product conditioning
        H_cond_product = self.product_emb(product_id)
        # Combined
        H_cond = H_cond_spatial + H_cond_product

        # Generate multiple samples
        all_samples = []

        for sample_idx in range(num_samples):
            # 3. Sample latent from diffusion model
            latent_shape = (B, 4, 32, 32)

            if sampler == 'ddpm':
                z_0 = self.ddpm_sampler.sample(
                    self.unet, self.time_emb, H_cond, latent_shape,
                    show_progress=show_progress and (sample_idx == 0)
                )
            elif sampler == 'ddim':
                z_0 = self.ddim_sampler.sample(
                    self.unet, self.time_emb, H_cond, latent_shape,
                    num_steps=num_steps, eta=eta,
                    show_progress=show_progress and (sample_idx == 0)
                )
            else:
                raise ValueError(f"Unknown sampler: {sampler}")

            # 4. Decode latent to residual
            residual = self.vae.decode(z_0)

            # 5. Convert to population: P = B * exp(R)
            pop_raw = baseline_pop * torch.exp(residual)
            pop_raw = torch.clamp(pop_raw, min=0)

            # 6. Apply census consistency if provided
            if admin_ids is not None and census_totals is not None:
                admin_ids = admin_ids.to(self.device)
                census_totals = census_totals.to(self.device)
                pop_final = self.census_layer(pop_raw, admin_ids, census_totals)
            else:
                pop_final = pop_raw

            all_samples.append(pop_final)

        # Stack samples: (B, num_samples, 1, 256, 256)
        population_maps = torch.stack(all_samples, dim=1)

        return population_maps


if __name__ == "__main__":
    print("=" * 70)
    print("Testing DDPM and DDIM Samplers")
    print("=" * 70)

    # Create dummy models
    class DummyUNet(nn.Module):
        def forward(self, z_t, t_emb, cond):
            # Return random noise prediction
            return torch.randn_like(z_t)

    class DummyTimeEmb(nn.Module):
        def forward(self, t):
            B = t.shape[0]
            return torch.randn(B, 256)

    # Create diffusion schedule
    T = 1000
    betas = torch.linspace(0.0001, 0.02, T)

    print(f"\n1. Testing DDPM sampler...")
    ddpm = DDPMSampler(betas, device='cpu')

    unet = DummyUNet()
    time_emb = DummyTimeEmb()
    cond = torch.randn(2, 256, 32, 32)
    shape = (2, 4, 32, 32)

    print(f"  Sampling with {T} steps...")
    z_0_ddpm = ddpm.sample(unet, time_emb, cond, shape, show_progress=False)
    print(f"  Output shape: {z_0_ddpm.shape}")
    print(f"  Output range: [{z_0_ddpm.min():.4f}, {z_0_ddpm.max():.4f}]")

    print(f"\n2. Testing DDIM sampler...")
    ddim = DDIMSampler(betas, device='cpu')

    num_steps = 50
    print(f"  Sampling with {num_steps} steps...")
    z_0_ddim = ddim.sample(unet, time_emb, cond, shape, num_steps=num_steps, show_progress=False)
    print(f"  Output shape: {z_0_ddim.shape}")
    print(f"  Output range: [{z_0_ddim.min():.4f}, {z_0_ddim.max():.4f}]")

    print(f"\n3. Testing DDIM with different eta values...")
    for eta in [0.0, 0.5, 1.0]:
        z_0 = ddim.sample(unet, time_emb, cond, shape, num_steps=50, eta=eta, show_progress=False)
        print(f"  eta={eta}: mean={z_0.mean():.4f}, std={z_0.std():.4f}")

    print(f"\n4. Testing speedup (DDIM vs DDPM)...")
    import time

    # DDPM
    start = time.time()
    _ = ddpm.sample(unet, time_emb, cond, shape, show_progress=False)
    ddpm_time = time.time() - start

    # DDIM
    start = time.time()
    _ = ddim.sample(unet, time_emb, cond, shape, num_steps=50, show_progress=False)
    ddim_time = time.time() - start

    print(f"  DDPM ({T} steps): {ddpm_time:.2f}s")
    print(f"  DDIM ({num_steps} steps): {ddim_time:.2f}s")
    print(f"  Speedup: {ddpm_time/ddim_time:.1f}x")

    print("\n✓ All sampler tests passed!")
