"""
VAE for Residual Maps in C3-LDM
Reference: C3-LDM.md Section "Component 2"

Encodes residual population maps (256x256) to latent space (32x32x4)
Used before diffusion process to reduce computational cost.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualVAE(nn.Module):
    """
    Variational Autoencoder for residual population maps.

    Encoder: 256x256x1 → 32x32x4 latent
    Decoder: 32x32x4 → 256x256x1 reconstruction
    """

    def __init__(self, latent_channels=4, base_channels=64):
        """
        Args:
            latent_channels: Number of latent channels (default: 4)
            base_channels: Base number of channels in encoder/decoder
        """
        super().__init__()
        self.latent_channels = latent_channels

        # Encoder: 256 → 128 → 64 → 32
        self.encoder = Encoder(latent_channels, base_channels)

        # Decoder: 32 → 64 → 128 → 256
        self.decoder = Decoder(latent_channels, base_channels)

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick: z = μ + σ * ε, where ε ~ N(0,1)

        Args:
            mu: (B, C, H, W) mean
            logvar: (B, C, H, W) log variance

        Returns:
            z: (B, C, H, W) sampled latent
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x):
        """
        Encode residual map to latent distribution.

        Args:
            x: (B, 1, 256, 256) residual map

        Returns:
            mu: (B, latent_channels, 32, 32) mean
            logvar: (B, latent_channels, 32, 32) log variance
        """
        return self.encoder(x)

    def decode(self, z):
        """
        Decode latent to residual map.

        Args:
            z: (B, latent_channels, 32, 32) latent

        Returns:
            reconstruction: (B, 1, 256, 256) reconstructed residual
        """
        return self.decoder(z)

    def forward(self, x):
        """
        Full VAE forward pass.

        Args:
            x: (B, 1, 256, 256) residual map

        Returns:
            reconstruction: (B, 1, 256, 256)
            mu: (B, latent_channels, 32, 32)
            logvar: (B, latent_channels, 32, 32)
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar

    def kl_loss(self, mu, logvar):
        """
        KL divergence loss: KL(q(z|x) || p(z)) where p(z) = N(0,1)

        Args:
            mu: (B, C, H, W) mean
            logvar: (B, C, H, W) log variance

        Returns:
            kl_div: scalar KL divergence
        """
        # KL(q || p) = -0.5 * sum(1 + log(σ²) - μ² - σ²)
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return kl_div / mu.shape[0]  # Average over batch


class Encoder(nn.Module):
    """
    Encoder: 256x256 → 32x32 latent

    Architecture:
    256 →[Conv] 256 →[Down] 128 →[Conv] 128 →[Down] 64 →[Conv] 64 →[Down] 32
    """

    def __init__(self, latent_channels=4, base_ch=64):
        super().__init__()

        # 256x256 → 256x256
        self.conv_in = nn.Conv2d(1, base_ch, kernel_size=3, stride=1, padding=1)

        # 256x256 → 128x128
        self.down1 = nn.Sequential(
            nn.GroupNorm(8, base_ch),
            nn.SiLU(),
            nn.Conv2d(base_ch, base_ch, kernel_size=3, stride=2, padding=1)
        )

        # 128x128 → 64x64
        self.down2 = nn.Sequential(
            nn.GroupNorm(8, base_ch),
            nn.SiLU(),
            nn.Conv2d(base_ch, base_ch * 2, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(16, base_ch * 2),
            nn.SiLU(),
            nn.Conv2d(base_ch * 2, base_ch * 2, kernel_size=3, stride=2, padding=1)
        )

        # 64x64 → 32x32
        self.down3 = nn.Sequential(
            nn.GroupNorm(16, base_ch * 2),
            nn.SiLU(),
            nn.Conv2d(base_ch * 2, base_ch * 4, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(32, base_ch * 4),
            nn.SiLU(),
            nn.Conv2d(base_ch * 4, base_ch * 4, kernel_size=3, stride=2, padding=1)
        )

        # Output layer: split into mu and logvar
        self.conv_out = nn.Conv2d(base_ch * 4, latent_channels * 2, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # x: (B, 1, 256, 256)
        h = self.conv_in(x)  # (B, 64, 256, 256)
        h = self.down1(h)  # (B, 64, 128, 128)
        h = self.down2(h)  # (B, 128, 64, 64)
        h = self.down3(h)  # (B, 256, 32, 32)
        h = self.conv_out(h)  # (B, latent_channels*2, 32, 32)

        # Split into mu and logvar
        mu, logvar = torch.chunk(h, 2, dim=1)
        return mu, logvar


class Decoder(nn.Module):
    """
    Decoder: 32x32 → 256x256

    Architecture:
    32 →[Up] 64 →[Conv] 64 →[Up] 128 →[Conv] 128 →[Up] 256 →[Conv] 256
    """

    def __init__(self, latent_channels=4, base_ch=64):
        super().__init__()

        # Input layer
        self.conv_in = nn.Conv2d(latent_channels, base_ch * 4, kernel_size=3, stride=1, padding=1)

        # 32x32 → 64x64
        self.up1 = nn.Sequential(
            nn.GroupNorm(32, base_ch * 4),
            nn.SiLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(base_ch * 4, base_ch * 4, kernel_size=3, stride=1, padding=1),
        )

        # 64x64 → 128x128
        self.up2 = nn.Sequential(
            nn.GroupNorm(32, base_ch * 4),
            nn.SiLU(),
            nn.Conv2d(base_ch * 4, base_ch * 2, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(16, base_ch * 2),
            nn.SiLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(base_ch * 2, base_ch * 2, kernel_size=3, stride=1, padding=1),
        )

        # 128x128 → 256x256
        self.up3 = nn.Sequential(
            nn.GroupNorm(16, base_ch * 2),
            nn.SiLU(),
            nn.Conv2d(base_ch * 2, base_ch, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, base_ch),
            nn.SiLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(base_ch, base_ch, kernel_size=3, stride=1, padding=1),
        )

        # Output layer
        self.conv_out = nn.Sequential(
            nn.GroupNorm(8, base_ch),
            nn.SiLU(),
            nn.Conv2d(base_ch, 1, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, z):
        # z: (B, latent_channels, 32, 32)
        h = self.conv_in(z)  # (B, 256, 32, 32)
        h = self.up1(h)  # (B, 256, 64, 64)
        h = self.up2(h)  # (B, 128, 128, 128)
        h = self.up3(h)  # (B, 64, 256, 256)
        reconstruction = self.conv_out(h)  # (B, 1, 256, 256)
        return reconstruction


if __name__ == "__main__":
    print("=" * 70)
    print("Testing Residual VAE")
    print("=" * 70)

    # Create model
    vae = ResidualVAE(latent_channels=4, base_channels=64)

    # Count parameters
    total_params = sum(p.numel() for p in vae.parameters())
    print(f"\nTotal parameters: {total_params:,}")

    # Test with dummy data
    B = 2
    x = torch.randn(B, 1, 256, 256)

    print(f"\n1. Testing encoder...")
    mu, logvar = vae.encode(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Mu shape: {mu.shape}")
    print(f"  Logvar shape: {logvar.shape}")

    print(f"\n2. Testing reparameterization...")
    z = vae.reparameterize(mu, logvar)
    print(f"  Latent shape: {z.shape}")
    print(f"  Latent range: [{z.min():.4f}, {z.max():.4f}]")

    print(f"\n3. Testing decoder...")
    reconstruction = vae.decode(z)
    print(f"  Reconstruction shape: {reconstruction.shape}")
    print(f"  Reconstruction range: [{reconstruction.min():.4f}, {reconstruction.max():.4f}]")

    print(f"\n4. Testing full forward pass...")
    recon, mu, logvar = vae(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {recon.shape}")
    print(f"  MSE reconstruction error: {F.mse_loss(recon, x):.6f}")

    print(f"\n5. Testing KL loss...")
    kl = vae.kl_loss(mu, logvar)
    print(f"  KL divergence: {kl:.6f}")

    print(f"\n6. Testing gradient flow...")
    loss = F.mse_loss(recon, x) + 0.0001 * kl
    loss.backward()
    encoder_grad_norm = torch.nn.utils.clip_grad_norm_(vae.encoder.parameters(), float('inf'))
    decoder_grad_norm = torch.nn.utils.clip_grad_norm_(vae.decoder.parameters(), float('inf'))
    print(f"  Encoder gradient norm: {encoder_grad_norm:.6f}")
    print(f"  Decoder gradient norm: {decoder_grad_norm:.6f}")

    print("\n✓ All tests passed!")
