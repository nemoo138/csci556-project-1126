"""
Diffusion U-Net for C3-LDM
Reference: C3-LDM.md Section "Component 4"

Simplified U-Net with proper channel tracking and skip connections.
"""

import torch
import torch.nn as nn
from .unet_blocks import FiLMResBlock, AttentionBlock, Downsample, Upsample


class DiffusionUNet(nn.Module):
    """
    U-Net for predicting noise in latent diffusion.

    Architecture: 32x32 -> 16x16 -> 8x8 -> 16x16 -> 32x32
    With FiLM conditioning from time + spatial features
    """

    def __init__(
        self,
        in_channels=4,
        out_channels=4,
        model_channels=128,
        time_emb_dim=256,
        cond_channels=256,
        channel_mult=(1, 2, 4),
        num_res_blocks=2,
        attention_resolutions=(32, 8),
        dropout=0.0
    ):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.time_emb_dim = time_emb_dim
        self.cond_channels = cond_channels

        # Input conv
        self.conv_in = nn.Conv2d(in_channels, model_channels, 3, padding=1)

        # Encoder
        self.down_blocks = nn.ModuleList()
        ch = model_channels

        for level, mult in enumerate(channel_mult):
            out_ch = model_channels * mult

            for _ in range(num_res_blocks):
                layers = [FiLMResBlock(ch, out_ch, time_emb_dim, cond_channels, dropout)]
                ch = out_ch

                # Add attention at specified resolutions
                res = 32 // (2 ** level)
                if res in attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads=8))

                self.down_blocks.append(nn.ModuleList(layers))

            # Downsample (except last level)
            if level < len(channel_mult) - 1:
                self.down_blocks.append(nn.ModuleList([Downsample(ch)]))

        # Middle
        self.mid_block1 = FiLMResBlock(ch, ch, time_emb_dim, cond_channels, dropout)
        self.mid_attn = AttentionBlock(ch, num_heads=8)
        self.mid_block2 = FiLMResBlock(ch, ch, time_emb_dim, cond_channels, dropout)

        # Decoder
        self.up_blocks = nn.ModuleList()

        # Track skip connection channels
        skip_channels = []
        temp_ch = model_channels
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                skip_channels.append(model_channels * mult)
            if level < len(channel_mult) - 1:
                skip_channels.append(model_channels * mult)

        skip_channels = list(reversed(skip_channels))
        skip_idx = 0

        for level, mult in enumerate(reversed(channel_mult)):
            out_ch = model_channels * mult

            for i in range(num_res_blocks + 1):
                # Account for skip connection channels
                in_ch = ch + skip_channels[skip_idx]
                skip_idx += 1

                layers = [FiLMResBlock(in_ch, out_ch, time_emb_dim, cond_channels, dropout)]
                ch = out_ch

                # Add attention
                res = 8 * (2 ** level)
                if res in attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads=8))

                self.up_blocks.append(nn.ModuleList(layers))

            # Upsample (except last level)
            if level < len(channel_mult) - 1:
                self.up_blocks.append(nn.ModuleList([Upsample(ch)]))

        # Output
        self.conv_out = nn.Sequential(
            nn.GroupNorm(32, ch),
            nn.SiLU(),
            nn.Conv2d(ch, out_channels, 3, padding=1)
        )

    def forward(self, x, t_emb, cond):
        """
        Args:
            x: (B, 4, 32, 32) noisy latent
            t_emb: (B, time_emb_dim) time embeddings
            cond: (B, cond_channels, 32, 32) spatial conditioning

        Returns:
            (B, 4, 32, 32) predicted noise
        """
        # Input
        h = self.conv_in(x)

        # Encoder with skip connections
        skips = []
        for block_list in self.down_blocks:
            for layer in block_list:
                if isinstance(layer, FiLMResBlock):
                    h = layer(h, t_emb, cond)
                    skips.append(h)
                elif isinstance(layer, AttentionBlock):
                    h = layer(h)
                elif isinstance(layer, Downsample):
                    h = layer(h)
                    skips.append(h)

        # Middle
        h = self.mid_block1(h, t_emb, cond)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb, cond)

        # Decoder with skip connections
        for block_list in self.up_blocks:
            for layer in block_list:
                if isinstance(layer, FiLMResBlock):
                    skip = skips.pop()
                    h = torch.cat([h, skip], dim=1)
                    h = layer(h, t_emb, cond)
                elif isinstance(layer, AttentionBlock):
                    h = layer(h)
                elif isinstance(layer, Upsample):
                    h = layer(h)

        # Output
        return self.conv_out(h)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')

    print("=" * 70)
    print("Testing Diffusion U-Net")
    print("=" * 70)

    unet = DiffusionUNet(
        in_channels=4,
        out_channels=4,
        model_channels=128,
        channel_mult=(1, 2, 4),
        num_res_blocks=2,
        attention_resolutions=(32, 8)
    )

    total_params = sum(p.numel() for p in unet.parameters())
    print(f"\nTotal parameters: {total_params:,}")

    # Test
    B = 2
    z_t = torch.randn(B, 4, 32, 32)
    t_emb = torch.randn(B, 256)
    cond = torch.randn(B, 256, 32, 32)

    print(f"\n1. Testing forward pass...")
    print(f"  Input shape: {z_t.shape}")
    print(f"  Time emb shape: {t_emb.shape}")
    print(f"  Conditioning shape: {cond.shape}")

    noise_pred = unet(z_t, t_emb, cond)
    print(f"  Output shape: {noise_pred.shape}")
    print(f"  Shape matches: {noise_pred.shape == z_t.shape}")

    print(f"\n2. Testing gradient flow...")
    loss = noise_pred.sum()
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(unet.parameters(), float('inf'))
    print(f"  Gradient norm: {grad_norm:.6f}")

    print("\n✓ All tests passed!")
