"""
Simplified working U-Net for C3-LDM

Basic U-Net that definitely works, for Phase 3 completion.
"""

import torch
import torch.nn as nn
from .unet_blocks import FiLMResBlock, AttentionBlock, Downsample, Upsample


class SimpleUNet(nn.Module):
    """Simple working U-Net for noise prediction."""

    def __init__(
        self,
        in_channels=4,
        model_channels=128,
        time_emb_dim=256,
        cond_channels=256
    ):
        super().__init__()

        # Input
        self.conv_in = nn.Conv2d(in_channels, model_channels, 3, padding=1)

        # Encoder: 32x32 -> 16x16 -> 8x8
        self.down1 = nn.ModuleList([
            FiLMResBlock(model_channels, model_channels, time_emb_dim, cond_channels),
            FiLMResBlock(model_channels, model_channels, time_emb_dim, cond_channels),
            AttentionBlock(model_channels, 8),
        ])
        self.downsample1 = Downsample(model_channels)

        self.down2 = nn.ModuleList([
            FiLMResBlock(model_channels, model_channels*2, time_emb_dim, cond_channels),
            FiLMResBlock(model_channels*2, model_channels*2, time_emb_dim, cond_channels),
        ])
        self.downsample2 = Downsample(model_channels*2)

        self.down3 = nn.ModuleList([
            FiLMResBlock(model_channels*2, model_channels*4, time_emb_dim, cond_channels),
            FiLMResBlock(model_channels*4, model_channels*4, time_emb_dim, cond_channels),
            AttentionBlock(model_channels*4, 8),
        ])

        # Middle
        self.mid = nn.ModuleList([
            FiLMResBlock(model_channels*4, model_channels*4, time_emb_dim, cond_channels),
            AttentionBlock(model_channels*4, 8),
            FiLMResBlock(model_channels*4, model_channels*4, time_emb_dim, cond_channels),
        ])

        # Decoder: 8x8 -> 16x16 -> 32x32
        self.up3 = nn.ModuleList([
            FiLMResBlock(model_channels*8, model_channels*4, time_emb_dim, cond_channels),  # concat skip
            FiLMResBlock(model_channels*4, model_channels*4, time_emb_dim, cond_channels),
            FiLMResBlock(model_channels*4, model_channels*4, time_emb_dim, cond_channels),
            AttentionBlock(model_channels*4, 8),
        ])
        self.upsample3 = Upsample(model_channels*4)

        self.up2 = nn.ModuleList([
            FiLMResBlock(model_channels*6, model_channels*2, time_emb_dim, cond_channels),  # concat skip
            FiLMResBlock(model_channels*2, model_channels*2, time_emb_dim, cond_channels),
            FiLMResBlock(model_channels*2, model_channels*2, time_emb_dim, cond_channels),
        ])
        self.upsample2 = Upsample(model_channels*2)

        self.up1 = nn.ModuleList([
            FiLMResBlock(model_channels*3, model_channels, time_emb_dim, cond_channels),  # concat skip
            FiLMResBlock(model_channels, model_channels, time_emb_dim, cond_channels),
            FiLMResBlock(model_channels, model_channels, time_emb_dim, cond_channels),
            AttentionBlock(model_channels, 8),
        ])

        # Output
        self.conv_out = nn.Sequential(
            nn.GroupNorm(32, model_channels),
            nn.SiLU(),
            nn.Conv2d(model_channels, in_channels, 3, padding=1)
        )

    def forward(self, x, t_emb, cond):
        # Input
        h = self.conv_in(x)

        # Encoder
        skips = []
        for layer in self.down1:
            if isinstance(layer, FiLMResBlock):
                h = layer(h, t_emb, cond)
            else:
                h = layer(h)
        skips.append(h)
        h = self.downsample1(h)

        for layer in self.down2:
            h = layer(h, t_emb, cond)
        skips.append(h)
        h = self.downsample2(h)

        for layer in self.down3:
            if isinstance(layer, FiLMResBlock):
                h = layer(h, t_emb, cond)
            else:
                h = layer(h)
        skips.append(h)

        # Middle
        for layer in self.mid:
            if isinstance(layer, FiLMResBlock):
                h = layer(h, t_emb, cond)
            else:
                h = layer(h)

        # Decoder
        h = torch.cat([h, skips.pop()], dim=1)
        for layer in self.up3:
            if isinstance(layer, FiLMResBlock):
                h = layer(h, t_emb, cond)
            else:
                h = layer(h)
        h = self.upsample3(h)

        h = torch.cat([h, skips.pop()], dim=1)
        for layer in self.up2:
            h = layer(h, t_emb, cond)
        h = self.upsample2(h)

        h = torch.cat([h, skips.pop()], dim=1)
        for layer in self.up1:
            if isinstance(layer, FiLMResBlock):
                h = layer(h, t_emb, cond)
            else:
                h = layer(h)

        return self.conv_out(h)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')

    print("Testing Simple U-Net")
    unet = SimpleUNet()
    print(f"Parameters: {sum(p.numel() for p in unet.parameters()):,}")

    x = torch.randn(2, 4, 32, 32)
    t = torch.randn(2, 256)
    c = torch.randn(2, 256, 32, 32)

    out = unet(x, t, c)
    print(f"Output shape: {out.shape}")
    print("✓ Test passed!")
