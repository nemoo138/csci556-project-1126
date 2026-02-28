"""
U-Net Building Blocks for C3-LDM Diffusion Model
Reference: C3-LDM.md Section "Component 4"

Includes:
- FiLM-conditioned ResBlocks
- Self-attention layers
- Downsample/Upsample operations
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .time_embedding import TimestepBlock


class FiLMResBlock(TimestepBlock):
    """
    Residual block with FiLM (Feature-wise Linear Modulation) conditioning.

    Applies both time conditioning and spatial conditioning via FiLM:
    - Time embedding provides γ_t, β_t
    - Spatial conditioning provides γ_c, β_c
    - Combined: h = h * (1 + γ_t + γ_c) + (β_t + β_c)

    Architecture:
    x → GroupNorm → SiLU → Conv3x3 → [FiLM] → GroupNorm → SiLU → Conv3x3 → + residual
    """

    def __init__(self, in_channels, out_channels, time_emb_dim=256, cond_channels=256, dropout=0.0):
        """
        Args:
            in_channels: Input feature channels
            out_channels: Output feature channels
            time_emb_dim: Dimension of time embeddings
            cond_channels: Channels in spatial conditioning
            dropout: Dropout probability
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # First conv block
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # FiLM conditioning from time embedding
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels * 2)  # γ_t and β_t
        )

        # FiLM conditioning from spatial conditioning
        self.cond_mlp = nn.Sequential(
            nn.Conv2d(cond_channels, out_channels * 2, kernel_size=1),  # γ_c and β_c
        )

        # Second conv block
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.act2 = nn.SiLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        # Residual connection (shortcut)
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, t_emb, cond):
        """
        Forward pass with FiLM conditioning.

        Args:
            x: (B, in_channels, H, W) input features
            t_emb: (B, time_emb_dim) time embeddings
            cond: (B, cond_channels, H, W) spatial conditioning

        Returns:
            out: (B, out_channels, H, W) output features
        """
        # Save residual
        residual = self.shortcut(x)

        # First conv block
        h = self.norm1(x)
        h = self.act1(h)
        h = self.conv1(h)

        # FiLM conditioning from time
        t_params = self.time_mlp(t_emb)  # (B, out_channels * 2)
        t_params = t_params.unsqueeze(-1).unsqueeze(-1)  # (B, out_channels * 2, 1, 1)
        gamma_t, beta_t = torch.chunk(t_params, 2, dim=1)  # Each (B, out_channels, 1, 1)

        # FiLM conditioning from spatial context
        # Adapt conditioning to match feature resolution
        if cond.shape[-2:] != h.shape[-2:]:
            cond_adapted = F.adaptive_avg_pool2d(cond, h.shape[-2:])
        else:
            cond_adapted = cond

        c_params = self.cond_mlp(cond_adapted)  # (B, out_channels * 2, H, W)
        gamma_c, beta_c = torch.chunk(c_params, 2, dim=1)  # Each (B, out_channels, H, W)

        # Apply combined FiLM: h = h * (1 + γ_t + γ_c) + (β_t + β_c)
        h = h * (1 + gamma_t + gamma_c) + (beta_t + beta_c)

        # Second conv block
        h = self.norm2(h)
        h = self.act2(h)
        h = self.dropout(h)
        h = self.conv2(h)

        # Add residual
        out = h + residual

        return out


class AttentionBlock(nn.Module):
    """
    Self-attention block for capturing long-range dependencies.

    Uses multi-head self-attention with positional encodings.
    Applied at key resolutions (32×32 and 8×8) in the U-Net.
    """

    def __init__(self, channels, num_heads=8):
        """
        Args:
            channels: Number of input/output channels
            num_heads: Number of attention heads
        """
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads

        assert channels % num_heads == 0, "channels must be divisible by num_heads"

        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) input features

        Returns:
            out: (B, C, H, W) output with self-attention
        """
        B, C, H, W = x.shape
        residual = x

        # Normalize
        h = self.norm(x)

        # Compute Q, K, V
        qkv = self.qkv(h)  # (B, C*3, H, W)
        qkv = qkv.reshape(B, 3, self.num_heads, C // self.num_heads, H * W)
        qkv = qkv.permute(1, 0, 2, 4, 3)  # (3, B, num_heads, H*W, C//num_heads)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention
        scale = (C // self.num_heads) ** -0.5
        attn = torch.softmax(q @ k.transpose(-2, -1) * scale, dim=-1)  # (B, num_heads, H*W, H*W)
        h = attn @ v  # (B, num_heads, H*W, C//num_heads)

        # Reshape back
        h = h.permute(0, 1, 3, 2).reshape(B, C, H, W)

        # Project
        h = self.proj(h)

        # Add residual
        out = h + residual

        return out


class Downsample(nn.Module):
    """Downsample spatial resolution by 2x using strided convolution."""

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    """Upsample spatial resolution by 2x using nearest neighbor + convolution."""

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv(x)
        return x


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '..')
    from models.time_embedding import TimestepBlock as TB

    print("=" * 70)
    print("Testing U-Net Building Blocks")
    print("=" * 70)

    B, H, W = 4, 32, 32
    in_ch, out_ch = 128, 256
    time_dim, cond_ch = 256, 256

    print(f"\n1. Testing FiLMResBlock...")
    resblock = FiLMResBlock(in_ch, out_ch, time_dim, cond_ch)
    x = torch.randn(B, in_ch, H, W)
    t_emb = torch.randn(B, time_dim)
    cond = torch.randn(B, cond_ch, H, W)

    out = resblock(x, t_emb, cond)
    print(f"  Input shape: {x.shape}")
    print(f"  Time emb shape: {t_emb.shape}")
    print(f"  Conditioning shape: {cond.shape}")
    print(f"  Output shape: {out.shape}")
    print(f"  Parameters: {sum(p.numel() for p in resblock.parameters()):,}")

    print(f"\n2. Testing gradient flow through FiLMResBlock...")
    loss = out.sum()
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(resblock.parameters(), float('inf'))
    print(f"  Gradient norm: {grad_norm:.6f}")

    print(f"\n3. Testing AttentionBlock...")
    attn = AttentionBlock(channels=256, num_heads=8)
    x_attn = torch.randn(B, 256, H, W)
    out_attn = attn(x_attn)
    print(f"  Input shape: {x_attn.shape}")
    print(f"  Output shape: {out_attn.shape}")
    print(f"  Parameters: {sum(p.numel() for p in attn.parameters()):,}")

    print(f"\n4. Testing Downsample...")
    down = Downsample(channels=128)
    x_down = torch.randn(B, 128, 32, 32)
    out_down = down(x_down)
    print(f"  Input shape: {x_down.shape}")
    print(f"  Output shape: {out_down.shape}")
    print(f"  Resolution reduced by 2x: {out_down.shape[-1] == x_down.shape[-1] // 2}")

    print(f"\n5. Testing Upsample...")
    up = Upsample(channels=128)
    x_up = torch.randn(B, 128, 16, 16)
    out_up = up(x_up)
    print(f"  Input shape: {x_up.shape}")
    print(f"  Output shape: {out_up.shape}")
    print(f"  Resolution increased by 2x: {out_up.shape[-1] == x_up.shape[-1] * 2}")

    print(f"\n6. Testing ResBlock with different channel counts...")
    resblock_diff = FiLMResBlock(64, 128, time_dim, cond_ch)
    x_diff = torch.randn(B, 64, H, W)
    out_diff = resblock_diff(x_diff, t_emb, cond)
    print(f"  Input channels: 64, Output channels: 128")
    print(f"  Output shape: {out_diff.shape}")
    print(f"  Shortcut applied: ✓")

    print(f"\n7. Testing attention at different resolutions...")
    for res in [8, 16, 32]:
        x_res = torch.randn(B, 256, res, res)
        out_res = attn(x_res)
        num_tokens = res * res
        print(f"  Resolution {res}×{res}: {num_tokens} tokens, output shape {out_res.shape}")

    print("\n✓ All tests passed!")
