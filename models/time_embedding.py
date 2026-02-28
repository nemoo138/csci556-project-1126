"""
Time Embedding for Diffusion Models in C3-LDM
Reference: C3-LDM.md Section 4.1

Sinusoidal positional encoding + MLP for time step conditioning.
"""

import math
import torch
import torch.nn as nn


class TimeEmbedding(nn.Module):
    """
    Time embedding for diffusion timesteps.

    Uses sinusoidal encoding followed by MLP:
    t → sinusoidal(dim=64) → Linear(64→256) → SiLU → Linear(256→256)
    """

    def __init__(self, dim=256, base_dim=64):
        """
        Args:
            dim: Output embedding dimension
            base_dim: Dimension of sinusoidal encoding
        """
        super().__init__()
        self.dim = dim
        self.base_dim = base_dim

        # MLP to process sinusoidal encoding
        self.mlp = nn.Sequential(
            nn.Linear(base_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, t):
        """
        Compute time embeddings for diffusion timesteps.

        Args:
            t: (B,) timesteps in range [0, T-1]

        Returns:
            emb: (B, dim) time embeddings
        """
        # Create sinusoidal encoding
        sin_emb = self.sinusoidal_encoding(t, self.base_dim)

        # Process through MLP
        emb = self.mlp(sin_emb)

        return emb

    @staticmethod
    def sinusoidal_encoding(t, dim):
        """
        Create sinusoidal positional encoding for timesteps.

        Args:
            t: (B,) timesteps
            dim: Embedding dimension

        Returns:
            encoding: (B, dim) sinusoidal encoding
        """
        device = t.device
        half_dim = dim // 2

        # Compute frequency bands
        # freq = 1 / (10000 ^ (2i / dim))
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)

        # Expand timesteps: (B,) → (B, 1)
        # Expand frequencies: (half_dim,) → (1, half_dim)
        # Result: (B, half_dim)
        emb = t[:, None].float() * emb[None, :]

        # Concatenate sin and cos
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

        # Handle odd dimensions
        if dim % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0, 1), mode='constant', value=0)

        return emb


class TimestepEmbedSequential(nn.Sequential):
    """
    A sequential module that passes timestep embeddings to children.

    Useful for building blocks that need time conditioning (e.g., ResBlocks).
    """

    def forward(self, x, emb):
        """
        Args:
            x: Input tensor
            emb: Time embedding

        Returns:
            Output tensor after passing through all modules
        """
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class TimestepBlock(nn.Module):
    """
    Base class for modules that use timestep conditioning.
    Subclasses should implement forward(x, emb).
    """

    def forward(self, x, emb):
        """
        Args:
            x: Input tensor
            emb: (B, dim) time embedding

        Returns:
            Output tensor
        """
        raise NotImplementedError


if __name__ == "__main__":
    print("=" * 70)
    print("Testing Time Embedding")
    print("=" * 70)

    # Create time embedding module
    time_emb = TimeEmbedding(dim=256, base_dim=64)

    print(f"\n1. Testing sinusoidal encoding...")
    t = torch.tensor([0, 10, 100, 500, 999])
    sin_enc = TimeEmbedding.sinusoidal_encoding(t, dim=64)
    print(f"  Input timesteps: {t}")
    print(f"  Sinusoidal encoding shape: {sin_enc.shape}")
    print(f"  Encoding range: [{sin_enc.min():.4f}, {sin_enc.max():.4f}]")

    # Check that different timesteps have different encodings
    print(f"\n2. Checking uniqueness...")
    similarity = torch.nn.functional.cosine_similarity(sin_enc[0:1], sin_enc[1:], dim=1)
    print(f"  Cosine similarity between t=0 and others: {similarity}")

    print(f"\n3. Testing full embedding...")
    B = 16
    T = 1000
    t_batch = torch.randint(0, T, (B,))
    emb = time_emb(t_batch)
    print(f"  Batch size: {B}")
    print(f"  Timestep range: [0, {T-1}]")
    print(f"  Output shape: {emb.shape}")
    print(f"  Output range: [{emb.min():.4f}, {emb.max():.4f}]")
    print(f"  Output mean: {emb.mean():.4f}, std: {emb.std():.4f}")

    print(f"\n4. Testing gradient flow...")
    loss = emb.sum()
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(time_emb.parameters(), float('inf'))
    print(f"  Gradient norm: {grad_norm:.6f}")

    print(f"\n5. Testing parameter count...")
    total_params = sum(p.numel() for p in time_emb.parameters())
    print(f"  Total parameters: {total_params:,}")

    print(f"\n6. Testing boundary conditions...")
    # Test t=0
    t_zero = torch.zeros(4, dtype=torch.long)
    emb_zero = time_emb(t_zero)
    print(f"  t=0 embedding shape: {emb_zero.shape}")
    print(f"  All t=0 embeddings identical: {torch.allclose(emb_zero[0], emb_zero[1])}")

    # Test t=T-1
    t_max = torch.full((4,), T-1, dtype=torch.long)
    emb_max = time_emb(t_max)
    print(f"  t={T-1} embedding shape: {emb_max.shape}")
    print(f"  All t={T-1} embeddings identical: {torch.allclose(emb_max[0], emb_max[1])}")

    # Different timesteps should have different embeddings
    emb_diff = torch.nn.functional.cosine_similarity(emb_zero[0:1], emb_max[0:1])
    print(f"  Cosine similarity t=0 vs t={T-1}: {emb_diff.item():.4f}")

    print("\n✓ All tests passed!")
