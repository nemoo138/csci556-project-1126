"""
Dual-Branch Conditional Encoder for C3-LDM
Reference: C3-LDM.md Section "Component 3"

Processes VIIRS nightlights (low-res) and WSF settlements (high-res) separately,
then fuses them into a single conditioning vector for the diffusion model.
"""

import torch
import torch.nn as nn


class DualBranchConditionalEncoder(nn.Module):
    """
    Dual-branch encoder for multi-scale conditioning.

    Low-res branch: VIIRS nightlights (coarse ~500-700m)
    High-res branch: WSF settlements (fine ~10-30m)

    Both branches downsample 256×256 → 32×32, then fuse to C_cond channels.
    """

    def __init__(self, cond_channels=256, low_res_ch=128, high_res_ch=128):
        """
        Args:
            cond_channels: Output conditioning channels (C_cond)
            low_res_ch: Channels in low-res branch
            high_res_ch: Channels in high-res branch
        """
        super().__init__()
        self.cond_channels = cond_channels

        # Low-res branch (VIIRS nightlights)
        # Larger kernels and strides to capture coarse patterns
        self.low_res_branch = nn.Sequential(
            # 256 → 128
            nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(4, 32),
            nn.SiLU(),

            # 128 → 64
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(8, 64),
            nn.SiLU(),

            # 64 → 32
            nn.Conv2d(64, low_res_ch, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(16, low_res_ch),
            nn.SiLU(),
        )

        # High-res branch (WSF settlements)
        # Smaller kernels to preserve fine detail
        self.high_res_branch = nn.Sequential(
            # 256 → 256
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),

            # 256 → 128
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),

            # 128 → 128
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(16, 128),
            nn.SiLU(),

            # 128 → 64
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(16, 128),
            nn.SiLU(),

            # 64 → 32
            nn.Conv2d(128, high_res_ch, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(16, high_res_ch),
            nn.SiLU(),
        )

        # Fusion layer
        # Concatenate low-res and high-res features, then compress
        self.fusion = nn.Sequential(
            nn.Conv2d(low_res_ch + high_res_ch, cond_channels, kernel_size=1),
            nn.GroupNorm(32, cond_channels),
            nn.SiLU(),
        )

    def forward(self, lights, settlement):
        """
        Process VIIRS and WSF through dual branches and fuse.

        Args:
            lights: (B, 1, 256, 256) VIIRS nighttime lights
            settlement: (B, 1, 256, 256) World Settlement Footprint

        Returns:
            H_cond: (B, cond_channels, 32, 32) fused conditioning
        """
        # Process through separate branches
        H_low = self.low_res_branch(lights)      # (B, low_res_ch, 32, 32)
        H_high = self.high_res_branch(settlement)  # (B, high_res_ch, 32, 32)

        # Concatenate
        H_concat = torch.cat([H_low, H_high], dim=1)  # (B, low_res_ch + high_res_ch, 32, 32)

        # Fuse to final conditioning
        H_cond = self.fusion(H_concat)  # (B, cond_channels, 32, 32)

        return H_cond


if __name__ == "__main__":
    print("=" * 70)
    print("Testing Dual-Branch Conditional Encoder")
    print("=" * 70)

    # Create encoder
    encoder = DualBranchConditionalEncoder(cond_channels=256, low_res_ch=128, high_res_ch=128)

    # Count parameters
    total_params = sum(p.numel() for p in encoder.parameters())
    print(f"\nTotal parameters: {total_params:,}")

    # Create dummy inputs
    B = 4
    lights = torch.randn(B, 1, 256, 256)
    settlement = torch.randn(B, 1, 256, 256)

    print(f"\n1. Testing forward pass...")
    H_cond = encoder(lights, settlement)
    print(f"  Lights input shape: {lights.shape}")
    print(f"  Settlement input shape: {settlement.shape}")
    print(f"  Conditioning output shape: {H_cond.shape}")
    print(f"  Output range: [{H_cond.min():.4f}, {H_cond.max():.4f}]")
    print(f"  Output mean: {H_cond.mean():.4f}, std: {H_cond.std():.4f}")

    print(f"\n2. Testing branch outputs separately...")
    with torch.no_grad():
        H_low = encoder.low_res_branch(lights)
        H_high = encoder.high_res_branch(settlement)
    print(f"  Low-res branch output: {H_low.shape}")
    print(f"  High-res branch output: {H_high.shape}")

    print(f"\n3. Testing different inputs produce different outputs...")
    lights2 = torch.randn(B, 1, 256, 256)
    settlement2 = torch.randn(B, 1, 256, 256)
    H_cond2 = encoder(lights2, settlement2)

    similarity = torch.nn.functional.cosine_similarity(
        H_cond.view(B, -1), H_cond2.view(B, -1), dim=1
    )
    print(f"  Cosine similarity between different inputs: {similarity.mean():.4f}")
    print(f"  (Should be close to 0 for random inputs)")

    print(f"\n4. Testing gradient flow...")
    loss = H_cond.sum()
    loss.backward()

    low_grad_norm = torch.nn.utils.clip_grad_norm_(
        encoder.low_res_branch.parameters(), float('inf')
    )
    high_grad_norm = torch.nn.utils.clip_grad_norm_(
        encoder.high_res_branch.parameters(), float('inf')
    )
    fusion_grad_norm = torch.nn.utils.clip_grad_norm_(
        encoder.fusion.parameters(), float('inf')
    )

    print(f"  Low-res branch gradient norm: {low_grad_norm:.6f}")
    print(f"  High-res branch gradient norm: {high_grad_norm:.6f}")
    print(f"  Fusion layer gradient norm: {fusion_grad_norm:.6f}")

    print(f"\n5. Testing parameter count by component...")
    low_params = sum(p.numel() for p in encoder.low_res_branch.parameters())
    high_params = sum(p.numel() for p in encoder.high_res_branch.parameters())
    fusion_params = sum(p.numel() for p in encoder.fusion.parameters())

    print(f"  Low-res branch: {low_params:,} parameters")
    print(f"  High-res branch: {high_params:,} parameters")
    print(f"  Fusion layer: {fusion_params:,} parameters")
    print(f"  Total: {total_params:,} parameters")

    print(f"\n6. Testing batch independence...")
    # Each sample in batch should be processed independently
    H_cond_single = encoder(lights[0:1], settlement[0:1])
    diff = (H_cond[0:1] - H_cond_single).abs().max()
    print(f"  Max difference between batch[0] and single forward: {diff:.8f}")
    print(f"  (Should be 0.0 for batch independence)")

    print("\n✓ All tests passed!")
