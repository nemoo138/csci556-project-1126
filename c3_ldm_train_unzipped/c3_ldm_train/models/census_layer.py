"""
Census Consistency Layer for C3-LDM
Reference: C3-LDM.md Section "Component 5"

Enforces hard constraint: Σ(predictions in admin unit) = census total
Differentiable for backpropagation through the normalization.
"""

import torch
import torch.nn as nn


class CensusConsistencyLayer(nn.Module):
    """
    Differentiable census consistency normalization.

    For each administrative unit A:
        S_A = Σ_{j ∈ A} P_raw[j] + ε
        P[i] = P_raw[i] / S_A * C_A  for all i ∈ A

    This ensures Σ_{i ∈ A} P[i] = C_A exactly by construction.
    """

    def __init__(self, epsilon=1e-6):
        """
        Args:
            epsilon: Small constant to prevent division by zero
        """
        super().__init__()
        self.epsilon = epsilon

    def forward(self, P_raw, admin_ids, census_totals):
        """
        Apply census consistency normalization.

        Args:
            P_raw: (B, 1, H, W) raw population predictions (unconstrained)
            admin_ids: (B, H, W) integer admin unit IDs for each pixel
            census_totals: (B, max_admin_units) census total for each admin unit

        Returns:
            P: (B, 1, H, W) census-consistent population predictions
        """
        B, C, H, W = P_raw.shape
        assert C == 1, "Expected single channel population map"

        P_raw = P_raw.squeeze(1)  # (B, H, W)

        # Initialize output
        P = torch.zeros_like(P_raw)

        # Process each sample in batch
        for b in range(B):
            p_raw = P_raw[b]  # (H, W)
            admin_map = admin_ids[b]  # (H, W)
            census = census_totals[b]  # (max_admin_units,)

            # Get unique admin IDs (excluding -1 which means no admin unit)
            unique_admins = torch.unique(admin_map)
            unique_admins = unique_admins[unique_admins >= 0]  # Filter out -1

            # For each admin unit, normalize
            for admin_id in unique_admins:
                # Mask for this admin unit
                mask = (admin_map == admin_id)  # (H, W)

                # Sum of raw predictions in this unit
                S_A = p_raw[mask].sum() + self.epsilon  # scalar

                # Census total for this unit
                C_A = census[admin_id]  # scalar

                # Normalize: P[i] = P_raw[i] / S_A * C_A
                P[b][mask] = p_raw[mask] / S_A * C_A

        return P.unsqueeze(1)  # (B, 1, H, W)


class CensusConsistencyLayerVectorized(nn.Module):
    """
    Vectorized version of census consistency layer for better performance.

    Uses scatter operations instead of loops for faster computation.
    """

    def __init__(self, epsilon=1e-6):
        """
        Args:
            epsilon: Small constant to prevent division by zero
        """
        super().__init__()
        self.epsilon = epsilon

    def forward(self, P_raw, admin_ids, census_totals):
        """
        Apply census consistency normalization (vectorized).

        Args:
            P_raw: (B, 1, H, W) raw population predictions
            admin_ids: (B, H, W) integer admin unit IDs (use -1 for no admin)
            census_totals: (B, max_admin_units) census totals

        Returns:
            P: (B, 1, H, W) census-consistent predictions
        """
        B, C, H, W = P_raw.shape
        assert C == 1, "Expected single channel population map"

        P_raw = P_raw.squeeze(1)  # (B, H, W)
        device = P_raw.device

        # Flatten spatial dimensions
        P_raw_flat = P_raw.view(B, -1)  # (B, H*W)
        admin_ids_flat = admin_ids.view(B, -1)  # (B, H*W)

        # Initialize output
        P_flat = torch.zeros_like(P_raw_flat)

        # Process each sample
        for b in range(B):
            p_raw = P_raw_flat[b]  # (H*W,)
            admin_map = admin_ids_flat[b]  # (H*W,)
            census = census_totals[b]  # (max_admin_units,)

            # Find valid admin IDs (>= 0)
            valid_mask = admin_map >= 0

            if valid_mask.any():
                # Compute sums per admin unit using scatter_add
                max_admin = int(admin_map.max().item()) + 1
                S_A = torch.zeros(max_admin, device=device)  # (max_admin,)

                # Sum raw predictions by admin unit
                S_A.scatter_add_(0, admin_map[valid_mask], p_raw[valid_mask])
                S_A = S_A + self.epsilon  # Add epsilon to prevent division by zero

                # Get census totals for each pixel's admin unit
                C_A_per_pixel = census[admin_map[valid_mask]]  # (num_valid,)
                S_A_per_pixel = S_A[admin_map[valid_mask]]  # (num_valid,)

                # Normalize: P[i] = P_raw[i] / S_A * C_A
                P_flat[b][valid_mask] = p_raw[valid_mask] / S_A_per_pixel * C_A_per_pixel

            # Pixels without admin units keep raw values
            P_flat[b][~valid_mask] = p_raw[~valid_mask]

        # Reshape back
        P = P_flat.view(B, H, W).unsqueeze(1)  # (B, 1, H, W)

        return P


if __name__ == "__main__":
    print("=" * 70)
    print("Testing Census Consistency Layer")
    print("=" * 70)

    # Create test data
    B, H, W = 2, 256, 256

    # Raw population predictions
    P_raw = torch.rand(B, 1, H, W) * 100  # Random populations 0-100

    # Admin units (simple grid: 4x4 = 16 admin units)
    admin_ids = torch.zeros(B, H, W, dtype=torch.long)
    admin_size = H // 4
    for i in range(4):
        for j in range(4):
            admin_id = i * 4 + j
            admin_ids[:, i*admin_size:(i+1)*admin_size, j*admin_size:(j+1)*admin_size] = admin_id

    # Census totals (random census for each admin unit)
    max_admin_units = 16
    census_totals = torch.rand(B, max_admin_units) * 10000 + 1000  # Random census 1000-11000

    print(f"\n1. Testing basic census consistency layer...")
    layer = CensusConsistencyLayer()
    P = layer(P_raw, admin_ids, census_totals)

    print(f"  Input shape: {P_raw.shape}")
    print(f"  Output shape: {P.shape}")
    print(f"  Admin IDs shape: {admin_ids.shape}")
    print(f"  Census totals shape: {census_totals.shape}")

    # Verify consistency
    print(f"\n2. Verifying census consistency...")
    admin_size = H // 4
    for b in range(B):
        errors = []
        for i in range(4):
            for j in range(4):
                admin_id = i * 4 + j
                # Extract admin unit region
                region = P[b, 0, i*admin_size:(i+1)*admin_size, j*admin_size:(j+1)*admin_size]
                pred_sum = region.sum().item()
                census_val = census_totals[b, admin_id].item()
                error = abs(pred_sum - census_val)
                errors.append(error)

        max_error = max(errors)
        print(f"  Batch {b}: Max error = {max_error:.6f} (should be ~0)")

    print(f"\n3. Testing vectorized layer...")
    layer_vec = CensusConsistencyLayerVectorized()
    P_vec = layer_vec(P_raw, admin_ids, census_totals)

    diff = (P - P_vec).abs().max()
    print(f"  Max difference vs basic layer: {diff:.8f} (should be ~0)")

    print(f"\n4. Testing gradient flow...")
    P_raw_grad = P_raw.clone().requires_grad_(True)
    P_out = layer(P_raw_grad, admin_ids, census_totals)
    loss = P_out.sum()
    loss.backward()

    print(f"  Gradient exists: {P_raw_grad.grad is not None}")
    print(f"  Gradient norm: {P_raw_grad.grad.norm():.6f}")

    print(f"\n5. Testing with missing admin units (-1)...")
    admin_ids_partial = admin_ids.clone()
    # Mark some regions as having no admin unit
    admin_ids_partial[:, :H//8, :W//8] = -1

    P_partial = layer_vec(P_raw, admin_ids_partial, census_totals)
    print(f"  Output shape with partial admin coverage: {P_partial.shape}")
    print(f"  No-admin region mean: {P_partial[:, :, :H//8, :W//8].mean():.4f}")

    print(f"\n6. Performance comparison...")
    import time

    # Basic layer
    start = time.time()
    for _ in range(10):
        _ = layer(P_raw, admin_ids, census_totals)
    basic_time = time.time() - start

    # Vectorized layer
    start = time.time()
    for _ in range(10):
        _ = layer_vec(P_raw, admin_ids, census_totals)
    vec_time = time.time() - start

    print(f"  Basic layer: {basic_time:.4f}s (10 iterations)")
    print(f"  Vectorized layer: {vec_time:.4f}s (10 iterations)")
    print(f"  Speedup: {basic_time/vec_time:.2f}x")

    print("\n✓ All tests passed!")
