"""
Baseline Dasymetric Module for C3-LDM
Reference: C3-LDM.md Section "Component 1"

Simple lights × settlement baseline that the diffusion model will correct.
Provides interpretable starting point and helps with heavy-tailed distributions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaselineDasymetric(nn.Module):
    """
    Baseline dasymetric allocation: score = (lights + λ_L) × (settlement + λ_S)
    Normalized per admin unit to match census totals.

    This is NOT learned - it's a simple heuristic baseline.
    The diffusion model learns residuals relative to this baseline.
    """

    def __init__(self, lambda_lights=0.01, lambda_settlement=0.01, eps=1e-8):
        """
        Args:
            lambda_lights: Small positive for dark but settled cells
            lambda_settlement: Small positive for lights-only areas (ports, highways)
            eps: Small constant to avoid division by zero
        """
        super().__init__()
        self.lambda_L = lambda_lights
        self.lambda_S = lambda_settlement
        self.eps = eps

    def forward(self, lights, settlement, admin_ids=None, census_totals=None):
        """
        Compute baseline population allocation.

        Args:
            lights: (B, 1, H, W) VIIRS nighttime lights
            settlement: (B, 1, H, W) World Settlement Footprint
            admin_ids: (B, H, W) admin unit IDs per pixel (optional)
            census_totals: dict {admin_id: total_population} (optional)

        Returns:
            baseline: (B, 1, H, W) baseline population density
        """
        # Compute raw scores
        # score[i] = (L[i] + λ_L) × (S[i] + λ_S)
        score = (lights + self.lambda_L) * (settlement + self.lambda_S)

        # If no census constraints, return raw score
        if admin_ids is None or census_totals is None:
            return score

        # Normalize to census totals per admin unit
        baseline = self._normalize_to_census(score, admin_ids, census_totals)

        return baseline

    def _normalize_to_census(self, score, admin_ids, census_totals):
        """
        Normalize scores to match census totals per admin unit.

        Args:
            score: (B, 1, H, W) raw dasymetric scores
            admin_ids: (B, H, W) admin unit ID per pixel
            census_totals: dict {admin_id: population_total}

        Returns:
            normalized: (B, 1, H, W) census-consistent population
        """
        B, C, H, W = score.shape
        device = score.device

        # Initialize output
        baseline = torch.zeros_like(score)

        # Process each sample in batch
        for b in range(B):
            score_b = score[b, 0]  # (H, W)
            admin_b = admin_ids[b]  # (H, W)

            # Get unique admin IDs in this tile
            unique_admins = torch.unique(admin_b)

            for admin_id in unique_admins:
                admin_id_item = admin_id.item()

                # Skip if no census data for this admin
                if admin_id_item not in census_totals:
                    continue

                # Get census total for this admin
                C_A = census_totals[admin_id_item]

                # Mask for this admin unit
                mask = (admin_b == admin_id)

                # Sum of scores in this admin unit
                S_A = score_b[mask].sum() + self.eps

                # Normalize: P[i] = score[i] / S_A * C_A for i ∈ A
                baseline[b, 0][mask] = score_b[mask] / S_A * C_A

        return baseline

    def compute_residual_target(self, population, lights, settlement,
                                admin_ids=None, census_totals=None, eps=1e-6):
        """
        Compute residual log-ratio for diffusion model training.

        Args:
            population: (B, 1, H, W) true population density
            lights, settlement: (B, 1, H, W) features
            admin_ids, census_totals: census constraints (optional)
            eps: small constant for numerical stability

        Returns:
            residual: (B, 1, H, W) log-ratio: log((pop + ε) / (baseline + ε))
        """
        # Compute baseline
        baseline = self.forward(lights, settlement, admin_ids, census_totals)

        # Compute log-residual
        # R[i] = log((Y[i] + ε) / (B[i] + ε))
        residual = torch.log((population + eps) / (baseline + eps))

        return residual, baseline

    def reconstruct_from_residual(self, residual, lights, settlement,
                                  admin_ids=None, census_totals=None):
        """
        Reconstruct population from residual prediction.

        Args:
            residual: (B, 1, H, W) predicted log-ratio
            lights, settlement: (B, 1, H, W) features
            admin_ids, census_totals: census constraints (optional)

        Returns:
            population: (B, 1, H, W) reconstructed population density
        """
        # Compute baseline
        baseline = self.forward(lights, settlement, admin_ids, census_totals)

        # Reconstruct: P[i] = B[i] × exp(R[i])
        population = baseline * torch.exp(residual)

        # Ensure non-negative
        population = torch.clamp(population, min=0.0)

        return population


if __name__ == "__main__":
    # Test baseline module
    print("=" * 70)
    print("Testing Baseline Dasymetric Module")
    print("=" * 70)

    # Create dummy data
    B, H, W = 2, 256, 256
    lights = torch.rand(B, 1, H, W) * 0.5  # VIIRS values typically 0-1
    settlement = torch.rand(B, 1, H, W) * 0.3  # WSF values typically 0-1
    population = torch.rand(B, 1, H, W) * 10  # Population density

    # Create dummy admin IDs (4 admin units per tile)
    admin_ids = torch.zeros(B, H, W, dtype=torch.long)
    admin_ids[:, :H//2, :W//2] = 1
    admin_ids[:, :H//2, W//2:] = 2
    admin_ids[:, H//2:, :W//2] = 3
    admin_ids[:, H//2:, W//2:] = 4

    # Create dummy census totals
    census_totals = {
        1: 10000.0,
        2: 15000.0,
        3: 12000.0,
        4: 8000.0,
    }

    # Initialize module
    baseline_model = BaselineDasymetric()

    print("\n1. Testing baseline allocation...")
    baseline = baseline_model(lights, settlement, admin_ids, census_totals)
    print(f"  Input lights shape: {lights.shape}")
    print(f"  Input settlement shape: {settlement.shape}")
    print(f"  Output baseline shape: {baseline.shape}")
    print(f"  Baseline range: [{baseline.min():.4f}, {baseline.max():.4f}]")

    # Check census consistency
    print("\n2. Checking census consistency...")
    for admin_id in [1, 2, 3, 4]:
        mask = (admin_ids[0] == admin_id)
        pred_total = baseline[0, 0][mask].sum().item()
        true_total = census_totals[admin_id]
        error = abs(pred_total - true_total)
        print(f"  Admin {admin_id}: predicted={pred_total:.2f}, census={true_total:.2f}, error={error:.6f}")

    print("\n3. Testing residual computation...")
    residual, baseline_check = baseline_model.compute_residual_target(
        population, lights, settlement, admin_ids, census_totals
    )
    print(f"  Residual shape: {residual.shape}")
    print(f"  Residual range: [{residual.min():.4f}, {residual.max():.4f}]")
    print(f"  Residual mean: {residual.mean():.4f}, std: {residual.std():.4f}")

    print("\n4. Testing reconstruction...")
    reconstructed = baseline_model.reconstruct_from_residual(
        residual, lights, settlement, admin_ids, census_totals
    )
    print(f"  Reconstructed shape: {reconstructed.shape}")
    print(f"  Reconstruction error (MSE): {F.mse_loss(reconstructed, population):.6f}")

    print("\n5. Testing without census constraints...")
    baseline_unconstrained = baseline_model(lights, settlement)
    print(f"  Unconstrained baseline shape: {baseline_unconstrained.shape}")
    print(f"  Unconstrained range: [{baseline_unconstrained.min():.4f}, {baseline_unconstrained.max():.4f}]")

    print("\n✓ All tests passed!")
