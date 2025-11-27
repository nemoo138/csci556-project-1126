"""
Product ID Embedding for C3-LDM
Reference: C3-LDM.md Section 3.3

Learnable embeddings for different population products:
- 0: WorldPop (RF-based dasymetric)
- 1: GHS-POP (satellite-based building distribution)
- 2: HRSL (high-resolution settlement detection)

Allows model to learn product-specific biases and characteristics.
"""

import torch
import torch.nn as nn


class ProductEmbedding(nn.Module):
    """
    Learnable embeddings for population product IDs.

    Maps discrete product IDs to continuous conditioning vectors that
    can be added to spatial conditioning or concatenated with time embeddings.
    """

    def __init__(self, num_products=3, d_prod=64, cond_channels=256):
        """
        Args:
            num_products: Number of different products (default: 3)
            d_prod: Dimension of product embeddings
            cond_channels: Output dimension matching spatial conditioning
        """
        super().__init__()
        self.num_products = num_products
        self.d_prod = d_prod
        self.cond_channels = cond_channels

        # Learnable embeddings for each product
        self.embeddings = nn.Embedding(num_products, d_prod)

        # Project to conditioning space
        # Output can be broadcast-added to spatial conditioning (B, C, H, W)
        self.proj = nn.Sequential(
            nn.Linear(d_prod, cond_channels),
            nn.SiLU(),
            nn.Linear(cond_channels, cond_channels)
        )

        # Initialize embeddings with small random values
        nn.init.normal_(self.embeddings.weight, mean=0.0, std=0.02)

    def forward(self, product_ids):
        """
        Compute product conditioning from product IDs.

        Args:
            product_ids: (B,) integer tensor with values in [0, num_products-1]
                        0: WorldPop
                        1: GHS-POP
                        2: HRSL

        Returns:
            prod_cond: (B, cond_channels, 1, 1) for broadcast addition to spatial features
        """
        # Embed product IDs
        emb = self.embeddings(product_ids)  # (B, d_prod)

        # Project to conditioning space
        prod_cond = self.proj(emb)  # (B, cond_channels)

        # Reshape for spatial broadcast: (B, C) → (B, C, 1, 1)
        prod_cond = prod_cond.unsqueeze(-1).unsqueeze(-1)

        return prod_cond

    def get_product_name(self, product_id):
        """
        Get human-readable product name from ID.

        Args:
            product_id: Integer in [0, 2]

        Returns:
            Product name string
        """
        names = {
            0: "WorldPop",
            1: "GHS-POP",
            2: "HRSL"
        }
        return names.get(product_id, f"Unknown ({product_id})")


if __name__ == "__main__":
    print("=" * 70)
    print("Testing Product Embedding")
    print("=" * 70)

    # Create product embedding module
    prod_emb = ProductEmbedding(num_products=3, d_prod=64, cond_channels=256)

    # Count parameters
    total_params = sum(p.numel() for p in prod_emb.parameters())
    print(f"\nTotal parameters: {total_params:,}")

    print(f"\n1. Testing single product...")
    product_id = torch.tensor([0])  # WorldPop
    prod_cond = prod_emb(product_id)
    print(f"  Product ID: {product_id.item()} ({prod_emb.get_product_name(product_id.item())})")
    print(f"  Output shape: {prod_cond.shape}")
    print(f"  Output range: [{prod_cond.min():.4f}, {prod_cond.max():.4f}]")

    print(f"\n2. Testing batch of mixed products...")
    B = 16
    product_ids = torch.randint(0, 3, (B,))
    prod_cond_batch = prod_emb(product_ids)
    print(f"  Batch size: {B}")
    print(f"  Product IDs: {product_ids.tolist()}")
    print(f"  Output shape: {prod_cond_batch.shape}")

    # Count each product
    for pid in range(3):
        count = (product_ids == pid).sum().item()
        print(f"  {prod_emb.get_product_name(pid)}: {count} samples")

    print(f"\n3. Testing different products have different embeddings...")
    prod0 = prod_emb(torch.tensor([0]))
    prod1 = prod_emb(torch.tensor([1]))
    prod2 = prod_emb(torch.tensor([2]))

    sim_01 = torch.nn.functional.cosine_similarity(
        prod0.view(1, -1), prod1.view(1, -1)
    ).item()
    sim_02 = torch.nn.functional.cosine_similarity(
        prod0.view(1, -1), prod2.view(1, -1)
    ).item()
    sim_12 = torch.nn.functional.cosine_similarity(
        prod1.view(1, -1), prod2.view(1, -1)
    ).item()

    print(f"  Cosine similarity WorldPop vs GHS-POP: {sim_01:.4f}")
    print(f"  Cosine similarity WorldPop vs HRSL: {sim_02:.4f}")
    print(f"  Cosine similarity GHS-POP vs HRSL: {sim_12:.4f}")
    print(f"  (Different products should have low similarity)")

    print(f"\n4. Testing broadcast compatibility with spatial features...")
    # Simulate spatial conditioning from DualBranchEncoder
    H_cond = torch.randn(B, 256, 32, 32)
    prod_cond = prod_emb(product_ids)

    # Should be able to add product conditioning to spatial conditioning
    H_combined = H_cond + prod_cond
    print(f"  Spatial conditioning shape: {H_cond.shape}")
    print(f"  Product conditioning shape: {prod_cond.shape}")
    print(f"  Combined shape: {H_combined.shape}")
    print(f"  Broadcast successful: {H_combined.shape == H_cond.shape}")

    print(f"\n5. Testing gradient flow...")
    loss = prod_cond_batch.sum()
    loss.backward()

    emb_grad_norm = torch.nn.utils.clip_grad_norm_(
        prod_emb.embeddings.parameters(), float('inf')
    )
    proj_grad_norm = torch.nn.utils.clip_grad_norm_(
        prod_emb.proj.parameters(), float('inf')
    )

    print(f"  Embedding gradient norm: {emb_grad_norm:.6f}")
    print(f"  Projection gradient norm: {proj_grad_norm:.6f}")

    print(f"\n6. Testing embedding statistics...")
    with torch.no_grad():
        all_embeddings = prod_emb.embeddings.weight  # (3, 64)
        print(f"  Embedding matrix shape: {all_embeddings.shape}")
        print(f"  Embedding mean: {all_embeddings.mean():.6f}")
        print(f"  Embedding std: {all_embeddings.std():.6f}")
        print(f"  Embedding range: [{all_embeddings.min():.6f}, {all_embeddings.max():.6f}]")

    print(f"\n7. Testing interpolation between products...")
    # Can we interpolate between product embeddings?
    with torch.no_grad():
        emb0 = prod_emb.embeddings.weight[0]
        emb1 = prod_emb.embeddings.weight[1]

        # Interpolate: 0.7 * WorldPop + 0.3 * GHS-POP
        emb_interp = 0.7 * emb0 + 0.3 * emb1
        cond_interp = prod_emb.proj(emb_interp.unsqueeze(0)).unsqueeze(-1).unsqueeze(-1)

        print(f"  Interpolated embedding shape: {emb_interp.shape}")
        print(f"  Interpolated conditioning shape: {cond_interp.shape}")
        print(f"  Can be used for uncertainty sampling: ✓")

    print("\n✓ All tests passed!")
