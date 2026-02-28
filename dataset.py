"""
Multi-Product Dataset for C3-LDM Training
Reference: IMPLEMENTATION_ROADMAP.md Phase 5.1

Loads paired data from WorldPop, GHS-POP, and HRSL
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class MultiProductDataset(Dataset):
    """
    Dataset for multi-product population mapping.

    Loads:
    - VIIRS nighttime lights (low-res, coarse)
    - WSF settlement footprint (high-res, fine)
    - Population targets from WorldPop, GHS-POP, or HRSL
    - Admin unit IDs and census totals (if available)
    """

    # Product name to ID mapping
    PRODUCT_TO_ID = {
        'WorldPop': 0,
        'GHS-POP': 1,
        'HRSL': 2
    }

    def __init__(
        self,
        pairing_csv,
        data_root='data',
        normalize=True,
        return_census=False,
        products=None  # Filter: list of products to include, e.g., ['WorldPop']
    ):
        """
        Args:
            pairing_csv: Path to multi_product_pairing.csv
            data_root: Root directory for data files
            normalize: Whether to normalize inputs to [0, 1]
            return_census: Whether to return admin IDs and census totals
            products: List of products to include. If None, include all.
                      Options: ['WorldPop', 'GHS-POP', 'HRSL']
        """
        self.data_root = data_root
        self.normalize = normalize
        self.return_census = return_census

        # Load pairing metadata
        self.pairing = pd.read_csv(pairing_csv)
        
        # Filter by products if specified
        if products is not None:
            original_len = len(self.pairing)
            self.pairing = self.pairing[self.pairing['product'].isin(products)].reset_index(drop=True)
            print(f"Filtered to products: {products}")
            print(f"  Before: {original_len:,} samples")
            print(f"  After:  {len(self.pairing):,} samples")
        
        print(f"Loaded {len(self.pairing):,} paired samples")
        print(f"Product distribution:")
        print(self.pairing['product'].value_counts())

        # Build lookup index for finding corresponding products by features file
        # This allows us to find WorldPop/GHS-POP tiles that match an HRSL tile
        self._build_features_lookup()

    def _build_features_lookup(self):
        """Build a lookup dictionary to find all products for a given features file."""
        self.features_to_rows = {}
        for idx, row in self.pairing.iterrows():
            features_file = row['wp_features_file']
            if features_file not in self.features_to_rows:
                self.features_to_rows[features_file] = {}
            self.features_to_rows[features_file][row['product']] = idx
        
        print(f"Built features lookup with {len(self.features_to_rows):,} unique locations")

    def __len__(self):
        return len(self.pairing)

    def __getitem__(self, idx):
        """
        Returns:
            dict with keys:
                'lights': (1, 256, 256) VIIRS nighttime lights
                'settlement': (1, 256, 256) WSF settlement footprint
                'lights_mask': (1, 256, 256) missingness mask for lights
                'settlement_mask': (1, 256, 256) missingness mask for settlement
                'target': (1, 256, 256) population density map
                'product_id': int (0=WorldPop, 1=GHS-POP, 2=HRSL)
                'admin_ids': (256, 256) admin unit IDs [optional]
                'census_totals': (max_admin_units,) census totals [optional]
        """
        row = self.pairing.iloc[idx]

        # Load features (VIIRS + WSF)
        # WorldPop features are in tiles_2020/project/tiles_2020/
        features_path = os.path.join(
            self.data_root,
            'tiles_2020/project/tiles_2020',
            row['wp_features_file']
        )
        features = np.load(features_path)  # Expected shape: (2, 256, 256)

        lights = features[0:1]  # (1, 256, 256) VIIRS
        settlement = features[1:2]  # (1, 256, 256) WSF

        # Missingness masks (1 = valid, 0 = missing/corrupt)
        lights_mask = np.isfinite(lights).astype(np.float32)
        settlement_mask = np.isfinite(settlement).astype(np.float32)

        # Check for NaN/Inf in features
        if np.isnan(lights).any() or np.isinf(lights).any():
            print(f"\n⚠️  WARNING: NaN/Inf in lights from {row['wp_features_file']}")
            lights = np.nan_to_num(lights, nan=0.0, posinf=0.0, neginf=0.0)
        if np.isnan(settlement).any() or np.isinf(settlement).any():
            print(f"\n⚠️  WARNING: NaN/Inf in settlement from {row['wp_features_file']}")
            settlement = np.nan_to_num(settlement, nan=0.0, posinf=0.0, neginf=0.0)

        # Normalize to [0, 1] if requested
        if self.normalize:
            lights = self._normalize(lights)
            settlement = self._normalize(settlement)

        # Load target based on product
        target = self._load_target(row)

        # Get product ID
        product_id = self.PRODUCT_TO_ID[row['product']]

        # Prepare output
        sample = {
            'lights': torch.from_numpy(lights).float(),
            'settlement': torch.from_numpy(settlement).float(),
            'lights_mask': torch.from_numpy(lights_mask).float(),
            'settlement_mask': torch.from_numpy(settlement_mask).float(),
            'target': torch.from_numpy(target).float(),
            'product_id': torch.tensor(product_id, dtype=torch.long),
        }

        # Optionally load census data
        if self.return_census:
            admin_ids, census_totals = self._load_census(row)
            sample['admin_ids'] = torch.from_numpy(admin_ids).long()
            sample['census_totals'] = torch.from_numpy(census_totals).float()

        return sample

    def _load_worldpop_target(self, features_file):
        """Load WorldPop target for a given features file."""
        if features_file not in self.features_to_rows:
            return None
        if 'WorldPop' not in self.features_to_rows[features_file]:
            return None
        
        wp_idx = self.features_to_rows[features_file]['WorldPop']
        wp_row = self.pairing.iloc[wp_idx]
        
        target_path = os.path.join(
            self.data_root,
            'tiles_2020/project/tiles_2020',
            wp_row['product_file']
        )
        
        if not os.path.exists(target_path):
            return None
            
        target = np.load(target_path)  # (1, 256, 256)
        target = np.maximum(target, 0.0)
        
        # Handle NaN in WorldPop (replace with 0)
        if np.isnan(target).any() or np.isinf(target).any():
            target = np.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)
        
        return target.astype(np.float32)

    def _load_ghspop_target(self, features_file):
        """Load GHS-POP target for a given features file."""
        if features_file not in self.features_to_rows:
            return None
        if 'GHS-POP' not in self.features_to_rows[features_file]:
            return None
        
        ghs_idx = self.features_to_rows[features_file]['GHS-POP']
        ghs_row = self.pairing.iloc[ghs_idx]
        
        file_path = os.path.join(
            self.data_root,
            'GHS_POP/GHS_POP_2020_USA_patch256_s128',
            ghs_row['product_file']
        )
        
        if not os.path.exists(file_path):
            return None
            
        target = np.load(file_path)  # (256, 256)
        target = target[np.newaxis, :, :]  # (1, 256, 256)
        target = np.maximum(target, 0.0)
        
        # Handle NaN in GHS-POP (replace with 0)
        if np.isnan(target).any() or np.isinf(target).any():
            target = np.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)
        
        return target.astype(np.float32)

    def _load_target(self, row):
        """Load population target based on product type."""
        product = row['product']
        product_file = row['product_file']

        if product == 'WorldPop':
            # WorldPop target is in tiles_2020/project/tiles_2020/ directory
            # Note: WorldPop files are already (1, 256, 256)
            target_path = os.path.join(
                self.data_root,
                'tiles_2020/project/tiles_2020',
                product_file
            )
            target = np.load(target_path)  # Already (1, 256, 256)

            # Check for NaN/Inf in WorldPop
            if np.isnan(target).any() or np.isinf(target).any():
                print(f"\n⚠️  WARNING: NaN/Inf detected in WorldPop file: {product_file}")
                print(f"  NaN count: {np.isnan(target).sum()}")
                print(f"  Inf count: {np.isinf(target).sum()}")
                print(f"  Replacing with zeros...")
                target = np.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)

        elif product == 'GHS-POP':
            # GHS-POP patches - each file is a single (256, 256) patch
            file_path = os.path.join(
                self.data_root,
                'GHS_POP/GHS_POP_2020_USA_patch256_s128',
                product_file
            )
            target = np.load(file_path)  # (256, 256)
            target = target[np.newaxis, :, :]  # (1, 256, 256)

            # Check for NaN/Inf in GHS-POP
            if np.isnan(target).any() or np.isinf(target).any():
                print(f"\n⚠️  WARNING: NaN/Inf detected in GHS-POP file: {product_file}")
                print(f"  NaN count: {np.isnan(target).sum()}")
                print(f"  Inf count: {np.isinf(target).sum()}")
                print(f"  Replacing with zeros...")
                target = np.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)

        elif product == 'HRSL':
            # HRSL patches - files are in patches_conus/ subdirectory
            file_path = os.path.join(
                self.data_root,
                'HRSL/hrsl_patches_100m/patches_conus',
                product_file
            )
            target = np.load(file_path)  # (256, 256)
            target = target[np.newaxis, :, :]  # (1, 256, 256)

            # Check for NaN/Inf in HRSL - replace with average of WorldPop and GHS-POP
            if np.isnan(target).any() or np.isinf(target).any():
                nan_mask = np.isnan(target) | np.isinf(target)
                nan_count = nan_mask.sum()
                
                # print(f"\n⚠️  WARNING: NaN/Inf detected in HRSL file: {product_file}")
                # print(f"  NaN count: {nan_count}")
                
                # Get corresponding features file to find matching products
                features_file = row['wp_features_file']
                
                # Load WorldPop and GHS-POP for the same location
                wp_target = self._load_worldpop_target(features_file)
                ghs_target = self._load_ghspop_target(features_file)
                
                # Compute replacement values
                if wp_target is not None and ghs_target is not None:
                    # Average of WorldPop and GHS-POP
                    replacement = (wp_target + ghs_target) / 2.0
                    # print(f"  Replacing with average of WorldPop and GHS-POP")
                elif wp_target is not None:
                    # Only WorldPop available
                    replacement = wp_target
                    print(f"  Replacing with WorldPop (GHS-POP not available)")
                elif ghs_target is not None:
                    # Only GHS-POP available
                    replacement = ghs_target
                    print(f"  Replacing with GHS-POP (WorldPop not available)")
                else:
                    # Neither available, fall back to zeros
                    replacement = np.zeros_like(target)
                    print(f"  Replacing with zeros (no matching products found)")
                
                # Apply replacement only to NaN/Inf pixels
                target = np.where(nan_mask, replacement, target)
                
                # Final safety check
                target = np.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)

        else:
            raise ValueError(f"Unknown product: {product}")

        # Ensure non-negative (population can't be negative)
        target = np.maximum(target, 0.0)

        return target.astype(np.float32)

    def _load_census(self, row):
        """
        Load admin unit IDs and census totals.

        For now, returns dummy data. In production, this would load
        actual administrative boundaries and census data.
        """
        # TODO: Implement actual census data loading
        # For now, return dummy data (no admin units)
        admin_ids = np.full((256, 256), -1, dtype=np.int64)  # -1 = no admin unit
        census_totals = np.zeros(100, dtype=np.float32)  # Max 100 admin units

        return admin_ids, census_totals

    def _normalize(self, x, eps=1e-6):
        """Normalize to [0, 1] range."""
        x_min = x.min()
        x_max = x.max()
        if x_max - x_min < eps:
            return np.zeros_like(x)
        return (x - x_min) / (x_max - x_min + eps)


if __name__ == "__main__":
    print("=" * 70)
    print("Testing MultiProductDataset")
    print("=" * 70)

    # Load dataset
    pairing_csv = 'data/paired_dataset/multi_product_pairing.csv'

    if not os.path.exists(pairing_csv):
        print(f"\nError: {pairing_csv} not found!")
        print("Please run the pairing script first to generate this file.")
        exit(1)

    # Test 1: WorldPop only (recommended for initial training)
    print(f"\n1. Loading WorldPop-only dataset...")
    dataset_wp = MultiProductDataset(
        pairing_csv=pairing_csv,
        data_root='data',
        normalize=True,
        return_census=False,
        products=['WorldPop']  # Filter to WorldPop only
    )
    print(f"   WorldPop-only size: {len(dataset_wp):,} samples")

    # Test 2: All products (for comparison)
    print(f"\n2. Loading full multi-product dataset...")
    dataset_all = MultiProductDataset(
        pairing_csv=pairing_csv,
        data_root='data',
        normalize=True,
        return_census=False,
        products=None  # All products
    )
    print(f"   Multi-product size: {len(dataset_all):,} samples")

    print(f"\n3. Testing __getitem__ on WorldPop dataset...")
    try:
        sample = dataset_wp[0]
        print(f"  Sample keys: {list(sample.keys())}")
        print(f"  Lights shape: {sample['lights'].shape}")
        print(f"  Settlement shape: {sample['settlement'].shape}")
        print(f"  Target shape: {sample['target'].shape}")
        print(f"  Product ID: {sample['product_id'].item()} (should be 0 for WorldPop)")
        print(f"  Lights range: [{sample['lights'].min():.4f}, {sample['lights'].max():.4f}]")
        print(f"  Settlement range: [{sample['settlement'].min():.4f}, {sample['settlement'].max():.4f}]")
        print(f"  Target range: [{sample['target'].min():.4f}, {sample['target'].max():.4f}]")
        print(f"  Target sum: {sample['target'].sum():.1f}")
    except Exception as e:
        print(f"  Error loading sample: {e}")
        print("  This is expected if data files haven't been generated yet.")

    print(f"\n4. Testing DataLoader with WorldPop...")
    from torch.utils.data import DataLoader

    loader = DataLoader(
        dataset_wp,
        batch_size=4,
        shuffle=True,
        num_workers=0  # Use 0 for testing
    )

    try:
        batch = next(iter(loader))
        print(f"  Batch lights shape: {batch['lights'].shape}")
        print(f"  Batch target shape: {batch['target'].shape}")
        print(f"  Product IDs in batch: {batch['product_id'].tolist()} (all should be 0)")
        print(f"  Target sums: {[f'{t.sum():.0f}' for t in batch['target']]}")
    except Exception as e:
        print(f"  Error loading batch: {e}")

    print(f"\n5. Verifying all samples are WorldPop...")
    try:
        all_wp = all(dataset_wp[i]['product_id'].item() == 0 for i in range(min(100, len(dataset_wp))))
        print(f"  First 100 samples all WorldPop: {all_wp}")
    except Exception as e:
        print(f"  Error checking: {e}")

    print("\n✓ Dataset tests completed!")
    print("\nTo train on WorldPop only:")
    print("  python train.py --products WorldPop")
