"""
Fast spatial pairing of multi-product population datasets.
Creates pairing metadata for C3-LDM multi-product supervision.
"""
import numpy as np
import pandas as pd
import os
from tqdm import tqdm


def build_spatial_index(meta_df):
    """Build a simple grid-based spatial index for fast lookups."""
    # Create a dictionary mapping lat/lon grid cells to tile indices
    grid_size = 0.5  # degrees

    spatial_index = {}
    for idx, row in meta_df.iterrows():
        center_lon = (row['lon_min'] + row['lon_max']) / 2
        center_lat = (row['lat_min'] + row['lat_max']) / 2

        grid_lon = int(np.floor(center_lon / grid_size))
        grid_lat = int(np.floor(center_lat / grid_size))

        key = (grid_lon, grid_lat)
        if key not in spatial_index:
            spatial_index[key] = []
        spatial_index[key].append(idx)

    return spatial_index, grid_size


def find_containing_tile(center_lon, center_lat, meta_df, spatial_index, grid_size):
    """
    Find the WorldPop tile that contains the given center point.
    Uses spatial index for fast lookup.
    """
    # Get candidate tiles from nearby grid cells
    grid_lon = int(np.floor(center_lon / grid_size))
    grid_lat = int(np.floor(center_lat / grid_size))

    candidates = []
    for dlon in [-1, 0, 1]:
        for dlat in [-1, 0, 1]:
            key = (grid_lon + dlon, grid_lat + dlat)
            if key in spatial_index:
                candidates.extend(spatial_index[key])

    # Check which candidate contains the point
    for idx in candidates:
        row = meta_df.iloc[idx]
        if (row['lon_min'] <= center_lon <= row['lon_max'] and
            row['lat_min'] <= center_lat <= row['lat_max']):
            return idx

    return None


def main():
    print("=" * 70)
    print("CREATING MULTI-PRODUCT PAIRING (FAST VERSION)")
    print("=" * 70)

    # Load metadata
    print("\nLoading metadata...")
    wp_meta = pd.read_csv('data/tiles_2020/project/tiles_2020/metadata.csv')
    ghs_meta = pd.read_csv('data/GHS_POP/GHS_POP_2020_USA_patch256_s128/metadata.csv')
    hrsl_meta = pd.read_csv('data/HRSL/hrsl_patches_100m/metadata_hrsl_conus.csv')

    print(f"  WorldPop: {len(wp_meta):,} tiles")
    print(f"  GHS-POP:  {len(ghs_meta):,} tiles")
    print(f"  HRSL:     {len(hrsl_meta):,} tiles")

    # Build spatial index for WorldPop
    print("\nBuilding spatial index for WorldPop tiles...")
    wp_index, grid_size = build_spatial_index(wp_meta)
    print(f"  Created {len(wp_index)} grid cells")

    pairings = []

    # Pair GHS-POP tiles with WorldPop
    print("\n1. Pairing GHS-POP tiles with WorldPop...")
    ghs_matched = 0
    for idx, ghs_row in tqdm(ghs_meta.iterrows(), total=len(ghs_meta), desc="GHS-POP"):
        center_lon = (ghs_row['lon_min'] + ghs_row['lon_max']) / 2
        center_lat = (ghs_row['lat_min'] + ghs_row['lat_max']) / 2

        wp_idx = find_containing_tile(center_lon, center_lat, wp_meta, wp_index, grid_size)

        if wp_idx is not None:
            ghs_matched += 1
            pairings.append({
                'wp_tile_id': wp_meta.iloc[wp_idx]['tile_id'],
                'wp_features_file': wp_meta.iloc[wp_idx]['features_file'],
                'wp_target_file': wp_meta.iloc[wp_idx]['target_file'],
                'product': 'GHS-POP',
                'product_file': ghs_row['filename'],
                'product_index': ghs_row['index'],
            })

    print(f"  Matched: {ghs_matched:,} / {len(ghs_meta):,} ({100*ghs_matched/len(ghs_meta):.1f}%)")

    # Pair HRSL tiles with WorldPop
    print("\n2. Pairing HRSL tiles with WorldPop...")
    hrsl_matched = 0
    for idx, hrsl_row in tqdm(hrsl_meta.iterrows(), total=len(hrsl_meta), desc="HRSL"):
        center_lon = (hrsl_row['lon_min'] + hrsl_row['lon_max']) / 2
        center_lat = (hrsl_row['lat_min'] + hrsl_row['lat_max']) / 2

        wp_idx = find_containing_tile(center_lon, center_lat, wp_meta, wp_index, grid_size)

        if wp_idx is not None:
            hrsl_matched += 1
            pairings.append({
                'wp_tile_id': wp_meta.iloc[wp_idx]['tile_id'],
                'wp_features_file': wp_meta.iloc[wp_idx]['features_file'],
                'wp_target_file': wp_meta.iloc[wp_idx]['target_file'],
                'product': 'HRSL',
                'product_file': hrsl_row['filename'],
                'product_index': hrsl_row['index'],
            })

    print(f"  Matched: {hrsl_matched:,} / {len(hrsl_meta):,} ({100*hrsl_matched/len(hrsl_meta):.1f}%)")

    # Add WorldPop as its own product
    print("\n3. Adding WorldPop tiles as their own product...")
    for idx, wp_row in tqdm(wp_meta.iterrows(), total=len(wp_meta), desc="WorldPop"):
        pairings.append({
            'wp_tile_id': wp_row['tile_id'],
            'wp_features_file': wp_row['features_file'],
            'wp_target_file': wp_row['target_file'],
            'product': 'WorldPop',
            'product_file': wp_row['target_file'],
            'product_index': wp_row['tile_id'],
        })

    # Create DataFrame
    pairing_df = pd.DataFrame(pairings)

    print("\n" + "=" * 70)
    print("PAIRING RESULTS")
    print("=" * 70)
    print(f"\nTotal pairings: {len(pairing_df):,}")
    print(f"\nBy product:")
    print(pairing_df['product'].value_counts())

    print(f"\nSample pairings:")
    print(pairing_df.head(10))

    # Save pairing metadata
    output_dir = 'data/paired_dataset'
    os.makedirs(output_dir, exist_ok=True)
    pairing_file = f'{output_dir}/multi_product_pairing.csv'
    pairing_df.to_csv(pairing_file, index=False)

    print(f"\n✓ Pairing metadata saved to: {pairing_file}")
    print(f"  File size: {os.path.getsize(pairing_file) / 1024:.1f} KB")

    # Statistics
    print("\n" + "=" * 70)
    print("USAGE FOR TRAINING")
    print("=" * 70)
    print("""
The pairing file maps population product files to WorldPop feature files:

Structure:
- wp_tile_id: WorldPop tile identifier
- wp_features_file: VIIRS + WSF features (256x256x2)
- wp_target_file: WorldPop population target (for reference)
- product: Product name (WorldPop, GHS-POP, HRSL)
- product_file: Population target file for this product
- product_index: Index in product's own metadata

Training usage:
1. Load pairing CSV
2. For each batch, sample rows from different products
3. Load features from wp_features_file
4. Load target from product_file (based on product type)
5. Pass product name as conditioning to the model

This enables multi-product supervision as described in C3-LDM Section 2.3
    """)


if __name__ == "__main__":
    main()
