"""
Create stratified train/test split for C3-LDM evaluation

Splits dataset 80/20 with:
- Stratification by population density (zero, low, medium, high)
- Random spatial distribution (not geographic holdout)
- Reproducible with fixed random seed
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import argparse


def compute_population_sum(data_root, row):
    """
    Compute total population for a tile.

    Args:
        data_root: Root directory for data
        row: DataFrame row with product and file info

    Returns:
        population_sum: Total population in the tile
    """
    product = row['product']
    product_file = row['product_file']

    try:
        if product == 'WorldPop':
            target_path = os.path.join(
                data_root,
                'tiles_2020/project/tiles_2020',
                product_file
            )
        elif product == 'GHS-POP':
            target_path = os.path.join(
                data_root,
                'GHS_POP/GHS_POP_2020_USA_patch256_s128',
                product_file
            )
        elif product == 'HRSL':
            target_path = os.path.join(
                data_root,
                'HRSL/hrsl_patches_100m/patches_conus',
                product_file
            )
        else:
            return 0.0

        if os.path.exists(target_path):
            target = np.load(target_path)
            # Replace NaN with 0
            target = np.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)
            return float(target.sum())
        else:
            print(f"Warning: File not found: {target_path}")
            return 0.0

    except Exception as e:
        print(f"Error loading {product_file}: {e}")
        return 0.0


def create_split(pairing_csv, data_root, output_dir, test_size=0.2,
                 products=None, random_state=42, min_pop_threshold=1.0):
    """
    Create stratified train/test split.

    Args:
        pairing_csv: Path to multi_product_pairing.csv
        data_root: Root directory for data
        output_dir: Directory to save split CSVs
        test_size: Fraction for test set (default: 0.2 = 20%)
        products: List of products to include (default: all)
        random_state: Random seed for reproducibility
        min_pop_threshold: Minimum population to not be considered "zero"
    """
    print("=" * 70)
    print("Creating Stratified Train/Test Split")
    print("=" * 70)

    # Load pairing data
    print(f"\nLoading pairing data from: {pairing_csv}")
    df = pd.read_csv(pairing_csv)
    print(f"Total samples: {len(df)}")

    # Filter by products
    if products is not None:
        df = df[df['product'].isin(products)]
        print(f"Filtered to products {products}: {len(df)} samples")

    print(f"\nProduct distribution:")
    print(df['product'].value_counts())

    # Compute population sums
    print(f"\nComputing population sums for all tiles...")
    pop_sums = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Loading tiles"):
        pop_sum = compute_population_sum(data_root, row)
        pop_sums.append(pop_sum)

    df['pop_sum'] = pop_sums

    # Statistics
    print(f"\nPopulation statistics:")
    print(f"  Total samples: {len(df)}")
    print(f"  Zero tiles (sum < {min_pop_threshold}): {(df['pop_sum'] < min_pop_threshold).sum()}")
    print(f"  Non-zero tiles: {(df['pop_sum'] >= min_pop_threshold).sum()}")
    print(f"  Min population: {df['pop_sum'].min():.2f}")
    print(f"  Max population: {df['pop_sum'].max():.2f}")
    print(f"  Mean population: {df['pop_sum'].mean():.2f}")
    print(f"  Median population: {df['pop_sum'].median():.2f}")

    # Create population density bins
    print(f"\nCreating population density bins...")
    bins = [0, min_pop_threshold, 100, 1000, np.inf]
    labels = ['zero', 'low', 'medium', 'high']
    df['pop_bin'] = pd.cut(df['pop_sum'], bins=bins, labels=labels, include_lowest=True)

    print(f"\nPopulation bin distribution:")
    print(df['pop_bin'].value_counts().sort_index())

    # Stratified split
    print(f"\nPerforming stratified train/test split ({(1-test_size)*100:.0f}/{test_size*100:.0f})...")

    try:
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            stratify=df['pop_bin'],
            random_state=random_state
        )
    except ValueError as e:
        print(f"Warning: Stratification failed ({e})")
        print("Falling back to random split...")
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state
        )

    print(f"\nTrain set: {len(train_df)} samples")
    print("  Population bins:")
    print(train_df['pop_bin'].value_counts().sort_index())
    print(f"  Mean population: {train_df['pop_sum'].mean():.2f}")

    print(f"\nTest set: {len(test_df)} samples")
    print("  Population bins:")
    print(test_df['pop_bin'].value_counts().sort_index())
    print(f"  Mean population: {test_df['pop_sum'].mean():.2f}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Save splits (without pop_sum and pop_bin columns for compatibility)
    train_output = os.path.join(output_dir, 'train_split.csv')
    test_output = os.path.join(output_dir, 'test_split.csv')

    # Keep original columns plus metadata
    train_df_save = train_df.copy()
    test_df_save = test_df.copy()

    train_df_save.to_csv(train_output, index=False)
    test_df_save.to_csv(test_output, index=False)

    print(f"\nSaved train split to: {train_output}")
    print(f"Saved test split to: {test_output}")

    # Save statistics
    stats_output = os.path.join(output_dir, 'split_statistics.txt')
    with open(stats_output, 'w') as f:
        f.write("Train/Test Split Statistics\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Random seed: {random_state}\n")
        f.write(f"Test size: {test_size*100:.0f}%\n")
        f.write(f"Min population threshold: {min_pop_threshold}\n\n")

        f.write(f"Total samples: {len(df)}\n")
        f.write(f"Train samples: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)\n")
        f.write(f"Test samples: {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)\n\n")

        f.write("Population bins:\n")
        f.write(f"  Zero: {bins[0]}-{bins[1]}\n")
        f.write(f"  Low: {bins[1]}-{bins[2]}\n")
        f.write(f"  Medium: {bins[2]}-{bins[3]}\n")
        f.write(f"  High: >{bins[3]}\n\n")

        f.write("Train set distribution:\n")
        f.write(str(train_df['pop_bin'].value_counts().sort_index()) + "\n\n")

        f.write("Test set distribution:\n")
        f.write(str(test_df['pop_bin'].value_counts().sort_index()) + "\n\n")

        f.write(f"Train mean population: {train_df['pop_sum'].mean():.2f}\n")
        f.write(f"Test mean population: {test_df['pop_sum'].mean():.2f}\n")

    print(f"Saved statistics to: {stats_output}")

    # Create a filtered test set (only non-zero tiles)
    test_nonzero_df = test_df[test_df['pop_sum'] >= min_pop_threshold].copy()
    if len(test_nonzero_df) > 0:
        test_nonzero_output = os.path.join(output_dir, 'test_split_nonzero.csv')
        test_nonzero_df.to_csv(test_nonzero_output, index=False)
        print(f"\nSaved non-zero test split to: {test_nonzero_output}")
        print(f"  Contains {len(test_nonzero_df)} tiles with population >= {min_pop_threshold}")

    print("\n" + "=" * 70)
    print("Split creation completed!")
    print("=" * 70)

    return train_df, test_df


def main():
    parser = argparse.ArgumentParser(description='Create stratified train/test split')

    parser.add_argument('--pairing_csv', type=str,
                       default='data/paired_dataset/multi_product_pairing.csv',
                       help='Path to multi_product_pairing.csv')
    parser.add_argument('--data_root', type=str, default='data',
                       help='Root directory for data')
    parser.add_argument('--output_dir', type=str, default='data/paired_dataset',
                       help='Directory to save split CSVs')
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Fraction for test set (default: 0.2)')
    parser.add_argument('--products', type=str, nargs='+', default=None,
                       help='Products to include (default: all)')
    parser.add_argument('--random_state', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--min_pop_threshold', type=float, default=1.0,
                       help='Minimum population to not be "zero" (default: 1.0)')

    args = parser.parse_args()

    create_split(
        pairing_csv=args.pairing_csv,
        data_root=args.data_root,
        output_dir=args.output_dir,
        test_size=args.test_size,
        products=args.products,
        random_state=args.random_state,
        min_pop_threshold=args.min_pop_threshold
    )


if __name__ == "__main__":
    main()
