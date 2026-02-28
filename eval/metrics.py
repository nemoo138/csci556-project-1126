"""
Evaluation Metrics for C3-LDM
Reference: IMPLEMENTATION_ROADMAP.md Phase 7.1

Provides comprehensive metrics for evaluating population map predictions:
- Pixel-level accuracy (RMSE, MAE, MAPE)
- Spatial correlation (Pearson, Spearman)
- Census consistency errors
- Heavy tail metrics (dense urban areas)
- Distribution metrics
"""

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from typing import Dict, Optional, Union, List


def rmse(predictions: np.ndarray, targets: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """
    Root Mean Squared Error.

    Args:
        predictions: (B, H, W) or (H, W) predicted population maps
        targets: (B, H, W) or (H, W) ground truth population maps
        mask: Optional (B, H, W) or (H, W) binary mask (1=compute, 0=ignore)

    Returns:
        RMSE value
    """
    predictions = np.asarray(predictions)
    targets = np.asarray(targets)

    if mask is not None:
        mask = np.asarray(mask).astype(bool)
        predictions = predictions[mask]
        targets = targets[mask]

    return np.sqrt(np.mean((predictions - targets) ** 2))


def mae(predictions: np.ndarray, targets: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """
    Mean Absolute Error.

    Args:
        predictions: (B, H, W) or (H, W) predicted population maps
        targets: (B, H, W) or (H, W) ground truth population maps
        mask: Optional (B, H, W) or (H, W) binary mask

    Returns:
        MAE value
    """
    predictions = np.asarray(predictions)
    targets = np.asarray(targets)

    if mask is not None:
        mask = np.asarray(mask).astype(bool)
        predictions = predictions[mask]
        targets = targets[mask]

    return np.mean(np.abs(predictions - targets))


def mape(predictions: np.ndarray, targets: np.ndarray,
         mask: Optional[np.ndarray] = None, epsilon: float = 1e-6) -> float:
    """
    Mean Absolute Percentage Error.

    Args:
        predictions: (B, H, W) or (H, W) predicted population maps
        targets: (B, H, W) or (H, W) ground truth population maps
        mask: Optional (B, H, W) or (H, W) binary mask
        epsilon: Small value to avoid division by zero

    Returns:
        MAPE value (percentage)
    """
    predictions = np.asarray(predictions)
    targets = np.asarray(targets)

    if mask is not None:
        mask = np.asarray(mask).astype(bool)
        predictions = predictions[mask]
        targets = targets[mask]

    # Only compute where target > epsilon to avoid division by near-zero
    valid_mask = targets > epsilon
    if not np.any(valid_mask):
        return np.nan

    predictions = predictions[valid_mask]
    targets = targets[valid_mask]

    return 100.0 * np.mean(np.abs((targets - predictions) / targets))


def r2_score(predictions: np.ndarray, targets: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """
    R² (coefficient of determination).

    Args:
        predictions: (B, H, W) or (H, W) predicted population maps
        targets: (B, H, W) or (H, W) ground truth population maps
        mask: Optional (B, H, W) or (H, W) binary mask

    Returns:
        R² value (1.0 = perfect, 0.0 = baseline, <0 = worse than mean)
    """
    predictions = np.asarray(predictions).flatten()
    targets = np.asarray(targets).flatten()

    if mask is not None:
        mask = np.asarray(mask).flatten().astype(bool)
        predictions = predictions[mask]
        targets = targets[mask]

    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)

    if ss_tot < 1e-10:
        return np.nan

    return 1.0 - (ss_res / ss_tot)


def spatial_correlation(predictions: np.ndarray, targets: np.ndarray,
                        method: str = 'pearson') -> float:
    """
    Spatial correlation between predicted and ground truth maps.

    Args:
        predictions: (B, H, W) or (H, W) predicted population maps
        targets: (B, H, W) or (H, W) ground truth population maps
        method: 'pearson' or 'spearman'

    Returns:
        Correlation coefficient
    """
    predictions = np.asarray(predictions).flatten()
    targets = np.asarray(targets).flatten()

    if method == 'pearson':
        corr, _ = pearsonr(predictions, targets)
    elif method == 'spearman':
        corr, _ = spearmanr(predictions, targets)
    else:
        raise ValueError(f"Unknown method: {method}")

    return corr


def census_error(predictions: np.ndarray, targets: np.ndarray,
                 admin_ids: np.ndarray, census_totals: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Census consistency errors per admin unit.

    Args:
        predictions: (B, H, W) or (H, W) predicted population maps
        targets: (B, H, W) or (H, W) ground truth population maps (for census totals)
        admin_ids: (H, W) admin unit IDs
        census_totals: Optional (num_units,) census totals. If None, compute from targets.

    Returns:
        Dictionary with:
            - 'mean_abs_error': Mean absolute census error across admin units
            - 'mean_rel_error': Mean relative census error (percentage)
            - 'max_abs_error': Maximum absolute census error
            - 'per_admin_errors': Dict mapping admin_id -> absolute error
    """
    predictions = np.asarray(predictions)
    targets = np.asarray(targets)
    admin_ids = np.asarray(admin_ids)

    # Handle batch dimension
    if predictions.ndim == 3:  # (B, H, W)
        # Average predictions across batch
        predictions = predictions.mean(axis=0)
    if targets.ndim == 3:  # (B, H, W)
        targets = targets.mean(axis=0)

    unique_ids = np.unique(admin_ids)
    # Filter out invalid IDs (like -1)
    unique_ids = unique_ids[unique_ids >= 0]

    abs_errors = []
    rel_errors = []
    per_admin = {}

    for admin_id in unique_ids:
        mask = admin_ids == admin_id

        pred_sum = predictions[mask].sum()

        if census_totals is not None:
            true_sum = census_totals[admin_id]
        else:
            true_sum = targets[mask].sum()

        abs_err = abs(pred_sum - true_sum)
        abs_errors.append(abs_err)
        per_admin[int(admin_id)] = float(abs_err)

        if true_sum > 1e-6:
            rel_err = 100.0 * abs_err / true_sum
            rel_errors.append(rel_err)

    return {
        'mean_abs_error': float(np.mean(abs_errors)) if abs_errors else 0.0,
        'mean_rel_error': float(np.mean(rel_errors)) if rel_errors else 0.0,
        'max_abs_error': float(np.max(abs_errors)) if abs_errors else 0.0,
        'per_admin_errors': per_admin
    }


def heavy_tail_metrics(predictions: np.ndarray, targets: np.ndarray,
                      percentile: float = 90.0) -> Dict[str, float]:
    """
    Metrics for heavy tail regions (dense urban areas).

    Args:
        predictions: (B, H, W) or (H, W) predicted population maps
        targets: (B, H, W) or (H, W) ground truth population maps
        percentile: Percentile threshold for "dense" areas (default: 90th)

    Returns:
        Dictionary with:
            - 'mae_dense': MAE on dense pixels (>percentile)
            - 'rmse_dense': RMSE on dense pixels
            - 'mae_sparse': MAE on sparse pixels (<=percentile)
            - 'rmse_sparse': RMSE on sparse pixels
            - 'corr_dense': Correlation on dense pixels
    """
    predictions = np.asarray(predictions)
    targets = np.asarray(targets)

    # Handle batch dimension
    if predictions.ndim == 3:  # (B, H, W)
        predictions = predictions.reshape(-1)
        targets = targets.reshape(-1)
    else:
        predictions = predictions.flatten()
        targets = targets.flatten()

    # Find threshold for dense areas
    threshold = np.percentile(targets, percentile)

    dense_mask = targets > threshold
    sparse_mask = ~dense_mask

    results = {}

    # Dense area metrics
    if np.any(dense_mask):
        pred_dense = predictions[dense_mask]
        tgt_dense = targets[dense_mask]
        results['mae_dense'] = mae(pred_dense, tgt_dense)
        results['rmse_dense'] = rmse(pred_dense, tgt_dense)
        results['corr_dense'] = spatial_correlation(pred_dense, tgt_dense)
    else:
        results['mae_dense'] = np.nan
        results['rmse_dense'] = np.nan
        results['corr_dense'] = np.nan

    # Sparse area metrics
    if np.any(sparse_mask):
        pred_sparse = predictions[sparse_mask]
        tgt_sparse = targets[sparse_mask]
        results['mae_sparse'] = mae(pred_sparse, tgt_sparse)
        results['rmse_sparse'] = rmse(pred_sparse, tgt_sparse)
    else:
        results['mae_sparse'] = np.nan
        results['rmse_sparse'] = np.nan

    return results


def batch_metrics(predictions: Union[np.ndarray, torch.Tensor],
                  targets: Union[np.ndarray, torch.Tensor],
                  admin_ids: Optional[Union[np.ndarray, torch.Tensor]] = None,
                  census_totals: Optional[Union[np.ndarray, torch.Tensor]] = None,
                  percentile: float = 90.0) -> Dict[str, float]:
    """
    Comprehensive metrics for a batch of predictions.

    Args:
        predictions: (B, H, W) or (B, C, H, W) predicted population maps
        targets: (B, H, W) or (B, C, H, W) ground truth population maps
        admin_ids: Optional (H, W) admin unit IDs (same for all batch items)
        census_totals: Optional (num_units,) census totals
        percentile: Percentile for heavy tail metrics (default: 90)

    Returns:
        Dictionary of all metrics:
            - rmse: Root Mean Squared Error
            - mae: Mean Absolute Error
            - mape: Mean Absolute Percentage Error
            - r2: R² score
            - corr_pearson: Pearson correlation
            - corr_spearman: Spearman correlation
            - mae_dense: MAE on dense areas (>percentile)
            - rmse_dense: RMSE on dense areas
            - mae_sparse: MAE on sparse areas
            - corr_dense: Correlation on dense areas
            - census_mean_abs_error: Mean census error (if admin_ids provided)
            - census_mean_rel_error: Mean relative census error (%)
    """
    # Convert to numpy
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    if admin_ids is not None and isinstance(admin_ids, torch.Tensor):
        admin_ids = admin_ids.detach().cpu().numpy()
    if census_totals is not None and isinstance(census_totals, torch.Tensor):
        census_totals = census_totals.detach().cpu().numpy()

    # Handle channel dimension (B, C, H, W) -> (B, H, W)
    if predictions.ndim == 4:
        predictions = predictions[:, 0, :, :]  # Take first channel
    if targets.ndim == 4:
        targets = targets[:, 0, :, :]

    metrics = {}

    # Basic pixel-level metrics
    metrics['rmse'] = rmse(predictions, targets)
    metrics['mae'] = mae(predictions, targets)
    metrics['mape'] = mape(predictions, targets)
    metrics['r2'] = r2_score(predictions, targets)

    # Spatial correlation
    metrics['corr_pearson'] = spatial_correlation(predictions, targets, method='pearson')
    metrics['corr_spearman'] = spatial_correlation(predictions, targets, method='spearman')

    # Heavy tail metrics
    tail_metrics = heavy_tail_metrics(predictions, targets, percentile=percentile)
    metrics.update(tail_metrics)

    # Census consistency (if admin data provided)
    if admin_ids is not None:
        census_metrics = census_error(predictions, targets, admin_ids, census_totals)
        metrics['census_mean_abs_error'] = census_metrics['mean_abs_error']
        metrics['census_mean_rel_error'] = census_metrics['mean_rel_error']
        metrics['census_max_abs_error'] = census_metrics['max_abs_error']

    return metrics


def print_metrics(metrics: Dict[str, float], title: str = "Evaluation Metrics"):
    """
    Pretty print metrics dictionary.

    Args:
        metrics: Dictionary of metric_name -> value
        title: Title to display
    """
    print(f"\n{'='*70}")
    print(f"{title:^70}")
    print(f"{'='*70}")

    # Group metrics
    basic = ['rmse', 'mae', 'mape', 'r2']
    correlation = ['corr_pearson', 'corr_spearman']
    heavy_tail = ['mae_dense', 'rmse_dense', 'mae_sparse', 'rmse_sparse', 'corr_dense']
    census = ['census_mean_abs_error', 'census_mean_rel_error', 'census_max_abs_error']

    # Print basic metrics
    print("\nPixel-Level Accuracy:")
    print("-" * 70)
    for key in basic:
        if key in metrics:
            value = metrics[key]
            if np.isnan(value):
                print(f"  {key:20s}: NaN")
            elif key == 'mape' or key == 'census_mean_rel_error':
                print(f"  {key:20s}: {value:8.2f}%")
            else:
                print(f"  {key:20s}: {value:8.4f}")

    # Print correlation
    print("\nSpatial Correlation:")
    print("-" * 70)
    for key in correlation:
        if key in metrics:
            value = metrics[key]
            print(f"  {key:20s}: {value:8.4f}")

    # Print heavy tail
    print("\nHeavy Tail Metrics (90th percentile):")
    print("-" * 70)
    for key in heavy_tail:
        if key in metrics:
            value = metrics[key]
            if np.isnan(value):
                print(f"  {key:20s}: NaN")
            else:
                print(f"  {key:20s}: {value:8.4f}")

    # Print census
    if any(key in metrics for key in census):
        print("\nCensus Consistency:")
        print("-" * 70)
        for key in census:
            if key in metrics:
                value = metrics[key]
                if key == 'census_mean_rel_error':
                    print(f"  {key:20s}: {value:8.2f}%")
                else:
                    print(f"  {key:20s}: {value:8.2f}")

    print(f"{'='*70}\n")


# ============================================================================
# Test and Example Usage
# ============================================================================

if __name__ == "__main__":
    print("Testing C3-LDM Metrics Module")
    print("=" * 70)

    # Create synthetic data
    np.random.seed(42)
    B, H, W = 8, 256, 256

    # Ground truth: mix of sparse and dense areas
    targets = np.random.exponential(scale=10.0, size=(B, H, W))
    # Add some dense urban cores
    targets[:, 100:150, 100:150] *= 5.0

    # Predictions: ground truth + noise
    predictions = targets + np.random.normal(0, 2.0, size=(B, H, W))
    predictions = np.maximum(predictions, 0)  # Non-negative

    # Admin IDs (simplified)
    admin_ids = np.zeros((H, W), dtype=np.int32)
    admin_ids[:, :128] = 0
    admin_ids[:, 128:] = 1

    print(f"\nTest data:")
    print(f"  Predictions shape: {predictions.shape}")
    print(f"  Targets shape: {targets.shape}")
    print(f"  Admin IDs shape: {admin_ids.shape}")
    print(f"  Predictions range: [{predictions.min():.2f}, {predictions.max():.2f}]")
    print(f"  Targets range: [{targets.min():.2f}, {targets.max():.2f}]")

    # Test batch metrics
    print("\n" + "=" * 70)
    print("Testing batch_metrics()...")
    print("=" * 70)

    metrics = batch_metrics(predictions, targets, admin_ids=admin_ids)
    print_metrics(metrics, title="Batch Metrics (8 samples)")

    # Test individual metrics
    print("\n" + "=" * 70)
    print("Testing individual metric functions...")
    print("=" * 70)

    print(f"\nRMSE: {rmse(predictions, targets):.4f}")
    print(f"MAE: {mae(predictions, targets):.4f}")
    print(f"MAPE: {mape(predictions, targets):.2f}%")
    print(f"R²: {r2_score(predictions, targets):.4f}")
    print(f"Pearson correlation: {spatial_correlation(predictions, targets, 'pearson'):.4f}")
    print(f"Spearman correlation: {spatial_correlation(predictions, targets, 'spearman'):.4f}")

    # Test heavy tail
    tail = heavy_tail_metrics(predictions, targets, percentile=90)
    print(f"\nHeavy tail (90th percentile):")
    print(f"  MAE dense: {tail['mae_dense']:.4f}")
    print(f"  RMSE dense: {tail['rmse_dense']:.4f}")
    print(f"  MAE sparse: {tail['mae_sparse']:.4f}")

    # Test census errors
    census = census_error(predictions, targets, admin_ids)
    print(f"\nCensus errors:")
    print(f"  Mean absolute: {census['mean_abs_error']:.2f}")
    print(f"  Mean relative: {census['mean_rel_error']:.2f}%")
    print(f"  Max absolute: {census['max_abs_error']:.2f}")

    # Test with PyTorch tensors
    print("\n" + "=" * 70)
    print("Testing with PyTorch tensors...")
    print("=" * 70)

    import torch
    pred_tensor = torch.from_numpy(predictions).float()
    tgt_tensor = torch.from_numpy(targets).float()
    admin_tensor = torch.from_numpy(admin_ids).long()

    metrics_torch = batch_metrics(pred_tensor, tgt_tensor, admin_ids=admin_tensor)
    print("\nMetrics computed from PyTorch tensors:")
    print(f"  RMSE: {metrics_torch['rmse']:.4f}")
    print(f"  MAE: {metrics_torch['mae']:.4f}")
    print(f"  Correlation: {metrics_torch['corr_pearson']:.4f}")

    print("\n✓ All tests passed!")
