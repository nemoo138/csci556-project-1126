# Evaluation Module

Comprehensive metrics and evaluation tools for C3-LDM population mapping models.

## Quick Start

### 1. Compute Metrics for a Batch

```python
from eval import batch_metrics, print_metrics
import numpy as np

# Your predictions and targets
predictions = np.random.rand(8, 256, 256)  # (batch, height, width)
targets = np.random.rand(8, 256, 256)

# Compute all metrics
metrics = batch_metrics(predictions, targets)
print_metrics(metrics)
```

**Output:**
```
======================================================================
                      Evaluation Metrics
======================================================================

Pixel-Level Accuracy:
----------------------------------------------------------------------
  rmse                :   0.2887
  mae                 :   0.2301
  mape                :   45.23%
  r2                  :   0.8234

Spatial Correlation:
----------------------------------------------------------------------
  corr_pearson        :   0.9123
  corr_spearman       :   0.8956

Heavy Tail Metrics (90th percentile):
----------------------------------------------------------------------
  mae_dense           :   0.2456
  rmse_dense          :   0.3012
  ...
```

### 2. Evaluate a Trained Model

```bash
# Evaluate on WorldPop test set
python eval/evaluate.py \
    --checkpoint checkpoints/checkpoint_best.pt \
    --pairing_csv data/paired_dataset/multi_product_pairing.csv \
    --products WorldPop \
    --sampler ddim \
    --num_steps 50 \
    --max_samples 100 \
    --output_dir eval_results
```

**Output:**
- `eval_results/metrics.json` - All metrics in JSON format
- `eval_results/summary.txt` - Human-readable summary
- `eval_results/predictions/` - Individual predictions (if `--save_predictions`)

---

## Available Metrics

### Pixel-Level Accuracy

| Metric | Function | Description |
|--------|----------|-------------|
| **RMSE** | `rmse()` | Root Mean Squared Error |
| **MAE** | `mae()` | Mean Absolute Error |
| **MAPE** | `mape()` | Mean Absolute Percentage Error |
| **R²** | `r2_score()` | Coefficient of determination (1.0 = perfect) |

### Spatial Correlation

| Metric | Function | Description |
|--------|----------|-------------|
| **Pearson** | `spatial_correlation(..., method='pearson')` | Linear correlation |
| **Spearman** | `spatial_correlation(..., method='spearman')` | Rank correlation |

### Heavy Tail (Dense Urban Areas)

Computed on pixels above the 90th percentile (customizable):

| Metric | Description |
|--------|-------------|
| `mae_dense` | MAE on dense areas (>90th percentile) |
| `rmse_dense` | RMSE on dense areas |
| `mae_sparse` | MAE on sparse areas (≤90th percentile) |
| `rmse_sparse` | RMSE on sparse areas |
| `corr_dense` | Correlation on dense areas |

### Census Consistency

Requires admin unit IDs and census totals:

| Metric | Description |
|--------|-------------|
| `census_mean_abs_error` | Mean absolute census error across admin units |
| `census_mean_rel_error` | Mean relative census error (percentage) |
| `census_max_abs_error` | Maximum absolute census error |

---

## API Reference

### `batch_metrics()`

Compute all metrics for a batch of predictions.

```python
from eval import batch_metrics

metrics = batch_metrics(
    predictions,      # (B, H, W) or (B, C, H, W)
    targets,          # (B, H, W) or (B, C, H, W)
    admin_ids=None,   # Optional (H, W) admin unit IDs
    census_totals=None,  # Optional (num_units,) census totals
    percentile=90.0   # Percentile for heavy tail metrics
)
```

**Returns:** Dictionary with all metrics

**Supports:**
- NumPy arrays
- PyTorch tensors (automatically converted)
- Batch or single samples
- 3D (B, H, W) or 4D (B, C, H, W) inputs

### Individual Metric Functions

```python
from eval import rmse, mae, spatial_correlation, census_error

# RMSE
error = rmse(predictions, targets, mask=None)

# MAE with mask
error = mae(predictions, targets, mask=valid_pixels)

# Correlation
corr = spatial_correlation(predictions, targets, method='pearson')

# Census errors
census = census_error(predictions, targets, admin_ids, census_totals)
print(census['mean_abs_error'])
print(census['per_admin_errors'])  # Dict: admin_id -> error
```

### `print_metrics()`

Pretty-print metrics dictionary.

```python
from eval import print_metrics

print_metrics(metrics, title="My Evaluation Results")
```

---

## Evaluation Script

### Basic Usage

```bash
# Evaluate checkpoint on 100 samples
python eval/evaluate.py \
    --checkpoint checkpoints/checkpoint_best.pt \
    --max_samples 100 \
    --output_dir eval_results
```

### Advanced Options

```bash
python eval/evaluate.py \
    --checkpoint checkpoints/checkpoint_best.pt \
    --pairing_csv data/paired_dataset/multi_product_pairing.csv \
    --products WorldPop GHS-POP HRSL \  # Evaluate on multiple products
    --num_samples 5 \                    # Generate 5 samples per input (ensemble)
    --sampler ddim \                     # Use DDIM (fast)
    --num_steps 50 \                     # 50 DDIM steps
    --max_samples 500 \                  # Evaluate 500 samples
    --output_dir eval_results \
    --save_predictions                   # Save individual predictions
```

### Output Files

After running evaluation:

```
eval_results/
├── metrics.json          # All metrics in JSON format
├── summary.txt          # Human-readable summary
└── predictions/         # Individual predictions (if --save_predictions)
    ├── pred_000000.npy
    ├── target_000000.npy
    ├── pred_000001.npy
    └── ...
```

**metrics.json:**
```json
{
  "rmse": 1.9084,
  "mae": 1.5089,
  "mape": 116.67,
  "r2": 0.9855,
  "corr_pearson": 0.9928,
  "corr_spearman": 0.9512,
  "mae_dense": 1.6033,
  "rmse_dense": 2.0098,
  ...
}
```

---

## Integration with Training

### Option 1: Use built-in quick_inference

Training already has evaluation every N epochs:

```bash
python train.py \
    --eval_every 5 \           # Evaluate every 5 epochs
    --eval_output_dir eval_wp  # Save to eval_wp/
```

### Option 2: Import metrics in training code

```python
from eval import batch_metrics, print_metrics

# During training
with torch.no_grad():
    predictions = model.generate(...)
    metrics = batch_metrics(predictions, targets)
    print_metrics(metrics)
```

---

## Example: Complete Evaluation Pipeline

```python
import numpy as np
from eval import batch_metrics, print_metrics

# 1. Load your predictions and targets
predictions = np.load('predictions.npy')  # (N, 256, 256)
targets = np.load('targets.npy')          # (N, 256, 256)

# 2. (Optional) Load census data
admin_ids = np.load('admin_ids.npy')     # (256, 256)
census_totals = np.load('census.npy')    # (num_units,)

# 3. Compute all metrics
metrics = batch_metrics(
    predictions,
    targets,
    admin_ids=admin_ids,
    census_totals=census_totals,
    percentile=90  # Dense areas = top 10%
)

# 4. Display results
print_metrics(metrics, title="My Model Evaluation")

# 5. Access individual metrics
print(f"RMSE: {metrics['rmse']:.4f}")
print(f"Correlation: {metrics['corr_pearson']:.4f}")
print(f"Census error: {metrics['census_mean_abs_error']:.2f}")
```

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'scipy'"

Install scipy:
```bash
pip install scipy
```

### NaN values in metrics

Some metrics may be NaN if:
- No data points satisfy the condition (e.g., no pixels >90th percentile)
- Division by zero (use epsilon parameter)
- All targets are zero (for relative errors)

This is expected behavior and metrics will show as "NaN" in output.

### Memory issues with large batches

For very large evaluations, process in chunks:

```python
chunk_size = 32
all_metrics = []

for i in range(0, len(predictions), chunk_size):
    chunk_pred = predictions[i:i+chunk_size]
    chunk_tgt = targets[i:i+chunk_size]

    metrics = batch_metrics(chunk_pred, chunk_tgt)
    all_metrics.append(metrics)

# Average metrics across chunks
final_metrics = {
    key: np.mean([m[key] for m in all_metrics])
    for key in all_metrics[0].keys()
}
```

---

## Performance

- **RMSE/MAE**: O(N) - very fast
- **Correlation**: O(N log N) - fast
- **Census errors**: O(N × num_admin_units) - fast for reasonable number of units
- **Heavy tail**: O(N) - fast

Typical performance on 1000 samples (256×256):
- **Batch metrics**: ~0.5 seconds
- **Individual metrics**: ~0.01 seconds each

---

## Citation

If you use these metrics in your research, please cite:

```bibtex
@article{c3ldm2024,
  title={Census-Consistent Conditional Latent Diffusion Models for Population Mapping},
  author={...},
  year={2024}
}
```
