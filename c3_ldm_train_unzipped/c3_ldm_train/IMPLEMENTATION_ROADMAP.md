# C3-LDM Implementation Roadmap

Based on the [C3-LDM.md](C3-LDM.md) specification, here's a structured implementation plan for the Census-Consistent, Conditional Latent Diffusion model for Population Mapping.

## Data Preparation ✅ COMPLETE

- [x] WorldPop dataset (24,020 tiles with VIIRS + WSF features)
- [x] GHS-POP dataset (16,479 tiles)
- [x] HRSL dataset (3,289 tiles)
- [x] Multi-product spatial pairing (41,896 samples)
- [x] Pairing metadata: `data/paired_dataset/multi_product_pairing.csv`

## Implementation Phases

### Phase 1: Core Components (Foundation)

#### 1.1 Baseline Dasymetric Module
**Location**: `models/baseline.py`
**Reference**: C3-LDM.md Section "Component 1"

```python
class BaselineDasymetric:
    """
    Simple lights × settlement baseline allocation.
    Used to compute residual targets for the diffusion model.
    """
    def forward(self, lights, settlement, admin_ids, census_totals):
        # score[i] = (L[i] + λ_L) * (S[i] + λ_S)
        # Normalize per admin unit to match census totals
        pass
```

**Hyperparameters**:
- `λ_L`: small positive (e.g., 0.01) for dark settled areas
- `λ_S`: small positive (e.g., 0.01) for lights-only areas

---

#### 1.2 VAE for Residual Maps
**Location**: `models/vae.py`
**Reference**: C3-LDM.md Section "Component 2"

**Encoder**: 256×256 → 32×32 latent (4 channels)
```
Input: R_true (B, 1, 256, 256)
↓ Conv(1→64, k3, s1) + GroupNorm + SiLU
↓ Conv(64→64, k3, s2)          # 256→128
↓ Conv(64→128, k3, s1) + GN + SiLU
↓ Conv(128→128, k3, s2)        # 128→64
↓ Conv(128→256, k3, s1) + GN + SiLU
↓ Conv(256→256, k3, s2)        # 64→32
↓ Conv(256→8, k3, s1)          # → μ and logσ
Output: z_0 (B, 4, 32, 32)
```

**Decoder**: Mirror architecture with upsampling
```
Input: z_0 (B, 4, 32, 32)
↓ Conv + Upsample layers
Output: R_hat (B, 1, 256, 256)
```

**Loss**: KL divergence + reconstruction in residual space

---

#### 1.3 Time Embedding
**Location**: `models/time_embedding.py`
**Reference**: C3-LDM.md Section 4.1

```python
class TimeEmbedding:
    """Sinusoidal positional encoding + MLP"""
    def forward(self, t):
        # t → sinusoidal (dim=64)
        # → Linear(64→256) + SiLU
        # → Linear(256→256)
        pass
```

---

### Phase 2: Conditional Architecture

#### 2.1 Dual-Branch Conditional Encoder
**Location**: `models/conditional_encoder.py`
**Reference**: C3-LDM.md Section "Component 3"

**Low-res branch** (VIIRS nightlights):
```
Input: L (B, 1, 256, 256)
↓ Conv(1→32, k5, s2) + GN + SiLU   # 256→128
↓ Conv(32→64, k5, s2) + GN + SiLU  # 128→64
↓ Conv(64→128, k3, s2) + GN + SiLU # 64→32
Output: H_low (B, 128, 32, 32)
```

**High-res branch** (WSF settlements):
```
Input: S (B, 1, 256, 256)
↓ Conv(1→64, k3, s1) + GN + SiLU
↓ Conv(64→64, k3, s2) + GN + SiLU  # 256→128
↓ Conv(64→128, k3, s1) + GN + SiLU
↓ Conv(128→128, k3, s2) + GN + SiLU # 128→64
↓ Conv(128→128, k3, s2) + GN + SiLU # 64→32
Output: H_high (B, 128, 32, 32)
```

**Fusion**:
```python
H_concat = cat(H_low, H_high, dim=1)  # (B, 256, 32, 32)
H_cond = Conv1x1(256→C_cond) + GN + SiLU
```

---

#### 2.2 Product ID Embedding
**Location**: `models/product_embedding.py`
**Reference**: C3-LDM.md Section 3.3

```python
class ProductEmbedding:
    """
    Learnable embeddings for WorldPop, GHS-POP, HRSL
    """
    def __init__(self, num_products=3, d_prod=64):
        self.embeddings = nn.Embedding(num_products, d_prod)
        self.proj = nn.Linear(d_prod, C_cond)

    def forward(self, product_ids):
        # product_ids: (B,) with values 0, 1, 2
        # → (B, C_cond, 1, 1) for broadcast add to H_cond
        pass
```

Mapping:
- 0: WorldPop
- 1: GHS-POP
- 2: HRSL

---

### Phase 3: Diffusion U-Net

#### 3.1 U-Net with FiLM Conditioning
**Location**: `models/unet.py`
**Reference**: C3-LDM.md Section "Component 4"

**Architecture**:
- Input: z_t (B, 4, 32, 32)
- Resolutions: 32 → 16 → 8 → 16 → 32
- FiLM conditioning from t_emb and H_cond at each ResBlock
- Self-attention at 32×32 and 8×8
- Output: predicted noise ε_pred (B, 4, 32, 32)

**ResBlock with FiLM**:
```python
class FiLMResBlock:
    def forward(self, x, t_emb, H_cond):
        # γ_t, β_t from t_emb
        # γ_c, β_c from H_cond
        # x = x * (1 + γ_t + γ_c) + (β_t + β_c)
        pass
```

---

### Phase 4: Census-Consistency Layer

#### 4.1 Differentiable Normalization
**Location**: `models/census_layer.py`
**Reference**: C3-LDM.md Section "Component 5"

```python
class CensusConsistencyLayer:
    """
    Enforces Σ(predictions in admin unit) = census total
    Differentiable for backprop
    """
    def forward(self, P_raw, admin_ids, census_totals):
        # For each admin A:
        #   S_A = Σ_{j ∈ A} P_raw[j] + ε
        #   P[i] = P_raw[i] / S_A * C_A  for i ∈ A
        pass
```

**Input**: Raw population predictions (unconstrained)
**Output**: Census-consistent predictions
**Constraint**: Hard constraint, exact match by construction

---

### Phase 5: Training Pipeline

#### 5.1 Multi-Product Dataloader
**Location**: `data/dataset.py`

```python
class MultiProductDataset:
    def __init__(self, pairing_csv):
        self.pairing = pd.read_csv(pairing_csv)

    def __getitem__(self, idx):
        row = self.pairing.iloc[idx]

        # Load features (VIIRS + WSF)
        features = np.load(features_path)

        # Load target based on product
        target = load_product_target(row['product'], row['product_file'])

        # Admin IDs and census totals (if available)
        admin_ids, census = load_census_data(row)

        return {
            'features': features,
            'target': target,
            'product': row['product'],
            'admin_ids': admin_ids,
            'census': census
        }
```

---

#### 5.2 Training Loop
**Location**: `train.py`
**Reference**: C3-LDM.md Section "Component 6"

```python
def training_step(batch):
    L, S, Y_true, product_id, admin_ids, census = batch

    # 1. Baseline
    B = baseline_dasymetric(L, S, admin_ids, census)

    # 2. Residual target
    R_true = log((Y_true + ε) / (B + ε))

    # 3. Encode to latent
    mu_z, logvar_z = vae_encoder(R_true)
    z_0 = reparameterize(mu_z, logvar_z)

    # 4. Diffusion forward
    t = torch.randint(0, T, (B,))
    noise = torch.randn_like(z_0)
    z_t = sqrt(alpha_bar[t]) * z_0 + sqrt(1 - alpha_bar[t]) * noise

    # 5. Conditioning
    H_cond = cond_encoder(L, S, product_id)
    t_emb = time_embedding(t)

    # 6. Predict noise
    noise_pred = unet(z_t, t_emb, H_cond)

    # 7. Diffusion loss
    L_diff = mse_loss(noise_pred, noise)

    # 8. Reconstruction loss (optional)
    z0_hat = (z_t - sqrt(1 - alpha_bar[t]) * noise_pred) / sqrt(alpha_bar[t])
    R_hat = vae_decoder(z0_hat)
    P_raw = B * torch.exp(R_hat)
    P = census_normalize(P_raw, admin_ids, census)
    L_pop = l1_loss(torch.log1p(P), torch.log1p(Y_true))

    # 9. Total loss
    loss = L_diff + λ_pop * L_pop + β_kl * KL(mu_z, logvar_z)

    return loss
```

**Hyperparameters**:
- `λ_pop`: reconstruction loss weight (e.g., 0.1)
- `β_kl`: KL divergence weight (e.g., 0.0001)
- Diffusion steps T: 1000
- Learning rate: 1e-4
- Batch size: 16-32

---

### Phase 6: Inference & Sampling

#### 6.1 DDPM/DDIM Sampling
**Location**: `models/sampler.py`

```python
def sample_population_map(features, product_id, census, num_steps=50):
    """
    Generate population map from VIIRS + WSF features
    """
    L, S = features
    admin_ids, census_totals = census

    # Conditioning
    H_cond = cond_encoder(L, S, product_id)

    # Start from noise
    z_T = torch.randn(B, 4, 32, 32)

    # DDIM sampling (faster)
    z_0 = ddim_sample(z_T, H_cond, num_steps)

    # Decode
    R_hat = vae_decoder(z_0)

    # Baseline
    B = baseline_dasymetric(L, S, admin_ids, census_totals)

    # Reconstruct
    P_raw = B * torch.exp(R_hat)

    # Apply census consistency
    P = census_normalize(P_raw, admin_ids, census_totals)

    return P
```

---

### Phase 7: Evaluation & Uncertainty

#### 7.1 Metrics
**Location**: `eval/metrics.py`

```python
def evaluate(predictions, targets, census):
    metrics = {}

    # Pixel-level accuracy
    metrics['MAE'] = mean_absolute_error(predictions, targets)
    metrics['RMSE'] = root_mean_squared_error(predictions, targets)

    # Census consistency
    for admin_id in census.keys():
        pred_sum = predictions[admin_mask == admin_id].sum()
        census_total = census[admin_id]
        metrics[f'census_error_{admin_id}'] = abs(pred_sum - census_total)

    # Heavy tail metrics
    dense_mask = targets > percentile(targets, 90)
    metrics['MAE_dense'] = MAE(predictions[dense_mask], targets[dense_mask])

    return metrics
```

---

#### 7.2 Uncertainty Estimation
**Location**: `eval/uncertainty.py`
**Reference**: C3-LDM.md Section 2.3

```python
def estimate_uncertainty(features, census, num_samples=10):
    """
    Generate samples from all products and compute uncertainty
    """
    samples = {}

    for product_id, product_name in enumerate(['WorldPop', 'GHS-POP', 'HRSL']):
        product_samples = []
        for _ in range(num_samples):
            P = sample_population_map(features, product_id, census)
            product_samples.append(P)
        samples[product_name] = torch.stack(product_samples)

    # Compute variance across products
    all_samples = torch.cat(list(samples.values()), dim=0)
    uncertainty_map = all_samples.var(dim=0)

    return uncertainty_map, samples
```

---

## File Structure

```
C3-LDM/
├── data/
│   ├── dataset.py              # Multi-product dataloader
│   ├── paired_dataset/
│   │   └── multi_product_pairing.csv
│   └── tiles_2020/             # WorldPop data
│
├── models/
│   ├── baseline.py             # Dasymetric baseline
│   ├── vae.py                  # Residual VAE
│   ├── time_embedding.py       # Sinusoidal time encoding
│   ├── conditional_encoder.py  # Dual-branch encoder
│   ├── product_embedding.py    # Product ID embeddings
│   ├── unet.py                 # Diffusion U-Net with FiLM
│   ├── census_layer.py         # Census normalization
│   └── sampler.py              # DDPM/DDIM sampling
│
├── eval/
│   ├── metrics.py              # Evaluation metrics
│   └── uncertainty.py          # Uncertainty quantification
│
├── train.py                    # Main training script
├── inference.py                # Sampling script
├── config.yaml                 # Hyperparameters
└── IMPLEMENTATION_ROADMAP.md   # This file
```

---

## Implementation Order (Recommended)

1. **Start with basics** (Phase 1):
   - Baseline dasymetric module
   - Simple VAE
   - Time embedding

2. **Build conditioning** (Phase 2):
   - Dual-branch encoder
   - Product embeddings
   - Test conditioning separately

3. **Core diffusion** (Phase 3):
   - U-Net with FiLM
   - Diffusion forward/reverse process
   - Train on single product first

4. **Add constraints** (Phase 4):
   - Census-consistency layer
   - Verify gradient flow

5. **Multi-product training** (Phase 5):
   - Dataloader with pairing
   - Train with all products
   - Monitor convergence

6. **Inference & evaluation** (Phase 6-7):
   - DDIM sampler for fast generation
   - Metrics and uncertainty
   - Visualizations

---

## Key Implementation Notes

### Census Consistency
- Must be differentiable (use masked operations)
- Apply AFTER exponentiating residuals
- Test gradient flow thoroughly

### Multi-Scale Conditioning
- Low-res branch: larger kernels, larger strides
- High-res branch: smaller kernels, preserve detail
- Fusion: concatenate then compress

### Product Conditioning
- Use learned embeddings (not one-hot)
- Add to spatial conditioning, not just global
- Test interpolation between products

### Residual Modeling
- Always work in log-space for stability
- Baseline prevents mode collapse
- Clamp extreme values if needed

### Training Stability
- Start with reconstruction loss only
- Gradually increase diffusion loss weight
- Use EMA for model weights
- Gradient clipping (max norm = 1.0)

---

## Expected Challenges

1. **Memory**: 256×256 tiles with diffusion → use gradient checkpointing
2. **Census data**: May not have admin boundaries for all tiles → use soft constraints
3. **Product disagreement**: Large variance → may need separate models per product initially
4. **Heavy tails**: Dense urban cores → use weighted losses
5. **Convergence**: Multiple losses → careful tuning of λ weights

---

## Success Criteria

- ✅ Census totals match exactly (by construction)
- ✅ Better pixel-level accuracy than baseline RF
- ✅ Captures uncertainty from product disagreement
- ✅ Reasonable samples from all three products
- ✅ Interpretable residuals (can explain corrections)

---

## Next Steps

Start with **Phase 1** - implement baseline and VAE, verify on small subset of data before scaling up.
