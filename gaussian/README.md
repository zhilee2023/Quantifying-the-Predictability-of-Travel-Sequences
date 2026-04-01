# Experiment 1 — Gaussian–Markov sequences and VQ-VAE rate–distortion

This folder is **self-contained**: synthetic 2D Gaussian–Markov trajectories, theoretical rate–distortion baselines (`data_gen.py`), and VQ-VAE training with a rate–distortion / predictability sweep (`experiment_1.py`).

**There is no CLI for these two scripts.** Hyperparameters live at the **top of each `.py` file** as constants; edit the file, save, then run.

---

## 1. Environment and how to start

```bash
cd gaussian
pip install -r requirements.txt
```

| Step | Command | Notes |
|------|---------|--------|
| (Optional) regenerate data | `python data_gen.py` | Long run; overwrites `data/*.npy`. Skip if you use the repo’s bundled `data/`. |
| Train + RD sweep | `python experiment_1.py` | Uses `data/` and writes under `rate_distortion_results_<MACHINE_ID>_<timestamp>/`. |

**Working directory:** always run from **`gaussian/`** so `data/` and imports resolve.

**PyTorch device:** `experiment_1.py` uses CUDA if available, else CPU (see `DEVICE = ...` in the script).

**Environment variable (optional):**

| Variable | Effect |
|----------|--------|
| `MACHINE_ID` | Prefix for output folder name. Default: `local`. |

```bash
# Linux / macOS
export MACHINE_ID=hpc01

# Windows PowerShell
$env:MACHINE_ID = "hpc01"
```

---

## 2. Parameters — `data_gen.py`

All settings are **constants at the top of the file** (lines ~11–32). Change them in an editor, then run `python data_gen.py`.

| Constant | Default (typical) | Meaning |
|----------|-------------------|---------|
| `SEQUENCE_LEN` | `1_000_000` | Training samples (after split logic). |
| `DIMENSION` | `2` | State dimension (2D). |
| `AR_ORDER` | `5` | Autoregressive order. |
| `SIGMA_Z` | `1.0` | White-noise std. |
| `THETA_VALUES` | `np.logspace(-3, 2, 50)` | Grid for theoretical RD curve. |
| `N_FREQ` | `100_000` | Frequency resolution for theoretical RD. |
| `TOTAL_LENGTH` | derived | Total simulated length; includes burn-in for AR. |
| `OUTPUT_DIR` | `"data"` | Folder for `X_train.npy`, `X_val.npy`, `coeff.npy`, `D_vals.npy`, `R_vals.npy`, `distortion_rate_curve.png`. |

**Outputs (under `gaussian/data/`):**

- `X_train.npy`, `X_val.npy` — sequences for VQ-VAE  
- `coeff.npy` — AR coefficients  
- `D_vals.npy`, `R_vals.npy` — theoretical distortion / rate pairs for the baseline curve  
- `distortion_rate_curve.png` — plot of the baseline RD curve  

---

## 3. Parameters — `experiment_1.py`

All settings are **constants at the top of the file** (lines ~21–47). Change them, then run `python experiment_1.py`.

### Training / optimization

| Constant | Default | Meaning |
|----------|---------|---------|
| `T` | `150` | Time steps per sliding window (sequence length fed to the model). |
| `BATCH_SIZE` | `512` | DataLoader batch size. |
| `NUM_EPOCHS` | `20` | Training epochs per distortion target in the sweep. |
| `PRETRAIN_EPOCHS` | `1` | Warm-up epochs before the main objective. |

### Model (`EC_VQVAE`)

| Constant | Default | Meaning |
|----------|---------|---------|
| `BETA` | `1.0` | Logged to metadata; see loss terms in `model_train`. |
| `SIGMA` | `1.0` | Passed into `model_train` (reconstruction / noise scale). |
| `KERNEL_SIZE` | `13` | Temporal conv kernel size. |
| `HIDDEN_CHANNELS` | `32` | Conv trunk width. |
| `EMBEDDING_DIM` | `6` | VQ embedding dimension. |
| `NUM_CONV_LAYERS` | `3` | Number of conv layers. |
| `CODEBOOK_SIZE` | `128` | Codebook size \(K\). |

### Distortion sweep

| Constant | Default | Meaning |
|----------|---------|---------|
| `DISTORTIONS` | `np.arange(1.75, 0.0, -0.1)` | Distortion **targets** swept in order (one model train per value). |

### Paths

| Constant | Default | Meaning |
|----------|---------|---------|
| `DATA_DIR` | `"data"` | Loads `X_train.npy`, `X_val.npy`, `R_vals.npy`, `D_vals.npy` from here. |
| `OUTPUT_BASE` | `"rate_distortion_results"` | Output root name. |
| Output folder | `rate_distortion_results_<MACHINE_ID>_<timestamp>/` | Contains `rate_distortion_results.txt`, plots, and logs. |

---

## 4. Layout (reference)

| File | Role |
|------|------|
| `data_gen.py` | Simulates AR process; writes `data/*.npy` and `distortion_rate_curve.png`. |
| `experiment_1.py` | Loads `data/`, trains VQ-VAE per distortion, logs metrics and figures. |
| `model.py`, `sequence_gen.py`, `ctw_estimate.py` | Model, batches, entropy / RD helpers. |

---

## 5. Citation

See the repository root `README.md` for BibTeX.

---

## 中文说明

- **启动**：在 `gaussian` 目录下先 `pip install -r requirements.txt`，再运行 `python experiment_1.py`（或先 `python data_gen.py` 重新生成数据）。  
- **参数**：`data_gen.py` 与 `experiment_1.py` **没有命令行参数**，均在各自文件**顶部常量**中修改（见上文表格）。  
- **输出**：`rate_distortion_results_*` 目录（已被 `.gitignore` 忽略）。  
- **环境变量**：`MACHINE_ID` 用于区分输出目录前缀，默认 `local`。
