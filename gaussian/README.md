# Experiment 1 — Gaussian–Markov sequences and VQ-VAE rate–distortion

This folder is **self-contained**: synthetic 2D Gaussian–Markov trajectories, theoretical rate–distortion baselines (`data_gen.py`), and VQ-VAE training with a rate–distortion / predictability sweep (`experiment_1.py`).

## Layout

| File | Role |
|------|------|
| `data_gen.py` | Simulates the AR process, writes `data/X_train.npy`, `data/X_val.npy`, `coeff.npy`, `D_vals.npy`, `R_vals.npy`, and a baseline RD figure `data/distortion_rate_curve.png`. |
| `experiment_1.py` | Loads the arrays under `data/`, trains `EC_VQVAE` over a grid of distortion targets, logs metrics, saves plots under `rate_distortion_results_<MACHINE_ID>_<timestamp>/`. |
| `model.py`, `sequence_gen.py`, `ctw_estimate.py` | Model, batches, and entropy / RD helpers used by the scripts above. |

Pre-generated tensors and the baseline curve are **included** under `data/` so you can run `experiment_1.py` immediately. Re-running `data_gen.py` overwrites those files (long run; uses ~1M training samples by default).

## Environment

```bash
cd gaussian
pip install -r requirements.txt
```

Use a recent **PyTorch** build (CPU or CUDA). `experiment_1.py` picks `cuda` when available.

Optional: set `MACHINE_ID` to tag output folders (default is `local`):

```bash
# Linux / macOS
export MACHINE_ID=hpc01

# Windows PowerShell
$env:MACHINE_ID = "hpc01"
```

## Run order

### 1) (Optional) Regenerate synthetic data and theoretical RD curve

```bash
cd gaussian
python data_gen.py
```

This can take a while and requires enough RAM for the configured sequence lengths.

### 2) Train VQ-VAE and sweep distortions

```bash
cd gaussian
python experiment_1.py
```

Outputs appear under `gaussian/rate_distortion_results_<MACHINE_ID>_<timestamp>/` (git-ignored).

## Citation

See the repository root `README.md` for BibTeX.

---

## 中文说明

- **依赖**：在 `gaussian` 目录下执行 `pip install -r requirements.txt`。
- **数据**：默认已附带 `data/` 中的 `.npy` 与基准图，可直接运行 `python experiment_1.py`。若需重新模拟数据，再运行 `python data_gen.py`（耗时较长）。
- **输出**：实验结果写入 `rate_distortion_results_*` 目录（已被 `.gitignore` 忽略）。
