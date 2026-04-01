# Quantifying the Predictability of Travel Sequences (VQ-VAE)

Official code companion to *Quantifying the Predictability of Travel Sequences Using a Vector-Quantized Variational Autoencoder*. This repository contains **two independent parts**:

| Part | Folder | What it does |
|------|--------|----------------|
| **Experiment 1 — Gaussian–Markov** | [`gaussian/`](gaussian/) | Synthetic 2D Gaussian–Markov sequences, theoretical rate–distortion curves, VQ-VAE training and predictability / rate–distortion sweeps. |
| **San Francisco (Cabspotting) GPS** | [`sf_2/`](sf_2/) | Real taxi-style trajectories: resampling, direct / Markov / VQ-VAE **CTW** predictability, bundled HPC-style outputs under `sf_2/sfexp_result/`, plotting scripts. SF-only (no Geolife / Beijing). |

Install and run **each** part from its own directory (different dependency sets). They do not share a single virtual environment requirement file.

---

## Quick start

### A. Gaussian–Markov experiment (`gaussian/`)

```bash
cd gaussian
pip install -r requirements.txt
# Optional: regenerate data (long); repo already includes gaussian/data/*.npy
# python data_gen.py
python experiment_1.py
```

Details: [`gaussian/README.md`](gaussian/README.md).

- **Data:** `gaussian/data/` ships precomputed `X_train`, `X_val`, coefficients, and theoretical `D_vals` / `R_vals` so `experiment_1.py` runs without regenerating data.
- **Outputs:** `rate_distortion_results_<MACHINE_ID>_<timestamp>/` (ignored by Git). Override tag with env var `MACHINE_ID` (default `local`).

### B. San Francisco experiment (`sf_2/`)

```bash
cd sf_2
pip install -r requirements.txt
# Place Cabspotting-style CSV at sf_2/data/sf_dataset.csv (see sf_2/data/README.md)
python src/run_sf_experiment.py --help
python scripts/plot_sf_figures.py --help
```

Details: [`sf_2/README.md`](sf_2/README.md).

---

## Dependencies (summary)

- **Gaussian:** `numpy`, `scipy`, `pandas`, `matplotlib`, `torch` — see [`gaussian/requirements.txt`](gaussian/requirements.txt).
- **SF:** PyTorch stack plus **geospatial** libraries (`geopandas`, `pyproj`, `shapely`, …) — see [`sf_2/requirements.txt`](sf_2/requirements.txt).

---

## Git Large File Storage (LFS)

The SF bundle stores model weights and pickles under `sf_2/sfexp_result/` using **Git LFS** for `*.pt` and `*.pkl` (see [`.gitattributes`](.gitattributes)). After cloning:

```bash
git lfs install
git lfs pull   # if pointers were not expanded
```

GitHub caps a **single LFS object at 2 GB**. Very large artifacts (e.g. some `latent_code_occurrences.csv` exports) must stay local or be shared separately; they are listed in [`.gitignore`](.gitignore). See [`sf_2/sfexp_result/README.md`](sf_2/sfexp_result/README.md).

---

## Citation

If you use this code, please cite:

```bibtex
@misc{Li2025,
  author       = {Zhi Li and Zhibin Chen and Minghui Zhong},
  title        = {Quantifying the Predictability of Travel Sequences Using a Vector-Quantized Variational Autoencoder},
  note         = {Preprint, under review},
  year         = {2025},
}
```

---

## 中文概览

- **高斯–马尔可夫实验**（`gaussian/`）：合成二维轨迹与率失真分析；已附带 `data/` 中数据可直接 `python experiment_1.py`；详见 [`gaussian/README.md`](gaussian/README.md)。
- **旧金山真实 GPS 实验**（`sf_2/`）：需自备 `sf_2/data/sf_dataset.csv`；权重等大体积极使用 Git LFS；详见 [`sf_2/README.md`](sf_2/README.md)。
