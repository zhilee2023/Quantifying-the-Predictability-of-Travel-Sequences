# Quantifying the Predictability of Travel Sequences (VQ-VAE)

Official code companion to *Quantifying the Predictability of Travel Sequences Using a Vector-Quantized Variational Autoencoder*. This repository contains **two independent parts**:

| Part | Folder | Entry scripts | How parameters are set |
|------|--------|---------------|-------------------------|
| **Experiment 1 — Gaussian–Markov** | [`gaussian/`](gaussian/) | `data_gen.py`, `experiment_1.py` | **Constants at top of each file** (no CLI). Full tables → [`gaussian/README.md`](gaussian/README.md). |
| **San Francisco (Cabspotting) GPS** | [`sf_2/`](sf_2/) | `src/run_sf_experiment.py`, `scripts/plot_sf_figures.py` | **Command-line arguments** (`--help`). Full tables → [`sf_2/README.md`](sf_2/README.md). |

Install and run **each** part from its own directory (different `requirements.txt`).

---

## How to start (summary)

### A. Gaussian–Markov (`gaussian/`)

```bash
cd gaussian
pip install -r requirements.txt
# Optional: regenerate data (slow) — tune constants in data_gen.py first
# python data_gen.py
python experiment_1.py
```

- **Working directory:** `gaussian/`.  
- **Device:** CUDA if available (see `experiment_1.py`).  
- **Optional env:** `MACHINE_ID` tags output folders (default `local`).  
- **Outputs:** `rate_distortion_results_<MACHINE_ID>_<timestamp>/` (git-ignored).  
- **Parameter reference:** [`gaussian/README.md`](gaussian/README.md) (data generation + VQ sweep constants).

### B. San Francisco (`sf_2/`)

```bash
cd sf_2
pip install -r requirements.txt
# Put Cabspotting CSV at data/sf_dataset.csv
python src/run_sf_experiment.py --help
python src/run_sf_experiment.py --tolerance-km 2.5 --sample-intervals 5 --interpolation-methods linear --codebook-sizes 256 --num-epochs 2 --max-users 50 --device cpu --data-dir data/sf_dataset.csv --output-dir sfexp_result --run-name smoke
python scripts/plot_sf_figures.py --workspace . --run-dir sfexp_result --mode all
```

- **Working directory:** `sf_2/`.  
- **Parameter reference:** [`sf_2/README.md`](sf_2/README.md) (all CLI flags + HPC sharding + plotting).

---

## Dependencies

- **Gaussian:** [`gaussian/requirements.txt`](gaussian/requirements.txt) — `numpy`, `scipy`, `pandas`, `matplotlib`, `torch`.  
- **SF:** [`sf_2/requirements.txt`](sf_2/requirements.txt) — PyTorch + geospatial stack (`geopandas`, `pyproj`, `shapely`, …).

---

## Git LFS (SF bundle)

Model weights and pickles under `sf_2/sfexp_result/` use **Git LFS** for `*.pt` / `*.pkl` ([`.gitattributes`](.gitattributes)). After clone:

```bash
git lfs install
git lfs pull   # if files show as pointers
```

GitHub limits a **single LFS object to 2 GB**; very large CSVs may be excluded — see [`.gitignore`](.gitignore) and [`sf_2/sfexp_result/README.md`](sf_2/sfexp_result/README.md).

---

## Citation

```bibtex
@misc{Li2025,
  author       = {Zhi Li and Zhibin Chen and Minghui Zhong},
  title        = {Quantifying the Predictability of Travel Sequences Using a Vector-Quantized Variational Autoencoder},
  note         = {Preprint, under review},
  year         = {2025},
}
```

---

## 中文说明

| 部分 | 启动 | 参数在哪里改 |
|------|------|----------------|
| **Gaussian** | `cd gaussian` → `pip install -r requirements.txt` → `python experiment_1.py` | 见 [`gaussian/README.md`](gaussian/README.md)：修改 `data_gen.py` / `experiment_1.py` **文件顶部常量**（无命令行参数）。 |
| **旧金山 SF** | `cd sf_2` → 放置 `data/sf_dataset.csv` → `python src/run_sf_experiment.py ...` | 见 [`sf_2/README.md`](sf_2/README.md)：全部用 **命令行参数**；`python src/run_sf_experiment.py --help` 查看完整列表。 |
| **作图** | `python scripts/plot_sf_figures.py --workspace . --run-dir sfexp_result` | 同上，`--help` 与子 README 中的表格。 |

更细的启动步骤、默认值与 HPC 任务编号说明以两个子目录下的 **README** 为准。
