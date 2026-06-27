# Quantifying the Predictability of Travel Sequences (VQ-VAE)

Code for *Quantifying the Predictability of Travel Sequences Using a Vector-Quantized Variational Autoencoder*.  
This repository has **two separate code packages** — install and run them **independently**:

| Package | Folder | What it is |
|--------|--------|------------|
| **Experiment 1 (synthetic)** | [`gaussian/`](gaussian/) | 2D Gaussian–Markov sequences + VQ-VAE rate–distortion sweep. |
| **Real GPS (San Francisco)** | [`sf_cabspotting/`](sf_cabspotting/) | Cabspotting-style SF taxi trajectories: resampling, baselines, VQ-VAE, CTW predictability. |
| **Entropy block-length test** | [`entropy_block_length/`](entropy_block_length/) | Binary entropy estimators (CTW, ACTW, LZ) with block/window-length sensitivity sweep. |

---

## Which folder should I use?

- **Paper Experiment 1 / synthetic data** → go to **`gaussian/`**  
- **Real-world SF trajectories / maps / HPC-style runs** → go to **`sf_cabspotting/`**

---

## 1. Gaussian experiment (`gaussian/`)

1. `cd gaussian`
2. `pip install -r requirements.txt`
3. `python experiment_1.py` (data is already under `gaussian/data/`; optional: `python data_gen.py` to regenerate)

**Parameters:** edited in the Python files (no CLI). See [`gaussian/README.md`](gaussian/README.md).

---

## 2. San Francisco Cabspotting experiment (`sf_cabspotting/`)

**Rule:** open a terminal **inside** `sf_cabspotting/` so paths like `data/sf_dataset.csv` work without extra typing.

1. `cd sf_cabspotting`
2. `pip install -r requirements.txt`
3. Put your dataset at **`data/sf_dataset.csv`** (format: [`sf_cabspotting/data/README.md`](sf_cabspotting/data/README.md))
4. Run the pipeline and figures:

```bash
python src/run_sf_experiment.py --data-dir data/sf_dataset.csv --output-dir sfexp_result --run-name my_run --device cpu
python scripts/plot_sf_figures.py --workspace . --run-dir sfexp_result --mode bar
```

**Parameters:** all via **command line** — `python src/run_sf_experiment.py --help`.  
Full tables (paths, grid, VQ, HPC tasks, plotting): [`sf_cabspotting/README.md`](sf_cabspotting/README.md).

---

## Dependencies

- [`gaussian/requirements.txt`](gaussian/requirements.txt) — NumPy stack + PyTorch  
- [`sf_cabspotting/requirements.txt`](sf_cabspotting/requirements.txt) — PyTorch + **geospatial** libs (`geopandas`, `pyproj`, `shapely`, …)

---

## Git LFS (real-GPS part only)

Checkpoints under `sf_cabspotting/sfexp_result/` use **Git LFS** for `*.pt` / `*.pkl`. After cloning:

```bash
git lfs install
git lfs pull
```

Single-file LFS limit on GitHub is **2 GB**; some huge CSV exports are not committed — see [`.gitignore`](.gitignore) and [`sf_cabspotting/sfexp_result/README.md`](sf_cabspotting/sfexp_result/README.md).

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

| 你要做… | 进入目录 | 说明 |
|--------|----------|------|
| 合成高斯实验 | `gaussian/` | 改脚本里常量；见 [`gaussian/README.md`](gaussian/README.md) |
| 旧金山真实 GPS | `sf_cabspotting/` | 放好 `data/sf_dataset.csv`，用命令行参数；见 [`sf_cabspotting/README.md`](sf_cabspotting/README.md) |

**注意：** 以前名为 `sf_2` 的目录已重命名为 **`sf_cabspotting`**，含义更清晰（Cabspotting / SF 出租车轨迹实验）。
