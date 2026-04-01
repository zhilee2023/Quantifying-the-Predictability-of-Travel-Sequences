# San Francisco (Cabspotting) trajectory predictability (`sf_2`)

End-to-end layout for the SF experiment (HPC-friendly). Code is **SF-only** (no other city datasets).

**Main entry:** `python src/run_sf_experiment.py` — full CLI (`--help`).  
**Figures:** `python scripts/plot_sf_figures.py` — CLI (`--help`).

Always run commands from **`sf_2/`** (so default paths to `data/` and `sfexp_result/` resolve).

---

## 1. Environment

```bash
cd sf_2
pip install -r requirements.txt
```

**Device:** `--device cpu` or `--device cuda` (default: cuda if available).  
**Multi-GPU:** one process per GPU, e.g. `CUDA_VISIBLE_DEVICES=0 python src/run_sf_experiment.py ...`.

---

## 2. Data

Place the Cabspotting-style CSV at **`data/sf_dataset.csv`** (see `data/README.md` for columns: `trajectory`, `timestamp`, `start_point`, `end_point` in WKT).

Default `--data-dir` is `sf_2/data/sf_dataset.csv` (path relative to repo layout when run from `sf_2/`).

---

## 3. How to start — `run_sf_experiment.py`

### 3.1 Quick smoke (single scenario, small run)

```bash
python src/run_sf_experiment.py ^
  --data-dir data/sf_dataset.csv ^
  --output-dir sfexp_result ^
  --run-name smoke_local ^
  --tolerance-km 2.5 ^
  --sample-intervals 5 ^
  --interpolation-methods linear ^
  --codebook-sizes 256 ^
  --num-epochs 2 ^
  --max-users 50 ^
  --device cpu
```

On **Linux/macOS** use `\` line continuation instead of `^`.

### 3.2 Full help

```bash
python src/run_sf_experiment.py --help
```

---

## 4. CLI parameters — `run_sf_experiment.py`

### Data / paths

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--dataset` | `sf` | `sf` | Only SF is supported in this bundle. |
| `--data-dir` | path | `data/sf_dataset.csv` | Input CSV (Cabspotting-style). |
| `--output-dir` | path | `sfexp_result` | Root for all runs and CSVs. |
| `--run-name` | str | auto timestamp | If set, outputs go to `output-dir/run-name/` instead of a new timestamp folder. |

### Grid / scenarios (spatial–temporal design)

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--tolerance-kms` | floats | `2.5 5 7.5 10` | Cell widths (km) for grid + `D_target`; one scenario per (tolerance × interval). |
| `--tolerance-km` | float | — | If set, **single** tolerance; overrides `--tolerance-kms`. |
| `--sample-intervals` | ints | `5 15 30` | Resampling intervals (minutes). |
| `--interpolation-methods` | str | `linear nearest` | Resampling methods (e.g. `linear` only for a smaller grid). |
| `--coordinate-scale-km` | float | `10.0` | Divides centered x/y (km) before VQ; must stay consistent across train/eval. |
| `--projection` | `utm` / `web_mercator` | `utm` | Planar projection for metric km. |

### VQ-VAE training

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--codebook-sizes` | ints | `1024 2048 4096` | Codebook sizes \(K\) to sweep. |
| `--time-steps` | int | `32` | Sequence length (windows). |
| `--window-stride` | int | `4` | Stride for sliding windows. |
| `--batch-size` | int | `64` | Training batch size. |
| `--eval-batch-size` | int | `256` | Eval batch size. |
| `--num-epochs` | int | `80` | Epochs per VQ scenario. |
| `--pretrain-epochs` | int | `5` | VQ-only warmup before augmented Lagrangian. |
| `--hidden-channels` | int | `64` | Conv trunk width. |
| `--embedding-dim` | int | `16` | VQ embedding dimension. |
| `--num-conv-layers` | int | `3` | Conv depth. |
| `--kernel-size` | int | `13` | Temporal kernel size. |
| `--commitment-cost` | float | `0.35` | VQ commitment loss weight. |
| `--lr` | float | `3e-4` | Adam learning rate. |
| `--sigma` | float | `1.08` | ALM multiplier when recon violates `D_target`. |
| `--gamma` | float | `0.5` | StepLR: multiply LR by this every `--step-size` epochs. |
| `--step-size` | int | `15` | StepLR period (epochs). |

### Baselines / filters

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--markov-alpha` | float | `1.0` | Markov baseline scaling. |
| `--min-points` | int | `5` | Minimum points per segment. |
| `--min-vq-length` | int | `32` | Minimum length for VQ windows. |
| `--max-users` | int | none | Cap number of users (debug / smoke). |
| `--max-segments` | int | none | Cap number of segments. |

### Execution

| Argument | Description |
|----------|-------------|
| `--device` | `cpu` or `cuda`. |
| `--skip-vq` | Skip VQ training (baselines only). |

### HPC task sharding (mutually exclusive)

| Argument | IDs | Description |
|----------|-----|-------------|
| `--hpc-scenario-task` | `0..11` | One **(tolerance × sample_interval)** from the default **4×3** grid: tolerances 2.5/5/7.5/10 km × intervals 5/15/30 min. `task = 3*tolerance_index + interval_index`. |
| `--hpc-coarse-task` | `0..5` | One **(interval × interpolation)** pair: 3 intervals × 2 methods; all codebook sizes in one process. |
| `--hpc-fine-task` | `0..17` | One **(interval × interpolation × codebook K)** triple: 3×2×3 grid. |

Use exactly **one** of these when splitting jobs on a cluster.

---

## 5. Reference config

`config/sf_default.json` mirrors **example** hyperparameters (not auto-loaded by the CLI; defaults are defined in `run_sf_experiment.py`). Use it as a human-readable reference.

---

## 6. Figures — `plot_sf_figures.py`

### Start

```bash
cd sf_2
python scripts/plot_sf_figures.py --workspace . --run-dir sfexp_result --mode all
```

### CLI parameters

| Argument | Default | Description |
|----------|---------|-------------|
| `--workspace` | `sf_2` (parent of `scripts/`) | Project root containing `src/`, `data/`. |
| `--run-dir` | `workspace/sfexp_result` | Where to search for `**/user_level_results.csv`. |
| `--mode` | `all` | `bar` — K predictability bars; `geo` — heatmap + basemap; `all` — both. |
| `--data-csv` | `workspace/data/sf_dataset.csv` | Trajectory CSV for **geo** mode. |
| `--scenario-dir` | example HPC path under `run-dir` | One `tol_*` folder (needs `latent_code_occurrences.csv` for some flows). |
| `--checkpoint` | matching `.pt` under scenario | VQ checkpoint for geo mode. |

If **geo** mode skips: ensure `data-csv` exists and `checkpoint` points to a real `.pt` file.

**Outputs:** figures such as `fig_bay_area_k_predictability_bar.*` and `fig_predictability_variation_geo.*` next to `sf_2/` (see script output paths).

---

## 7. Results (`sfexp_result/`)

- Large outputs and checkpoints may use **Git LFS** (`*.pt`, `*.pkl`) — see root `.gitattributes` and `sfexp_result/README.md`.
- Plotting expects paths like `sfexp_result/.../tol_2p5km_int_5min/` with `user_level_results.csv` and `vqvae_*_meta.json` / `.pt` as needed.

---

## 8. Citation

Cite the paper; BibTeX is in the repository root `README.md`.

---

## 中文说明

- **启动**：在 `sf_2` 目录执行 `pip install -r requirements.txt`，再运行 `python src/run_sf_experiment.py`（见上文参数表）。  
- **参数**：全部通过 **命令行** 传入；完整列表以 `python src/run_sf_experiment.py --help` 为准。  
- **作图**：`python scripts/plot_sf_figures.py --help`；常用 `--workspace .`、`--run-dir sfexp_result`、`--mode bar|geo|all`。  
- **数据**：`data/sf_dataset.csv` 需自行放置（大文件一般不提交）。
