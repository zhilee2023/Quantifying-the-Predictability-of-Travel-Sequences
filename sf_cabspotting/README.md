# San Francisco — Cabspotting trajectory predictability

This directory (`sf_cabspotting/`) is the **real-GPS** experiment: Cabspotting-style San Francisco taxi trajectories, grid resampling, direct / Markov / VQ-VAE **CTW** predictability, and plotting.

It does **not** depend on `gaussian/` (synthetic experiment). Treat this folder as a **standalone project**.

---

## Folder layout

```
sf_cabspotting/
  data/
    sf_dataset.csv          ← you provide (large; not committed by default)
    bay_area_counties_ca.geojson
  sfexp_result/             ← outputs, logs, checkpoints (see README inside)
  src/
    run_sf_experiment.py    ← main CLI
    model.py, sf_preprocess.py, …
  scripts/
    plot_sf_figures.py      ← figures (bar / geo)
  config/
    sf_default.json         ← reference hyperparameters (defaults live in the CLI)
```

---

## Before you run

1. **Working directory:** always `cd` into **`sf_cabspotting/`** first.  
   The code assumes defaults like `data/sf_dataset.csv` relative to this folder.

2. **Data file:** place the Cabspotting-style CSV at **`data/sf_dataset.csv`**.  
   Column expectations: `trajectory`, `timestamp`, `start_point`, `end_point` (WKT), etc. — details in [`data/README.md`](data/README.md).

---

## Step-by-step

### Step 1 — Environment

```bash
cd sf_cabspotting
pip install -r requirements.txt
```

Use `--device cuda` if you have a GPU; use `--device cpu` otherwise.

### Step 2 — Run the experiment

Minimal example (small smoke test):

```bash
python src/run_sf_experiment.py ^
  --data-dir data/sf_dataset.csv ^
  --output-dir sfexp_result ^
  --run-name smoke ^
  --tolerance-km 2.5 ^
  --sample-intervals 5 ^
  --interpolation-methods linear ^
  --codebook-sizes 256 ^
  --num-epochs 2 ^
  --max-users 50 ^
  --device cpu
```

(Linux/macOS: use `\` at line ends instead of `^`.)

**Full list of flags:** run

```bash
python src/run_sf_experiment.py --help
```

### Step 3 — Figures

```bash
python scripts/plot_sf_figures.py --workspace . --run-dir sfexp_result --mode all
```

`--mode bar` — K predictability bar chart.  
`--mode geo` — map heatmap (needs `data/sf_dataset.csv` + valid checkpoint paths).  
`--mode all` — both.

**Plotting help:**

```bash
python scripts/plot_sf_figures.py --help
```

Figures are written as `fig_*.pdf` / `fig_*.png` next to this folder (see script output).

---

## Default paths (if you omit arguments)

| Argument | Default when running from `sf_cabspotting/` |
|----------|-----------------------------------------------|
| `--data-dir` | `data/sf_dataset.csv` |
| `--output-dir` | `sfexp_result` |
| `--run-name` | auto timestamp folder if omitted |

---

## CLI reference (summary)

### Data / IO

| Argument | Meaning |
|----------|---------|
| `--data-dir` | Path to input CSV |
| `--output-dir` | Root for all outputs |
| `--run-name` | Subfolder name under `output-dir` (optional) |

### Scenario grid

| Argument | Meaning |
|----------|---------|
| `--tolerance-kms` | List of cell widths (km), default `2.5 5 7.5 10` |
| `--tolerance-km` | Single width; overrides `--tolerance-kms` |
| `--sample-intervals` | Minutes, default `5 15 30` |
| `--interpolation-methods` | e.g. `linear nearest` |
| `--coordinate-scale-km` | Normalizes coordinates before VQ (default `10`) |
| `--projection` | `utm` or `web_mercator` |

### VQ training (high level)

| Argument | Meaning |
|----------|---------|
| `--codebook-sizes` | Codebook sizes K to sweep |
| `--time-steps`, `--window-stride` | Windowing |
| `--batch-size`, `--eval-batch-size` | Batches |
| `--num-epochs`, `--pretrain-epochs` | Training |
| `--hidden-channels`, `--embedding-dim`, `--kernel-size`, … | Architecture |
| `--lr`, `--sigma`, `--gamma`, `--step-size` | Optimization / ALM |

### HPC sharding (pick **one**)

| Argument | Range | Meaning |
|----------|-------|---------|
| `--hpc-scenario-task` | `0..11` | One (tolerance × interval) from 4×3 grid |
| `--hpc-coarse-task` | `0..5` | One (interval × interpolation); all K in-process |
| `--hpc-fine-task` | `0..17` | One (interval × interpolation × K) |

### Other

| Argument | Meaning |
|----------|---------|
| `--max-users`, `--max-segments` | Subsample for debugging |
| `--skip-vq` | Skip VQ, baselines only |
| `--device` | `cpu` or `cuda` |

### `plot_sf_figures.py`

| Argument | Meaning |
|----------|---------|
| `--workspace` | Usually `.` (this folder) |
| `--run-dir` | Where `user_level_results.csv` trees live (default `sfexp_result`) |
| `--mode` | `bar` \| `geo` \| `all` |
| `--data-csv` | Override path to `sf_dataset.csv` for geo mode |
| `--scenario-dir`, `--checkpoint` | Scenario folder + `.pt` for geo mode |

---

## `config/sf_default.json`

Example JSON for reading only; **CLI defaults** are defined in `run_sf_experiment.py`, not loaded from this file automatically.

---

## Results and Git LFS

Large `*.pt` / `*.pkl` files may be tracked with **Git LFS**. See repository root `.gitattributes` and [`sfexp_result/README.md`](sfexp_result/README.md).

---

## Citation

BibTeX is in the repository root `README.md`.

---

## 中文说明

1. **必须先** `cd sf_cabspotting`，再执行命令（路径都相对这个目录）。  
2. **数据**：把 CSV 放到 `data/sf_dataset.csv`。  
3. **主程序**：`python src/run_sf_experiment.py`，所有参数用命令行；完整列表：`--help`。  
4. **作图**：`python scripts/plot_sf_figures.py --workspace . --run-dir sfexp_result`。  
5. 目录曾用名 **`sf_2`**，现改名为 **`sf_cabspotting`**，避免与“版本号”混淆。
