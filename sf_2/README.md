# San Francisco (Cabspotting) trajectory predictability (`sf_2`)

End-to-end layout for the SF experiment (HPC-friendly). Code is SF-only (no other city datasets).

## Layout

```
sf_2/
  data/
    sf_dataset.csv              # you provide (not committed if large)
    bay_area_counties_ca.geojson # optional; else downloaded on first map use
  sfexp_result/                 # outputs + copied HPC runs (see below)
  src/
    run_sf_experiment.py        # main entry: resample → direct/Markov/VQ → CSVs
    model.py, sequence_gen.py, ctw_estimate.py, sf_preprocess.py, ...
  scripts/
    plot_sf_figures.py          # bar chart + optional geo heatmap
    bay_area_map_style.py
  config/
    sf_default.json             # example hyperparameters (reference only)
```

## Environment

```bash
cd sf_2
pip install -r requirements.txt
```

**Device:** pass `--device cpu` or `--device cuda`. Multi-GPU: use one process per GPU, e.g. `CUDA_VISIBLE_DEVICES=0 python src/run_sf_experiment.py ...` and `CUDA_VISIBLE_DEVICES=1 ...` for another slice, or rely on a single GPU / CPU.

## Data

Place the Cabspotting-style CSV as **`data/sf_dataset.csv`** (columns include `trajectory`, `timestamp`, `start_point`, `end_point` in WKT).

## Run experiment (local or HPC)

From **`sf_2`** (so paths resolve):

```bash
# quick smoke (single tolerance, one K, few epochs) — CPU or GPU
python src/run_sf_experiment.py \
  --data-dir data/sf_dataset.csv \
  --output-dir sfexp_result \
  --run-name smoke_local \
  --tolerance-km 2.5 \
  --sample-intervals 5 \
  --interpolation-methods linear \
  --codebook-sizes 256 \
  --num-epochs 2 \
  --max-users 50 \
  --device cpu
```

On Windows PowerShell, use backtick line continuation or a single line; for GPU use `--device cuda` and set `CUDA_VISIBLE_DEVICES` if you run multiple jobs.

Full sweeps match your HPC settings via `--tolerance-kms`, `--sample-intervals`, `--codebook-sizes`, `--hpc-*` task ids, etc. (see `python src/run_sf_experiment.py --help`).

## Results folder (`sfexp_result`)

- On this machine, `sfexp_result/` can hold the **full HPC output** (all tasks and checkpoints). Total size is several GB; the repo uses **Git LFS** for `*.pt` / `*.pkl` (see root `.gitattributes`). See `sfexp_result/README.md`.
- Plotting expects paths like `sfexp_result/sf_run_1980198_task_0/tol_2p5km_int_5min/` with `user_level_results.csv` and `vqvae_*_meta.json` / `.pt` as needed.

## Figures

```bash
python scripts/plot_sf_figures.py --workspace . --run-dir sfexp_result --mode all
```

- **`--mode bar`:** K × resolution predictability bar chart (needs aggregated `user_level_results.csv` under `--run-dir`).
- **`--mode geo`:** viridis heatmap + Bay Area basemap (needs `data/sf_dataset.csv` and a valid `--scenario-dir` / `--checkpoint`).
- Override paths if your HPC tree differs.

Outputs are written next to `sf_2/` as `fig_bay_area_k_predictability_bar.*` and `fig_predictability_variation_geo.*`.

## Citation

If you use the predictability / VQ-VAE methodology, cite the paper; see the repository root `README.md` for BibTeX.
