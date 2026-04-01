# Experiment outputs (`sfexp_result`)

This folder is a **full mirror of the local HPC run tree** `sf_results_hpc_2` (job id `1980198`, tasks `0`–`9`): aggregated summary tables at the root, plus per-task folders `sf_run_1980198_task_*` with `sf_samples.pkl`, scenario subfolders `tol_*km_int_*min`, checkpoints (`vqvae_*.pt`), CSV/JSON logs, and figures.

## Layout (typical)

- `sf_*.csv` / `sf_*.png` — cross-run summary tables and plots.
- `sf_run_1980198_task_N/` — one HPC task; contains resampled `sf_samples.pkl` and one or more `tol_*km_int_*min/` scenario directories with `user_level_results.csv`, `vqvae_*.pt`, etc.

## Git / GitHub

**`latent_code_occurrences.csv`** can exceed **GitHub LFS’s 2 GB per-file limit** and is listed in the repo **`.gitignore`**. Keep it on disk from your HPC copy; publish via **release zip** or **Zenodo** if needed.

The repository uses **Git LFS** for `*.pt` / `*.pkl` where tracked. The full tree on disk is large; link supplemental archives in the main README if needed.

## Regenerate locally

Copy from your machine:

`robocopy <path_to>\sf_results_hpc_2  sfexp_result  /E /COPY:DAT`
