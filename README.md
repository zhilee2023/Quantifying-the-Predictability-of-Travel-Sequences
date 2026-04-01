# Quantifying the Predictability of Travel Sequences (San Francisco experiment)

This repository contains **only** the **`sf_2/`** tree: a self-contained pipeline for the San Francisco (Cabspotting-style) GPS experiment—resampling, direct / Markov / VQ-VAE **CTW** predictability, bundled HPC-style outputs under `sf_2/sfexp_result/`, and plotting scripts. It is **SF-only** (no Geolife / Beijing code).

- **Documentation and runbook:** [`sf_2/README.md`](sf_2/README.md)
- **Dependencies:** `pip install -r sf_2/requirements.txt` (from `sf_2/` or with `-r` path as needed)
- **Large files:** `.gitattributes` enables Git LFS for `*.pt` and `*.pkl`. Run `git lfs install` before clone/commit. Files such as `latent_code_occurrences.csv` can exceed GitHub’s **2 GB per LFS object** limit and are **not** committed (see `sf_2/sfexp_result/README.md`).

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
