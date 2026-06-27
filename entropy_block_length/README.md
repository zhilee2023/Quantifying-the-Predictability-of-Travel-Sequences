# Entropy Block-Length Sensitivity Test

Simulation code for comparing entropy estimators under different **block / window lengths** on binary symbolic sequences. This package supports the entropy-estimator comparison study used in the predictability paper revision.

## What it tests

| Estimator | Block-length role |
|-----------|-------------------|
| `CTW` | Full-sequence baseline (no block length). |
| `ACTW_fixed` | Non-overlapping fixed blocks; block length is swept. Does **not** use change-point labels. |
| `ACTW_random` | Random local windows centered around each candidate block length. Does **not** know segment boundaries in advance. |
| `LZ_GKB` | Sliding-window match-length estimator; window size is swept. |

Sources:

- **Stationary:** Bernoulli, Markov-1, Markov-2 (random parameters each replicate)
- **Nonstationary:** piecewise Bernoulli, drifting Bernoulli, piecewise Markov (random parameters each replicate)

Default settings: `N = 100`, sequence length `15000`.

## Quick start

```bash
cd entropy_block_length
python run_block_length_sweep.py
```

Custom sweep:

```bash
python run_block_length_sweep.py \
  --length 15000 \
  --n 100 \
  --block-lengths 512,1024,2048,4096,8192 \
  --output-dir results
```

## Outputs

Written to `results/` (or `--output-dir`):

- `block_length_raw.csv` — per-replicate estimates
- `block_length_summary.csv` — mean, bias, std, var by family × estimator × block length
- `block_length_bias.svg` — bias vs block length
- `block_length_mean.svg` — mean estimate vs block length

## Interpretation

- For **stationary** sources, reference entropy is the true entropy rate (averaged across sources in each replicate).
- For **nonstationary** sources, reference is a time-averaged local / segment-wise benchmark, not a unique stationary entropy rate.
- ACTW block lengths are **hyperparameters**, not tuned to synthetic segment boundaries.

## 中文说明

本目录用于测试不同 **block / window 长度** 对熵估计器的影响：

- `ACTW_fixed`：固定 block 长度、非重叠分块 CTW
- `ACTW_random`：随机局部窗口，**事先不知道**分段位置
- `LZ_GKB`：滑动窗口 LZ 估计器

默认：`N=100` 条序列，每条长度 `15000`。结果输出 CSV 和 SVG 敏感性曲线。
