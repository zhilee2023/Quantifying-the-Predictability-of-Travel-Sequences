#!/bin/bash
#SBATCH --job-name=geolife_fine
#SBATCH --array=0-17
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
## #SBATCH --partition=gpu
## #SBATCH --account=YOUR_PROJECT
#SBATCH -o geolife_fine_%A_%a.out
#SBATCH -e geolife_fine_%A_%a.err
#
# 18 路并行：每个 array task = 一种 (间隔 × 插值 × 码本 K)。
# 注意：direct_ctw / Markov 会在每个 task 里重复算，合并时按需去重。
#
# 提交：
#   sbatch scripts/submit_geolife_slurm_fine.sh

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT"

export PYTHONUNBUFFERED=1

# source ~/miniconda3/etc/profile.d/conda.sh && conda activate train_env

TASK="${SLURM_ARRAY_TASK_ID:?}"
RUN_NAME="geolife_hpc_fine_${TASK}"

exec python experiment_geolife.py \
  --num-epochs 20 \
  --hpc-fine-task "${TASK}" \
  --run-name "${RUN_NAME}" \
  --device cuda
