#!/bin/bash
#SBATCH --job-name=geolife
#SBATCH --array=0-5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
# 若集群要求分区/账号，取消下面两行注释并改成你的队列名、项目号：
# #SBATCH --partition=gpu
# #SBATCH --account=YOUR_PROJECT
#SBATCH -o geolife_slurm_%A_%a.out
#SBATCH -e geolife_slurm_%A_%a.err
#
# 6 路并行：每个 array task = 一种 (采样间隔 × 插值)，同一任务内顺序训 K=64,128,256。
# 训练轮数：--num-epochs 20（另加 1 epoch 预训练，见 model_train）。
#
# 提交（在仓库根目录或任意目录，建议 cd 到 Quantifying-the-Predictability-of-Travel-Sequences 再 sbatch 本文件）：
#   sbatch scripts/submit_geolife_slurm.sh
#
# 合并：各任务输出在 geolife_results/geolife_hpc_coarse_<id>/ ，用 pandas 合并 CSV。

set -euo pipefail

# 作业提交目录（sbatch 时所在目录）；若集群未设置则退化为脚本所在目录的上一级
ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT"

export PYTHONUNBUFFERED=1

# ---------- 环境：按你集群改 ----------
# module purge
# module load cuda/12.x
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate train_env
# ------------------------------------

TASK="${SLURM_ARRAY_TASK_ID:?}"
RUN_NAME="geolife_hpc_coarse_${TASK}"

exec python experiment_geolife.py \
  --num-epochs 20 \
  --hpc-coarse-task "${TASK}" \
  --run-name "${RUN_NAME}" \
  --device cuda
