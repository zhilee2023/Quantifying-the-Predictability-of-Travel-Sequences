#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH -t 24:00:00
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -e output_error_file/gaussian_nonstationary.e.%J.txt
#SBATCH -o output_error_file/gaussian_nonstationary.o.%J.txt
#SBATCH --mem=32G

set -euo pipefail

WORKDIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${WORKDIR}" || exit 1
mkdir -p output_error_file

echo "[gaussian-ns] JOB=${SLURM_JOB_ID} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

python experiment_gaussian_nonstationary.py \
  --device cuda \
  --scenario matrix_shift \
  --shift-levels 0.0 0.33 0.66 1.0 \
  --train-length 120000 \
  --val-length 60000 \
  --distortions 1.50 1.00 0.75 0.50 \
  --num-epochs 20 \
  --output-dir "${WORKDIR}/gaussian_nonstationary_results" \
  --run-name "job_${SLURM_JOB_ID}"
