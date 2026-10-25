#!/bin/bash
# Prints the HPRC onboarding evidence (module list, nvidia-smi, torch GPU check).
# Run INSIDE an allocated interactive GPU job, never on a login node:
#   srun --partition=gpu --gres=gpu:1 --ntasks=1 --cpus-per-task=8 --mem=32G --time=01:00:00 --pty bash -l
#   bash edge/hprc_evidence.sh
if [ -z "$SLURM_JOB_ID" ]; then
  echo "Not inside a SLURM job -- request a GPU node with srun first." >&2
  exit 1
fi

echo "=== $(date) | host $(hostname) | job $SLURM_JOB_ID | partition $SLURM_JOB_PARTITION"
echo "=== module list"
module list 2>&1
echo "=== nvidia-smi"
nvidia-smi
echo "=== python GPU check"
python -c 'import torch; print("torch", torch.__version__, "| cuda_available", torch.cuda.is_available(), "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no GPU")'
