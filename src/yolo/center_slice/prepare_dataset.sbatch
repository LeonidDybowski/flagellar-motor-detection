#!/bin/bash
#SBATCH --job-name=yolo-center-prepare
#SBATCH --partition=alpha
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/home/h5/ledy730h/flagella/logs/yolo/%x-%j.out
#SBATCH --error=/home/h5/ledy730h/flagella/logs/yolo/%x-%j.err

# Logs land in logs/yolo/yolo-center-prepare-<jobid>.{out,err}
# CPU-only work; --gres=gpu:1 is required by this cluster's job-submission
# policy even for non-GPU jobs (see src/precompute_volumes.sbatch).
# Prerequisite: src/yolo/baseline/prepare_dataset.py (baseline tiles are linked, not rebuilt).

cd /home/h5/ledy730h/flagella
source .venv/bin/activate

python src/yolo/center_slice/prepare_dataset.py
