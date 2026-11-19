#!/bin/bash
#SBATCH --job-name=rtdetr-analyze-loss
#SBATCH --partition=alpha
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/home/h5/ledy730h/flagella/logs/rtdetr/%x-%j.out
#SBATCH --error=/home/h5/ledy730h/flagella/logs/rtdetr/%x-%j.err

# Logs land in logs/rtdetr/rtdetr-analyze-loss-<jobid>.{out,err}
# CPU-only work (reads results.csv, no model inference); --gres=gpu:1 is
# required by this cluster's job-submission policy even for non-GPU jobs
# (see src/precompute_volumes.sbatch).
# Prerequisite: src/rtdetr/baseline/train.py must have produced
# results/rtdetr/baseline/train/results.csv.

cd /home/h5/ledy730h/flagella
source .venv/bin/activate

python src/rtdetr/baseline/analyze_loss.py
