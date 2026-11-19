#!/bin/bash
#SBATCH --job-name=yolo-diag-slice-offset
#SBATCH --partition=alpha
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/h5/ledy730h/flagella/logs/yolo/%x-%j.out
#SBATCH --error=/home/h5/ledy730h/flagella/logs/yolo/%x-%j.err

# Diagnostic of the first baseline run (results/yolo/baseline_first_run/):
# score at each validation motor by slice offset.

cd /home/h5/ledy730h/flagella
source .venv/bin/activate

python src/yolo/diagnostics/slice_offset_scores.py
