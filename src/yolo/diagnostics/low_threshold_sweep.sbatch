#!/bin/bash
#SBATCH --job-name=yolo-diag-low-threshold
#SBATCH --partition=alpha
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/h5/ledy730h/flagella/logs/yolo/%x-%j.out
#SBATCH --error=/home/h5/ledy730h/flagella/logs/yolo/%x-%j.err

# Diagnostic of the first baseline run (results/yolo/baseline_first_run/):
# whole-tomogram F-beta of best.pt swept down to threshold 0.001.

cd /home/h5/ledy730h/flagella
source .venv/bin/activate

python src/yolo/diagnostics/low_threshold_sweep.py
