#!/bin/bash
#SBATCH --job-name=rtdetr-val-vs-predict
#SBATCH --partition=alpha
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/h5/ledy730h/flagella/logs/rtdetr/%x-%j.out
#SBATCH --error=/home/h5/ledy730h/flagella/logs/rtdetr/%x-%j.err

cd /home/h5/ledy730h/flagella
source .venv/bin/activate

python src/rtdetr/diagnostics/val_vs_predict.py
