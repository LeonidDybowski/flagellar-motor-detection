#!/bin/bash
#SBATCH --job-name=rtdetr-bn-recal
#SBATCH --partition=alpha
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/home/h5/ledy730h/flagella/logs/rtdetr/%x-%j.out
#SBATCH --error=/home/h5/ledy730h/flagella/logs/rtdetr/%x-%j.err

cd /home/h5/ledy730h/flagella
source .venv/bin/activate

python src/rtdetr/diagnostics/bn_recal.py
