#!/bin/bash
#SBATCH --job-name=unet2d-train
#SBATCH --partition=alpha
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:4
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00
#SBATCH --output=/home/h5/ledy730h/flagella/logs/unet2d/%x-%j.out
#SBATCH --error=/home/h5/ledy730h/flagella/logs/unet2d/%x-%j.err

# Logs land in logs/unet2d/unet2d-train-<jobid>.{out,err}
# Experiment: baseline (results in results/unet2d/baseline/).

cd /home/h5/ledy730h/flagella
source .venv/bin/activate

export OMP_NUM_THREADS=6

torchrun --standalone --nnodes=1 --nproc_per_node=4 src/unet2d/baseline/train.py