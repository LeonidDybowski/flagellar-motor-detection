#!/bin/bash
#SBATCH --job-name=unet3d-eval
#SBATCH --partition=alpha
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:4
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=/home/h5/ledy730h/flagella/logs/unet3d/%x-%j.out
#SBATCH --error=/home/h5/ledy730h/flagella/logs/unet3d/%x-%j.err

# Logs land in logs/unet3d/unet3d-eval-<jobid>.{out,err}
# Experiment: mse (results in results/unet3d/mse/).
# Override the checkpoint with FLAGELLA_CKPT=<path>.

cd /home/h5/ledy730h/flagella
source .venv/bin/activate

export OMP_NUM_THREADS=6

torchrun --standalone --nnodes=1 --nproc_per_node=4 src/unet3d/mse/evaluate.py