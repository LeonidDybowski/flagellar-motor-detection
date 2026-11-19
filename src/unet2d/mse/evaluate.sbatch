#!/bin/bash
#SBATCH --job-name=unet2d-eval
#SBATCH --partition=alpha
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:4
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=/home/h5/ledy730h/flagella/logs/unet2d/%x-%j.out
#SBATCH --error=/home/h5/ledy730h/flagella/logs/unet2d/%x-%j.err

# Logs land in logs/unet2d/unet2d-eval-<jobid>.{out,err}
# Experiment: mse (results in results/unet2d/mse/).
# Override the checkpoint with FLAGELLA_CKPT=<path>.

cd /home/h5/ledy730h/flagella
source .venv/bin/activate

export OMP_NUM_THREADS=6

torchrun --standalone --nnodes=1 --nproc_per_node=4 src/unet2d/mse/evaluate.py