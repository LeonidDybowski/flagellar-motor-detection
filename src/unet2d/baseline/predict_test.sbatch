#!/bin/bash
#SBATCH --job-name=unet2d-predict-test
#SBATCH --partition=alpha
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:4
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=/home/h5/ledy730h/flagella/logs/unet2d/%x-%j.out
#SBATCH --error=/home/h5/ledy730h/flagella/logs/unet2d/%x-%j.err

# Logs land in logs/unet2d/unet2d-predict-test-<jobid>.{out,err}
# Predictions of the baseline weights on the unlabelled test tomograms
# (results/unet2d/baseline/test_predictions.csv, at the validation threshold).
# Override the checkpoint with FLAGELLA_CKPT=<path>.

cd /home/h5/ledy730h/flagella
source .venv/bin/activate

export OMP_NUM_THREADS=6

torchrun --standalone --nnodes=1 --nproc_per_node=4 src/unet2d/baseline/predict_test.py