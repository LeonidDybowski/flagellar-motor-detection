#!/bin/bash
#SBATCH --job-name=rtdetr-evaluate
#SBATCH --partition=alpha
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:4
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=/home/h5/ledy730h/flagella/logs/rtdetr/%x-%j.out
#SBATCH --error=/home/h5/ledy730h/flagella/logs/rtdetr/%x-%j.err

# Logs land in logs/rtdetr/rtdetr-evaluate-<jobid>.{out,err}
# Prerequisite: src/rtdetr/baseline/train.py must have produced
# results/rtdetr/baseline/train/weights/best.pt (or set FLAGELLA_RTDETR_CKPT).

cd /home/h5/ledy730h/flagella
source .venv/bin/activate

python src/rtdetr/baseline/evaluate.py

# per-motor F-beta of the same checkpoint on whole validation tomograms.
python src/rtdetr/baseline/fbeta.py --best

