#!/bin/bash
#SBATCH --job-name=rtdetr-train
#SBATCH --partition=alpha
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:4
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/h5/ledy730h/flagella/logs/rtdetr/%x-%j.out
#SBATCH --error=/home/h5/ledy730h/flagella/logs/rtdetr/%x-%j.err

# Logs land in logs/rtdetr/rtdetr-train-<jobid>.{out,err}
# Prerequisite: src/rtdetr/baseline/prepare_dataset.py (or its sbatch) must have
# run first, to build {DATA_ROOT}/train_tiles_rtdetr/ and src/rtdetr/baseline/data.yaml.
#
# No torchrun: Ultralytics spawns its own multi-GPU DDP internally via the
# device="0,1,2,3" kwarg set in train.py.

cd /home/h5/ledy730h/flagella
source .venv/bin/activate

# fp32_trainer.py must be importable by Ultralytics' generated multi-GPU script.
export PYTHONPATH=/home/h5/ledy730h/flagella/src/rtdetr/baseline:$PYTHONPATH

python src/rtdetr/baseline/train.py

# per-motor F-beta of every epoch's weights on whole validation tomograms
# (results/rtdetr/baseline/fbeta_history.csv) and of best.pt (fbeta.csv, peaks.csv, matches.csv).
python src/rtdetr/baseline/fbeta.py
