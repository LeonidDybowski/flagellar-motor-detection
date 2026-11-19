#!/bin/bash
#SBATCH --job-name=yolo-train
#SBATCH --partition=alpha
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/home/h5/ledy730h/flagella/logs/yolo/%x-%j.out
#SBATCH --error=/home/h5/ledy730h/flagella/logs/yolo/%x-%j.err

# Logs land in logs/yolo/yolo-train-<jobid>.{out,err}
# Prerequisite: src/yolo/baseline/prepare_dataset.py (or its sbatch) must have run
# first, to build {DATA_ROOT}/train_tiles/ and src/yolo/data.yaml.
#
# No torchrun: Ultralytics spawns its own multi-GPU DDP internally via the
# device="0,1,2,3" kwarg set in train.py.

cd /home/h5/ledy730h/flagella
source .venv/bin/activate

python src/yolo/baseline/train.py

# per-motor F-beta of every epoch's weights on whole validation tomograms
# (results/yolo/baseline/fbeta_history.csv) and of best.pt (fbeta.csv, peaks.csv, matches.csv).
python src/yolo/baseline/fbeta.py
