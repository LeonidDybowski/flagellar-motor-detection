#!/bin/bash
#SBATCH --job-name=yolo-evaluate
#SBATCH --partition=alpha
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:4
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=/home/h5/ledy730h/flagella/logs/yolo/%x-%j.out
#SBATCH --error=/home/h5/ledy730h/flagella/logs/yolo/%x-%j.err

# Logs land in logs/yolo/yolo-evaluate-<jobid>.{out,err}
# Prerequisite: src/yolo/baseline/train.py must have produced
# results/yolo/baseline/train/weights/best.pt (or set FLAGELLA_YOLO_CKPT).

cd /home/h5/ledy730h/flagella
source .venv/bin/activate

python src/yolo/baseline/evaluate.py

# per-motor F-beta of the same checkpoint on whole validation tomograms.
python src/yolo/baseline/fbeta.py --best

