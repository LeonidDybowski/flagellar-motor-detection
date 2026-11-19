#!/bin/bash
#SBATCH --job-name=yolo-predict-test
#SBATCH --partition=alpha
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/home/h5/ledy730h/flagella/logs/yolo/%x-%j.out
#SBATCH --error=/home/h5/ledy730h/flagella/logs/yolo/%x-%j.err

# Logs land in logs/yolo/yolo-predict-test-<jobid>.{out,err}
# Predictions of the baseline weights on the unlabelled test tomograms
# (results/yolo/baseline/test_predictions.csv, at the validation threshold).

cd /home/h5/ledy730h/flagella
source .venv/bin/activate

python src/yolo/baseline/predict_test.py
