#!/bin/bash
#SBATCH --job-name=rtdetr-predict-test
#SBATCH --partition=alpha
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/home/h5/ledy730h/flagella/logs/rtdetr/%x-%j.out
#SBATCH --error=/home/h5/ledy730h/flagella/logs/rtdetr/%x-%j.err

# Logs land in logs/rtdetr/rtdetr-predict-test-<jobid>.{out,err}
# Predictions of the baseline weights on the unlabelled test tomograms
# (results/rtdetr/baseline/test_predictions.csv, at the validation threshold).

cd /home/h5/ledy730h/flagella
source .venv/bin/activate

python src/rtdetr/baseline/predict_test.py
