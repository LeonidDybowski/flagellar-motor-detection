#!/bin/bash
#SBATCH --job-name=precompute-volumes
#SBATCH --partition=alpha
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/home/h5/ledy730h/flagella/logs/precompute-volumes-%j.out
#SBATCH --error=/home/h5/ledy730h/flagella/logs/precompute-volumes-%j.err

# Logs land in logs/precompute-volumes-<jobid>.{out,err}

cd /home/h5/ledy730h/flagella
source .venv/bin/activate

python src/precompute_volumes.py --workers 6
