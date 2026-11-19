#!/bin/bash
#SBATCH --job-name=torch-cnn
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00

# Adjust for your cluster, e.g. `module load python/3.12 cuda` or `source .venv/bin/activate`.
python train.py --epochs 10
