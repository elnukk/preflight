#!/bin/bash
#SBATCH --job-name=toy-train
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00

python train.py --epochs 5
