#!/bin/bash
#SBATCH -o log_trainer_modify.out
#SBATCH -J trainer_modify
#SBATCH -p gpu
#SBATCH -t 7-00:00:00
#SBATCH --gres=gpu:1
#SBATCH -N 1

export WANDB_MODE="offline"

ulimit -s unlimited
ulimit -u 65535
module purge
module load apps/anaconda3/2022.10 compiler/cuda/11.4 compiler/gcc/10.2.0

# Initialize conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate allegro-rsp

# install package
pip install -e .

python trainer_modify.py
