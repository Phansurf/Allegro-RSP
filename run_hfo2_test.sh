#!/bin/bash
#SBATCH -o log_hfo2_test.out
#SBATCH -J HfO2_test
#SBATCH -p gpu
#SBATCH -t 7-00:00:00
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=12

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

# test
mkdir -p "results/reciprocal/HfO2/test_HfO2/HfO2(allegro)"
nequip-evaluate --train-dir "results/reciprocal/HfO2/HfO2(allegro)" --dataset-config "configs/HfO2/test/reciprocal_test.yaml" --batch-size 16 --output-fields-from-original-dataset total_energy,forces --output "results/reciprocal/HfO2/test_HfO2/HfO2(allegro)/HfO2(allegro)_test.xyz" --log "results/reciprocal/HfO2/test_HfO2/HfO2(allegro)/HfO2(allegro)_evaluate.log"
mkdir -p "results/reciprocal/HfO2/test_HfO2/HfO2(1-step)"
nequip-evaluate --train-dir "results/reciprocal/HfO2/HfO2(1-step)" --dataset-config "configs/HfO2/test/reciprocal_test.yaml" --batch-size 16 --output-fields atomic_charges,total_charge,short_energy,reciprocal_energy --output-fields-from-original-dataset total_energy,forces --output "results/reciprocal/HfO2/test_HfO2/HfO2(1-step)/HfO2(1-step)_test.xyz" --log "results/reciprocal/HfO2/test_HfO2/HfO2(1-step)/HfO2(1-step)_evaluate.log"
mkdir -p "results/reciprocal/HfO2/test_HfO2/HfO2(3-step)"
nequip-evaluate --train-dir "results/reciprocal/HfO2/HfO2(3-step)" --dataset-config "configs/HfO2/test/reciprocal_test.yaml" --batch-size 16 --output-fields atomic_charges,total_charge,short_energy,reciprocal_energy --output-fields-from-original-dataset total_energy,forces --output "results/reciprocal/HfO2/test_HfO2/HfO2(3-step)/HfO2(3-step)_test.xyz" --log "results/reciprocal/HfO2/test_HfO2/HfO2(3-step)/HfO2(3-step)_evaluate.log"

# defect test
mkdir -p "results/reciprocal/HfO2/test_HfO2-defect/HfO2(allegro)"
nequip-evaluate --train-dir "results/reciprocal/HfO2/HfO2(allegro)" --dataset-config "configs/HfO2/test/reciprocal_defect.yaml" --batch-size 16 --output-fields-from-original-dataset total_energy,forces --output "results/reciprocal/HfO2/test_HfO2-defect/HfO2(allegro)/HfO2(allegro)_test.xyz" --log "results/reciprocal/HfO2/test_HfO2-defect/HfO2(allegro)/HfO2(allegro)_evaluate.log"
mkdir -p "results/reciprocal/HfO2/test_HfO2-defect/HfO2(1-step)"
nequip-evaluate --train-dir "results/reciprocal/HfO2/HfO2(1-step)" --dataset-config "configs/HfO2/test/reciprocal_defect.yaml" --batch-size 16 --output-fields atomic_charges,total_charge,short_energy,reciprocal_energy --output-fields-from-original-dataset total_energy,forces --output "results/reciprocal/HfO2/test_HfO2-defect/HfO2(1-step)/HfO2(1-step)_test.xyz" --log "results/reciprocal/HfO2/test_HfO2-defect/HfO2(1-step)/HfO2(1-step)_evaluate.log"
mkdir -p "results/reciprocal/HfO2/test_HfO2-defect/HfO2(3-step)"
nequip-evaluate --train-dir "results/reciprocal/HfO2/HfO2(3-step)" --dataset-config "configs/HfO2/test/reciprocal_defect.yaml" --batch-size 16 --output-fields atomic_charges,total_charge,short_energy,reciprocal_energy --output-fields-from-original-dataset total_energy,forces --output "results/reciprocal/HfO2/test_HfO2-defect/HfO2(3-step)/HfO2(3-step)_test.xyz" --log "results/reciprocal/HfO2/test_HfO2-defect/HfO2(3-step)/HfO2(3-step)_evaluate.log"

python plot.py

nequip-deploy build --train-dir "results/reciprocal/HfO2/HfO2(allegro)/" "results/reciprocal/HfO2/HfO2(allegro)/jit.pth"
nequip-deploy build --train-dir "results/reciprocal/HfO2/HfO2(1-step)/" "results/reciprocal/HfO2/HfO2(1-step)/jit.pth"
nequip-deploy build --train-dir "results/reciprocal/HfO2/HfO2(3-step)/" "results/reciprocal/HfO2/HfO2(3-step)/jit.pth"

python get_phonon_HfO2.py
