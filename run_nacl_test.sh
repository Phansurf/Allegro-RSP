#!/bin/bash
#SBATCH -o log_nacl_test.out
#SBATCH -J NaCl_test
#SBATCH -p gpu
#SBATCH -t 7-00:00:00
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=32

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
mkdir -p "results/reciprocal/NaCl/test_NaCl/NaCl(allegro)"
nequip-evaluate --train-dir "results/reciprocal/NaCl/NaCl(allegro)" --dataset-config "configs/NaCl/test/reciprocal_test.yaml" --batch-size 16 --output-fields-from-original-dataset total_energy,forces --output "results/reciprocal/NaCl/test_NaCl/NaCl(allegro)/NaCl(allegro)_test.xyz" --log "results/reciprocal/NaCl/test_NaCl/NaCl(allegro)/NaCl(allegro)_evaluate.log"
mkdir -p "results/reciprocal/NaCl/test_NaCl/NaCl(1-step)"
nequip-evaluate --train-dir "results/reciprocal/NaCl/NaCl(1-step)" --dataset-config "configs/NaCl/test/reciprocal_test.yaml" --batch-size 16 --output-fields atomic_charges,total_charge,short_energy,reciprocal_energy --output-fields-from-original-dataset total_energy,forces --output "results/reciprocal/NaCl/test_NaCl/NaCl(1-step)/NaCl(1-step)_test.xyz" --log "results/reciprocal/NaCl/test_NaCl/NaCl(1-step)/NaCl(1-step)_evaluate.log"
mkdir -p "results/reciprocal/NaCl/test_NaCl/NaCl(3-step)"
nequip-evaluate --train-dir "results/reciprocal/NaCl/NaCl(3-step)" --dataset-config "configs/NaCl/test/reciprocal_test.yaml" --batch-size 16 --output-fields atomic_charges,total_charge,short_energy,reciprocal_energy --output-fields-from-original-dataset total_energy,forces --output "results/reciprocal/NaCl/test_NaCl/NaCl(3-step)/NaCl(3-step)_test.xyz" --log "results/reciprocal/NaCl/test_NaCl/NaCl(3-step)/NaCl(3-step)_evaluate.log"

# defect test
mkdir -p "results/reciprocal/NaCl/test_NaCl-defect/NaCl(allegro)"
nequip-evaluate --train-dir "results/reciprocal/NaCl/NaCl(allegro)" --dataset-config "configs/NaCl/test/reciprocal_defect.yaml" --batch-size 8 --output-fields-from-original-dataset total_energy,forces --output "results/reciprocal/NaCl/test_NaCl-defect/NaCl(allegro)/NaCl(allegro)_test.xyz" --log "results/reciprocal/NaCl/test_NaCl-defect/NaCl(allegro)/NaCl(allegro)_evaluate.log"
mkdir -p "results/reciprocal/NaCl/test_NaCl-defect/NaCl(1-step)"
nequip-evaluate --train-dir "results/reciprocal/NaCl/NaCl(1-step)" --dataset-config "configs/NaCl/test/reciprocal_defect.yaml" --batch-size 8 --output-fields atomic_charges,total_charge,short_energy,reciprocal_energy --output-fields-from-original-dataset total_energy,forces --output "results/reciprocal/NaCl/test_NaCl-defect/NaCl(1-step)/NaCl(1-step)_test.xyz" --log "results/reciprocal/NaCl/test_NaCl-defect/NaCl(1-step)/NaCl(1-step)_evaluate.log"
mkdir -p "results/reciprocal/NaCl/test_NaCl-defect/NaCl(3-step)"
nequip-evaluate --train-dir "results/reciprocal/NaCl/NaCl(3-step)" --dataset-config "configs/NaCl/test/reciprocal_defect.yaml" --batch-size 8 --output-fields atomic_charges,total_charge,short_energy,reciprocal_energy --output-fields-from-original-dataset total_energy,forces --output "results/reciprocal/NaCl/test_NaCl-defect/NaCl(3-step)/NaCl(3-step)_test.xyz" --log "results/reciprocal/NaCl/test_NaCl-defect/NaCl(3-step)/NaCl(3-step)_evaluate.log"

# expansion test
mkdir -p "results/reciprocal/NaCl/test_NaCl-expand/NaCl(allegro)"
nequip-evaluate --train-dir "results/reciprocal/NaCl/NaCl(allegro)" --dataset-config "configs/NaCl/test/reciprocal_expand.yaml" --batch-size 4 --output-fields-from-original-dataset total_energy,forces --output "results/reciprocal/NaCl/test_NaCl-expand/NaCl(allegro)/NaCl(allegro)_test.xyz" --log "results/reciprocal/NaCl/test_NaCl-expand/NaCl(allegro)/NaCl(allegro)_evaluate.log"
mkdir -p "results/reciprocal/NaCl/test_NaCl-expand/NaCl(1-step)"
nequip-evaluate --train-dir "results/reciprocal/NaCl/NaCl(1-step)" --dataset-config "configs/NaCl/test/reciprocal_expand.yaml" --batch-size 4 --output-fields atomic_charges,total_charge,short_energy,reciprocal_energy --output-fields-from-original-dataset total_energy,forces --output "results/reciprocal/NaCl/test_NaCl-expand/NaCl(1-step)/NaCl(1-step)_test.xyz" --log "results/reciprocal/NaCl/test_NaCl-expand/NaCl(1-step)/NaCl(1-step)_evaluate.log"
mkdir -p "results/reciprocal/NaCl/test_NaCl-expand/NaCl(3-step)"
nequip-evaluate --train-dir "results/reciprocal/NaCl/NaCl(3-step)" --dataset-config "configs/NaCl/test/reciprocal_expand.yaml" --batch-size 4 --output-fields atomic_charges,total_charge,short_energy,reciprocal_energy --output-fields-from-original-dataset total_energy,forces --output "results/reciprocal/NaCl/test_NaCl-expand/NaCl(3-step)/NaCl(3-step)_test.xyz" --log "results/reciprocal/NaCl/test_NaCl-expand/NaCl(3-step)/NaCl(3-step)_evaluate.log"

python plot.py

nequip-deploy build --train-dir "results/reciprocal/NaCl/NaCl(allegro)/" "results/reciprocal/NaCl/NaCl(allegro)/jit.pth"
nequip-deploy build --train-dir "results/reciprocal/NaCl/NaCl(1-step)/" "results/reciprocal/NaCl/NaCl(1-step)/jit.pth"
nequip-deploy build --train-dir "results/reciprocal/NaCl/NaCl(3-step)/" "results/reciprocal/NaCl/NaCl(3-step)/jit.pth"

python data_gen.py
python get_force_constant.py
python get_lattice_constant.py
