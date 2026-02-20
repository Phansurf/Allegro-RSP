#!/bin/bash
#SBATCH -o log_nacl.out
#SBATCH -J NaCl
#SBATCH -p gpu
#SBATCH -t 7-00:00:00
#SBATCH --gres=gpu:3
#SBATCH -N 1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=64

echo "Starting NVIDIA-SMI Monitor..."
(
    while true; do
        echo "----------------------------------------------------------------"
        echo "Timestamp: $(date)"
        nvidia-smi
        echo "----------------------------------------------------------------"
        sleep 600
    done
) &
MONITOR_PID=$!

export WANDB_MODE="offline"

ulimit -s unlimited
ulimit -u 65535
module purge
module load apps/anaconda3/2022.10 compiler/cuda/11.4 compiler/gcc/10.2.0

# Initialize conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate allegro-rsp

model_types=("3-step" "1-step" "allegro")
cuda_devices=(0 1 2)

# install package
pip install -e .

# prepare dataset
cd datasets/NaCl/NaCl
python data.py
python process-loss_weight.py

# test datasets
cd ../test_NaCl
python data.py
cd ../test_NaCl-defect
python data.py
cd ../test_NaCl-expand
python data.py
python process.py
cd ../../..

# train
for idx in "${!model_types[@]}"; do
  model_type="${model_types[$idx]}"
  cuda_device="${cuda_devices[$idx]}"
  (
    # train model
    export CUDA_VISIBLE_DEVICES="$cuda_device"
    nequip-train "configs/NaCl/reciprocal_${model_type}.yaml" --equivariance-test --warn-unused

    # test model
    output_fields_arg=""
    if [ ${model_type} != "allegro" ]; then
      output_fields_arg="--output-fields atomic_charges,total_charge,short_energy,reciprocal_energy"
    fi
    # test NaCl
    mkdir -p "results/reciprocal/NaCl/test_NaCl/NaCl(${model_type})"
    nequip-evaluate --train-dir "results/reciprocal/NaCl/NaCl(${model_type})" \
      --dataset-config "configs/NaCl/test/reciprocal_test.yaml" \
      --batch-size 4 \
      ${output_fields_arg} \
      --output-fields-from-original-dataset total_energy,forces \
      --output "results/reciprocal/NaCl/test_NaCl/NaCl(${model_type})/NaCl(${model_type})_test.xyz" \
      --log "results/reciprocal/NaCl/test_NaCl/NaCl(${model_type})/NaCl(${model_type})_evaluate.log"
    # test expansion
    mkdir -p "results/reciprocal/NaCl/test_NaCl-expand/NaCl(${model_type})"
    nequip-evaluate --train-dir "results/reciprocal/NaCl/NaCl(${model_type})" \
      --dataset-config "configs/NaCl/test/reciprocal_expand.yaml" \
      --batch-size 4 \
      ${output_fields_arg} \
      --output-fields-from-original-dataset total_energy,forces \
      --output "results/reciprocal/NaCl/test_NaCl-expand/NaCl(${model_type})/NaCl(${model_type})_test.xyz" \
      --log "results/reciprocal/NaCl/test_NaCl-expand/NaCl(${model_type})/NaCl(${model_type})_evaluate.log"
    # test defect
    mkdir -p "results/reciprocal/NaCl/test_NaCl-defect/NaCl(${model_type})"
    nequip-evaluate --train-dir "results/reciprocal/NaCl/NaCl(${model_type})" \
      --dataset-config "configs/NaCl/test/reciprocal_defect.yaml" \
      --batch-size 4 \
      ${output_fields_arg} \
      --output-fields-from-original-dataset total_energy,forces \
      --output "results/reciprocal/NaCl/test_NaCl-defect/NaCl(${model_type})/NaCl(${model_type})_test.xyz" \
      --log "results/reciprocal/NaCl/test_NaCl-defect/NaCl(${model_type})/NaCl(${model_type})_evaluate.log"
  ) > "log_nacl_${model_type}.out" 2>&1 &
done

wait

kill $MONITOR_PID

python plot.py
