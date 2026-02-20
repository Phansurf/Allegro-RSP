#!/bin/bash
#SBATCH -o log_hfo2.out
#SBATCH -J HfO2
#SBATCH -p gpu
#SBATCH -t 7-00:00:00
#SBATCH --gres=gpu:3
#SBATCH -N 1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=12

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
cd datasets/HfO2/HfO2
python data-loss_weight.py
cd ../../..

# train
for idx in "${!model_types[@]}"; do
  model_type="${model_types[$idx]}"
  cuda_device="${cuda_devices[$idx]}"
  (
    # train model
    export CUDA_VISIBLE_DEVICES="$cuda_device"
    nequip-train "configs/HfO2/reciprocal_${model_type}.yaml" --equivariance-test --warn-unused

    # test model
    output_fields_arg=""
    if [ ${model_type} != "allegro" ]; then
      output_fields_arg="--output-fields atomic_charges,total_charge,short_energy,reciprocal_energy"
    fi
    # test HfO2
    mkdir -p "results/reciprocal/HfO2/test_HfO2/HfO2(${model_type})"
    nequip-evaluate --train-dir "results/reciprocal/HfO2/HfO2(${model_type})" \
      --dataset-config "configs/HfO2/test/reciprocal_test.yaml" \
      --batch-size 16 \
      ${output_fields_arg} \
      --output-fields-from-original-dataset total_energy,forces \
      --output "results/reciprocal/HfO2/test_HfO2/HfO2(${model_type})/HfO2(${model_type})_test.xyz" \
      --log "results/reciprocal/HfO2/test_HfO2/HfO2(${model_type})/HfO2(${model_type})_evaluate.log"
    # test defect
    mkdir -p "results/reciprocal/HfO2/test_HfO2-defect/HfO2(${model_type})"
    nequip-evaluate --train-dir "results/reciprocal/HfO2/HfO2(${model_type})" \
      --dataset-config "configs/HfO2/test/reciprocal_defect.yaml" \
      --batch-size 4 \
      ${output_fields_arg} \
      --output-fields-from-original-dataset total_energy,forces \
      --output "results/reciprocal/HfO2/test_HfO2-defect/HfO2(${model_type})/HfO2(${model_type})_test.xyz" \
      --log "results/reciprocal/HfO2/test_HfO2-defect/HfO2(${model_type})/HfO2(${model_type})_evaluate.log"
  ) > "log_hfo2_${model_type}.out" 2>&1 &
done

wait

kill $MONITOR_PID

python plot.py
