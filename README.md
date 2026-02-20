# Allegro-RSP

## Introduction
This repository implements the **Reciprocal Space Neural Network (RSNN)** framework, designed to capture long-range interactions (such as Coulombic and van der Waals forces) in machine learning interatomic potentials. By transforming spatial information into reciprocal space, the model overcomes the cutoff limitations of standard local descriptors.

The project integrates RSNN with state-of-the-art architectures and includes significant enhancements to the training pipeline:

- **Allegro + RSP**: Integration of the Reciprocal Space Potential (RSP) with the **Allegro** model. Validated on **NaCl** (Coulomb + van der Waals) and **HfO$_2$** (pristine & defective) datasets.
- **Enhanced NequIP Framework**: A modified `nequip` core supporting **loss-weighted training** and a robust **3-step training strategy** (Local $\rightarrow$ Long-range $\rightarrow$ Joint Fine-tuning) for improved convergence.

---

## Installation

### With Internet

**1. Create Python environment**

```bash
conda create -n allegro-rsp python=3.9 -y
conda activate allegro-rsp
```

**2. Install dependencies**

Load CUDA:
```bash
module load compiler/cuda/11.4
```

Install numpy, pymatgen, wandb:
```bash
conda install -c conda-forge numpy=1.26.0 pymatgen wandb -y
```

Install PyTorch 1.11.0:
```bash
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
```

Install torch_scatter (Python 3.9, CUDA 11.3, Linux x86_64):
```bash
wget https://data.pyg.org/whl/torch-1.11.0%2Bcu113/torch_scatter-2.0.9-cp39-cp39-linux_x86_64.whl
pip install torch_scatter-2.0.9-cp39-cp39-linux_x86_64.whl
```

**3. Install Allegro-RSP**

```bash
pip install -e .
```

This installs both the bundled `nequip` and `allegro` packages together with all remaining dependencies declared in `pyproject.toml`. No separate `pip install nequip` or `pip install allegro` is needed.

### Without Internet

**Step 1 — on a machine with internet access**, download all required wheels:

```bash
mkdir -p allegro_pkgs && cd allegro_pkgs
pip download torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
rm numpy* pillow*
pip download numpy==1.26.0 pymatgen wandb setuptools>=61.0 wheel pybind11>=2.10.0 --platform manylinux2014_x86_64 --python-version 39 --only-binary=:all:
wget https://data.pyg.org/whl/torch-1.11.0%2Bcu113/torch_scatter-2.0.9-cp39-cp39-linux_x86_64.whl
pip download absl-py==0.10.0 adjustText ase==3.23.0 e3nn==0.5.6 grpcio markdown==2.6.11 markupsafe==2.1.1 opt_einsum_fx==0.1.4 opt-einsum phonopy pillow==11.3.0 scikit-learn tensorboard tensorboard-data-server==0.7.0 threadpoolctl==3.1.0 torch-runstats==0.2.0 torch-ema==0.3.0 werkzeug --platform manylinux2014_x86_64 --python-version 39 --only-binary=:all: --no-deps
wget https://files.pythonhosted.org/packages/ec/f1/923d8dcf2d54d165bb8eb1f3782ba2aa58a85d7dfd8c43d0173dde836f76/calorine-3.2.tar.gz
```

Copy the `allegro_pkgs/` directory to the target server.

**Step 2 — on the server without internet**, create a virtual environment and install:

```bash
module purge
module load apps/anaconda3/2022.10 compiler/cuda/11.4 compiler/gcc/10.2.0
python -m venv $HOME/allegro-rsp
source $HOME/allegro-rsp/bin/activate
```

```bash
export PKG_DIR=/path/to/allegro_pkgs
pip install --no-index --find-links=$PKG_DIR setuptools wheel pybind11
pip install --no-index --find-links=$PKG_DIR $PKG_DIR/*
```

Then install Allegro-RSP from the repo:
```bash
pip install -e .
```

## Dataset Preparation

Before training, generate the dataset files from raw data:

```bash
# NaCl training set
cd datasets/NaCl/NaCl
python data.py                  # generate NaCl.xyz
python process-loss_weight.py   # embed per-sample loss weights

# NaCl test sets
cd ../test_NaCl && python data.py
cd ../test_NaCl-defect && python data.py
cd ../test_NaCl-expand && python data.py && python process.py
cd ../../..

# HfO2 training set
cd datasets/HfO2/HfO2
python data-loss_weight.py      # generate HfO2.xyz with loss weights
cd ../../..
```

## Usage

### Training

```bash
# Single GPU
nequip-train configs/NaCl/reciprocal_3-step.yaml --equivariance-test --warn-unused

# Specific GPU
CUDA_VISIBLE_DEVICES=0 nequip-train configs/NaCl/reciprocal_3-step.yaml --equivariance-test --warn-unused

# Offline W&B logging
WANDB_MODE=offline nequip-train configs/NaCl/reciprocal_3-step.yaml
```

Three config variants are provided for each material (`NaCl`, `HfO2`):

| Config | Description |
|--------|-------------|
| `reciprocal_3-step.yaml` | RSP model with 3-step training (`start_stage: 1`) |
| `reciprocal_1-step.yaml` | RSP model with single-step training (`start_stage: 3`) |
| `reciprocal_allegro.yaml` | Baseline Allegro without RSP |

The RSP model is selected via `model_builders` in the config:

```yaml
model_builders:
  - allegro.model.ReciprocalNN_Allegro_charge_equilibrium
  - StressForceOutput
```

### 3-Step Training

Set `start_stage: 1` in the config to enable 3-step training:

1. **Stage 1**: Train short-range Allegro only; ChargeMLP and ReciprocalNN are frozen
2. **Stage 2**: Freeze Allegro; train ChargeMLP + ReciprocalNN
3. **Stage 3**: Joint fine-tuning of all parameters

Stage transitions are triggered when the learning rate drops below the thresholds in `early_stopping_lower_bounds.LR` (a list of 3 values, one per stage). Use `start_stage: 3` for standard single-step training.

### Evaluation

```bash
# Baseline Allegro model
nequip-evaluate \
  --train-dir results/reciprocal/NaCl/NaCl\(allegro\) \
  --dataset-config configs/NaCl/test/reciprocal_test.yaml \
  --batch-size 4 \
  --output-fields-from-original-dataset total_energy,forces \
  --output results/NaCl_allegro_test.xyz \
  --log results/NaCl_allegro_evaluate.log

# RSP model (1-step or 3-step) — add --output-fields for RSP quantities
nequip-evaluate \
  --train-dir results/reciprocal/NaCl/NaCl\(3-step\) \
  --dataset-config configs/NaCl/test/reciprocal_test.yaml \
  --batch-size 4 \
  --output-fields atomic_charges,total_charge,short_energy,reciprocal_energy \
  --output-fields-from-original-dataset total_energy,forces \
  --output results/NaCl_3step_test.xyz \
  --log results/NaCl_3step_evaluate.log
```

Test configs available under `configs/{NaCl,HfO2}/test/`:

| Config | Description |
|--------|-------------|
| `reciprocal_test.yaml` | Standard test set |
| `reciprocal_defect.yaml` | Defect structures |
| `reciprocal_expand.yaml` | Expanded lattice (NaCl only) |

### Deployment

```bash
nequip-deploy build --train-dir <run_dir> <deployed_model.pth>
```

### Benchmarking

```bash
nequip-benchmark configs/NaCl/reciprocal_3-step.yaml
```

### Unit Tests

```bash
pytest tests/                              # all tests
pytest tests/model/test_allegro.py        # single file
pytest tests/integration/                 # integration tests only
```

## Running Full Experiments

Shell scripts are provided to run the complete train + evaluate + deploy pipeline for each material. They are designed for SLURM clusters but can also be run directly.

**NaCl** — trains all 3 model variants in parallel on 3 GPUs:

```bash
# Submit via SLURM
sbatch run_nacl.sh

# Or run in background locally
bash task_nacl.sh
```

**HfO2** — same structure:

```bash
sbatch run_hfo2.sh
bash task_hfo2.sh
```

**Test only** (requires pre-trained models in `results/`):

```bash
sbatch run_nacl_test.sh
sbatch run_hfo2_test.sh
```

**Trainer modification experiment:**

```bash
sbatch run_trainer_modify.sh
```

Each `run_*.sh` script handles: dataset prep → parallel training → evaluation on all test sets → deploy → post-processing (`run_nacl_test.sh` calls `plot.py`, `data_gen.py`, `get_force_constant.py`, `get_lattice_constant.py`; `run_hfo2_test.sh` calls `plot.py`, `get_phonon_HfO2.py`).

## Key Config Parameters

| Parameter | Description |
|-----------|-------------|
| `r_max` | Real-space cutoff (Å) |
| `k_max` | Reciprocal-space cutoff |
| `eta` | Ewald summation smearing parameter |
| `start_stage` | Training stage (1=3-step, 3=1-step) |
| `FCN_hidden` / `FCN_layers` | ReciprocalNN MLP size |
| `charge_mlp_latent_dimensions` | ChargeMLP hidden dims |
| `model_input_fields` | Declare extra per-atom fields fed to the model (e.g. `atomic_charges: 1x0e`, `total_charge: 1x0e`, `loss_weight: 1x0e`) |
| `include_keys` | Extra fields to load from dataset (e.g. `[atomic_charges, total_charge, loss_weight]`); adding `loss_weight` automatically enables per-sample loss weighting |
| `loss_coeffs` | Loss terms — RSP training requires `total_charge` and `stress` (e.g. `{total_energy: [10., PerAtomMSELoss], forces: 1., total_charge: 1., stress: 1.}`) |
| `early_stopping_lower_bounds.LR` | List of 3 LR thresholds triggering stage 1→2→3 transitions |

## Article

For a detailed description of the method, mathematical derivation, and benchmark results, please refer to our paper:

**Capturing long-range interactions with a reciprocal-space neural network** *Ruijie Guo, Hongyu Yu, Liangliang Hong, Shiyou Chen, Xingao Gong, and Hongjun Xiang* **arXiv**: [2211.16684](https://arxiv.org/abs/2211.16684)
