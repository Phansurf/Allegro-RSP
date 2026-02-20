# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Allegro-RSP** — a self-contained package that integrates the Allegro model with a Reciprocal Space Potential (RSP) for long-range Coulombic interactions in machine learning interatomic potentials. Paper: arXiv:2211.16684.

The package bundles two Python packages installed together via `pip install .`:
- `nequip/` — Modified NequIP 0.5.x framework (3-step training strategy, per-sample loss weighting)
- `allegro/` — Allegro model augmented with ChargeMLP + ReciprocalNN (Ewald summation in k-space)

## Installation

```bash
conda create -n allegro-rsp python=3.9
conda activate allegro-rsp
# Install PyTorch 1.11.0 with CUDA (see https://pytorch.org/get-started/previous-versions/)
pip install torch==1.11.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
# Install torch_scatter (download wheel matching your torch/CUDA version from https://github.com/rusty1s/pytorch_scatter)
pip install torch_scatter-*.whl
pip install -e .
```

`pip install .` auto-installs all remaining dependencies declared in `pyproject.toml`:
`numpy`, `ase`, `tqdm`, `e3nn>=0.4.4,<0.6.0`, `pyyaml`, `torch-runstats`, `torch-ema`, `opt_einsum_fx==0.1.4`, `wandb`, `packaging`, `importlib-metadata` (Python<3.10), `scipy`, `pymatgen`.

It also installs both bundled packages — `nequip` and `allegro` — so no separate `pip install nequip` or `pip install allegro` is needed.

The two packages that require manual pre-installation are `torch` and `torch_scatter`, because they depend on the specific CUDA version of the environment.

## Commands

**Train:**
```bash
nequip-train configs/NaCl/reciprocal_3-step.yaml
```

**Evaluate:**
```bash
nequip-evaluate --train-dir <run_dir> --dataset-config <config.yaml> --output <out.xyz>
```

**Deploy:**
```bash
nequip-deploy build --train-dir <run_dir> <deployed_model.pth>
```

**Tests:**
```bash
pytest tests/                                  # all tests
pytest tests/model/test_allegro.py            # single file
pytest tests/integration/                     # integration tests only
pytest tests/unit/                            # unit tests only
```

## Architecture

### Allegro + RSP model (`allegro/`)

The main model builder is `allegro.model.ReciprocalNN_Allegro_charge_equilibrium` (in `allegro/model/_ReciprocalNN_allegro_charge_equilibrium.py`). It assembles a `SequentialGraphNetwork` with these layers:

1. `OneHotAtomEncoding` → `RadialBasisEdgeEncoding` (with optional `NormalizedBasis`) → `SphericalHarmonicEdgeAttrs`
2. `Allegro_Module` — short-range equivariant message passing (`allegro/nn/_allegro.py`)
3. `ScalarMLP` → `edge_short_energy`; `EdgewiseReduce` → `atomic_short_energy`; `AtomwiseReduce` → `short_energy`
4. `ChargeMLP` (`allegro/nn/_long_range_modules.py`) — predicts atomic charges from edge/node invariants
5. `ReciprocalNN` (`allegro/nn/_long_range_modules.py`) — Ewald summation in reciprocal space → `reciprocal_energy` + `total_energy`

Custom field keys are in `allegro/_keys.py` and auto-registered via the `nequip.extension` entry point in `pyproject.toml`.

### 3-Step Training Strategy

Controlled by `start_stage` in the YAML config. Implemented in `nequip/train/trainer.py` (`_switch_stage()`, `_set_model_grad()`):

- **Stage 1** (`start_stage: 1`): Train short-range Allegro only; `ChargeMLP` and `ReciprocalNN` are frozen and return zeros
- **Stage 2**: Freeze Allegro; train `ChargeMLP` + `ReciprocalNN`
- **Stage 3**: Joint fine-tuning of all parameters

Stage transitions are triggered when `LR` drops below the thresholds in `early_stopping_lower_bounds.LR` (one per stage). Use `start_stage: 3` for single-step training (standard Allegro baseline).

### Loss-Weighted Training

`nequip/train/loss.py` supports per-sample loss weighting. It is activated automatically at runtime when `loss_weight` is present in the batch data — there is no separate config flag. To use it: add `loss_weight: 1x0e` to `model_input_fields`, add `loss_weight` to `include_keys`, and ensure weights are precomputed in the dataset file.

### Config Structure

YAML configs in `configs/{NaCl,HfO2}/`:
- `reciprocal_3-step.yaml` — RSP model with 3-step training
- `reciprocal_1-step.yaml` — RSP model with single-step training
- `reciprocal_allegro.yaml` — baseline Allegro (no RSP)
- `test/` — evaluation configs for test sets (normal, defect, expand)

Key config parameters:
- `model_builders` — `[allegro.model.ReciprocalNN_Allegro_charge_equilibrium, StressForceOutput]` for RSP; `[allegro.model.Allegro, StressForceOutput]` for baseline
- `model_input_fields` — declares extra per-atom fields fed to the model (e.g. `atomic_charges: 1x0e`, `total_charge: 1x0e`, `loss_weight: 1x0e`)
- `r_max` — real-space cutoff (Å)
- `k_max` — reciprocal-space cutoff
- `eta` — Ewald summation smearing parameter
- `start_stage` — training stage (1 = 3-step, 3 = 1-step)
- `FCN_hidden`, `FCN_layers` — ReciprocalNN MLP dimensions
- `charge_mlp_latent_dimensions` — ChargeMLP hidden dimensions
- `early_stopping_lower_bounds.LR` — list of 3 LR thresholds triggering stage 1→2→3 transitions (only relevant when `start_stage: 1`)
- `include_keys: [atomic_charges, total_charge, loss_weight]` — extra fields to load from the dataset file

HfO2 test configs only have `reciprocal_test.yaml` and `reciprocal_defect.yaml` (no expand variant).

## Bundled Dependencies

`nequip/utils/torch_geometric/` is a trimmed-down subset of PyTorch Geometric bundled directly into the package. It provides only the basic graph data structures needed by NequIP and avoids pulling in the full PyG dependency. Do not replace it with an external `torch_geometric` import.

## Top-Level Utility Scripts

The repo root contains standalone analysis scripts (not part of the installable package):
- `data_gen.py` — dataset generation
- `get_force_constant.py`, `get_lattice_constant.py` — structure optimization helpers
- `get_phonon_NaCl.py`, `get_phonon_HfO2.py` — phonon calculations via ASE/Phonopy
- `plot.py` — result visualization
- `trainer_modify.py` — experimental trainer patches
