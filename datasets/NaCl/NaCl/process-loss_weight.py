import os
import random

import matplotlib.pyplot as plt
import numpy as np
from ase.io import read, write

# %% read files
# # full structure
# perturb_input_dir = "perturb"
# perturb_structures = []
# perturb_energies = []
# for file in os.listdir(perturb_input_dir):
#     if file.endswith(".xyz"):
#         path = os.path.join(perturb_input_dir, file)
#         atoms = read(path)
#         perturb_structures.append(atoms)
#         total_energy = atoms.get_total_energy()
#         perturb_energies.append(total_energy)
#
# # defect structure
# defect_input_dir = "defect"
# defect_structures = []
# defect_energies = []
# for file in os.listdir(defect_input_dir):
#     if file.endswith(".xyz"):
#         path = os.path.join(defect_input_dir, file)
#         atoms = read(path)
#         defect_structures.append(atoms)
#         total_energy = atoms.get_total_energy()
#         defect_energies.append(total_energy)
# full structure
perturb_file = "perturb.xyz"
perturb_structures = []
perturb_energies = []
perturb_atoms = read(perturb_file, index=":")
for atoms in perturb_atoms:
    perturb_structures.append(atoms)
    total_energy = atoms.get_total_energy()
    perturb_energies.append(total_energy)

# defect structure
defect_file = "defect.xyz"
defect_structures = []
defect_energies = []
defect_atoms = read(defect_file, index=":")
for atoms in defect_atoms:
    defect_structures.append(atoms)
    total_energy = atoms.get_total_energy()
    defect_energies.append(total_energy)

# %% calculate loss weights
# full structure
min_e = np.min(perturb_energies)
max_e = np.max(perturb_energies)
num_bins = 200
hist, bin_edges = np.histogram(perturb_energies, bins=num_bins, range=(min_e, max_e))
hist_replaced = np.where(hist == 0, 1, hist)
# calculate loss weights
loss_weight = np.mean(hist) / hist_replaced
loss_weight = np.where(hist == 0, 0, loss_weight)
# plot figure
plt.figure(figsize=(10, 6))
bars = plt.bar(
    bin_edges[:-1], hist, width=np.diff(bin_edges),
    edgecolor='k', align='edge'
)
plt.xlabel('Total Energy (eV)', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.title('Total Energy Distribution (full)', fontsize=14)
# display results above bars
# for bar, weight in zip(bars, loss_weight):
#     height = bar.get_height()
#     plt.text(bar.get_x() + bar.get_width() / 2.,  # middle x position
#              height + 0.02 * max(hist),  # 2% height above y position
#              f'{weight:.3e}',
#              ha='center', va='bottom',
#              fontsize=5)
# optimize display
plt.ylim(0, max(hist) * 1.15)
plt.grid(axis='y', alpha=0.5)
plt.tight_layout()
plt.savefig(f'energy_distribution_full.png', dpi=300)

# map to each frame
bin_indices = np.digitize(perturb_energies, bin_edges) - 1
bin_indices = np.clip(bin_indices, 0, len(loss_weight) - 1)
loss_weight = loss_weight[bin_indices]
for i, atoms in enumerate(perturb_structures):
    atoms.info["loss_weight"] = loss_weight[i]

# defect_structures
if defect_energies:
    min_e = np.min(defect_energies)
    max_e = np.max(defect_energies)
    num_bins = 100
    hist, bin_edges = np.histogram(defect_energies, bins=num_bins, range=(min_e, max_e))
    hist_replaced = np.where(hist == 0, 1, hist)
    # calculate loss weights
    loss_weight = np.mean(hist) / hist_replaced
    loss_weight = np.where(hist == 0, 0, loss_weight)
    # plot figure
    plt.figure(figsize=(10, 6))
    bars = plt.bar(bin_edges[:-1], hist, width=np.diff(bin_edges),
                   edgecolor='k', align='edge')
    plt.xlabel('Total Energy (eV)', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title('Total Energy Distribution (defect)', fontsize=14)
    # display results above bars
    # for bar, weight in zip(bars, loss_weight):
    #     height = bar.get_height()
    #     plt.text(bar.get_x() + bar.get_width() / 2.,  # middle x position
    #              height + 0.02 * max(hist),  # 2% height above y position
    #              f'{weight:.3e}',
    #              ha='center', va='bottom',
    #              fontsize=5)
    # optimize display
    plt.ylim(0, max(hist) * 1.15)
    plt.grid(axis='y', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f'energy_distribution_defect.png', dpi=300)
    
    # map to each frame
    bin_indices = np.digitize(defect_energies, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, len(loss_weight) - 1)
    loss_weight = loss_weight[bin_indices]
    for i, atoms in enumerate(defect_structures):
        atoms.info["loss_weight"] = loss_weight[i]

# %% write into xyz file
structures = perturb_structures + defect_structures
random.shuffle(structures)
write(
    "NaCl.xyz",
    images=structures,
    format="extxyz",
    write_info=True,
)
