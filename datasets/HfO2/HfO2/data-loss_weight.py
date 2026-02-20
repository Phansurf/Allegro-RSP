from ase.io import iread, write
from ase import Atoms
import random
import numpy as np
import matplotlib.pyplot as plt
import os
from adjustText import adjust_text

# %% read extxyz file
full_structures = []
equal_defect_structures = []
inequal_defect_structures = []

full_energy = []
equal_defect_energy = []
inequal_defect_energy = []

atoms_list = []
for atoms in iread("data/HfO2.extxyz"):
    atoms_list.append(atoms)
random.shuffle(atoms_list)
for atoms in atoms_list:
    energy = atoms.get_total_energy()
    elements = atoms.get_chemical_symbols()
    processed_atoms = Atoms(
        symbols=elements,
        positions=atoms.get_positions(),
        cell=atoms.cell,
        pbc=atoms.pbc,
    )
    processed_atoms.arrays["forces"] = atoms.get_forces()
    # atomic charge
    atomic_charges = []
    for symbol in elements:
        if symbol == 'Hf':
            atomic_charges.append(4.0)
        elif symbol == 'O':
            atomic_charges.append(-2.0)
    processed_atoms.arrays['atomic_charges'] = np.array(atomic_charges)
    # total charge
    hf_count = elements.count("Hf")
    o_count = elements.count("O")
    structure_info = {}
    structure_info["stress"] = atoms.get_stress(voigt=False)
    structure_info["energy"] = atoms.get_total_energy()
    # structure_info["total_charge"] = 0.
    structure_info["total_charge"] = np.sum(atomic_charges)
    processed_atoms.info = structure_info

    if len(atoms) % 12 == 0:
        full_structures.append(processed_atoms)
        full_energy.append(energy)
    elif structure_info["total_charge"] == 0.0:
        equal_defect_structures.append(processed_atoms)
        equal_defect_energy.append(energy)
    else:
        inequal_defect_structures.append(processed_atoms)
        inequal_defect_energy.append(energy)

full_energy = np.array(full_energy)
equal_defect_energy = np.array(equal_defect_energy)
inequal_defect_energy = np.array(inequal_defect_energy)

# %% full structures
# calculate loss weights
min_e = np.min(full_energy)
max_e = np.max(full_energy)
num_bins = 100
hist, bin_edges = np.histogram(full_energy, bins=num_bins, range=(min_e, max_e))
hist_replaced = np.where(hist == 0, 1, hist)
# calculate loss weights
loss_weight = np.mean(hist) / hist_replaced
loss_weight = np.where(hist == 0, 0, loss_weight)

# plot figure
plt.figure(figsize=(10, 6))
bars = plt.bar(bin_edges[:-1], hist, width=np.diff(bin_edges), edgecolor='k', align='edge')
plt.xlabel('Total Energy (eV)', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.title('Total Energy Distribution (full)', fontsize=14)
# display results above bars
texts = []
i_bin = 0
for bar, weight in zip(bars, loss_weight):
    if weight > 0.0 and i_bin % 2 == 0:
        height = bar.get_height()
        t = plt.text(
            bar.get_x() + bar.get_width() / 2., height, f'{weight:.3f}',
            ha='center', va='bottom', fontsize=8
        )
        texts.append(t)
    i_bin += 1
# optimize display
plt.ylim(0, max(hist) * 1.15)
plt.grid(axis='y', alpha=0.5)
adjust_text(texts, only_move={'points':'y', 'text':'y'}, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))
plt.tight_layout()
plt.savefig(f'./energy_distribution_full.png', dpi=300)

bin_indices = np.digitize(full_energy, bin_edges) - 1
bin_indices = np.clip(bin_indices, 0, len(loss_weight) - 1)
loss_weight = loss_weight[bin_indices]

for i, atoms in enumerate(full_structures):
    atoms.info["loss_weight"] = loss_weight[i]

# %% equal defect structures
if equal_defect_structures:
    # calculate loss weights
    min_e = np.min(equal_defect_energy)
    max_e = np.max(equal_defect_energy)
    num_bins = 100
    hist, bin_edges = np.histogram(equal_defect_energy, bins=num_bins, range=(min_e, max_e))
    hist_replaced = np.where(hist == 0, 1, hist)
    # calculate loss weights
    loss_weight = np.mean(hist) / hist_replaced
    loss_weight = np.where(hist == 0, 0, loss_weight)

    # plot figure
    plt.figure(figsize=(10, 6))
    bars = plt.bar(bin_edges[:-1], hist, width=np.diff(bin_edges), edgecolor='k', align='edge')
    plt.xlabel('Total Energy (eV)', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title('Total Energy Distribution (equal defect)', fontsize=14)
    # display results above bars
    texts = []
    i_bin = 0
    for bar, weight in zip(bars, loss_weight):
        if weight > 0.0 and i_bin % 2 == 0:
            height = bar.get_height()
            t = plt.text(
                bar.get_x() + bar.get_width() / 2., height, f'{weight:.3f}',
                ha='center', va='bottom', fontsize=8
            )
            texts.append(t)
        i_bin += 1
    # optimize display
    plt.ylim(0, max(hist) * 1.15)
    plt.grid(axis='y', alpha=0.5)
    adjust_text(texts, only_move={'points':'y', 'text':'y'}, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))
    plt.tight_layout()
    plt.savefig(f'./energy_distribution_equal_defect.png', dpi=300)

    bin_indices = np.digitize(equal_defect_energy, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, len(loss_weight) - 1)
    loss_weight = loss_weight[bin_indices]

    for i, atoms in enumerate(equal_defect_structures):
        atoms.info["loss_weight"] = loss_weight[i]

# %% inequal defect structures
if inequal_defect_structures:
    # calculate loss weights
    min_e = np.min(inequal_defect_energy)
    max_e = np.max(inequal_defect_energy)
    num_bins = 100
    hist, bin_edges = np.histogram(inequal_defect_energy, bins=num_bins, range=(min_e, max_e))
    hist_replaced = np.where(hist == 0, 1, hist)
    # calculate loss weights
    loss_weight = np.mean(hist) / hist_replaced
    loss_weight = np.where(hist == 0, 0, loss_weight)

    # plot figure
    plt.figure(figsize=(10, 6))
    bars = plt.bar(bin_edges[:-1], hist, width=np.diff(bin_edges), edgecolor='k', align='edge')
    plt.xlabel('Total Energy (eV)', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title('Total Energy Distribution (inequal defect)', fontsize=14)
    # display results above bars
    texts = []
    i_bin = 0
    for bar, weight in zip(bars, loss_weight):
        if weight > 0.0 and i_bin % 2 == 0:
            height = bar.get_height()
            t = plt.text(
                bar.get_x() + bar.get_width() / 2., height, f'{weight:.3f}',
                ha='center', va='bottom', fontsize=8
            )
            texts.append(t)
        i_bin += 1
    # optimize display
    plt.ylim(0, max(hist) * 1.15)
    plt.grid(axis='y', alpha=0.5)
    adjust_text(texts, only_move={'points':'y', 'text':'y'}, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))
    plt.tight_layout()
    plt.savefig(f'./energy_distribution_inequal_defect.png', dpi=300)

    bin_indices = np.digitize(inequal_defect_energy, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, len(loss_weight) - 1)
    loss_weight = loss_weight[bin_indices]

    for i, atoms in enumerate(inequal_defect_structures):
        atoms.info["loss_weight"] = loss_weight[i]

# %% write into xyz files
num_full = len(full_structures)
num_equal_defect = len(equal_defect_structures)
num_inequal_defect = len(inequal_defect_structures)
print("full:", num_full)
print("equal_defect:", num_equal_defect)
print("inequal_defect:", num_inequal_defect)

if equal_defect_structures and inequal_defect_structures:
    train_data = full_structures[:2300] + equal_defect_structures[:810] + inequal_defect_structures[:890]
    test_data = full_structures[2300:2800]
    test_defect_data = equal_defect_structures[810:1110] + inequal_defect_structures[890:]
elif equal_defect_structures:
    train_data = full_structures[:2000] + equal_defect_structures[:2000]
    test_data = full_structures[2000:2500]
    test_defect_data = equal_defect_structures[2000:2500]
elif inequal_defect_structures:
    train_data = full_structures[:2000] + inequal_defect_structures[:2000]
    test_data = full_structures[2000:2500]
    test_defect_data = inequal_defect_structures[2000:2500]
else:
    train_data = full_structures[:4000]
    test_data = full_structures[4000:4500]

# train
write(
    "HfO2.xyz",
    images=train_data,
    format="extxyz",
    write_info=True,
)

# test
os.makedirs("../test_HfO2", exist_ok=True)
write(
    "../test_HfO2/HfO2.xyz",
    images=test_data,
    format="extxyz",
    write_info=True,
)
os.makedirs("../test_HfO2-defect", exist_ok=True)
write(
    "../test_HfO2-defect/HfO2.xyz",
    images=test_defect_data,
    format="extxyz",
    write_info=True,
)
