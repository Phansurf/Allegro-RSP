from ase.io import iread, write
from ase import Atoms
import random
import numpy as np
import matplotlib.pyplot as plt
import os

# %% read extxyz file
full_structures = []
defect_structures = []
full_energy = []
defect_energy = []
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

    if len(atoms) % 12 == 0 and energy > -2000:
        full_structures.append(processed_atoms)
        full_energy.append(energy)
    elif energy > -2000:
        defect_structures.append(processed_atoms)
        defect_energy.append(energy)

full_energy = np.array(full_energy)
defect_energy = np.array(defect_energy)

# %% full data distribution
min_e = np.min(full_energy)
max_e = np.max(full_energy)
num_bins = 20
hist, bin_edges = np.histogram(full_energy, bins=num_bins, range=(min_e, max_e))
hist_replaced = np.where(hist == 0, 1, hist)

# plot figure
plt.figure(figsize=(10, 6))
bars = plt.bar(bin_edges[:-1], hist, width=np.diff(bin_edges),
               edgecolor='k', align='edge')
plt.xlabel('Total Energy (eV)', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.title('Total Energy Distribution (full)', fontsize=14)
# display results above bars
# for bar, count in zip(bars, hist):
#     height = bar.get_height()
#     plt.text(bar.get_x() + bar.get_width() / 2.,  # middle x position
#              height + 0.02 * max(hist),  # 2% height above y position
#              f'{count:.3e}',
#              ha='center', va='bottom',
#              fontsize=5)
# optimize display
plt.ylim(0, max(hist) * 1.15)
plt.grid(axis='y', alpha=0.5)
plt.tight_layout()
plt.savefig(f'./data/energy_distribution_full.png', dpi=300)
plt.show()

# %% defect data distribution
os.makedirs("../test_HfO2-defect/data", exist_ok=True)
min_e = np.min(defect_energy)
max_e = np.max(defect_energy)
num_bins = 20
hist, bin_edges = np.histogram(defect_energy, bins=num_bins, range=(min_e, max_e))
# plot figure
plt.figure(figsize=(10, 6))
bars = plt.bar(bin_edges[:-1], hist, width=np.diff(bin_edges),
               edgecolor='k', align='edge')
plt.xlabel('Total Energy (eV)', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.title('Total Energy Distribution (defect)', fontsize=14)
# display results above bars
# for bar, count in zip(bars, hist):
#     height = bar.get_height()
#     plt.text(bar.get_x() + bar.get_width() / 2.,  # middle x position
#              height + 0.02 * max(hist),  # 2% height above y position
#              f'{count:.3e}',
#              ha='center', va='bottom',
#              fontsize=5)
# optimize display
plt.ylim(0, max(hist) * 1.15)
plt.grid(axis='y', alpha=0.5)
plt.tight_layout()
plt.savefig(f'../test_HfO2-defect/data/energy_distribution_defect.png', dpi=300)
plt.show()

# %% write into xyz files
write(
    "HfO2.xyz",
    images=full_structures[:3000] + defect_structures[:3000],
    format="extxyz",
    write_info=True,
)

os.makedirs("../test_HfO2", exist_ok=True)
write(
    "../test_HfO2/HfO2.xyz",
    images=full_structures[3000:],
    format="extxyz",
    write_info=True,
)

os.makedirs("../test_HfO2-defect", exist_ok=True)
write(
    "../test_HfO2-defect/HfO2.xyz",
    images=defect_structures[3000:],
    format="extxyz",
    write_info=True,
)
