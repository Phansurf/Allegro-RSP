import random

from ase.io import read, write

# %% read files
# full structure
perturb_file = "perturb.xyz"
perturb_structures = []
perturb_atoms = read(perturb_file, index=":")
for atoms in perturb_atoms:
    perturb_structures.append(atoms)

# defect structure
defect_file = "defect.xyz"
defect_structures = []
defect_atoms = read(defect_file, index=":")
for atoms in defect_atoms:
    defect_structures.append(atoms)

# %% write into xyz file
structures = perturb_structures + defect_structures
random.shuffle(structures)
write(
    "NaCl.xyz",
    images=structures,
    format="extxyz",
    write_info=True,
)