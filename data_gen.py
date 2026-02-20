# !/usr/bin/env python3
# --- coding: utf-8 ---

from ase.io import write
from ase import Atoms
import numpy as np
from pymatgen.io.vasp import Poscar
from pymatgen.core import Lattice, Structure, Species
import os
import multiprocessing as mp
from tqdm import tqdm


nacl_expand = [
    [2, 0, 0], [0, 2, 0], [0, 0, 2]
]
nacl_defect = [
    [[2, 0, 0], [0, 2, 0], [0, 0, 2]], 2
]

r_max = 6.0
k_max = 7.0
alpha = 0.6

a_start = 5.05
a_end = 5.15
a_extend_start = 4.8
a_extend_end = 5.8
n_samples = 101


def expand(units: Structure, scale, material: str = None, save_file: str = None):
    """
    Provide material: generate structure for calculating phonon spectrum.
    Provide save_path: generate structure for predicting lattice constant.
    """
    expand = units.copy()  # copy the structure from input
    expand.make_supercell(scale)
    structure = expand.copy()
    # elements
    chemical_symbols = [site.specie.symbol for site in structure]

    # write .xyz file
    atoms = Atoms(
        symbols=chemical_symbols,
        positions=structure.cart_coords,
        cell=structure.lattice.matrix,
        pbc=structure.pbc,
    )

    if material is not None:
        save_path = f"results/reciprocal/{material}/test_{material}"
        write(
            f"{save_path}/original_frame_{material}.xyz",
            images=atoms,
            format="extxyz",
            write_info=True,
        )
        pos_file = Poscar(structure)
        os.system(f"mkdir -p {save_path}/VASP")
        pos_file.write_file(f"{save_path}/VASP/POSCAR")
        # unit cell
        unit_file = Poscar(units)
        unit_file.write_file(f"{save_path}/VASP/POSCAR-unitcell")
    elif save_file is not None:
        write(
            save_file,
            images=atoms,
            format="extxyz",
            write_info=True,
        )


def expand_remove(units: Structure, scale, num_remove: int, material: str = None, save_file: str = None):
    # expand
    expand = units.copy()  # copy the structure from input
    expand.make_supercell(scale)
    structure = expand.copy()
    # defect
    all_indices = list(range(len(structure)))
    indices_to_remove = np.random.choice(all_indices, size=num_remove, replace=False)
    for idx in sorted(indices_to_remove, reverse=True):
        structure.remove_sites([idx])
    # elements
    chemical_symbols = [site.specie.symbol for site in structure]

    # write .xyz file
    atoms = Atoms(
        symbols=chemical_symbols,
        positions=structure.cart_coords,
        cell=structure.lattice.matrix,
        pbc=structure.pbc,
    )

    if material is not None:
        # structure for calculating phonon spectrum
        save_path = f"results/reciprocal/{material}/test_{material}-defect"
        write(
            f"{save_path}/original_frame_{material}-defect.xyz",
            images=atoms,
            format="extxyz",
            write_info=True,
        )
        pos_file = Poscar(structure)
        os.system(f"mkdir -p {save_path}/VASP")
        pos_file.write_file(f"{save_path}/VASP/POSCAR")
        # unit cell
        unit_file = Poscar(units)
        unit_file.write_file(f"{save_path}/VASP/POSCAR-unitcell")
    elif save_file is not None:
        # structure for lattice constant prediction
        write(
            save_file,
            images=atoms,
            format="extxyz",
            write_info=True,
        )


def worker(args):
    a, species, coords, site_properties = args
    lattice = Lattice.from_parameters(
        a=a, b=a, c=a, alpha=90, beta=90, gamma=90
    )
    nacl_unit_cell = Structure(
        lattice=lattice,
        species=species,
        coords=coords,
        site_properties=site_properties,
    )
    save_file = f"results/reciprocal/NaCl/lattice_constant/data/{a:.3f}.xyz"
    expand(nacl_unit_cell, nacl_expand, save_file=save_file)


if __name__ == "__main__":
    a0 = 5.1124
    a_list = np.linspace(a_start, a_end, n_samples)
    a_list_extend = np.linspace(a_extend_start, a_extend_end, n_samples)
    a_list = np.concatenate([a_list, a_list_extend])

    coords = [[0.0, 0.0, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0],
              [0.0, 0.5, 0.0], [0.0, 0.0, 0.5], [0.5, 0.5, 0.5], [0.5, 0.0, 0.0]]
    species = [Species("Na", 1), Species("Na", 1), Species("Na", 1), Species("Na", 1),
               Species("Cl", -1), Species("Cl", -1), Species("Cl", -1), Species("Cl", -1)]
    charges = [1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, ]
    site_properties = {"charge": charges}
    lattice = Lattice.from_parameters(
        a=a0, b=a0, c=a0, alpha=90, beta=90, gamma=90
    )
    nacl_unit_cell = Structure(
        lattice=lattice,
        species=species,
        coords=coords,
        site_properties=site_properties,
    )
    try:
        print("=" * 20)
        print("Generating NaCl data for phonon spectrum calculation...")
        expand(nacl_unit_cell, nacl_expand, material="NaCl")
        print("Generating NaCl-defect data for phonon spectrum calculation...")
        expand_remove(nacl_unit_cell, nacl_defect[0], nacl_defect[1], material="NaCl")
        # lattice constant
        print('-' * 20)
        print("Generating NaCl data for lattice constant prediction...")
        os.system(f"mkdir -p results/reciprocal/NaCl/lattice_constant/data")
        task_list = [
            (a, species, coords, site_properties)
            for a in a_list
        ]
        with mp.Pool(processes=mp.cpu_count()) as pool:
            results = list(tqdm(
                pool.imap(worker, task_list),
                total=len(task_list),
                desc="Lattice constant structures"
            ))

    except Exception as e:
        print(e)
        print("NaCl test data doesn't exist !")
