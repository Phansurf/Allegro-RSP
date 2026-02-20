# !/usr/bin/env python3
# --- coding: utf-8 ---
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import json
import multiprocessing as mp
import shutil
from multiprocessing import Pool

import numpy as np
from ase import Atoms
from ase.io import write
from pymatgen.core import Lattice, Structure, Species
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from tqdm import tqdm

from allegro.nn._strided import EwaldSummation, LennardJonesSummation

scale = [
    [[2, 0, 0], [0, 2, 0], [0, 0, 2]],
    [[1, 0, 0], [0, 2, 0], [0, 0, 3]],
    [[2, 0, 0], [0, 4, 0], [0, 0, 1]],
    [[5, 0, 0], [0, 1, 0], [0, 0, 2]],
    [[2, 0, 0], [0, 6, 0], [0, 0, 1]],
    [[2, 0, 0], [0, 7, 0], [0, 0, 1]],
]
# [n_perturb]
num = [
    10,
    10,
    10,
    10,
    10,
    10,
]
# r_max_ref = 6.0
# k_max_ref = 7.0
# alpha_ref = 0.6
r_max_ref = 8.0
k_max_ref = 10.0
alpha_ref = 0.6

lj_params_file = "../NaCl/NaCl_VASP/lj_params.json"
lj_params = json.load(open(lj_params_file))
energy_shift = lj_params.pop("energy_shift")
lj_params.pop("final_loss", None)
lj_params.pop("weights", None)

# Lattice Scan Parameters (from poscar.py)
a_start = 5.0
a_end = 6.0
n_samples = 11
cif_file = "../NaCl/NaCl_VASP/NaCl_mp-22862_primitive.cif"

# Perturbation Parameters
d_min = 0.0
d_max = 0.5
min_dist_check = 2.5
max_retries = 100


def expand_perturb(args):
    unit = args['unit']
    scale = args['scale']

    expand = unit.copy()  # copy the structure from input
    expand.make_supercell(scale)

    success = False
    max_min_dist_found = -1.0
    best_failed_struct = None
    for _ in range(max_retries):
        struct = expand.copy()
        struct.perturb(distance=d_max, min_distance=d_min)

        dist_matrix = struct.distance_matrix
        np.fill_diagonal(dist_matrix, float('inf'))
        current_min_dist = np.min(dist_matrix)

        if current_min_dist >= min_dist_check:
            success = True
            break

        if current_min_dist > max_min_dist_found:
            max_min_dist_found = current_min_dist
            best_failed_struct = struct

    if not success:
        # Fallback: use the best attempt we found
        if best_failed_struct is not None:
            struct = best_failed_struct
            print(f"Warning: Failed to satisfy min_dist {min_dist_check}. "
                  f"Using best attempt with min_dist={max_min_dist_found:.4f}")
        else:
            # Should theoretically not happen unless max_retries=0 or structure is empty
            print(f"Warning: Failed completely. Using unperturbed structure.")
            struct = expand.copy()

    # elements
    chemical_symbols = [site.specie.symbol for site in struct]
    # charges
    charges = np.array(struct.site_properties["charge"])

    # total_energy, forces, stress = calculate_stress(struct, k_max=k_max_ref, r_max=r_max_ref, eta=alpha_ref)
    # total_energy, _, _, _, forces, stress = from_structure(struct, k_max=k_max_ref, r_max=r_max_ref, eta=alpha_ref)
    total_energy, total_forces, total_stress = calculate(
        struct, k_max=k_max_ref, r_max=r_max_ref, eta=alpha_ref, lj_params=lj_params
    )

    # write .xyz file
    atoms = Atoms(
        symbols=chemical_symbols,
        positions=struct.cart_coords,
        cell=struct.lattice.matrix,
        pbc=struct.pbc,
    )
    atoms.arrays["forces"] = total_forces
    atoms.arrays['atomic_charges'] = charges
    structure_info = {}
    structure_info["stress"] = total_stress
    structure_info["energy"] = total_energy
    structure_info["total_charge"] = np.sum(charges)
    atoms.info = structure_info

    return atoms


def calculate(struct: Structure, k_max, r_max, eta, lj_params):
    coulomb_ewald = EwaldSummation(
        struct,
        real_space_cut=r_max,
        recip_space_cut=k_max,
        eta=eta,
        compute_forces=True,
        compute_stress=True,
    )
    coulomb_energy = coulomb_ewald.total_energy
    coulomb_forces = coulomb_ewald.forces
    coulomb_stress = coulomb_ewald.stress
    lj = LennardJonesSummation(
        struct,
        compute_forces=True,
        compute_stress=True,
        **lj_params
    )
    lj_energy = lj.total_energy
    lj_forces = lj.forces
    lj_stress = lj.stress
    total_energy = coulomb_energy + lj_energy + energy_shift
    total_forces = coulomb_forces + lj_forces
    total_stress = coulomb_stress + lj_stress

    return total_energy, total_forces, total_stress


if __name__ == "__main__":
    mp.set_start_method('spawn')
    output_file = "NaCl.xyz"

    ref_structure = Structure.from_file(cif_file)
    sga = SpacegroupAnalyzer(ref_structure)
    ref_unit_cell = sga.get_conventional_standard_structure()
    a0 = ref_unit_cell.lattice.a

    task_list = []

    coords = [[0.0, 0.0, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0],
              [0.0, 0.5, 0.0], [0.0, 0.0, 0.5], [0.5, 0.5, 0.5], [0.5, 0.0, 0.0]]
    species = [Species("Na", 1), Species("Na", 1), Species("Na", 1), Species("Na", 1),
               Species("Cl", -1), Species("Cl", -1), Species("Cl", -1), Species("Cl", -1)]
    charges = [1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, ]

    a_list = np.linspace(a_start, a_end, n_samples).tolist()
    a_list.append(a0)
    for a in a_list:
        lattice = Lattice.from_parameters(
            a=a, b=a, c=a, alpha=90, beta=90, gamma=90
        )
        unit_cell = Structure(
            lattice=lattice,
            species=species,
            coords=coords,
            site_properties={"charge": charges},
        )
        for scale_matrix, perturb_number in zip(scale, num):
            for _ in range(perturb_number):
                task_list.append({
                    "unit": unit_cell,
                    "scale": scale_matrix,
                })
    print(f"Total Perturb Tasks: {len(task_list)}")

    with Pool(processes=mp.cpu_count()) as pool:
        print(f"Running Perturbation Tasks -> Saving to {output_file} ...")
        with open(output_file, "w") as f_out:
            for atoms in tqdm(pool.imap_unordered(expand_perturb, task_list), total=len(task_list)):
                if atoms is not None:
                    write(f_out, atoms, format='extxyz')

    print("All tasks completed.")