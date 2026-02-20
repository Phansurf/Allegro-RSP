import gc
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.io import read
from joblib import Parallel, delayed
from nequip.ase import NequIPCalculator
from phonopy.api_phonopy import Phonopy
from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections
from phonopy.structure.atoms import PhonopyAtoms
from pymatgen.io.ase import AseAtomsAdaptor

from allegro.nn._strided import EwaldSummation, LennardJonesSummation

# ==========================================
# %% Global Configurations
# ==========================================
material = "NaCl"
species_to_type_name = {"Na": "Na", "Cl": "Cl"}
model_list = ["1-step", "3-step", "allegro"]
task_list = [f"{material}", f"{material}-defect"]

# Ewald parameters
ewald_params = {
    "eta": 0.6,
    "real_space_cut": 6.0,
    "recip_space_cut": 7.0,
    "compute_forces": True
}
# Ewald atomic charge
charges = {
    "Na": 1.0,
    "Cl": -1.0
}
# lj parameters
lj_params_file = "datasets/NaCl/NaCl/NaCl_VASP/lj_params.json"
lj_params = json.load(open(lj_params_file))
energy_shift = lj_params.pop("energy_shift")
lj_params.pop("final_loss", None)
lj_params.pop("weights", None)

# phonon calculation parameters
device = "cpu"
n_jobs = int(os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count()))
print(f"Using {n_jobs} CPU cores for parallel Ewald+LJ calculations.")
n_points = 101
supercell_matrix = [1, 1, 1]
# path = [
#     [[0.0, 0.0, 0.0], # Gamma
#     [0.5, 0.0, 0.0], # X
#     [0.5, 0.5, 0.0], # S
#     [0.0, 0.5, 0.0], # Y
#     [0.0, 0.0, 0.0], # Gamma
#     [0.0, 0.0, 0.5], # Z
#     [0.5, 0.0, 0.5], # U
#     [0.5, 0.5, 0.5], # R
#     [0.0, 0.5, 0.5], # T
#     [0.0, 0.0, 0.5], # Z
#     [0.0, 0.5, 0.0], # Y
#     [0.0, 0.5, 0.5], # T
#     [0.5, 0.0, 0.5], # U
#     [0.5, 0.0, 0.0], # X
#     [0.5, 0.5, 0.0], # S
#     [0.5, 0.5, 0.5]] # R
# ]
# labels = ['G','X','S','Y','G','Z','U','R','T','Z','Y','T','U','X','S','R']
path = [
    [
        [0.0, 0.0, 0.0], # Gamma
        [0.5, 0.0, 0.0], # X
        [0.5, 0.5, 0.0], # S
        [0.0, 0.5, 0.0], # Y
        [0.0, 0.0, 0.0], # Gamma
        [0.0, 0.0, 0.5],
        [0.5, 0.0, 0.5],
        [0.5, 0.5, 0.5],
        [0.0, 0.5, 0.5],
        [0.0, 0.0, 0.5],
        [0.5, 0.0, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.0],
        [0.0, 0.5, 0.5],
        [0.5, 0.5, 0.0],
        [0.5, 0.5, 0.5],
    ]
]
labels = ["$\Gamma$", "X", "S", "Y", "$\Gamma$", "Z", "U", "R", "T", "Z", "X", "U", "Y", "T", "S", "R"]


# plot parameters
n_bands_to_plot = 25
style_dict = {
    "Ewald+LJ": {
        "color": "black",
        "linewidth": 2.0,
        "alpha": 1.0,
        "zorder": 2, # 绘图层级: 最上层
    },
    "1-step": {
        "color": "blue",
        "linewidth": 1.0,
        "alpha": 1.0,
        "zorder": 1,
    },
    "3-step": {
        "color": "green",
        "linewidth": 1.0,
        "alpha": 1.0,
        "zorder": 1,
    },
    "allegro": {
        "color": "red",
        "linewidth": 1.0,
        "alpha": 1.0,
        "zorder": 1,
    },
}

# ==========================================
# %% Single Ewald summation
# ==========================================
def compute_ewald_single(atoms, charges, ewald_params):
    """
    Perform Ewald summation on single Ase atoms structure.
    """
    # ASE -> Pymatgen
    struct = AseAtomsAdaptor.get_structure(atoms)

    # 添加氧化态
    struct.add_oxidation_state_by_element(charges)

    # 计算
    ewald = EwaldSummation(struct, **ewald_params)
    lj = LennardJonesSummation(
        struct,
        compute_forces=True,
        compute_stress=True,
        **lj_params
    )

    return {
        'energy': ewald.total_energy + lj.total_energy,
        'forces': np.array(ewald.forces) + np.array(lj.forces),
    }


# ==========================================
# %% Ewald ASE Calculator
# ==========================================
class PymatgenEwaldCalculator(Calculator):
    """
    Serial Ewald calculator.
    """
    implemented_properties = ['energy', 'forces']

    def __init__(self, charges: dict, ewald_params: dict, **kwargs):
        super().__init__(**kwargs)
        self.charges = charges
        self.ewald_params = ewald_params

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        if properties is None:
            properties = ['energy']
        super().calculate(atoms, properties, system_changes)
        result = compute_ewald_single(self.atoms, self.charges, self.ewald_params)
        self.results['energy'] = result['energy']
        self.results['forces'] = result['forces']


# ==========================================
# %% Serial get_force_constants
# ==========================================
def get_force_constants_serial(structure, model_path, species_to_type_name, supercell_matrix, device="cpu"):
    print("  [Serial Phonon] Initializing Phonopy...")
    phonopy_atoms = PhonopyAtoms(
        symbols=structure.get_chemical_symbols(),
        cell=structure.get_cell(),
        scaled_positions=structure.get_scaled_positions()
    )
    phonon = Phonopy(phonopy_atoms, supercell_matrix=supercell_matrix)
    phonon.generate_displacements(distance=0.01)
    supercells = phonon.supercells_with_displacements
    print(f"  [Serial Phonon] Total displacements to calculate: {len(supercells)}")
    forces_sets = []
    for i, sc in enumerate(supercells):
        atoms_disp = Atoms(
            symbols=sc.symbols,
            cell=np.array(sc.cell, dtype=float),
            positions=np.array(sc.positions, dtype=float),
            pbc=True
        )
        if i % 10 == 0:
            print(f"  [Serial Phonon] Calculating structure {i + 1}/{len(supercells)}...")
        calc = NequIPCalculator.from_deployed_model(
            model_path=model_path,
            species_to_type_name=species_to_type_name,
            set_global_options=True,
            device=device
        )
        atoms_disp.calc = calc
        f = atoms_disp.get_forces()
        forces_sets.append(f)
        del calc
        del atoms_disp
        if i % 5 == 0:
            gc.collect()
    print("  [Serial Phonon] Producing force constants...")
    phonon.forces = forces_sets
    phonon.produce_force_constants()
    return phonon


# ==========================================
# %% Parallel get_force_constants
# ==========================================
def get_force_constants_parallel(structure, charges, ewald_params, supercell_matrix, n_jobs):
    """
    使用 phonopy 生成位移结构，并行计算力，然后构建力常数矩阵。
    """
    # 1. 将 ASE atoms 转为 PhonopyAtoms
    phonopy_atoms = PhonopyAtoms(
        symbols=structure.get_chemical_symbols(),
        cell=structure.get_cell(),
        scaled_positions=structure.get_scaled_positions()
    )

    # 2. 创建 Phonopy 对象
    phonon = Phonopy(phonopy_atoms, supercell_matrix=supercell_matrix)

    # 3. 生成位移
    phonon.generate_displacements(distance=0.01)  # 位移量 0.01 Å

    # 4. 获取所有位移结构
    supercells = phonon.supercells_with_displacements
    print(f"Total displaced structures: {len(supercells)}")

    # 5. 将 PhonopyAtoms 转回 ASE Atoms (方便调用我们的函数)
    from ase import Atoms
    displaced_atoms_list = []
    for sc in supercells:
        ase_atoms = Atoms(
            symbols=sc.symbols,
            cell=sc.cell,
            positions=sc.positions,
            pbc=True
        )
        displaced_atoms_list.append(ase_atoms)

    # 6. 并行计算所有位移结构的力
    print(f"Calculating forces in parallel with {n_jobs} workers...")
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(compute_ewald_single)(atoms, charges, ewald_params)
        for atoms in displaced_atoms_list
    )

    # 7. 提取力并设置到 phonopy
    forces_sets = [r['forces'] for r in results]
    phonon.forces = forces_sets

    # 8. 生成力常数
    print("Producing force constants...")
    phonon.produce_force_constants()

    return phonon


# ==========================================
# %% Get Reference Band Indices
# ==========================================
def get_reference_band_indices(phonon, n_bands=None):
    band_dict = phonon.get_band_structure_dict()
    frequencies = band_dict['frequencies']

    all_freqs_at_gamma = frequencies[0][0, :]
    n_total_bands = len(all_freqs_at_gamma)

    positive_band_indices = [i for i in range(n_total_bands) if all_freqs_at_gamma[i] >= -1e-6]
    positive_band_indices = sorted(positive_band_indices, key=lambda i: all_freqs_at_gamma[i])

    n_plot = n_bands if n_bands is not None else len(positive_band_indices)
    n_plot = min(n_plot, len(positive_band_indices))
    bands_to_plot = positive_band_indices[:n_plot]

    return bands_to_plot


# ==========================================
# %% Plot single model
# ==========================================
def plot_band_structure(phonon, output_path, reference_band_indices, method_name):
    band_dict = phonon.get_band_structure_dict()
    distances = band_dict['distances']
    frequencies = band_dict['frequencies']

    style = style_dict.get(method_name, {"color": "black", "linewidth": 1.0, "alpha": 1.0, "zorder": 1})
    color = style["color"]
    linewidth = style["linewidth"]
    alpha = style["alpha"]

    x_min = distances[0][0]
    x_max = distances[-1][-1]
    tick_positions = [seg[0] for seg in distances] + [distances[-1][-1]]

    fig, ax = plt.subplots(figsize=(10, 8))

    for seg_idx, (dist, freq) in enumerate(zip(distances, frequencies)):
        for plot_idx, band_idx in enumerate(reference_band_indices):
            lbl = method_name if (seg_idx == 0 and plot_idx == 0) else None
            ax.plot(dist, freq[:, band_idx], color=color, linewidth=linewidth, alpha=alpha, label=lbl)

    ax.set_xlim(x_min, x_max)
    ax.set_ylabel("Frequency (THz)", fontsize=14)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(labels, fontsize=12)

    for pos in tick_positions:
        ax.axvline(x=pos, color='gray', linestyle='-', linewidth=0.5, alpha=0.7)

    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.tick_params(axis='x', which='both', direction='in', top=True, bottom=True)
    ax.tick_params(axis='y', which='both', direction='in', left=True, right=True)

    ax.legend(loc='upper right', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


# ==========================================
# %% Plot all models combined
# ==========================================
def plot_combined_band_structure(phonon_dict, output_path, reference_band_indices):
    fig, ax = plt.subplots(figsize=(10, 8))

    first_method = True
    tick_positions = None
    x_min, x_max = None, None

    for method_name, phonon in phonon_dict.items():
        band_dict = phonon.get_band_structure_dict()
        distances = band_dict['distances']
        frequencies = band_dict['frequencies']

        style = style_dict.get(method_name, {"color": "black", "linewidth": 1.0, "alpha": 1.0, "zorder": 1})
        color = style["color"]
        linewidth = style["linewidth"]
        alpha = style["alpha"]
        zorder = style["zorder"]

        for seg_idx, (dist, freq) in enumerate(zip(distances, frequencies)):
            for plot_idx, band_idx in enumerate(reference_band_indices):
                lbl = method_name if (seg_idx == 0 and plot_idx == 0) else None
                ax.plot(dist, freq[:, band_idx],
                        color=color, linewidth=linewidth, alpha=alpha, zorder=zorder, label=lbl)

        if first_method:
            tick_positions = [seg[0] for seg in distances] + [distances[-1][-1]]
            x_min = distances[0][0]
            x_max = distances[-1][-1]
            first_method = False

    ax.set_xlim(x_min, x_max)
    ax.set_ylabel("Frequency (THz)", fontsize=14)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(labels, fontsize=12)

    for pos in tick_positions:
        ax.axvline(x=pos, color='gray', linestyle='-', linewidth=0.5, alpha=0.7)

    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.tick_params(axis='x', which='both', direction='in', top=True, bottom=True)
    ax.tick_params(axis='y', which='both', direction='in', left=True, right=True)

    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax.set_title(f"{material} Phonon Spectrum Comparison", fontsize=14)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved combined plot: {output_path}")


# ==========================================
# %% main loop
# ==========================================
if __name__ == "__main__":
    for task in task_list:
        print("=" * 100)
        # structure_path = f"results/reciprocal/{material}/test_{task}/distinct_frame_{task}.xyz"
        structure_path = f"results/reciprocal/{material}/test_{task}/original_frame_{task}.xyz"
        print(f"Loading structure from {structure_path}...")
        atoms = read(structure_path, format="extxyz")
        all_phonons = {}

        print('-' * 50)
        # Ewald reference
        print("Calculating force constants (Parallel Ewald+LJ)...")
        phonon_ewald = get_force_constants_parallel(
            structure=atoms,
            charges=charges,
            ewald_params=ewald_params,
            supercell_matrix=supercell_matrix,
            n_jobs=n_jobs
        )
        print("Calculating band structure...")
        q_points, connections = get_band_qpoints_and_path_connections(path, npoints=n_points)
        phonon_ewald.run_band_structure(q_points, path_connections=connections, labels=labels)
        all_phonons["Ewald+LJ"] = phonon_ewald
        reference_band_indices = get_reference_band_indices(phonon_ewald, n_bands=n_bands_to_plot)
        print(f"Reference band indices (from Ewald+LJ): {reference_band_indices}")
        # save
        output_yaml = f'results/reciprocal/{material}/test_{task}/band_{task}_Ewald.yaml'
        output_img = f'results/reciprocal/{material}/test_{task}/phonon_{task}_Ewald.png'
        phonon_ewald.write_yaml_band_structure(filename=output_yaml)
        print(f"Saved: {output_yaml}")
        # phonon_ewald.plot_band_structure().savefig(output_img, dpi=500)
        plot_band_structure(phonon_ewald, output_img, reference_band_indices, method_name="Ewald+LJ")
        print(f"Saved: {output_img}")

        for model in model_list:
            print('-' * 50)
            model_path = f"results/reciprocal/{material}/{material}({model})/jit.pth"
            print(f"Calculating force constants {material}({model})...")
            phonon = get_force_constants_serial(
                structure=atoms,
                model_path=model_path,
                species_to_type_name=species_to_type_name,
                supercell_matrix=supercell_matrix,
                device=device
            )
            print("Calculating band structure...")
            q_points, connections = get_band_qpoints_and_path_connections(path, npoints=n_points)
            phonon.run_band_structure(q_points, path_connections=connections, labels=labels)
            all_phonons[model] = phonon

            output_yaml = f'results/reciprocal/{material}/test_{task}/band_{task}_{material}({model}).yaml'
            phonon.write_yaml_band_structure(filename=output_yaml)
            print(f"Saved: {output_yaml}")

            output_img = f'results/reciprocal/{material}/test_{task}/phonon_{task}_{material}({model}).png'
            plot_band_structure(phonon, output_img, reference_band_indices, method_name=model)
            print(f"Saved: {output_img}")
        print('-' * 50)
        combined_output = f'results/reciprocal/{material}/test_{task}/phonon_{task}.png'
        plot_combined_band_structure(all_phonons, combined_output, reference_band_indices)
    print("Done!")