import json
import os

import matplotlib

matplotlib.use("pgf")

import matplotlib.pyplot as plt
import numpy as np
import torch
from ase.build import make_supercell
from ase.io import read
from nequip.ase import NequIPCalculator
from numpy.linalg import norm


plt.rcParams.update({
    "pgf.texsystem": "xelatex",
    "text.usetex": True,
    "pgf.rcfonts": False,

    "pgf.preamble": "\n".join([
        r"\usepackage{amsmath}",
        r"\usepackage{mathspec}",
        r"\setmainfont{Arial}",
        r"\setsansfont{Arial}",
        r"\setmathsfont(Digits,Latin){Arial}",
        r"\usepackage{bm}"
    ]),

    "font.family": "sans-serif",
    "font.size": 14,
    "axes.linewidth": 1.5,
    "xtick.major.width": 1.5,
    "ytick.major.width": 1.5,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 5,
    "ytick.major.size": 5,
})

# --- 1. Color Map Definitions ---
color_map = {
    "green": '#009E73',
    "orange": '#D55E00',
    "blue": '#0072B2',
    "purple": '#CC79A7',
    "gray": '#333333',
    "black": '#000000',
}

# --- 2. Style Definitions ---
title_style = {"fontsize": 16, "fontweight": 'bold', "pad": 10}
xlabel_style = {"fontsize": 14, "fontweight": 'bold'}
legend_style = {
    "loc": "center left", "fontsize": 11, "frameon": True,
    "edgecolor": 'gray', "framealpha": 0.9, "fancybox": False
}

# Common style for scatter points
common_scatter_style = {
    "s": 60,              # Marker size
    "alpha": 0.5,         # Transparency
    # "linewidths": 0.5,    # Edge width
    # "edgecolors": 'white',# Edge color
    "zorder": 1,
    "rasterized": True,
}

# Mapping specific models to colors and markers (Consistent with Lattice Constant script)
style_map = {
    "allegro": {
        "c": color_map["green"],
        "marker": 'o',
        "label": "Allegro",
        **common_scatter_style
    },
    "1-step": {
        "c": color_map["blue"],
        "marker": 'D',
        "label": "Allegro + RSP (1-step)",
        **common_scatter_style
    },
    "3-step": {
        "c": color_map["purple"],
        "marker": 's',
        "label": "Allegro + RSP (3-step)",
        **common_scatter_style
    },
}

# Style for auxiliary lines (Cutoff, Decay)
ref_line_style = {
    "color": color_map['gray'],
    "linestyle": '--',
    "linewidth": 1.5,
    "alpha": 0.8,
    "zorder": 5
}

decay_curve_style = {
    "color": color_map["black"],
    "linestyle": "-",
    "linewidth": 2.0,
    "alpha": 0.8,
    "zorder": 5,
    "label": 'Dipole Decay ($1/r^3$)'
}

material = "NaCl"
species_mapping = {"Na": "Na", "Cl": "Cl"}
supercell_matrix = [[5, 0, 0], [0, 5, 0], [0, 0, 5]]

displacement_distance = 0.01

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_list = ["allegro", "1-step", "3-step"]
work_dir = f"results/reciprocal/{material}/test_{material}"
unitcell_path = f"results/reciprocal/{material}/test_{material}/VASP/POSCAR-unitcell"


def calculate_force_constants_decay(unitcell_path, model_path, supercell_matrix):
    print(f"Loading model: {model_path}")
    print(f"Using device: {device}")

    unit_atoms = read(unitcell_path)
    P = np.array(supercell_matrix)
    atoms = make_supercell(unit_atoms, P)
    print(f"Supercell built, total atoms: {len(atoms)}")

    center_of_cell = np.dot(np.sum(P, axis=0) / 2, unit_atoms.cell)
    positions = atoms.get_positions()
    dists_to_center = norm(positions - center_of_cell, axis=1)
    center_idx = np.argmin(dists_to_center)
    center_symbol = atoms.get_chemical_symbols()[center_idx]
    print(f"Selected center atom: Index {center_idx}, Element {center_symbol}")

    phi_0j = np.zeros((len(atoms), 3, 3))

    print("Starting finite displacement calculations...")

    original_pos = atoms.get_positions()

    for alpha in range(3):  # x, y, z directions
        print(f"  - Calculating direction {['x', 'y', 'z'][alpha]} ...")

        # positive shift (+d)
        calc = NequIPCalculator.from_deployed_model(
            model_path=model_path,
            device=device,
            species_to_type_name=species_mapping,
            set_global_options=True
        )
        pos_plus = original_pos.copy()
        pos_plus[center_idx, alpha] += displacement_distance
        atoms.set_positions(pos_plus)
        atoms.calc = calc
        f_plus = atoms.get_forces()  # [N_atoms, 3]
        del calc

        # negative shift (-d)
        calc = NequIPCalculator.from_deployed_model(
            model_path=model_path,
            device=device,
            species_to_type_name=species_mapping,
            set_global_options=True
        )
        pos_minus = original_pos.copy()
        pos_minus[center_idx, alpha] -= displacement_distance
        atoms.set_positions(pos_minus)
        atoms.calc = calc
        f_minus = atoms.get_forces()  # [N_atoms, 3]
        del calc

        phi_0j[:, :, alpha] = - (f_plus - f_minus) / (2 * displacement_distance)

    print("Calculation finished, processing data...")

    distances = []
    fc_norms = []

    for j in range(len(atoms)):
        # Skip the center atom itself
        if j == center_idx:
            continue

        d = atoms.get_distance(center_idx, j, mic=True)

        fc_matrix = phi_0j[j]
        fc_norm = norm(fc_matrix)

        distances.append(d)
        fc_norms.append(fc_norm)

    return np.array(distances), np.array(fc_norms)


def calculate_force_decay(unitcell_path, model_path, supercell_matrix):
    print(f"Loading model: {model_path}")
    print(f"Using device: {device}")
    unit_atoms = read(unitcell_path)
    P = np.array(supercell_matrix)
    atoms = make_supercell(unit_atoms, P)  # Perfect supercell
    print(f"Supercell built, total atoms: {len(atoms)}")

    center_of_cell = np.dot(np.sum(P, axis=0) / 2, unit_atoms.cell)
    positions = atoms.get_positions()
    dists_to_center = norm(positions - center_of_cell, axis=1)
    center_idx = np.argmin(dists_to_center)
    center_symbol = atoms.get_chemical_symbols()[center_idx]
    print(f"Selected center atom: Index {center_idx}, Element {center_symbol}")

    # --- Modification Start: Replace finite displacement with defect difference method ---

    # 1. Calculate forces for the perfect supercell (Perfect Forces)
    print("Calculating forces for perfect supercell...")
    calc_perfect = NequIPCalculator.from_deployed_model(
        model_path=model_path,
        device=device,
        species_to_type_name=species_mapping,
        set_global_options=True
    )
    atoms.calc = calc_perfect
    f_perfect = atoms.get_forces()  # [N, 3]
    # Clear calculator to free GPU memory
    del calc_perfect

    # 2. Build defect supercell (Remove center atom)
    print("Building defect supercell (removing center atom) and calculating forces...")
    atoms_defect = atoms.copy()
    del atoms_defect[center_idx]  # Physically remove center atom

    calc_defect = NequIPCalculator.from_deployed_model(
        model_path=model_path,
        device=device,
        species_to_type_name=species_mapping,
        set_global_options=True
    )
    atoms_defect.calc = calc_defect
    f_defect = atoms_defect.get_forces()  # [N-1, 3]
    del calc_defect

    print("Calculation finished, processing data...")
    distances = []
    force_diff_norms = []  # Variable name changed from fc_norms to force_diff_norms

    # Iterate through all atoms in perfect supercell (skip center atom)
    for j in range(len(atoms)):
        if j == center_idx:
            continue

        # Calculate distance (in perfect lattice)
        d = atoms.get_distance(center_idx, j, mic=True)

        # Get force on this atom in perfect supercell
        f_p = f_perfect[j]

        # Get force on this atom in defect supercell (need to handle index shift)
        # If j < center_idx, index remains j in defect supercell
        # If j > center_idx, index becomes j-1 in defect supercell due to removed atom
        if j < center_idx:
            f_d = f_defect[j]
        else:
            f_d = f_defect[j - 1]

        # Calculate force difference vector (Defect - Perfect)
        # This represents the effect of removing the center atom on the force of atom j
        f_diff = f_d - f_p

        diff_norm = norm(f_diff)
        distances.append(d)
        force_diff_norms.append(diff_norm)

    return np.array(distances), np.array(force_diff_norms)


def get_r_max_from_model(model_path, device):
    """
    Attempt to extract cutoff radius r_max from TorchScript model.
    """
    try:
        # Load model (only for reading metadata, no gradients needed)
        loaded_model = torch.jit.load(model_path, map_location=device)

        # Strategy 1: Read top-level attribute directly (NequIP/Allegro standard approach)
        if hasattr(loaded_model, "r_max"):
            r_max = float(loaded_model.r_max)
            print(f"Successfully extracted r_max from model top-level attribute: {r_max} Angstrom")
            return r_max

        # Strategy 2: Recursively search all submodules (specific to your model structure)
        # Debug results show r_max hidden inside deep modules like model.func.radial_basis.basis
        # named_modules() automatically traverses all hierarchy levels
        for name, submodule in loaded_model.named_modules():
            if hasattr(submodule, "r_max"):
                # Finding the first submodule containing r_max is sufficient
                r_max = float(submodule.r_max)
                print(f"Successfully extracted r_max from submodule '{name}': {r_max} Angstrom")
                return r_max

        # Strategy 3: If all fail, return default value and warn
        print("Warning: Could not automatically extract r_max, using default value 6.0 (please verify manually)")
        return 6.0

    except Exception as e:
        print(f"Error occurred while extracting r_max: {e}")
        print("Using default value 6.0")
        return 6.0


def save_plotting_data(filepath, data_dict):
    """Save plotting data to JSON."""
    # Numpy arrays cannot be directly JSON serialized, need to convert to list
    json_ready_data = {}
    for plot_type, models_data in data_dict.items():
        json_ready_data[plot_type] = {}
        for model_name, content in models_data.items():
            json_ready_data[plot_type][model_name] = {
                "r": content["r"].tolist(),
                "fc": content["fc"].tolist(),
                "r_max": content.get("r_max", None)
            }

    with open(filepath, 'w') as f:
        json.dump(json_ready_data, f, indent=4)
    print(f"Plotting data saved to: {filepath}")


def load_plotting_data(filepath):
    """Load plotting data from JSON."""
    with open(filepath, 'r') as f:
        raw_data = json.load(f)

    # Convert lists back to numpy arrays
    restored_data = {}
    for plot_type, models_data in raw_data.items():
        restored_data[plot_type] = {}
        for model_name, content in models_data.items():
            restored_data[plot_type][model_name] = {
                "r": np.array(content["r"]),
                "fc": np.array(content["fc"]),
                "r_max": content.get("r_max", None)
            }
    print(f"Loaded existing plotting data: {filepath}")
    return restored_data


if __name__ == "__main__":
    data_save_path = f"{work_dir}/{material}_decay_data.json"

    # --- Stage 1: Data Acquisition (Load or Calculate) ---
    if os.path.exists(data_save_path):
        # 1. If file exists, load directly
        plot_data = load_plotting_data(data_save_path)
    else:
        # 2. If file does not exist, perform calculations and save
        print("Existing data not found, starting calculations...")
        plot_data = {
            "force_constant": {},
            "force_decay": {}
        }

        for model in model_list:
            model_path = f"results/reciprocal/{material}/{material}({model})/jit.pth"

            # Get r_max (get once for storage)
            # Note: If loading from file, we don't need this step, just read from JSON
            current_r_max = None
            if model == "allegro" or True:  # Try getting for all models, or set condition
                try:
                    current_r_max = get_r_max_from_model(model_path, device)
                except:
                    current_r_max = None

            # --- Calculation 1: Force Constant Decay ---
            r_fc, val_fc = calculate_force_constants_decay(unitcell_path, model_path, supercell_matrix)
            plot_data["force_constant"][model] = {
                "r": r_fc,
                "fc": val_fc,
                "r_max": current_r_max
            }

            # --- Calculation 2: Defect Force Decay ---
            r_fd, val_fd = calculate_force_decay(unitcell_path, model_path, supercell_matrix)
            plot_data["force_decay"][model] = {
                "r": r_fd,
                "fc": val_fd,
                "r_max": current_r_max
            }

        # Save calculation results
        save_plotting_data(data_save_path, plot_data)

    # --- Stage 2: Plotting (Pure plotting logic, no calculations) ---

    # Plot 1: Force Constant Decay
    output_path = f"{work_dir}/{material}_force_constant.png"
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.grid(True, which='major', linestyle='--', linewidth=0.7, color='gray', alpha=0.3, zorder=0)

    # Iterate through data dictionary instead of directly through model list for safety
    # However, to maintain legend order, we still iterate through model_list
    for model in model_list:
        if model not in plot_data["force_constant"]: continue

        data = plot_data["force_constant"][model]
        r, fc = data["r"], data["fc"]
        r_max = data.get("r_max")

        if model == "allegro" and r_max is not None:
            ax.axvline(x=r_max, **ref_line_style)
            ax.text(
                r_max * 1.05,
                np.min(fc) * 1.5,
                f'$r_{{cutoff}}$ = {r_max} ' + r'\AA{}',
                fontsize=12, color=color_map['gray'], verticalalignment='bottom'
            )

            # 1/r^3 Reference Line logic
            valid_mask = r < r_max
            inner_indices = np.where(valid_mask)[0]
            if len(inner_indices) > 0:
                ref_idx = inner_indices[np.argmax(r[inner_indices])]
                r_ref = r[ref_idx]
                fc_ref = fc[ref_idx]
                x_plot = np.linspace(min(r), max(r), 100)
                y_plot = fc_ref * (x_plot / r_ref) ** (-3)
                ax.plot(x_plot, y_plot, **decay_curve_style)

        ax.scatter(r, fc, **style_map[model])

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Distance (\AA{})', **xlabel_style)
    ax.set_ylabel('Force Constant Norm (eV/\AA{}$^2$)', **xlabel_style)
    ax.set_title(f'{material} Force Constant', **title_style)
    ax.legend(**legend_style)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved figure: {output_path}")

    # Plot 2: Defect Interaction Force Decay
    output_path = f"{work_dir}/{material}_force_decay.png"
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.grid(True, which='major', linestyle='--', linewidth=0.7, color='gray', alpha=0.3, zorder=0)

    for model in model_list:
        if model not in plot_data["force_decay"]: continue

        data = plot_data["force_decay"][model]
        r, fc = data["r"], data["fc"]
        r_max = data.get("r_max")

        ax.scatter(r, fc, **style_map[model])

        if model == "allegro" and r_max is not None:
            ax.axvline(x=r_max, **ref_line_style)
            ax.text(
                r_max * 1.05,
                np.min(fc) * 1.5,
                f'$r_{{cutoff}}$ = {r_max} ' + r'\AA{}',
                fontsize=12, color=color_map['gray'], verticalalignment='bottom'
            )

            # 1/r^3 Reference Line logic
            valid_mask = r < r_max
            inner_indices = np.where(valid_mask)[0]
            if len(inner_indices) > 0:
                ref_idx = inner_indices[np.argmax(r[inner_indices])]
                r_ref = r[ref_idx]
                fc_ref = fc[ref_idx]
                x_plot = np.linspace(min(r), max(r), 100)
                y_plot = fc_ref * (x_plot / r_ref) ** (-3)
                ax.plot(x_plot, y_plot, **decay_curve_style)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Distance (\AA{})', **xlabel_style)
    ax.set_ylabel('Force Difference Norm (eV/\AA{})', **xlabel_style)
    ax.set_title(f'{material} Defect Force Decay', **title_style)
    ax.legend(**legend_style)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved figure: {output_path}")