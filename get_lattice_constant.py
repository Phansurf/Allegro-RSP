#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lattice Constant Prediction via Energy Minimization
"""
import glob
import json
import os

import matplotlib

matplotlib.use("pgf")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # Added for CSV handling
from ase.io import read
from joblib import Parallel, delayed
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from nequip.ase import NequIPCalculator
from pymatgen.io.ase import AseAtomsAdaptor
from scipy.interpolate import UnivariateSpline
from scipy.optimize import minimize_scalar
from tqdm import tqdm

# from pymatgen.analysis.ewald import EwaldSummation
from allegro.nn._strided import EwaldSummation, LennardJonesSummation


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

color_map = {
    "green": '#009E73',
    "orange": '#D55E00',
    "blue": '#0072B2',
    "purple": '#CC79A7',
    "gray": '#333333',
    "black": '#000000',
    "red":   '#D55E00', # Mapping red to orange/vermilion if needed, or keeping distinct
}

title_style = {"fontsize": 16, "fontweight": 'bold', "pad": 10}
xlabel_style = {"fontsize": 14, "fontweight": 'bold'}
legend_style = {
    "loc": "lower right", "fontsize": 10, "frameon": True, "edgecolor": 'gray', "framealpha": 0.9, "fancybox": False
}

common_curve_style_dict = {
    "linewidth": 1.5,
    "linestyle": "-",
}
common_marker_style_dict = {
    "s": 80,               # Marker size
    "alpha": 1.0,
    # "facecolors": 'white', # Hollow center
    "zorder": 10           # Ensure markers are on top
}

curve_style_dict = {
    "Ewald+LJ": {
        "color": color_map["black"],
        "label": "Reference (Ewald+LJ)",
        "zorder": 5,
        **common_curve_style_dict
    },
    "allegro": {
        "color": color_map["green"],
        "label": "Allegro",
        "zorder": 2,
        **common_curve_style_dict
    },
    "1-step": {
        "color": color_map["blue"],
        "label": "Allegro + RSP (1-step)",
        "zorder": 4,
        **common_curve_style_dict
    },
    "3-step": {
        "color": color_map["purple"],
        "label": "Allegro + RSP (3-step)",
        "zorder": 3,
        **common_curve_style_dict
    },
}
marker_style_dict = {
    "Ewald+LJ": {
        "c": color_map["black"],
        "marker": "*",
        **common_marker_style_dict
    },
    "allegro": {
        "c": color_map['green'],
        "marker": 'o',
        **common_marker_style_dict
    },
    "1-step": {
        "c": color_map['blue'],
        "marker": 'D',
        **common_marker_style_dict
    },
    "3-step": {
        "c": color_map['purple'],
        "marker": 's',
        **common_marker_style_dict
    },
}
zoom_style_dict = {
    "enable": True,
    "ax_kwargs": {
        "width": "50%",        # Width relative to parent axes
        "height": "50%",       # Height relative to parent axes
        "loc": "upper left", # Location code
        "bbox_to_anchor": (0.1, 0.05, 1.0, 0.95), # Fine-tune position (x, y, w, h)
        "bbox_transform": None # Will be set to ax.transAxes inside function
    },
    # Zoom window limits (Customize for NaCl equilibrium region)
    "xlim": (5.07, 5.15),
    "ylim": (-14.6, -14.45),
    # Connection lines style
    "mark_inset_kwargs": {"loc1": 3, "loc2": 4, "fc": "none", "ec": 'k', "lw": 1}
}

# ==========================================
# %% Global Configurations
# ==========================================
material = "NaCl"
model_list = ["allegro", "1-step", "3-step"]

# Ewald parameters
ewald_params = {
    "eta": 0.6,
    "real_space_cut": 10.0,
    "recip_space_cut": 12.0,
    "compute_forces": False
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

species_to_type_name = {
    "Na": "Na",
    "Cl": "Cl"
}

# CPU cores for parallel Ewald
n_jobs = int(os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count()))
print(f"Using {n_jobs} CPU cores for parallel Ewald calculations.")

# Device
device = "cpu"

# ==========================================
# %% Data Persistence (Save/Load)
# ==========================================
def save_results_to_file(results_dict, output_dir, material, experimental_a=None):
    """
    Save calculation results to JSON (for plotting) and CSV (for summary).
    """
    # 1. Save Plotting Data (JSON)
    json_data = {}
    summary_data = []

    for method, (a_vals, energies, a_eq, _) in results_dict.items():
        # Convert numpy arrays to lists for JSON serialization
        json_data[method] = {
            "a_values": a_vals.tolist(),
            "energies": energies.tolist(),
            "a_eq": float(a_eq) if a_eq else None
        }

        # Prepare CSV row
        row = {
            "Method": method,
            "Equilibrium_a (Å)": a_eq,
        }
        if experimental_a is not None and a_eq is not None:
            row["Deviation (%)"] = (a_eq - experimental_a) / experimental_a * 100
        else:
            row["Deviation (%)"] = None
        summary_data.append(row)

    json_path = os.path.join(output_dir, f"{material}_energy_data.json")
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=4)
    print(f"Plotting data saved to {json_path}")

    # 2. Save Summary Table (CSV)
    df = pd.DataFrame(summary_data)
    csv_path = os.path.join(output_dir, f"{material}_lattice_summary.csv")
    df.to_csv(csv_path, index=False, float_format="%.6f")
    print(f"Summary table saved to {csv_path}")


def load_results_from_file(output_dir, material):
    """
    Load results from JSON file to skip recalculation.
    Returns reconstructed results_dict (splines are re-fitted).
    """
    json_path = os.path.join(output_dir, f"{material}_energy_data.json")
    if not os.path.exists(json_path):
        return None

    print(f"Loading existing data from {json_path}...")
    with open(json_path, "r") as f:
        data = json.load(f)

    results_dict = {}
    for method, content in data.items():
        a_values = np.array(content["a_values"])
        energies = np.array(content["energies"])

        # Re-fit spline to get the function object
        # Note: We re-calculate a_eq here to ensure consistency with the spline,
        # though we could also read it from file.
        a_eq, E_eq, spline = find_equilibrium_lattice_constant(
            a_values, energies, method_name=method
        )
        results_dict[method] = (a_values, energies, a_eq, spline)

    return results_dict


# ==========================================
# %% Ewald Energy Calculation (Single)
# ==========================================
def compute_ewald_energy(atoms, charges, ewald_params):
    """
    Perform Ewald summation on single ASE atoms structure.
    Returns energy only.
    """
    struct = AseAtomsAdaptor.get_structure(atoms)
    struct.add_oxidation_state_by_element(charges)
    ewald = EwaldSummation(struct, **ewald_params)
    lj = LennardJonesSummation(
        struct,
        compute_forces=True,
        compute_stress=True,
        **lj_params
    )
    total_energy = ewald.total_energy + lj.total_energy
    return total_energy


# ==========================================
# %% Ewald Energy Calculation (Parallel)
# ==========================================
def compute_ewald_energies_parallel(atoms_list, charges, ewald_params, n_jobs):
    """
    Compute Ewald energies for a list of structures in parallel.
    """
    print(f"Computing Ewald energies for {len(atoms_list)} structures...")
    energies = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(compute_ewald_energy)(atoms, charges, ewald_params)
        for atoms in atoms_list
    )
    return np.array(energies)


# ==========================================
# %% NequIP Energy Calculation (Serial)
# ==========================================
def compute_nequip_energies(atoms_list, model_path, species_to_type_name, device="cpu"):
    """
    Compute energies using NequIP calculator (serial).
    NequIP calculator is not thread-safe, so we compute one by one.
    """
    energies = []
    for atoms in tqdm(atoms_list, desc="NequIP energy calculation"):
        calc = NequIPCalculator.from_deployed_model(
            model_path=model_path,
            species_to_type_name=species_to_type_name,
            set_global_options=True,
            device=device,
        )
        atoms_copy = atoms.copy()
        atoms_copy.calc = calc
        energy = atoms_copy.get_potential_energy()
        energies.append(energy)
        del calc

    return np.array(energies)


# ==========================================
# %% Find Equilibrium Lattice Constant
# ==========================================
def find_equilibrium_lattice_constant(a_values, energies, method_name="Unknown"):
    """
    Find the equilibrium lattice constant by fitting a spline and finding the minimum.

    Returns:
        a_eq: equilibrium lattice constant
        E_eq: energy at equilibrium
        spline: fitted spline function for plotting
    """
    # Remove any NaN or Inf values
    valid_mask = np.isfinite(energies)
    a_valid = a_values[valid_mask]
    E_valid = energies[valid_mask]

    if len(a_valid) < 10:
        print(f"Warning: Too few valid data points for {method_name}")
        return None, None, None

    # Fit a spline
    # Sort by a values first
    sort_idx = np.argsort(a_valid)
    a_sorted = a_valid[sort_idx]
    E_sorted = E_valid[sort_idx]

    try:
        # Use smoothing spline
        spline = UnivariateSpline(a_sorted, E_sorted, s=len(a_sorted) * 0.001)

        # Find minimum using scipy
        result = minimize_scalar(
            spline,
            bounds=(a_sorted.min(), a_sorted.max()),
            method='bounded'
        )

        a_eq = result.x
        E_eq = result.fun

        print(f"{method_name}: Equilibrium lattice constant = {a_eq:.6f} Å, Energy = {E_eq:.6f} eV")

        return a_eq, E_eq, spline

    except Exception as e:
        print(f"Error fitting spline for {method_name}: {e}")
        return None, None, None


# ==========================================
# %% Plot Energy per Atom vs Lattice Constant
# ==========================================
def plot_energy_per_atom(results_dict, n_atoms, output_path,
                         curve_styles, marker_styles, zoom_config=None,
                         title="Energy per Atom vs Lattice Constant",
                         best_prediction=None):
    """
    Plot energy per atom vs lattice constant with separate styles for curves and markers,
    and optional local magnification (inset).
    """
    fig, ax = plt.subplots(figsize=(6, 6))

    # Grid style
    ax.grid(True, which='major', linestyle='--', linewidth=0.7, color='gray', alpha=0.3, zorder=0)

    # --- Setup Inset Axis (if enabled) ---
    axins = None
    if zoom_config and zoom_config.get("enable", False):
        # Need to set bbox_transform to the parent axes dynamically
        ax_kwargs = zoom_config["ax_kwargs"].copy()
        if "bbox_transform" in ax_kwargs and ax_kwargs["bbox_transform"] is None:
            ax_kwargs["bbox_transform"] = ax.transAxes

        axins = inset_axes(ax, **ax_kwargs)

        # Set Grid for Inset
        axins.grid(True, which='major', linestyle='--', linewidth=0.5, color='gray', alpha=0.2)

        # Set Limits for Inset
        if "xlim" in zoom_config:
            axins.set_xlim(zoom_config["xlim"])
        if "ylim" in zoom_config:
            axins.set_ylim(zoom_config["ylim"])

        # Hide tick labels if needed (optional, keeping them for now for readability)
        axins.tick_params(labelsize=10)

    plot_order = [k for k in curve_styles.keys() if k in results_dict]
    remaining = [k for k in results_dict.keys() if k not in curve_styles]
    plot_order.extend(remaining)

    legend_handles = []

    # --- Plot Loop ---
    for method_name in plot_order:
        (a_values, energies, a_eq, spline) = results_dict[method_name]

        # Retrieve styles
        c_style = curve_styles.get(method_name)
        m_style = marker_styles.get(method_name)

        # Energy per atom
        E_per_atom = energies / n_atoms

        # 1. Plot Spline (Curve)
        if spline is not None:
            a_fine = np.linspace(a_values.min(), a_values.max(), 500)
            E_fine = spline(a_fine) / n_atoms

            # Plot on Main Axes
            if method_name == best_prediction:
                c_style["label"] += r": $\boxed{\textbf{" + f"{a_eq:.3f}" + r"}}$"
            else:
                c_style["label"] += r": $\textbf{" + f"{a_eq:.3f}" + r"}$"
            ax.plot(a_fine, E_fine, **c_style)

            # Plot on Inset Axes
            if axins is not None:
                axins.plot(a_fine, E_fine, **c_style)

        # 2. Mark Equilibrium Point & Vertical Line
        if a_eq is not None and axins is not None:
            E_eq_per_atom = spline(a_eq) / n_atoms

            # Common Vertical Line Params
            vline_kwargs = {
                "x": a_eq,
                "ymin": zoom_config["ylim"][0],
                "ymax": zoom_config["ylim"][1],
                "color": c_style.get("color", "black"),
                "linestyle": '--',
                "linewidth": 1.2,
                "alpha": 0.7,
                "zorder": c_style.get("zorder", 2) - 0.5
            }

            axins.vlines(**vline_kwargs)
            axins.scatter([a_eq], [E_eq_per_atom], **m_style)
            ax.scatter([a_eq], [E_eq_per_atom], **m_style)

        ms = np.sqrt(m_style.get('s', 80))
        custom_handle = Line2D(
            [], [],
            color=c_style['color'],
            linewidth=c_style.get('linewidth', 1.5),
            linestyle=c_style.get('linestyle', '-'),
            marker=m_style['marker'],
            markersize=ms,
            markerfacecolor=m_style['c'],
            markeredgecolor=m_style['c'],
            label=c_style['label'],
        )
        legend_handles.append(custom_handle)

    # --- Configuration ---
    ax.set_xlabel("Lattice Constant (Å)", **xlabel_style)
    ax.set_ylabel("Energy per Atom (eV)", **xlabel_style)
    ax.set_title(title, **title_style)
    ax.legend(handles=legend_handles, title="Lattice Constant (\AA{})",**legend_style)

    # --- Draw Connecting Lines for Inset ---
    if axins is not None and "mark_inset_kwargs" in zoom_config:
        mark_inset(ax, axins, **zoom_config["mark_inset_kwargs"])

    # plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved figure: {output_path}")


# ==========================================
# %% Summary Table
# ==========================================
def print_summary_table(results_dict, experimental_a=None):
    """
    Print a summary table of equilibrium lattice constants.
    """
    print("\n" + "=" * 60)
    print("SUMMARY: Predicted Equilibrium Lattice Constants")
    print("=" * 60)
    print(f"{'Method':<20} {'a_eq (Å)':<15} {'Deviation (%)':<15}")
    print("-" * 60)

    min_deviation = float("inf")
    best_model = None
    for method_name, (a_values, energies, a_eq, spline) in results_dict.items():
        if a_eq is not None:
            if experimental_a is not None:
                deviation = (a_eq - experimental_a) / experimental_a * 100
                print(f"{method_name:<20} {a_eq:<15.6f} {deviation:<15.2f}")
                if np.abs(deviation) < min_deviation and deviation != 0:
                    min_deviation = np.abs(deviation)
                    best_model = method_name
            else:
                print(f"{method_name:<20} {a_eq:<15.6f} {'N/A':<15}")
        else:
            print(f"{method_name:<20} {'Failed':<15} {'N/A':<15}")

    if experimental_a is not None:
        print("-" * 60)
        print(f"{'Experimental':<20} {experimental_a:<15.6f} {'0.00':<15}")

    print("=" * 60 + "\n")
    return best_model


# ==========================================
# %% Main
# ==========================================
if __name__ == "__main__":
    # ---------------------------------------------------------
    # 1. Setup Paths
    # ---------------------------------------------------------
    structures_dir = f"results/reciprocal/{material}/lattice_constant/data"
    output_dir = f"results/reciprocal/{material}/lattice_constant"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Working on material: {material}")
    print(f"Output directory: {output_dir}")

    # ---------------------------------------------------------
    # 2. Try Loading Existing Results
    # ---------------------------------------------------------
    results_dict = load_results_from_file(output_dir, material)

    # We need n_atoms for plotting (Energy/atom).
    # Read one file to get it, regardless of whether we loaded results or need to calculate.
    xyz_files_check = glob.glob(os.path.join(structures_dir, "*.xyz"))
    if not xyz_files_check:
        print(f"Error: No .xyz files found in {structures_dir}")
        exit(1)

    # Read the first structure just to get the number of atoms
    sample_atoms = read(xyz_files_check[0], format="extxyz")
    n_atoms = len(sample_atoms)
    print(f"Number of atoms per structure (from {os.path.basename(xyz_files_check[0])}): {n_atoms}")

    # ---------------------------------------------------------
    # 3. Calculation Loop (if no results loaded)
    # ---------------------------------------------------------
    if results_dict is None:
        print("\n" + "=" * 60)
        print("No existing results found. Starting fresh calculations...")
        print("=" * 60)

        # --- A. Load All Structures ---
        xyz_files = sorted(xyz_files_check)
        print(f"Found {len(xyz_files)} structures.")

        a_values = []
        atoms_list = []

        for xyz_file in tqdm(xyz_files, desc="Loading structures"):
            # Extract lattice constant from filename (format: {a:.3f}.xyz)
            filename = os.path.basename(xyz_file)
            try:
                a = float(filename.replace(".xyz", ""))
                a_values.append(a)
                atoms = read(xyz_file, format="extxyz")
                atoms_list.append(atoms)
            except ValueError:
                print(f"Skipping invalid filename: {filename}")

        a_values = np.array(a_values)
        print(f"Lattice constants range: {a_values.min():.4f} - {a_values.max():.4f} Å")

        results_dict = {}

        # --- B. Ewald Summation (Ground Truth) ---
        print("\n--> Computing Ewald energies (Parallel)...")
        ewald_energies = compute_ewald_energies_parallel(
            atoms_list, charges, ewald_params, n_jobs
        )
        a_eq_ewald, E_eq_ewald, spline_ewald = find_equilibrium_lattice_constant(
            a_values, ewald_energies, method_name="Ewald+LJ"
        )
        results_dict["Ewald+LJ"] = (a_values, ewald_energies, a_eq_ewald, spline_ewald)

        # --- C. NequIP/Allegro Models ---
        for model_name in model_list:
            print(f"\n--> Computing {model_name} energies...")

            # Construct model path
            model_path = f"results/reciprocal/{material}/{material}({model_name})/jit.pth"

            if not os.path.exists(model_path):
                print(f"Warning: Model file not found at {model_path}, skipping.")
                continue

            try:
                nequip_energies = compute_nequip_energies(
                    atoms_list, model_path, species_to_type_name, device=device
                )

                a_eq, E_eq, spline = find_equilibrium_lattice_constant(
                    a_values, nequip_energies, method_name=model_name
                )

                results_dict[model_name] = (a_values, nequip_energies, a_eq, spline)

            except Exception as e:
                print(f"Error computing energies for {model_name}: {e}")
                continue

        # --- D. Save Results ---
        # Get experimental (Ewald) a_eq for deviation calculation in CSV
        experimental_a = results_dict.get("Ewald+LJ", (None, None, None, None))[2]
        save_results_to_file(results_dict, output_dir, material, experimental_a)

    else:
        print("\n" + "=" * 60)
        print("Loaded results from file. Skipping calculations.")
        print("=" * 60)

    # ---------------------------------------------------------
    # 4. Plotting & Summary
    # ---------------------------------------------------------
    print("\nGenerating plots and summary...")

    # Print Summary to Console (CSV is already saved)
    # Extract 'experimental' value again for the print function
    experimental_a = results_dict.get("Ewald+LJ", (None, None, None, None))[2]
    best_prediction = print_summary_table(results_dict, experimental_a=experimental_a)

    # Define Output Path
    plot_output_path = os.path.join(output_dir, f"{material}_lattice_constant.png")
    # Plot
    plot_energy_per_atom(
        results_dict,
        n_atoms,
        plot_output_path,
        curve_styles=curve_style_dict,
        marker_styles=marker_style_dict,
        zoom_config=zoom_style_dict,
        title=f"{material} Lattice Constant Prediction",
        best_prediction=best_prediction,
    )

    print("\nDone!")