import os

import ase.io
import matplotlib

matplotlib.use("pgf")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset


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
}

title_style = {"fontsize": 16, "fontweight": 'bold', "pad": 10}

xlabel_style = {"fontsize": 14, "fontweight": 'bold'}
diag_line_style = {"color": color_map['black'], "linestyle": '--', "linewidth": 1.5, "alpha": 0.8, 'zorder': 1}
parity_common_style = {
    "s": 40,                  # Slightly increase dot size
    "alpha": 0.5,             # Increase opacity, combine with white edge
    # "edgecolors": 'white',  # Key: white edge
    # "linewidths": 0.6,      # Edge width
    "rasterized": True,
}
parity_legend_style = {
    "loc": "lower right", "fontsize": 10, "frameon": True, "edgecolor": 'gray', "framealpha": 0.9, "fancybox": False
}

hist_common_style = {
    "bins": 50,
    "density": True,
    "histtype": 'stepfilled',
    "alpha": 0.5,
    "linewidth": 0,
}
hist_legend_style = {
    "loc": "upper center", "fontsize": 10, "frameon": True, "edgecolor": 'gray', "framealpha": 0.9, "fancybox": False
}


def plot_parity(file_dir: str, model_cmap: dict, test_data_name: str, test_data_dict: dict, original_file_path: str, ax1, ax2) -> None:
    lims_total = [float('inf'), float('-inf')]
    lims_per_atom = [float('inf'), float('-inf')]
    # Flag to track if any model data was successfully plotted
    has_plotted_data = False

    for ax in [ax1, ax2]:
        ax.grid(True, which='major', linestyle='--', linewidth=0.7, color='gray', alpha=0.3, zorder=0)
        ax.set_title(f"{test_data_dict['title']}", **title_style)

    axins = None
    inset_settings = test_data_dict.get("inset_axes", None)
    if inset_settings:
        axins = inset_axes(ax1, bbox_transform=ax1.transAxes, **inset_settings)
        axins.patch.set_facecolor('white')
        axins.patch.set_alpha(0.9)
        axins.grid(True, which='major', linestyle='--', linewidth=0.5, color='gray', alpha=0.3, zorder=0)
        axins.xaxis.set_major_locator(MaxNLocator(integer=True, nbins='auto'))
        axins.yaxis.set_major_locator(MaxNLocator(integer=True, nbins='auto'))

    plot_data = []
    min_mae_total = float('inf')
    min_mae_per_atom = float('inf')
    for model_type, style in model_cmap.items():
        pred_file_path = f"{file_dir}/{model_type}/{model_type}_test.xyz"

        # --- Read predicted data ---
        # Check if the prediction file exists; skip if missing
        if not os.path.exists(pred_file_path):
            print(f"Warning: File not found {pred_file_path}, skipping.")
            continue

        try:
            pred_atoms = list(ase.io.iread(pred_file_path))
        except Exception as e:
            print(f"Warning: Failed to read {pred_file_path}: {e}, skipping.")
            continue

        # Skip if the file exists but contains no atomic configurations
        if not pred_atoms:
            print(f"Warning: No atoms found in {pred_file_path}, skipping.")
            continue

        # Extract predicted total energy and number of atoms
        pred_total_e = np.array([atom.get_total_energy() for atom in pred_atoms])
        n_atoms = np.array([len(atom) for atom in pred_atoms])
        pred_per_atom_e = pred_total_e / n_atoms

        # --- Read original (ground truth) data ---
        # Verify the existence of the ground truth file
        if not os.path.exists(original_file_path):
            print(f"Error: Original file not found at {original_file_path}. Skipping this model.")
            continue

        original_atoms_all = list(ase.io.iread(original_file_path))
        # Ensure indices match between predicted and original datasets
        try:
            original_indices = [atom.info["original_dataset_index"] for atom in pred_atoms]
            target_atoms = [original_atoms_all[i] for i in original_indices]
        except (KeyError, IndexError) as e:
            print(f"Error matching indices for {model_type}: {e}. Skipping.")
            continue

        target_total_e = np.array([atom.get_total_energy() for atom in target_atoms])
        # Double check the number of atoms
        target_n_atoms = np.array([len(atom) for atom in target_atoms])
        target_per_atom_e = target_total_e / target_n_atoms

        # --- Calculate Mean Absolute Error (MAE) ---
        mae_total = np.mean(np.abs(pred_total_e - target_total_e))
        mae_per_atom = np.mean(np.abs(pred_per_atom_e - target_per_atom_e))
        # Update global minimums
        if mae_total < min_mae_total:
            min_mae_total = mae_total
        if mae_per_atom < min_mae_per_atom:
            min_mae_per_atom = mae_per_atom

        # --- Sort data for consistent plotting order ---
        sort_idx = np.argsort(target_total_e)

        # Update limits for the diagonal line based on data range
        lims_total[0] = min(lims_total[0], target_total_e.min(), pred_total_e.min())
        lims_total[1] = max(lims_total[1], target_total_e.max(), pred_total_e.max())

        # Update limits for the diagonal line based on data range
        lims_per_atom[0] = min(lims_per_atom[0], target_per_atom_e.min(), pred_per_atom_e.min())
        lims_per_atom[1] = max(lims_per_atom[1], target_per_atom_e.max(), pred_per_atom_e.max())

        # Mark that valid data has been plotted for at least one model
        has_plotted_data = True

        plot_data.append({
            "target_total": target_total_e,
            "pred_total": pred_total_e,
            "target_per_atom": target_per_atom_e,
            "pred_per_atom": pred_per_atom_e,
            "sort_idx": sort_idx,
            "style": style,
            "mae_total": mae_total,
            "mae_per_atom": mae_per_atom,
        })

    for data in plot_data:
        target_total_e = data["target_total"]
        pred_total_e = data["pred_total"]
        target_per_atom_e = data["target_per_atom"]
        pred_per_atom_e = data["pred_per_atom"]
        sort_idx = data["sort_idx"]
        style = data["style"]
        mae_total = data["mae_total"]
        mae_per_atom = data["mae_per_atom"]

        # --- Plot Total Energy (ax1) ---
        total_energy_style = style.copy()
        if mae_total == min_mae_total:
            total_energy_style["label"] += r": $\boxed{\textbf{" + f"{mae_total:.3f}" + r"}}$" # Add border to highlight
        else:
            total_energy_style["label"] += r": $\textbf{" + f"{mae_total:.3f}" + r"}$"
        ax1.scatter(target_total_e[sort_idx], pred_total_e[sort_idx], **total_energy_style)

        # --- Plot Per Atom Energy (ax2) ---
        per_atom_energy_style = style.copy()
        if mae_per_atom == min_mae_per_atom:
            per_atom_energy_style["label"] += r": $\boxed{\textbf{" + f"{mae_per_atom * 1e3:.3f}" + r"}}$" # Add border to highlight
        else:
            per_atom_energy_style["label"] += r": $\textbf{" + f"{mae_per_atom * 1e3:.3f}" + r"}$"
        ax2.scatter(target_per_atom_e[sort_idx], pred_per_atom_e[sort_idx], **per_atom_energy_style)

        if axins is not None:
            axins.scatter(target_total_e[sort_idx], pred_total_e[sort_idx], **style)

    # --- Critical Check: If no data was plotted, exit early ---
    if not has_plotted_data:
        print(f"Skipping plot for {test_data_name}: No valid model data found.")
        return

    # --- Configure Total Energy Subplot ---
    ax1.set_xlabel("Target Total Energy (eV)", **xlabel_style)
    ax1.legend(title="MAE E (eV)", **parity_legend_style)

    # Draw diagonal reference line (y=x)
    # Add a small margin for better visualization
    margin_total = (lims_total[1] - lims_total[0]) * 0.05
    # Safety check to prevent errors if limits are identical
    if margin_total == 0: margin_total = 1.0

    ax1.plot(
        [lims_total[0] - margin_total, lims_total[1] + margin_total],
        [lims_total[0] - margin_total, lims_total[1] + margin_total],
        **diag_line_style
    )
    ax1.set_xlim(lims_total[0] - margin_total, lims_total[1] + margin_total)
    ax1.set_ylim(lims_total[0] - margin_total, lims_total[1] + margin_total)

    # --- Configure Per Atom Energy Subplot ---
    ax2.set_xlabel("Target Energy/Atom (eV)", **xlabel_style)
    ax2.legend(title="MAE E/N (meV)", **parity_legend_style)

    # Draw diagonal reference line (y=x)
    margin_per_atom = (lims_per_atom[1] - lims_per_atom[0]) * 0.05
    if margin_per_atom == 0: margin_per_atom = 0.1

    ax2.plot(
        [lims_per_atom[0] - margin_per_atom, lims_per_atom[1] + margin_per_atom],
        [lims_per_atom[0] - margin_per_atom, lims_per_atom[1] + margin_per_atom],
        **diag_line_style
    )
    ax2.set_xlim(lims_per_atom[0] - margin_per_atom, lims_per_atom[1] + margin_per_atom)
    ax2.set_ylim(lims_per_atom[0] - margin_per_atom, lims_per_atom[1] + margin_per_atom)

    if axins is not None:
        xlims = test_data_dict.get("axins_lim")
        ylims = test_data_dict.get("axins_lim")

        axins.set_xlim(xlims)
        axins.set_ylim(ylims)

        ins_x_min, ins_x_max = axins.get_xlim()
        axins.plot([ins_x_min, ins_x_max], [ins_x_min, ins_x_max],  **diag_line_style)

        axins.tick_params(axis='both', which='major', labelsize=8)

        mark_settings = test_data_dict.get("mark_inset")
        mark_inset(ax1, axins, **mark_settings)


def plot_charge_histogram(file_dir: str, hist_style: dict, test_data_name: str, material: str) -> None:
    """
    Plot histogram of charge distribution.
    The style is completely controlled by the hist_style dictionary.
    """
    fig, ax = plt.subplots(figsize=(6, 6))

    # Set basic chart style
    ax.grid(True, which='major', linestyle='--', linewidth=0.7, color='gray', alpha=0.3, zorder=0)
    ax.set_xlabel("Atomic Charge Feature", **xlabel_style)
    ax.set_ylabel("Density", **xlabel_style)
    ax.set_title(f"{test_data_name} Atomic Charge Feature Distribution", **title_style)

    has_plotted = False

    global_min = float('inf')
    global_max = float('-inf')

    # Iterate through models defined in hist_style
    for model_type, style in hist_style.items():
        pred_file_path = f"{file_dir}/{model_type}/{model_type}_test.xyz"

        if not os.path.exists(pred_file_path):
            continue

        # --- Read data (keep unchanged) ---
        try:
            # Quick check
            first_atom = next(ase.io.iread(pred_file_path))
            if "atomic_charges" not in first_atom.arrays:
                continue
        except:
            continue

        all_charges = []
        try:
            for atom in ase.io.iread(pred_file_path):
                all_charges.extend(atom.get_array("atomic_charges"))
        except Exception as e:
            print(f"Error reading charges for {model_type}: {e}")
            continue

        if not all_charges:
            continue
        all_charges = np.array(all_charges)
        # ---------------------------

        global_min = min(global_min, all_charges.min())
        global_max = max(global_max, all_charges.max())

        # Core modification: directly unpack the style dictionary for arguments
        # The style dictionary should contain all parameters like bins, color, label, alpha, histtype, etc.
        ax.hist(all_charges, **style, zorder=2)
        has_plotted = True

    if not has_plotted:
        plt.close(fig)
        return

    # --- Draw reference lines ---
    ref_line_kwargs = {"linestyle": "--", "color": color_map['black'], "linewidth": 1.5, "alpha": 0.8, "zorder": 3}

    if material == "NaCl":
        ax.axvline(x=1.0, label=r"Reference: $\pm$1", **ref_line_kwargs)
        ax.axvline(x=-1.0, **ref_line_kwargs)

    ax.set_xlim(global_min, global_max)

    ax.legend(**hist_legend_style)

    plt.tight_layout()
    save_path = f"{file_dir}/{material}_charge_distribution.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    print(f"Charge histogram saved to: {save_path}")
    plt.close(fig)


if __name__ == '__main__':
    # Define the mapping for test datasets
    test_data_map = {
        "NaCl": {
            "NaCl": {
                "title": "NaCl(without defect)",
                "inset_axes": {
                    "width": "50%", "height": "50%",
                    "loc": "upper left", "bbox_to_anchor": (0.05, 0.05, 0.85, 0.95)
                },
                "axins_lim": [-1360, -1260],
                "mark_inset": {"loc1": 3, "loc2": 4, "fc": "none", "ec": 'k', "lw": 1},
            },
            "NaCl-defect": {
                "title": "NaCl(with defect)",
                "inset_axes": {
                    "width": "50%", "height": "50%",
                    "loc": "upper left", "bbox_to_anchor": (0.05, 0.05, 0.85, 0.95)
                },
                "axins_lim": [-1150, -1000],
                "mark_inset": {"loc1": 3, "loc2": 4, "fc": "none", "ec": 'k', "lw": 1},
            },
            "NaCl-expand": {
                "title": "NaCl(cell expansion)",
                "inset_axes": {
                    "width": "50%", "height": "50%",
                    "loc": "upper left", "bbox_to_anchor": (0.05, 0.05, 0.85, 0.95)
                },
                "axins_lim": [-1860, -1800],
                "mark_inset": {"loc1": 3, "loc2": 4, "fc": "none", "ec": 'k', "lw": 1},
            },
        },
        "HfO2": {
            # "HfO2": {
            #     "title": "HfO2(without defect)",
            #     "inset_axes": {
            #         "width": "50%", "height": "50%",
            #         "loc": "upper left", "bbox_to_anchor": (0.05, 0.05, 0.85, 0.95)
            #     },
            #     "axins_lim": [-960, -950],
            #     "mark_inset": {"loc1": 3, "loc2": 4, "fc": "none", "ec": 'k', "lw": 1},
            # },
            "HfO2-defect": {
                "title": "HfO2(with defect)",
                "inset_axes": {
                    "width": "50%", "height": "50%",
                    "loc": "upper left", "bbox_to_anchor": (0.05, 0.05, 0.85, 0.95)
                },
                "axins_lim": [-960, -950],
                "mark_inset": {"loc1": 3, "loc2": 4, "fc": "none", "ec": 'k', "lw": 1},
            },
        },
    }

    for material, material_data_dict in test_data_map.items():
        print("=" * 50)

        n_datasets = len(material_data_dict)
        fig1, axes1 = plt.subplots(1, n_datasets, figsize=(6 * n_datasets, 6))
        fig2, axes2 = plt.subplots(1, n_datasets, figsize=(6 * n_datasets, 6))

        axes1 = np.atleast_1d(axes1)
        axes2 = np.atleast_1d(axes2)

        axes1[0].set_ylabel("Predicted Total Energy (eV)", **xlabel_style)
        axes2[0].set_ylabel("Predicted Energy/Atom (eV)", **xlabel_style)

        # Define color mapping for different models
        parity_style = {
            f"{material}(allegro)": {
                "c": color_map['green'],     # Teal
                "marker": 'o',               # Circle
                "label": 'Allegro',
                "zorder": 2,
                **parity_common_style
            },
            f"{material}(1-step)": {
                "c": color_map['blue'],       # Blue
                "marker": 'D',                # Diamond
                "label": 'Allegro + RSP (1-step)',
                "zorder": 1,
                **parity_common_style
            },
            f"{material}(3-step)": {
                "c": color_map['purple'],   # Reddish Purple
                "marker": 's',              # Square
                "label": 'Allegro + RSP (3-step)',
                "zorder": 2,
                **parity_common_style
            },
            f"{material}(LES)": {
                "c": color_map['orange'],   # Vermilion
                "marker": "^",              # Triangle
                "label": "CACE + LES",
                "zorder": 1,
                **parity_common_style
            },
        }

        hist_style = {
            f"{material}(1-step)": {
                "color": color_map['blue'],
                "edgecolor": color_map['blue'],
                "label": 'Allegro + RSP (1-step)',
                **hist_common_style
            },
            f"{material}(3-step)": {
                "color": color_map['purple'],
                "edgecolor": color_map['purple'],
                "label": 'Allegro + RSP (3-step)',
                **hist_common_style
            },
            # f"{material}(LES)": {
            #     "bins": 100,
            #     "color": color_map['purple'],
            #     "edgecolor": color_map['purple'],
            #     "label": "CACE + LES",
            #     **hist_common_style
            # },
        }

        for i, test_data in enumerate(material_data_dict):
            test_data_dict = material_data_dict[test_data]
            print(f"Processing: {material} - {test_data} ...")

            # Construct file paths
            # Assumed directory structure: results/reciprocal/NaCl/test_NaCl-expand/
            file_dir = f"results/reciprocal/{material}/test_{test_data}"
            original_file_path = f"datasets/{material}/test_{test_data}/{material}.xyz"

            # Execute the plotting function
            ax1 = axes1[i]
            ax2 = axes2[i]
            plot_parity(file_dir, parity_style, test_data, test_data_dict, original_file_path, ax1, ax2)
            plot_charge_histogram(file_dir, hist_style, test_data, material)

        # Save the figure
        # fig1.tight_layout()
        save_path1 = f"results/reciprocal/{material}/{material}_total_energy_parity.png"
        if os.path.isdir(os.path.dirname(save_path1)):
            fig1.savefig(save_path1, dpi=300, bbox_inches='tight')
            print(f"Figure saved to: {save_path1}")

        # fig2.tight_layout()
        save_path2 = f"results/reciprocal/{material}/{material}_per_atom_energy_parity.png"
        if os.path.isdir(os.path.dirname(save_path2)):
            fig2.savefig(save_path2, dpi=300, bbox_inches='tight')
            print(f"Figure saved to: {save_path2}")