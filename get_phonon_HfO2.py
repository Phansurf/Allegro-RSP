import os
import matplotlib
matplotlib.use("pgf")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from ase import Atoms
from ase.io import read
from nequip.ase import NequIPCalculator
from phonopy import Phonopy
from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections
from phonopy.structure.atoms import PhonopyAtoms
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
label_style = {"fontsize": 14, "fontweight": 'bold'}
legend_style = {
    "title": "RMSE (THz)",
    "loc": "lower right",
    "fontsize": 12,
    "frameon": True,
    "edgecolor": 'gray',
    "framealpha": 0.9,
    "fancybox": False,
}

# plot parameters
n_bands_to_plot = 20
style_dict = {
    "VASP": {
        "color": color_map["black"],
        "linewidth": 4.0,
        "alpha": 0.8,
        "zorder": 0,
        "linestyle": "-",
        "label": "Reference (VASP)",
    },
    "allegro": {
        "color": color_map["green"],
        "linewidth": 2.0,
        "alpha": 1.0,
        "zorder": 1,
        "linestyle": "-",
        "label": "Allegro",
    },
    "1-step": {
        "color": color_map["blue"],
        "linewidth": 2.0,
        "alpha": 1.0,
        "zorder": 1,
        "linestyle": ":",
        "label": "Allegro + RSP (1-step)",
    },
    "3-step": {
        "color": color_map["purple"],
        "linewidth": 2.0,
        "alpha": 1.0,
        "zorder": 1,
        "linestyle": "-",
        "label": "Allegro + RSP (3-step)",
    },
    # "LES": {
    #     "color": color_map["orange"],
    #     "linewidth": 1.5,
    #     "alpha": 1.0,
    #     "zorder": 1,
    #     "label": "CACE + LES",
    # }
}

inset_config = {
    "HfO2": {
        "title": "HfO2 Phonon Spectrum",
        "inset_axes": {
            "width": "40%",
            "height": "40%",
            "loc": "lower left",
            "bbox_to_anchor": (0.01, 0.01, 0.9, 0.9)
        },
        "xlim": (0.465, 0.575),
        "ylim": (2.5, 3.1),
        "mark_inset": {"loc1": 4, "loc2": 2, "fc": "none", "ec": '0.0', "lw": 2, "linestyle": "-"},
    },
    "HfO2-defect": {
        "title": "HfO2(defect) Phonon Spectrum",
        "inset_axes": {
            "width": "40%",
            "height": "40%",
            "loc": "lower left",
            "bbox_to_anchor": (0.01, 0.01, 0.9, 0.9)
        },
        "xlim": (0.24, 0.3),
        "ylim": (2.5, 3.2),
        "mark_inset": {"loc1": 4, "loc2": 2, "fc": "none", "ec": '0.0', "lw": 2, "linestyle": "-"},
    }
}

# ==========================================
# %% Global Configurations
# ==========================================
material = "HfO2"
species_to_type_name = {"Hf": "Hf", "O": "O"}

task = f"{material}"
# task = f"{material}-defect"
work_dir = f"results/reciprocal/{material}/test_{task}"
unitcell_path = f"{work_dir}/VASP/POSCAR-unitcell"
# output file
combined_output = f'{work_dir}/{task}_phonon.png'
# yaml files
yaml_files = {
    "VASP": os.path.join(work_dir, "VASP/band.yaml"),
    "allegro": os.path.join(work_dir, "HfO2(allegro)/band.yaml"),
    # "1-step": os.path.join(work_dir, "HfO2(1-step)/band.yaml"),
    "3-step": os.path.join(work_dir, "HfO2(3-step)/band.yaml"),
}

# phonon calculation parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
npoints = 51
distance = 0.01
# neutral defect
# supercell_scale = [2, 2, 2]
# path = [
#     [
#         [0.0, 0.0, 0.0], # Gamma
#         [0.0, 0.5, 0.0], # Z
#         [0.0, 0.5, 0.5], # D
#         [0.0, 0.0, 0.5], # B
#         [0.0, 0.0, 0.0], # Gamma
#         [-0.5, 0.0, 0.5], # A
#         [-0.5, 0.5, 0.5], # E
#         [0.0, 0.5, 0.0], # Z
#         [-0.5, 0.5, 0.0], # C2
#         [-0.5, 0.0, 0.0], # Y2
#         [0.0, 0.0, 0.0], # Gamma
#     ]
# ]
# labels = ["$\Gamma$", "Z", "D", "B", "$\Gamma$", "A", "E", "Z", "$C_2$", "$Y_2$", "$\Gamma$"]

# charged defect
supercell_scale = [2, 2, 1]
# supercell_scale = [2, 1, 1]
path = [
    [
        [0.0, 0.0, 0.0], # Gamma
        [0.5, 0.0, 0.0], # X
        [0.5, 0.5, 0.0], # S
        [0.0, 0.5, 0.0], # Y
        [0.0, 0.0, 0.0], # Gamma
        [0.0, 0.0, 0.5], # Z
        [0.5, 0.0, 0.5], # U
        [0.5, 0.5, 0.5], # R
        [0.0, 0.5, 0.5], # T
        [0.0, 0.0, 0.5], # Z
    ],
    [
        [0.5, 0.0, 0.0], # X
        [0.5, 0.0, 0.5], # U
    ],
    [
        [0.0, 0.5, 0.0], # Y
        [0.0, 0.5, 0.5], # T
    ],
    [
        [0.5, 0.5, 0.0], # S
        [0.5, 0.5, 0.5], # R
    ],
]
labels = ["$\Gamma$", "X", "S", "Y", "$\Gamma$", "Z", "U", "R", "T", "Z", "X", "U", "Y", "T", "S", "R"]


def read_band_yaml(filename):
    print(f"Reading {filename} ...")

    with open(filename, 'r') as f:
        try:
            data = yaml.load(f, Loader=yaml.CLoader)
        except AttributeError:
            data = yaml.load(f, Loader=yaml.Loader)

    all_distances = []
    all_frequencies = []

    for point in data['phonon']:
        all_distances.append(point['distance'])
        freqs = [band['frequency'] for band in point['band']]
        all_frequencies.append(freqs)

    all_distances = np.array(all_distances)
    all_frequencies = np.array(all_frequencies)

    labels_raw = data.get('labels', [])
    segment_nqpoint = data.get('segment_nqpoint', [])

    label_positions = []
    label_texts = []

    if segment_nqpoint and labels_raw:
        current_idx = 0
        label_positions.append(all_distances[0])
        label_texts.append(labels_raw[0][0])

        for i, nq in enumerate(segment_nqpoint):
            current_idx += nq
            pos = all_distances[current_idx - 1]
            label_end = labels_raw[i][1]

            if i < len(segment_nqpoint) - 1:
                label_next_start = labels_raw[i + 1][0]
                if label_end != label_next_start:
                    final_label = f"{label_end}|{label_next_start}"
                else:
                    final_label = label_end
            else:
                final_label = label_end

            label_positions.append(pos)
            label_texts.append(final_label)
    else:
        label_positions = [all_distances[0], all_distances[-1]]
        label_texts = ["Start", "End"]
        if labels_raw:
            label_texts = [labels_raw[0][0], labels_raw[-1][1]]

    formatted_labels = []
    for l in label_texts:
        if 'Gamma' in l and '$\Gamma$' not in l:
            l = l.replace('Gamma', r'\Gamma')
            l = f"${l}$"
        elif '$' not in l:
            l = f"${l}$"
        formatted_labels.append(l)

    return {
        "distances": all_distances,
        "frequencies": all_frequencies,
        "segment_nqpoint": segment_nqpoint,
        "label_positions": label_positions,
        "label_texts": formatted_labels
    }


def plot_on_ax(ax, band_data, style_dict, max_bands):
    distances = band_data["distances"]
    frequencies = band_data["frequencies"]
    segment_nqpoint = band_data["segment_nqpoint"]

    total_bands = frequencies.shape[1]

    if max_bands is not None and max_bands < total_bands:
        num_bands = max_bands
    else:
        num_bands = total_bands

    plot_kwargs = style_dict.copy()
    model_label = plot_kwargs.pop("label", None)
    for i in range(num_bands):
        band_freqs = frequencies[:, i]
        if segment_nqpoint:
            start_idx = 0
            for j, nq in enumerate(segment_nqpoint):
                end_idx = start_idx + nq
                current_label = model_label if (i == 0 and j == 0) else None
                ax.plot(distances[start_idx:end_idx],
                        band_freqs[start_idx:end_idx],
                        label=current_label,
                        **plot_kwargs)
                start_idx = end_idx
        else:
            current_label = model_label if i == 0 else None
            ax.plot(distances, band_freqs, label=current_label, **plot_kwargs)


def decorate_ax(ax, band_data):
    label_positions = band_data["label_positions"]
    label_texts = band_data["label_texts"]
    distances = band_data["distances"]

    ax.set_xlim(distances[0], distances[-1])
    ax.set_xticks(label_positions)
    ax.set_xticklabels(label_texts)

    for pos in label_positions:
        ax.axvline(x=pos, color='gray', linestyle='--', linewidth=0.5, zorder=0)

    ax.axhline(y=0, color='gray', linestyle=':', linewidth=0.5, zorder=0)

    ax.set_ylabel("Frequency (THz)", **label_style)
    ax.set_xlabel("Wave Vector", **label_style)

    ax.tick_params(axis='both', which='major', labelsize=12)


def get_force_constants(
        model_path: str,
        unitcell_path: str,
        supercell_matrix,
        distance: float,
) -> Phonopy:
    # prepare primitive unit cell for phonopy
    unitcell_ase = read(unitcell_path, format='vasp')
    unitcell_phonopy = PhonopyAtoms(
        symbols=unitcell_ase.get_chemical_symbols(),
        cell=unitcell_ase.get_cell(),
        scaled_positions=unitcell_ase.get_scaled_positions()
    )

    # make sure we are using the masses intended by the user
    unitcell_phonopy.masses = unitcell_ase.get_masses()

    # prepare supercells
    phonon = Phonopy(unitcell_phonopy, supercell_matrix)
    phonon.generate_displacements(distance)

    # compute force constant matrix
    forces = []
    for structure_ph in phonon.supercells_with_displacements:
        calculator = NequIPCalculator.from_deployed_model(
            model_path=model_path, species_to_type_name=species_to_type_name, set_global_options=True, device=device,
        )
        structure_ase = Atoms(
            cell=structure_ph.cell,
            numbers=structure_ph.numbers,
            positions=structure_ph.positions,
            pbc=True,
        )
        structure_ase.calc = calculator
        forces.append(structure_ase.get_forces().copy())
        del calculator

    phonon.forces = forces
    phonon.produce_force_constants()

    return phonon


if __name__ == "__main__":
    fig, (ax_main, ax_err) = plt.subplots(2, 1, figsize=(10, 10), sharex=True, gridspec_kw={'height_ratios': [4, 1], 'hspace': 0.05})

    loaded_data = {}
    for model_name, filepath in yaml_files.items():
        if not os.path.isfile(filepath):
            model_path = f"results/reciprocal/{material}/{material}({model_name})/jit.pth"
            phonon: Phonopy = get_force_constants(model_path, unitcell_path, supercell_scale, distance)
            qpoints, connections = get_band_qpoints_and_path_connections(path, npoints=npoints)
            phonon.run_band_structure(qpoints, path_connections=connections, labels=labels)
            phonon.write_yaml_band_structure(filename=filepath)

        loaded_data[model_name] = read_band_yaml(filepath)

    ref_data = loaded_data.get("VASP")
    err_dict = {}
    num_bands = 0

    if ref_data is not None:
        ref_freqs = ref_data["frequencies"]
        total_bands = ref_freqs.shape[1]
        num_bands = min(n_bands_to_plot, total_bands) if n_bands_to_plot else total_bands

        for model_name, data in loaded_data.items():
            if model_name == "VASP":
                continue
            pred_freqs = data["frequencies"]
            diff = ref_freqs - pred_freqs
            mae = np.mean(np.abs(diff))
            rmse = np.sqrt(np.mean(np.square(diff)))
            err_dict[model_name] = rmse

    min_err = min(err_dict.values()) if err_dict else float('inf')

    config = inset_config.get(task, None)
    axins = None
    if config:
        axins = inset_axes(ax_main, bbox_transform=ax_main.transAxes, **config["inset_axes"])
        axins.patch.set_facecolor('white')
        axins.patch.set_alpha(0.9)
        axins.grid(True, which='major', linestyle='--', linewidth=0.5, color='gray', alpha=0.3)
        axins.tick_params(labelbottom=False, labelleft=True, labelsize=10)

    # --- Plotting Loop ---
    for model_name, data in loaded_data.items():
        style = style_dict.get(model_name)
        if not style:
            continue

        # Update legend labels
        plot_style = style.copy()
        if model_name != "VASP" and model_name in err_dict:
            err = err_dict[model_name]
            if err == min_err:
                plot_style["label"] += ": " + r"$\boxed{\textbf{" + f"{err:.3f}" + r"}}$"
            else:
                plot_style["label"] += ": " + r"$\textbf{" + f"{err:.3f}" + r"}$"

        # 1. Plot phonon spectrum on main axis (ax_main)
        plot_on_ax(ax_main, data, plot_style, num_bands)

        # 2. Plot error bands on error axis (ax_err)
        if model_name != "VASP" and ref_data is not None:
            ref_f = ref_data["frequencies"]
            pred_f = data["frequencies"]
            # Calculate Mean Absolute Error per q-point (MAE along bands)
            diff = ref_f - pred_f
            mae_per_q = np.mean(np.abs(diff), axis=1)
            rmse_per_q = np.sqrt(np.mean(np.square(diff), axis=1))
            err = rmse_per_q

            distances = data["distances"]
            segment_nqpoint = data["segment_nqpoint"]

            # Error line style (remove label to avoid legend duplication)
            err_style = style.copy()
            err_style.pop("label", None)

            if segment_nqpoint:
                start_idx = 0
                for nq in segment_nqpoint:
                    end_idx = start_idx + nq
                    ax_err.plot(distances[start_idx:end_idx], err[start_idx:end_idx], **err_style)
                    start_idx = end_idx
            else:
                ax_err.plot(distances, err, **err_style)

        # 3. Plot in inset
        if axins is not None:
            ins_style = style.copy()
            ins_style.pop("label", None)
            plot_on_ax(axins, data, ins_style, num_bands)

    # --- Axis Decoration ---
    layout_ref = ref_data if ref_data else list(loaded_data.values())[0]
    label_positions = layout_ref["label_positions"]
    label_texts = layout_ref["label_texts"]
    distances = layout_ref["distances"]
    xlims = (distances[0], distances[-1])

    # 1. Decorate main axis (ax_main)
    ax_main.set_title(config.get("title", "Phonon Spectrum") if config else "Phonon Spectrum", **title_style)
    ax_main.set_ylabel("Frequency (THz)", **label_style)
    ax_main.set_xlim(xlims)
    ax_main.set_xticks(label_positions)
    ax_main.set_xticklabels([])  # Hide x-axis label text for main plot
    ax_main.legend(**legend_style)

    # Draw vertical lines and zero line
    for pos in label_positions:
        ax_main.axvline(x=pos, color='gray', linestyle='--', linewidth=0.5, zorder=0)
    ax_main.axhline(y=0, color='gray', linestyle=':', linewidth=0.5, zorder=0)
    ax_main.tick_params(axis='y', which='major', labelsize=12)

    # 2. Decorate error axis (ax_err)
    ax_err.set_ylabel(r"RMSE (THz)", **label_style)  # Label y-axis
    ax_err.set_xlabel("Wave Vector", **label_style)
    ax_err.set_xlim(xlims)
    ax_err.set_xticks(label_positions)
    ax_err.set_xticklabels(label_texts)  # Show label text only at the bottom
    ax_err.set_ylim(bottom=0)  # Error usually starts from 0

    # Draw vertical lines
    for pos in label_positions:
        ax_err.axvline(x=pos, color='gray', linestyle='--', linewidth=0.5, zorder=0)
    ax_err.tick_params(axis='both', which='major', labelsize=12)

    # 3. Decorate inset
    if axins is not None:
        axins.set_xlim(config["xlim"])
        axins.set_ylim(config["ylim"])
        axins.set_xticks([])
        axins.set_yticks([])
        mark_inset(ax_main, axins, **config["mark_inset"])

    plt.savefig(combined_output, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {combined_output}")