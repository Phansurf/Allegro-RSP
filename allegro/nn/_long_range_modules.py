# Author: Liangliang Hong and Hongyu Yu and Ruijie Guo
import math
from typing import Tuple, Dict, List, Callable, Optional

import torch
from torch.nn import Parameter
from e3nn import o3
from e3nn.util.jit import compile_mode
from nequip.data import AtomicDataDict
from nequip.nn import GraphModuleMixin
from scipy import constants
from torch_runstats.scatter import scatter
from torch.nn.utils.rnn import pad_sequence

from allegro.nn.cutoffs import cosine_cutoff
from ._fc import ScalarMLPFunction
from .. import _keys


# %% reciprocal space NN
@compile_mode("script")
class ChargeMLP(GraphModuleMixin, torch.nn.Module):
    """
    Charge prediction model.
    """
    def __init__(
            self,
            start_stage: int,
            charge_mlp_latent_dimensions: List[int],
            charge_mlp_nonlinearity: str,
            charge_mlp_initialization: str,
            edge_invariant_field: str = AtomicDataDict.EDGE_EMBEDDING_KEY,
            node_invariant_field: str = AtomicDataDict.NODE_ATTRS_KEY,
            charge_mlp: Callable = ScalarMLPFunction,
            embed_initial_edge: bool = True,
            irreps_in: Dict[str, o3.Irreps]=None,
    ):
        super(ChargeMLP, self).__init__()
        self.register_buffer('stage', torch.as_tensor(start_stage))

        self.edge_invariant_field = edge_invariant_field
        self.node_invariant_field = node_invariant_field
        self.embed_initial_edge = embed_initial_edge

        # set up irreps
        self._init_irreps(
            irreps_in=irreps_in,
            required_irreps_in=[
                self.edge_invariant_field,
                self.node_invariant_field,
            ],
        )

        # mlp
        self.mlp = charge_mlp(
            mlp_input_dimension=(
                (
                    # Node invariants for center and neighbor (chemistry)
                    2 * self.irreps_in[self.node_invariant_field].num_irreps
                    # Plus edge invariants for the edge (radius).
                    + self.irreps_in[self.edge_invariant_field].num_irreps
                )
            ),
            mlp_output_dimension=1,
            mlp_nonlinearity=charge_mlp_nonlinearity,
            mlp_initialization=charge_mlp_initialization,
            mlp_latent_dimensions=charge_mlp_latent_dimensions,
        )

        if self.stage != 1:
            self.irreps_out.update({_keys.ATOMIC_CHARGES: o3.Irreps("1x0e")})

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        if self.stage == 1:
            return data
        else:
            data = AtomicDataDict.with_batch(data)
            edge_center = data[AtomicDataDict.EDGE_INDEX_KEY][0]
            edge_neighbor = data[AtomicDataDict.EDGE_INDEX_KEY][1]
            edge_length = data[AtomicDataDict.EDGE_LENGTH_KEY]
            edge_invariants = data[self.edge_invariant_field]
            node_invariants = data[self.node_invariant_field]

            # For the first layer, we use the input invariants:
            # The center and neighbor invariants and edge invariants
            latent_inputs_to_cat = [node_invariants[edge_center], node_invariants[edge_neighbor], edge_invariants]

            # Compute charge
            edge_charges = self.mlp(torch.cat(latent_inputs_to_cat, dim=-1))
            atomic_charges = scatter(
                edge_charges,
                edge_center,
                dim=0,
                dim_size=len(data[AtomicDataDict.POSITIONS_KEY]),
                reduce="sum",
            )
            total_charge = scatter(
                atomic_charges,
                data[AtomicDataDict.BATCH_KEY],
                dim=0,
                dim_size=len(data[AtomicDataDict.BATCH_PTR_KEY]) - 1,
                reduce="sum",
            )
            data[_keys.ATOMIC_CHARGES] = atomic_charges  # (batch_size * num_atoms, 1)
            data[_keys.TOTAL_CHARGE] = total_charge  # (batch_size, 1)

            return data


# %% reciprocal space NN
@compile_mode("script")
class ReciprocalNN(GraphModuleMixin, torch.nn.Module):
    """
    Construct the model for fitting reciprocal space energy.
    Put after the short range model.
    Analogous to EwaldSummation in pymatgen.
    """
    def __init__(
            self,
            start_stage: int,
            n_hidden: int,
            n_layers: int,
            r_max: float,
            k_max: Optional[float],
            eta: Optional[float],
            acc_factor: Optional[float],
            e3_tolerance: float,
            irreps_in: Dict[str, o3.Irreps] = None,
    ):
        """
        Initialize an instance of the neural network model
        Args:
            n_hidden (int): FCN latent dimension
            n_layers (int): number of FCN layers
            k_max (float): reciprocal space cutoff radius
        """
        super(ReciprocalNN, self).__init__()
        self.register_buffer('stage', torch.as_tensor(start_stage))
        self.register_buffer('r_max', torch.as_tensor(r_max))
        self.register_buffer('e3_tolerance', torch.as_tensor(e3_tolerance))
        self.CONV_FACT = 1e10 * constants.e / (4 * math.pi * constants.epsilon_0)

        # check eta and acc_factor parameter
        if eta is not None:
            _eta = eta
            _acc_val = acc_factor if acc_factor is not None else eta * r_max**2
        elif acc_factor is not None:
            _eta = acc_factor / r_max**2
            _acc_val = acc_factor
        else:
            raise ValueError("At least one of `eta` or `acc_factor` must be provided.")
        # check k_max
        if k_max is not None and k_max > 0.0:
            _k_max = k_max
        else:
            _k_max = 2 * math.sqrt(_eta) * math.sqrt(_acc_val)

        self.register_buffer('k_max', torch.as_tensor(_k_max))
        self.register_buffer('eta', torch.as_tensor(_eta))

        # FCN layers
        self.nn = torch.nn.Sequential(
            torch.nn.Linear(1, n_hidden),
            torch.nn.LayerNorm(n_hidden),
            torch.nn.SiLU(),
            *[
                torch.nn.Sequential(
                    torch.nn.Linear(n_hidden, n_hidden),
                    torch.nn.LayerNorm(n_hidden),
                    torch.nn.SiLU(),
                )
                for _ in range(n_layers)
            ],
            torch.nn.Linear(n_hidden, 1)
        )

        self.cut_off = cosine_cutoff  # cut off function

        # E_cell factor
        self.alpha = Parameter(torch.tensor(1.0), requires_grad=True)

        # initialize irreps
        irreps_out = irreps_in.copy()
        irreps_out[_keys.RECIPROCAL_ENERGY] = o3.Irreps("1x0e")
        irreps_out[_keys.ATOMIC_RECIPROCAL_ENERGY] = o3.Irreps("1x0e")
        irreps_out[AtomicDataDict.TOTAL_ENERGY_KEY] = o3.Irreps("1x0e")
        irreps_out[AtomicDataDict.PER_ATOM_ENERGY_KEY] = o3.Irreps("1x0e")
        self._init_irreps(irreps_in=irreps_in, irreps_out=irreps_out)
    
    def calculate_E_recip(
            self, lattice: torch.Tensor, positions: torch.Tensor, charges: torch.Tensor, batch: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the k-points and S(k) according to the input data
        Args:
            lattice: the lattice vectors of the target structure.  # (batch_size, 3, 3)
            positions: the coordinates of each atom.  # (batch_size * n_atoms, 3)
            charges: the charges of each atom in the unit cell.  # (batch_size * n_atoms)
            batch: the batch index of each atom. # (batch_size * n_atoms)
        """
        # n_graphs = batch.max() + 1
        # volume = torch.abs(torch.sum(lattice[:, 0] * torch.cross(lattice[:, 1], lattice[:, 2], dim=1), dim=1))  # (batch_size)
        # # get the required k-points from the reciprocal lattice space
        # rcp_lattice = torch.linalg.inv(lattice).transpose(1, 2) * (2 * torch.pi)  # (batch_size, 3, 3)
        # ks, batch_k = self.get_points_in_sphere(rcp_lattice, self.k_max, self.e3_tolerance)  # (n_k * batch_size, 3) (n_k * batch_size)
        # ks_norm = torch.linalg.norm(ks, dim=1).view(-1, 1)  # (n_k * batch_size, 1)
        # # mask_ki
        # mask_ki = (batch_k.unsqueeze(1) == batch.unsqueeze(0)).float()  # (batch_size * n_k, batch_size * n_atoms)
        # indices = torch.nonzero(mask_ki)
        # k_idx = indices[:, 0]
        # i_idx = indices[:, 1]
        # # phi(k)
        # phi_k = self.nn(ks_norm)  # (n_k * batch_size, 1)
        # # k·ri
        # krs = torch.sum(ks[k_idx] * positions[i_idx], dim=-1)  # (n_ki_valid)
        # # |qi·e^(-I·k·ri)|
        # s_k_all = charges[i_idx] * torch.exp(-1j * krs)  # (n_ki_valid)
        # s_k = torch.abs(
        #     torch.zeros(len(ks), dtype=s_k_all.dtype, device=s_k_all.device).index_add(index=k_idx, source=s_k_all, dim=0)
        # )  # (n_k * batch_size)
        # e_recip = self.CONV_FACT * phi_k.squeeze(-1) * s_k**2 / volume[batch_k]  # (n_k * batch_size)
        # E_recip = torch.zeros(
        #     n_graphs, dtype=e_recip.dtype, device=e_recip.device
        # ).scatter_add(index=batch_k, src=e_recip, dim=0).reshape(-1, 1)  # (batch_size, 1)

        n_graphs = batch.max() + 1
        volume = torch.abs(torch.sum(lattice[:, 0] * torch.cross(lattice[:, 1], lattice[:, 2], dim=1), dim=1))  # (batch_size)
        # 1. Get k-points (Flattened)
        rcp_lattice = torch.linalg.inv(lattice).transpose(1, 2) * (2 * torch.pi)
        ks, batch_k = self.get_points_in_sphere(rcp_lattice, self.k_max, self.e3_tolerance)
        # 2. Compute phi(k) on flattened ks (Element-wise, cheap)
        ks_norm = torch.linalg.norm(ks, dim=1).view(-1, 1)
        phi_k = self.nn(ks_norm)  # (n_k_total, 1)
        # Goal: Compute s_k without creating (Total_K, Total_Atoms) matrix
        # A. Prepare counts for splitting
        # Ensure minlength to handle empty batches correctly
        atom_counts: List[int] = torch.bincount(batch, minlength=n_graphs).tolist()
        k_counts: List[int] = torch.bincount(batch_k, minlength=n_graphs).tolist()
        # B. Pad Atoms & Charges -> (Batch, Max_Atoms, ...)
        # split requires tensor to be CPU for the section sizes usually, but list works
        pos_list = torch.split(positions, atom_counts)
        q_list = torch.split(charges, atom_counts)
        # pad_sequence stacks them: (Batch, Max_Atoms, 3)
        pos_padded = pad_sequence(pos_list, batch_first=True)
        # charges padded with 0 is perfect (0 charge = no contribution)
        q_padded = pad_sequence(q_list, batch_first=True, padding_value=0.0)
        # C. Pad K-points -> (Batch, Max_K, 3)
        k_list = torch.split(ks, k_counts)
        k_padded = pad_sequence(k_list, batch_first=True)
        # D. Batch Matmul
        # (B, Max_K, 3) @ (B, 3, Max_A) -> (B, Max_K, Max_A)
        # This only computes interactions WITHIN the same batch!
        krs_batch = torch.matmul(k_padded, pos_padded.transpose(1, 2))
        # E. Compute Structure Factor S(k) per batch
        # q: (B, 1, Max_A) * exp(...): (B, Max_K, Max_A) -> Sum over Atoms -> (B, Max_K)
        s_k_batch = torch.sum(q_padded.unsqueeze(1) * torch.exp(-1j * krs_batch), dim=-1)
        s_k_batch = torch.abs(s_k_batch)  # Take magnitude
        # F. Flatten back to (Total_K) to match original format
        # We need to remove the padded k-points
        mask_k = torch.arange(s_k_batch.shape[1], device=s_k_batch.device).unsqueeze(0) < torch.tensor(k_counts, device=s_k_batch.device).unsqueeze(1)
        s_k = s_k_batch[mask_k]
        # Calculate Energy
        # s_k is now (Total_K), matching phi_k and volume[batch_k]
        e_recip = self.CONV_FACT * phi_k.squeeze(-1) * s_k ** 2 / volume[batch_k]
        E_recip = torch.zeros(
            n_graphs, dtype=e_recip.dtype, device=e_recip.device
        ).scatter_add(index=batch_k, src=e_recip, dim=0).reshape(-1, 1)

        return E_recip, volume.reshape(-1, 1)  # (batch_size, 1) (batch_size, 1)

    @staticmethod
    def get_points_in_sphere(lattice: torch.Tensor, rcut: torch.Tensor, eps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Similar to pymatgen implementation
        """
        batch_size = lattice.shape[0]
        device = lattice.device
        # Calculate reciprocal lengths and n_max as before
        recp_len = torch.linalg.norm(lattice, dim=2)  # (batch_size, 3)
        # Ensure rcut is broadcastable if it's a scalar or 1D tensor
        if rcut.ndim == 0:
            rcut = rcut.unsqueeze(0)
        n_max = rcut.view(-1, 1) / recp_len  # (batch_size, 3)
        mins = torch.floor(-n_max).long()  # (batch_size, 3)
        maxes = torch.ceil(n_max).long()  # (batch_size, 3)
        # 1. Determine the maximum range needed across the entire batch
        # We need to cover the widest range from min to max across all batches.
        # Note: The original code used arange(min, max), which excludes max.
        # We find the largest magnitude index we need to represent.
        # e.g., if one batch needs [-2, 3] and another [-5, 1], we need a grid covering [-5, 3].
        global_min = mins.min(dim=0)[0]  # (3)
        global_max = maxes.max(dim=0)[0]  # (3)
        # 2. Generate the grid coordinates
        # We create a meshgrid covering the global extents.
        arange_x = torch.arange(global_min[0], global_max[0], device=device)
        arange_y = torch.arange(global_min[1], global_max[1], device=device)
        arange_z = torch.arange(global_min[2], global_max[2], device=device)
        # Create the 3D grid
        grid_x, grid_y, grid_z = torch.meshgrid(arange_x, arange_y, arange_z, indexing='ij')
        # Stack to get (N_points, 3)
        # This is a single grid template we will apply to every batch
        grid = torch.stack([grid_x.flatten(), grid_y.flatten(), grid_z.flatten()], dim=1)  # (N_grid_total, 3)
        # 3. Broadcast grid to batch size
        # Shape: (batch_size, N_grid_total, 3)
        batch_grid = grid.unsqueeze(0).expand(batch_size, -1, -1)
        # 4. Create a mask to select valid points for each batch
        # We need points where: mins[b] <= point < maxes[b]
        # mins shape: (batch_size, 1, 3), batch_grid shape: (batch_size, N_grid, 3)
        mins_expanded = mins.unsqueeze(1)
        maxes_expanded = maxes.unsqueeze(1)
        # Check bounds (batch_size, N_grid_total)
        valid_mask = (batch_grid >= mins_expanded) & (batch_grid < maxes_expanded)
        valid_mask = valid_mask.all(dim=2)  # Combine x, y, z checks
        # 5. Extract valid indices
        # torch.nonzero gives us indices [batch_idx, grid_idx]
        valid_indices = torch.nonzero(valid_mask)
        batch_ids = valid_indices[:, 0]  # This corresponds to 'batch_k'
        grid_indices = valid_indices[:, 1]
        # Get the actual integer coordinates (h, k, l)
        iter_tensor = batch_grid[batch_ids, grid_indices].to(lattice.dtype)  # (Total_valid_points, 3)
        # 6. Calculate real space vectors (rs)
        # lattice[batch_ids] selects the correct lattice matrix for each point
        latt_expanded = lattice[batch_ids]  # (Total_valid_points, 3, 3)
        # Matrix multiplication: (N, 1, 3) @ (N, 3, 3) -> (N, 1, 3) -> (N, 3)
        rs_raw = torch.matmul(iter_tensor.unsqueeze(1), latt_expanded).squeeze(1)
        # 7. Filter by radius (rcut) and epsilon (eps)
        rs_norm = torch.linalg.norm(rs_raw, dim=1)
        # Handle rcut broadcasting if necessary for the final mask
        if rcut.numel() > 1:
            rcut_expanded = rcut[batch_ids]
        else:
            rcut_expanded = rcut
        dist_mask = (rs_norm > eps) & (rs_norm < rcut_expanded)
        rs = rs_raw[dist_mask]
        batch_k = batch_ids[dist_mask].to(device)  # Ensure output is on correct device

        return rs, batch_k

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        if self.stage == 1:
            E_recip = torch.zeros_like(data[_keys.TOTAL_CHARGE])
            atomic_E_recip = torch.zeros_like(data[_keys.ATOMIC_SHORT_ENERGY])
            E_cell = torch.zeros_like(data[_keys.TOTAL_CHARGE])
            atomic_E_cell = torch.zeros_like(data[_keys.ATOMIC_SHORT_ENERGY])
        else:
            data = AtomicDataDict.with_batch(data)
            atomic_charges = data[_keys.ATOMIC_CHARGES].reshape(-1)  # (n_atoms)
            total_charge = data[_keys.TOTAL_CHARGE]  # (batch_size, 1)
            cell = data[AtomicDataDict.CELL_KEY].view(-1, 3, 3)  # (batch_size, 3, 3)
            batch = data[AtomicDataDict.BATCH_KEY]
            pos = data[AtomicDataDict.POSITIONS_KEY]
            # count atom numbers per batch
            num_atoms_per_batch = torch.bincount(batch)
            E_recip, volume = self.calculate_E_recip(cell, pos, atomic_charges, batch)
            E_cell = -self.CONV_FACT / 2 * math.pi / volume / self.eta * total_charge**2 * self.alpha
            atomic_E_recip = torch.repeat_interleave(
                E_recip / num_atoms_per_batch.unsqueeze(-1),
                num_atoms_per_batch,
                dim=0
            )
            atomic_E_cell = torch.repeat_interleave(
                E_cell / num_atoms_per_batch.unsqueeze(-1),
                num_atoms_per_batch,
                dim=0
            )

        data[_keys.RECIPROCAL_ENERGY] = E_recip
        data[_keys.ATOMIC_RECIPROCAL_ENERGY] = atomic_E_recip
        # short range energy
        E_short = data[_keys.SHORT_ENERGY]
        atomic_E_short = data[_keys.ATOMIC_SHORT_ENERGY]
        # total energy
        data[AtomicDataDict.TOTAL_ENERGY_KEY] = E_short + E_recip + E_cell
        data[AtomicDataDict.PER_ATOM_ENERGY_KEY] = atomic_E_short + atomic_E_recip + atomic_E_cell

        return data
