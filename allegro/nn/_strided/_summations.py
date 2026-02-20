"""This module provides classes for calculating the Ewald sum of a structure."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING
from warnings import warn

import numpy as np
from monty.json import MSONable
from pymatgen.analysis.ewald import compute_average_oxidation_state, EwaldSummation
from pymatgen.core.structure import Structure
from pymatgen.util.due import Doi, due
from scipy import constants
from scipy.special import erfc
from typing import Dict, Tuple, Optional

if TYPE_CHECKING:
    from typing import Any

    from typing_extensions import Self

__author__ = "Shyue Ping Ong, William Davidson Richard"
__copyright__ = "Copyright 2011, The Materials Project"
__credits__ = "Christopher Fischer"
__version__ = "1.0"
__maintainer__ = "Shyue Ping Ong"
__email__ = "shyuep@gmail.com"
__status__ = "Production"
__date__ = "Aug 1 2012"


@due.dcite(
    Doi("10.1016/0010-4655(96)00016-1"),
    description="Ewald summation techniques in perspective: a survey",
    path="pymatgen.analysis.ewald.EwaldSummation",
)
class EwaldSummation(MSONable):
    """
    Calculates the electrostatic energy of a periodic array of charges using
    the Ewald technique.

    Ref:
        Ewald summation techniques in perspective: a survey
        Abdulnour Y. Toukmaji and John A. Board Jr.
        DOI: 10.1016/0010-4655(96)00016-1
        URL: http://www.ee.duke.edu/~ayt/ewaldpaper/ewaldpaper.html

    This matrix can be used to do fast calculations of Ewald sums after species
    removal.

    E = E_recip + E_real + E_point

    Atomic units used in the code, then converted to eV.
    """

    # Converts unit of q*q/r into eV
    CONV_FACT = 1e10 * constants.e / (4 * math.pi * constants.epsilon_0)

    def __init__(
            self,
            structure: Structure,
            real_space_cut: float=None,
            recip_space_cut: float=None,
            eta: float=None,
            acc_factor: float=12.0,
            w: float=1 / 2**0.5,
            compute_forces: bool=False,
            compute_stress: bool=False,
    ):
        """Initialize and calculate the Ewald sum. Default convergence
        parameters have been specified, but you can override them if you wish.

        Args:
            structure (Structure): Input structure that must have proper
                Species on all sites, i.e. Element with oxidation state. Use
                Structure.add_oxidation_state... for example.
            real_space_cut (float): Real space cutoff radius dictating how
                many terms are used in the real space sum. Defaults to None,
                which means determine automatically using the formula given
                in gulp 3.1 documentation.
            recip_space_cut (float): Reciprocal space cutoff radius.
                Defaults to None, which means determine automatically using
                the formula given in gulp 3.1 documentation.
            eta (float): The screening parameter. Defaults to None, which means
                determine automatically.
            acc_factor (float): No. of significant figures each sum is
                converged to.
            w (float): Weight parameter, w, has been included that represents
                the relative computational expense of calculating a term in
                real and reciprocal space. Default of 0.7 reproduces result
                similar to GULP 4.2. This has little effect on the total
                energy, but may influence speed of computation in large
                systems. Note that this parameter is used only when the
                cutoffs are set to None.
            compute_forces (bool): Whether to compute forces. False by
                default since it is usually not needed.
        """
        self._struct = structure
        self._charged = abs(structure.charge) > 1e-8
        self._vol = structure.volume
        self._compute_forces = compute_forces
        self._compute_stress = compute_stress

        self._acc_factor = acc_factor
        # set screening length
        self._eta = eta or (len(structure) * w / (self._vol**2)) ** (1 / 3) * math.pi
        self._sqrt_eta = math.sqrt(self._eta)

        # acc factor used to automatically determine the optimal real and
        # reciprocal space cutoff radii
        self._accf = math.sqrt(math.log(10**acc_factor))

        self._rmax = real_space_cut or self._accf / self._sqrt_eta
        self._gmax = recip_space_cut or 2 * self._sqrt_eta * self._accf

        # The next few lines pre-compute certain quantities and store them.
        # Ewald summation is rather expensive, and these shortcuts are
        # necessary to obtain several factors of improvement in speedup.
        self._oxi_states = [compute_average_oxidation_state(site) for site in structure]

        self._coords = np.array(self._struct.cart_coords)

        # Define the private attributes to lazy compute reciprocal and real
        # space terms.
        self._initialized = False
        self._recip = self._real = self._point = self._forces = self._stress = None

        # Compute the correction for a charged cell
        self._charged_cell_energy = (
            -EwaldSummation.CONV_FACT / 2 * np.pi / structure.volume / self._eta * structure.charge**2
        )

    def compute_partial_energy(self, removed_indices):
        """Get total Ewald energy for certain sites being removed, i.e. zeroed out."""
        total_energy_matrix = self.total_energy_matrix.copy()
        for idx in removed_indices:
            total_energy_matrix[idx, :] = 0
            total_energy_matrix[:, idx] = 0
        return sum(sum(total_energy_matrix))

    def compute_sub_structure(self, sub_structure, tol: float = 1e-3):
        """Get total Ewald energy for an sub structure in the same
        lattice. The sub_structure must be a subset of the original
        structure, with possible different charges.

        Args:
            substructure (Structure): Substructure to compute Ewald sum for.
            tol (float): Tolerance for site matching in fractional coordinates.

        Returns:
            Ewald sum of substructure.
        """
        total_energy_matrix = self.total_energy_matrix.copy()

        def find_match(site):
            for test_site in sub_structure:
                frac_diff = abs(np.array(site.frac_coords) - np.array(test_site.frac_coords)) % 1
                frac_diff = [abs(a) < tol or abs(a) > 1 - tol for a in frac_diff]
                if all(frac_diff):
                    return test_site
            return None

        matches = []
        for idx, site in enumerate(self._struct):
            matching_site = find_match(site)
            if matching_site:
                new_charge = compute_average_oxidation_state(matching_site)
                old_charge = self._oxi_states[idx]
                scaling_factor = new_charge / old_charge
                matches.append(matching_site)
            else:
                scaling_factor = 0
            total_energy_matrix[idx, :] *= scaling_factor
            total_energy_matrix[:, idx] *= scaling_factor

        if len(matches) != len(sub_structure):
            output = ["Missing sites."]
            for site in sub_structure:
                if site not in matches:
                    output.append(f"unmatched = {site}")
            raise ValueError("\n".join(output))

        return sum(sum(total_energy_matrix))

    @property
    def reciprocal_space_energy(self):
        """The reciprocal space energy."""
        if not self._initialized:
            self._calc_ewald_terms()
            self._initialized = True
        return sum(sum(self._recip))

    @property
    def reciprocal_space_energy_matrix(self):
        """The reciprocal space energy matrix. Each matrix element (i, j)
        corresponds to the interaction energy between site i and site j in
        reciprocal space.
        """
        if not self._initialized:
            self._calc_ewald_terms()
            self._initialized = True
        return self._recip

    @property
    def real_space_energy(self):
        """The real space energy."""
        if not self._initialized:
            self._calc_ewald_terms()
            self._initialized = True
        return sum(sum(self._real))

    @property
    def real_space_energy_matrix(self):
        """The real space energy matrix. Each matrix element (i, j) corresponds to
        the interaction energy between site i and site j in real space.
        """
        if not self._initialized:
            self._calc_ewald_terms()
            self._initialized = True
        return self._real

    @property
    def point_energy(self):
        """The point energy."""
        if not self._initialized:
            self._calc_ewald_terms()
            self._initialized = True
        return sum(self._point)

    @property
    def point_energy_matrix(self):
        """The point space matrix. A diagonal matrix with the point terms for each
        site in the diagonal elements.
        """
        if not self._initialized:
            self._calc_ewald_terms()
            self._initialized = True
        return self._point

    @property
    def total_energy(self):
        """The total energy."""
        if not self._initialized:
            self._calc_ewald_terms()
            self._initialized = True
        return sum(sum(self._recip)) + sum(sum(self._real)) + sum(self._point) + self._charged_cell_energy

    @property
    def total_energy_matrix(self):
        """The total energy matrix. Each matrix element (i, j) corresponds to the
        total interaction energy between site i and site j.

        Note that this does not include the charged-cell energy, which is only important
        when the simulation cell is not charge balanced.
        """
        if not self._initialized:
            self._calc_ewald_terms()
            self._initialized = True

        total_energy = self._recip + self._real
        for idx, energy in enumerate(self._point):
            total_energy[idx, idx] += energy
        return total_energy

    @property
    def forces(self):
        """The forces on each site as a Nx3 matrix. Each row corresponds to a site."""
        if not self._initialized:
            self._calc_ewald_terms()
            self._initialized = True

        if not self._compute_forces:
            raise AttributeError("Forces are available only if compute_forces is True!")
        return self._forces

    @property
    def stress(self):
        """The stress tensor as a 3x3 matrix in eV/A^3."""
        if not self._initialized:
            self._calc_ewald_terms()
            self._initialized = True

        if not self._compute_stress:
            raise AttributeError("Stress is available only if compute_stress is True!")
        return self._stress

    def get_site_energy(self, site_index):
        """Compute the energy for a single site in the structure.

        Args:
            site_index (int): Index of site

        Returns:
            float: Energy of that site
        """
        if not self._initialized:
            self._calc_ewald_terms()
            self._initialized = True

        if self._charged:
            warn("Per atom energies for charged structures not supported in EwaldSummation")
        return np.sum(self._recip[:, site_index]) + np.sum(self._real[:, site_index]) + self._point[site_index]

    def _calc_ewald_terms(self):
        """Calculate and sets all Ewald terms (point, real and reciprocal)."""
        self._recip, recip_forces, recip_stress = self._calc_recip()
        self._real, self._point, real_point_forces, real_point_stress = self._calc_real_and_point()

        if self._compute_forces:
            self._forces = recip_forces + real_point_forces

        if self._compute_stress:
            charged_cell_stress = np.zeros((3, 3))
            if self._charged:
                charged_cell_stress = np.eye(3) * (self._charged_cell_energy / self._vol)
            self._stress = recip_stress + real_point_stress + charged_cell_stress

    def _calc_recip(self):
        """
        Perform the reciprocal space summation. Calculates the quantity
        E_recip = 1/(2PiV) sum_{G < Gmax} exp(-(G.G/4/eta))/(G.G) S(G)S(-G)
        where
        S(G) = sum_{k=1,N} q_k exp(-i G.r_k)
        S(G)S(-G) = |S(G)|**2.

        This method is heavily vectorized to utilize numpy's C backend for speed.
        """
        n_sites = len(self._struct)
        prefactor = 2 * math.pi / self._vol
        e_recip = np.zeros((n_sites, n_sites), dtype=np.float64)
        forces = np.zeros((n_sites, 3), dtype=np.float64)
        stress = np.zeros((3, 3), dtype=np.float64)
        coords = self._coords
        rcp_latt = self._struct.lattice.reciprocal_lattice
        recip_nn = rcp_latt.get_points_in_sphere([[0, 0, 0]], [0, 0, 0], self._gmax)

        frac_coords = [frac_coords for (frac_coords, dist, _idx, _img) in recip_nn if dist != 0]

        gs = rcp_latt.get_cartesian_coords(frac_coords)
        g2s = np.sum(gs**2, 1)
        exp_vals = np.exp(-g2s / (4 * self._eta))
        grs = np.sum(gs[:, None] * coords[None, :], 2)

        oxi_states = np.array(self._oxi_states)

        # create array where q_2[i,j] is qi * qj
        qi_qj = oxi_states[None, :] * oxi_states[:, None]

        # calculate the structure factor
        s_reals = np.sum(oxi_states[None, :] * np.cos(grs), 1)
        s_imags = np.sum(oxi_states[None, :] * np.sin(grs), 1)
        # Pre-calculate |S(G)|^2 for stress
        norm_s_sq = s_reals**2 + s_imags**2

        for g, g2, gr, exp_val, s_real, s_imag, s_sq in zip(gs, g2s, grs, exp_vals, s_reals, s_imags, norm_s_sq):
            # Uses the identity sin(x)+cos(x) = 2**0.5 sin(x + pi/4)
            m = np.sin((gr[None, :] + math.pi / 4) - gr[:, None])
            m *= exp_val / g2

            e_recip += m

            if self._compute_forces:
                pref = 2 * exp_val / g2 * oxi_states
                factor = prefactor * pref * (s_real * np.sin(gr) - s_imag * np.cos(gr))

                forces += factor[:, None] * g[None, :]

            if self._compute_stress:
                # Stress calculation
                # Term 1: Volume dependence (Hydrostatic) -> -E_recip_term * delta_ab
                term_val = exp_val / g2
                energy_term = prefactor * term_val * s_sq
                stress -= energy_term * np.eye(3)
                # Term 2: G vector dependence
                # dQ/d(G^2) = Q * [ -1/(4eta) - 1/G^2 ] = - Q * [ (G^2 + 4eta) / (4eta G^2) ]
                dQ_dG2 = -term_val * ((g2 + 4 * self._eta) / (4 * self._eta * g2))
                # Factor: - 4pi/V * |S|^2 * dQ_dG2
                stress_factor = -2 * prefactor * s_sq * dQ_dG2
                # Outer product G_alpha * G_beta
                stress += stress_factor * np.outer(g, g)

        forces *= EwaldSummation.CONV_FACT
        stress *= (EwaldSummation.CONV_FACT / self._vol)
        e_recip *= prefactor * EwaldSummation.CONV_FACT * qi_qj * 2**0.5
        return e_recip, forces, stress

    def _calc_real_and_point(self):
        """Determine the self energy -(eta/pi)**(1/2) * sum_{i=1}^{N} q_i**2."""
        frac_coords = self._struct.frac_coords
        force_pf = 2 * self._sqrt_eta / math.sqrt(math.pi)
        coords = self._coords
        n_sites = len(self._struct)
        e_real = np.empty((n_sites, n_sites), dtype=np.float64)

        forces = np.zeros((n_sites, 3), dtype=np.float64)
        stress = np.zeros((3, 3), dtype=np.float64) # Initialize stress

        qs = np.array(self._oxi_states)

        e_point = -(qs**2) * math.sqrt(self._eta / math.pi)

        for idx in range(n_sites):
            nf_coords, rij, js, _ = self._struct.lattice.get_points_in_sphere(
                frac_coords, coords[idx], self._rmax, zip_results=False
            )

            # remove the rii term
            inds = rij > 1e-8
            js = js[inds]
            rij = rij[inds]
            nf_coords = nf_coords[inds]

            qi = qs[idx]
            qj = qs[js]

            erfc_val = erfc(self._sqrt_eta * rij)
            new_ereals = erfc_val * qi * qj / rij

            # insert new_ereals
            for key in range(n_sites):
                e_real[key, idx] = np.sum(new_ereals[js == key])

            if self._compute_forces or self._compute_stress:
                # Common term for forces and stress
                # dE/dr term (scalar part)
                # d/dr(erfc(sqrt(eta)r)/r) = -1/r^2 * (erfc + 2/sqrt(pi)*sqrt(eta)*r*exp(-eta*r^2))
                # Note: The force_pf defined above is 2*sqrt(eta)/sqrt(pi)
                # fijpf corresponds to magnitude of force / r
                fijpf = qj / rij**3 * (erfc_val + force_pf * rij * np.exp(-self._eta * rij**2))
                # Vector from neighbor to central atom idx: r_i - r_j
                # nc_coords are neighbors r_j
                # dr_vec = r_i - r_j
                nc_coords = self._struct.lattice.get_cartesian_coords(nf_coords)
                dr_vecs = np.array([coords[idx]]) - nc_coords
                if self._compute_forces:
                    forces[idx] += np.sum(
                        np.expand_dims(fijpf, 1) * dr_vecs * qi * EwaldSummation.CONV_FACT,
                        axis=0,
                    )

                if self._compute_stress:
                    # Stress = (1/V) * sum_pairs (dE/dr) * (r_a * r_b / r)
                    # dE/dr = - fijpf * r^2 * qi (magnitude)
                    # But we iterate per atom, so we double count.
                    # Factor 0.5 is needed for total stress.
                    # Contribution per pair: 0.5 * (dE/dr) * (r_a * r_b / r)
                    # dE/dr / r = - fijpf * qi
                    # Contribution = -0.5 * fijpf * qi * r_a * r_b
                    # d_cart is (N_neighbors, 3)
                    # fijpf is (N_neighbors,)
                    # qi is scalar
                    prefactors = -0.5 * fijpf * qi
                    # Vectorized outer product sum: sum_k prefactor_k * r_k_a * r_k_b
                    stress += np.einsum('k,ka,kb->ab', prefactors, dr_vecs, dr_vecs)

        stress *= (EwaldSummation.CONV_FACT / self._vol)
        e_real *= 0.5 * EwaldSummation.CONV_FACT
        e_point *= EwaldSummation.CONV_FACT
        return e_real, e_point, forces, stress

    @property
    def eta(self):
        """Eta value used in Ewald summation."""
        return self._eta

    def __str__(self):
        output = [
            f"Real = {self.real_space_energy}",
            f"Reciprocal = {self.reciprocal_space_energy}",
            f"Point = {self.point_energy}",
            f"Total = {self.total_energy}",
            f"Forces:\n{self.forces}" if self._compute_forces else "Forces were not computed",
            f"Stress:\n{self.stress}" if self._compute_stress else "Stress was not computed",
        ]
        return "\n".join(output)

    def as_dict(self, verbosity: int = 0) -> dict:
        """
        JSON-serialization dict representation of EwaldSummation.

        Args:
            verbosity (int): Verbosity level. Default of 0 only includes the
                matrix representation. Set to 1 for more details.
        """
        return {
            "@module": type(self).__module__,
            "@class": type(self).__name__,
            "structure": self._struct.as_dict(),
            "compute_forces": self._compute_forces,
            "compute_stress": self._compute_stress,
            "eta": self._eta,
            "acc_factor": self._acc_factor,
            "real_space_cut": self._rmax,
            "recip_space_cut": self._gmax,
            "_recip": None if self._recip is None else self._recip.tolist(),
            "_real": None if self._real is None else self._real.tolist(),
            "_point": None if self._point is None else self._point.tolist(),
            "_forces": None if self._forces is None else self._forces.tolist(),
            "_stress": None if self._stress is None else self._stress.tolist(),
        }

    @classmethod
    def from_dict(cls, dct: dict[str, Any], fmt: str | None = None, **kwargs) -> Self:
        """Create an EwaldSummation instance from JSON-serialized dictionary.

        Args:
            dct (dict): Dictionary representation
            fmt (str, optional): Unused. Defaults to None.

        Returns:
            EwaldSummation: class instance
        """
        summation = cls(
            structure=Structure.from_dict(dct["structure"]),
            real_space_cut=dct["real_space_cut"],
            recip_space_cut=dct["recip_space_cut"],
            eta=dct["eta"],
            acc_factor=dct["acc_factor"],
            compute_forces=dct["compute_forces"],
        )

        # set previously computed private attributes
        if dct["_recip"] is not None:
            summation._recip = np.array(dct["_recip"])
            summation._real = np.array(dct["_real"])
            summation._point = np.array(dct["_point"])
            summation._forces = np.array(dct["_forces"])
            summation._initialized = True

        return summation


"""
Lennard-Jones Summation Module
Optimized for fitting workflows (Energy & Forces).
"""
class LennardJonesSummation(MSONable):
    """
    Calculates the total potential energy and forces using the Lennard-Jones (12-6) potential.

    Potential Form:
        V(r) = 4 * epsilon * [(sigma/r)^12 - (sigma/r)^6]

    Force Form (Analytical Derivative):
        F_i = -dV/dr * (r_vec / r)
            = (24 * epsilon / r^2) * [2*(sigma/r)^12 - (sigma/r)^6] * (r_i - r_j)

    Features:
        - Real-space truncated summation.
        - Supports multi-component systems (Lorentz-Berthelot mixing).
        - Fully vectorized neighbor interactions.
    """

    def __init__(
            self,
            structure: Structure,
            epsilon: Dict[str, float],
            sigma: Dict[str, float],
            cutoff: float = 12.0,
            r_min: float = 0.0,
            compute_forces: bool = False,
            compute_stress: bool = False,
            custom_pairs: Optional[Dict[Tuple[str, str], Tuple[float, float]]] = None,
    ):
        """
        Args:
            structure: Pymatgen Structure.
            epsilon: Dict of depth of potential well (eV) e.g. {'Na': 0.05, 'Cl': 0.1}.
            sigma: Dict of zero-potential distance (A) e.g. {'Na': 2.3, 'Cl': 4.4}.
            cutoff: Real space cutoff radius (A).
            r_min: Minimum distance to avoid singularity (A).
            compute_forces: Whether to calculate forces.
            compute_stress: Whether to calculate the analytical stress tensor.
            custom_pairs: Override mixing rules {(A, B): (eps, sig)}.
        """
        self.structure = structure
        self.cutoff = cutoff
        self.r_min = r_min
        self.compute_forces = compute_forces
        self.compute_stress = compute_stress

        # --- 1. Parameter Matrix Construction ---
        # Map species to integer indices for fast numpy lookup
        elements = sorted([e.symbol for e in structure.composition.elements])
        self.species_map = {el: i for i, el in enumerate(elements)}
        n_types = len(elements)

        # Initialize matrices
        self.eps_mat = np.zeros((n_types, n_types))
        self.sig_mat = np.zeros((n_types, n_types))

        # Fill matrices using mixing rules
        for s1 in elements:
            for s2 in elements:
                i, j = self.species_map[s1], self.species_map[s2]

                # Check custom overrides first (sort key to ensure A-B == B-A)
                pair_key = tuple(sorted((s1, s2)))

                if custom_pairs and pair_key in custom_pairs:
                    ep, si = custom_pairs[pair_key]
                else:
                    # Lorentz-Berthelot Rules
                    # sigma = arithmetic mean, epsilon = geometric mean
                    si = 0.5 * (sigma[s1] + sigma[s2])
                    ep = math.sqrt(epsilon[s1] * epsilon[s2])

                self.eps_mat[i, j] = self.eps_mat[j, i] = ep
                self.sig_mat[i, j] = self.sig_mat[j, i] = si

        # Create an array of type indices for the atoms in the structure
        # e.g., [0, 1, 0, 1] for Na Cl Na Cl
        self.atom_types = np.array(
            [self.species_map[s.specie.symbol] for s in structure], dtype=int
        )

        # --- 2. Perform Calculation ---
        self._energy = 0.0
        self._forces = np.zeros((len(structure), 3)) if compute_forces else None
        self._stress = np.zeros((3, 3)) if compute_stress else None
        self._calculate()

    @property
    def total_energy(self) -> float:
        """Total Energy in eV."""
        return self._energy

    @property
    def forces(self) -> np.ndarray:
        """Forces in eV/A. Shape (N, 3)."""
        if self._forces is None:
            raise ValueError("Forces were not computed.")
        return self._forces

    @property
    def stress(self) -> np.ndarray:
        """Analytical Stress Tensor in eV/A^3."""
        if self._stress is None:
            raise ValueError("Stress was not computed. Set compute_stress=True in __init__.")
        return self._stress

    def _calculate(self):
        """Core vectorized calculation engine."""
        lattice = self.structure.lattice
        cart_coords = np.array(self.structure.cart_coords)
        frac_coords = self.structure.frac_coords
        n_atoms = len(self.structure)
        vol = self.structure.volume

        total_energy = 0.0
        total_stress = np.zeros((3, 3)) if self.compute_stress else None

        # Loop over each atom (Central Atom i)
        for i in range(n_atoms):
            # 1. Get Neighbors
            # get_points_in_sphere returns neighbors including periodic images
            # format: [coords, distance, index, image]
            neighbors = lattice.get_points_in_sphere(
                frac_coords, cart_coords[i], self.cutoff, zip_results=False
            )

            # Unpack results
            # neighbor_frac_coords = neighbors[0] # Not needed if we use cartesian
            dists = neighbors[1]
            indices = neighbors[2] # Indices of neighbor atoms j

            # 2. Filter Self-interaction and Short-range singularities
            # We must remove the i-i interaction (dist ~ 0)
            mask = dists > self.r_min

            if not np.any(mask):
                continue

            dists = dists[mask]
            indices = indices[mask]
            neighbor_frac_coords = neighbors[0][mask]

            # 3. Retrieve Parameters Vectorized
            # Type of central atom i
            type_i = self.atom_types[i]
            # Types of neighbor atoms j
            types_j = self.atom_types[indices]

            # Look up parameters from matrices using integer array indexing
            eps_vals = self.eps_mat[type_i, types_j]
            sig_vals = self.sig_mat[type_i, types_j]

            # 4. Calculate Energy Terms
            # (sigma / r)
            sig_r = sig_vals / dists

            # Powers
            sig_r_6 = sig_r ** 6
            sig_r_12 = sig_r_6 * sig_r_6

            # LJ Potential: 4 * eps * (s^12 - s^6)
            e_terms = 4.0 * eps_vals * (sig_r_12 - sig_r_6)
            total_energy += np.sum(e_terms)

            # 5. Calculate Forces
            if self.compute_forces or self.compute_stress:
                # Need vector from Neighbor j -> Central i
                # r_vec = r_i - r_j
                neighbor_cart_coords = lattice.get_cartesian_coords(neighbor_frac_coords)
                dr_vecs = cart_coords[i] - neighbor_cart_coords
                # Force Scalar Prefactor
                # F = (24 * eps / r^2) * (2 * s^12 - s^6) * r_vec
                # Note: The r^2 in denominator cancels the r in derivative and r in normalization
                # Calculate (24 * eps / r^2)
                # We use dists**2 to avoid extra sqrt operations if we had squared dists,
                # but here we have dists.
                prefactor = (24.0 * eps_vals / (dists * dists)) * (2.0 * sig_r_12 - sig_r_6)

                if self.compute_forces:
                    # Broadcast prefactor to (N, 1) to multiply (N, 3) vectors
                    force_contributions = dr_vecs * prefactor[:, np.newaxis]
                    # Sum contributions from all neighbors to get total force on atom i
                    self._forces[i] += np.sum(force_contributions, axis=0)

                if self.compute_stress:
                    # Virial Stress Term: - (1/V) * sum ( r_a * F_b )
                    # F_b = prefactor * r_b
                    # Term = - (1/V) * prefactor * r_a * r_b
                    # We sum (prefactor * r_a * r_b) here, apply -1/V later.
                    # Vectorized outer product summation: sum_k (S[k] * v[k, a] * v[k, b])
                    # einsum: k=neighbor index, a=dim1, b=dim2
                    batch_stress = np.einsum('k,ka,kb->ab', prefactor, dr_vecs, dr_vecs)
                    total_stress += -0.5 * batch_stress / vol

        # 6. Final Corrections
        # Energy: Each pair counted twice (i-j and j-i), so divide by 2
        self._energy = total_energy * 0.5

        # Forces: Each pair interaction adds F_ij to atom i and F_ji to atom j separately.
        # No division needed.

        self._stress = total_stress

    def as_dict(self) -> dict:
        return {
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
            "total_energy": self._energy,
            "forces": self._forces.tolist() if self._forces is not None else None,
            "stress": self._stress.tolist() if self._stress is not None else None,
            "cutoff": self.cutoff
        }