import logging
from typing import Optional

from e3nn import o3
from nequip.data import AtomicDataDict, AtomicDataset
from nequip.model import builder_utils
from nequip.nn import SequentialGraphNetwork, AtomwiseReduce
from nequip.nn.embedding import (
    OneHotAtomEncoding,
    SphericalHarmonicEdgeAttrs,
    RadialBasisEdgeEncoding,
)
from nequip.nn.radial_basis import BesselBasis

from allegro._keys import (
    EDGE_FEATURES,
    EDGE_SHORT_ENERGY,
    ATOMIC_SHORT_ENERGY,
    SHORT_ENERGY,
)
from allegro.nn import (
    NormalizedBasis,
    EdgewiseReduce,
    Allegro_Module,
    ScalarMLP,
    ReciprocalNN,
    ChargeMLP,
)


def ReciprocalNN_Allegro_charge_equilibrium(config, initialize: bool, dataset: Optional[AtomicDataset] = None):
    logging.debug("Building ReciprocalNN_Allegro model...")

    # Handle avg num neighbors auto
    builder_utils.add_avg_num_neighbors(
        config=config, initialize=initialize, dataset=dataset
    )

    # Handle simple irreps
    if "l_max" in config:
        l_max = int(config["l_max"])
        parity_setting = config["parity"]
        assert parity_setting in ("o3_full", "o3_restricted", "so3")
        irreps_edge_sh = repr(
            o3.Irreps.spherical_harmonics(
                l_max, p=(1 if parity_setting == "so3" else -1)
            )
        )
        nonscalars_include_parity = parity_setting == "o3_full"
        # check consistant
        assert config.get("irreps_edge_sh", irreps_edge_sh) == irreps_edge_sh
        assert (
            config.get("nonscalars_include_parity", nonscalars_include_parity)
            == nonscalars_include_parity
        )
        config["irreps_edge_sh"] = irreps_edge_sh
        config["nonscalars_include_parity"] = nonscalars_include_parity

    layers = {
        # %% -- Encode --
        # Get various edge invariants
        "one_hot": OneHotAtomEncoding,
        "radial_basis": (
            RadialBasisEdgeEncoding,
            dict(
                basis=(
                    NormalizedBasis
                    if config.get("normalize_basis", True)
                    else BesselBasis
                ),
                out_field=AtomicDataDict.EDGE_EMBEDDING_KEY,
            ),
        ),
        # Get edge nonscalars
        "spharm": SphericalHarmonicEdgeAttrs,

        # %% -- Short Range Model (allegro model) --
        "allegro": (
            Allegro_Module,
            dict(
                field=AtomicDataDict.EDGE_ATTRS_KEY,  # initial input is the edge SH
                edge_invariant_field=AtomicDataDict.EDGE_EMBEDDING_KEY,
                node_invariant_field=AtomicDataDict.NODE_ATTRS_KEY,
            ),
        ),
        "edge_short_eng": (
            ScalarMLP,
            dict(field=EDGE_FEATURES, out_field=EDGE_SHORT_ENERGY, mlp_output_dimension=1),
        ),
        "edge_short_eng_sum": (
            EdgewiseReduce,
            dict(field=EDGE_SHORT_ENERGY, out_field=ATOMIC_SHORT_ENERGY),
        ),
        "short_eng_sum": (
            AtomwiseReduce,
            dict(
                reduce="sum",
                field=ATOMIC_SHORT_ENERGY,
                out_field=SHORT_ENERGY,
            ),
        ),

        # %% -- Long Range Model (reciprocal model) --
        # Get atomic charges (learnable)
        "charge_mlp": (
            ChargeMLP,
            dict(
                edge_invariant_field=AtomicDataDict.EDGE_EMBEDDING_KEY,
                node_invariant_field=AtomicDataDict.NODE_ATTRS_KEY,
            ),
        ),
        # ReciprocalNN
        "ReciprocalNN": (
            ReciprocalNN,
            dict(
                n_hidden=config["FCN_hidden"],
                n_layers=config["FCN_layers"],
                k_max=config.get("k_max"),
                eta=config.get("eta"),
                acc_factor=config.get("acc_factor"),
            ),
        ),

        # %% -- Sum Total Energy --
        # Sum system energy:
        "total_energy_sum": (
            AtomwiseReduce,
            dict(
                reduce="sum",
                field=AtomicDataDict.PER_ATOM_ENERGY_KEY,
                out_field=AtomicDataDict.TOTAL_ENERGY_KEY,
            ),
        ),
    }

    model = SequentialGraphNetwork.from_parameters(shared_params=config, layers=layers)

    return model
