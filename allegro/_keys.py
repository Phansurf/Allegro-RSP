"""Keys file to overcome TorchScript constants bug."""

import sys

if sys.version_info[1] >= 8:
    from typing import Final
else:
    from typing_extensions import Final

from nequip.data import register_fields

EDGE_ENERGY: Final[str] = "edge_energy"
EDGE_FEATURES: Final[str] = "edge_features"
EDGE_CHARGES: Final[str] = "edge_charges"
EDGE_SHORT_ENERGY: Final[str] = "edge_short_energy"
ATOMIC_CHARGES: Final[str] = "atomic_charges"
ATOMIC_RECIPROCAL_ENERGY: Final[str] = "atomic_reciprocal_energy"
ATOMIC_SHORT_ENERGY: Final[str] = "atomic_short_energy"
RECIPROCAL_ENERGY: Final[str] = "reciprocal_energy"
SHORT_ENERGY: Final[str] = "short_energy"
TOTAL_CHARGE: Final[str] = "total_charge"

register_fields(
    edge_fields=[EDGE_ENERGY, EDGE_FEATURES, EDGE_CHARGES, EDGE_SHORT_ENERGY],
    node_fields=[ATOMIC_CHARGES, ATOMIC_RECIPROCAL_ENERGY, ATOMIC_SHORT_ENERGY],
    graph_fields=[RECIPROCAL_ENERGY, SHORT_ENERGY, TOTAL_CHARGE],
)
