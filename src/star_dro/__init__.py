"""STaR-DRO's dependency-free adversary; PyTorch integration is optional."""
from .core import Config, StarDRO, backmap, schedule
from .grouping import GroupInventory
from .standard_dro import StandardDRO

__all__ = ["Config", "StarDRO", "backmap", "schedule", "GroupInventory", "StandardDRO"]

__version__ = '0.5.0'

