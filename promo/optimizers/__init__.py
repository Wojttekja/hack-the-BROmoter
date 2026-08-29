"""Optimizer implementations and a name-based registry for the runner."""

from __future__ import annotations

from .base import Optimizer
from .cmaes import CMAES
from .koth import KingOfTheHill
from .map_elites import MapElites
from .random import RandomSearch

#: Maps ``--optimizer`` names to classes. The runner uses this to instantiate.
REGISTRY: dict[str, type[Optimizer]] = {
    "random": RandomSearch,
    "koth": KingOfTheHill,
    "cmaes": CMAES,
    "map_elites": MapElites,
}

__all__ = [
    "CMAES",
    "REGISTRY",
    "KingOfTheHill",
    "MapElites",
    "Optimizer",
    "RandomSearch",
]
