"""promo: black-box promoter optimization scaffold for the HYPPE hackathon.

All oracle access flows through :mod:`promo.backend`, which returns objects
satisfying the Protocols in :mod:`promo.interfaces`. Swap the backend by setting
the ``PROMO_BACKEND`` environment variable (``mock`` or ``real``); nothing else in
the codebase imports a concrete backend.
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
