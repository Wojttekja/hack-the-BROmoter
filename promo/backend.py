"""The ONE swap point for all oracle access.

Every part of the codebase obtains its Judge and Navigator here. Nothing else may
import a concrete backend module. On the morning of the hackathon we change exactly
one thing -- the ``PROMO_BACKEND`` environment variable (or ``--backend`` flag,
which sets it) -- and the real oracles flow through unchanged.

    PROMO_BACKEND=mock   (default) -> promo.mock_backend
    PROMO_BACKEND=real            -> promo.real_backend
"""

from __future__ import annotations

import os

from .interfaces import Judge, Navigator

_VALID = ("mock", "real")


def _selected(explicit: str | None) -> str:
    """Resolve the backend name from an explicit arg or ``PROMO_BACKEND``."""
    name = (explicit or os.environ.get("PROMO_BACKEND", "mock")).lower()
    if name not in _VALID:
        raise ValueError(f"PROMO_BACKEND must be one of {_VALID}, got {name!r}")
    return name


def get_judge(backend: str | None = None, **kwargs: object) -> Judge:
    """Return a raw (uncached) Judge for the selected backend.

    The caller is responsible for wrapping this in
    :class:`promo.cache.CachedJudge`; the probe suite deliberately uses the raw
    handle so it can observe determinism and latency.

    Args:
        backend: Override for ``PROMO_BACKEND`` (``"mock"`` or ``"real"``).
        **kwargs: Passed through to the backend constructor.

    Returns:
        An object satisfying the :class:`~promo.interfaces.Judge` Protocol.
    """
    name = _selected(backend)
    if name == "mock":
        from .mock_backend import MockJudge

        return MockJudge(**kwargs)  # type: ignore[arg-type]
    from .real_backend import RealJudge

    return RealJudge(**kwargs)  # type: ignore[arg-type]


def get_navigator(backend: str | None = None, **kwargs: object) -> Navigator:
    """Return a Navigator for the selected backend.

    Args:
        backend: Override for ``PROMO_BACKEND``.
        **kwargs: Passed through to the backend constructor.

    Returns:
        An object satisfying the :class:`~promo.interfaces.Navigator` Protocol.
    """
    name = _selected(backend)
    if name == "mock":
        from .mock_backend import MockNavigator

        return MockNavigator(**kwargs)  # type: ignore[arg-type]
    from .real_backend import RealNavigator

    return RealNavigator(**kwargs)  # type: ignore[arg-type]
