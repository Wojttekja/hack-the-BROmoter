"""Adapter skeleton for the real HYPPE oracles. FILL IN ON THE MORNING.

The real HYPPE library and its exact API are not visible until the event starts.
This module is the *only* place that will import it. Everything else in the codebase
already talks to the :class:`~promo.interfaces.Judge` and
:class:`~promo.interfaces.Navigator` Protocols, so once the three TODO regions below
are filled in and ``PROMO_BACKEND=real`` is set, the whole pipeline runs unchanged.

Keep edits surgical. Do NOT reshape the rest of the codebase to match HYPPE; reshape
HYPPE to these Protocols here. See the "MORNING OF THE HACKATHON" checklist in
README.md for the exact sequence.
"""

from __future__ import annotations

import numpy as np

from .interfaces import Winner

# TODO(hackathon) #0 -- IMPORT GUARD.
# Import the real library lazily so that merely importing this module (e.g. during
# `python -m promo.probe --backend mock`) never fails when HYPPE is absent. Replace
# the placeholder name with the real package once known.
try:  # pragma: no cover - depends on the real library being installed
    import hyppe  # type: ignore  # noqa: F401

    _HYPPE_AVAILABLE = True
except Exception:  # noqa: BLE001 - any import failure means "not installed yet"
    _HYPPE_AVAILABLE = False


def _require_hyppe() -> None:
    """Raise a clear error if the real library is not importable yet."""
    if not _HYPPE_AVAILABLE:
        raise RuntimeError(
            "The real HYPPE library is not importable. Fill in the TODOs in "
            "promo/real_backend.py and ensure `hyppe` is installed, or run with "
            "--backend mock."
        )


class RealJudge:
    """Adapts the real pairwise Judge to the :class:`~promo.interfaces.Judge` Protocol."""

    def __init__(self, **kwargs: object) -> None:
        """Construct the real judge handle.

        TODO(hackathon) #1 -- JUDGE CONSTRUCTION.
        Replace the body with however HYPPE hands you a judge, e.g.::

            from hyppe import load_judge
            self._judge = load_judge(**kwargs)

        Store the handle privately; do not expose it (the cache must not be
        bypassable). Keep this constructor cheap -- no oracle calls here.
        """
        _require_hyppe()
        # TODO: self._judge = hyppe.<something>(**kwargs)
        self._judge: object = None
        raise NotImplementedError("Fill in RealJudge.__init__ (TODO #1).")

    def compare(self, seq_a: str, seq_b: str) -> Winner:
        """Return ``"A"`` if ``seq_a`` is predicted stronger, else ``"B"``.

        TODO(hackathon) #2 -- VERDICT MAPPING.
        Call the real judge and normalize whatever it returns into ``"A"``/``"B"``.
        Expect one of these shapes and adapt accordingly:

          * a boolean / int ("is A stronger?")     -> "A" if truthy else "B"
          * the winning sequence string             -> "A" if == seq_a else "B"
          * a 0/1 class index or {-1,+1}            -> map explicitly
          * a probability p that A wins             -> "A" if p >= 0.5 else "B"

        Do NOT invent a numeric score; the real Judge has none. If HYPPE only offers
        batch scoring, wrap it so a single pairwise verdict still comes out here.
        """
        raise NotImplementedError("Map the real verdict to 'A'/'B' (TODO #2).")


class RealNavigator:
    """Adapts the real Navigator to the :class:`~promo.interfaces.Navigator` Protocol."""

    def __init__(self, **kwargs: object) -> None:
        """Construct the real navigator handle.

        TODO(hackathon) #3 -- NAVIGATOR CONSTRUCTION.
        Replace with HYPPE's encoder/decoder loading, e.g.::

            from hyppe import load_navigator
            self._nav = load_navigator(**kwargs)
        """
        _require_hyppe()
        self._nav: object = None
        raise NotImplementedError("Fill in RealNavigator.__init__ (TODO #3).")

    def encode(self, seq: str) -> np.ndarray:
        """Encode a sequence to a latent vector.

        TODO(hackathon) #4 -- ENCODE MAPPING.
        Return a 1-D ``np.ndarray``. If HYPPE returns a torch tensor, do
        ``.detach().cpu().numpy()``. Confirm dtype is float and shape is ``(dim,)``.
        """
        raise NotImplementedError("Map encode() output to np.ndarray (TODO #4).")

    def decode(self, z: np.ndarray) -> str:
        """Decode a latent vector back to a sequence.

        TODO(hackathon) #5 -- DECODE MAPPING.
        Accept a 1-D ``np.ndarray`` and return an ACGT string. Convert to whatever
        tensor type HYPPE expects on the way in, and uppercase/clean on the way out.
        """
        raise NotImplementedError("Map decode() input/output (TODO #5).")

    # OPTIONAL members below. The Navigator Protocol allows them to be absent; the
    # probe suite detects at runtime whether they exist.

    @property
    def dim(self) -> int:
        """Latent dimensionality.

        TODO(hackathon) #6 (optional) -- return the real latent dim, or delete this
        property if HYPPE does not expose one (probe will infer it from a sample).
        """
        raise NotImplementedError

    def distance(self, z1: np.ndarray, z2: np.ndarray) -> float:
        """Native latent distance, if HYPPE provides one.

        TODO(hackathon) #7 (optional) -- forward to HYPPE's distance if it exists.
        If it does not, DELETE this method entirely and rely on promo.latent, whose
        geometry the probe will have detected. Do not fabricate a Euclidean distance
        here if the space is hyperbolic.
        """
        raise NotImplementedError
