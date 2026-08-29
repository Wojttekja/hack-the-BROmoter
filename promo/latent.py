"""Latent-space geometry: Euclidean and Poincare-ball (hyperbolic) operations.

The real Navigator's architecture name mentions "hyperbolic", so linear arithmetic
in the latent space may be invalid. This module implements both geometries behind a
single dispatching :class:`LatentOps` so optimizers never branch on geometry
themselves. It also provides :func:`detect_geometry` to pick the right mode from a
sample of real latent vectors on the morning of the event.

Poincare-ball formulas use curvature ``c = 1`` (unit ball). Near the origin every
operation reduces to its Euclidean counterpart, which the tests assert.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

Geometry = Literal["euclidean", "poincare"]

_EPS = 1e-9
_MAX_NORM = 1.0 - 1e-5  # keep points strictly inside the open unit ball


def _clip_to_ball(x: np.ndarray) -> np.ndarray:
    """Project ``x`` to lie strictly inside the open unit ball."""
    norm = np.linalg.norm(x)
    if norm >= _MAX_NORM:
        return x * (_MAX_NORM / (norm + _EPS))
    return x


# --- Mobius / hyperbolic primitives (curvature c=1) -------------------------


def mobius_add(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Mobius addition ``x (+) y`` in the unit Poincare ball.

    This is the hyperbolic analogue of vector addition. It is non-commutative.
    """
    xy = float(np.dot(x, y))
    x2 = float(np.dot(x, x))
    y2 = float(np.dot(y, y))
    num = (1 + 2 * xy + y2) * x + (1 - x2) * y
    den = 1 + 2 * xy + x2 * y2
    return num / (den + _EPS)


def _lambda(x: np.ndarray) -> float:
    """Conformal factor lambda_x = 2 / (1 - ||x||^2) at point ``x``."""
    return 2.0 / (1.0 - float(np.dot(x, x)) + _EPS)


def expmap(x: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Exponential map: move from base point ``x`` along tangent vector ``v``.

    Maps a Euclidean tangent vector at ``x`` to a point on the ball. This is how we
    take an optimizer step in the tangent space and land back on the manifold.
    """
    vn = float(np.linalg.norm(v))
    if vn < _EPS:
        return x
    lam = _lambda(x)
    second = np.tanh(lam * vn / 2.0) * v / vn
    return _clip_to_ball(mobius_add(x, second))


def logmap(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Logarithmic map: tangent vector at ``x`` pointing toward ``y``.

    Inverse of :func:`expmap`. ``expmap(x, logmap(x, y)) == y``.
    """
    diff = mobius_add(-x, y)
    dn = float(np.linalg.norm(diff))
    if dn < _EPS:
        return np.zeros_like(x)
    lam = _lambda(x)
    return (2.0 / lam) * np.arctanh(min(dn, _MAX_NORM)) * diff / dn


def poincare_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Geodesic distance between two points in the unit Poincare ball."""
    diff = mobius_add(-x, y)
    dn = float(np.linalg.norm(diff))
    return float(2.0 * np.arctanh(min(dn, _MAX_NORM)))


# --- Dispatching operations -------------------------------------------------


@dataclass(slots=True)
class LatentOps:
    """Geometry-aware latent operations chosen once by ``geometry``.

    Optimizers hold a ``LatentOps`` and call ``step``/``interpolate``/``perturb``
    without knowing whether the space is Euclidean or hyperbolic.
    """

    geometry: Geometry = "euclidean"

    def step(self, z: np.ndarray, direction: np.ndarray, scale: float = 1.0) -> np.ndarray:
        """Move from ``z`` along ``direction`` (a tangent vector) by ``scale``."""
        v = np.asarray(direction, dtype=float) * scale
        if self.geometry == "euclidean":
            return np.asarray(z, dtype=float) + v
        return expmap(np.asarray(z, dtype=float), v)

    def interpolate(self, z1: np.ndarray, z2: np.ndarray, t: float) -> np.ndarray:
        """Point at fraction ``t`` along the geodesic from ``z1`` to ``z2``."""
        z1 = np.asarray(z1, dtype=float)
        z2 = np.asarray(z2, dtype=float)
        if self.geometry == "euclidean":
            return (1 - t) * z1 + t * z2
        return expmap(z1, t * logmap(z1, z2))

    def midpoint(self, z1: np.ndarray, z2: np.ndarray) -> np.ndarray:
        """Geodesic midpoint of ``z1`` and ``z2``."""
        return self.interpolate(z1, z2, 0.5)

    def perturb(
        self, z: np.ndarray, scale: float, rng: np.random.Generator
    ) -> np.ndarray:
        """Random perturbation of ``z`` with the given ``scale`` (std of the step)."""
        z = np.asarray(z, dtype=float)
        noise = rng.standard_normal(z.shape) * scale
        return self.step(z, noise, 1.0)

    def distance(self, z1: np.ndarray, z2: np.ndarray) -> float:
        """Distance appropriate to the geometry."""
        z1 = np.asarray(z1, dtype=float)
        z2 = np.asarray(z2, dtype=float)
        if self.geometry == "euclidean":
            return float(np.linalg.norm(z1 - z2))
        return poincare_distance(z1, z2)

    def radius(self, z: np.ndarray) -> float:
        """Distance from the origin (hyperbolic radius in Poincare mode)."""
        origin = np.zeros_like(np.asarray(z, dtype=float))
        return self.distance(origin, np.asarray(z, dtype=float))


def detect_geometry(sample: np.ndarray, *, boundary_frac: float = 0.1) -> Geometry:
    """Guess the latent geometry from a sample of latent vectors.

    Heuristic: if every vector has norm < 1 and a non-trivial fraction sits near the
    boundary (norm > ``1 - boundary_frac``), treat the space as a Poincare ball.
    Otherwise assume Euclidean.

    Args:
        sample: Array of shape ``(n, d)`` of encoded latent vectors.
        boundary_frac: Distance-from-boundary threshold for "near the boundary".

    Returns:
        ``"poincare"`` or ``"euclidean"``.
    """
    sample = np.atleast_2d(np.asarray(sample, dtype=float))
    norms = np.linalg.norm(sample, axis=1)
    if norms.size == 0:
        return "euclidean"
    all_inside = bool(np.all(norms < 1.0 + 1e-6))
    near_boundary = float(np.mean(norms > (1.0 - boundary_frac)))
    max_norm = float(np.max(norms))
    if all_inside and (near_boundary > 0.05 or max_norm > 0.9):
        return "poincare"
    return "euclidean"
