"""Hyperbolic latent ops: stay in the ball, reduce to Euclidean near the origin."""

from __future__ import annotations

import numpy as np

from promo.latent import (
    LatentOps,
    detect_geometry,
    expmap,
    logmap,
    mobius_add,
    poincare_distance,
)

RNG = np.random.default_rng(0)


def _random_ball_point(dim: int = 8, scale: float = 0.5) -> np.ndarray:
    v = RNG.standard_normal(dim)
    return v / np.linalg.norm(v) * (scale * RNG.random())


def test_expmap_stays_inside_ball() -> None:
    """Expmap never produces a point on or outside the unit ball, even for big steps."""
    x = _random_ball_point()
    for _ in range(50):
        v = RNG.standard_normal(8) * 10.0  # deliberately huge tangent vectors
        y = expmap(x, v)
        assert np.linalg.norm(y) < 1.0


def test_expmap_logmap_are_inverse() -> None:
    """expmap(x, logmap(x, y)) recovers y."""
    x = _random_ball_point()
    y = _random_ball_point()
    recovered = expmap(x, logmap(x, y))
    assert np.linalg.norm(recovered - y) < 1e-6


def test_near_origin_reduces_to_euclidean() -> None:
    """Near the origin, hyperbolic ops match their Euclidean counterparts."""
    v = np.zeros(8)
    v[0] = 1e-5
    # expmap at origin ~ identity displacement
    assert np.linalg.norm(expmap(np.zeros(8), v) - v) < 1e-9
    # Mobius addition ~ vector addition for tiny vectors
    a = np.zeros(8)
    a[1] = 1e-5
    assert np.linalg.norm(mobius_add(a, v) - (a + v)) < 1e-9
    # Geodesic distance reduces to Euclidean up to the metric's conformal factor at
    # the origin (lambda_0 = 2 for the curvature-1 ball), i.e. d_hyp -> 2 * d_euc.
    d_hyp = poincare_distance(a, v)
    d_euc = float(np.linalg.norm(a - v))
    assert abs(d_hyp - 2.0 * d_euc) < 1e-10


def test_latentops_dispatch() -> None:
    """LatentOps routes to the requested geometry and keeps poincare in the ball."""
    eu = LatentOps("euclidean")
    z = np.array([0.1, 0.2, 0.0])
    stepped = eu.step(z, np.array([1.0, 0.0, 0.0]), 0.5)
    assert np.allclose(stepped, z + np.array([0.5, 0.0, 0.0]))

    po = LatentOps("poincare")
    base = np.array([0.3, 0.0, 0.0])
    for _ in range(20):
        p = po.perturb(base, 2.0, RNG)
        assert np.linalg.norm(p) < 1.0


def test_detect_geometry() -> None:
    """detect_geometry flags boundary-hugging samples as poincare, else euclidean."""
    euclid = RNG.standard_normal((50, 8)) * 5.0
    assert detect_geometry(euclid) == "euclidean"
    dirs = RNG.standard_normal((50, 8))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    ball = dirs * RNG.uniform(0.9, 0.99, size=(50, 1))
    assert detect_geometry(ball) == "poincare"
