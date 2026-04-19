"""Sampling primitives ported from Cycles bevel.h.

Cubic BSSRDF falloff + quintic root-find for disk-radius sampling, and
Halton-based stratified 2D sampling with per-texel Cranley-Patterson
rotation so rays are decorrelated across texels and stratified across
the sample index.
"""

from __future__ import annotations

import math

import numpy as np


def svm_bevel_cubic_eval(radius: float, r: np.ndarray | float) -> np.ndarray | float:
    """Cubic BSSRDF falloff. Matches Cycles svm_bevel_cubic_eval."""
    Rm5 = radius ** 5
    if np.isscalar(r):
        if r >= radius:
            return 0.0
        f = radius - r
        return (10.0 * f * f * f) / (Rm5 * math.pi)
    out = np.zeros_like(r, dtype=np.float32)
    mask = r < radius
    f = (radius - r[mask]).astype(np.float32)
    out[mask] = (10.0 * f * f * f) / (Rm5 * math.pi)
    return out


def svm_bevel_cubic_quintic_root_find(xi: np.ndarray, max_iter: int = 10, tol: float = 1e-6) -> np.ndarray:
    """Solve 10x^2 - 20x^3 + 15x^4 - 4x^5 - xi = 0 by Newton-Raphson.

    Vectorized. Starts at x=0.25, clamps to [0, 1] each step.
    """
    x = np.full_like(xi, 0.25, dtype=np.float32)
    for _ in range(max_iter):
        x2 = x * x
        x3 = x2 * x
        nx = 1.0 - x
        f = 10.0 * x2 - 20.0 * x3 + 15.0 * x2 * x2 - 4.0 * x2 * x3 - xi
        fp = 20.0 * (x * nx) * (nx * nx)
        step = np.where(np.abs(fp) < 1e-20, 0.0, f / fp)
        x = np.clip(x - step, 0.0, 1.0)
        if np.all(np.abs(f) < tol):
            break
    return x


def svm_bevel_cubic_sample(radius: float, xi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns (disk_r, disk_height) per input xi. Cycles-faithful."""
    r_norm = svm_bevel_cubic_quintic_root_find(xi.astype(np.float32))
    r = r_norm * radius
    h = np.sqrt(np.maximum(0.0, radius * radius - r * r))
    return r.astype(np.float32), h.astype(np.float32)


def _radical_inverse(i: int, base: int) -> float:
    f = 1.0
    r = 0.0
    while i > 0:
        f /= base
        r += f * (i % base)
        i //= base
    return r


def halton_2d(s: int) -> tuple[float, float]:
    return _radical_inverse(s + 1, 2), _radical_inverse(s + 1, 3)


def per_texel_offsets(n: int, seed: int = 0) -> np.ndarray:
    """Per-texel Cranley-Patterson rotation offsets in [0, 1)^2."""
    rng = np.random.default_rng(seed)
    return rng.random((n, 2), dtype=np.float32)


def make_orthonormals_batch(N: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build an orthonormal frame (T, B) given a unit normal N per row.

    Stable against alignment with any coordinate axis: helper axis chosen
    based on which component of N is smallest.
    """
    ax = np.abs(N)
    # Pick the axis whose component of N is smallest
    smallest = np.argmin(ax, axis=-1)
    helper = np.zeros_like(N)
    helper[np.arange(N.shape[0]), smallest] = 1.0
    T = np.cross(N, helper)
    T /= np.maximum(np.linalg.norm(T, axis=-1, keepdims=True), 1e-20)
    B = np.cross(N, T)
    return T.astype(np.float32), B.astype(np.float32)
