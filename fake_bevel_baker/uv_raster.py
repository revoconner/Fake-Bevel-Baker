"""Stage 3: UV rasterization.

Walk each UV triangle's bounding box, test texel centers, and record
(triangle_index, bary_u, bary_v, bary_w) for every covered texel. First
write wins on UV overlaps.

Texel convention: texel at integer pixel (x, y) has UV
((x + 0.5) / W, (y + 0.5) / H). Array layout is `[y, x]`; v=0 sits at
y=0. PNG flipping, if needed, happens at write time.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RasterResult:
    tri_idx: np.ndarray   # (H, W) int32, -1 for empty
    bary: np.ndarray      # (H, W, 3) float32 -> (u, v, w)
    valid: np.ndarray     # (H, W) bool


def rasterize_uvs(
    uvs: np.ndarray,
    faces: np.ndarray,
    width: int,
    height: int,
    edge_eps: float = 1e-5,
) -> RasterResult:
    """Rasterize triangles to a texel grid.

    uvs:    (Nv, 2) float, in [0, 1] typically (out-of-range UVs are
            clipped to the image bounds by the AABB).
    faces:  (F, 3) int indices into uvs.
    width, height: output resolution in texels.
    edge_eps: tolerance on barycentric edge tests. Positive values keep
              texels exactly on an edge classified as inside.
    """
    if width < 1 or height < 1:
        raise ValueError("width and height must be >= 1")

    uvs_px = np.empty_like(uvs, dtype=np.float64)
    uvs_px[:, 0] = uvs[:, 0].astype(np.float64) * width - 0.5
    uvs_px[:, 1] = uvs[:, 1].astype(np.float64) * height - 0.5

    tri_idx = np.full((height, width), -1, dtype=np.int32)
    bary = np.zeros((height, width, 3), dtype=np.float32)
    valid = np.zeros((height, width), dtype=bool)

    F = int(faces.shape[0])
    for f in range(F):
        a = uvs_px[faces[f, 0]]
        b = uvs_px[faces[f, 1]]
        c = uvs_px[faces[f, 2]]

        min_x = min(a[0], b[0], c[0])
        max_x = max(a[0], b[0], c[0])
        min_y = min(a[1], b[1], c[1])
        max_y = max(a[1], b[1], c[1])

        x0 = max(0, int(np.floor(min_x)))
        x1 = min(width - 1, int(np.ceil(max_x)))
        y0 = max(0, int(np.floor(min_y)))
        y1 = min(height - 1, int(np.ceil(max_y)))
        if x0 > x1 or y0 > y1:
            continue

        denom = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        if abs(denom) < 1e-20:
            continue
        inv_denom = 1.0 / denom

        ys = np.arange(y0, y1 + 1, dtype=np.float64)
        xs = np.arange(x0, x1 + 1, dtype=np.float64)
        px, py = np.meshgrid(xs, ys)

        u = ((b[1] - c[1]) * (px - c[0]) + (c[0] - b[0]) * (py - c[1])) * inv_denom
        v = ((c[1] - a[1]) * (px - c[0]) + (a[0] - c[0]) * (py - c[1])) * inv_denom
        w = 1.0 - u - v

        inside = (u >= -edge_eps) & (v >= -edge_eps) & (w >= -edge_eps)
        if not inside.any():
            continue

        yy, xx = np.where(inside)
        pix_y = yy + y0
        pix_x = xx + x0
        empty = tri_idx[pix_y, pix_x] == -1
        if not empty.any():
            continue
        pix_y = pix_y[empty]
        pix_x = pix_x[empty]

        u_sel = np.clip(u[yy[empty], xx[empty]], 0.0, 1.0).astype(np.float32)
        v_sel = np.clip(v[yy[empty], xx[empty]], 0.0, 1.0).astype(np.float32)
        w_sel = np.clip(w[yy[empty], xx[empty]], 0.0, 1.0).astype(np.float32)
        # Renormalize so the three weights always sum to 1 after clipping.
        s = u_sel + v_sel + w_sel
        s = np.where(s < 1e-20, 1.0, s)
        u_sel /= s
        v_sel /= s
        w_sel /= s

        tri_idx[pix_y, pix_x] = f
        bary[pix_y, pix_x, 0] = u_sel
        bary[pix_y, pix_x, 1] = v_sel
        bary[pix_y, pix_x, 2] = w_sel
        valid[pix_y, pix_x] = True

    return RasterResult(tri_idx=tri_idx, bary=bary, valid=valid)
