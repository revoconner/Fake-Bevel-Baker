"""
edge padding for UV islands.

Invalid texels within `radius_px` of a valid texel inherit the nearest
valid value. One `distance_transform_edt` call does distance + nearest
source index in a single pass.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt


def dilate(
    image: np.ndarray,
    valid: np.ndarray,
    radius_px: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    """Fill invalid texels within `radius_px` of a valid one.

    image: (H, W) or (H, W, C). Untouched outside the dilation ring.
    valid: (H, W) bool.
    Returns (dilated_image, dilated_valid).
    """
    if radius_px < 0:
        raise ValueError("radius_px must be >= 0")
    if valid.shape != image.shape[:2]:
        raise ValueError("valid mask and image must have matching HxW")
    if radius_px == 0:
        return image.copy(), valid.copy()

    dist, indices = distance_transform_edt(~valid, return_indices=True)
    iy = indices[0]
    ix = indices[1]

    ring = (dist > 0.0) & (dist <= float(radius_px))

    out = image.copy()
    if out.ndim == 3:
        out[ring] = image[iy[ring], ix[ring]]
    else:
        out[ring] = image[iy[ring], ix[ring]]

    dilated_valid = valid | ring
    return out, dilated_valid
