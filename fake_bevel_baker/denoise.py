"""Post-bake denoising of the world-space normal map.

Runs Intel Open Image Denoise (OIDN) on the raw per-texel bake output
*before* tangent-space projection, dilation, and encoding. Operates on
the world-space normal buffer because that signal is continuous across
UV seams; tangent-space values flip sign across seams and would confuse
the filter.

The `valid` mask is used to restrict noisy pixels from bleeding into
invalid texels and vice versa. OIDN itself has no mask input, so we
feed the valid region directly and renormalize / remask afterward.
"""

from __future__ import annotations

import numpy as np


def _require_oidn():
    try:
        import oidn
    except ImportError as exc:
        raise RuntimeError(
            "oidn package not installed. Install with: "
            "bebvenv\\Scripts\\pip install oidn"
        ) from exc
    return oidn


def denoise_world_normal(
    world_normal: np.ndarray,
    valid: np.ndarray,
    aux_normal: np.ndarray | None = None,
) -> np.ndarray:
    """Run OIDN on a world-space normal image.

    world_normal: (H, W, 3) float32 unit vectors in [-1, 1]. Invalid
        texels should already be zero (that's how `bake()` leaves them).
    valid: (H, W) bool mask.
    aux_normal: optional (H, W, 3) float32 guide normals (e.g. per-texel
        geometric normal `Ng`). Helps the denoiser respect hard edges.
        If None, no normal guide is passed.

    Returns a new (H, W, 3) float32 array, unit-length on valid texels,
    zero elsewhere.
    """
    oidn = _require_oidn()

    if world_normal.ndim != 3 or world_normal.shape[2] != 3:
        raise ValueError("world_normal must be (H, W, 3)")
    if valid.shape != world_normal.shape[:2]:
        raise ValueError("valid mask shape must match world_normal")

    H, W, _ = world_normal.shape

    # OIDN's RT filter wants [0, 1] color data. Encode normals the same
    # way PNG does so the filter sees plausible "color" values. We'll
    # decode after.
    color_in = np.ascontiguousarray(
        (world_normal * 0.5 + 0.5).astype(np.float32)
    )
    output = np.zeros_like(color_in)

    device = oidn.NewDevice(oidn.DEVICE_TYPE_CPU)
    oidn.CommitDevice(device)
    err = oidn.GetDeviceError(device)
    if err != oidn.ERROR_NONE:
        oidn.ReleaseDevice(device)
        raise RuntimeError(f"OIDN device init failed ({err})")

    flt = oidn.NewFilter(device, "RT")
    oidn.SetSharedFilterImage(flt, "color", color_in,
                              oidn.FORMAT_FLOAT3, W, H)
    if aux_normal is not None:
        if aux_normal.shape != world_normal.shape:
            raise ValueError("aux_normal shape mismatch")
        aux = np.ascontiguousarray(aux_normal.astype(np.float32))
        oidn.SetSharedFilterImage(flt, "normal", aux,
                                  oidn.FORMAT_FLOAT3, W, H)
    oidn.SetSharedFilterImage(flt, "output", output,
                              oidn.FORMAT_FLOAT3, W, H)
    oidn.CommitFilter(flt)
    oidn.ExecuteFilter(flt)
    err = oidn.GetDeviceError(device)
    oidn.ReleaseFilter(flt)
    oidn.ReleaseDevice(device)
    if err != oidn.ERROR_NONE:
        raise RuntimeError(f"OIDN filter execution failed ({err})")

    # Decode [0, 1] back to [-1, 1] and renormalize on the unit sphere.
    decoded = output * 2.0 - 1.0
    lens = np.linalg.norm(decoded, axis=-1, keepdims=True)
    safe = np.where(lens < 1e-20, 1.0, lens)
    decoded /= safe

    # Zero out everything OIDN hallucinated into invalid texels.
    decoded[~valid] = 0.0
    return decoded.astype(np.float32)
