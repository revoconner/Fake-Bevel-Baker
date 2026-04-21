"""Post-bake denoising of the world-space normal map.

Runs Intel Open Image Denoise (OIDN) on the raw per-texel bake output
*before* tangent-space projection, dilation, and encoding. Operates on
the world-space normal buffer because that signal is continuous across
UV seams; tangent-space values flip sign across seams and would confuse
the filter.

Improvements over a naive "RT filter on color only" call:

- Feed the per-texel geometric normal (Ng) as the `normal` AOV so the
  denoiser knows where hard edges live and does not smooth across them.
- Feed a synthesized `albedo` AOV (we have no material albedo, but OIDN
  is calibrated against one; a constant neutral value works as a stable
  guide that won't introduce false edges).
- Pre-fill invalid texels with the encoded Ng before filtering so OIDN
  does not see a black hole at UV island borders and bleed darkness
  inward.
- Set `cleanAux=True` so OIDN trusts the AOVs (our Ng is noise-free).
- Request `quality="high"` and disable HDR (our encoded normals live in
  [0, 1]). These flags aren't exposed by the thin `oidn` Python wrapper
  so they're set via ctypes against the OIDN shared library directly.

The `valid` mask is used to restrict the filter's output back to the UV
islands after it runs. Everything OIDN hallucinated outside the valid
region is re-zeroed.
"""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np


# ----- ctypes bindings for the OIDN attribute setters ---------------------
# The `oidn` Python package wraps only SetSharedFilterImage; the scalar
# attribute setters (hdr, quality, cleanAux, ...) aren't exposed. We bind
# them against the same DLL the Python package ships.

_EXTRA_BINDINGS: dict[str, object] = {}


def _load_extra_bindings():
    if _EXTRA_BINDINGS:
        return _EXTRA_BINDINGS
    import oidn as _oidn
    pkg_dir = Path(_oidn.__file__).parent
    candidates = list(pkg_dir.rglob("OpenImageDenoise.dll")) + \
                 list(pkg_dir.rglob("libOpenImageDenoise*.so*")) + \
                 list(pkg_dir.rglob("libOpenImageDenoise*.dylib"))
    if not candidates:
        raise RuntimeError("Could not locate the OpenImageDenoise shared library.")
    lib = ctypes.CDLL(str(candidates[0]))

    lib.oidnSetFilter1b.restype = None
    lib.oidnSetFilter1b.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_bool]
    lib.oidnSetFilter1i.restype = None
    lib.oidnSetFilter1i.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]

    _EXTRA_BINDINGS["lib"] = lib
    _EXTRA_BINDINGS["SetFilter1b"] = lib.oidnSetFilter1b
    _EXTRA_BINDINGS["SetFilter1i"] = lib.oidnSetFilter1i
    return _EXTRA_BINDINGS


def _set_filter_bool(flt_handle: int, name: str, value: bool) -> None:
    b = _load_extra_bindings()
    b["SetFilter1b"](ctypes.c_void_p(flt_handle), name.encode("ascii"), bool(value))


def _set_filter_int(flt_handle: int, name: str, value: int) -> None:
    b = _load_extra_bindings()
    b["SetFilter1i"](ctypes.c_void_p(flt_handle), name.encode("ascii"), int(value))


# OIDN quality enum values per OpenImageDenoise.h, stable across 2.x releases.
_QUALITY_DEFAULT = 0
_QUALITY_BALANCED = 4
_QUALITY_HIGH = 5


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
    quality: str = "high",
) -> np.ndarray:
    """Run OIDN on a world-space normal image.

    world_normal: (H, W, 3) float32 unit vectors in [-1, 1]. Invalid
        texels should already be zero (that's how `bake()` leaves them).
    valid: (H, W) bool mask.
    aux_normal: optional (H, W, 3) float32 guide normals (e.g. per-texel
        geometric normal Ng). If supplied, fed to OIDN's `normal` AOV
        input with `cleanAux=True`. Dramatically improves hard-edge
        preservation.
    quality: "high" (default), "balanced", or "default". "high" costs
        roughly 2x the time of "balanced" and gives sharper output.

    Returns a new (H, W, 3) float32 array, unit-length on valid texels,
    zero elsewhere.
    """
    oidn = _require_oidn()

    if world_normal.ndim != 3 or world_normal.shape[2] != 3:
        raise ValueError("world_normal must be (H, W, 3)")
    if valid.shape != world_normal.shape[:2]:
        raise ValueError("valid mask shape must match world_normal")

    H, W, _ = world_normal.shape

    # Encode the signed normal into [0, 1] "color" space so OIDN sees a
    # value distribution close to what its RT model was trained on.
    color_in = (world_normal * 0.5 + 0.5).astype(np.float32)

    # Pre-fill invalid texels with the encoded aux_normal (or neutral 0.5
    # if no aux was provided). This prevents a dark hole outside UV
    # islands from bleeding into the filter's output along the border.
    if aux_normal is not None:
        fill = (aux_normal * 0.5 + 0.5).astype(np.float32)
    else:
        fill = np.full_like(color_in, 0.5, dtype=np.float32)
    inv = ~valid
    color_in[inv] = fill[inv]
    color_in = np.ascontiguousarray(color_in)

    # Synthesize a stable albedo AOV: neutral grey on valid pixels,
    # neutral grey on invalid (same value everywhere). This is the
    # minimal input that keeps OIDN's AOV-guided mode active without
    # injecting fake edges — it tells the denoiser "there is nothing in
    # the albedo to disambiguate", so all guidance comes from the
    # normal AOV.
    albedo_in = np.full((H, W, 3), 0.5, dtype=np.float32)

    normal_in: np.ndarray | None = None
    if aux_normal is not None:
        if aux_normal.shape != world_normal.shape:
            raise ValueError("aux_normal shape mismatch")
        normal_in = np.ascontiguousarray(aux_normal.astype(np.float32))

    output = np.zeros_like(color_in)

    device = oidn.NewDevice(oidn.DEVICE_TYPE_CPU)
    oidn.CommitDevice(device)
    err = oidn.GetDeviceError(device)
    if err != oidn.ERROR_NONE:
        oidn.ReleaseDevice(device)
        raise RuntimeError(f"OIDN device init failed ({err})")

    flt = oidn.NewFilter(device, "RT")
    oidn.SetSharedFilterImage(flt, "color", color_in, oidn.FORMAT_FLOAT3, W, H)
    oidn.SetSharedFilterImage(flt, "albedo", albedo_in, oidn.FORMAT_FLOAT3, W, H)
    if normal_in is not None:
        oidn.SetSharedFilterImage(flt, "normal", normal_in, oidn.FORMAT_FLOAT3, W, H)
    oidn.SetSharedFilterImage(flt, "output", output, oidn.FORMAT_FLOAT3, W, H)

    # Extra attributes via ctypes (not wrapped by the Python binding).
    try:
        _set_filter_bool(flt, "hdr", False)         # values are LDR in [0, 1]
        _set_filter_bool(flt, "cleanAux", True)     # Ng and albedo are noise-free
        _set_filter_bool(flt, "srgb", False)        # normals are linear, not sRGB
        q = {
            "default": _QUALITY_DEFAULT,
            "balanced": _QUALITY_BALANCED,
            "high": _QUALITY_HIGH,
        }.get(quality, _QUALITY_HIGH)
        _set_filter_int(flt, "quality", q)
    except Exception:
        # If the ctypes shim fails (odd OIDN build, missing symbols), the
        # filter still runs with default attributes — not ideal but fine.
        pass

    oidn.CommitFilter(flt)
    oidn.ExecuteFilter(flt)
    err = oidn.GetDeviceError(device)
    oidn.ReleaseFilter(flt)
    oidn.ReleaseDevice(device)
    if err != oidn.ERROR_NONE:
        raise RuntimeError(f"OIDN filter execution failed ({err})")

    # Decode [0, 1] back to [-1, 1] and renormalize onto the unit sphere.
    decoded = output * 2.0 - 1.0
    lens = np.linalg.norm(decoded, axis=-1, keepdims=True)
    safe = np.where(lens < 1e-20, 1.0, lens)
    decoded /= safe

    # Zero out anything OIDN hallucinated into invalid texels.
    decoded[~valid] = 0.0
    return decoded.astype(np.float32)
