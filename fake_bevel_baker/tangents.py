"""Stage 4 pre-req: MikkTSpace tangents via ctypes against the reference C.

The `pygltfio` package listed in the spec does not provide a
MikkTSpace binding (it is a glTF parser). We build the reference
implementation into a shared library via
`fake_bevel_baker.native.build_mikkt` and load it here.

Output is per-face-corner `(xyz, sign)`. Callers should interpolate
tangents using the texel's face index + barycentrics (do NOT average
to per-vertex — MikkT corner tangents that share a vertex are only
identical when MikkT's welding rules say they are).
"""

from __future__ import annotations

import ctypes as C
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_NATIVE_DIR = _HERE / "native"


def _lib_name() -> str:
    if sys.platform == "win32":
        return "mikkt.dll"
    if sys.platform == "darwin":
        return "libmikkt.dylib"
    return "libmikkt.so"


def _load_lib() -> C.CDLL:
    path = _NATIVE_DIR / _lib_name()
    if not path.exists():
        raise FileNotFoundError(
            f"MikkTSpace shared library not found at {path}. "
            f"Build it with: python -m fake_bevel_baker.native.build_mikkt"
        )
    lib = C.CDLL(str(path))
    lib.fbb_compute_mikkt_tangents.argtypes = [
        C.POINTER(C.c_float),  # positions
        C.POINTER(C.c_float),  # normals
        C.POINTER(C.c_float),  # uvs
        C.POINTER(C.c_int),    # faces
        C.c_int,               # num_faces
        C.POINTER(C.c_float),  # out_tangents (F*3*4)
    ]
    lib.fbb_compute_mikkt_tangents.restype = C.c_int
    return lib


_LIB: C.CDLL | None = None


def _lib() -> C.CDLL:
    global _LIB
    if _LIB is None:
        _LIB = _load_lib()
    return _LIB


def compute_tangents(
    positions: np.ndarray,
    normals: np.ndarray,
    uvs: np.ndarray,
    faces: np.ndarray,
) -> np.ndarray:
    """Return per-face-corner tangents, shape (F, 3, 4) -> (tx, ty, tz, sign).

    All inputs are for a single triangulated mesh (the tangent_mesh from
    Stage 1). Arrays are coerced to contiguous float32 / int32.
    Bitangent = sign * cross(normal, tangent) at the corner.
    """
    pos = np.ascontiguousarray(positions, dtype=np.float32)
    nrm = np.ascontiguousarray(normals, dtype=np.float32)
    uv = np.ascontiguousarray(uvs, dtype=np.float32)
    f = np.ascontiguousarray(faces, dtype=np.int32)

    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError("positions must be (V, 3)")
    if nrm.shape != pos.shape:
        raise ValueError("normals must match positions shape")
    if uv.ndim != 2 or uv.shape[1] != 2 or uv.shape[0] != pos.shape[0]:
        raise ValueError("uvs must be (V, 2) with same V as positions")
    if f.ndim != 2 or f.shape[1] != 3:
        raise ValueError("faces must be (F, 3)")

    F = int(f.shape[0])
    out = np.zeros((F, 3, 4), dtype=np.float32)

    ok = _lib().fbb_compute_mikkt_tangents(
        pos.ctypes.data_as(C.POINTER(C.c_float)),
        nrm.ctypes.data_as(C.POINTER(C.c_float)),
        uv.ctypes.data_as(C.POINTER(C.c_float)),
        f.ctypes.data_as(C.POINTER(C.c_int)),
        F,
        out.ctypes.data_as(C.POINTER(C.c_float)),
    )
    if ok == 0:
        raise RuntimeError("genTangSpaceDefault returned failure")
    return out


def bitangent_from_tangent(normal: np.ndarray, tangent_xyzs: np.ndarray) -> np.ndarray:
    """Reconstruct per-corner bitangent = sign * cross(normal, tangent).

    normal: (..., 3), tangent_xyzs: (..., 4) with last component = sign.
    Returns: (..., 3).
    """
    t = tangent_xyzs[..., :3]
    s = tangent_xyzs[..., 3:4]
    return s * np.cross(normal, t)
