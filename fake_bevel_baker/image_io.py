"""Image writers.

- `write_png_rgb16`: 16-bit RGB PNG via stdlib zlib+struct (Pillow/imageio
  cannot write 16-bit RGB PNG in this env).
- `write_exr_rgb32`: 32-bit float RGB OpenEXR via the `OpenEXR` package


Both expect `img_float` in [0, 1] encoded form. Callers that have raw
normals in [-1, 1] should pass them through `encode_normal_to_uint16` /
`encode_normal_to_float01` first. Keeping the encoding consistent
between PNG and EXR outputs is important: most PBR pipelines (Substance,
Marmoset, engine importers) assume [0, 1] encoded normals regardless of
format.
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np


# ----- PNG 16-bit RGB -----------------------------------------------------

def _png_chunk(tag: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + tag + data
        + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    )


def write_png_rgb16(path: Path, img_uint16: np.ndarray) -> None:
    if img_uint16.dtype != np.uint16 or img_uint16.ndim != 3 or img_uint16.shape[2] != 3:
        raise ValueError("expected (H, W, 3) uint16 image")
    h, w, _ = img_uint16.shape
    be = img_uint16.astype(">u2").tobytes()
    row_bytes = w * 3 * 2
    body = bytearray()
    for y in range(h):
        body.append(0)  # filter type: None
        body.extend(be[y * row_bytes : (y + 1) * row_bytes])
    compressed = zlib.compress(bytes(body), 6)
    ihdr = struct.pack(">IIBBBBB", w, h, 16, 2, 0, 0, 0)  # 16-bit, RGB
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(_png_chunk(b"IHDR", ihdr))
        f.write(_png_chunk(b"IDAT", compressed))
        f.write(_png_chunk(b"IEND", b""))


# ----- EXR 32-bit float RGB via the OpenEXR package -----------------------

def encode_normal_to_float01(img_signed: np.ndarray) -> np.ndarray:
    """Map a unit-vector image in [-1, 1] to float32 [0, 1].

    Matches the encoding used by `encode_normal_to_uint16` so PNG and
    EXR outputs have the same color interpretation in downstream tools.
    """
    return np.clip(img_signed * 0.5 + 0.5, 0.0, 1.0).astype(np.float32)


def write_exr_rgb32(path: Path, img_float: np.ndarray) -> None:
    """Write a (H, W, 3) float32 RGB EXR, ZIP_COMPRESSION, linear.

    `img_float` is written as-is — callers are responsible for any
    encoding (e.g. normals going [-1, 1] -> [0, 1]).
    """
    try:
        import OpenEXR
    except ImportError as exc:  # pragma: no cover - depends on user env
        raise RuntimeError(
            "OpenEXR package not installed. "
            "Install with: bebvenv\\Scripts\\pip install OpenEXR"
        ) from exc

    if img_float.ndim != 3 or img_float.shape[2] != 3:
        raise ValueError("expected (H, W, 3) float image")
    rgb = np.ascontiguousarray(img_float, dtype=np.float32)

    header = {
        "compression": OpenEXR.ZIPS_COMPRESSION,
        "type": OpenEXR.scanlineimage,
    }
    channels = {"RGB": rgb}
    with OpenEXR.File(header, channels) as f:
        f.write(str(path))
