"""Tests for PNG 16-bit and EXR 32-bit writers."""

from __future__ import annotations

import struct

import numpy as np
import pytest

from fake_bevel_baker.image_io import (
    encode_normal_to_float01,
    write_exr_rgb32,
    write_png_rgb16,
)


def test_png16_header(tmp_path):
    img = np.zeros((16, 24, 3), dtype=np.uint16)
    img[..., 0] = 65535
    img[..., 1] = 32768
    out = tmp_path / "t.png"
    write_png_rgb16(out, img)
    data = out.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    w, h = struct.unpack(">II", data[16:24])
    assert (w, h) == (24, 16)
    assert data[24] == 16 and data[25] == 2  # 16-bit, RGB


def test_png16_rejects_wrong_shape(tmp_path):
    out = tmp_path / "bad.png"
    with pytest.raises(ValueError):
        write_png_rgb16(out, np.zeros((4, 4), dtype=np.uint16))


def test_exr_roundtrip_values(tmp_path):
    OpenEXR = pytest.importorskip("OpenEXR")
    h, w = 8, 12
    img = np.stack([
        np.linspace(0, 1, w)[None, :].repeat(h, axis=0),
        np.linspace(0, 1, h)[:, None].repeat(w, axis=1),
        np.full((h, w), 0.5),
    ], axis=-1).astype(np.float32)
    out = tmp_path / "rt.exr"
    write_exr_rgb32(out, img)

    with OpenEXR.File(str(out)) as f:
        ch = f.parts[0].channels
        assert "RGB" in ch
        back = ch["RGB"].pixels
        assert back.shape == (h, w, 3)
        np.testing.assert_allclose(back, img, atol=1e-6)


def test_exr_rejects_wrong_shape(tmp_path):
    out = tmp_path / "bad.exr"
    with pytest.raises(ValueError):
        write_exr_rgb32(out, np.zeros((4, 4), dtype=np.float32))


def test_encode_normal_to_float01_matches_png_encoding():
    # Check it matches the uint16 encoding used for PNGs.
    from fake_bevel_baker.bake import encode_normal_to_uint16
    n = np.array([[[0, 0, 1.0], [1, 0, 0], [-1, -1, -1]]], dtype=np.float32)
    enc01 = encode_normal_to_float01(n)
    enc16 = encode_normal_to_uint16(n)
    # enc16 / 65535 should be ~= enc01
    np.testing.assert_allclose(enc16.astype(np.float32) / 65535.0, enc01, atol=1e-4)
