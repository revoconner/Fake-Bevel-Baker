"""Command-line entry for the bevel baker.

Example:
    python -m fake_bevel_baker.main \\
        --mesh tests/fixtures/cube.obj \\
        --out cube_bevel.png \\
        --resolution 512 --samples 16 --angle 45 --radius 2.0
"""

from __future__ import annotations

import argparse
import struct
import sys
import time
import zlib
from pathlib import Path

import numpy as np

from .bake import bake, encode_normal_to_uint16
from .bvh import BVH
from .dilation import dilate
from .mesh_prep import prepare_mesh
from .uv_raster import rasterize_uvs


def _png_chunk(tag: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + tag + data
        + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    )


def _write_png_rgb16(path: Path, img_uint16: np.ndarray) -> None:
    """Write a 16-bit RGB PNG. Pillow can't, imageio plugins not installed."""
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

    ihdr = struct.pack(">IIBBBBB", w, h, 16, 2, 0, 0, 0)  # 16-bit, color_type 2 = RGB
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(_png_chunk(b"IHDR", ihdr))
        f.write(_png_chunk(b"IDAT", compressed))
        f.write(_png_chunk(b"IEND", b""))


def _write_png(path: Path, img_uint16: np.ndarray) -> None:
    # Flip vertically so v=0 UV row sits at the bottom of the saved image
    # (PNG rows go top-down; our raster has v=0 at row 0).
    _write_png_rgb16(path, img_uint16[::-1])


def _write_exr(path: Path, img_float: np.ndarray) -> None:
    try:
        import imageio.v3 as iio
    except ImportError as e:
        raise RuntimeError("imageio not installed") from e
    img_flipped = img_float[::-1].astype(np.float32)
    iio.imwrite(str(path), img_flipped)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Fake bevel normal map baker (command-line)")
    p.add_argument("--mesh", required=True, help="Input mesh (.obj, .gltf, .ply, ...)")
    p.add_argument("--out", required=True, help="Output tangent-space normal map path")
    p.add_argument("--out-world", default=None, help="Optional world-space output path")
    p.add_argument("--resolution", type=int, default=512, help="Square output resolution (default 512)")
    p.add_argument("--samples", type=int, default=16, help="Samples per texel (default 16)")
    p.add_argument("--angle", type=float, default=30.0, help="Soften-by-angle degrees (default 30)")
    p.add_argument("--radius", type=float, default=None,
                   help="Bevel radius in world units (default = 0.2%% of mesh bbox diagonal)")
    p.add_argument("--seed", type=int, default=0, help="RNG seed (default 0)")
    p.add_argument("--format", choices=("png", "exr"), default="png",
                   help="Output format (default png = 16-bit)")
    p.add_argument("--dilation", type=int, default=16,
                   help="Edge padding ring width in texels (default 16, 0 disables)")
    args = p.parse_args(argv)

    mesh_path = Path(args.mesh)
    out_path = Path(args.out)
    res = args.resolution

    print(f"Loading mesh: {mesh_path}")
    t0 = time.time()
    prep = prepare_mesh(mesh_path, angle_deg=args.angle)
    print(f"  split mesh: {prep.split_positions.shape[0]} verts, "
          f"{prep.split_faces.shape[0]} tris")
    print(f"  tangent mesh: {prep.tangent_positions.shape[0]} verts")

    # Default radius = 0.2% of bbox diagonal
    if args.radius is None:
        bbox_min = prep.tangent_positions.min(axis=0)
        bbox_max = prep.tangent_positions.max(axis=0)
        diag = float(np.linalg.norm(bbox_max - bbox_min))
        radius = 0.002 * diag
        print(f"  bbox diagonal: {diag:.4f}, auto radius: {radius:.4f}")
    else:
        radius = float(args.radius)
        print(f"  radius: {radius:.4f}")

    print(f"Rasterizing UVs at {res}x{res}...")
    raster = rasterize_uvs(prep.tangent_uvs, prep.tangent_faces, res, res)
    covered = int(raster.valid.sum())
    print(f"  covered texels: {covered} / {res * res}")

    print("Building BVH (Embree)...")
    bvh = BVH(prep.split_positions, prep.split_faces)

    print(f"Baking: {args.samples} samples/texel, seed={args.seed}")
    last_pct = -1

    def prog(i, total):
        nonlocal last_pct
        pct = int(100 * i / max(total, 1))
        if pct != last_pct and pct % 5 == 0:
            last_pct = pct
            print(f"  {pct}%  ({i}/{total})")

    t1 = time.time()
    result = bake(prep, raster, bvh, radius=radius, num_samples=args.samples,
                  seed=args.seed, progress=prog)
    t_bake = time.time() - t1
    print(f"Bake done in {t_bake:.2f}s")

    tan_img = result.tangent_normal
    world_img = result.world_normal
    if args.dilation > 0:
        td = time.time()
        tan_img, _ = dilate(tan_img, result.valid, radius_px=args.dilation)
        world_img, _ = dilate(world_img, result.valid, radius_px=args.dilation)
        print(f"Dilation ({args.dilation}px): {time.time() - td:.2f}s")

    if args.format == "png":
        print(f"Writing 16-bit PNG: {out_path}")
        _write_png(out_path, encode_normal_to_uint16(tan_img))
        if args.out_world:
            _write_png(Path(args.out_world), encode_normal_to_uint16(world_img))
    else:
        print(f"Writing EXR: {out_path}")
        _write_exr(out_path, tan_img)
        if args.out_world:
            _write_exr(Path(args.out_world), world_img)

    print(f"Total: {time.time() - t0:.2f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
