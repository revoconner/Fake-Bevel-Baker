"""Stage 5: per-texel bevel bake.

Port of Blender Cycles `svm_bevel`. For each valid texel we shoot
`num_samples` probe rays on a BSSRDF-weighted disk around the shading
point, gather multi-hit normals through Embree, MIS-combine across the
3-axis disk frame, and average. World-space result is then rotated into
MikkTSpace tangent space via the tangent_mesh face corners.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .bvh import BVH, LOCAL_MAX_HITS
from .mesh_prep import PreparedMesh
from .sampling import (
    halton_2d,
    make_orthonormals_batch,
    per_texel_offsets,
    svm_bevel_cubic_eval,
    svm_bevel_cubic_sample,
)
from .tangents import compute_tangents
from .uv_raster import RasterResult


@dataclass
class BakeResult:
    world_normal: np.ndarray   # (H, W, 3) float32, world-space, unit
    tangent_normal: np.ndarray # (H, W, 3) float32, MikkT tangent-space, unit
    valid: np.ndarray          # (H, W) bool


def bake(
    prep: PreparedMesh,
    raster: RasterResult,
    bvh: BVH,
    radius: float,
    num_samples: int,
    seed: int = 0,
    progress: Callable[[int, int], None] | None = None,
) -> BakeResult:
    """Run the bevel bake.

    prep, raster, bvh: prepared inputs.
    radius: bevel world-space disk radius.
    num_samples: per-texel probe rays.
    seed: RNG seed for the per-texel Cranley-Patterson offsets.
    progress: optional callback called as progress(sample_index, num_samples).
    """
    if radius <= 0.0 or num_samples < 1:
        raise ValueError("radius must be > 0 and num_samples >= 1")

    H, W = raster.valid.shape
    world_out = np.zeros((H, W, 3), dtype=np.float32)
    tan_out = np.zeros((H, W, 3), dtype=np.float32)

    tangents_corner = compute_tangents(
        prep.tangent_positions, prep.tangent_normals,
        prep.tangent_uvs, prep.tangent_faces,
    )  # (F, 3, 4)

    ys, xs = np.where(raster.valid)
    tri_ids = raster.tri_idx[ys, xs].astype(np.int64)
    barys = raster.bary[ys, xs].astype(np.float32)
    N = int(ys.shape[0])
    if N == 0:
        return BakeResult(world_out, tan_out, raster.valid)

    # Split-mesh data for sampling
    sp = prep.split_positions
    sn = prep.split_normals
    sf = prep.split_faces
    fn = prep.face_normals

    tri_sp = sp[sf[tri_ids]]           # (N, 3, 3)
    tri_sn = sn[sf[tri_ids]]           # (N, 3, 3)
    P = (tri_sp * barys[..., None]).sum(axis=1).astype(np.float32)
    Ng = fn[tri_ids].astype(np.float32)

    # Tangent frame at texel (from tangent_mesh corner tangents + normals)
    tf = prep.tangent_faces
    T_corner = tangents_corner[tri_ids]                    # (N, 3, 4)
    N_corner = prep.tangent_normals[tf[tri_ids]]           # (N, 3, 3)
    T_interp = (T_corner[..., :3] * barys[..., None]).sum(axis=1)
    N_tan_interp = (N_corner * barys[..., None]).sum(axis=1)
    N_tan_interp /= np.maximum(np.linalg.norm(N_tan_interp, axis=-1, keepdims=True), 1e-20)
    # Re-orthogonalize T against N to keep the basis clean after interpolation
    T_interp -= (T_interp * N_tan_interp).sum(axis=-1, keepdims=True) * N_tan_interp
    T_interp /= np.maximum(np.linalg.norm(T_interp, axis=-1, keepdims=True), 1e-20)
    T_sign = T_corner[:, 0, 3]  # corners of one face share sign
    B_interp = T_sign[:, None] * np.cross(N_tan_interp, T_interp)

    # Base Ng frame
    T_base, B_base = make_orthonormals_batch(Ng)

    # Stack (Ng, T_base, B_base) so we can permute per-axis-choice
    frame = np.stack([Ng, T_base, B_base], axis=1)  # (N, 3, 3)

    # Per-texel CP offsets
    offsets = per_texel_offsets(N, seed=seed)

    sum_N = np.zeros((N, 3), dtype=np.float64)

    for s in range(num_samples):
        if progress is not None:
            progress(s, num_samples)
        h1, h2 = halton_2d(s)
        u1 = (h1 + offsets[:, 0]) % 1.0
        u2 = (h2 + offsets[:, 1]) % 1.0

        axis_choice = np.where(u1 < 0.5, 0, np.where(u1 < 0.75, 1, 2)).astype(np.int32)
        rand_x = np.where(
            axis_choice == 0, u1 * 2.0,
            np.where(axis_choice == 1, (u1 - 0.5) * 4.0, (u1 - 0.75) * 4.0),
        ).astype(np.float32)

        # Permutation of frame rows per axis choice:
        # axis 0 (Ng): (0,1,2)  axis 1 (T): (1,0,2)  axis 2 (B): (2,1,0)
        perm = np.empty((N, 3), dtype=np.int64)
        m0 = axis_choice == 0
        m1 = axis_choice == 1
        m2 = axis_choice == 2
        perm[m0] = (0, 1, 2)
        perm[m1] = (1, 0, 2)
        perm[m2] = (2, 1, 0)
        picked = np.take_along_axis(frame, perm[..., None].repeat(3, axis=-1), axis=1)
        disk_N = picked[:, 0].astype(np.float32)
        disk_T = picked[:, 1].astype(np.float32)
        disk_B = picked[:, 2].astype(np.float32)

        pick_pdf = np.empty((N, 3), dtype=np.float32)
        pick_pdf[m0] = (0.5, 0.25, 0.25)
        pick_pdf[m1] = (0.25, 0.5, 0.25)
        pick_pdf[m2] = (0.25, 0.25, 0.5)

        disk_r, disk_height = svm_bevel_cubic_sample(radius, u2.astype(np.float32))
        disk_pdf_self = svm_bevel_cubic_eval(radius, disk_r)  # (N,)

        phi = 2.0 * math.pi * rand_x
        cos_phi = np.cos(phi).astype(np.float32)
        sin_phi = np.sin(phi).astype(np.float32)
        disk_offset = disk_T * (disk_r * cos_phi)[:, None] + disk_B * (disk_r * sin_phi)[:, None]

        ray_origins = (P + disk_N * disk_height[:, None] + disk_offset).astype(np.float32)
        ray_dirs = (-disk_N).astype(np.float32)

        hits = bvh.multi_hit(
            ray_origins, ray_dirs,
            tmin=np.zeros(N, dtype=np.float32),
            tmax=(2.0 * disk_height).astype(np.float32),
            max_hits=LOCAL_MAX_HITS,
        )

        for hit_i in range(LOCAL_MAX_HITS):
            mask = hits.primID[:, hit_i] != -1
            if not mask.any():
                continue
            idx = np.where(mask)[0]
            pids = hits.primID[idx, hit_i].astype(np.int64)
            us = hits.u[idx, hit_i]
            vs = hits.v[idx, hit_i]
            ng_raw = hits.Ng[idx, hit_i]
            ng_len = np.linalg.norm(ng_raw, axis=-1, keepdims=True)
            hit_Ng = ng_raw / np.maximum(ng_len, 1e-20)

            # Smooth normal via barycentric interpolation on split mesh.
            # Embree barys: u is weight of v1, v is weight of v2, (1-u-v) is weight of v0.
            w0 = 1.0 - us - vs
            w1 = us
            w2 = vs
            tri_sn_h = sn[sf[pids]]       # (M, 3, 3)
            tri_sp_h = sp[sf[pids]]
            wb = np.stack([w0, w1, w2], axis=-1)[..., None].astype(np.float32)
            hit_N = (tri_sn_h * wb).sum(axis=1)
            hit_N /= np.maximum(np.linalg.norm(hit_N, axis=-1, keepdims=True), 1e-20)
            hit_P = (tri_sp_h * wb).sum(axis=1)

            dN = disk_N[idx]; dT = disk_T[idx]; dB = disk_B[idx]
            pdf_N_axis = pick_pdf[idx, 0] * np.abs((dN * hit_Ng).sum(axis=-1))
            pdf_T_axis = pick_pdf[idx, 1] * np.abs((dT * hit_Ng).sum(axis=-1))
            pdf_B_axis = pick_pdf[idx, 2] * np.abs((dB * hit_Ng).sum(axis=-1))
            denom = pdf_N_axis ** 2 + pdf_T_axis ** 2 + pdf_B_axis ** 2
            w = pdf_N_axis / np.maximum(denom, 1e-20)

            r_real = np.linalg.norm(hit_P - P[idx], axis=-1).astype(np.float32)
            pdf_real = svm_bevel_cubic_eval(radius, r_real)
            disk_pdf_i = disk_pdf_self[idx]
            w *= pdf_real / np.maximum(disk_pdf_i, 1e-20)

            sum_N[idx] += (w[:, None] * hit_N).astype(np.float64)

    if progress is not None:
        progress(num_samples, num_samples)

    lens = np.linalg.norm(sum_N, axis=-1, keepdims=True)
    nonzero = lens.squeeze(-1) > 1e-20
    N_world = np.where(
        nonzero[:, None],
        sum_N / np.maximum(lens, 1e-20),
        Ng,
    ).astype(np.float32)

    # World -> tangent space per texel.
    t_dot = (N_world * T_interp).sum(axis=-1)
    b_dot = (N_world * B_interp).sum(axis=-1)
    n_dot = (N_world * N_tan_interp).sum(axis=-1)
    N_tan = np.stack([t_dot, b_dot, n_dot], axis=-1)
    N_tan /= np.maximum(np.linalg.norm(N_tan, axis=-1, keepdims=True), 1e-20)

    world_out[ys, xs] = N_world
    tan_out[ys, xs] = N_tan

    return BakeResult(world_normal=world_out, tangent_normal=tan_out, valid=raster.valid)


def encode_normal_to_uint16(img: np.ndarray) -> np.ndarray:
    """Map unit-vector image [-1, 1] -> uint16 [0, 65535] with 0.5-offset encoding."""
    v = np.clip(img * 0.5 + 0.5, 0.0, 1.0)
    return (v * 65535.0 + 0.5).astype(np.uint16)
